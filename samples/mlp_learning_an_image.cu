/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   mlp-learning-an-image.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  Sample application that uses the tiny cuda nn framework to learn a
            2D function that represents an image.
 */

#include <tiny-cuda-nn/common_device.h>

#include <tiny-cuda-nn/config.h>

#include <stbi/stbi_wrapper.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace tcnn;
using precision_t = network_precision_t;

GPUMemory<float> load_image(const std::string& filename, int& width, int& height) {
	// width * height * RGBA
	float* out = load_stbi(&width, &height, filename.c_str());

	GPUMemory<float> result(width * height * 4);
	result.copy_from_host(out);
	free(out); // release memory of image data

	return result;
}

template <typename T>
__global__ void to_ldr(const uint64_t num_elements, const uint32_t n_channels, const uint32_t stride, const T* __restrict__ in, uint8_t* __restrict__ out) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint64_t pixel = i / n_channels;
	const uint32_t channel = i - pixel * n_channels;

	out[i] = (uint8_t)(powf(fmaxf(fminf(in[pixel * stride + channel], 1.0f), 0.0f), 1.0f/2.2f) * 255.0f + 0.5f);
}

template <typename T>
void save_image(const T* image, int width, int height, int n_channels, int channel_stride, const std::string& filename) {
	GPUMemory<uint8_t> image_ldr(width * height * n_channels);
	linear_kernel(to_ldr<T>, 0, nullptr, width * height * n_channels, n_channels, channel_stride, image, image_ldr.data());

	std::vector<uint8_t> image_ldr_host(width * height * n_channels);
	CUDA_CHECK_THROW(cudaMemcpy(image_ldr_host.data(), image_ldr.data(), image_ldr.size(), cudaMemcpyDeviceToHost));

	save_stbi(image_ldr_host.data(), width, height, n_channels, filename.c_str());
}

template <uint32_t stride>
__global__ void eval_image(uint32_t n_elements, cudaTextureObject_t texture, float* __restrict__ xs_and_ys, float* __restrict__ result) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	uint32_t output_idx = i * stride;
	uint32_t input_idx = i * 2;

	float4 val = tex2D<float4>(texture, xs_and_ys[input_idx], xs_and_ys[input_idx+1]);
	result[output_idx + 0] = val.x;
	result[output_idx + 1] = val.y;
	result[output_idx + 2] = val.z;

	for (uint32_t i = 3; i < stride; ++i) {
		result[output_idx + i] = 1;
	}
}

int main(int argc, char* argv[]) {
	try {
		uint32_t compute_capability = cuda_compute_capability();
		if (compute_capability < MIN_GPU_ARCH) {
			std::cerr
				<< "Warning: Insufficient compute capability " << compute_capability << " detected. "
				<< "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly." << std::endl;
		}

		if (argc < 2) {
			std::cout << "USAGE: " << argv[0] << " " << "path-to-image.jpg [path-to-optional-config.json]" << std::endl;
			std::cout << "Sample EXR files are provided in 'data/images'." << std::endl;
			return 0;
		}

		json config = {
			{"loss", {
				{"otype", "RelativeL2"}
			}},
			{"optimizer", {
				{"otype", "Adam"},
				// {"otype", "Shampoo"},
				{"learning_rate", 1e-2},
				{"beta1", 0.9f},
				{"beta2", 0.99f},
				{"l2_reg", 0.0f},
				// The following parameters are only used when the optimizer is "Shampoo".
				{"beta3", 0.9f},
				{"beta_shampoo", 0.0f},
				{"identity", 0.0001f},
				{"cg_on_momentum", false},
				{"frobenius_normalization", true},
			}},
			{"encoding", {
				{"otype", "OneBlob"},
				{"n_bins", 32},
			}},
			{"network", {
				{"otype", "FullyFusedMLP"},
				// {"otype", "CutlassMLP"},
				{"n_neurons", 64},
				{"n_hidden_layers", 4},
				{"activation", "ReLU"},
				{"output_activation", "None"},
			}},
		};

		if (argc >= 3) {
			std::cout << "Loading custom json config '" << argv[2] << "'." << std::endl;
			std::ifstream f{argv[2]};
			config = json::parse(f, nullptr, true, /*skip_comments=*/true);
		}

		// First step: load an image that we'd like to learn
		int width, height;
		GPUMemory<float> image = load_image(argv[1], width, height);

		// Second step: create a cuda texture out of this image. It'll be used to generate training data efficiently on the fly
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypePitch2D;
		resDesc.res.pitch2D.devPtr = image.data();
		resDesc.res.pitch2D.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		resDesc.res.pitch2D.width = width;
		resDesc.res.pitch2D.height = height;
		resDesc.res.pitch2D.pitchInBytes = width * 4 * sizeof(float);

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.normalizedCoords = true;
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.addressMode[2] = cudaAddressModeClamp;

		cudaTextureObject_t texture;
		CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));

		// Third step: sample a reference image to dump to disk. Visual comparison of this reference image and the learned
		//             function will be eventually possible.

		int sampling_width = width;
		int sampling_height = height;

		// Uncomment to fix the resolution of the training task independent of input image
		// int sampling_width = 1024;
		// int sampling_height = 1024;

		uint32_t n_coords = sampling_width * sampling_height;
		uint32_t n_coords_padded = next_multiple(n_coords, batch_size_granularity);

		GPUMemory<float> sampled_image(n_coords * 3);
		GPUMemory<float> xs_and_ys(n_coords_padded * 2);

		std::vector<float> host_xs_and_ys(n_coords * 2);
		for (int y = 0; y < sampling_height; ++y) {
			for (int x = 0; x < sampling_width; ++x) {
				int idx = (y * sampling_width + x) * 2;
				host_xs_and_ys[idx+0] = (float)(x + 0.5) / (float)sampling_width;
				host_xs_and_ys[idx+1] = (float)(y + 0.5) / (float)sampling_height;
			}
		}

		xs_and_ys.copy_from_host(host_xs_and_ys.data());

		linear_kernel(eval_image<3>, 0, nullptr, n_coords, texture, xs_and_ys.data(), sampled_image.data());

		save_image(sampled_image.data(), sampling_width, sampling_height, 3, 3, "reference.jpg");

		// Fourth step: train the model by sampling the above image and optimizing an error metric

		// Various constants for the network and optimization
		const uint32_t batch_size = 1 << 18;
		const uint32_t n_training_steps = argc >= 4 ? atoi(argv[3]) : 10000000;
		const uint32_t n_input_dims = 2; // 2-D image coordinate
		const uint32_t n_output_dims = 3; // RGB color

		cudaStream_t inference_stream;
		CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
		cudaStream_t training_stream = inference_stream;

		default_rng_t rng{1337};

		// Auxiliary matrices for training
		GPUMatrix<float> training_target(n_output_dims, batch_size);
		GPUMatrix<float> training_batch(n_input_dims, batch_size);

		// Auxiliary matrices for evaluation
		GPUMatrix<float> prediction(n_output_dims, n_coords_padded);
		GPUMatrix<float> inference_batch(xs_and_ys.data(), n_input_dims, n_coords_padded);

		json encoding_opts = config.value("encoding", json::object());
		json loss_opts = config.value("loss", json::object());
		json optimizer_opts = config.value("optimizer", json::object());
		json network_opts = config.value("network", json::object());

		std::shared_ptr<Loss<precision_t>> loss{create_loss<precision_t>(loss_opts)};
		std::shared_ptr<Optimizer<precision_t>> optimizer{create_optimizer<precision_t>(optimizer_opts)};
		std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = std::make_shared<NetworkWithInputEncoding<precision_t>>(n_input_dims, n_output_dims, encoding_opts, network_opts);

		auto trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		float tmp_loss = 0;
		uint32_t tmp_loss_counter = 0;

		std::cout << "Beginning optimization with " << n_training_steps << " training steps." << std::endl;

		uint32_t interval = 10;

		for (uint32_t i = 0; i < n_training_steps; ++i) {
			bool print_loss = i % interval == 0;
			bool visualize_learned_func = argc < 5 && i % interval == 0;

			// Compute reference values at random coordinates
			{
				generate_random_uniform<float>(training_stream, rng, batch_size * n_input_dims, training_batch.data());
				linear_kernel(eval_image<n_output_dims>, 0, training_stream, batch_size, texture, training_batch.data(), training_target.data());
			}

			// Training step
			{
				auto ctx = trainer->training_step(training_stream, training_batch, training_target);

				if (i % std::min(interval, (uint32_t)100) == 0) {
					tmp_loss += trainer->loss(training_stream, *ctx);
					++tmp_loss_counter;
				}
			}

			// Debug outputs
			{
				if (print_loss) {
					std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
					std::cout << "Step#" << i << ": " << "loss=" << tmp_loss/(float)tmp_loss_counter << " time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

					tmp_loss = 0;
					tmp_loss_counter = 0;
				}

				if (visualize_learned_func) {
					network->inference(inference_stream, inference_batch, prediction);
					auto filename = fmt::format("{}.jpg", i);
					std::cout << "Writing '" << filename << "'... ";
					save_image(prediction.data(), sampling_width, sampling_height, 3, n_output_dims, filename);
					std::cout << "done." << std::endl;
				}

				// Don't count visualizing as part of timing
				// (assumes visualize_learned_pdf is only true when print_loss is true)
				if (print_loss) {
					begin = std::chrono::steady_clock::now();
				}
			}

			if (print_loss && i > 0 && interval < 1000) {
				interval *= 10;
			}
		}

		// Dump final image if a name was specified
		if (argc >= 5) {
			network->inference(inference_stream, inference_batch, prediction);
			save_image(prediction.data(), sampling_width, sampling_height, 3, n_output_dims, argv[4]);
		}

		free_all_gpu_memory_arenas();

		// If only the memory arenas pertaining to a single stream are to be freed, use
		//free_gpu_memory_arena(stream);
	} catch (std::exception& e) {
		std::cout << "Uncaught exception: " << e.what() << std::endl;
	}

	return EXIT_SUCCESS;
}

