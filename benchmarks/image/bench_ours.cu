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

/** @file   bench-ours.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  Generates performance data for comparison with TensorFlow.
 */

#include <tiny-cuda-nn/common_device.h>

#include <tiny-cuda-nn/config.h>

#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/gpu_matrix.h>

#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>

#include <tiny-cuda-nn/trainer.h>

#include <fmt/core.h>

#include <stbi/stbi_wrapper.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fstream>


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
__global__ void eval_image(uint32_t n_elements, cudaTextureObject_t texture, bool filter, int width, int height, float* __restrict__ xs_and_ys, float* __restrict__ result) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	uint32_t output_idx = i * stride;
	uint32_t input_idx = i * 2;

	float2 pos = {xs_and_ys[input_idx], xs_and_ys[input_idx+1]};
	if (!filter) {
		pos.x = (roundf(pos.x * width - 0.5f) + 0.5f) / width;
		pos.y = (roundf(pos.y * height - 0.5f) + 0.5f) / height;
	}

	float4 val = tex2D<float4>(texture, pos.x, pos.y);
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

		if (argc < 3) {
			std::cout << "USAGE: " << argv[0] << " " << "path-to-image.jpg path-to-config.json" << std::endl;
			std::cout << "A sample image is provided in 'data/images'." << std::endl;
			return 0;
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

		default_rng_t rng{1337};

		// Third step: sample a reference image to dump to disk. Visual comparison of this reference image and the learned
		//             function will be eventually possible.

		int sampling_width = 1024;
		int sampling_height = 1024;

		uint32_t n_coords = sampling_width * sampling_height;

		GPUMemory<float> sampled_image(n_coords * 3);
		GPUMemory<float> xs_and_ys(n_coords * 2);

		std::vector<float> host_xs_and_ys(n_coords * 2);
		for (int y = 0; y < sampling_height; ++y) {
			for (int x = 0; x < sampling_width; ++x) {
				int idx = (y * sampling_width + x) * 2;
				host_xs_and_ys[idx+0] = (float)(x + 0.5) / (float)sampling_width;
				host_xs_and_ys[idx+1] = (float)(y + 0.5) / (float)sampling_height;
			}
		}

		xs_and_ys.copy_from_host(host_xs_and_ys.data());

		bool filter = false;

		eval_image<3><<<n_blocks_linear(n_coords), n_threads_linear>>>(n_coords, texture, filter, width, height, xs_and_ys.data(), sampled_image.data());

		save_image(sampled_image.data(), sampling_width, sampling_height, 3, 3, "reference.jpg");

		// Fourth step: train the model by sampling the above image and optimizing relative squared error using Adam.
		std::vector<uint32_t> batch_sizes = {1 << 21, 1 << 20, 1 << 19, 1 << 18, 1 << 17, 1 << 16, 1 << 15, 1 << 14};
		std::vector<std::string> methods = {"fully_fused", "cutlass"};
		json bench_result;

		for (std::string method : methods) {
			bench_result[method] = json::array();
			for (uint32_t batch_size : batch_sizes) {
				// Various constants for the network and optimization
				uint32_t n_iterations = std::max(1000 * (1 << 18) / batch_size, 250u);
				uint32_t n_iterations_warmup = n_iterations / 2;

				const uint32_t num_dims_encoded = 2;
				const uint32_t num_output_dims = 3;

				// Input & corresponding RNG
				GPUMemory<float> batch(batch_size * num_dims_encoded);

				cudaStream_t inference_stream;
				CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
				cudaStream_t training_stream = inference_stream;

				std::ifstream f{argv[2]};
				json config = json::parse(f, nullptr, true, /*skip_comments=*/true);

				json encoding_opts = config.value("encoding", json::object());

				// Auxiliary matrices for training
				GPUMatrix<float> bench_target(num_output_dims, batch_size);

				// Auxiliary matrices for evaluation
				GPUMemory<float> prediction_data(num_output_dims * n_coords);
				GPUMatrix<float> prediction(prediction_data.data(), num_output_dims, n_coords);

				json loss_opts = config.value("loss", json::object());
				json optimizer_opts = config.value("optimizer", json::object());
				json network_opts = config.value("network", json::object());
				network_opts["otype"] = method == "cutlass" ? "MLP" : "FullyFusedMLP";

				std::shared_ptr<Loss<precision_t>> loss{create_loss<precision_t>(loss_opts)};
				std::shared_ptr<Optimizer<precision_t>> optimizer{create_optimizer<precision_t>(optimizer_opts)};

				std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = std::make_shared<NetworkWithInputEncoding<precision_t>>(num_dims_encoded, num_output_dims, encoding_opts, network_opts);

				auto trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);

				std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

				float tmp_loss = 0;
				uint32_t tmp_loss_counter = 0;

				uint32_t print_interval = n_iterations / 10;
				const uint32_t STEPS_INCREMENT = 5;

				double mean_training_throughput = 0;
				size_t mean_counter = 0;

				for (uint32_t i = 0; i < n_iterations; i += STEPS_INCREMENT) {
					bool print_loss = i % print_interval == 0;

					for (uint32_t j = 0; j < STEPS_INCREMENT; ++j) {
						// Compute reference values at random coordinates
						generate_random_uniform<float>(training_stream, rng, batch_size * num_dims_encoded, batch.data());
						linear_kernel(eval_image<num_output_dims>, 0, training_stream, batch_size, texture, filter, width, height, batch.data(), bench_target.data());

						auto ctx = trainer->training_step(training_stream, GPUMatrix<float>{batch.data(), num_dims_encoded, batch_size}, bench_target);
						if (j == STEPS_INCREMENT-1) {
							tmp_loss += trainer->loss(training_stream, *ctx);
							++tmp_loss_counter;
						}
					}

					// Debug outputs
					if (print_loss) {
						CUDA_CHECK_THROW(cudaDeviceSynchronize());
						std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
						auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
						double throughput = print_interval * batch_size / ((double)microseconds / 1000000.0);
						std::cout << "Iteration#" << i << ": " << "loss=" << tmp_loss/(float)tmp_loss_counter << " time=" << microseconds << "[µs] thp=" << throughput << "/s" << std::endl;

						begin = end;
						tmp_loss = 0;
						tmp_loss_counter = 0;

						if (i >= n_iterations_warmup) {
							mean_training_throughput += throughput;
							++mean_counter;
						}
					}
				}

				mean_training_throughput /= (double)mean_counter;

				// Dump learned image for sanity checking
				network->inference(inference_stream, GPUMatrix<float>{xs_and_ys.data(), num_dims_encoded, n_coords}, prediction);

				save_image(prediction_data.data(), sampling_width, sampling_height, 3, num_output_dims, fmt::format("{}-after-{}-iters-{}.jpg", batch_size, n_iterations, method));

				std::cout << "Finished training benchmark. Mean throughput is " << mean_training_throughput << "/s. Waiting 10 seconds for GPU to cool down." << std::endl;
				std::this_thread::sleep_for(std::chrono::seconds{10});

				// Inference benchmark
				double mean_inference_throughput = 0;
				mean_counter = 0;

				print_interval *= 5;
				n_iterations *= 5;
				n_iterations_warmup *= 5;
				for (uint32_t i = 0; i < n_iterations; ++i) {
					bool print_loss = i % print_interval == 0;

					// Compute inference values at random coordinates
					generate_random_uniform<float>(inference_stream, rng, batch_size * num_dims_encoded, batch.data());

					// Inference step
					network->inference(inference_stream, GPUMatrix<float>{batch.data(), num_dims_encoded, batch_size}, bench_target);

					// Debug outputs
					if (print_loss) {
						cudaDeviceSynchronize();
						std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
						auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
						double throughput = print_interval * batch_size / ((double)microseconds / 1000000.0);
						std::cout << "Iteration#" << i << ": " << "time=" << microseconds << "[µs] thp=" << throughput << "/s" << std::endl;

						begin = end;

						if (i >= n_iterations_warmup) {
							mean_inference_throughput += throughput;
							++mean_counter;
						}
					}
				}

				mean_inference_throughput /= (double)mean_counter;

				std::cout << "Finished inference benchmark. Mean throughput is " << mean_inference_throughput << "/s. Waiting 10 seconds for GPU to cool down." << std::endl;
				std::this_thread::sleep_for(std::chrono::seconds{10});

				bench_result[method].push_back({
					{"batch_size", batch_size},
					{"training_throughput", mean_training_throughput},
					{"inference_throughput", mean_inference_throughput},
				});
			}
		}

		std::string json_string = bench_result.dump(4);
		std::ofstream out{"bench_result_ours.json"};
		out << json_string;
	} catch (std::exception& e) {
		std::cout << "Uncaught exception: " << e.what() << std::endl;
	}


	return EXIT_SUCCESS;
}

