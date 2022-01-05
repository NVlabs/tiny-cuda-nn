/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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
 *//*
 */

/** @file   bench-ours.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  Generates performance data for comparison with TensorFlow.
 */

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/encodings/oneblob.h>

#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>

#include <tiny-cuda-nn/trainer.h>

#include <tinyexr/tinyexr.h>

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


bool SaveEXR(const float* data, int width, int height, int nChannels, int channelStride, const char* outfilename) {
	EXRHeader header;
	InitEXRHeader(&header);

	EXRImage image;
	InitEXRImage(&image);

	image.num_channels = nChannels;

	std::vector<std::vector<float>> images(nChannels);
	std::vector<float*> image_ptr(nChannels);
	for (int i = 0; i < nChannels; ++i) {
		images[i].resize(width * height);
	}

	for (int i = 0; i < nChannels; ++i) {
		image_ptr[i] = images[nChannels - i - 1].data();
	}

	for (size_t i = 0; i < (size_t)width * height; i++) {
		for (int c = 0; c < nChannels; ++c) {
			images[c][i] = data[channelStride*i+c];
		}
	}

	image.images = (unsigned char**)image_ptr.data();
	image.width = width;
	image.height = height;

	header.num_channels = nChannels;
	header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
	// Must be (A)BGR order, since most of EXR viewers expect this channel order.
	strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
	if (nChannels > 1) {
		strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
	}
	if (nChannels > 2) {
		strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';
	}
	if (nChannels > 3) {
		strncpy(header.channels[3].name, "A", 255); header.channels[3].name[strlen("A")] = '\0';
	}

	header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
	header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
	for (int i = 0; i < header.num_channels; i++) {
		header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
		header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
	}

	const char* err = NULL; // or nullptr in C++11 or later.
	int ret = SaveEXRImageToFile(&image, &header, outfilename, &err);
	if (ret != TINYEXR_SUCCESS) {
		fprintf(stderr, "Save EXR err: %s\n", err);
		FreeEXRErrorMessage(err); // free's buffer for an error message
		return ret;
	}
	printf("Saved exr file. [ %s ] \n", outfilename);

	free(header.channels);
	free(header.pixel_types);
	free(header.requested_pixel_types);
	return true;
}


GPUMemory<float> load_image(const std::string& filename, int& width, int& height) {
	float* out; // width * height * RGBA
	const char* err = nullptr;

	int ret = LoadEXR(&out, &width, &height, filename.c_str(), &err);

	if (ret != TINYEXR_SUCCESS) {
		if (err) {
			std::string error_message = std::string("Failed to load EXR image: ") + err;
			FreeEXRErrorMessage(err);
			throw std::runtime_error(error_message);
		} else {
			throw std::runtime_error("Failed to load EXR image");
		}
	}

	GPUMemory<float> result(width * height * 4);
	result.copy_from_host(out);
	free(out); // release memory of image data

	return result;
}

template <typename T>
void save_image(const GPUMemory<T>& image, int width, int height, int n_channels, int channel_stride, const std::string& filename) {
	std::vector<T> host_data(image.size());
	image.copy_to_host(host_data.data());

	std::vector<float> float_host_data(host_data.size());
	for (size_t i = 0; i < host_data.size(); ++i) {
		float_host_data[i] = (float)host_data[i];
	}

	SaveEXR(float_host_data.data(), width, height, n_channels, channel_stride, filename.c_str());
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
	if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
		std::cout << "Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
		return -1;
	}

	cudaDeviceProp props;

	cudaError_t error = cudaGetDeviceProperties(&props, 0);
	if (error != cudaSuccess) {
		std::cout << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
		return -1;
	}

	if (!((props.major * 10 + props.minor) >= 75)) {
		std::cout << "Turing Tensor Core operations must be run on a machine with compute capability at least 75."
					<< std::endl;
		return -1;
	}

	if (argc < 3) {
		std::cout << "USAGE: " << argv[0] << " " << "path-to-image.exr path-to-config.json" << std::endl;
		std::cout << "Sample EXR files are provided in 'data/images'." << std::endl;
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

	cudaResourceViewDesc viewDesc;
	memset(&viewDesc, 0, sizeof(viewDesc));
	viewDesc.format = cudaResViewFormatFloat4;
	viewDesc.width = width;
	viewDesc.height = height;

	cudaTextureObject_t texture;
	CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &resDesc, &texDesc, &viewDesc));

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

	save_image(sampled_image, sampling_width, sampling_height, 3, 3, "reference.exr");

	// Fourth step: train the model by sampling the above image and optimizing relative squared error using Adam.
	try {
		std::vector<uint32_t> batch_sizes = {1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21};
		std::vector<std::string> methods = {"cutlass", "fully_fused"};
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
				std::shared_ptr<Encoding<precision_t>> encoding{create_encoding<precision_t>(num_dims_encoded, encoding_opts, 16)};
				const uint32_t padded_num_input_dims = encoding->num_encoded_dims();

				// Auxiliary matrices for training
				GPUMatrix<precision_t> bench_obe_out(padded_num_input_dims, batch_size);
				GPUMatrix<float> bench_target(num_output_dims, batch_size);

				// Auxiliary matrices for evaluation
				GPUMatrix<precision_t> eval_obe_out(padded_num_input_dims, n_coords);
				GPUMemory<float> prediction_data(num_output_dims * n_coords);
				GPUMatrix<float> prediction(prediction_data.data(), num_output_dims, n_coords);

				json loss_opts = config.value("loss", json::object());
				json optimizer_opts = config.value("optimizer", json::object());
				json network_opts = config.value("network", json::object());
				network_opts["otype"] = method == "cutlass" ? "MLP" : "FullyFusedMLP";
				network_opts["n_output_dims"] = num_output_dims;
				network_opts["n_input_dims"] = padded_num_input_dims;

				std::shared_ptr<Loss<precision_t>> loss{create_loss<precision_t>(loss_opts)};
				std::shared_ptr<Optimizer<precision_t>> optimizer{create_optimizer<precision_t>(optimizer_opts)};
				std::shared_ptr<Network<precision_t>> network{create_network<precision_t>(network_opts)};

				auto trainer = std::make_shared<Trainer<precision_t, precision_t>>(network, optimizer, loss);

				std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

				float tmp_loss = 0;
				uint32_t tmp_loss_counter = 0;

				uint32_t print_interval = n_iterations / 10;
				const uint32_t STEPS_INCREMENT = 5;

				double mean_training_throughput = 0;
				size_t mean_counter = 0;

				for (uint32_t i = 0; i < n_iterations; i += STEPS_INCREMENT) {
					bool print_loss = i % print_interval == 0;

					float loss_value;
					for (uint32_t j = 0; j < STEPS_INCREMENT; ++j) {
						// Compute reference values at random coordinates
						generate_random_uniform<float>(training_stream, rng, batch_size * num_dims_encoded, batch.data());

						// Training step
						float* p_loss = j == (STEPS_INCREMENT - 1) ? &loss_value : nullptr;
						encoding->encode(training_stream, batch_size, {batch.data(), num_dims_encoded}, {bench_obe_out.data(), num_output_dims});
						trainer->training_step(training_stream, bench_obe_out, bench_target, p_loss);
					}

					tmp_loss += loss_value;
					++tmp_loss_counter;

					// Debug outputs
					if (print_loss) {
						cudaDeviceSynchronize();
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
				encoding->encode(inference_stream, n_coords, {xs_and_ys.data(), num_dims_encoded}, {eval_obe_out.data(), num_output_dims});
				network->inference(inference_stream, eval_obe_out, prediction);

				save_image(prediction_data, sampling_width, sampling_height, 3, num_output_dims, std::to_string(batch_size) + "-after-" + std::to_string(n_iterations) + "-iters-" + method + ".exr");

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
					encoding->encode(inference_stream, batch_size, {batch.data(), num_dims_encoded}, {bench_obe_out.data(), num_output_dims});
					network->inference(inference_stream, bench_obe_out, bench_target);

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

