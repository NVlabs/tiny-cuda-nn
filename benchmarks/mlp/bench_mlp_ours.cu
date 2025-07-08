/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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
 *  @brief  Generates performance data for tiny-cuda-nn's supported MLPs.
 */

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/trainer.h>

#include <fmt/core.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace tcnn;
using precision_t = network_precision_t;

int main(int argc, char* argv[]) {
	try {
		uint32_t compute_capability = cuda_compute_capability();
		if (compute_capability < MIN_GPU_ARCH) {
			std::cerr
				<< "Warning: Insufficient compute capability " << compute_capability << " detected. "
				<< "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly." << std::endl;
		}

		default_rng_t rng{1337};

		// Fourth step: train the model by sampling the above image and optimizing relative squared error using Adam.
		std::vector<uint32_t> batch_sizes = {1 << 20};
		std::vector<std::string> methods = {"fully_fused_jit", "fully_fused", "cutlass"};
		json bench_result;

		for (std::string method : methods) {
			bench_result[method] = json::array();
			for (uint32_t batch_size : batch_sizes) {
				// Various constants for the network and optimization
				uint32_t n_iterations = std::max(5000u * (1u << 18) / batch_size, 2000u);
				uint32_t n_iterations_warmup = n_iterations / 2;

				const uint32_t n_hidden_layers = 3;
				const uint32_t dim = 32; // 64 and 128 are also efficient
				const uint32_t n_input_dims = dim;
				const uint32_t n_output_dims = dim;

				// Input. Most efficient in RM layout when used with JIT, CM layout otherwise.
				GPUMatrixDynamic<precision_t> batch(n_input_dims, batch_size, method == "fully_fused_jit" ? RM : CM);
				GPUMatrix<precision_t, RM> bench_target(n_output_dims, batch_size);

				cudaStream_t inference_stream;
				CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));

				json network_opts = {
					{"otype", method == "cutlass" ? "MLP" : "FullyFusedMLP"},
					{"n_input_dims", n_input_dims},
					{"n_output_dims", n_output_dims},
					{"n_neurons", dim},
					{"n_hidden_layers", n_hidden_layers},
					{"activation", "ReLU"},
					{"output_activation", "None"},
				};

				std::shared_ptr<Loss<precision_t>> loss{create_loss<precision_t>(nlohmann::json::object())};
				std::shared_ptr<Optimizer<precision_t>> optimizer{create_optimizer<precision_t>(nlohmann::json::object())};
				std::shared_ptr<Network<precision_t>> network{create_network<precision_t>(network_opts)};
				if (method == "fully_fused_jit") {
					network->set_jit_fusion(true);
				}

				auto trainer = std::make_shared<Trainer<precision_t, precision_t, precision_t>>(network, optimizer, loss);

				// Compute inference values at random coordinates
				generate_random_uniform<precision_t>(inference_stream, rng, batch_size * n_input_dims, batch.data());

				std::cout << "warming up... ";
				for (uint32_t i = 0; i < n_iterations_warmup; ++i) {
					network->inference_mixed_precision(inference_stream, batch, bench_target);
				}

				cudaDeviceSynchronize();

				std::cout << "benchmarking... " << std::endl;

				std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
				for (uint32_t i = 0; i < n_iterations; ++i) {
					network->inference_mixed_precision(inference_stream, batch, bench_target);
				}

				cudaDeviceSynchronize();

				std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
				double throughput = n_iterations * batch_size / ((double)microseconds / 1000000.0);

				std::cout << "  " << method << ": dim=" << dim << " time=" << microseconds << "[µs] thp=" << throughput << "/s time_per_batch=" << (batch_size / throughput) << "/s" << std::endl;
				std::cout << "waiting 10 seconds for GPU to cool down... ";
				std::this_thread::sleep_for(std::chrono::seconds{10});

				bench_result[method].push_back({
					{"batch_size", batch_size},
					{"inference_throughput", throughput},
				});
			}
		}

		std::cout << std::endl;

		std::string json_string = bench_result.dump(4);
		std::ofstream out{"bench_result_ours.json"};
		out << json_string;
	} catch (std::exception& e) {
		std::cout << "Uncaught exception: " << e.what() << std::endl;
	}

	return EXIT_SUCCESS;
}

