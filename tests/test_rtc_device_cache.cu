/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "test_common.h"

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

using namespace tcnn;

TEST_CASE("JIT inference caches kernels per CUDA device", "[jit][multi-gpu]") {
	tcnn_test_setup();

	if (cuda_device_count() < 2 || !supports_jit_fusion(0) || !supports_jit_fusion(1)) {
		WARN("Test requires two CUDA devices with JIT fusion support.");
		return;
	}

	const int original_device = cuda_device();
	ScopeGuard restore_device{[original_device]() { set_cuda_device(original_device); }};
	auto encoding = default_encoding<float>(3, "Identity");
	const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	std::vector<float> input_host(3 * batch_size);
	for (size_t i = 0; i < input_host.size(); ++i) {
		input_host[i] = (float)i / input_host.size();
	}

	auto run = [&](int device, bool jit) {
		set_cuda_device(device);
		GPUMatrix<float> input{3, batch_size};
		GPUMatrix<float> output{encoding->padded_output_width(), batch_size};
		CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));
		encoding->set_jit_fusion(jit);
		encoding->inference_mixed_precision(input, output);
		if (jit) {
			REQUIRE(encoding->jit_fusion());
		}
		return output.to_cpu_vector();
	};

	const auto native_0 = run(0, false);
	const auto native_1 = run(1, false);
	const auto jit_0 = run(0, true);
	const auto jit_1 = run(1, true);
	const auto jit_0_again = run(0, true);

	vector_match_rae(native_0, jit_0, 1e-6);
	vector_match_rae(native_1, jit_1, 1e-6);
	vector_match_rae(native_0, jit_0_again, 1e-6);
}

TEST_CASE("Combined network JIT caches conversion kernels per CUDA device", "[jit][multi-gpu][network]") {
	tcnn_test_setup();

	if (cuda_device_count() < 2 || !supports_jit_fusion(0) || !supports_jit_fusion(1)) {
		WARN("Test requires two CUDA devices with JIT fusion support.");
		return;
	}

	using T = network_precision_t;
	const auto network_type = GENERATE("FullyFusedMLP", "CutlassMLP");
	const int original_device = cuda_device();
	ScopeGuard restore_device{[original_device]() { set_cuda_device(original_device); }};
	json network_config = {
		{"otype", network_type},
		{"activation", "ReLU"},
		{"output_activation", "None"},
		{"n_neurons", 64},
		{"n_hidden_layers", 1},
	};
	auto model = std::make_shared<NetworkWithInputEncoding<T>>(16, 16, json{{"otype", "Identity"}}, network_config);
	const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	std::vector<float> input_host(16 * batch_size);
	for (size_t i = 0; i < input_host.size(); ++i) {
		input_host[i] = (float)(i % 31) / 31.0f;
	}
	std::vector<T> params_host(model->n_params());
	for (size_t i = 0; i < params_host.size(); ++i) {
		params_host[i] = (T)(((int)(i % 17) - 8) * 0.001f);
	}

	auto run = [&](int device, bool jit) {
		set_cuda_device(device);
		GPUMatrix<float> input{16, batch_size};
		GPUMatrix<T> output{model->padded_output_width(), batch_size};
		GPUMemory<T> params{model->n_params()};
		CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));
		params.copy_from_host(params_host);
		model->set_params(params.data(), params.data(), params.data());
		model->set_jit_fusion(jit);
		model->inference_mixed_precision(input, output);
		if (jit) {
			REQUIRE(model->jit_fusion());
		}
		return output.to_cpu_vector();
	};

	const auto native_0 = run(0, false);
	const auto native_1 = run(1, false);
	const auto jit_0 = run(0, true);
	const auto jit_1 = run(1, true);
	const auto jit_0_again = run(0, true);

	vector_match_rae(native_0, jit_0, 1e-2);
	vector_match_rae(native_1, jit_1, 1e-2);
	vector_match_rae(native_0, jit_0_again, 1e-2);
}
