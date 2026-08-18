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

/** @file   test_networks.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  Test various invariances of neural networks. E.g. that `inference`
 *          and `inference_mixed_precision` produce the same results, that JIT
 *          produces the same result as no-JIT, as well as the same for derivatives.
 */

 #include "test_common.h"

#include <tiny-cuda-nn/network.h>

using namespace tcnn;

TEST_CASE("Various invariance checks for neural networks", "[network][jit]") {
	using T = network_precision_t;

	tcnn_test_setup();

	std::vector<uint32_t> input_sizes = {16, 32, 48, 64, 128};
	std::vector<uint32_t> hidden_sizes = {32, 64, 128};
	std::vector<uint32_t> output_sizes = {16, 32};
	std::vector<uint32_t> depths = {2};

	for (uint32_t n_in : input_sizes) {
		for (uint32_t n_hidden : hidden_sizes) {
			for (uint32_t n_out : output_sizes) {
				for (uint32_t depth : depths) {
					SECTION(fmt::format("MLP testing {}->{}x{}->{}", n_in, n_hidden, depth, n_out)) {
						json config = {
							{"otype", "CutlassMLP"},
							{"n_input_dims", n_in},
							{"n_neurons", n_hidden},
							{"n_output_dims", n_out},
							{"n_hidden_layers", depth},
						};

						std::shared_ptr<Network<T>> cutlass_mlp{create_network<T>(config)};
						SECTION("CutlassMLP") { test_differentiable_object<T, T, T>(cutlass_mlp); }

						if (MIN_GPU_ARCH > 70) {
							config["otype"] = "FullyFusedMLP";
							std::shared_ptr<Network<T>> fully_fused_mlp{create_network<T>(config)};
							SECTION("FullyFusedMLP") { test_differentiable_object<T, T, T>(fully_fused_mlp); }

							// This would be a good place to check that FullyFusedMLP produces the same
							// results as CutlassMLP if this was not already implicitly checked in the JIT
							// tests from within `test_differentiable_object`, because the JIT is the same
							// across CutlassMLP and FullyFusedMLP.
						}
					}
				}
			}
		}
	}
}

TEST_CASE("FullyFusedMLP non-JIT double backward", "[network][double-backward]") {
	using T = network_precision_t;

	tcnn_test_setup();
	if (MIN_GPU_ARCH <= 70) {
		return;
	}

	const auto [input_width, hidden_width, output_width, batch_size] = GENERATE(table<uint32_t, uint32_t, uint32_t, uint32_t>({
		{32, 64, 16, 256},
		{16, 16, 1, 256},
		{32, 32, 16, 256},
		{48, 64, 17, 256},
		{128, 128, 32, 512},
	}));
	CAPTURE(input_width, hidden_width, output_width, batch_size);

	json config = {
		{"otype", "FullyFusedMLP"},
		{"activation", "ReLU"},
		{"output_activation", "None"},
		{"n_input_dims", input_width},
		{"n_output_dims", output_width},
		{"n_neurons", hidden_width},
		{"n_hidden_layers", 1},
	};

	std::shared_ptr<Network<T>> network{create_network<T>(config)};
	std::shared_ptr<Optimizer<T>> optimizer{create_optimizer<T>(json::object())};
	std::shared_ptr<Loss<T>> loss{create_loss<T>(json::object())};
	auto trainer = std::make_shared<Trainer<T, T, T>>(network, optimizer, loss);
	network->set_jit_fusion(false);

	pcg32 rng{0xdeadbeef};
	GPUMatrix<T> input{network->input_width(), batch_size};
	GPUMatrix<T> output{network->padded_output_width(), batch_size};
	GPUMatrix<T> output_gradient{network->padded_output_width(), batch_size};
	input.initialize_uniform(rng, -1.0f, 1.0f);
	output_gradient.initialize_uniform(rng, -0.01f, 0.01f);

	auto ctx = network->forward(input, &output, false, true);
	auto read_gradients = [&]() {
		std::vector<T> result(network->n_params());
		CUDA_CHECK_THROW(cudaMemcpy(result.data(), trainer->param_gradients(), result.size() * sizeof(T), cudaMemcpyDeviceToHost));
		return result;
	};
	auto require_matching_max_error = [](const std::vector<T>& expected, const std::vector<T>& actual, size_t begin, size_t end) {
		REQUIRE(expected.size() == actual.size());
		float max_expected = 0.0f;
		float max_error = 0.0f;
		for (size_t i = begin; i < end; ++i) {
			max_expected = std::max(max_expected, std::abs((float)expected[i]));
			max_error = std::max(max_error, std::abs((float)expected[i] - (float)actual[i]));
		}
		CAPTURE(begin, end, max_expected, max_error);
		REQUIRE(max_expected > 0.0f);
		REQUIRE(max_error < max_expected * 2e-2f + 1e-5f);
	};

	// ReLU is positively homogeneous. With dL/ddLdx = x, its parameter double gradient
	// equals the ordinary parameter gradient for the same output gradient.
	network->backward(*ctx, input, output, output_gradient, nullptr, false, GradientMode::Overwrite);
	const auto reference = read_gradients();
	network->backward_backward_input(*ctx, input, input, output_gradient, nullptr, nullptr, false, GradientMode::Overwrite);
	const auto double_backward = read_gradients();
	vector_match_rae(reference, double_backward, 2e-2, 0.999, true);
	const size_t input_weight_count = input_width * hidden_width;
	require_matching_max_error(reference, double_backward, 0, input_weight_count);
	require_matching_max_error(reference, double_backward, input_weight_count, reference.size());

	GPUMatrix<T> upstream_gradient{network->padded_output_width(), batch_size};
	network->backward_backward_input(*ctx, input, input, output_gradient, &upstream_gradient, nullptr, false, GradientMode::Ignore);
	const auto expected_upstream_gradient = output.to_cpu_vector();
	const auto actual_upstream_gradient = upstream_gradient.to_cpu_vector();
	vector_match_rae(expected_upstream_gradient, actual_upstream_gradient, 2e-2, 0.999, true);
	require_matching_max_error(expected_upstream_gradient, actual_upstream_gradient, 0, expected_upstream_gradient.size());

	if (input_width != 32 || hidden_width != 64 || output_width != 16 || batch_size != BATCH_SIZE_GRANULARITY) {
		return;
	}

	network->backward_backward_input(*ctx, input, input, output_gradient, nullptr, nullptr, false, GradientMode::Accumulate);
	const auto accumulated = read_gradients();
	std::vector<float> doubled(double_backward.size());
	std::transform(double_backward.begin(), double_backward.end(), doubled.begin(), [](T value) { return 2.0f * (float)value; });
	vector_match_rae(doubled, accumulated, 2e-2, 0.999, true);

	network->backward_backward_input(*ctx, input, input, output_gradient, nullptr, nullptr, false, GradientMode::Ignore);
	REQUIRE(read_gradients() == accumulated);

	GPUMatrix<T> unsupported_input_gradient{network->input_width(), batch_size};
	REQUIRE_THROWS_WITH(
		network->backward_backward_input(*ctx, input, input, output_gradient, nullptr, &unsupported_input_gradient),
		"FullyFusedMLP non-JIT double backward does not support input Hessians."
	);

	GPUMatrix<T> empty_input{network->input_width(), 0};
	GPUMatrix<T> empty_output_gradient{network->padded_output_width(), 0};
	network->backward_backward_input(*ctx, empty_input, empty_input, empty_output_gradient, nullptr, nullptr, false, GradientMode::Accumulate);
	REQUIRE(read_gradients() == accumulated);
	network->backward_backward_input(*ctx, empty_input, empty_input, empty_output_gradient, nullptr, nullptr, false, GradientMode::Overwrite);
	const auto empty_gradients = read_gradients();
	REQUIRE(std::all_of(empty_gradients.begin(), empty_gradients.end(), [](T value) { return (float)value == 0.0f; }));
}
