/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
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

/** @file   test_permuto.cu
 *  @brief  Test Permuto and hard level-of-detail encodings.
 */

#include "test_common.h"

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/encodings/multi_level_interface.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>

#include <array>
#include <limits>

using namespace tcnn;

namespace {

json permuto_config(uint32_t n_levels = 16, uint32_t log2_hashmap_size = 19) {
	return {
		{"otype",                "Permuto"         },
		{"n_levels",             n_levels          },
		{"n_features_per_level", 2                 },
		{"log2_hashmap_size",    log2_hashmap_size },
		{"per_level_scale",      1.4472692374403782},
		{"base_scale",           16.0              },
		{"interpolation",        "Linear"          },
		{"max_input_grad_dims",  3                 },
	};
}

json hard_lod_config(uint32_t n_levels = 16, uint32_t log2_hashmap_size = 19) {
	return {
		{"otype", "MultiLevelEncodingLoD"},
		{"lod_type", "Hard"},
		{"base", permuto_config(n_levels, log2_hashmap_size)},
	};
}

std::vector<float> hard_lod_input(uint32_t batch_size, const std::vector<float>& ratios) {
	std::vector<float> result(6 * batch_size);
	for (uint32_t row = 0; row < batch_size; ++row) {
		for (uint32_t dim = 0; dim < 5; ++dim) {
			result[row * 6 + dim] = 0.05f + (float)((row + dim) % 13) / 20.0f;
		}
		result[row * 6 + 5] = ratios[row % ratios.size()];
	}
	return result;
}

std::array<float, 2> reference_permuto_level(
	const std::array<float, 5>& position,
	const std::vector<float>& params,
	float scale,
	uint32_t seed,
	uint32_t level,
	uint32_t parameter_offset,
	uint32_t hashmap_size
) {
	pcg32 rng{seed};
	rng.advance(level * 5);
	std::array<float, 5> scales;
	std::array<float, 5> shifts;
	std::array<float, 6> elevated;
	std::array<int, 6> rem0{};
	std::array<int, 6> rank{};

	for (uint32_t dim = 0; dim < 5; ++dim) {
		scales[dim] = scale / std::sqrt((dim + 1) * (dim + 2));
		shifts[dim] = std::fma(rng.next_float(), 10.0f, -5.0f);
	}

	float sum = 0.0f;
	for (int dim = 5; dim > 0; --dim) {
		const uint32_t position_dim = static_cast<uint32_t>(dim - 1);
		const float coordinate = (position[position_dim] + shifts[position_dim]) * scales[position_dim];
		elevated[dim] = sum - dim * coordinate;
		sum += coordinate;
	}
	elevated[0] = sum;

	int rem0_sum = 0;
	for (uint32_t dim = 0; dim <= 5; ++dim) {
		const float value = elevated[dim] / 6.0f;
		const float up = std::ceil(value) * 6.0f;
		const float down = std::floor(value) * 6.0f;
		rem0[dim] = up - elevated[dim] < elevated[dim] - down ? static_cast<int>(up) : static_cast<int>(down);
		rem0_sum += rem0[dim];
	}
	rem0_sum /= 6;

	for (uint32_t dim = 0; dim < 5; ++dim) {
		const float difference = elevated[dim] - rem0[dim];
		for (uint32_t other_dim = dim + 1; other_dim <= 5; ++other_dim) {
			if (difference < elevated[other_dim] - rem0[other_dim]) {
				++rank[dim];
			} else {
				++rank[other_dim];
			}
		}
	}
	for (uint32_t dim = 0; dim <= 5; ++dim) {
		rank[dim] += rem0_sum;
		if (rank[dim] < 0) {
			rank[dim] += 6;
			rem0[dim] += 6;
		} else if (rank[dim] > 5) {
			rank[dim] -= 6;
			rem0[dim] -= 6;
		}
	}

	std::array<float, 7> barycentric{};
	for (uint32_t dim = 0; dim <= 5; ++dim) {
		const float delta = (elevated[dim] - rem0[dim]) / 6.0f;
		barycentric[5 - rank[dim]] += delta;
		barycentric[6 - rank[dim]] -= delta;
	}
	barycentric[0] += 1.0f + barycentric[6];

	std::array<float, 2> result{};
	for (uint32_t vertex = 0; vertex <= 5; ++vertex) {
		uint32_t hash = 0;
		for (uint32_t dim = 0; dim < 5; ++dim) {
			int coordinate = rem0[dim] + static_cast<int>(vertex);
			if (rank[dim] > static_cast<int>(5 - vertex)) {
				coordinate -= 6;
			}
			hash += static_cast<uint32_t>(coordinate);
			hash *= 2531011u;
		}
		const uint32_t parameter = parameter_offset + (hash % hashmap_size) * 2;
		for (uint32_t feature = 0; feature < 2; ++feature) {
			result[feature] = std::fma(barycentric[vertex], params[parameter + feature], result[feature]);
		}
	}
	return result;
}

std::array<float, 4>
	reference_permuto_forward(const std::array<float, 5>& position, const std::vector<float>& params, float base_scale, uint32_t seed) {
	std::array<float, 4> result{};
	constexpr uint32_t hashmap_size = 8;
	constexpr uint32_t parameters_per_level = hashmap_size * 2;
	for (uint32_t level = 0; level < 2; ++level) {
		const auto level_result = reference_permuto_level(
			position, params, std::ldexp(base_scale, static_cast<int>(level)), seed, level, level * parameters_per_level, hashmap_size
		);
		result[level * 2] = level_result[0];
		result[level * 2 + 1] = level_result[1];
	}
	return result;
}

} // namespace

TEST_CASE("Permuto validates its public configuration", "[encoding][permuto]") {
	using T = network_precision_t;
	REQUIRE_NOTHROW(create_encoding<T>(5, permuto_config(2, 4), 16));
	REQUIRE_NOTHROW(create_encoding<T>(6, hard_lod_config(), 16));
	REQUIRE_THROWS_AS(create_encoding<T>(4, permuto_config(), 16), std::runtime_error);

	const auto require_invalid = [&](const char* key, const json& value) {
		json config = permuto_config();
		config[key] = value;
		INFO("key=" << key << ", value=" << value);
		REQUIRE_THROWS_AS(create_encoding<T>(5, config, 16), std::runtime_error);
	};

	require_invalid("n_features_per_level", 0);
	require_invalid("n_features_per_level", 4);
	require_invalid("n_features_per_level", -1);
	require_invalid("n_features_per_level", 2.5);
	require_invalid("n_levels", 0);
	require_invalid("n_levels", 33);
	require_invalid("n_levels", -1);
	require_invalid("n_levels", 16.5);
	require_invalid("n_levels", std::numeric_limits<uint64_t>::max());
	require_invalid("log2_hashmap_size", -1);
	require_invalid("log2_hashmap_size", 19.5);
	require_invalid("log2_hashmap_size", 32);
	require_invalid("log2_hashmap_size", 27);
	require_invalid("log2_hashmap_size", std::numeric_limits<uint64_t>::max());
	require_invalid("max_input_grad_dims", -1);
	require_invalid("max_input_grad_dims", 3.5);
	require_invalid("max_input_grad_dims", 6);
	require_invalid("max_input_grad_dims", std::numeric_limits<uint64_t>::max());
	require_invalid("seed", -1);
	require_invalid("seed", 1.5);
	require_invalid("seed", std::numeric_limits<uint64_t>::max());
	require_invalid("base_scale", "invalid");
	require_invalid("base_scale", std::numeric_limits<double>::infinity());
	require_invalid("base_scale", std::numeric_limits<double>::quiet_NaN());
	require_invalid("base_scale", 1e30);
	require_invalid("per_level_scale", "invalid");
	require_invalid("per_level_scale", 0.0);
	require_invalid("per_level_scale", -1.0);
	require_invalid("per_level_scale", std::numeric_limits<double>::infinity());
	require_invalid("per_level_scale", std::numeric_limits<double>::quiet_NaN());
	require_invalid("per_level_scale", 1e30);
	require_invalid("interpolation", "Smoothstep");
	require_invalid("n_features", 32);
	require_invalid("n_grid_features", 32);
}

TEST_CASE("Permuto matches an independent CPU reference", "[encoding][permuto]") {
	tcnn_test_setup();

	json config = permuto_config(2, 3);
	config["base_scale"] = 1.0f;
	config["per_level_scale"] = 2.0f;
	config["max_input_grad_dims"] = 3;
	config["seed"] = 42;
	std::shared_ptr<Encoding<float>> encoding{create_encoding<float>(5, config, 2)};
	auto optimizer = std::shared_ptr<Optimizer<float>>{create_optimizer<float>(json::object())};
	auto loss = std::shared_ptr<Loss<float>>{create_loss<float>(json::object())};
	auto trainer = std::make_shared<Trainer<float, float, float>>(encoding, optimizer, loss);

	std::vector<float> params(encoding->n_params());
	for (size_t i = 0; i < params.size(); ++i) {
		params[i] = static_cast<float>(static_cast<int>((i * 17) % 29) - 14) / 128.0f;
	}
	CUDA_CHECK_THROW(cudaMemcpy(encoding->params(), params.data(), params.size() * sizeof(float), cudaMemcpyHostToDevice));

	constexpr uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	const std::array<float, 5> position = {0.137f, 0.271f, 0.419f, 0.583f, 0.731f};
	const std::array<float, 4> upstream = {0.75f, -0.5f, 0.25f, 1.0f};
	std::vector<float> input_host(5 * batch_size);
	std::vector<float> upstream_host(4 * batch_size, 0.0f);
	for (uint32_t element = 0; element < batch_size; ++element) {
		std::copy(position.begin(), position.end(), input_host.begin() + element * 5);
	}
	std::copy(upstream.begin(), upstream.end(), upstream_host.begin());

	GPUMatrix<float> input{5, batch_size};
	GPUMatrix<float> output{4, batch_size};
	GPUMatrix<float> output_gradient{4, batch_size};
	GPUMatrix<float> input_gradient{5, batch_size};
	CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(cudaMemcpy(output_gradient.data(), upstream_host.data(), output_gradient.n_bytes(), cudaMemcpyHostToDevice));
	auto context = encoding->forward(input, &output, false, true);
	encoding->backward(*context, input, output, output_gradient, &input_gradient, false, GradientMode::Overwrite);

	const auto reference_output = reference_permuto_forward(position, params, 1.0f, 42);
	REQUIRE(std::any_of(reference_output.begin(), reference_output.end(), [](float value) { return value != 0.0f; }));
	const auto output_host = output.to_cpu_vector();
	for (uint32_t element = 0; element < batch_size; ++element) {
		for (uint32_t feature = 0; feature < 4; ++feature) {
			REQUIRE(output_host[element * 4 + feature] == Approx(reference_output[feature]).margin(2e-5f).epsilon(2e-4f));
		}
	}

	const auto reference_loss = [&](const std::array<float, 5>& reference_position, const std::vector<float>& reference_params) {
		const auto reference = reference_permuto_forward(reference_position, reference_params, 1.0f, 42);
		float result = 0.0f;
		for (uint32_t feature = 0; feature < 4; ++feature) {
			result += reference[feature] * upstream[feature];
		}
		return result;
	};
	std::vector<float> parameter_gradient(params.size());
	CUDA_CHECK_THROW(
		cudaMemcpy(parameter_gradient.data(), encoding->gradients(), parameter_gradient.size() * sizeof(float), cudaMemcpyDeviceToHost)
	);
	constexpr float parameter_epsilon = 1.0f / 256.0f;
	bool has_parameter_gradient = false;
	for (size_t i = 0; i < params.size(); ++i) {
		auto lower = params;
		auto upper = params;
		lower[i] -= parameter_epsilon;
		upper[i] += parameter_epsilon;
		const float finite_difference = (reference_loss(position, upper) - reference_loss(position, lower)) / (2.0f * parameter_epsilon);
		has_parameter_gradient |= finite_difference != 0.0f;
		REQUIRE(parameter_gradient[i] == Approx(finite_difference).margin(5e-4f).epsilon(2e-3f));
	}
	REQUIRE(has_parameter_gradient);

	const auto input_gradient_host = input_gradient.to_cpu_vector();
	constexpr float input_epsilon = 1.0f / 1024.0f;
	bool has_input_gradient = false;
	for (uint32_t dim = 0; dim < 3; ++dim) {
		auto lower = position;
		auto upper = position;
		lower[dim] -= input_epsilon;
		upper[dim] += input_epsilon;
		const float expected = (reference_loss(upper, params) - reference_loss(lower, params)) / (2.0f * input_epsilon);
		lower[dim] = position[dim] - input_epsilon * 0.5f;
		upper[dim] = position[dim] + input_epsilon * 0.5f;
		const float half_interval = (reference_loss(upper, params) - reference_loss(lower, params)) / input_epsilon;
		REQUIRE(half_interval == Approx(expected).margin(1e-4f).epsilon(1e-3f));
		has_input_gradient |= expected != 0.0f;
		REQUIRE(input_gradient_host[dim] == Approx(expected).margin(1e-3f).epsilon(5e-3f));
	}
	REQUIRE(has_input_gradient);
	REQUIRE(input_gradient_host[3] == 0.0f);
	REQUIRE(input_gradient_host[4] == 0.0f);
	for (uint32_t element = 1; element < batch_size; ++element) {
		for (uint32_t dim = 0; dim < 5; ++dim) {
			REQUIRE(input_gradient_host[element * 5 + dim] == 0.0f);
		}
	}
}

TEST_CASE("Five-dimensional Permuto supports forward, backward, and optimization", "[encoding][permuto]") {
	tcnn_test_setup();

	const json config = permuto_config();

	using T = network_precision_t;
	REQUIRE_THROWS_AS(create_encoding<T>(4, config, 16), std::runtime_error);
	json unsupported_features_config = config;
	unsupported_features_config["n_features_per_level"] = 4;
	REQUIRE_THROWS_AS(create_encoding<T>(5, unsupported_features_config, 16), std::runtime_error);
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(5, config, 16)};

	REQUIRE(encoding->input_width() == 5);
	REQUIRE(encoding->output_width() == 32);
	REQUIRE(encoding->n_params() == 16777216);

	auto* multi_level = dynamic_cast<MultiLevelEncoding<T>*>(encoding.get());
	REQUIRE(multi_level != nullptr);
	for (uint32_t level = 0; level < 16; ++level) {
		REQUIRE(multi_level->level_params_offset(level) == level * (1u << 19));
		REQUIRE(multi_level->level_n_params(level) == (1u << 19));
	}
	REQUIRE(multi_level->level_params_offset(16) == 16u * (1u << 19));

	auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
	auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
	auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);

	const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	GPUMatrix<float> input{5, batch_size};
	GPUMatrix<float> input_gradient{5, batch_size};
	GPUMatrix<T> output{32, batch_size};
	GPUMatrix<T> output_gradient{32, batch_size};

	pcg32 rng{0xdeadbeef};
	input.initialize_uniform(rng, 0.001f, 0.999f);
	output_gradient.initialize_uniform(rng, -default_loss_scale<T>(), default_loss_scale<T>());
	std::vector<T> params_before(encoding->n_params());
	CUDA_CHECK_THROW(cudaMemcpy(params_before.data(), encoding->params(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost));

	auto context = encoding->forward(input, &output, false, true);
	encoding->backward(*context, input, output, output_gradient, &input_gradient);

	const auto output_host = output.to_cpu_vector();
	REQUIRE(std::any_of(output_host.begin(), output_host.end(), [](T value) { return (float)value != 0.0f; }));
	REQUIRE(std::all_of(output_host.begin(), output_host.end(), [](T value) { return std::isfinite((float)value); }));

	const auto input_gradient_host = input_gradient.to_cpu_vector();
	bool xyz_gradient_is_nonzero = false;
	for (uint32_t row = 0; row < batch_size; ++row) {
		for (uint32_t dim = 0; dim < 3; ++dim) {
			const float value = input_gradient_host[row * 5 + dim];
			REQUIRE(std::isfinite(value));
			xyz_gradient_is_nonzero |= value != 0.0f;
		}
		REQUIRE(input_gradient_host[row * 5 + 3] == 0.0f);
		REQUIRE(input_gradient_host[row * 5 + 4] == 0.0f);
	}
	REQUIRE(xyz_gradient_is_nonzero);

	std::vector<T> parameter_gradient_host(encoding->n_params());
	CUDA_CHECK_THROW(
		cudaMemcpy(parameter_gradient_host.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost)
	);
	REQUIRE(std::any_of(parameter_gradient_host.begin(), parameter_gradient_host.end(), [](T value) { return (float)value != 0.0f; }));
	REQUIRE(std::all_of(parameter_gradient_host.begin(), parameter_gradient_host.end(), [](T value) { return std::isfinite((float)value); }));

	trainer->optimizer_step(default_loss_scale<T>());
	std::vector<T> params_after(encoding->n_params());
	CUDA_CHECK_THROW(cudaMemcpy(params_after.data(), encoding->params(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost));
	REQUIRE(params_after != params_before);
}

TEST_CASE("Hard LoD hyperparameters round trip through the factory", "[encoding][permuto][lod]") {
	tcnn_test_setup();

	json config = hard_lod_config(2, 4);
	config["base"]["seed"] = 42;

	using T = network_precision_t;
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(6, config, 16)};
	const json hyperparams = encoding->hyperparams();

	REQUIRE(hyperparams.at("otype") == "MultiLevelEncodingLoD");
	REQUIRE(hyperparams.at("lod_type") == "Hard");
	REQUIRE(hyperparams.at("base").at("otype") == "Permuto");
	REQUIRE(hyperparams.at("base").at("seed") == 42);

	std::shared_ptr<Encoding<T>> restored{create_encoding<T>(encoding->input_width(), hyperparams, 16)};
	REQUIRE(restored->input_width() == encoding->input_width());
	REQUIRE(restored->output_width() == encoding->output_width());
	REQUIRE(restored->n_params() == encoding->n_params());
	REQUIRE(restored->hyperparams() == hyperparams);

	auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
	auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
	auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);
	auto restored_optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
	auto restored_loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
	auto restored_trainer = std::make_shared<Trainer<float, T, T>>(restored, restored_optimizer, restored_loss);
	CUDA_CHECK_THROW(cudaMemcpy(restored->params(), encoding->params(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToDevice));

	const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	GPUMatrix<float> input{6, batch_size};
	GPUMatrix<T> output{encoding->padded_output_width(), batch_size};
	GPUMatrix<T> restored_output{restored->padded_output_width(), batch_size};
	const auto input_host = hard_lod_input(batch_size, {0.0f, 0.5f, 1.0f});
	CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));
	encoding->forward(input, &output);
	restored->forward(input, &restored_output);
	REQUIRE(output.to_cpu_vector() == restored_output.to_cpu_vector());
}

TEST_CASE("Hard LoD retains context without a forward output", "[encoding][permuto][lod]") {
	tcnn_test_setup();

	using T = network_precision_t;
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(6, hard_lod_config(2, 4), 16)};
	auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
	auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
	auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);

	const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	GPUMatrix<float> input{6, batch_size};
	GPUMatrix<float> input_gradient{6, batch_size};
	GPUMatrix<T> output{encoding->padded_output_width(), batch_size};
	GPUMatrix<T> output_gradient{encoding->padded_output_width(), batch_size};
	const auto input_host = hard_lod_input(batch_size, {1.0f});
	CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));

	pcg32 rng{0xdeadbeef};
	output_gradient.initialize_uniform(rng, -default_loss_scale<T>(), default_loss_scale<T>());
	auto context = encoding->forward(input, nullptr, false, true);
	encoding->backward(*context, input, output, output_gradient, &input_gradient, false, GradientMode::Ignore);

	const auto input_gradient_host = input_gradient.to_cpu_vector();
	REQUIRE(std::any_of(input_gradient_host.begin(), input_gradient_host.end(), [](float value) { return value != 0.0f; }));
	for (uint32_t row = 0; row < batch_size; ++row) {
		REQUIRE(input_gradient_host[row * 6 + 3] == 0.0f);
		REQUIRE(input_gradient_host[row * 6 + 4] == 0.0f);
		REQUIRE(input_gradient_host[row * 6 + 5] == 0.0f);
	}

	context = encoding->forward(input, &output);
	encoding->backward(*context, input, output, output_gradient, nullptr, false, GradientMode::Overwrite);
	std::vector<T> parameter_gradient_host(encoding->n_params());
	CUDA_CHECK_THROW(
		cudaMemcpy(parameter_gradient_host.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost)
	);
	REQUIRE(std::any_of(parameter_gradient_host.begin(), parameter_gradient_host.end(), [](T value) { return (float)value != 0.0f; }));
}

TEST_CASE("Permuto preserves parameter gradient modes", "[encoding][permuto]") {
	tcnn_test_setup();

	using T = network_precision_t;
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(5, permuto_config(2, 4), 16)};
	auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
	auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
	auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);

	const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	GPUMatrix<float> input{5, batch_size};
	GPUMatrix<T> output{encoding->padded_output_width(), batch_size};
	GPUMatrix<T> output_gradient{encoding->padded_output_width(), batch_size};
	pcg32 rng{0xdeadbeef};
	input.initialize_uniform(rng, 0.001f, 0.999f);
	std::vector<T> output_gradient_host(output_gradient.n_elements(), (T)0.0f);
	std::fill_n(output_gradient_host.begin(), encoding->output_width(), (T)1.0f);
	CUDA_CHECK_THROW(cudaMemcpy(output_gradient.data(), output_gradient_host.data(), output_gradient.n_bytes(), cudaMemcpyHostToDevice));
	auto context = encoding->forward(input, &output);

	encoding->backward(*context, input, output, output_gradient, nullptr, false, GradientMode::Overwrite);
	std::vector<T> overwrite(encoding->n_params());
	CUDA_CHECK_THROW(cudaMemcpy(overwrite.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost));

	encoding->backward(*context, input, output, output_gradient, nullptr, false, GradientMode::Accumulate);
	std::vector<T> accumulated(encoding->n_params());
	CUDA_CHECK_THROW(cudaMemcpy(accumulated.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost));
	bool found_nonzero = false;
	for (size_t i = 0; i < overwrite.size(); ++i) {
		const float expected = 2.0f * (float)overwrite[i];
		const float tolerance = 0.02f * std::max(1.0f, std::abs(expected));
		REQUIRE(std::abs((float)accumulated[i] - expected) <= tolerance);
		found_nonzero |= expected != 0.0f;
	}
	REQUIRE(found_nonzero);

	encoding->backward(*context, input, output, output_gradient, nullptr, false, GradientMode::Ignore);
	std::vector<T> ignored(encoding->n_params());
	CUDA_CHECK_THROW(cudaMemcpy(ignored.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost));
	REQUIRE(ignored == accumulated);
}

TEST_CASE("Permuto encodings preserve parameter gradients for empty batches", "[encoding][permuto]") {
	tcnn_test_setup();

	using T = network_precision_t;
	const std::vector<std::pair<uint32_t, json>> configurations = {
		{5, permuto_config(2,  4)},
		{6, hard_lod_config(2, 4)},
	};
	for (const auto& [input_width, config] : configurations) {
		std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(input_width, config, 16)};
		auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
		auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
		auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);

		GPUMatrix<float> input{input_width, 0};
		GPUMatrix<T> output{encoding->padded_output_width(), 0};
		GPUMatrix<T> output_gradient{encoding->padded_output_width(), 0};
		std::vector<T> sentinel(encoding->n_params(), (T)0.5f);
		std::vector<T> gradients(encoding->n_params());

		for (GradientMode mode : {GradientMode::Overwrite, GradientMode::Accumulate, GradientMode::Ignore}) {
			CUDA_CHECK_THROW(cudaMemcpy(encoding->gradients(), sentinel.data(), encoding->n_params() * sizeof(T), cudaMemcpyHostToDevice));
			auto context = encoding->forward(input, &output);
			encoding->backward(*context, input, output, output_gradient, nullptr, false, mode);
			CUDA_CHECK_THROW(cudaMemcpy(gradients.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost));

			if (mode == GradientMode::Overwrite) {
				REQUIRE(std::all_of(gradients.begin(), gradients.end(), [](T value) { return (float)value == 0.0f; }));
			} else {
				REQUIRE(gradients == sentinel);
			}
		}
	}
}

TEST_CASE("Permuto honors pitched SoA matrix strides", "[encoding][permuto]") {
	tcnn_test_setup();

	using T = float;
	constexpr uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	constexpr uint32_t output_stride = batch_size + 7;
	constexpr uint32_t input_gradient_stride = batch_size + 5;
	constexpr float guard = 0.5f;
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(5, permuto_config(3, 4), 16)};
	auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
	auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
	auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);
	const uint32_t output_width = encoding->padded_output_width();
	constexpr uint32_t active_width = 6;

	StreamAndEvent caller_stream;
	const cudaStream_t stream = caller_stream.get();
	GPUMatrix<float> input{5, batch_size};
	pcg32 rng{0xdeadbeef};
	input.initialize_uniform(rng, 0.001f, 0.999f);

	GPUMatrix<T, SoA> contiguous_output{output_width, batch_size};
	GPUMemory<T> pitched_output_storage{output_width * output_stride};
	std::vector<T> pitched_output_host(output_width * output_stride, (T)guard);
	CUDA_CHECK_THROW(cudaMemcpyAsync(
		pitched_output_storage.data(), pitched_output_host.data(), pitched_output_host.size() * sizeof(T), cudaMemcpyHostToDevice, stream
	));
	GPUMatrixDynamic<T> pitched_output{pitched_output_storage.data(), output_width, batch_size, SoA, output_stride};

	auto contiguous_context = encoding->forward(stream, input, &contiguous_output, false, true);
	auto pitched_context = encoding->forward(stream, input, &pitched_output, false, true);
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	const auto contiguous_output_host = contiguous_output.to_cpu_vector();
	CUDA_CHECK_THROW(
		cudaMemcpy(pitched_output_host.data(), pitched_output_storage.data(), pitched_output_host.size() * sizeof(T), cudaMemcpyDeviceToHost)
	);
	for (uint32_t row = 0; row < output_width; ++row) {
		for (uint32_t element = 0; element < batch_size; ++element) {
			REQUIRE((float)pitched_output_host[row * output_stride + element] == (float)contiguous_output_host[row * batch_size + element]);
		}
		for (uint32_t element = batch_size; element < output_stride; ++element) {
			REQUIRE((float)pitched_output_host[row * output_stride + element] == guard);
		}
	}
	for (uint32_t row = active_width; row < output_width; ++row) {
		for (uint32_t element = 0; element < batch_size; ++element) {
			REQUIRE((float)pitched_output_host[row * output_stride + element] == 0.0f);
		}
	}

	GPUMatrix<T, SoA> contiguous_upstream{output_width, batch_size};
	contiguous_upstream.initialize_uniform(rng, -default_loss_scale<T>(), default_loss_scale<T>());
	const auto contiguous_upstream_host = contiguous_upstream.to_cpu_vector();
	GPUMemory<T> pitched_upstream_storage{output_width * output_stride};
	std::vector<T> pitched_upstream_host(output_width * output_stride, (T)guard);
	for (uint32_t row = 0; row < output_width; ++row) {
		std::copy_n(contiguous_upstream_host.data() + row * batch_size, batch_size, pitched_upstream_host.data() + row * output_stride);
	}
	CUDA_CHECK_THROW(cudaMemcpyAsync(
		pitched_upstream_storage.data(), pitched_upstream_host.data(), pitched_upstream_host.size() * sizeof(T), cudaMemcpyHostToDevice, stream
	));
	GPUMatrixDynamic<T> pitched_upstream{pitched_upstream_storage.data(), output_width, batch_size, SoA, output_stride};
	GPUMatrix<float, SoA> contiguous_input_gradient{5, batch_size};
	GPUMemory<float> pitched_input_gradient_storage{5 * input_gradient_stride};
	std::vector<float> pitched_input_gradient_host(5 * input_gradient_stride, guard);
	CUDA_CHECK_THROW(cudaMemcpyAsync(
		pitched_input_gradient_storage.data(),
		pitched_input_gradient_host.data(),
		pitched_input_gradient_host.size() * sizeof(float),
		cudaMemcpyHostToDevice,
		stream
	));
	GPUMatrixDynamic<float> pitched_input_gradient{pitched_input_gradient_storage.data(), 5, batch_size, SoA, input_gradient_stride};

	encoding->backward(
		stream, *pitched_context, input, pitched_output, pitched_upstream, &pitched_input_gradient, false, GradientMode::Overwrite
	);
	std::vector<T> pitched_parameter_gradient(encoding->n_params());
	CUDA_CHECK_THROW(cudaMemcpyAsync(
		pitched_parameter_gradient.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost, stream
	));
	encoding->backward(
		stream, *contiguous_context, input, contiguous_output, contiguous_upstream, &contiguous_input_gradient, false, GradientMode::Overwrite
	);
	std::vector<T> contiguous_parameter_gradient(encoding->n_params());
	CUDA_CHECK_THROW(cudaMemcpyAsync(
		contiguous_parameter_gradient.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost, stream
	));
	CUDA_CHECK_THROW(cudaMemcpyAsync(
		pitched_input_gradient_host.data(),
		pitched_input_gradient_storage.data(),
		pitched_input_gradient_host.size() * sizeof(float),
		cudaMemcpyDeviceToHost,
		stream
	));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	const auto contiguous_input_gradient_host = contiguous_input_gradient.to_cpu_vector();

	for (uint32_t dim = 0; dim < 5; ++dim) {
		for (uint32_t element = 0; element < batch_size; ++element) {
			const float expected = contiguous_input_gradient_host[dim * batch_size + element];
			const float actual = pitched_input_gradient_host[dim * input_gradient_stride + element];
			if (expected == 0.0f) {
				REQUIRE(actual == 0.0f);
			} else {
				const float difference = std::abs(actual - expected);
				REQUIRE(difference <= 1e-4f * std::max(1.0f, std::abs(expected)));
			}
		}
		for (uint32_t element = batch_size; element < input_gradient_stride; ++element) {
			REQUIRE(pitched_input_gradient_host[dim * input_gradient_stride + element] == guard);
		}
	}
	for (size_t i = 0; i < pitched_parameter_gradient.size(); ++i) {
		const float expected = (float)contiguous_parameter_gradient[i];
		const float actual = (float)pitched_parameter_gradient[i];
		REQUIRE(std::abs(actual - expected) <= 0.02f * std::max(1.0f, std::abs(expected)));
	}
}

TEST_CASE("Hard LoD rejects soft level selection", "[encoding][permuto][lod]") {
	json config = hard_lod_config();
	config["lod_type"] = "Soft";
	REQUIRE_THROWS_AS(create_encoding<network_precision_t>(6, config, 16), std::runtime_error);
}

TEST_CASE("Hard LoD isolates rows and retains its forward context", "[encoding][permuto][lod]") {
	tcnn_test_setup();

	using T = network_precision_t;
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(6, hard_lod_config(), 16)};
	auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
	auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
	auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);

	REQUIRE(encoding->input_width() == 6);
	REQUIRE(encoding->output_width() == 32);
	REQUIRE(encoding->n_params() == 16777216);

	const std::vector<float> ratios = {0.0f, 10.0f / 16.0f, 12.0f / 16.0f, 15.0f / 16.0f, 1.0f};
	const std::vector<uint32_t> n_enabled_levels = {1, 11, 13, 16, 16};
	const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	GPUMatrix<float> input{6, batch_size};
	GPUMatrix<float> input_gradient{6, batch_size};
	GPUMatrix<T> output{32, batch_size};
	GPUMatrix<T> output_gradient{32, batch_size};
	const auto input_host = hard_lod_input(batch_size, ratios);
	CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));

	pcg32 rng{0xdeadbeef};
	output_gradient.initialize_uniform(rng, -default_loss_scale<T>(), default_loss_scale<T>());
	auto context = encoding->forward(input, &output, false, true);
	const auto output_host = output.to_cpu_vector();

	for (uint32_t row = 0; row < ratios.size(); ++row) {
		bool last_active_level_is_nonzero = false;
		for (uint32_t feature = 2 * (n_enabled_levels[row] - 1); feature < 2 * n_enabled_levels[row]; ++feature) {
			last_active_level_is_nonzero |= (float)output_host[row * 32 + feature] != 0.0f;
		}
		REQUIRE(last_active_level_is_nonzero);

		for (uint32_t feature = 2 * n_enabled_levels[row]; feature < 32; ++feature) {
			REQUIRE((float)output_host[row * 32 + feature] == 0.0f);
		}
	}

	GPUMatrix<float> repeated_input{6, batch_size};
	GPUMatrix<T> repeated_output{32, batch_size};
	std::vector<float> repeated_input_host(repeated_input.n_elements());
	for (uint32_t source_row = 0; source_row < ratios.size(); ++source_row) {
		for (uint32_t row = 0; row < batch_size; ++row) {
			std::copy_n(&input_host[source_row * 6], 6, &repeated_input_host[row * 6]);
		}
		CUDA_CHECK_THROW(cudaMemcpy(repeated_input.data(), repeated_input_host.data(), repeated_input.n_bytes(), cudaMemcpyHostToDevice));
		encoding->forward(repeated_input, &repeated_output);

		const auto repeated_output_host = repeated_output.to_cpu_vector();
		for (uint32_t feature = 0; feature < 32; ++feature) {
			REQUIRE((float)output_host[source_row * 32 + feature] == (float)repeated_output_host[feature]);
		}
	}

	encoding->backward(*context, input, output, output_gradient, &input_gradient);
	const auto input_gradient_host = input_gradient.to_cpu_vector();
	bool xyz_gradient_is_nonzero = false;
	for (uint32_t row = 0; row < batch_size; ++row) {
		for (uint32_t dim = 0; dim < 6; ++dim) {
			const float value = input_gradient_host[row * 6 + dim];
			REQUIRE(std::isfinite(value));
			xyz_gradient_is_nonzero |= dim < 3 && value != 0.0f;
		}
		REQUIRE(input_gradient_host[row * 6 + 3] == 0.0f);
		REQUIRE(input_gradient_host[row * 6 + 4] == 0.0f);
		REQUIRE(input_gradient_host[row * 6 + 5] == 0.0f);
	}
	REQUIRE(xyz_gradient_is_nonzero);

	std::vector<T> parameter_gradient_host(encoding->n_params());
	CUDA_CHECK_THROW(
		cudaMemcpy(parameter_gradient_host.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost)
	);
	REQUIRE(std::any_of(parameter_gradient_host.begin(), parameter_gradient_host.end(), [](T value) { return (float)value != 0.0f; }));
	REQUIRE(std::all_of(parameter_gradient_host.begin(), parameter_gradient_host.end(), [](T value) { return std::isfinite((float)value); }));
}

TEST_CASE("Hard LoD masks parameter gradients above level zero", "[encoding][permuto][lod]") {
	tcnn_test_setup();

	using T = network_precision_t;
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(6, hard_lod_config(), 16)};
	auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
	auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
	auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);

	const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	GPUMatrix<float> input{6, batch_size};
	GPUMatrix<T> output{32, batch_size};
	GPUMatrix<T> output_gradient{32, batch_size};
	const auto input_host = hard_lod_input(batch_size, {0.0f});
	CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));

	pcg32 rng{0xdeadbeef};
	output_gradient.initialize_uniform(rng, -default_loss_scale<T>(), default_loss_scale<T>());
	std::vector<T> parameter_gradient_host(encoding->n_params(), (T)1.0f);
	CUDA_CHECK_THROW(
		cudaMemcpy(encoding->gradients(), parameter_gradient_host.data(), encoding->n_params() * sizeof(T), cudaMemcpyHostToDevice)
	);
	auto context = encoding->forward(input, &output);
	encoding->backward(*context, input, output, output_gradient);

	CUDA_CHECK_THROW(
		cudaMemcpy(parameter_gradient_host.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost)
	);
	const auto first_level_end = parameter_gradient_host.begin() + (1u << 20);
	REQUIRE(std::all_of(parameter_gradient_host.begin(), parameter_gradient_host.end(), [](T value) { return std::isfinite((float)value); }));
	REQUIRE(std::any_of(parameter_gradient_host.begin(), first_level_end, [](T value) { return (float)value != 0.0f; }));
	REQUIRE(std::all_of(first_level_end, parameter_gradient_host.end(), [](T value) { return (float)value == 0.0f; }));
}

TEST_CASE("Hard LoD excludes the epsilon boundary from forward and backward", "[encoding][permuto][lod]") {
	tcnn_test_setup();

	using T = network_precision_t;
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(6, hard_lod_config(), 16)};
	auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
	auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
	auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);

	const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	const float boundary_ratio = (10.0f - 1e-3f) / 16.0f;
	REQUIRE((boundary_ratio * 32.0f) / 2.0f + 1e-3f == 10.0f);

	GPUMatrix<float> input{6, batch_size};
	GPUMatrix<float> input_gradient{6, batch_size};
	GPUMatrix<T> output{32, batch_size};
	GPUMatrix<T> output_gradient{32, batch_size};
	const auto input_host = hard_lod_input(batch_size, {boundary_ratio});
	std::vector<T> output_gradient_host(output_gradient.n_elements(), (T)0.0f);
	for (uint32_t row = 0; row < batch_size; ++row) {
		output_gradient_host[row * 32 + 20] = (T)1.0f;
		output_gradient_host[row * 32 + 21] = (T)1.0f;
	}
	std::vector<T> parameter_gradient_host(encoding->n_params(), (T)1.0f);
	CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(cudaMemcpy(output_gradient.data(), output_gradient_host.data(), output_gradient.n_bytes(), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(
		cudaMemcpy(encoding->gradients(), parameter_gradient_host.data(), encoding->n_params() * sizeof(T), cudaMemcpyHostToDevice)
	);

	auto context = encoding->forward(input, &output, false, true);
	const auto output_host = output.to_cpu_vector();
	for (uint32_t row = 0; row < batch_size; ++row) {
		for (uint32_t feature = 20; feature < 32; ++feature) {
			REQUIRE((float)output_host[row * 32 + feature] == 0.0f);
		}
	}

	encoding->backward(*context, input, output, output_gradient, &input_gradient);
	const auto input_gradient_host = input_gradient.to_cpu_vector();
	REQUIRE(std::all_of(input_gradient_host.begin(), input_gradient_host.end(), [](float value) { return value == 0.0f; }));

	CUDA_CHECK_THROW(
		cudaMemcpy(parameter_gradient_host.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost)
	);
	REQUIRE(std::all_of(parameter_gradient_host.begin(), parameter_gradient_host.end(), [](T value) { return (float)value == 0.0f; }));
}

TEST_CASE("Hard LoD training supports CUDA graph capture", "[encoding][permuto][lod]") {
	tcnn_test_setup();

	using T = network_precision_t;
	json config = hard_lod_config();
	config["base"]["log2_hashmap_size"] = 4;
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(6, config, 16)};
	auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
	auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
	auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);

	const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	GPUMatrix<float> input{6, batch_size};
	GPUMatrix<float> input_gradient{6, batch_size};
	GPUMatrix<float> target{32, batch_size};
	const auto input_host = hard_lod_input(batch_size, {0.0f, 10.0f / 16.0f, 1.0f});
	CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));

	pcg32 rng{0xdeadbeef};
	target.initialize_uniform(rng, -1.0f, 1.0f);
	StreamAndEvent training_stream;
	auto context = trainer->training_step(training_stream.get(), input, target, nullptr, false, &input_gradient);
	REQUIRE(context != nullptr);
	CUDA_CHECK_THROW(cudaStreamSynchronize(training_stream.get()));
}

TEST_CASE("Permuto supports native double backward", "[encoding][permuto][double-backward]") {
	tcnn_test_setup();

	json config = permuto_config(2, 3);
	config["base_scale"] = 1.0f;
	config["per_level_scale"] = 2.0f;
	config["seed"] = 42;
	std::shared_ptr<Encoding<float>> encoding{create_encoding<float>(5, config, 2)};
	auto optimizer = std::shared_ptr<Optimizer<float>>{create_optimizer<float>(json::object())};
	auto loss = std::shared_ptr<Loss<float>>{create_loss<float>(json::object())};
	auto trainer = std::make_shared<Trainer<float, float, float>>(encoding, optimizer, loss);

	std::vector<float> params(encoding->n_params());
	for (size_t i = 0; i < params.size(); ++i) {
		params[i] = static_cast<float>(static_cast<int>((i * 17) % 29) - 14) / 128.0f;
	}
	CUDA_CHECK_THROW(cudaMemcpy(encoding->params(), params.data(), params.size() * sizeof(float), cudaMemcpyHostToDevice));

	constexpr uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	constexpr std::array<float, 5> position = {0.137f, 0.271f, 0.419f, 0.583f, 0.731f};
	constexpr std::array<float, 4> upstream = {0.75f, -0.5f, 0.25f, 1.0f};
	constexpr std::array<float, 5> second_seed = {1.0f, -0.5f, 0.25f, 7.0f, -9.0f};
	std::vector<float> input_host(5 * batch_size, 0.0f);
	std::vector<float> upstream_host(4 * batch_size, 0.0f);
	std::vector<float> second_seed_host(5 * batch_size, 0.0f);
	std::copy(position.begin(), position.end(), input_host.begin());
	std::copy(upstream.begin(), upstream.end(), upstream_host.begin());
	std::copy(second_seed.begin(), second_seed.end(), second_seed_host.begin());

	GPUMatrix<float> input{5, batch_size};
	GPUMatrix<float> output{4, batch_size};
	GPUMatrix<float> dL_doutput{4, batch_size};
	GPUMatrix<float> dL_ddLdinput{5, batch_size};
	GPUMatrix<float> dL_ddLdoutput{4, batch_size};
	GPUMatrix<float> dL_dinput{5, batch_size};
	CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(cudaMemcpy(dL_doutput.data(), upstream_host.data(), dL_doutput.n_bytes(), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(cudaMemcpy(dL_ddLdinput.data(), second_seed_host.data(), dL_ddLdinput.n_bytes(), cudaMemcpyHostToDevice));
	dL_dinput.memset(0x7f);

	auto context = encoding->forward(input, &output, false, true);
	encoding->backward_backward_input(
		*context, input, dL_ddLdinput, dL_doutput, &dL_ddLdoutput, &dL_dinput, false, GradientMode::Overwrite
	);

	const auto dL_ddLdoutput_host = dL_ddLdoutput.to_cpu_vector();
	const auto dL_dinput_host = dL_dinput.to_cpu_vector();
	REQUIRE(std::all_of(dL_dinput_host.begin(), dL_dinput_host.end(), [](float value) { return value == 0.0f; }));

	constexpr float input_epsilon = 1.0f / 1024.0f;
	std::array<float, 4> reference_dL_ddLdoutput{};
	for (uint32_t dim = 0; dim < 3; ++dim) {
		auto lower = position;
		auto upper = position;
		lower[dim] -= input_epsilon;
		upper[dim] += input_epsilon;
		const auto lower_output = reference_permuto_forward(lower, params, 1.0f, 42);
		const auto upper_output = reference_permuto_forward(upper, params, 1.0f, 42);
		for (uint32_t feature = 0; feature < 4; ++feature) {
			reference_dL_ddLdoutput[feature] += second_seed[dim] * (upper_output[feature] - lower_output[feature]) / (2.0f * input_epsilon);
		}
	}
	for (uint32_t feature = 0; feature < 4; ++feature) {
		REQUIRE(dL_ddLdoutput_host[feature] == Approx(reference_dL_ddLdoutput[feature]).margin(2e-3f).epsilon(1e-2f));
	}
	REQUIRE(std::all_of(dL_ddLdoutput_host.begin() + 4, dL_ddLdoutput_host.end(), [](float value) { return value == 0.0f; }));

	const auto reference_contraction = [&](const std::vector<float>& reference_params) {
		float result = 0.0f;
		for (uint32_t dim = 0; dim < 3; ++dim) {
			auto lower = position;
			auto upper = position;
			lower[dim] -= input_epsilon;
			upper[dim] += input_epsilon;
			const auto lower_output = reference_permuto_forward(lower, reference_params, 1.0f, 42);
			const auto upper_output = reference_permuto_forward(upper, reference_params, 1.0f, 42);
			for (uint32_t feature = 0; feature < 4; ++feature) {
				result += second_seed[dim] * upstream[feature] * (upper_output[feature] - lower_output[feature]) / (2.0f * input_epsilon);
			}
		}
		return result;
	};

	std::vector<float> parameter_gradient(params.size());
	CUDA_CHECK_THROW(
		cudaMemcpy(parameter_gradient.data(), encoding->gradients(), parameter_gradient.size() * sizeof(float), cudaMemcpyDeviceToHost)
	);
	constexpr float parameter_epsilon = 1.0f / 256.0f;
	bool has_parameter_gradient = false;
	for (size_t i = 0; i < params.size(); ++i) {
		auto lower = params;
		auto upper = params;
		lower[i] -= parameter_epsilon;
		upper[i] += parameter_epsilon;
		const float expected = (reference_contraction(upper) - reference_contraction(lower)) / (2.0f * parameter_epsilon);
		has_parameter_gradient |= expected != 0.0f;
		REQUIRE(parameter_gradient[i] == Approx(expected).margin(5e-3f).epsilon(2e-2f));
	}
	REQUIRE(has_parameter_gradient);

	encoding->backward_backward_input(
		*context, input, dL_ddLdinput, dL_doutput, &dL_ddLdoutput, nullptr, false, GradientMode::Accumulate
	);
	std::vector<float> accumulated(params.size());
	CUDA_CHECK_THROW(cudaMemcpy(accumulated.data(), encoding->gradients(), accumulated.size() * sizeof(float), cudaMemcpyDeviceToHost));
	for (size_t i = 0; i < params.size(); ++i) {
		REQUIRE(accumulated[i] == Approx(2.0f * parameter_gradient[i]).margin(1e-5f).epsilon(1e-5f));
	}
	encoding->backward_backward_input(*context, input, dL_ddLdinput, dL_doutput, &dL_ddLdoutput, nullptr, false, GradientMode::Ignore);
	std::vector<float> ignored(params.size());
	CUDA_CHECK_THROW(cudaMemcpy(ignored.data(), encoding->gradients(), ignored.size() * sizeof(float), cudaMemcpyDeviceToHost));
	REQUIRE(ignored == accumulated);

	std::fill(second_seed_host.begin(), second_seed_host.end(), 0.0f);
	second_seed_host[3] = 1.0f;
	second_seed_host[4] = -2.0f;
	CUDA_CHECK_THROW(cudaMemcpy(dL_ddLdinput.data(), second_seed_host.data(), dL_ddLdinput.n_bytes(), cudaMemcpyHostToDevice));
	encoding->backward_backward_input(
		*context, input, dL_ddLdinput, dL_doutput, &dL_ddLdoutput, &dL_dinput, false, GradientMode::Overwrite
	);
	CUDA_CHECK_THROW(cudaMemcpy(parameter_gradient.data(), encoding->gradients(), parameter_gradient.size() * sizeof(float), cudaMemcpyDeviceToHost));
	REQUIRE(std::all_of(parameter_gradient.begin(), parameter_gradient.end(), [](float value) { return value == 0.0f; }));
	const auto masked_dL_ddLdoutput = dL_ddLdoutput.to_cpu_vector();
	REQUIRE(std::all_of(masked_dL_ddLdoutput.begin(), masked_dL_ddLdoutput.end(), [](float value) { return value == 0.0f; }));
}

TEST_CASE("Hard LoD delegates native double backward", "[encoding][permuto][lod][double-backward]") {
	tcnn_test_setup();

	json config = hard_lod_config(2, 3);
	config["base"]["base_scale"] = 1.0f;
	config["base"]["per_level_scale"] = 2.0f;
	config["base"]["seed"] = 42;
	std::shared_ptr<Encoding<float>> encoding{create_encoding<float>(6, config, 2)};
	auto optimizer = std::shared_ptr<Optimizer<float>>{create_optimizer<float>(json::object())};
	auto loss = std::shared_ptr<Loss<float>>{create_loss<float>(json::object())};
	auto trainer = std::make_shared<Trainer<float, float, float>>(encoding, optimizer, loss);

	std::vector<float> params(encoding->n_params());
	for (size_t i = 0; i < params.size(); ++i) {
		params[i] = static_cast<float>(static_cast<int>((i * 17) % 29) - 14) / 128.0f;
	}
	CUDA_CHECK_THROW(cudaMemcpy(encoding->params(), params.data(), params.size() * sizeof(float), cudaMemcpyHostToDevice));

	constexpr uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	constexpr std::array<float, 5> position = {0.137f, 0.271f, 0.419f, 0.583f, 0.731f};
	constexpr std::array<float, 4> upstream = {0.75f, -0.5f, 0.25f, 1.0f};
	const float boundary_ratio = (1.0f - 1e-3f) / 2.0f;
	std::vector<float> input_host(6 * batch_size, 0.0f);
	std::vector<float> upstream_host(4 * batch_size, 0.0f);
	std::vector<float> second_seed_host(6 * batch_size, 0.0f);
	for (uint32_t element = 0; element < batch_size; ++element) {
		std::copy(position.begin(), position.end(), input_host.begin() + element * 6);
	}
	input_host[5] = 0.0f;
	input_host[11] = boundary_ratio;
	input_host[17] = 1.0f;
	for (uint32_t element = 0; element < 3; ++element) {
		std::copy(upstream.begin(), upstream.end(), upstream_host.begin() + element * 4);
		second_seed_host[element * 6] = 1.0f;
		second_seed_host[element * 6 + 1] = -0.5f;
		second_seed_host[element * 6 + 2] = 0.25f;
		second_seed_host[element * 6 + 5] = 100.0f;
	}

	GPUMatrix<float> input{6, batch_size};
	GPUMatrix<float> output{4, batch_size};
	GPUMatrix<float> dL_doutput{4, batch_size};
	GPUMatrix<float> dL_ddLdinput{6, batch_size};
	GPUMatrix<float> dL_ddLdoutput{4, batch_size};
	GPUMatrix<float> dL_dinput{6, batch_size};
	CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(cudaMemcpy(dL_doutput.data(), upstream_host.data(), dL_doutput.n_bytes(), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(cudaMemcpy(dL_ddLdinput.data(), second_seed_host.data(), dL_ddLdinput.n_bytes(), cudaMemcpyHostToDevice));

	auto context = encoding->forward(input, &output, false, true);
	std::vector<float> gradient_sentinel(params.size(), 0.5f);
	CUDA_CHECK_THROW(cudaMemcpy(encoding->gradients(), gradient_sentinel.data(), gradient_sentinel.size() * sizeof(float), cudaMemcpyHostToDevice));
	encoding->backward_backward_input(*context, input, dL_ddLdinput, dL_doutput, &dL_ddLdoutput, nullptr, false, GradientMode::Ignore);
	const auto dL_ddLdoutput_host = dL_ddLdoutput.to_cpu_vector();
	for (uint32_t element : {0u, 1u}) {
		REQUIRE(dL_ddLdoutput_host[element * 4 + 2] == 0.0f);
		REQUIRE(dL_ddLdoutput_host[element * 4 + 3] == 0.0f);
	}
	REQUIRE((dL_ddLdoutput_host[10] != 0.0f || dL_ddLdoutput_host[11] != 0.0f));
	std::vector<float> ignored(params.size());
	CUDA_CHECK_THROW(cudaMemcpy(ignored.data(), encoding->gradients(), ignored.size() * sizeof(float), cudaMemcpyDeviceToHost));
	REQUIRE(ignored == gradient_sentinel);

	std::fill(upstream_host.begin() + 4, upstream_host.end(), 0.0f);
	std::fill(second_seed_host.begin() + 6, second_seed_host.end(), 0.0f);
	CUDA_CHECK_THROW(cudaMemcpy(dL_doutput.data(), upstream_host.data(), dL_doutput.n_bytes(), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(cudaMemcpy(dL_ddLdinput.data(), second_seed_host.data(), dL_ddLdinput.n_bytes(), cudaMemcpyHostToDevice));
	dL_dinput.memset(0x7f);
	encoding->backward_backward_input(
		*context, input, dL_ddLdinput, dL_doutput, &dL_ddLdoutput, &dL_dinput, false, GradientMode::Overwrite
	);
	std::vector<float> parameter_gradient(params.size());
	CUDA_CHECK_THROW(cudaMemcpy(parameter_gradient.data(), encoding->gradients(), parameter_gradient.size() * sizeof(float), cudaMemcpyDeviceToHost));
	REQUIRE(std::any_of(parameter_gradient.begin(), parameter_gradient.begin() + parameter_gradient.size() / 2, [](float value) { return value != 0.0f; }));
	REQUIRE(std::all_of(parameter_gradient.begin() + parameter_gradient.size() / 2, parameter_gradient.end(), [](float value) { return value == 0.0f; }));
	const auto hessian = dL_dinput.to_cpu_vector();
	REQUIRE(std::all_of(hessian.begin(), hessian.end(), [](float value) { return value == 0.0f; }));

	std::fill(second_seed_host.begin(), second_seed_host.end(), 0.0f);
	second_seed_host[5] = 1.0f;
	CUDA_CHECK_THROW(cudaMemcpy(dL_ddLdinput.data(), second_seed_host.data(), dL_ddLdinput.n_bytes(), cudaMemcpyHostToDevice));
	encoding->backward_backward_input(
		*context, input, dL_ddLdinput, dL_doutput, &dL_ddLdoutput, &dL_dinput, false, GradientMode::Overwrite
	);
	CUDA_CHECK_THROW(cudaMemcpy(parameter_gradient.data(), encoding->gradients(), parameter_gradient.size() * sizeof(float), cudaMemcpyDeviceToHost));
	REQUIRE(std::all_of(parameter_gradient.begin(), parameter_gradient.end(), [](float value) { return value == 0.0f; }));
	const auto ratio_dL_ddLdoutput = dL_ddLdoutput.to_cpu_vector();
	REQUIRE(std::all_of(ratio_dL_ddLdoutput.begin(), ratio_dL_ddLdoutput.end(), [](float value) { return value == 0.0f; }));
	const auto ratio_dL_dinput = dL_dinput.to_cpu_vector();
	REQUIRE(std::all_of(ratio_dL_dinput.begin(), ratio_dL_dinput.end(), [](float value) { return value == 0.0f; }));
}
