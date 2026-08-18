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
 *  @brief  Test Permuto and multilevel level-of-detail encodings.
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

json permuto_config(uint32_t n_levels = 16, uint32_t log2_hashmap_size = 19, uint32_t n_features_per_level = 2) {
	return {
		{"otype",                "Permuto"           },
		{"n_levels",             n_levels            },
		{"n_features_per_level", n_features_per_level},
		{"log2_hashmap_size",    log2_hashmap_size   },
		{"per_level_scale",      1.4472692374403782  },
		{"base_scale",           16.0                },
		{"interpolation",        "Linear"            },
		{"max_input_grad_dims",  3                   },
	};
}

json hard_lod_config(uint32_t n_levels = 16, uint32_t log2_hashmap_size = 19, uint32_t n_features_per_level = 2) {
	return {
		{"otype", "MultiLevelEncodingLoD"},
		{"lod_type", "Hard"},
		{"base", permuto_config(n_levels, log2_hashmap_size, n_features_per_level)},
	};
}

json soft_lod_config(uint32_t n_levels = 16, uint32_t log2_hashmap_size = 19, uint32_t n_features_per_level = 2) {
	json result = hard_lod_config(n_levels, log2_hashmap_size, n_features_per_level);
	result["lod_type"] = "Soft";
	return result;
}

json grid_config(uint32_t n_levels = 2, uint32_t n_features_per_level = 2) {
	return {
		{"otype",                "HashGrid"},
		{"n_levels",             n_levels            },
		{"n_features_per_level", n_features_per_level},
		{"log2_hashmap_size",    4         },
		{"base_resolution",      4         },
		{"per_level_scale",      2.0f      },
		{"interpolation",        "Linear"  },
	};
}

json grid_lod_config(const char* lod_type, uint32_t n_levels = 2, uint32_t n_features_per_level = 2) {
	return {
		{"otype",    "MultiLevelEncodingLoD"},
		{"lod_type", lod_type                 },
		{"base",     grid_config(n_levels, n_features_per_level)},
	};
}

class LegacyMultiLevelEncoding : public MultiLevelEncoding<float> {
public:
	explicit LegacyMultiLevelEncoding(uint32_t n_offsets) { m_offsets.size = n_offsets; }

#if !defined(TCNN_NO_FWD_BWD)
	std::unique_ptr<Context> forward_impl(
		cudaStream_t,
		const GPUMatrixDynamic<float>&,
		GPUMatrixDynamic<float>* = nullptr,
		bool = false,
		bool = false
	) override {
		return std::make_unique<Context>();
	}

	void backward_impl(
		cudaStream_t,
		const Context&,
		const GPUMatrixDynamic<float>&,
		const GPUMatrixDynamic<float>&,
		const GPUMatrixDynamic<float>&,
		GPUMatrixDynamic<float>* = nullptr,
		bool = false,
		GradientMode = GradientMode::Overwrite
	) override { }
#endif

	uint32_t input_width() const override { return 1; }
	uint32_t padded_output_width() const override { return 0; }
	uint32_t output_width() const override { return 0; }
	uint32_t required_input_alignment() const override { return 1; }
	void set_padded_output_width(uint32_t) override { }
	uint32_t required_output_alignment() const override { return 1; }
	MatrixLayout preferred_output_layout() const override { return AoS; }
	uint32_t n_pos_dims() const override { return 1; }
	uint32_t n_features_per_level() const override { return 1; }
	size_t level_n_params(uint32_t) const override { return 0; }
	size_t level_params_offset(uint32_t) const override { return 0; }
	const ParamsOffsetTable& params_offset_table() const override { return m_offsets; }
	json hyperparams() const override { return {{"otype", "LegacyMultiLevelEncoding"}}; }

private:
	ParamsOffsetTable m_offsets;
};

float soft_lod_weight(float ratio, uint32_t n_levels, uint32_t level) {
	const float level_f = ratio * n_levels + 1e-3f;
	if (level_f < 0.0f) {
		return 0.0f;
	}
	if (level_f >= (float)n_levels) {
		return 1.0f;
	}
	const int32_t level_i = (int32_t)std::floor(level_f);
	if ((int32_t)level < level_i) {
		return 1.0f;
	}
	return (int32_t)level == level_i ? level_f - level_i : 0.0f;
}

struct GuardedMatrix {
	GuardedMatrix(uint32_t rows, uint32_t cols, MatrixLayout layout, uint32_t padding, float guard)
	: rows{rows}, cols{cols}, layout{layout}, stride{(layout == AoS ? rows : cols) + padding}, guard{guard},
	  storage{(layout == AoS ? cols : rows) * stride}, matrix{storage.data(), rows, cols, layout, stride},
	  host(storage.size(), guard) { }

	size_t index(uint32_t row, uint32_t col) const {
		return layout == AoS ? col * stride + row : row * stride + col;
	}

	void set(uint32_t row, uint32_t col, float value) {
		host[index(row, col)] = value;
	}

	float get(uint32_t row, uint32_t col) const {
		return host[index(row, col)];
	}

	void upload(cudaStream_t stream) {
		CUDA_CHECK_THROW(cudaMemcpyAsync(storage.data(), host.data(), storage.get_bytes(), cudaMemcpyHostToDevice, stream));
	}

	void download(cudaStream_t stream) {
		CUDA_CHECK_THROW(cudaMemcpyAsync(host.data(), storage.data(), storage.get_bytes(), cudaMemcpyDeviceToHost, stream));
	}

	void require_matches(const std::vector<float>& contiguous) const {
		const uint32_t contiguous_stride = layout == AoS ? rows : cols;
		for (uint32_t row = 0; row < rows; ++row) {
			for (uint32_t col = 0; col < cols; ++col) {
				const size_t contiguous_index = layout == AoS ? col * contiguous_stride + row : row * contiguous_stride + col;
				CAPTURE(row, col);
				REQUIRE(get(row, col) == Approx(contiguous[contiguous_index]).margin(1e-5f).epsilon(1e-4f));
			}
		}

		if (layout == AoS) {
			for (uint32_t col = 0; col < cols; ++col) {
				for (uint32_t row = rows; row < stride; ++row) {
					REQUIRE(host[col * stride + row] == guard);
				}
			}
		} else {
			for (uint32_t row = 0; row < rows; ++row) {
				for (uint32_t col = cols; col < stride; ++col) {
					REQUIRE(host[row * stride + col] == guard);
				}
			}
		}
	}

	uint32_t rows;
	uint32_t cols;
	MatrixLayout layout;
	uint32_t stride;
	float guard;
	GPUMemory<float> storage;
	GPUMatrixDynamic<float> matrix;
	std::vector<float> host;
};

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

std::vector<float> reference_permuto_level(
	const std::array<float, 5>& position,
	const std::vector<float>& params,
	float scale,
	uint32_t seed,
	uint32_t level,
	uint32_t parameter_offset,
	uint32_t hashmap_size,
	uint32_t n_features_per_level
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

	std::vector<float> result(n_features_per_level);
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
		const uint32_t parameter = parameter_offset + (hash % hashmap_size) * n_features_per_level;
		for (uint32_t feature = 0; feature < n_features_per_level; ++feature) {
			result[feature] = std::fma(barycentric[vertex], params[parameter + feature], result[feature]);
		}
	}
	return result;
}

std::vector<float> reference_permuto_forward(
	const std::array<float, 5>& position,
	const std::vector<float>& params,
	float base_scale,
	uint32_t seed,
	uint32_t n_features_per_level = 2
) {
	std::vector<float> result(2 * n_features_per_level);
	constexpr uint32_t hashmap_size = 8;
	const uint32_t parameters_per_level = hashmap_size * n_features_per_level;
	for (uint32_t level = 0; level < 2; ++level) {
		const auto level_result = reference_permuto_level(
			position,
			params,
			std::ldexp(base_scale, static_cast<int>(level)),
			seed,
			level,
			level * parameters_per_level,
			hashmap_size,
			n_features_per_level
		);
		for (uint32_t feature = 0; feature < n_features_per_level; ++feature) {
			result[level * n_features_per_level + feature] = level_result[feature];
		}
	}
	return result;
}

} // namespace

TEST_CASE("Permuto validates its public configuration", "[encoding][permuto]") {
	using T = network_precision_t;
	REQUIRE_NOTHROW(create_encoding<T>(5, permuto_config(2, 4), 16));
	REQUIRE_NOTHROW(create_encoding<T>(6, hard_lod_config(), 16));

	const auto require_invalid = [&](const char* key, const json& value) {
		json config = permuto_config();
		config[key] = value;
		INFO("key=" << key << ", value=" << value);
		REQUIRE_THROWS_AS(create_encoding<T>(5, config, 16), std::runtime_error);
	};

	require_invalid("n_features_per_level", 0);
	require_invalid("n_features_per_level", 3);
	require_invalid("n_features_per_level", 16);
	require_invalid("n_features_per_level", -1);
	require_invalid("n_features_per_level", 2.5);
	require_invalid("n_features_per_level", true);
	require_invalid("n_features_per_level", "2");
	require_invalid("n_features_per_level", nullptr);
	require_invalid("n_levels", 0);
	require_invalid("n_levels", 33);
	require_invalid("n_levels", -1);
	require_invalid("n_levels", 16.5);
	require_invalid("n_levels", std::numeric_limits<uint64_t>::max());
	require_invalid("n_levels", true);
	require_invalid("n_levels", "16");
	require_invalid("n_levels", nullptr);
	require_invalid("log2_hashmap_size", -1);
	require_invalid("log2_hashmap_size", 19.5);
	require_invalid("log2_hashmap_size", 32);
	require_invalid("log2_hashmap_size", 27);
	require_invalid("log2_hashmap_size", std::numeric_limits<uint64_t>::max());
	require_invalid("log2_hashmap_size", true);
	require_invalid("log2_hashmap_size", "19");
	require_invalid("log2_hashmap_size", nullptr);
	require_invalid("max_input_grad_dims", -1);
	require_invalid("max_input_grad_dims", 3.5);
	require_invalid("max_input_grad_dims", 6);
	require_invalid("max_input_grad_dims", std::numeric_limits<uint64_t>::max());
	require_invalid("max_input_grad_dims", true);
	require_invalid("max_input_grad_dims", "3");
	require_invalid("max_input_grad_dims", nullptr);
	require_invalid("seed", -1);
	require_invalid("seed", 1.5);
	require_invalid("seed", std::numeric_limits<uint64_t>::max());
	require_invalid("seed", true);
	require_invalid("seed", "1337");
	require_invalid("seed", nullptr);
	require_invalid("base_scale", "invalid");
	require_invalid("base_scale", std::numeric_limits<double>::infinity());
	require_invalid("base_scale", std::numeric_limits<double>::quiet_NaN());
	require_invalid("base_scale", 1e30);
	require_invalid("base_scale", 1e100);
	require_invalid("per_level_scale", "invalid");
	require_invalid("per_level_scale", 0.0);
	require_invalid("per_level_scale", -1.0);
	require_invalid("per_level_scale", std::numeric_limits<double>::infinity());
	require_invalid("per_level_scale", std::numeric_limits<double>::quiet_NaN());
	require_invalid("per_level_scale", 1e30);
	require_invalid("per_level_scale", 1e100);
	require_invalid("interpolation", "Smoothstep");
	require_invalid("interpolation", 1);

	REQUIRE_THROWS_WITH(
		create_encoding<T>(11, permuto_config(1, 2), 1),
		"PermutoEncoding: input dimensions must be one of 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, or 24."
	);
	REQUIRE_THROWS_WITH(
		create_encoding<T>(5, permuto_config(1, 2, 3), 1),
		"PermutoEncoding: n_features_per_level must be 1, 2, 4, or 8."
	);

	json alias_conflict = permuto_config(2, 2, 4);
	alias_conflict["n_features"] = 8;
	REQUIRE_THROWS_WITH(
		create_encoding<T>(5, alias_conflict, 1), "PermutoEncoding: total feature aliases and n_levels are mutually exclusive."
	);

	json nondivisible = permuto_config(1, 2, 4);
	nondivisible.erase("n_levels");
	nondivisible["n_features"] = 6;
	REQUIRE_THROWS_WITH(
		create_encoding<T>(5, nondivisible, 1), "PermutoEncoding: n_features must be divisible by n_features_per_level."
	);

	json parameter_overflow = permuto_config(1, 31, 2);
	REQUIRE_THROWS_WITH(
		create_encoding<T>(5, parameter_overflow, 1),
		"PermutoEncoding: parameter count=4294967296 exceeds the supported maximum=4294967295"
	);
}

TEST_CASE("Permuto constructs its supported specialization matrix", "[encoding][permuto]") {
	using T = network_precision_t;
	constexpr std::array<uint32_t, 13> supported_dimensions = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24};
	constexpr std::array<uint32_t, 4> supported_feature_widths = {1, 2, 4, 8};

	for (uint32_t n_dims : supported_dimensions) {
		for (uint32_t n_features_per_level : supported_feature_widths) {
			CAPTURE(n_dims, n_features_per_level);
			json config = permuto_config(2, 2, n_features_per_level);
			config["max_input_grad_dims"] = n_dims;
			std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(n_dims, config, 1)};
			auto multi_level = std::dynamic_pointer_cast<MultiLevelEncoding<T>>(encoding);

			REQUIRE(multi_level != nullptr);
			REQUIRE(encoding->input_width() == n_dims);
			REQUIRE(encoding->output_width() == 2 * n_features_per_level);
			REQUIRE(encoding->required_output_alignment() == n_features_per_level);
			REQUIRE(encoding->n_params() == 8 * n_features_per_level);
			REQUIRE(multi_level->n_pos_dims() == n_dims);
			REQUIRE(multi_level->n_levels() == 2);
			REQUIRE(multi_level->n_features_per_level() == n_features_per_level);
			REQUIRE(encoding->hyperparams().at("n_levels") == 2);
		}
	}

	for (uint32_t unsupported_dimensions : {0u, 11u, 13u, 25u}) {
		CAPTURE(unsupported_dimensions);
		REQUIRE_THROWS_AS(create_encoding<T>(unsupported_dimensions, permuto_config(1, 2), 1), std::runtime_error);
	}
}

TEST_CASE("Permuto supports defaults and total-feature aliases", "[encoding][permuto]") {
	using T = network_precision_t;
	std::shared_ptr<Encoding<T>> defaults{create_encoding<T>(3, {{"otype", "Permuto"}}, 1)};
	const json default_hyperparams = defaults->hyperparams();
	REQUIRE(defaults->input_width() == 3);
	REQUIRE(defaults->output_width() == 32);
	REQUIRE(defaults->n_params() == 16u * (1u << 19) * 2u);
	REQUIRE(default_hyperparams.at("n_levels") == 16);
	REQUIRE(default_hyperparams.at("n_features_per_level") == 2);
	REQUIRE(default_hyperparams.at("log2_hashmap_size") == 19);
	REQUIRE(default_hyperparams.at("base_scale") == 16.0f);
	REQUIRE(default_hyperparams.at("per_level_scale") == 2.0f);
	REQUIRE(default_hyperparams.at("max_input_grad_dims") == 3);
	REQUIRE(default_hyperparams.at("seed") == 1337);

	for (const char* alias : {"n_features", "n_grid_features"}) {
		CAPTURE(alias);
		json config = permuto_config(1, 2, 4);
		config.erase("n_levels");
		config[alias] = 12;
		std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(5, config, 1)};
		const json hyperparams = encoding->hyperparams();
		REQUIRE(encoding->output_width() == 12);
		REQUIRE(hyperparams.at("n_levels") == 3);
		REQUIRE(hyperparams.at("n_features_per_level") == 4);
		REQUIRE_FALSE(hyperparams.contains("n_features"));
		REQUIRE_FALSE(hyperparams.contains("n_grid_features"));
	}

	json both_aliases = permuto_config(1, 2, 4);
	both_aliases.erase("n_levels");
	both_aliases["n_features"] = 8;
	both_aliases["n_grid_features"] = 8;
	REQUIRE_THROWS_AS(create_encoding<T>(5, both_aliases, 1), std::runtime_error);

	const auto require_invalid_alias = [&](const json& value) {
		json config = permuto_config(1, 2, 4);
		config.erase("n_levels");
		config["n_features"] = value;
		INFO("n_features=" << value);
		REQUIRE_THROWS_AS(create_encoding<T>(5, config, 1), std::runtime_error);
	};
	require_invalid_alias(0);
	require_invalid_alias(-1);
	require_invalid_alias(4.5);
	require_invalid_alias(6);
	require_invalid_alias(std::numeric_limits<uint64_t>::max());
	require_invalid_alias(true);
	require_invalid_alias("8");
	require_invalid_alias(nullptr);

	json maximum_levels = permuto_config(1, 2, 4);
	maximum_levels.erase("n_levels");
	maximum_levels["n_features"] = 32 * 4;
	std::shared_ptr<Encoding<T>> boundary_encoding;
	REQUIRE_NOTHROW(boundary_encoding.reset(create_encoding<T>(5, maximum_levels, 1)));
	maximum_levels["n_features"] = 33 * 4;
	REQUIRE_THROWS_AS(create_encoding<T>(5, maximum_levels, 1), std::runtime_error);

	json maximum_hash = permuto_config(1, 31, 1);
	maximum_hash["max_input_grad_dims"] = 1;
	REQUIRE_NOTHROW(boundary_encoding.reset(create_encoding<T>(1, maximum_hash, 1)));
	maximum_hash["n_features_per_level"] = 2;
	REQUIRE_THROWS_AS(create_encoding<T>(1, maximum_hash, 1), std::runtime_error);

	json boundary_values = permuto_config(1, 2, 2);
	boundary_values["max_input_grad_dims"] = 0;
	boundary_values["seed"] = std::numeric_limits<uint32_t>::max();
	boundary_values["base_scale"] = -16.0f;
	boundary_values["interpolation"] = "linear";
	REQUIRE_NOTHROW(boundary_encoding.reset(create_encoding<T>(5, boundary_values, 1)));
	boundary_values["max_input_grad_dims"] = 5;
	REQUIRE_NOTHROW(boundary_encoding.reset(create_encoding<T>(5, boundary_values, 1)));
}

TEST_CASE("Permuto canonical hyperparameters preserve generalized configurations", "[encoding][permuto]") {
	tcnn_test_setup();

	using T = network_precision_t;
	json config = permuto_config(3, 2, 4);
	config["max_input_grad_dims"] = 7;
	config["seed"] = 42;
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(7, config, 1)};
	const json hyperparams = encoding->hyperparams();

	REQUIRE(hyperparams.at("n_levels") == 3);
	REQUIRE(hyperparams.at("n_features_per_level") == 4);
	REQUIRE(hyperparams.at("scales_table").size() == 3 * 7);
	REQUIRE(hyperparams.at("shifts_table").size() == 3 * 7);
	std::shared_ptr<Encoding<T>> restored{create_encoding<T>(7, hyperparams, 1)};
	REQUIRE(restored->input_width() == encoding->input_width());
	REQUIRE(restored->output_width() == encoding->output_width());
	REQUIRE(restored->n_params() == encoding->n_params());
	REQUIRE(restored->hyperparams() == hyperparams);

	json compatibility_config = config;
	compatibility_config["base_resolution"] = 8;
	compatibility_config["max_resolution"] = 2048;
	compatibility_config["n_dims_to_encode"] = 24;
	std::shared_ptr<Encoding<T>> compatible{create_encoding<T>(7, compatibility_config, 1)};
	REQUIRE(compatible->hyperparams() == hyperparams);

	GPUMemory<T> params_gpu{encoding->n_params()};
	pcg32 rng{0xdeadbeef};
	generate_random_uniform(rng, params_gpu.size(), params_gpu.data(), (T)-0.125f, (T)0.125f);
	encoding->set_params(params_gpu.data(), params_gpu.data(), nullptr);
	compatible->set_params(params_gpu.data(), params_gpu.data(), nullptr);

	const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	GPUMatrix<float> input{7, batch_size};
	GPUMatrix<T> output{encoding->padded_output_width(), batch_size};
	GPUMatrix<T> compatible_output{compatible->padded_output_width(), batch_size};
	input.initialize_uniform(rng, 0.05f, 0.95f);
	encoding->inference_mixed_precision(input, output);
	compatible->inference_mixed_precision(input, compatible_output);
	const auto output_host = output.to_cpu_vector();
	REQUIRE(std::any_of(output_host.begin(), output_host.end(), [](T value) { return (float)value != 0.0f; }));
	REQUIRE(output_host == compatible_output.to_cpu_vector());

	for (const char* key : {"scales_table", "shifts_table"}) {
		CAPTURE(key);
		const json expected = hyperparams.at(key);
		const auto require_invalid_table = [&](const json& value) {
			json invalid = hyperparams;
			invalid[key] = value;
			REQUIRE_THROWS_AS(create_encoding<T>(7, invalid, 1), std::runtime_error);
		};

		require_invalid_table(json::array({expected.at(0)}));
		require_invalid_table(1.0f);

		json nonnumeric = expected;
		nonnumeric[0] = "invalid";
		require_invalid_table(nonnumeric);

		json not_finite = expected;
		not_finite[0] = std::numeric_limits<double>::quiet_NaN();
		require_invalid_table(not_finite);
		not_finite[0] = std::numeric_limits<double>::infinity();
		require_invalid_table(not_finite);

		json mismatched = expected;
		mismatched[0] = expected.at(0).get<float>() + 1.0f;
		require_invalid_table(mismatched);
	}
}

TEST_CASE("Hard LoD constructs a generalized Permuto base", "[encoding][permuto][lod]") {
	using T = network_precision_t;
	json config = hard_lod_config(3, 2, 4);
	config["base"]["max_input_grad_dims"] = 3;
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(4, config, 1)};
	const json hyperparams = encoding->hyperparams();

	REQUIRE(encoding->input_width() == 4);
	REQUIRE(encoding->output_width() == 12);
	REQUIRE(encoding->required_output_alignment() == 4);
	REQUIRE(encoding->n_params() == 3 * 4 * 4);
	REQUIRE(hyperparams.at("otype") == "MultiLevelEncodingLoD");
	REQUIRE(hyperparams.at("lod_type") == "Hard");
	REQUIRE(hyperparams.at("base").at("n_levels") == 3);
	REQUIRE(hyperparams.at("base").at("n_features_per_level") == 4);
}

TEST_CASE("Permuto matches an independent CPU reference and the frozen five-dimensional baseline", "[encoding][permuto]") {
	tcnn_test_setup();
	constexpr uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	const std::array<float, 5> position = {0.137f, 0.271f, 0.419f, 0.583f, 0.731f};

	for (uint32_t n_features_per_level : {1u, 2u, 4u, 8u}) {
		CAPTURE(n_features_per_level);
		json config = permuto_config(2, 3, n_features_per_level);
		config["base_scale"] = 1.0f;
		config["per_level_scale"] = 2.0f;
		config["max_input_grad_dims"] = 3;
		config["seed"] = 42;
		std::shared_ptr<Encoding<float>> encoding{create_encoding<float>(5, config, n_features_per_level)};
		auto optimizer = std::shared_ptr<Optimizer<float>>{create_optimizer<float>(json::object())};
		auto loss = std::shared_ptr<Loss<float>>{create_loss<float>(json::object())};
		auto trainer = std::make_shared<Trainer<float, float, float>>(encoding, optimizer, loss);
		if (n_features_per_level == 2) {
			const json expected_hyperparams = {
				{"otype", "Permuto"},
				{"n_levels", 2},
				{"n_features_per_level", 2},
				{"base_scale", 1.0f},
				{"per_level_scale", 2.0f},
				{"log2_hashmap_size", 3},
				{"max_input_grad_dims", 3},
				{"seed", 42},
				{"scales_table", std::vector<float>{
					0.707106769f, 0.408248305f, 0.288675129f, 0.223606795f, 0.182574183f,
					1.414213538f, 0.816496611f, 0.577350259f, 0.447213590f, 0.365148365f,
				}},
				{"shifts_table", std::vector<float>{
					-1.955292225f, 3.965381384f, -1.526242495f, 4.526897430f, 4.039040565f,
					-0.820634365f, 0.276217461f, -0.485876799f, -4.565489292f, -2.433936596f,
				}},
			};
			REQUIRE(encoding->n_params() == 32);
			auto multi_level = std::dynamic_pointer_cast<MultiLevelEncoding<float>>(encoding);
			REQUIRE(multi_level != nullptr);
			REQUIRE(multi_level->level_params_offset(0) == 0);
			REQUIRE(multi_level->level_params_offset(1) == 8);
			REQUIRE(multi_level->level_params_offset(2) == 16);
			REQUIRE(multi_level->level_n_params(0) == 8);
			REQUIRE(multi_level->level_n_params(1) == 8);
			REQUIRE(encoding->hyperparams() == expected_hyperparams);
		}

		std::vector<float> params(encoding->n_params());
		for (size_t i = 0; i < params.size(); ++i) {
			params[i] = static_cast<float>(static_cast<int>((i * 17) % 29) - 14) / 128.0f;
		}
		CUDA_CHECK_THROW(cudaMemcpy(encoding->params(), params.data(), params.size() * sizeof(float), cudaMemcpyHostToDevice));

		const uint32_t output_width = 2 * n_features_per_level;
		std::vector<float> upstream(output_width);
		if (n_features_per_level == 2) {
			upstream = {0.75f, -0.5f, 0.25f, 1.0f};
		} else {
			for (uint32_t feature = 0; feature < output_width; ++feature) {
				upstream[feature] = static_cast<float>(static_cast<int>(feature % 5) - 2) / 2.0f;
			}
		}
		std::vector<float> input_host(5 * batch_size);
		std::vector<float> upstream_host(output_width * batch_size, 0.0f);
		for (uint32_t element = 0; element < batch_size; ++element) {
			std::copy(position.begin(), position.end(), input_host.begin() + element * 5);
		}
		std::copy(upstream.begin(), upstream.end(), upstream_host.begin());

		GPUMatrix<float> input{5, batch_size};
		GPUMatrix<float> output{output_width, batch_size};
		GPUMatrix<float> output_gradient{output_width, batch_size};
		GPUMatrix<float> input_gradient{5, batch_size};
		CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));
		CUDA_CHECK_THROW(cudaMemcpy(output_gradient.data(), upstream_host.data(), output_gradient.n_bytes(), cudaMemcpyHostToDevice));
		auto context = encoding->forward(input, &output, false, true);
		encoding->backward(*context, input, output, output_gradient, &input_gradient, false, GradientMode::Overwrite);

		const auto reference_output = reference_permuto_forward(position, params, 1.0f, 42, n_features_per_level);
		REQUIRE(std::any_of(reference_output.begin(), reference_output.end(), [](float value) { return value != 0.0f; }));
		const auto output_host = output.to_cpu_vector();
		if (n_features_per_level == 2) {
			const std::array<float, 4> frozen_output = {0.005886105f, 0.027815659f, -0.015179069f, -0.013158527f};
			for (uint32_t feature = 0; feature < output_width; ++feature) {
				REQUIRE(output_host[feature] == Approx(frozen_output[feature]).margin(2e-5f).epsilon(2e-4f));
			}
		}
		for (uint32_t element = 0; element < batch_size; ++element) {
			for (uint32_t feature = 0; feature < output_width; ++feature) {
				REQUIRE(
					output_host[element * output_width + feature] == Approx(reference_output[feature]).margin(2e-5f).epsilon(2e-4f)
				);
			}
		}

		const auto reference_loss = [&](const std::array<float, 5>& reference_position, const std::vector<float>& reference_params) {
			const auto reference = reference_permuto_forward(reference_position, reference_params, 1.0f, 42, n_features_per_level);
			float result = 0.0f;
			for (uint32_t feature = 0; feature < output_width; ++feature) {
				result += reference[feature] * upstream[feature];
			}
			return result;
		};
		std::vector<float> parameter_gradient(params.size());
		CUDA_CHECK_THROW(cudaMemcpy(
			parameter_gradient.data(), encoding->gradients(), parameter_gradient.size() * sizeof(float), cudaMemcpyDeviceToHost
		));
		if (n_features_per_level == 2) {
			const std::array<float, 32> frozen_parameter_gradient = {
				0.120784193f, -0.080522791f, 0.020353220f, -0.013568814f, 0.083992347f, -0.055994898f, 0.0f, 0.0f,
				0.0f, 0.0f, 0.367060781f, -0.244707197f, 0.0f, 0.0f, 0.157809451f, -0.105206303f,
				0.0f, 0.0f, 0.025160966f, 0.100643866f, 0.0f, 0.0f, 0.063755088f, 0.255020350f,
				0.090111226f, 0.360444903f, 0.015566609f, 0.062266435f, 0.055406105f, 0.221624419f, 0.0f, 0.0f,
			};
			for (size_t i = 0; i < parameter_gradient.size(); ++i) {
				REQUIRE(parameter_gradient[i] == Approx(frozen_parameter_gradient[i]).margin(2e-5f).epsilon(2e-4f));
			}
		}
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
		if (n_features_per_level == 2) {
			const std::array<float, 5> frozen_input_gradient = {0.092071205f, 0.005581521f, -0.020861289f, 0.0f, 0.0f};
			for (uint32_t dim = 0; dim < 5; ++dim) {
				REQUIRE(input_gradient_host[dim] == Approx(frozen_input_gradient[dim]).margin(2e-5f).epsilon(2e-4f));
			}
		}
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
}

TEST_CASE("Default five-dimensional Permuto supports forward, backward, and optimization", "[encoding][permuto]") {
	tcnn_test_setup();

	const json config = permuto_config();

	using T = network_precision_t;
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

TEST_CASE("Generalized Permuto specializations support native forward and backward", "[encoding][permuto]") {
	tcnn_test_setup();

	constexpr std::array<std::pair<uint32_t, uint32_t>, 4> test_configs = {{{1, 1}, {3, 4}, {5, 2}, {24, 8}}};
	using T = network_precision_t;

	for (const auto& [n_dims, n_features_per_level] : test_configs) {
		CAPTURE(n_dims, n_features_per_level);
		const bool maximum_width = n_features_per_level == 8;
		json config = permuto_config(maximum_width ? 32 : 2, 2, n_features_per_level);
		if (maximum_width) {
			config["base_scale"] = 1.0f;
			config["per_level_scale"] = 1.0f;
		}
		const uint32_t max_input_grad_dims = n_dims == 24 ? 7 : n_dims;
		config["max_input_grad_dims"] = max_input_grad_dims;
		std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(n_dims, config, n_features_per_level)};
		auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
		auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
		auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);

		const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
		GPUMatrix<float> input{n_dims, batch_size};
		GPUMatrix<float> input_gradient{n_dims, batch_size};
		GPUMatrix<T> output{encoding->output_width(), batch_size};
		GPUMatrix<T> output_gradient{encoding->output_width(), batch_size};
		pcg32 rng{0xdeadbeef};
		input.initialize_uniform(rng, 0.05f, 0.95f);
		output_gradient.initialize_uniform(rng, -default_loss_scale<T>(), default_loss_scale<T>());

		auto context = encoding->forward(input, &output, false, true);
		encoding->backward(*context, input, output, output_gradient, &input_gradient, false, GradientMode::Overwrite);

		const auto output_host = output.to_cpu_vector();
		REQUIRE(std::any_of(output_host.begin(), output_host.end(), [](T value) { return (float)value != 0.0f; }));
		REQUIRE(std::all_of(output_host.begin(), output_host.end(), [](T value) { return std::isfinite((float)value); }));

		const auto input_gradient_host = input_gradient.to_cpu_vector();
		REQUIRE(std::any_of(
			input_gradient_host.begin(), input_gradient_host.end(), [](float value) { return value != 0.0f; }
		));
		REQUIRE(std::all_of(
			input_gradient_host.begin(), input_gradient_host.end(), [](float value) { return std::isfinite(value); }
		));
		for (uint32_t element = 0; element < batch_size; ++element) {
			for (uint32_t dim = max_input_grad_dims; dim < n_dims; ++dim) {
				REQUIRE(input_gradient_host[element * n_dims + dim] == 0.0f);
			}
		}

		std::vector<T> parameter_gradient(encoding->n_params());
		CUDA_CHECK_THROW(cudaMemcpy(
			parameter_gradient.data(), encoding->gradients(), parameter_gradient.size() * sizeof(T), cudaMemcpyDeviceToHost
		));
		REQUIRE(std::any_of(parameter_gradient.begin(), parameter_gradient.end(), [](T value) { return (float)value != 0.0f; }));
		REQUIRE(std::all_of(parameter_gradient.begin(), parameter_gradient.end(), [](T value) { return std::isfinite((float)value); }));
	}
}

TEST_CASE("Generalized Permuto preserves requested output alignment", "[encoding][permuto]") {
	tcnn_test_setup();

	json config = permuto_config(2, 2, 4);
	config["max_input_grad_dims"] = 3;
	std::shared_ptr<Encoding<float>> encoding{create_encoding<float>(3, config, 16)};
	REQUIRE(encoding->output_width() == 16);
	REQUIRE(encoding->required_output_alignment() == 4);

	auto optimizer = std::shared_ptr<Optimizer<float>>{create_optimizer<float>(json::object())};
	auto loss = std::shared_ptr<Loss<float>>{create_loss<float>(json::object())};
	auto trainer = std::make_shared<Trainer<float, float, float>>(encoding, optimizer, loss);
	const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	GPUMatrix<float> input{3, batch_size};
	GPUMatrix<float> output{encoding->output_width(), batch_size};
	pcg32 rng{0xdeadbeef};
	input.initialize_uniform(rng, 0.05f, 0.95f);
	encoding->forward(input, &output);

	const auto output_host = output.to_cpu_vector();
	for (uint32_t element = 0; element < batch_size; ++element) {
		for (uint32_t feature = 8; feature < encoding->output_width(); ++feature) {
			REQUIRE(output_host[element * encoding->output_width() + feature] == 0.0f);
		}
	}
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
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(5, permuto_config(2, 4, 1), 16)};
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

TEST_CASE("Multilevel encodings preserve parameter gradients for empty batches", "[encoding][permuto][grid][lod]") {
	tcnn_test_setup();

	using T = network_precision_t;
	const std::vector<std::pair<uint32_t, json>> configurations = {
		{5, permuto_config(2,  4)},
		{6, hard_lod_config(2, 4)},
		{6, soft_lod_config(2, 4)},
		{3, grid_lod_config("Hard")},
		{3, grid_lod_config("Soft")},
	};
	for (const auto& [input_width, config] : configurations) {
		std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(input_width, config, 16)};
		auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
		auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
		auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);

		GPUMatrix<float> input{input_width, 0};
		GPUMatrix<float> dL_ddLdinput{input_width, 0};
		GPUMatrix<T> output{encoding->padded_output_width(), 0};
		GPUMatrix<T> output_gradient{encoding->padded_output_width(), 0};
		GPUMatrix<T> dL_ddLdoutput{encoding->padded_output_width(), 0};
		std::vector<T> sentinel(encoding->n_params(), (T)0.5f);
		std::vector<T> gradients(encoding->n_params());

		for (GradientMode mode : {GradientMode::Overwrite, GradientMode::Accumulate, GradientMode::Ignore}) {
			CUDA_CHECK_THROW(cudaMemcpy(encoding->gradients(), sentinel.data(), encoding->n_params() * sizeof(T), cudaMemcpyHostToDevice));
			auto context = encoding->forward(input, &output, false, true);
			encoding->backward(*context, input, output, output_gradient, nullptr, false, mode);
			CUDA_CHECK_THROW(cudaMemcpy(gradients.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost));

			if (mode == GradientMode::Overwrite) {
				REQUIRE(std::all_of(gradients.begin(), gradients.end(), [](T value) { return (float)value == 0.0f; }));
			} else {
				REQUIRE(gradients == sentinel);
			}

			CUDA_CHECK_THROW(cudaMemcpy(encoding->gradients(), sentinel.data(), encoding->n_params() * sizeof(T), cudaMemcpyHostToDevice));
			encoding->backward_backward_input(
				*context, input, dL_ddLdinput, output_gradient, &dL_ddLdoutput, nullptr, false, mode
			);
			CUDA_CHECK_THROW(
				cudaMemcpy(gradients.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost)
			);
			if (mode == GradientMode::Overwrite) {
				REQUIRE(std::all_of(gradients.begin(), gradients.end(), [](T value) { return (float)value == 0.0f; }));
			} else {
				REQUIRE(gradients == sentinel);
			}
		}
	}
}

TEST_CASE("Grid-backed LoD forwards parameter gradient modes at both derivative orders", "[encoding][grid][lod]") {
	tcnn_test_setup();

	using T = float;
	constexpr uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	for (const char* lod_type : {"Hard", "Soft"}) {
		CAPTURE(lod_type);
		std::shared_ptr<Encoding<T>> base{create_encoding<T>(2, grid_config(), 4)};
		std::shared_ptr<Encoding<T>> wrapped{create_encoding<T>(3, grid_lod_config(lod_type), 4)};
		auto base_optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
		auto wrapped_optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
		auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
		auto base_trainer = std::make_shared<Trainer<float, T, T>>(base, base_optimizer, loss);
		auto wrapped_trainer = std::make_shared<Trainer<float, T, T>>(wrapped, wrapped_optimizer, loss);

		REQUIRE(base->n_params() == wrapped->n_params());
		std::vector<T> params(base->n_params());
		for (size_t i = 0; i < params.size(); ++i) {
			params[i] = (T)(static_cast<int>((i * 19) % 37) - 18) / 64.0f;
		}
		CUDA_CHECK_THROW(cudaMemcpy(base->params(), params.data(), base->n_params() * sizeof(T), cudaMemcpyHostToDevice));
		CUDA_CHECK_THROW(
			cudaMemcpy(wrapped->params(), params.data(), wrapped->n_params() * sizeof(T), cudaMemcpyHostToDevice)
		);

		std::vector<float> base_input_host(2 * batch_size);
		std::vector<float> wrapped_input_host(3 * batch_size);
		for (uint32_t element = 0; element < batch_size; ++element) {
			base_input_host[element * 2] = wrapped_input_host[element * 3] = 0.05f + (float)(element % 13) / 20.0f;
			base_input_host[element * 2 + 1] = wrapped_input_host[element * 3 + 1] =
				0.1f + (float)((element + 5) % 11) / 16.0f;
			wrapped_input_host[element * 3 + 2] = 1.0f;
		}

		GPUMatrix<float> base_input{2, batch_size};
		GPUMatrix<float> wrapped_input{3, batch_size};
		GPUMatrix<T> base_output{base->padded_output_width(), batch_size};
		GPUMatrix<T> wrapped_output{wrapped->padded_output_width(), batch_size};
		GPUMatrix<T> upstream{base->padded_output_width(), batch_size};
		GPUMatrix<float> base_second_seed{2, batch_size};
		GPUMatrix<float> wrapped_second_seed{3, batch_size};
		GPUMatrix<float> base_input_gradient{2, batch_size};
		GPUMatrix<float> wrapped_input_gradient{3, batch_size};
		GPUMatrix<T> base_upstream_gradient{base->padded_output_width(), batch_size};
		GPUMatrix<T> wrapped_upstream_gradient{wrapped->padded_output_width(), batch_size};
		CUDA_CHECK_THROW(cudaMemcpy(base_input.data(), base_input_host.data(), base_input.n_bytes(), cudaMemcpyHostToDevice));
		CUDA_CHECK_THROW(
			cudaMemcpy(wrapped_input.data(), wrapped_input_host.data(), wrapped_input.n_bytes(), cudaMemcpyHostToDevice)
		);
		pcg32 rng{0xdeadbeef};
		upstream.initialize_uniform(rng, -0.5f, 0.5f);
		base_second_seed.initialize_uniform(rng, -0.5f, 0.5f);
		std::vector<float> wrapped_second_seed_host(3 * batch_size, 100.0f);
		const auto base_second_seed_host = base_second_seed.to_cpu_vector();
		for (uint32_t element = 0; element < batch_size; ++element) {
			std::copy_n(base_second_seed_host.data() + element * 2, 2, wrapped_second_seed_host.data() + element * 3);
		}
		CUDA_CHECK_THROW(cudaMemcpy(
			wrapped_second_seed.data(), wrapped_second_seed_host.data(), wrapped_second_seed.n_bytes(), cudaMemcpyHostToDevice
		));

		auto base_context = base->forward(base_input, &base_output, false, true);
		auto wrapped_context = wrapped->forward(wrapped_input, &wrapped_output, false, true);

		std::vector<T> sentinel(base->n_params(), 0.25f);
		std::vector<T> first_overwrite;
		std::vector<T> second_overwrite;
		for (bool second_order : {false, true}) {
			CAPTURE(second_order);
			std::vector<T>& overwrite = second_order ? second_overwrite : first_overwrite;
			for (GradientMode mode : {GradientMode::Overwrite, GradientMode::Accumulate, GradientMode::Ignore}) {
				CAPTURE((int)mode);
				CUDA_CHECK_THROW(cudaMemcpy(base->gradients(), sentinel.data(), base->n_params() * sizeof(T), cudaMemcpyHostToDevice));
				CUDA_CHECK_THROW(
					cudaMemcpy(wrapped->gradients(), sentinel.data(), wrapped->n_params() * sizeof(T), cudaMemcpyHostToDevice)
				);
				if (second_order) {
					base->backward_backward_input(
						*base_context, base_input, base_second_seed, upstream, &base_upstream_gradient, nullptr, false, mode
					);
					wrapped->backward_backward_input(
						*wrapped_context,
						wrapped_input,
						wrapped_second_seed,
						upstream,
						&wrapped_upstream_gradient,
						nullptr,
						false,
						mode
					);
				} else {
					base->backward(*base_context, base_input, base_output, upstream, &base_input_gradient, false, mode);
					wrapped->backward(
						*wrapped_context, wrapped_input, wrapped_output, upstream, &wrapped_input_gradient, false, mode
					);
				}

				std::vector<T> base_gradient(base->n_params());
				std::vector<T> wrapped_gradient(wrapped->n_params());
				CUDA_CHECK_THROW(cudaMemcpy(
					base_gradient.data(), base->gradients(), base->n_params() * sizeof(T), cudaMemcpyDeviceToHost
				));
				CUDA_CHECK_THROW(cudaMemcpy(
					wrapped_gradient.data(), wrapped->gradients(), wrapped->n_params() * sizeof(T), cudaMemcpyDeviceToHost
				));
				for (size_t i = 0; i < base_gradient.size(); ++i) {
					CAPTURE(i);
					REQUIRE(wrapped_gradient[i] == Approx(base_gradient[i]).margin(1e-5f).epsilon(1e-4f));
				}
				if (mode == GradientMode::Overwrite) {
					overwrite = base_gradient;
					REQUIRE(std::any_of(overwrite.begin(), overwrite.end(), [](T value) { return value != 0.0f; }));
				} else {
					for (size_t i = 0; i < base_gradient.size(); ++i) {
						const T expected = mode == GradientMode::Accumulate ? sentinel[i] + overwrite[i] : sentinel[i];
						REQUIRE(base_gradient[i] == Approx(expected).margin(1e-5f).epsilon(1e-4f));
					}
				}
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

TEST_CASE("Multilevel LoD honors pitched matrix strides at both derivative orders", "[encoding][permuto][grid][lod]") {
	tcnn_test_setup();

	using Configuration = std::tuple<const char*, uint32_t, json, MatrixLayout>;
	const std::vector<Configuration> configurations = {
		{"soft-permuto-soa", 6, soft_lod_config(2, 3), SoA},
		{"hard-grid-soa",    3, grid_lod_config("Hard"), SoA},
		{"hard-wide-grid-aos", 3, grid_lod_config("Hard", 17, 8), AoS},
		{"soft-wide-grid-aos", 3, grid_lod_config("Soft", 17, 8), AoS},
	};
	constexpr uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	constexpr float guard = 0.5f;

	for (const auto& [case_name, input_width, config, layout] : configurations) {
		CAPTURE(case_name);
		std::shared_ptr<Encoding<float>> encoding{create_encoding<float>(input_width, config, 8)};
		auto optimizer = std::shared_ptr<Optimizer<float>>{create_optimizer<float>(json::object())};
		auto loss = std::shared_ptr<Loss<float>>{create_loss<float>(json::object())};
		auto trainer = std::make_shared<Trainer<float, float, float>>(encoding, optimizer, loss);
		const uint32_t output_width = encoding->padded_output_width();
		const MatrixLayout matrix_layout = layout;
		const auto contiguous_index = [matrix_layout](uint32_t row, uint32_t col, uint32_t rows, uint32_t cols) {
			return matrix_layout == AoS ? col * rows + row : row * cols + col;
		};

		StreamAndEvent caller_stream;
		const cudaStream_t stream = caller_stream.get();
		std::vector<float> params(encoding->n_params());
		for (size_t i = 0; i < params.size(); ++i) {
			params[i] = (float)(static_cast<int>((i * 19) % 37) - 18) / 64.0f;
		}
		CUDA_CHECK_THROW(cudaMemcpyAsync(
			encoding->params(), params.data(), params.size() * sizeof(float), cudaMemcpyHostToDevice, stream
		));

		GPUMatrixDynamic<float> input{input_width, batch_size, layout};
		std::vector<float> input_host(input.n_elements());
		GuardedMatrix pitched_input{input_width, batch_size, layout, 5, guard};
		for (uint32_t element = 0; element < batch_size; ++element) {
			for (uint32_t dim = 0; dim + 1 < input_width; ++dim) {
				const float value = 0.05f + (float)((element + dim) % 13) / 20.0f;
				input_host[contiguous_index(dim, element, input_width, batch_size)] = value;
				pitched_input.set(dim, element, value);
			}
			input_host[contiguous_index(input_width - 1, element, input_width, batch_size)] = 0.75f;
			pitched_input.set(input_width - 1, element, 0.75f);
		}
		CUDA_CHECK_THROW(cudaMemcpyAsync(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice, stream));
		pitched_input.upload(stream);

		GPUMatrixDynamic<float> contiguous_output{output_width, batch_size, layout};
		GuardedMatrix pitched_output{output_width, batch_size, layout, 7, guard};
		pitched_output.upload(stream);
		auto contiguous_context = encoding->forward(stream, input, &contiguous_output, false, true);
		auto pitched_context = encoding->forward(stream, pitched_input.matrix, &pitched_output.matrix, false, true);
		pitched_output.download(stream);
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		const auto contiguous_output_host = contiguous_output.to_cpu_vector();
		pitched_output.require_matches(contiguous_output_host);
		REQUIRE(std::any_of(
			contiguous_output_host.begin(), contiguous_output_host.end(), [](float value) { return value != 0.0f; }
		));

		GPUMatrixDynamic<float> contiguous_upstream{output_width, batch_size, layout};
		std::vector<float> contiguous_upstream_host(contiguous_upstream.n_elements());
		GuardedMatrix pitched_upstream{output_width, batch_size, layout, 7, guard};
		for (uint32_t row = 0; row < output_width; ++row) {
			for (uint32_t element = 0; element < batch_size; ++element) {
				const size_t index = contiguous_index(row, element, output_width, batch_size);
				const float value = (float)(static_cast<int>((index * 13) % 23) - 11) / 32.0f;
				contiguous_upstream_host[index] = value;
				pitched_upstream.set(row, element, value);
			}
		}
		CUDA_CHECK_THROW(cudaMemcpyAsync(
			contiguous_upstream.data(),
			contiguous_upstream_host.data(),
			contiguous_upstream.n_bytes(),
			cudaMemcpyHostToDevice,
			stream
		));
		pitched_upstream.upload(stream);

		GPUMatrixDynamic<float> contiguous_input_gradient{input_width, batch_size, layout};
		GuardedMatrix pitched_input_gradient{input_width, batch_size, layout, 5, guard};
		pitched_input_gradient.upload(stream);
		encoding->backward(
			stream,
			*pitched_context,
			pitched_input.matrix,
			pitched_output.matrix,
			pitched_upstream.matrix,
			&pitched_input_gradient.matrix,
			false,
			GradientMode::Overwrite
		);
		std::vector<float> pitched_parameter_gradient(encoding->n_params());
		CUDA_CHECK_THROW(cudaMemcpyAsync(
			pitched_parameter_gradient.data(),
			encoding->gradients(),
			encoding->n_params() * sizeof(float),
			cudaMemcpyDeviceToHost,
			stream
		));
		encoding->backward(
			stream,
			*contiguous_context,
			input,
			contiguous_output,
			contiguous_upstream,
			&contiguous_input_gradient,
			false,
			GradientMode::Overwrite
		);
		std::vector<float> contiguous_parameter_gradient(encoding->n_params());
		CUDA_CHECK_THROW(cudaMemcpyAsync(
			contiguous_parameter_gradient.data(),
			encoding->gradients(),
			encoding->n_params() * sizeof(float),
			cudaMemcpyDeviceToHost,
			stream
		));
		pitched_input_gradient.download(stream);
		pitched_upstream.download(stream);
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		pitched_input_gradient.require_matches(contiguous_input_gradient.to_cpu_vector());
		pitched_upstream.require_matches(contiguous_upstream_host);
		bool found_parameter_gradient = false;
		for (size_t i = 0; i < pitched_parameter_gradient.size(); ++i) {
			REQUIRE(
				pitched_parameter_gradient[i] == Approx(contiguous_parameter_gradient[i]).margin(1e-5f).epsilon(1e-4f)
			);
			found_parameter_gradient |= pitched_parameter_gradient[i] != 0.0f;
		}
		REQUIRE(found_parameter_gradient);
		for (uint32_t element = 0; element < batch_size; ++element) {
			REQUIRE(pitched_input_gradient.get(input_width - 1, element) == 0.0f);
		}

		GPUMatrixDynamic<float> second_seed{input_width, batch_size, layout};
		std::vector<float> second_seed_host(second_seed.n_elements());
		GuardedMatrix pitched_second_seed{input_width, batch_size, layout, 6, guard};
		for (uint32_t element = 0; element < batch_size; ++element) {
			for (uint32_t dim = 0; dim + 1 < input_width; ++dim) {
				const float value = dim == 0 ? 1.0f : (dim == 1 ? -0.25f : 0.125f);
				second_seed_host[contiguous_index(dim, element, input_width, batch_size)] = value;
				pitched_second_seed.set(dim, element, value);
			}
			second_seed_host[contiguous_index(input_width - 1, element, input_width, batch_size)] = 100.0f;
			pitched_second_seed.set(input_width - 1, element, 100.0f);
		}
		CUDA_CHECK_THROW(cudaMemcpyAsync(
			second_seed.data(), second_seed_host.data(), second_seed.n_bytes(), cudaMemcpyHostToDevice, stream
		));
		pitched_second_seed.upload(stream);

		GPUMatrixDynamic<float> contiguous_upstream_gradient{output_width, batch_size, layout};
		GPUMatrixDynamic<float> contiguous_second_input_gradient{input_width, batch_size, layout};
		GuardedMatrix pitched_upstream_gradient{output_width, batch_size, layout, 7, guard};
		GuardedMatrix pitched_second_input_gradient{input_width, batch_size, layout, 5, guard};
		pitched_upstream_gradient.upload(stream);
		pitched_second_input_gradient.upload(stream);
		encoding->backward_backward_input(
			stream,
			*pitched_context,
			pitched_input.matrix,
			pitched_second_seed.matrix,
			pitched_upstream.matrix,
			&pitched_upstream_gradient.matrix,
			&pitched_second_input_gradient.matrix,
			false,
			GradientMode::Overwrite
		);
		std::vector<float> pitched_second_parameter_gradient(encoding->n_params());
		CUDA_CHECK_THROW(cudaMemcpyAsync(
			pitched_second_parameter_gradient.data(),
			encoding->gradients(),
			encoding->n_params() * sizeof(float),
			cudaMemcpyDeviceToHost,
			stream
		));
		encoding->backward_backward_input(
			stream,
			*contiguous_context,
			input,
			second_seed,
			contiguous_upstream,
			&contiguous_upstream_gradient,
			&contiguous_second_input_gradient,
			false,
			GradientMode::Overwrite
		);
		std::vector<float> contiguous_second_parameter_gradient(encoding->n_params());
		CUDA_CHECK_THROW(cudaMemcpyAsync(
			contiguous_second_parameter_gradient.data(),
			encoding->gradients(),
			encoding->n_params() * sizeof(float),
			cudaMemcpyDeviceToHost,
			stream
		));
		pitched_upstream_gradient.download(stream);
		pitched_second_input_gradient.download(stream);
		pitched_input.download(stream);
		pitched_second_seed.download(stream);
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		pitched_upstream_gradient.require_matches(contiguous_upstream_gradient.to_cpu_vector());
		pitched_second_input_gradient.require_matches(contiguous_second_input_gradient.to_cpu_vector());
		pitched_input.require_matches(input_host);
		pitched_second_seed.require_matches(second_seed_host);
		bool found_second_parameter_gradient = false;
		if (layout == AoS) {
			vector_match_rae(pitched_second_parameter_gradient, contiguous_second_parameter_gradient, 1.2e-2, 0.999, true);
		} else {
			for (size_t i = 0; i < pitched_second_parameter_gradient.size(); ++i) {
				REQUIRE(
					pitched_second_parameter_gradient[i] ==
					Approx(contiguous_second_parameter_gradient[i]).margin(1e-5f).epsilon(1e-4f)
				);
			}
		}
		for (float value : pitched_second_parameter_gradient) {
			found_second_parameter_gradient |= value != 0.0f;
		}
		REQUIRE(found_second_parameter_gradient);
		for (uint32_t element = 0; element < batch_size; ++element) {
			REQUIRE(pitched_second_input_gradient.get(input_width - 1, element) == 0.0f);
		}
	}
}

TEST_CASE("Multilevel LoD accepts generic bases and canonicalizes modes", "[encoding][permuto][grid][lod]") {
	using T = network_precision_t;
	for (const auto& [input_mode, canonical_mode] : std::array<std::pair<const char*, const char*>, 4>{
		{{"Hard", "Hard"}, {"discontinuous", "Hard"}, {"Soft", "Soft"}, {"continuous", "Soft"}}
	}) {
		CAPTURE(input_mode);
		json config = hard_lod_config(2, 3);
		config["lod_type"] = input_mode;
		std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(6, config, 8)};
		auto multilevel = std::dynamic_pointer_cast<MultiLevelEncoding<T>>(encoding);
		REQUIRE(multilevel != nullptr);
		REQUIRE(multilevel->n_pos_dims() == 5);
		REQUIRE(multilevel->n_levels() == 2);
		REQUIRE(multilevel->n_features_per_level() == 2);
		REQUIRE(encoding->hyperparams().at("lod_type") == canonical_mode);
		std::shared_ptr<Encoding<T>> restored{create_encoding<T>(encoding->input_width(), encoding->hyperparams(), 8)};
		REQUIRE(restored->hyperparams() == encoding->hyperparams());
		REQUIRE(restored->n_params() == encoding->n_params());
	}

	json invalid_mode = hard_lod_config(2, 3);
	invalid_mode["lod_type"] = "Smooth";
	REQUIRE_THROWS_WITH(
		create_encoding<T>(6, invalid_mode, 8),
		"MultiLevelEncodingLoD: lod_type must be Hard, Discontinuous, Soft, or Continuous."
	);

	json non_multilevel = hard_lod_config(2, 3);
	non_multilevel["base"] = {{"otype", "Identity"}};
	REQUIRE_THROWS_WITH(
		create_encoding<T>(6, non_multilevel, 8), "MultiLevelEncodingLoD requires a multi-level base encoding."
	);

	json nested = hard_lod_config(2, 3);
	nested["base"] = hard_lod_config(2, 3);
	REQUIRE_THROWS_WITH(
		create_encoding<T>(7, nested, 8), "MultiLevelEncodingLoD cannot wrap another MultiLevelEncodingLoD."
	);

	std::shared_ptr<Encoding<T>> grid{create_encoding<T>(2, grid_config(), 8)};
	std::shared_ptr<Encoding<T>> wrapped_grid{create_encoding<T>(3, grid_lod_config("Hard"), 8)};
	auto grid_multilevel = std::dynamic_pointer_cast<MultiLevelEncoding<T>>(grid);
	auto wrapped_multilevel = std::dynamic_pointer_cast<MultiLevelEncoding<T>>(wrapped_grid);
	REQUIRE(grid_multilevel != nullptr);
	REQUIRE(wrapped_multilevel != nullptr);
	REQUIRE(wrapped_grid->input_width() == grid->input_width() + 1);
	REQUIRE(wrapped_grid->output_width() == grid->output_width());
	REQUIRE(wrapped_grid->required_output_alignment() == grid->required_output_alignment());
	REQUIRE(wrapped_grid->preferred_output_layout() == grid->preferred_output_layout());
	REQUIRE(wrapped_grid->n_params() == grid->n_params());
	REQUIRE(wrapped_multilevel->n_pos_dims() == grid_multilevel->n_pos_dims());
	REQUIRE(wrapped_multilevel->n_levels() == grid_multilevel->n_levels());
	REQUIRE(wrapped_multilevel->n_features_per_level() == grid_multilevel->n_features_per_level());
	REQUIRE(wrapped_multilevel->params_offset_table().size == grid_multilevel->params_offset_table().size);
	for (uint32_t level = 0; level < grid_multilevel->n_levels(); ++level) {
		REQUIRE(wrapped_multilevel->level_n_params(level) == grid_multilevel->level_n_params(level));
		REQUIRE(wrapped_multilevel->level_params_offset(level) == grid_multilevel->level_params_offset(level));
	}
	for (uint32_t i = 0; i < grid_multilevel->params_offset_table().size; ++i) {
		REQUIRE(wrapped_multilevel->params_offset_table().data[i] == grid_multilevel->params_offset_table().data[i]);
	}
}

TEST_CASE("Multilevel interface derives a backward-compatible level count", "[encoding][permuto][grid][lod]") {
	LegacyMultiLevelEncoding empty{0};
	LegacyMultiLevelEncoding three_offsets{3};

	REQUIRE_FALSE(std::is_abstract<LegacyMultiLevelEncoding>::value);
	REQUIRE(empty.n_levels() == 0);
	REQUIRE(three_offsets.n_levels() == 2);
}

TEST_CASE("Multilevel LoD composes inherited scalar and GPU level controls", "[encoding][permuto][grid][lod][double-backward]") {
	tcnn_test_setup();

	using T = float;
	constexpr uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	for (const std::string base : {"Grid", "Permuto"}) {
		for (const char* lod_type : {"Hard", "Soft"}) {
			for (const bool use_gpu_control : {false, true}) {
				CAPTURE(base, lod_type, use_gpu_control);
				json config = base == "Grid" ? grid_lod_config(lod_type) : hard_lod_config(2, 3);
				config["lod_type"] = lod_type;
				const uint32_t input_width = base == "Grid" ? 3 : 6;

				std::shared_ptr<Encoding<T>> controlled{create_encoding<T>(input_width, config, 4)};
				std::shared_ptr<Encoding<T>> reference{create_encoding<T>(input_width, config, 4)};
				auto controlled_multilevel = std::dynamic_pointer_cast<MultiLevelEncoding<T>>(controlled);
				REQUIRE(controlled_multilevel != nullptr);
				controlled->set_jit_fusion(false);
				reference->set_jit_fusion(false);

				auto controlled_optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
				auto reference_optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
				auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
				auto controlled_trainer = std::make_shared<Trainer<float, T, T>>(controlled, controlled_optimizer, loss);
				auto reference_trainer = std::make_shared<Trainer<float, T, T>>(reference, reference_optimizer, loss);

				std::vector<T> params(controlled->n_params());
				for (size_t i = 0; i < params.size(); ++i) {
					params[i] = (T)(static_cast<int>((i * 23) % 41) - 20) / 128.0f;
				}
				CUDA_CHECK_THROW(cudaMemcpy(controlled->params(), params.data(), controlled->n_params() * sizeof(T), cudaMemcpyHostToDevice));
				CUDA_CHECK_THROW(cudaMemcpy(reference->params(), params.data(), reference->n_params() * sizeof(T), cudaMemcpyHostToDevice));

				std::vector<float> controlled_input_host(input_width * batch_size);
				std::vector<float> reference_input_host(input_width * batch_size);
				std::vector<float> gpu_controls(batch_size);
				for (uint32_t element = 0; element < batch_size; ++element) {
					for (uint32_t dim = 0; dim + 1 < input_width; ++dim) {
						const float value = 0.05f + (float)((element * 7 + dim * 11) % 89) / 100.0f;
						controlled_input_host[element * input_width + dim] = value;
						reference_input_host[element * input_width + dim] = value;
					}
					gpu_controls[element] = element % 2 == 0 ? 0.25f : 0.75f;
					controlled_input_host[element * input_width + input_width - 1] = 1.0f;
					reference_input_host[element * input_width + input_width - 1] =
						use_gpu_control ? gpu_controls[element] : 0.25f;
				}

				GPUMatrix<float> gpu_control{1, batch_size};
				if (use_gpu_control) {
					CUDA_CHECK_THROW(cudaMemcpy(gpu_control.data(), gpu_controls.data(), gpu_control.n_bytes(), cudaMemcpyHostToDevice));
					controlled_multilevel->set_max_level_gpu(gpu_control.data());
				} else {
					controlled_multilevel->set_max_level(0.25f);
				}

				GPUMatrix<float> controlled_input{input_width, batch_size};
				GPUMatrix<float> reference_input{input_width, batch_size};
				GPUMatrix<T> controlled_output{controlled->padded_output_width(), batch_size};
				GPUMatrix<T> reference_output{reference->padded_output_width(), batch_size};
				GPUMatrix<T> dL_doutput{controlled->padded_output_width(), batch_size};
				GPUMatrix<float> dL_ddLdinput{input_width, batch_size};
				GPUMatrix<float> controlled_dL_dinput{input_width, batch_size};
				GPUMatrix<float> reference_dL_dinput{input_width, batch_size};
				GPUMatrix<T> controlled_dL_ddLdoutput{controlled->padded_output_width(), batch_size};
				GPUMatrix<T> reference_dL_ddLdoutput{reference->padded_output_width(), batch_size};
				GPUMatrix<float> controlled_second_dL_dinput{input_width, batch_size};
				GPUMatrix<float> reference_second_dL_dinput{input_width, batch_size};
				CUDA_CHECK_THROW(cudaMemcpy(
					controlled_input.data(), controlled_input_host.data(), controlled_input.n_bytes(), cudaMemcpyHostToDevice
				));
				CUDA_CHECK_THROW(cudaMemcpy(
					reference_input.data(), reference_input_host.data(), reference_input.n_bytes(), cudaMemcpyHostToDevice
				));
				pcg32 rng{0xdeadbeef};
				dL_doutput.initialize_uniform(rng, -0.5f, 0.5f);
				dL_ddLdinput.initialize_uniform(rng, -0.5f, 0.5f);

				auto controlled_context = controlled->forward(controlled_input, &controlled_output, false, true);
				auto reference_context = reference->forward(reference_input, &reference_output, false, true);
				vector_match_rae(controlled_output.to_cpu_vector(), reference_output.to_cpu_vector(), 1e-5);

				controlled->backward(
					*controlled_context, controlled_input, controlled_output, dL_doutput, &controlled_dL_dinput, false, GradientMode::Overwrite
				);
				reference->backward(
					*reference_context, reference_input, reference_output, dL_doutput, &reference_dL_dinput, false, GradientMode::Overwrite
				);
				vector_match_rae(controlled_dL_dinput.to_cpu_vector(), reference_dL_dinput.to_cpu_vector(), 1e-5);
				std::vector<T> controlled_parameter_gradient(controlled->n_params());
				std::vector<T> reference_parameter_gradient(reference->n_params());
				CUDA_CHECK_THROW(cudaMemcpy(
					controlled_parameter_gradient.data(), controlled->gradients(), controlled->n_params() * sizeof(T), cudaMemcpyDeviceToHost
				));
				CUDA_CHECK_THROW(cudaMemcpy(
					reference_parameter_gradient.data(), reference->gradients(), reference->n_params() * sizeof(T), cudaMemcpyDeviceToHost
				));
				vector_match_rae(controlled_parameter_gradient, reference_parameter_gradient, 1.2e-2, 0.999, true);

				controlled->backward_backward_input(
					*controlled_context,
					controlled_input,
					dL_ddLdinput,
					dL_doutput,
					&controlled_dL_ddLdoutput,
					&controlled_second_dL_dinput,
					false,
					GradientMode::Overwrite
				);
				reference->backward_backward_input(
					*reference_context,
					reference_input,
					dL_ddLdinput,
					dL_doutput,
					&reference_dL_ddLdoutput,
					&reference_second_dL_dinput,
					false,
					GradientMode::Overwrite
				);
				vector_match_rae(controlled_dL_ddLdoutput.to_cpu_vector(), reference_dL_ddLdoutput.to_cpu_vector(), 1e-5);
				vector_match_rae(controlled_second_dL_dinput.to_cpu_vector(), reference_second_dL_dinput.to_cpu_vector(), 1e-5);
				CUDA_CHECK_THROW(cudaMemcpy(
					controlled_parameter_gradient.data(), controlled->gradients(), controlled->n_params() * sizeof(T), cudaMemcpyDeviceToHost
				));
				CUDA_CHECK_THROW(cudaMemcpy(
					reference_parameter_gradient.data(), reference->gradients(), reference->n_params() * sizeof(T), cudaMemcpyDeviceToHost
				));
				vector_match_rae(controlled_parameter_gradient, reference_parameter_gradient, 1.2e-2, 0.999, true);
			}
		}
	}
}

TEST_CASE("Soft LoD weights native Permuto forward and backward", "[encoding][permuto][lod]") {
	tcnn_test_setup();

	using T = float;
	constexpr uint32_t n_levels = 2;
	constexpr uint32_t n_features_per_level = 2;
	constexpr uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	std::shared_ptr<Encoding<T>> base{create_encoding<T>(5, permuto_config(n_levels, 3, n_features_per_level), 8)};
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(6, soft_lod_config(n_levels, 3, n_features_per_level), 8)};
	auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
	auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
	auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);
	auto base_optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
	auto base_loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
	auto base_trainer = std::make_shared<Trainer<float, T, T>>(base, base_optimizer, base_loss);

	REQUIRE(encoding->n_params() == base->n_params());
	std::vector<T> params(encoding->n_params());
	for (size_t i = 0; i < params.size(); ++i) {
		params[i] = (T)(static_cast<int>((i * 17) % 31) - 15) / 128.0f;
	}
	CUDA_CHECK_THROW(cudaMemcpy(encoding->params(), params.data(), encoding->n_params() * sizeof(T), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(cudaMemcpy(base->params(), params.data(), base->n_params() * sizeof(T), cudaMemcpyHostToDevice));

	const float boundary_ratio = (1.0f - 1e-3f) / n_levels;
	const std::vector<float> ratios = {
		-std::numeric_limits<float>::max(), -0.25f, 0.0f, boundary_ratio, 0.5f, 0.75f, 1.0f, 1.25f,
		std::numeric_limits<float>::max(),
	};
	const auto input_host = hard_lod_input(batch_size, ratios);
	std::vector<float> base_input_host(5 * batch_size);
	for (uint32_t element = 0; element < batch_size; ++element) {
		std::copy_n(input_host.data() + element * 6, 5, base_input_host.data() + element * 5);
	}

	GPUMatrix<float> input{6, batch_size};
	GPUMatrix<float> base_input{5, batch_size};
	GPUMatrix<T> output{encoding->padded_output_width(), batch_size};
	GPUMatrix<T> base_output{base->padded_output_width(), batch_size};
	CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(cudaMemcpy(base_input.data(), base_input_host.data(), base_input.n_bytes(), cudaMemcpyHostToDevice));

	auto context = encoding->forward(input, &output, false, true);
	auto base_context = base->forward(base_input, &base_output, false, true);
	const auto output_host = output.to_cpu_vector();
	const auto base_output_host = base_output.to_cpu_vector();
	for (uint32_t element = 0; element < batch_size; ++element) {
		const float ratio = ratios[element % ratios.size()];
		for (uint32_t feature = 0; feature < encoding->padded_output_width(); ++feature) {
			const float weight = feature < n_levels * n_features_per_level
				? soft_lod_weight(ratio, n_levels, feature / n_features_per_level)
				: 1.0f;
			const float expected = weight * base_output_host[element * base->padded_output_width() + feature];
			REQUIRE(output_host[element * encoding->padded_output_width() + feature] == Approx(expected).margin(1e-6f));
		}
	}

	GPUMatrix<T> dL_doutput{encoding->padded_output_width(), batch_size};
	GPUMatrix<T> base_dL_doutput{base->padded_output_width(), batch_size};
	std::vector<T> dL_doutput_host(dL_doutput.n_elements());
	std::vector<T> base_dL_doutput_host(base_dL_doutput.n_elements());
	for (uint32_t element = 0; element < batch_size; ++element) {
		const float ratio = ratios[element % ratios.size()];
		for (uint32_t feature = 0; feature < encoding->padded_output_width(); ++feature) {
			const size_t index = element * encoding->padded_output_width() + feature;
			dL_doutput_host[index] = (T)(static_cast<int>((index * 13) % 23) - 11) / 32.0f;
			const float weight = feature < n_levels * n_features_per_level
				? soft_lod_weight(ratio, n_levels, feature / n_features_per_level)
				: 1.0f;
			base_dL_doutput_host[index] = weight * dL_doutput_host[index];
		}
	}
	CUDA_CHECK_THROW(cudaMemcpy(dL_doutput.data(), dL_doutput_host.data(), dL_doutput.n_bytes(), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(
		cudaMemcpy(base_dL_doutput.data(), base_dL_doutput_host.data(), base_dL_doutput.n_bytes(), cudaMemcpyHostToDevice)
	);

	GPUMatrix<float> dL_dinput{6, batch_size};
	GPUMatrix<float> base_dL_dinput{5, batch_size};
	encoding->backward(*context, input, output, dL_doutput, &dL_dinput, false, GradientMode::Overwrite);
	base->backward(*base_context, base_input, base_output, base_dL_doutput, &base_dL_dinput, false, GradientMode::Overwrite);
	const auto dL_dinput_host = dL_dinput.to_cpu_vector();
	const auto base_dL_dinput_host = base_dL_dinput.to_cpu_vector();
	for (uint32_t element = 0; element < batch_size; ++element) {
		for (uint32_t dim = 0; dim < 5; ++dim) {
			const float expected = base_dL_dinput_host[element * 5 + dim];
			REQUIRE(dL_dinput_host[element * 6 + dim] == Approx(expected).margin(1e-5f).epsilon(1e-4f));
		}
		REQUIRE(dL_dinput_host[element * 6 + 5] == 0.0f);
	}

	std::vector<T> parameter_gradient(encoding->n_params());
	std::vector<T> base_parameter_gradient(base->n_params());
	CUDA_CHECK_THROW(
		cudaMemcpy(parameter_gradient.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost)
	);
	CUDA_CHECK_THROW(
		cudaMemcpy(base_parameter_gradient.data(), base->gradients(), base->n_params() * sizeof(T), cudaMemcpyDeviceToHost)
	);
	bool found_parameter_gradient = false;
	for (size_t i = 0; i < parameter_gradient.size(); ++i) {
		REQUIRE(parameter_gradient[i] == Approx(base_parameter_gradient[i]).margin(1e-5f).epsilon(1e-4f));
		found_parameter_gradient |= parameter_gradient[i] != 0.0f;
	}
	REQUIRE(found_parameter_gradient);

	encoding->backward(*context, input, output, dL_doutput, nullptr, false, GradientMode::Accumulate);
	std::vector<T> accumulated(encoding->n_params());
	CUDA_CHECK_THROW(
		cudaMemcpy(accumulated.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost)
	);
	for (size_t i = 0; i < parameter_gradient.size(); ++i) {
		REQUIRE(accumulated[i] == Approx(2.0f * parameter_gradient[i]).margin(1e-5f).epsilon(1e-4f));
	}
	encoding->backward(*context, input, output, dL_doutput, nullptr, false, GradientMode::Ignore);
	std::vector<T> ignored(encoding->n_params());
	CUDA_CHECK_THROW(cudaMemcpy(ignored.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost));
	REQUIRE(ignored == accumulated);

	auto context_only = encoding->forward(input, nullptr, false, true);
	GPUMatrix<float> context_only_dL_dinput{6, batch_size};
	encoding->backward(
		*context_only, input, output, dL_doutput, &context_only_dL_dinput, false, GradientMode::Ignore
	);
	const auto context_only_gradient = context_only_dL_dinput.to_cpu_vector();
	for (size_t i = 0; i < context_only_gradient.size(); ++i) {
		REQUIRE(context_only_gradient[i] == Approx(dL_dinput_host[i]).margin(1e-5f).epsilon(1e-4f));
	}

	const auto scalar_loss = [&](const std::vector<float>& candidate_input) {
		CUDA_CHECK_THROW(cudaMemcpy(input.data(), candidate_input.data(), input.n_bytes(), cudaMemcpyHostToDevice));
		encoding->forward(input, &output);
		const auto candidate_output = output.to_cpu_vector();
		float result = 0.0f;
		for (size_t i = 0; i < candidate_output.size(); ++i) {
			result += candidate_output[i] * dL_doutput_host[i];
		}
		return result;
	};
	constexpr float input_epsilon = 1e-4f;
	constexpr uint32_t finite_difference_element = 5;
	constexpr uint32_t finite_difference_index = finite_difference_element * 6;
	auto lower_input = input_host;
	auto upper_input = input_host;
	lower_input[finite_difference_index] -= input_epsilon;
	upper_input[finite_difference_index] += input_epsilon;
	const float finite_difference = (scalar_loss(upper_input) - scalar_loss(lower_input)) / (2.0f * input_epsilon);
	REQUIRE(dL_dinput_host[finite_difference_index] != 0.0f);
	REQUIRE(finite_difference != 0.0f);
	REQUIRE(dL_dinput_host[finite_difference_index] == Approx(finite_difference).margin(1e-2f).epsilon(2e-2f));

	const auto parameter_it = std::find_if(
		parameter_gradient.begin(), parameter_gradient.end(), [](T value) { return std::abs(value) > 1e-5f; }
	);
	REQUIRE(parameter_it != parameter_gradient.end());
	const size_t parameter_index = parameter_it - parameter_gradient.begin();
	constexpr float parameter_epsilon = 1.0f / 256.0f;
	auto lower_params = params;
	auto upper_params = params;
	lower_params[parameter_index] -= parameter_epsilon;
	upper_params[parameter_index] += parameter_epsilon;
	CUDA_CHECK_THROW(
		cudaMemcpy(encoding->params(), upper_params.data(), encoding->n_params() * sizeof(T), cudaMemcpyHostToDevice)
	);
	const float upper_loss = scalar_loss(input_host);
	CUDA_CHECK_THROW(
		cudaMemcpy(encoding->params(), lower_params.data(), encoding->n_params() * sizeof(T), cudaMemcpyHostToDevice)
	);
	const float lower_loss = scalar_loss(input_host);
	CUDA_CHECK_THROW(cudaMemcpy(encoding->params(), params.data(), encoding->n_params() * sizeof(T), cudaMemcpyHostToDevice));
	const float parameter_finite_difference = (upper_loss - lower_loss) / (2.0f * parameter_epsilon);
	REQUIRE(parameter_gradient[parameter_index] == Approx(parameter_finite_difference).margin(1e-2f).epsilon(2e-2f));
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

TEST_CASE("Grid-backed hard LoD uses one exact boundary at both derivative orders", "[encoding][grid][lod][double-backward]") {
	tcnn_test_setup();

	using T = float;
	constexpr uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(3, grid_lod_config("Hard"), 4)};
	auto multilevel = std::dynamic_pointer_cast<MultiLevelEncoding<T>>(encoding);
	auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
	auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
	auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);
	REQUIRE(multilevel != nullptr);
	REQUIRE(multilevel->n_levels() == 2);

	std::vector<T> params(encoding->n_params());
	for (size_t i = 0; i < params.size(); ++i) {
		params[i] = (T)(static_cast<int>((i * 19) % 37) - 18) / 64.0f;
	}
	CUDA_CHECK_THROW(cudaMemcpy(encoding->params(), params.data(), encoding->n_params() * sizeof(T), cudaMemcpyHostToDevice));
	const size_t level_one_offset = multilevel->level_params_offset(1) * multilevel->n_features_per_level();

	const float boundary_ratio = (1.0f - 1e-3f) / 2.0f;
	for (const auto& [ratio, level_one_enabled] : std::array<std::pair<float, bool>, 2>{{
		{boundary_ratio, false}, {0.5f, true}
	}}) {
		CAPTURE(ratio, level_one_enabled);
		std::vector<float> input_host(3 * batch_size);
		std::vector<T> dL_doutput_host(encoding->padded_output_width() * batch_size, 0.0f);
		std::vector<float> dL_ddLdinput_host(3 * batch_size);
		for (uint32_t element = 0; element < batch_size; ++element) {
			input_host[element * 3] = 0.137f;
			input_host[element * 3 + 1] = 0.619f;
			input_host[element * 3 + 2] = ratio;
			dL_doutput_host[element * encoding->padded_output_width() + 2] = 0.75f;
			dL_doutput_host[element * encoding->padded_output_width() + 3] = -0.5f;
			dL_ddLdinput_host[element * 3] = 1.0f;
			dL_ddLdinput_host[element * 3 + 1] = -0.25f;
			dL_ddLdinput_host[element * 3 + 2] = 100.0f;
		}

		GPUMatrix<float> input{3, batch_size};
		GPUMatrix<T> output{encoding->padded_output_width(), batch_size};
		GPUMatrix<T> dL_doutput{encoding->padded_output_width(), batch_size};
		GPUMatrix<float> dL_dinput{3, batch_size};
		GPUMatrix<float> dL_ddLdinput{3, batch_size};
		GPUMatrix<T> dL_ddLdoutput{encoding->padded_output_width(), batch_size};
		GPUMatrix<float> second_dL_dinput{3, batch_size};
		CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));
		CUDA_CHECK_THROW(cudaMemcpy(dL_doutput.data(), dL_doutput_host.data(), dL_doutput.n_bytes(), cudaMemcpyHostToDevice));
		CUDA_CHECK_THROW(
			cudaMemcpy(dL_ddLdinput.data(), dL_ddLdinput_host.data(), dL_ddLdinput.n_bytes(), cudaMemcpyHostToDevice)
		);

		auto context = encoding->forward(input, &output, false, true);
		const auto output_host = output.to_cpu_vector();
		bool has_level_one_output = false;
		for (uint32_t element = 0; element < batch_size; ++element) {
			for (uint32_t feature = 2; feature < 4; ++feature) {
				has_level_one_output |= output_host[element * encoding->padded_output_width() + feature] != 0.0f;
			}
		}
		REQUIRE(has_level_one_output == level_one_enabled);

		encoding->backward(*context, input, output, dL_doutput, &dL_dinput, false, GradientMode::Overwrite);
		const auto first_input_gradient = dL_dinput.to_cpu_vector();
		std::vector<T> first_parameter_gradient(encoding->n_params());
		CUDA_CHECK_THROW(cudaMemcpy(
			first_parameter_gradient.data(),
			encoding->gradients(),
			encoding->n_params() * sizeof(T),
			cudaMemcpyDeviceToHost
		));
		const bool has_first_input_gradient = std::any_of(
			first_input_gradient.begin(), first_input_gradient.end(), [](float value) { return value != 0.0f; }
		);
		const bool has_first_parameter_gradient = std::any_of(
			first_parameter_gradient.begin() + level_one_offset,
			first_parameter_gradient.end(),
			[](T value) { return value != 0.0f; }
		);
		REQUIRE(has_first_input_gradient == level_one_enabled);
		REQUIRE(has_first_parameter_gradient == level_one_enabled);
		for (uint32_t element = 0; element < batch_size; ++element) {
			REQUIRE(first_input_gradient[element * 3 + 2] == 0.0f);
		}

		encoding->backward_backward_input(
			*context,
			input,
			dL_ddLdinput,
			dL_doutput,
			&dL_ddLdoutput,
			&second_dL_dinput,
			false,
			GradientMode::Overwrite
		);
		const auto upstream_gradient = dL_ddLdoutput.to_cpu_vector();
		const auto second_input_gradient = second_dL_dinput.to_cpu_vector();
		std::vector<T> second_parameter_gradient(encoding->n_params());
		CUDA_CHECK_THROW(cudaMemcpy(
			second_parameter_gradient.data(),
			encoding->gradients(),
			encoding->n_params() * sizeof(T),
			cudaMemcpyDeviceToHost
		));
		bool has_level_one_upstream_gradient = false;
		for (uint32_t element = 0; element < batch_size; ++element) {
			for (uint32_t feature = 2; feature < 4; ++feature) {
				has_level_one_upstream_gradient |= upstream_gradient[element * encoding->padded_output_width() + feature] != 0.0f;
			}
			REQUIRE(second_input_gradient[element * 3 + 2] == 0.0f);
		}
		const bool has_second_input_gradient = std::any_of(
			second_input_gradient.begin(), second_input_gradient.end(), [](float value) { return value != 0.0f; }
		);
		const bool has_second_parameter_gradient = std::any_of(
			second_parameter_gradient.begin() + level_one_offset,
			second_parameter_gradient.end(),
			[](T value) { return value != 0.0f; }
		);
		REQUIRE(has_level_one_upstream_gradient == level_one_enabled);
		REQUIRE(has_second_input_gradient == level_one_enabled);
		REQUIRE(has_second_parameter_gradient == level_one_enabled);
	}
}

TEST_CASE("Grid-backed soft LoD weights native double backward", "[encoding][grid][lod][double-backward]") {
	tcnn_test_setup();

	using T = float;
	constexpr uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	constexpr float ratio = 0.75f;
	std::shared_ptr<Encoding<T>> base{create_encoding<T>(2, grid_config(), 4)};
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(3, grid_lod_config("Soft"), 4)};
	auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
	auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
	auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);
	auto base_optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
	auto base_loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
	auto base_trainer = std::make_shared<Trainer<float, T, T>>(base, base_optimizer, base_loss);

	std::vector<T> params(encoding->n_params());
	for (size_t i = 0; i < params.size(); ++i) {
		params[i] = (T)((static_cast<int>((i * 23) % 41) - 20) / 128.0f);
	}
	CUDA_CHECK_THROW(cudaMemcpy(encoding->params(), params.data(), encoding->n_params() * sizeof(T), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(cudaMemcpy(base->params(), params.data(), base->n_params() * sizeof(T), cudaMemcpyHostToDevice));

	std::vector<float> input_host(3 * batch_size);
	std::vector<float> base_input_host(2 * batch_size);
	std::vector<float> dL_ddLdinput_host(3 * batch_size);
	std::vector<float> base_dL_ddLdinput_host(2 * batch_size);
	std::vector<T> dL_doutput_host(encoding->padded_output_width() * batch_size);
	std::vector<T> base_dL_doutput_host(base->padded_output_width() * batch_size);
	for (uint32_t element = 0; element < batch_size; ++element) {
		input_host[element * 3] = base_input_host[element * 2] = 0.173f;
		input_host[element * 3 + 1] = base_input_host[element * 2 + 1] = 0.587f;
		input_host[element * 3 + 2] = ratio;
		dL_ddLdinput_host[element * 3] = base_dL_ddLdinput_host[element * 2] = 0.75f;
		dL_ddLdinput_host[element * 3 + 1] = base_dL_ddLdinput_host[element * 2 + 1] = -0.5f;
		dL_ddLdinput_host[element * 3 + 2] = 100.0f;
		for (uint32_t feature = 0; feature < encoding->padded_output_width(); ++feature) {
			const size_t index = element * encoding->padded_output_width() + feature;
			dL_doutput_host[index] = (T)(static_cast<int>((index * 11) % 29) - 14) / 32.0f;
			const float weight = feature < 4 ? soft_lod_weight(ratio, 2, feature / 2) : 1.0f;
			base_dL_doutput_host[index] = weight * dL_doutput_host[index];
		}
	}

	GPUMatrix<float> input{3, batch_size};
	GPUMatrix<float> base_input{2, batch_size};
	GPUMatrix<T> output{encoding->padded_output_width(), batch_size};
	GPUMatrix<T> base_output{base->padded_output_width(), batch_size};
	GPUMatrix<T> dL_doutput{encoding->padded_output_width(), batch_size};
	GPUMatrix<T> base_dL_doutput{base->padded_output_width(), batch_size};
	GPUMatrix<float> dL_ddLdinput{3, batch_size};
	GPUMatrix<float> base_dL_ddLdinput{2, batch_size};
	GPUMatrix<T> dL_ddLdoutput{encoding->padded_output_width(), batch_size};
	GPUMatrix<T> base_dL_ddLdoutput{base->padded_output_width(), batch_size};
	GPUMatrix<float> dL_dinput{3, batch_size};
	GPUMatrix<float> base_dL_dinput{2, batch_size};
	CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(cudaMemcpy(base_input.data(), base_input_host.data(), base_input.n_bytes(), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(cudaMemcpy(dL_doutput.data(), dL_doutput_host.data(), dL_doutput.n_bytes(), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(
		cudaMemcpy(base_dL_doutput.data(), base_dL_doutput_host.data(), base_dL_doutput.n_bytes(), cudaMemcpyHostToDevice)
	);
	CUDA_CHECK_THROW(
		cudaMemcpy(dL_ddLdinput.data(), dL_ddLdinput_host.data(), dL_ddLdinput.n_bytes(), cudaMemcpyHostToDevice)
	);
	CUDA_CHECK_THROW(cudaMemcpy(
		base_dL_ddLdinput.data(), base_dL_ddLdinput_host.data(), base_dL_ddLdinput.n_bytes(), cudaMemcpyHostToDevice
	));

	auto context = encoding->forward(input, &output, false, true);
	auto base_context = base->forward(base_input, &base_output, false, true);
	encoding->backward_backward_input(
		*context, input, dL_ddLdinput, dL_doutput, &dL_ddLdoutput, &dL_dinput, false, GradientMode::Overwrite
	);
	base->backward_backward_input(
		*base_context,
		base_input,
		base_dL_ddLdinput,
		base_dL_doutput,
		&base_dL_ddLdoutput,
		&base_dL_dinput,
		false,
		GradientMode::Overwrite
	);

	const auto upstream_gradient = dL_ddLdoutput.to_cpu_vector();
	const auto base_upstream_gradient = base_dL_ddLdoutput.to_cpu_vector();
	const auto input_gradient = dL_dinput.to_cpu_vector();
	const auto base_input_gradient = base_dL_dinput.to_cpu_vector();
	bool found_upstream_gradient = false;
	bool found_input_gradient = false;
	for (uint32_t element = 0; element < batch_size; ++element) {
		for (uint32_t feature = 0; feature < encoding->padded_output_width(); ++feature) {
			const size_t index = element * encoding->padded_output_width() + feature;
			const float weight = feature < 4 ? soft_lod_weight(ratio, 2, feature / 2) : 1.0f;
			const float expected = weight * base_upstream_gradient[index];
			REQUIRE(upstream_gradient[index] == Approx(expected).margin(1e-5f).epsilon(1e-4f));
			found_upstream_gradient |= upstream_gradient[index] != 0.0f;
		}
		for (uint32_t dim = 0; dim < 2; ++dim) {
			const float expected = base_input_gradient[element * 2 + dim];
			REQUIRE(input_gradient[element * 3 + dim] == Approx(expected).margin(1e-5f).epsilon(1e-4f));
			found_input_gradient |= input_gradient[element * 3 + dim] != 0.0f;
		}
		REQUIRE(input_gradient[element * 3 + 2] == 0.0f);
	}
	REQUIRE(found_upstream_gradient);
	REQUIRE(found_input_gradient);

	std::vector<T> parameter_gradient(encoding->n_params());
	std::vector<T> base_parameter_gradient(base->n_params());
	CUDA_CHECK_THROW(
		cudaMemcpy(parameter_gradient.data(), encoding->gradients(), encoding->n_params() * sizeof(T), cudaMemcpyDeviceToHost)
	);
	CUDA_CHECK_THROW(
		cudaMemcpy(base_parameter_gradient.data(), base->gradients(), base->n_params() * sizeof(T), cudaMemcpyDeviceToHost)
	);
	bool found_parameter_gradient = false;
	for (size_t i = 0; i < parameter_gradient.size(); ++i) {
		REQUIRE(parameter_gradient[i] == Approx(base_parameter_gradient[i]).margin(1e-5f).epsilon(1e-4f));
		found_parameter_gradient |= parameter_gradient[i] != 0.0f;
	}
	REQUIRE(found_parameter_gradient);
}

TEST_CASE("Multilevel LoD training supports CUDA graph capture", "[encoding][permuto][lod]") {
	tcnn_test_setup();

	using T = network_precision_t;
	for (const char* lod_type : {"Hard", "Soft"}) {
		CAPTURE(lod_type);
		json config = hard_lod_config();
		config["lod_type"] = lod_type;
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
}

TEST_CASE("Generalized Permuto supports native double backward", "[encoding][permuto][double-backward]") {
	tcnn_test_setup();

	const auto run_cases = [](auto precision) {
		using T = decltype(precision);
		(void)precision;
		constexpr std::array<std::pair<uint32_t, uint32_t>, 2> test_configs = {{{1, 1}, {24, 8}}};
		for (const auto& [n_dims, n_features_per_level] : test_configs) {
			CAPTURE(n_dims, n_features_per_level);
			const bool maximum_width = n_features_per_level == 8;
			json config = permuto_config(maximum_width ? 32 : 2, 2, n_features_per_level);
			if (maximum_width) {
				config["base_scale"] = 1.0f;
				config["per_level_scale"] = 1.0f;
			}
			config["max_input_grad_dims"] = n_dims;
			std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(n_dims, config, n_features_per_level)};
			auto optimizer = std::shared_ptr<Optimizer<T>>{create_optimizer<T>(json::object())};
			auto loss = std::shared_ptr<Loss<T>>{create_loss<T>(json::object())};
			auto trainer = std::make_shared<Trainer<float, T, T>>(encoding, optimizer, loss);

			const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
			GPUMatrix<float> input{n_dims, batch_size};
			GPUMatrix<T> output{encoding->output_width(), batch_size};
			GPUMatrix<T> dL_doutput{encoding->output_width(), batch_size};
			GPUMatrix<float> dL_ddLdinput{n_dims, batch_size};
			GPUMatrix<T> dL_ddLdoutput{encoding->output_width(), batch_size};
			GPUMatrix<float> dL_dinput{n_dims, batch_size};
			pcg32 rng{0xdeadbeef};
			input.initialize_uniform(rng, 0.05f, 0.95f);
			dL_doutput.initialize_uniform(rng, -0.25f, 0.25f);
			dL_ddLdinput.initialize_uniform(rng, -1.0f, 1.0f);
			dL_dinput.memset(0x7f);

			auto context = encoding->forward(input, &output, false, true);
			encoding->backward_backward_input(
				*context, input, dL_ddLdinput, dL_doutput, &dL_ddLdoutput, &dL_dinput, false, GradientMode::Overwrite
			);

			const auto dL_ddLdoutput_host = dL_ddLdoutput.to_cpu_vector();
			REQUIRE(std::any_of(
				dL_ddLdoutput_host.begin(), dL_ddLdoutput_host.end(), [](T value) { return (float)value != 0.0f; }
			));
			REQUIRE(std::all_of(
				dL_ddLdoutput_host.begin(), dL_ddLdoutput_host.end(), [](T value) { return std::isfinite((float)value); }
			));
			const auto dL_dinput_host = dL_dinput.to_cpu_vector();
			REQUIRE(std::all_of(dL_dinput_host.begin(), dL_dinput_host.end(), [](float value) { return value == 0.0f; }));

			std::vector<T> parameter_gradient(encoding->n_params());
			CUDA_CHECK_THROW(cudaMemcpy(
				parameter_gradient.data(), encoding->gradients(), parameter_gradient.size() * sizeof(T), cudaMemcpyDeviceToHost
			));
			REQUIRE(std::any_of(parameter_gradient.begin(), parameter_gradient.end(), [](T value) { return (float)value != 0.0f; }));
			REQUIRE(std::all_of(parameter_gradient.begin(), parameter_gradient.end(), [](T value) { return std::isfinite((float)value); }));
		}
	};

	run_cases(float{});
	if constexpr (!std::is_same<network_precision_t, float>::value) {
		run_cases(network_precision_t{});
	}
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
