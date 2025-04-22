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

/** @file   test_common.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Common routines used by unit tests.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/object.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/trainer.h>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <string>
#include <vector>

inline void tcnn_test_setup() {
	using namespace tcnn;

	set_log_callback([](LogSeverity severity, const std::string& msg) {
		switch (severity) {
			case LogSeverity::Warning: FAIL(fmt::format("Warning: {}", msg));
			case LogSeverity::Error: FAIL(fmt::format("Error: {}", msg));
			default: break;
		}
	});

	rtc_set_cache_dir(".");
}

// Match two vectors according to symmetric relative absolute error (RAE), which is defined as |a| - |b| / 0.5*(|a| + |b| + epsilon).
// This function additionally allows measuring the error only on a lower percentile of the dataset (to ignore outliers) and to measure
// the mean of the error only (if noise is tolerable but bias is not).
template <typename T, typename U>
void vector_match_rae(const std::vector<T>& v1, const std::vector<U>& v2, double threshold = 1e-2, double percentile = 1.0f, bool mean_error_only = false) {
	REQUIRE(v1.size() == v2.size());

	if (v1.empty()) {
		return;
	}

	double mean = 0.0;

	std::vector<double> relative_errors;
	for (size_t i = 0; i < v1.size(); ++i) {
		double d1 = v1[i], d2 = v2[i];
		REQUIRE(std::isfinite(d1));
		REQUIRE(std::isfinite(d2));

		mean += d1 + d2;
	}

	mean /= v1.size() * 2;

	for (size_t i = 0; i < v1.size(); ++i) {
		double d1 = v1[i], d2 = v2[i];
		// Symmetric relative absolute error. The epsilon value of `mean * tolerance` ensures that values closer to
		// the mean tolerance become less and less relative as the ratios can tend to infinity close to zero.
		// We are fine with absolute errors in that regime, because we know that this regime is only a small
		// magnitude compared to the average value.
		double e = (std::abs(d1) < 1.0e-15 && std::abs(d2) < 1.0e-15) ? 0 : ((std::abs(d1 - d2) / (0.5 * (std::abs(d1) + std::abs(d2)) + mean * threshold)));
		relative_errors.push_back(e);
	}

	// Optionally disregard the X% elements with largest error. This is useful for the parameter and
	// input gradients of the grid encoding, which are picewise-constant, meaning that negligible
	// rouding errors could push a coordinate to a different piecewise-constant region that may cause
	// arbitrary error. With this percentile setting, we allow for a very small number of such cases.
	size_t n_to_check = relative_errors.size();
	if (percentile < 1.0f) {
		n_to_check = (size_t)floor(relative_errors.size() * percentile);
		std::nth_element(std::begin(relative_errors), std::begin(relative_errors) + n_to_check, std::end(relative_errors));
	}

	if (n_to_check == 0) {
		return;
	}

	double mean_relative_error = 0.0;
	for (size_t i = 0; i < n_to_check; ++i) {
		mean_relative_error += relative_errors[i];
		if (!mean_error_only) {
			REQUIRE(relative_errors[i] < threshold);
		}
	}

	mean_relative_error /= n_to_check;
	REQUIRE(mean_relative_error < threshold);
}

template <typename T, typename U>
void vector_match_rae(const std::string& section_name, const std::vector<T>& v1, const std::vector<U>& v2, double threshold = 1e-2, double percentile = 1.0f, bool mean_error_only = false) {
	SECTION(section_name) { vector_match_rae(v1, v2, threshold, percentile, mean_error_only); }
}

template <typename T, typename PARAMS_T, typename COMPUTE_T>
void test_differentiable_object(const std::shared_ptr<tcnn::DifferentiableObject<T, PARAMS_T, COMPUTE_T>>& object) {
	using namespace tcnn;

	const float backward_scale = default_loss_scale<PARAMS_T>();
	const uint32_t batch_size = 32 * BATCH_SIZE_GRANULARITY;
	const uint32_t n_dims = object->input_width();

	if (object->padded_output_width() == 0) {
		return;
	}

	pcg32 rng{0xdeadbeef};

	GPUMatrix<T> input{n_dims, batch_size};
	GPUMatrix<T> input_gradient{n_dims, batch_size};

	std::shared_ptr<Optimizer<COMPUTE_T>> dummy_optimizer{create_optimizer<COMPUTE_T>(json::object())};
	std::shared_ptr<Loss<COMPUTE_T>> dummy_loss{create_loss<COMPUTE_T>(json::object())};
	auto trainer = std::make_shared<Trainer<T, PARAMS_T, COMPUTE_T>>(object, dummy_optimizer, dummy_loss);

	GPUMatrix<COMPUTE_T> output{object->padded_output_width(), batch_size};
	GPUMatrix<COMPUTE_T> output_gradient{object->padded_output_width(), batch_size};
	GPUMatrix<float> output_fp{object->output_width(), batch_size};

	// Encodings expect an input range in [0, 1]^n_dims
	input.initialize_uniform(rng, 0.001f, 0.999f);
	output_gradient.initialize_uniform(rng, -backward_scale, backward_scale);

	SECTION("`inference` and `inference_mixed_precision` match") {
		object->inference(input, output_fp);
		object->inference_mixed_precision(input, output);

		vector_match_rae(output.to_cpu_vector(), output_fp.to_cpu_vector(), 1e-4);
	}

	SECTION("`inference_mixed_precision` and `forward` match") {
		object->inference_mixed_precision(input, output);
		auto v1 = output.to_cpu_vector();
		object->forward(input, &output);

		vector_match_rae(v1, output.to_cpu_vector(), 1e-4);
	}

	if (supports_jit_fusion()) {
		SECTION("JIT matches `inference`") {
			object->set_jit_fusion(true);
			object->inference(input, output_fp);
			auto v1 = output_fp.to_cpu_vector();

			object->set_jit_fusion(false);
			object->inference(input, output_fp);

			vector_match_rae(v1, output_fp.to_cpu_vector(), 1e-2, 0.99);
		}

		SECTION("JIT matches `inference_mixed_precision`") {
			object->set_jit_fusion(true);
			object->inference_mixed_precision(input, output);
			auto v1 = output.to_cpu_vector();

			object->set_jit_fusion(false);
			object->inference_mixed_precision(input, output);

			vector_match_rae(v1, output.to_cpu_vector(), 1e-2, 0.99);
		}

		SECTION("JIT matches `forward` and `backward`") {
			object->set_jit_fusion(true);

			{
				auto ctx = object->forward(input, &output, false, true);
				object->backward(*ctx, input, output, output_gradient, &input_gradient);
			}

			auto o1 = output.to_cpu_vector();
			auto b1 = input_gradient.to_cpu_vector();
			std::vector<PARAMS_T> p1(object->n_params());
			CUDA_CHECK_THROW(cudaMemcpy(p1.data(), object->gradients(), object->n_params() * sizeof(PARAMS_T), cudaMemcpyDeviceToHost));

			object->set_jit_fusion(false);

			{
				auto ctx = object->forward(input, &output, false, true);
				object->backward(*ctx, input, output, output_gradient, &input_gradient);
			}

			auto o2 = output.to_cpu_vector();
			auto b2 = input_gradient.to_cpu_vector();
			std::vector<PARAMS_T> p2(object->n_params());
			CUDA_CHECK_THROW(cudaMemcpy(p2.data(), object->gradients(), object->n_params() * sizeof(PARAMS_T), cudaMemcpyDeviceToHost));

			vector_match_rae("Output", o1, o2, 1e-2, 0.99);
			vector_match_rae("Input gradient", b1, b2, 1e-2, 0.99);
			vector_match_rae("Params gradient", p1, p2, 1.2e-2, 0.999, true);
		}


	}
}
