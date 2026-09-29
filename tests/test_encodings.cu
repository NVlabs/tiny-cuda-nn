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
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   test_encodings.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  Test various invariances of input encodings. E.g. that `inference`
 *          and `inference_mixed_precision` produce the same results, that JIT
 *          produces the same result as no-JIT, as well as the same for derivatives.
 */

#include "test_common.h"

#include <tiny-cuda-nn/encoding.h>

#include <algorithm>

using namespace tcnn;

#if defined(TCNN_NO_FWD_BWD)
TEST_CASE("Offline-only encodings are absent without forward and backward", "[encoding][no-fwd-bwd]") {
	const auto encodings = builtin_encodings();
	const auto contains = [&encodings](const char* name) { return std::find(encodings.begin(), encodings.end(), name) != encodings.end(); };

	REQUIRE_FALSE(contains("Permuto"));
	REQUIRE_FALSE(contains("MultiLevelEncodingLoD"));
	const json permuto_config = {
		{"otype", "Permuto"}
	};
	const json lod_config = {
		{"otype", "MultiLevelEncodingLoD"}
	};
	REQUIRE_THROWS_AS(create_encoding<float>(5, permuto_config, 16), std::runtime_error);
	REQUIRE_THROWS_AS(create_encoding<float>(6, lod_config, 16), std::runtime_error);
}
#endif

TEMPLATE_TEST_CASE("Various invariance checks for input encodings", "[encoding][jit]", network_precision_t, float) {
	using T = TestType;

	tcnn_test_setup();

	for (const auto& encoding_name : builtin_encodings()) {
		if (equals_case_insensitive(encoding_name, "Permuto") || equals_case_insensitive(encoding_name, "MultiLevelEncodingLoD")) {
			continue;
		}

		SECTION(fmt::format("Testing {}", encoding_name)) {
			// Typical number of input dims is 3D (e.g. 3D space), but we need to special-case for some encodings that require more.
			const uint32_t n_dims = equals_case_insensitive(encoding_name, "NRC") || equals_case_insensitive(encoding_name, "OneBlobFrequency") ? 8 : 3;
			const uint32_t alignment = 16; // Common value due to tensor core width

			std::shared_ptr<Encoding<T>> encoding = default_encoding<T>(n_dims, encoding_name);
			encoding->set_alignment(alignment);
			test_differentiable_object<float, T, T>(encoding);
		}
	}
}

#if !defined(TCNN_NO_FWD_BWD)
TEMPLATE_TEST_CASE("Identity double backward", "[encoding][double-backward]", network_precision_t, float) {
	using T = TestType;

	tcnn_test_setup();

	const uint32_t n_dims = 3;
	const uint32_t batch_size = 256;
	const float scale = 2.0f;
	const json config = {
		{"otype", "Identity"},
		{"scale", scale},
		{"offset", 0.5f},
	};
	std::shared_ptr<Encoding<T>> encoding{create_encoding<T>(n_dims, config, 16)};
	const uint32_t n_outputs = encoding->padded_output_width();
	REQUIRE(n_outputs > n_dims);

	for (const auto layout : {CM, RM}) {
		CAPTURE(layout == CM);

		pcg32 rng{1337};
		GPUMatrix<float> input{n_dims, batch_size};
		GPUMatrix<float> dL_ddLdinput{n_dims, batch_size};
		GPUMatrixDynamic<T> dL_doutput{n_outputs, batch_size, layout};
		GPUMatrixDynamic<T> dL_ddLdoutput{n_outputs, batch_size, layout};
		GPUMatrix<float> dL_dinput{n_dims, batch_size};
		input.initialize_uniform(rng, -1.0f, 1.0f);
		dL_ddLdinput.initialize_uniform(rng, -1.0f, 1.0f);
		dL_doutput.initialize_uniform(rng, -1.0f, 1.0f);
		dL_ddLdoutput.initialize_uniform(rng, 1.0f, 2.0f);
		dL_dinput.initialize_uniform(rng, 1.0f, 2.0f);

		auto ctx = encoding->forward(input, nullptr, false, true);
		encoding->backward_backward_input(*ctx, input, dL_ddLdinput, dL_doutput, &dL_ddLdoutput, &dL_dinput);

		const auto second_order = dL_ddLdinput.to_cpu_vector();
		const auto upstream = dL_ddLdoutput.to_cpu_vector();
		for (uint32_t i = 0; i < batch_size; ++i) {
			for (uint32_t j = 0; j < n_outputs; ++j) {
				const float actual = (float)upstream[layout == CM ? i * n_outputs + j : j * batch_size + i];
				const float expected = j < n_dims ? scale * second_order[i * n_dims + j] : 0.0f;
				CAPTURE(i, j);
				REQUIRE(actual == Approx(expected).epsilon(1e-2).margin(1e-3));
			}
		}

		const auto input_gradient = dL_dinput.to_cpu_vector();
		REQUIRE(std::all_of(input_gradient.begin(), input_gradient.end(), [](float value) { return value == 0.0f; }));
	}
}
#endif
