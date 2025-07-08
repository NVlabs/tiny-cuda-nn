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

/** @file   test_encodings.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  Test various invariances of input encodings. E.g. that `inference`
 *          and `inference_mixed_precision` produce the same results, that JIT
 *          produces the same result as no-JIT, as well as the same for derivatives.
 */

#include "test_common.h"

#include <tiny-cuda-nn/encoding.h>

using namespace tcnn;

TEMPLATE_TEST_CASE("Various invariance checks for input encodings", "[encoding][jit]", network_precision_t, float) {
	using T = TestType;

	tcnn_test_setup();

	for (const auto& encoding_name : builtin_encodings()) {
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
