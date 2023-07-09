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

/** @file   vec_pybind11.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  pybind11 bindings for NGP's vector and matrix types. Adapted from
 *          Patrik Huber's glm binding code per the BSD license of pybind11.
 */

#pragma once

#include <tiny-cuda-nn/vec.h>

#include <cstddef>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

namespace pybind11 {
namespace detail {

template <typename T, uint32_t N>
struct type_caster<tcnn::tvec<T, N>> {
	using vector_type = tcnn::tvec<T, N>;
	using Scalar = T;
	static constexpr std::size_t num_elements = N;

	bool load(handle src, bool) {
		array_t<Scalar> buf(src, true);
		if (!buf.check()) {
			return false;
		}

		if (buf.ndim() != 1) {
			return false; // not a rank-1 tensor (i.e. vector)
		}

		if (buf.shape(0) != num_elements) {
			return false; // not a 2-elements vector
		}

		for (size_t i = 0; i < num_elements; ++i) {
			value[i] = *buf.data(i);
		}

		return true;
	}

	static handle cast(const vector_type& src, return_value_policy, handle) {
		return array(
			num_elements,
			src.data()
		).release();
	}

	// Specifies the doc-string for the type in Python:
	PYBIND11_TYPE_CASTER(vector_type, _("vec"));
};

template <typename T, uint32_t N, uint32_t M>
struct type_caster<tcnn::tmat<T, N, M>> {
	using matrix_type = tcnn::tmat<T, N, M>;
	using Scalar = T;
	static constexpr std::size_t num_rows = M;
	static constexpr std::size_t num_cols = N;

	bool load(handle src, bool) {
		array_t<Scalar> buf(src, true);
		if (!buf.check()) {
			return false;
		}

		if (buf.ndim() != 2) {
			return false; // not a rank-2 tensor (i.e. matrix)
		}

		if (buf.shape(0) != num_rows || buf.shape(1) != num_cols) {
			return false; // not a 4x4 matrix
		}

		for (size_t i = 0; i < num_cols; ++i) {
			for (size_t j = 0; j < num_rows; ++j) {
				value[i][j] = *buf.data(j, i);
			}
		}

		return true;
	}

	static handle cast(const matrix_type& src, return_value_policy, handle) {
		return array(
			{ num_rows, num_cols },
			{ sizeof(Scalar), sizeof(Scalar) * num_rows }, // strides - flip the row/col layout!
			src.data()
		).release();
	}

	// Specifies the doc-string for the type in Python:
	PYBIND11_TYPE_CASTER(matrix_type, _("mat"));
};

}
}

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
