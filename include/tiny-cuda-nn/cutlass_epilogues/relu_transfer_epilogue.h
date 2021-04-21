/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

/** @file   relu_transfer_epilogue.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  ReLU transfer epilogue for use with CUTLASS.
 *          Parts of this file were adapted by starting from the CUTLASS sample code (see its BSD 3-clause license).
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/array.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>


TCNN_NAMESPACE_BEGIN

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct relu_transfer_op {
	CUTLASS_HOST_DEVICE
	T operator()(T lhs, T rhs) const {
		return ((float)rhs <= 0) ? (T)0 : lhs;
	}
};

template <typename T, int N>
struct relu_transfer_op<cutlass::Array<T, N>> {

	CUTLASS_HOST_DEVICE
	cutlass::Array<T, N> operator()(cutlass::Array<T, N> const &lhs, cutlass::Array<T, N> const &rhs) const {

	cutlass::Array<T, N> result;
	relu_transfer_op<T> scalar_op;

	CUTLASS_PRAGMA_UNROLL
	for (int i = 0; i < N; ++i) {
	  result[i] = scalar_op(lhs[i], rhs[i]);
	}

	return result;
  }
};

template <
	typename ElementOutput_,                             ///< Data type used to load and store tensors
	int Count,                                           ///< Number of elements computed per operation
	typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
	typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
	cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest
>
class ReLUTransferEpilogue {
public:

	using ElementOutput = ElementOutput_;
	using ElementAccumulator = ElementAccumulator_;
	using ElementCompute = ElementCompute_;

	static int const kCount = Count;

	using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
	using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
	using ComputeFragment = cutlass::Array<ElementCompute, kCount>;

	static cutlass::FloatRoundStyle const kRound = Round;

	/// Host-constructable parameters structure
	struct Params {
	CUTLASS_HOST_DEVICE
	Params() { }
	};

public:

	/// Constructs the function object, possibly loading from pointers in host memory
	CUTLASS_HOST_DEVICE
	ReLUTransferEpilogue(Params const &params) {
	}

	/// Returns true if source is needed
	CUTLASS_HOST_DEVICE
	bool is_source_needed() const {
	return true;
	}

	/// Functionally required for serial reduction in the epilogue
	CUTLASS_HOST_DEVICE
	void set_k_partition(int k_partition) {
	}

	CUTLASS_HOST_DEVICE
	FragmentOutput operator()(
		FragmentAccumulator const &accumulator,
		FragmentOutput const &source) const {

		// Convert source to interal compute numeric type
		cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
		cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

		ComputeFragment converted_source = source_converter(source);
		ComputeFragment converted_accumulator = accumulator_converter(accumulator);

		// Convert to destination numeric type
		relu_transfer_op<ComputeFragment> trans;
		cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

		return destination_converter(trans(converted_accumulator, converted_source));
	}

	CUTLASS_HOST_DEVICE
	FragmentOutput operator()(
		FragmentAccumulator const &accumulator) const {

		// Convert source to interal compute numeric type
		cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

		ComputeFragment converted_accumulator = accumulator_converter(accumulator);

		// Convert to destination numeric type
		cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

		return destination_converter(converted_accumulator);
	}
};

TCNN_NAMESPACE_END
