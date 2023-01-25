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

/** @file   cutlass_matmul.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Matrix multiplication wrappers that call into CUTLASS (plus some custom modifications).
 *          The parameters are optimized to give optimal performance in a variety of situations.
 *          Parts of this file were adapted by starting from the CUTLASS sample code (see its BSD 3-clause license).
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>

#include <cutlass/cutlass.h>

#include <cutlass/array.h>
#include <cutlass/functional.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <iostream>
#include <map>
#include <type_traits>

TCNN_NAMESPACE_BEGIN

#define CUTLASS_CHECK_THROW(x)                                                                                        \
	do {                                                                                                                   \
		cutlass::Status error = x;                                                                                    \
		if (error != cutlass::Status::kSuccess)                                                                            \
			throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + cutlassGetStatusString(error)); \
	} while(0)

using SmArch = std::conditional_t<MIN_GPU_ARCH >= 80,
	std::conditional_t<std::is_same<network_precision_t, float>::value, cutlass::arch::Sm75, cutlass::arch::Sm80>,
	std::conditional_t<MIN_GPU_ARCH >= 75,
		cutlass::arch::Sm75,
		cutlass::arch::Sm70
	>
>;

using TypeAccumulator = std::conditional_t<std::is_same<network_precision_t, float>::value, float, cutlass::half_t>;
using TypeCompute = std::conditional_t<std::is_same<network_precision_t, float>::value, float, cutlass::half_t>;

template <typename T>
using MMAOp = typename std::conditional<
	std::is_same<T, float>::value || MIN_GPU_ARCH < 70,
	cutlass::arch::OpClassSimt,
	cutlass::arch::OpClassTensorOp
>::type;

template <typename T>
using ShapeMMAOp = typename std::conditional<
	std::is_same<MMAOp<T>, cutlass::arch::OpClassTensorOp>::value,
	typename std::conditional<
		std::is_same<SmArch, cutlass::arch::Sm80>::value || std::is_same<SmArch, cutlass::arch::Sm75>::value,
		cutlass::gemm::GemmShape<16, 8, 8>,
		cutlass::gemm::GemmShape<8, 8, 4>
	>::type,
	cutlass::gemm::GemmShape<1, 1, 1>
>::type;

template <typename thread_block, typename warp>
struct LayerConfig {
	using k_thread_block = thread_block;
	using k_warp = warp;
};

using FullLayerK = typename std::conditional<
	std::is_same<MMAOp<network_precision_t>, cutlass::arch::OpClassSimt>::value,
	LayerConfig<cutlass::gemm::GemmShape<128, 128, 8>, cutlass::gemm::GemmShape<32, 64, 8>>,
	LayerConfig<cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>>
>::type;
using LastLayerK = FullLayerK;

using FullLayer = typename std::conditional<
	std::is_same<MMAOp<network_precision_t>, cutlass::arch::OpClassSimt>::value,
	LayerConfig<cutlass::gemm::GemmShape<128, 128, 8>, cutlass::gemm::GemmShape<32, 64, 8>>,
	LayerConfig<cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>>
>::type;
using LastLayer = FullLayer;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// This code section describes the epilogue part of the kernel

template <typename V>
struct CutlassFragmentWrapper {
	static const uint32_t num_elements = V::kElements;
	V x;
};

template <
	typename ElementOutput_,                             ///< Data type used to load and store tensors
	int Count,                                           ///< Number of elements computed per operation
	typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
	typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
	cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest
>
class ActivationEpilogue {
public:
	using ElementOutput = ElementOutput_;
	using ElementAccumulator = ElementAccumulator_;
	using ElementCompute = ElementCompute_;

	static int const kCount = Count;

	using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
	using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
	using ComputeFragment = cutlass::Array<ElementCompute, kCount>;

	static cutlass::FloatRoundStyle const kRound = Round;

	struct Params {
		Activation activation;
		bool sum_source;
	};

public:
	CUTLASS_HOST_DEVICE
	ActivationEpilogue(Params const &params) : m_activation{params.activation}, m_sum_source{params.sum_source} { }

	CUTLASS_HOST_DEVICE
	bool is_source_needed() const {
		return m_sum_source;
	}

	/// Functionally required for serial reduction in the epilogue
	CUTLASS_HOST_DEVICE
	void set_k_partition(int k_partition, int k_partition_count) { }

	CUTLASS_HOST_DEVICE
	FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
		cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

		auto intermediate = CutlassFragmentWrapper<ComputeFragment>{accumulator_converter(accumulator)};
		intermediate = warp_activation<ElementCompute>(m_activation, intermediate);

		cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
		return destination_converter(intermediate.x);
	}

	CUTLASS_HOST_DEVICE
	FragmentOutput operator()(FragmentAccumulator const &accumulator, FragmentOutput const &source) const {
		cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
		cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

		cutlass::plus<ComputeFragment> plus_op;
		auto intermediate = CutlassFragmentWrapper<ComputeFragment>{accumulator_converter(accumulator)};
		if (m_sum_source) {
			intermediate.x = plus_op(intermediate.x, source_converter(source));
		}
		intermediate = warp_activation<ElementCompute>(m_activation, intermediate);

		cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
		return destination_converter(intermediate.x);
	}

private:
	Activation m_activation;
	bool m_sum_source;
};

template <
	typename ElementOutput_,                             ///< Data type used to load and store tensors
	int Count,                                           ///< Number of elements computed per operation
	typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
	typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
	cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest
>
class ActivationTransferEpilogue {
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
		Activation activation;
	};

public:
	/// Constructs the function object, possibly loading from pointers in host memory
	CUTLASS_HOST_DEVICE
	ActivationTransferEpilogue(Params const &params) : m_activation{params.activation} { }

	/// Returns true if source is needed
	CUTLASS_HOST_DEVICE
	bool is_source_needed() const {
		return true;
	}

	/// Functionally required for serial reduction in the epilogue
	CUTLASS_HOST_DEVICE
	void set_k_partition(int k_partition, int k_partition_count) { }

	CUTLASS_HOST_DEVICE
	FragmentOutput operator()(
		FragmentAccumulator const &accumulator,
		FragmentOutput const &source) const {

		cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
		cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

		auto converted_source = CutlassFragmentWrapper<ComputeFragment>{source_converter(source)};
		auto intermediate = CutlassFragmentWrapper<ComputeFragment>{accumulator_converter(accumulator)};

		intermediate = warp_activation_backward<ElementCompute>(m_activation, intermediate, converted_source);

		cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
		return destination_converter(intermediate.x);
	}

	CUTLASS_HOST_DEVICE
	FragmentOutput operator()(
		FragmentAccumulator const &accumulator) const {

		cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

		ComputeFragment converted_accumulator = accumulator_converter(accumulator);

		cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

		return destination_converter(converted_accumulator);
	}

private:
	Activation m_activation;
};


template <typename T>
static constexpr int n_vectorized_elements = std::is_same<MMAOp<T>, cutlass::arch::OpClassTensorOp>::value ? (128 / cutlass::sizeof_bits<T>::value) : 1;

template <typename T>
using SumOp = cutlass::epilogue::thread::LinearCombination<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

template <typename T>
using ActivationOp = ActivationEpilogue<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

template <typename T>
using ActivationTransferOp = ActivationTransferEpilogue<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;


template <typename EPILOGUE, typename LayerConfig, typename TypeA, typename LayoutA, typename TypeB, typename LayoutB, typename TypeOutput, typename LayoutOutput>
using OurGemm = cutlass::gemm::device::Gemm<
	TypeA,
	LayoutA,
	TypeB,
	LayoutB,
	TypeOutput,
	LayoutOutput,
	TypeAccumulator,
	MMAOp<TypeA>,
	SmArch,
	typename LayerConfig::k_thread_block,
	typename LayerConfig::k_warp,
	ShapeMMAOp<TypeA>,
	EPILOGUE,
	SwizzleThreadBlock,
	2
>;

template <typename EPILOGUE, typename LayerConfig, typename TypeA, typename LayoutA, typename TypeB, typename LayoutB, typename TypeOutput, typename LayoutOutput>
using SplitKGemm = cutlass::gemm::device::GemmSplitKParallel<
	TypeA,
	LayoutA,
	TypeB,
	LayoutB,
	TypeOutput,
	LayoutOutput,
	TypeAccumulator,
	MMAOp<TypeA>,
	SmArch,
	typename LayerConfig::k_thread_block,
	typename LayerConfig::k_warp,
	ShapeMMAOp<TypeA>,
	EPILOGUE
>;

template <class Gemm>
void fc_multiply_impl(cudaStream_t stream, const typename Gemm::Arguments& args) {
	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = Gemm::get_workspace_size(args);

	// Instantiate CUTLASS kernel depending on templates
	Gemm gemm_op;

	// Initialize CUTLASS kernel with arguments and workspace pointer
	auto workspace = allocate_workspace(stream, workspace_size);
	cutlass::Status status = gemm_op.initialize(args, workspace.data(), stream);
	CUTLASS_CHECK_THROW(status);

	// Launch initialized CUTLASS kernel
	status = gemm_op(stream);
	CUTLASS_CHECK_THROW(status);
}

template <class Gemm>
void fc_multiply_split_k_impl(cudaStream_t stream, const typename Gemm::Arguments& args) {
	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = Gemm::get_workspace_size(args);

	// Instantiate CUTLASS kernel depending on templates
	Gemm gemm_op;

	// Initialize CUTLASS kernel with arguments and workspace pointer
	auto workspace = allocate_workspace(stream, workspace_size);
	cutlass::Status status = gemm_op.initialize(args, workspace.data());
	CUTLASS_CHECK_THROW(status);

	// Launch initialized CUTLASS kernel
	status = gemm_op(stream);
	CUTLASS_CHECK_THROW(status);
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, MatrixLayout LayoutC, typename TypeD, MatrixLayout LayoutD>
void fc_multiply(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrix<TypeC, LayoutC>& C, const GPUMatrix<TypeD, LayoutD>& D, Activation act = Activation::None, bool transfer = false, bool sum_source = false) {
	using CutlassLayoutA = typename std::conditional<LayoutA == RM, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutB = typename std::conditional<LayoutB == RM, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutC = typename std::conditional<LayoutC == RM, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutD = typename std::conditional<LayoutD == RM, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;

	static_assert(std::is_same<TypeA, TypeB>::value, "Type of matrix A and B must be equal");
	static_assert(std::is_same<TypeC, TypeD>::value, "Type of matrix C and D must be equal");
	static_assert(std::is_same<CutlassLayoutC, CutlassLayoutD>::value, "Layout of matrix C and D must be equal");

	using MatmulTypeCompute = std::conditional_t<std::is_same<TypeA, float>::value, float, cutlass::half_t>;
	using MatmulTypeAccumulator = std::conditional_t<std::is_same<TypeC, float>::value, float, cutlass::half_t>;

	if (A.n() != B.m()) {
		throw std::runtime_error("Matrices A and B can not be multiplied together");
	}

	const int M = A.m();
	const int K = A.n();
	const int N = B.n();

	if (C.m() != M || C.n() != N) {
		throw std::runtime_error{fmt::format("Matrix C has incorrect size {}x{} != {}x{}", C.m(), C.n(), M, N)};
	}

	if (D.m() != M || D.n() != N) {
		throw std::runtime_error{fmt::format("Matrix D has incorrect size {}x{} != {}x{}", D.m(), D.n(), M, N)};
	}

	if (transfer) {
		using Gemm = OurGemm<ActivationTransferOp<MatmulTypeAccumulator>, config, MatmulTypeCompute, CutlassLayoutA, MatmulTypeCompute, CutlassLayoutB, MatmulTypeAccumulator, CutlassLayoutC>;
		typename Gemm::Arguments arguments{
			{M, N, K},
			{(MatmulTypeCompute*)A.data(), (int)A.stride()},
			{(MatmulTypeCompute*)B.data(), (int)B.stride()},
			{(MatmulTypeAccumulator*)C.data(), (int)C.stride()},
			{(MatmulTypeAccumulator*)D.data(), (int)D.stride()},
			{act},
			1
		};

		fc_multiply_impl<Gemm>(stream, arguments);
	} else {
		using Gemm = OurGemm<ActivationOp<MatmulTypeAccumulator>, config, MatmulTypeCompute, CutlassLayoutA, MatmulTypeCompute, CutlassLayoutB, MatmulTypeAccumulator, CutlassLayoutC>;
		typename Gemm::Arguments arguments{
			{M, N, K},
			{(MatmulTypeCompute*)A.data(), (int)A.stride()},
			{(MatmulTypeCompute*)B.data(), (int)B.stride()},
			{(MatmulTypeAccumulator*)C.data(), (int)C.stride()},
			{(MatmulTypeAccumulator*)D.data(), (int)D.stride()},
			{act, sum_source},
			1
		};

		fc_multiply_impl<Gemm>(stream, arguments);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, typename TypeD>
void fc_multiply(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrixDynamic<TypeC>& C, const GPUMatrixDynamic<TypeD>& D, Activation act = Activation::None, bool transfer = false, bool sum_source = false) {
	if (C.layout() != D.layout()) {
		throw std::runtime_error{"fc_multiply: Layout of GPUMatrixDynamic C and D must be equal"};
	}

	if (D.layout() == CM) {
		fc_multiply<config>(stream, A, B, C.cm(), D.cm(), act, transfer, sum_source);
	} else {
		fc_multiply<config>(stream, A, B, C.rm(), D.rm(), act, transfer, sum_source);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeC>& C, const GPUMatrixDynamic<TypeD>& D, Activation act = Activation::None, bool transfer = false, bool sum_source = false) {
	if (B.layout() == CM) {
		fc_multiply<config>(stream, A, B.cm(), C, D, act, transfer, sum_source);
	} else {
		fc_multiply<config>(stream, A, B.rm(), C, D, act, transfer, sum_source);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeD>
void fc_multiply(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeD>& D, Activation act = Activation::None) {
	fc_multiply<config>(stream, A, B, D, D, act);
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, MatrixLayout LayoutC, typename TypeD, MatrixLayout LayoutD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrix<TypeC, LayoutC>& C, const GPUMatrix<TypeD, LayoutD>& D, int split_k_slices = 1, float beta = 0.0f) {
	using CutlassLayoutA = typename std::conditional<LayoutA == RM, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutB = typename std::conditional<LayoutB == RM, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutC = typename std::conditional<LayoutC == RM, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutD = typename std::conditional<LayoutD == RM, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;

	using MatmulTypeCompute = std::conditional_t<std::is_same<TypeA, float>::value, float, cutlass::half_t>;
	using MatmulTypeAccumulator = std::conditional_t<std::is_same<TypeC, float>::value, float, cutlass::half_t>;

	static_assert(std::is_same<TypeA, TypeB>::value, "Type of matrix A and B must be equal");
	static_assert(std::is_same<TypeC, TypeD>::value, "Type of matrix C and D must be equal");
	static_assert(std::is_same<CutlassLayoutC, CutlassLayoutD>::value, "Layout of matrix C and D must be equal");

	if (A.n() != B.m()) {
		throw std::runtime_error("Matrices A and B can not be multiplied together");
	}

	const int M = A.m();
	const int K = A.n();
	const int N = B.n();

	if (C.m() != M || C.n() != N) {
		throw std::runtime_error{fmt::format("Matrix C has incorrect size {}x{} != {}x{}", C.m(), C.n(), M, N)};
	}

	if (D.m() != M || D.n() != N) {
		throw std::runtime_error{fmt::format("Matrix D has incorrect size {}x{} != {}x{}", D.m(), D.n(), M, N)};
	}

	using Gemm = SplitKGemm<SumOp<MatmulTypeAccumulator>, config, MatmulTypeCompute, CutlassLayoutA, MatmulTypeCompute, CutlassLayoutB, MatmulTypeAccumulator, CutlassLayoutC>;
	typename Gemm::Arguments arguments{
		{M, N, K},
		{(MatmulTypeCompute*)A.data(), (int)A.stride()},
		{(MatmulTypeCompute*)B.data(), (int)B.stride()},
		{(MatmulTypeAccumulator*)C.data(), (int)C.stride()},
		{(MatmulTypeAccumulator*)D.data(), (int)D.stride()},
		{(TypeCompute)1.0f, (TypeCompute)beta},
		split_k_slices
	};

	fc_multiply_split_k_impl<Gemm>(stream, arguments);
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, typename TypeD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrixDynamic<TypeC>& C, const GPUMatrixDynamic<TypeD>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (C.layout() != D.layout()) {
		throw std::runtime_error{"fc_multiply: Layout of GPUMatrixDynamic C and D must be equal"};
	}

	if (D.layout() == CM) {
		fc_multiply_split_k<config>(stream, A, B, C.cm(), D.cm(), split_k_slices, beta);
	} else {
		fc_multiply_split_k<config>(stream, A, B, C.rm(), D.rm(), split_k_slices, beta);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeC>& C, const GPUMatrixDynamic<TypeD>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (B.layout() == CM) {
		fc_multiply_split_k<config>(stream, A, B.cm(), C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k<config>(stream, A, B.rm(), C, D, split_k_slices, beta);
	}
}

template <typename config, typename TypeA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrixDynamic<TypeA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeC>& C, const GPUMatrixDynamic<TypeD>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (A.layout() == CM) {
		fc_multiply_split_k<config>(stream, A.cm(), B, C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k<config>(stream, A.rm(), B, C, D, split_k_slices, beta);
	}
}

template <typename config, typename TypeA, typename TypeB, typename TypeD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrixDynamic<TypeA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeD>& D, int split_k_slices, float beta) {
	fc_multiply_split_k<config>(stream, A, B, D, D, split_k_slices, beta);
}

TCNN_NAMESPACE_END
