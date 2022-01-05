/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

#define CUTLASS_CHECK(status)                                                                      \
{                                                                                                  \
	cutlass::Status error = status;                                                                \
	if (error != cutlass::Status::kSuccess) {                                                      \
		std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
		          << std::endl;                                                                    \
		exit(EXIT_FAILURE);                                                                        \
	}                                                                                              \
}

#define CUDA_CHECK(status)                                                \
{                                                                         \
	cudaError_t error = status;                                           \
	if (error != cudaSuccess) {                                           \
		std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
		          << " at line: " << __LINE__ << std::endl;               \
		exit(EXIT_FAILURE);                                               \
	}                                                                     \
}

#ifdef TCNN_AMPERE
using SmArch = typename std::conditional<std::is_same<network_precision_t, float>::value, cutlass::arch::Sm75, cutlass::arch::Sm80>::type;
#else
using SmArch = cutlass::arch::Sm75;
#endif

using TypeAccumulator = std::conditional_t<std::is_same<network_precision_t, float>::value, float, cutlass::half_t>;
using TypeCompute = std::conditional_t<std::is_same<network_precision_t, float>::value, float, cutlass::half_t>;

template <typename T>
using MMAOp = typename std::conditional<
	std::is_same<T, float>::value,
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

using FullLayerK = LayerConfig<cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>>;
using LastLayerK = LayerConfig<cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>>;

using FullLayer = typename std::conditional<
	std::is_same<TypeCompute, float>::value,
	LayerConfig<cutlass::gemm::GemmShape<128, 128, 8>, cutlass::gemm::GemmShape<32, 64, 8>>,
	LayerConfig<cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>>
>::type;

using FullLayerPreReLU = typename std::conditional<
	std::is_same<TypeCompute, float>::value,
	LayerConfig<cutlass::gemm::GemmShape<128, 128, 8, true>, cutlass::gemm::GemmShape<32, 64, 8, true>>,
	LayerConfig<cutlass::gemm::GemmShape<128, 128, 32, true>, cutlass::gemm::GemmShape<64, 64, 32, true>>
>::type;

using LastLayer = typename std::conditional<
	std::is_same<TypeCompute, float>::value,
	LayerConfig<cutlass::gemm::GemmShape<128, 128, 8>, cutlass::gemm::GemmShape<32, 64, 8>>,
	typename std::conditional<
		std::is_same<SmArch, cutlass::arch::Sm80>::value || std::is_same<SmArch, cutlass::arch::Sm75>::value,
		LayerConfig<cutlass::gemm::GemmShape<128, 32, 32>, cutlass::gemm::GemmShape<32, 32, 32>>,
		LayerConfig<cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>>
	>::type
>::type;

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
	void set_k_partition(int k_partition) { }

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
	void set_k_partition(int k_partition) { }

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
using IntermediateActivationOp = ActivationEpilogue<T, 4, TypeAccumulator, TypeCompute>;

template <typename T>
using IntermediateActivationTransferOp = ActivationTransferEpilogue<T, 4, TypeAccumulator, TypeCompute>;

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

inline std::map<cudaStream_t, GPUMemory<uint8_t>>& cutlass_workspaces() {
	static std::map<cudaStream_t, GPUMemory<uint8_t>> s_workspaces;
	return s_workspaces;
}

inline uint8_t* cutlass_get_workspace(size_t size, cudaStream_t stream) {
	GPUMemory<uint8_t>& workspace = cutlass_workspaces()[stream];
	if (size > workspace.size()) {
		size *= 2;
#ifdef TCNN_VERBOSE_MEMORY_ALLOCS
		std::cout << "CUTLASS GEMM: Allocating temporary workspace of " << bytes_to_string(size) << "." << std::endl;
#endif

		// Allocate twice the requested size to make sure we're not constantly allocating small increments.
		workspace.resize(size);
	}
	return workspace.data();
}

inline void cutlass_free_workspace(cudaStream_t stream) {
	if (cutlass_workspaces().count(stream) == 0) {
		return;
	}

#ifdef TCNN_VERBOSE_MEMORY_ALLOCS
	std::cout << "CUTLASS GEMM: Freeing temporary workspace of " << bytes_to_string(cutlass_workspaces().at(stream).size()) << "." << std::endl;
#endif
	cutlass_workspaces().erase(stream);
}


template <class Gemm>
void fc_multiply_impl(cudaStream_t stream, const typename Gemm::Arguments& args) {
	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = Gemm::get_workspace_size(args);

	// Instantiate CUTLASS kernel depending on templates
	Gemm gemm_op;

	// Initialize CUTLASS kernel with arguments and workspace pointer
	cutlass::Status status = gemm_op.initialize(args, cutlass_get_workspace(workspace_size, stream), stream);
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_op(stream);
	CUTLASS_CHECK(status);
}

template <class Gemm>
void fc_multiply_split_k_impl(cudaStream_t stream, const typename Gemm::Arguments& args) {
	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = Gemm::get_workspace_size(args);

	// Instantiate CUTLASS kernel depending on templates
	Gemm gemm_op;

	// Initialize CUTLASS kernel with arguments and workspace pointer
	cutlass::Status status = gemm_op.initialize(args, cutlass_get_workspace(workspace_size, stream));
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_op(stream);
	CUTLASS_CHECK(status);
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, MatrixLayout LayoutC, typename TypeD, MatrixLayout LayoutD>
void fc_multiply(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrix<TypeC, LayoutC>& C, GPUMatrix<TypeD, LayoutD>& D, Activation act = Activation::None, bool transfer = false, bool sum_source = false) {
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
		throw std::runtime_error(std::string("Matrix C has incorrect size ") + std::to_string(C.m()) + "," + std::to_string(C.n()) + "!=" + std::to_string(M) + "," + std::to_string(N));
	}

	if (D.m() != M || D.n() != N) {
		throw std::runtime_error(std::string("Matrix D has incorrect size ") + std::to_string(D.m()) + "," + std::to_string(D.n()) + "!=" + std::to_string(M) + "," + std::to_string(N));
	}

	const int lda = LayoutA == RM ? A.n() : A.m();
	const int ldb = LayoutB == RM ? B.n() : B.m();
	const int ldc = LayoutC == RM ? C.n() : C.m();
	const int ldd = LayoutD == RM ? D.n() : D.m();

	if (transfer) {
		using Gemm = OurGemm<ActivationTransferOp<MatmulTypeAccumulator>, config, MatmulTypeCompute, CutlassLayoutA, MatmulTypeCompute, CutlassLayoutB, MatmulTypeAccumulator, CutlassLayoutC>;
		typename Gemm::Arguments arguments{
			{M, N, K},
			{(MatmulTypeCompute*)A.data(), lda},
			{(MatmulTypeCompute*)B.data(), ldb},
			{(MatmulTypeAccumulator*)C.data(), ldc},
			{(MatmulTypeAccumulator*)D.data(), ldd},
			{act},
			1
		};

		fc_multiply_impl<Gemm>(stream, arguments);
	} else {
		using Gemm = OurGemm<ActivationOp<MatmulTypeAccumulator>, config, MatmulTypeCompute, CutlassLayoutA, MatmulTypeCompute, CutlassLayoutB, MatmulTypeAccumulator, CutlassLayoutC>;
		typename Gemm::Arguments arguments{
			{M, N, K},
			{(MatmulTypeCompute*)A.data(), lda},
			{(MatmulTypeCompute*)B.data(), ldb},
			{(MatmulTypeAccumulator*)C.data(), ldc},
			{(MatmulTypeAccumulator*)D.data(), ldd},
			{act, sum_source},
			1
		};

		fc_multiply_impl<Gemm>(stream, arguments);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, typename TypeD>
void fc_multiply(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrixDynamic<TypeC>& C, GPUMatrixDynamic<TypeD>& D, Activation act = Activation::None, bool transfer = false, bool sum_source = false) {
	if (C.layout() != D.layout()) {
		throw std::runtime_error{"fc_multiply: Layout of GPUMatrixDynamic C and D must be equal"};
	}

	if (D.layout() == CM) {
		auto C_CM = GPUMatrix<TypeC, CM>{C};
		auto D_CM = GPUMatrix<TypeD, CM>{D};
		fc_multiply<config>(stream, A, B, C_CM, D_CM, act, transfer, sum_source);
	} else {
		auto C_RM = GPUMatrix<TypeC, RM>{C};
		auto D_RM = GPUMatrix<TypeD, RM>{D};
		fc_multiply<config>(stream, A, B, C_RM, D_RM, act, transfer, sum_source);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeC>& C, GPUMatrixDynamic<TypeD>& D, Activation act = Activation::None, bool transfer = false, bool sum_source = false) {
	if (B.layout() == CM) {
		auto B_CM = GPUMatrix<TypeB, CM>{B};
		fc_multiply<config>(stream, A, B_CM, C, D, act, transfer, sum_source);
	} else {
		auto B_RM = GPUMatrix<TypeB, RM>{B};
		// Only column-major output is supported by CUTLASS, then B is row-major.
		// The following constructors will throw if that assumption isn't met.
		auto C_CM = GPUMatrix<TypeC, CM>{C};
		auto D_CM = GPUMatrix<TypeD, CM>{D};
		fc_multiply<config>(stream, A, B_RM, C_CM, D_CM, act, transfer, sum_source);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeD>
void fc_multiply(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrixDynamic<TypeB>& B, GPUMatrixDynamic<TypeD>& D, Activation act = Activation::None) {
	fc_multiply<config>(stream, A, B, D, D, act);
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, MatrixLayout LayoutC, typename TypeD, MatrixLayout LayoutD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrix<TypeC, LayoutC>& C, GPUMatrix<TypeD, LayoutD>& D, int split_k_slices = 1) {
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
		throw std::runtime_error(std::string("Matrix C has incorrect size ") + std::to_string(C.m()) + "," + std::to_string(C.n()) + "!=" + std::to_string(M) + "," + std::to_string(N));
	}

	if (D.m() != M || D.n() != N) {
		throw std::runtime_error(std::string("Matrix D has incorrect size ") + std::to_string(D.m()) + "," + std::to_string(D.n()) + "!=" + std::to_string(M) + "," + std::to_string(N));
	}

	const int lda = LayoutA == RM ? A.n() : A.m();
	const int ldb = LayoutB == RM ? B.n() : B.m();
	const int ldc = LayoutC == RM ? C.n() : C.m();
	const int ldd = LayoutD == RM ? D.n() : D.m();

	using Gemm = SplitKGemm<SumOp<MatmulTypeAccumulator>, config, MatmulTypeCompute, CutlassLayoutA, MatmulTypeCompute, CutlassLayoutB, MatmulTypeAccumulator, CutlassLayoutC>;
	typename Gemm::Arguments arguments{
		{M, N, K},
		{(MatmulTypeCompute*)A.data(), lda},
		{(MatmulTypeCompute*)B.data(), ldb},
		{(MatmulTypeAccumulator*)C.data(), ldc},
		{(MatmulTypeAccumulator*)D.data(), ldd},
		{(TypeCompute)1.0f, (TypeCompute)0.0f},
		split_k_slices
	};

	fc_multiply_split_k_impl<Gemm>(stream, arguments);
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, typename TypeD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrixDynamic<TypeC>& C, GPUMatrixDynamic<TypeD>& D, int split_k_slices = 1) {
	if (C.layout() != D.layout()) {
		throw std::runtime_error{"fc_multiply: Layout of GPUMatrixDynamic C and D must be equal"};
	}

	if (D.layout() == CM) {
		auto C_CM = GPUMatrix<TypeC, CM>{C};
		auto D_CM = GPUMatrix<TypeD, CM>{D};
		fc_multiply_split_k<config>(stream, A, B, C_CM, D_CM, split_k_slices);
	} else {
		auto C_RM = GPUMatrix<TypeC, RM>{C};
		auto D_RM = GPUMatrix<TypeD, RM>{D};
		fc_multiply_split_k<config>(stream, A, B, C_RM, D_RM, split_k_slices);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeC>& C, GPUMatrixDynamic<TypeD>& D, int split_k_slices = 1) {
	if (B.layout() == CM) {
		auto B_CM = GPUMatrix<TypeB, CM>{B};
		fc_multiply_split_k<config>(stream, A, B_CM, C, D, split_k_slices);
	} else {
		auto B_RM = GPUMatrix<TypeB, RM>{B};
		fc_multiply_split_k<config>(stream, A, B_RM, C, D, split_k_slices);
	}
}

template <typename config, typename TypeA, typename TypeB, MatrixLayout LayoutB, typename TypeC, typename TypeD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrixDynamic<TypeA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrixDynamic<TypeC>& C, GPUMatrixDynamic<TypeD>& D, int split_k_slices = 1) {
	if (A.layout() == CM) {
		auto A_CM = GPUMatrix<TypeA, CM>{A};
		fc_multiply_split_k<config>(stream, A_CM, B, C, D, split_k_slices);
	} else {
		auto A_RM = GPUMatrix<TypeA, RM>{A};
		fc_multiply_split_k<config>(stream, A_RM, B, C, D, split_k_slices);
	}
}

template <typename config, typename TypeA, typename TypeB, MatrixLayout LayoutB, typename TypeD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrixDynamic<TypeA>& A, const GPUMatrix<TypeB, LayoutB>& B, GPUMatrixDynamic<TypeD>& D, int split_k_slices) {
	fc_multiply_split_k<config>(stream, A, B, D, D, split_k_slices);
}

TCNN_NAMESPACE_END
