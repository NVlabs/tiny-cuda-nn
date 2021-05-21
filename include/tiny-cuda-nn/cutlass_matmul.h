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

/** @file   cutlass_matmul.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Matrix multiplication wrappers that call into CUTLASS (plus some custom modifications).
 *          The parameters are optimized to give optimal performance in a variety of situations.
 *          Parts of this file were adapted by starting from the CUTLASS sample code (see its BSD 3-clause license).
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/cutlass_matmul_interface.h>

#include <tiny-cuda-nn/cutlass_epilogues/exponential_epilogue.h>
#include <tiny-cuda-nn/cutlass_epilogues/relu_transfer_epilogue.h>
#include <tiny-cuda-nn/cutlass_epilogues/sine_epilogue.h>

#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/misc_kernels.h>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>

#include <cutlass_b2b/device/b2b_gemm.h>

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

using SmArch = typename std::conditional<std::is_same<network_precision_t, float>::value, cutlass::arch::Sm75, cutlass::arch::Sm80>::type;

using TypeAccumulator = network_precision_t;
using TypeCompute = network_precision_t;

template <typename T>
using MMAOp = typename std::conditional<
	std::is_same<T, cutlass::half_t>::value,
	cutlass::arch::OpClassTensorOp,
	cutlass::arch::OpClassSimt
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
using FullLayerKPreReLU = LayerConfig<cutlass::gemm::GemmShape<64, 64, 32, true>, cutlass::gemm::GemmShape<32, 32, 32, true>>;
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

using FullLayerB2bPreReLU64 = LayerConfig<cutlass::gemm::GemmShape<128, 64, 32, true>, cutlass::gemm::GemmShape<32, 64, 32, true>>;
using FullLayerB2bPreReLU128 = LayerConfig<cutlass::gemm::GemmShape<128, 128, 32, true>, cutlass::gemm::GemmShape<32, 128, 32, true>>;
using FullLayerB2bPreReLU256 = LayerConfig<cutlass::gemm::GemmShape<128, 256, 16, true>, cutlass::gemm::GemmShape<32, 256, 16, true>>;

using FullLayerB2b64 = LayerConfig<cutlass::gemm::GemmShape<128, 64, 32>, cutlass::gemm::GemmShape<32, 64, 32>>;
using FullLayerB2b128 = LayerConfig<cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<32, 128, 32>>;
using FullLayerB2b256 = LayerConfig<cutlass::gemm::GemmShape<128, 256, 16>, cutlass::gemm::GemmShape<32, 256, 16>>;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// This code section describes the epilogue part of the kernel

template <typename T>
static constexpr int n_vectorized_elements = std::is_same<MMAOp<T>, cutlass::arch::OpClassTensorOp>::value ? (128 / cutlass::sizeof_bits<T>::value) : 1;

template <typename T>
using IntermediateOp = cutlass::epilogue::thread::LinearCombinationRelu<T, 4, TypeAccumulator, TypeCompute>;

template <typename T>
using SumOp = cutlass::epilogue::thread::LinearCombination<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

template <typename T>
using ReLUOp = cutlass::epilogue::thread::LinearCombinationRelu<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

template <typename T>
using ReLUIntermediateOp = cutlass::epilogue::thread::LinearCombinationRelu<T, 4, TypeAccumulator, TypeCompute>;

template <typename T>
using ReLUTransferOp = ReLUTransferEpilogue<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

template <typename T>
using ExponentialOp = ExponentialEpilogue<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

template <typename T>
using ExponentialIntermediateOp = ExponentialEpilogue<T, 4, TypeAccumulator, TypeCompute>;

template <typename T>
using SineOp = SineEpilogue<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

template <typename T>
using SineIntermediateOp = SineEpilogue<T, 4, TypeAccumulator, TypeCompute>;

template <typename T>
using ConversionOp = cutlass::epilogue::thread::Convert<T, n_vectorized_elements<T>, TypeAccumulator>;


// Number of pipelines you want to use
constexpr int NumStages = 2;

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
	NumStages
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

template <typename Epilogue1, typename Epilogue2, typename LayerConfig1, typename LayerConfig2, typename TypeA, typename LayoutA, typename TypeB, typename LayoutB, typename TypeOutput, typename LayoutOutput>
using OurB2bGemm = cutlass::gemm::device::B2bGemm<
	TypeA,
	LayoutA,
	TypeB,
	LayoutB,
	TypeOutput,
	LayoutOutput,
	TypeAccumulator,
	MMAOp<TypeA>,
	SmArch,
	typename LayerConfig1::k_thread_block,
	typename LayerConfig2::k_thread_block,
	typename LayerConfig1::k_warp,
	typename LayerConfig2::k_warp,
	ShapeMMAOp<TypeA>,
	Epilogue1,
	Epilogue2,
	cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
	NumStages
>;


static std::map<cudaStream_t, GPUMemory<uint8_t>> workspaces;

inline uint8_t* get_workspace(size_t size, cudaStream_t stream) {
	GPUMemory<uint8_t>& workspace = workspaces[stream];
	if (size > workspace.get_num_elements()) {
		size *= 2;
		std::cout << "CUTLASS GEMM: Allocating temporary workspace of " << size << " bytes." << std::endl;

		// Allocate twice the requested size to make sure we're not constantly allocating small increments.
		workspace.resize(size);
	}
	return workspace.data();
}

inline void free_workspace(cudaStream_t stream) {
	if (workspaces.count(stream) == 0) {
		return;
	}

	std::cout << "CUTLASS GEMM: Freeing temporary workspace of " << workspaces.at(stream).get_num_elements() << " bytes." << std::endl;
	workspaces.erase(stream);
}


template <class Gemm>
void fc_multiply_impl(cudaStream_t stream, const typename Gemm::Arguments& args) {
	// Using the arguments, query for extra workspace required for matrix multiplication computation
	size_t workspace_size = Gemm::get_workspace_size(args);

	// Instantiate CUTLASS kernel depending on templates
	Gemm gemm_op;

	// Initialize CUTLASS kernel with arguments and workspace pointer
	cutlass::Status status = gemm_op.initialize(args, get_workspace(workspace_size, stream), stream);
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
	cutlass::Status status = gemm_op.initialize(args, get_workspace(workspace_size, stream));
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_op(stream);
	CUTLASS_CHECK(status);
}

template <class Gemm>
void fc_multiply_b2b_impl(cudaStream_t stream, const typename Gemm::Arguments& args) {
	// Instantiate CUTLASS kernel depending on templates
	Gemm gemm_op;

	// Initialize CUTLASS kernel with arguments and workspace pointer
	cutlass::Status status = gemm_op.initialize(args, nullptr, stream);
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_op(stream);
	CUTLASS_CHECK(status);
}

template <Activation activation, typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, MatrixLayout LayoutC, typename TypeD, MatrixLayout LayoutD>
void fc_multiply(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrix<TypeC, LayoutC>& C, GPUMatrix<TypeD, LayoutD>& D, TypeCompute beta, TypeCompute alpha) {
	using CutlassLayoutA = typename std::conditional<LayoutA == MatrixLayout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutB = typename std::conditional<LayoutB == MatrixLayout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutC = typename std::conditional<LayoutC == MatrixLayout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutD = typename std::conditional<LayoutD == MatrixLayout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;

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

	const int lda = LayoutA == MatrixLayout::RowMajor ? A.n() : A.m();
	const int ldb = LayoutB == MatrixLayout::RowMajor ? B.n() : B.m();
	const int ldc = LayoutC == MatrixLayout::RowMajor ? C.n() : C.m();
	const int ldd = LayoutD == MatrixLayout::RowMajor ? D.n() : D.m();

	if (activation == Activation::None) {
		using Gemm = OurGemm<SumOp<TypeC>, config, TypeA, CutlassLayoutA, TypeB, CutlassLayoutB, TypeC, CutlassLayoutC>;
		typename Gemm::Arguments arguments{
			{M, N, K},
			{A.data(), lda},
			{B.data(), ldb},
			{C.data(), ldc},
			{D.data(), ldd},
			{alpha, beta},
			1
		};

		fc_multiply_impl<Gemm>(stream, arguments);
	} else if (activation == Activation::ReLU) {
		using Gemm = OurGemm<ReLUOp<TypeC>, config, TypeA, CutlassLayoutA, TypeB, CutlassLayoutB, TypeC, CutlassLayoutC>;
		typename Gemm::Arguments arguments{
			{M, N, K},
			{A.data(), lda},
			{B.data(), ldb},
			{C.data(), ldc},
			{D.data(), ldd},
			{alpha, beta, (TypeCompute)0.0f},
			1
		};

		fc_multiply_impl<Gemm>(stream, arguments);
	} else if (activation == Activation::Exponential) {
		using Gemm = OurGemm<ExponentialOp<TypeC>, config, TypeA, CutlassLayoutA, TypeB, CutlassLayoutB, TypeC, CutlassLayoutC>;
		typename Gemm::Arguments arguments{
			{M, N, K},
			{A.data(), lda},
			{B.data(), ldb},
			{C.data(), ldc},
			{D.data(), ldd},
			{beta},
			1
		};

		fc_multiply_impl<Gemm>(stream, arguments);
	} else if (activation == Activation::Sine) {
		using Gemm = OurGemm<SineOp<TypeC>, config, TypeA, CutlassLayoutA, TypeB, CutlassLayoutB, TypeC, CutlassLayoutC>;
		typename Gemm::Arguments arguments{
			{M, N, K},
			{A.data(), lda},
			{B.data(), ldb},
			{C.data(), ldc},
			{D.data(), ldd},
			typename SineOp<TypeC>::Params(),
			1
		};

		fc_multiply_impl<Gemm>(stream, arguments);
	} else if (activation == Activation::ReLUTransfer) {
		using Gemm = OurGemm<ReLUTransferOp<TypeC>, config, TypeA, CutlassLayoutA, TypeB, CutlassLayoutB, TypeC, CutlassLayoutC>;
		typename Gemm::Arguments arguments{
			{M, N, K},
			{A.data(), lda},
			{B.data(), ldb},
			{C.data(), ldc},
			{D.data(), ldd},
			typename ReLUTransferOp<TypeC>::Params(),
			1
		};

		fc_multiply_impl<Gemm>(stream, arguments);
	} else {
        throw std::runtime_error{
            "Unsupported activation type in fc_multiply()"
        };
    }
}

template <Activation activation, typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeD, MatrixLayout LayoutD>
void fc_multiply(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, GPUMatrix<TypeD, LayoutD>& D, TypeCompute beta = (TypeCompute)0) {
	fc_multiply<activation, config>(stream, A, B, D, D, beta);
}

template <Activation activation, typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, MatrixLayout LayoutC, typename TypeD, MatrixLayout LayoutD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrix<TypeC, LayoutC>& C, GPUMatrix<TypeD, LayoutD>& D, TypeCompute beta = (TypeCompute)0, int split_k_slices = 1, TypeCompute alpha = (TypeCompute)1) {
	using CutlassLayoutA = typename std::conditional<LayoutA == MatrixLayout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutB = typename std::conditional<LayoutB == MatrixLayout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutC = typename std::conditional<LayoutC == MatrixLayout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutD = typename std::conditional<LayoutD == MatrixLayout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;

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

	const int lda = LayoutA == MatrixLayout::RowMajor ? A.n() : A.m();
	const int ldb = LayoutB == MatrixLayout::RowMajor ? B.n() : B.m();
	const int ldc = LayoutC == MatrixLayout::RowMajor ? C.n() : C.m();
	const int ldd = LayoutD == MatrixLayout::RowMajor ? D.n() : D.m();

	if (activation == Activation::None) {
		using Gemm = SplitKGemm<SumOp<TypeC>, config, TypeA, CutlassLayoutA, TypeB, CutlassLayoutB, TypeC, CutlassLayoutC>;
		typename Gemm::Arguments arguments{
			{M, N, K},
			{A.data(), lda},
			{B.data(), ldb},
			{C.data(), ldc},
			{D.data(), ldd},
			{alpha, beta},
			split_k_slices
		};

		fc_multiply_split_k_impl<Gemm>(stream, arguments);
	} else if (activation == Activation::ReLU) {
		using Gemm = SplitKGemm<ReLUOp<TypeC>, config, TypeA, CutlassLayoutA, TypeB, CutlassLayoutB, TypeC, CutlassLayoutC>;
		typename Gemm::Arguments arguments{
			{M, N, K},
			{A.data(), lda},
			{B.data(), ldb},
			{C.data(), ldc},
			{D.data(), ldd},
			{alpha, beta, (TypeCompute)0.0f},
			split_k_slices
		};

		fc_multiply_split_k_impl<Gemm>(stream, arguments);
	} else if (activation == Activation::Exponential) {
		using Gemm = SplitKGemm<ExponentialOp<TypeC>, config, TypeA, CutlassLayoutA, TypeB, CutlassLayoutB, TypeC, CutlassLayoutC>;
		typename Gemm::Arguments arguments{
			{M, N, K},
			{A.data(), lda},
			{B.data(), ldb},
			{C.data(), ldc},
			{D.data(), ldd},
			{beta},
			split_k_slices
		};

		fc_multiply_split_k_impl<Gemm>(stream, arguments);
	} else if (activation == Activation::Sine) {
		using Gemm = SplitKGemm<SineOp<TypeC>, config, TypeA, CutlassLayoutA, TypeB, CutlassLayoutB, TypeC, CutlassLayoutC>;
		typename Gemm::Arguments arguments{
			{M, N, K},
			{A.data(), lda},
			{B.data(), ldb},
			{C.data(), ldc},
			{D.data(), ldd},
			typename SineOp<TypeC>::Params(),
			split_k_slices
		};

		fc_multiply_split_k_impl<Gemm>(stream, arguments);
	} else if (activation == Activation::ReLUTransfer) {
		using Gemm = SplitKGemm<ReLUTransferOp<TypeC>, config, TypeA, CutlassLayoutA, TypeB, CutlassLayoutB, TypeC, CutlassLayoutC>;
		typename Gemm::Arguments arguments{
			{M, N, K},
			{A.data(), lda},
			{B.data(), ldb},
			{C.data(), ldc},
			{D.data(), ldd},
			typename ReLUTransferOp<TypeC>::Params(),
			split_k_slices
		};

		fc_multiply_split_k_impl<Gemm>(stream, arguments);
	} else {
        throw std::runtime_error{
            "Unsupported activation type in fc_multiply_split_k()"
        };
    }
}

template <Activation activation, typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeD, MatrixLayout LayoutD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, GPUMatrix<TypeD, LayoutD>& D, int split_k_slices, TypeCompute alpha = (TypeCompute)1, TypeCompute beta = (TypeCompute)0) {
	fc_multiply_split_k<activation, config>(stream, A, B, D, D, beta, split_k_slices, alpha);
}

template <Activation activation, typename config1, typename config2, typename TypeA, MatrixLayout LayoutA, typename TypeB1, MatrixLayout LayoutB1, typename TypeC1, MatrixLayout LayoutC1, typename TypeB2, MatrixLayout LayoutB2, typename TypeC2, MatrixLayout LayoutC2, typename TypeD, MatrixLayout LayoutD>
void fc_multiply_b2b(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB1, LayoutB1>& B1, const GPUMatrix<TypeC1, LayoutC1>& C1, const GPUMatrix<TypeB2, LayoutB2>& B2, const GPUMatrix<TypeC2, LayoutC2>& C2, GPUMatrix<TypeD, LayoutD>& D, TypeCompute beta1 = (TypeCompute)0, TypeCompute beta2 = (TypeCompute)0) {
	using CutlassLayoutA = typename std::conditional<LayoutA == MatrixLayout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutB1 = typename std::conditional<LayoutB1 == MatrixLayout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutC1 = typename std::conditional<LayoutC1 == MatrixLayout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutB2 = typename std::conditional<LayoutB2 == MatrixLayout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutC2 = typename std::conditional<LayoutC2 == MatrixLayout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutD = typename std::conditional<LayoutD == MatrixLayout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;

	static_assert(std::is_same<TypeA, TypeB1>::value, "Type of matrix A and B1 must be equal");
	static_assert(std::is_same<TypeB1, TypeC1>::value, "Type of matrix B1 and C2 must be equal");
	static_assert(std::is_same<TypeC2, TypeD>::value, "Type of matrix C2 and D must be equal");
	static_assert(std::is_same<CutlassLayoutC2, CutlassLayoutD>::value, "Layout of matrix C and D must be equal");

	if (A.n() != B1.m()) {
		throw std::runtime_error("Matrices A and B can not be multiplied together");
	}

	const int M1 = A.m();
	const int K1 = A.n();
	const int N1 = B1.n();

	if (C1.m() != M1 || C1.n() != N1) {
		throw std::runtime_error(std::string("Matrix C has incorrect size ") + std::to_string(C1.m()) + "," + std::to_string(C1.n()) + "!=" + std::to_string(M1) + "," + std::to_string(N1));
	}

	const int M2 = C1.m();
	const int K2 = C1.n();
	const int N2 = B2.n();

	if (C2.m() != M2 || C2.n() != N2) {
		throw std::runtime_error(std::string("Matrix C has incorrect size ") + std::to_string(C2.m()) + "," + std::to_string(C2.n()) + "!=" + std::to_string(M2) + "," + std::to_string(N2));
	}

	if (D.m() != M2 || D.n() != N2) {
		throw std::runtime_error(std::string("Matrix D has incorrect size ") + std::to_string(D.m()) + "," + std::to_string(D.n()) + "!=" + std::to_string(M2) + "," + std::to_string(N2));
	}

	const int lda = LayoutA == MatrixLayout::RowMajor ? A.n() : A.m();
	const int ldb1 = LayoutB1 == MatrixLayout::RowMajor ? B1.n() : B1.m();
	const int ldc1 = LayoutC1 == MatrixLayout::RowMajor ? C1.n() : C1.m();
	const int ldb2 = LayoutB2 == MatrixLayout::RowMajor ? B2.n() : B2.m();
	const int ldc2 = LayoutC2 == MatrixLayout::RowMajor ? C2.n() : C2.m();
	const int ldd = LayoutD == MatrixLayout::RowMajor ? D.n() : D.m();

	if (activation == Activation::None) {
		using Gemm = OurB2bGemm<IntermediateOp<TypeC1>, SumOp<TypeC2>, config1, config2, TypeA, CutlassLayoutA, TypeB1, CutlassLayoutB1, TypeC1, CutlassLayoutC1>;
		typename Gemm::Arguments arguments{
			{M1, N1, K1},
			{M2, N2, K2},
			{A.data(), lda},
			{B1.data(), ldb1},
			{C1.data(), ldc1},
			{B2.data(), ldb2},
			{C2.data(), ldc2},
			{D.data(), ldd},
			{(TypeCompute)1.0f, beta1},
			{(TypeCompute)1.0f, beta2},
		};

		fc_multiply_b2b_impl<Gemm>(stream, arguments);
	} else if (activation == Activation::ReLU) {
		using Gemm = OurB2bGemm<ReLUIntermediateOp<TypeC1>, ReLUOp<TypeC2>, config1, config2, TypeA, CutlassLayoutA, TypeB1, CutlassLayoutB1, TypeC1, CutlassLayoutC1>;
		typename Gemm::Arguments arguments{
			{M1, N1, K1},
			{M2, N2, K2},
			{A.data(), lda},
			{B1.data(), ldb1},
			{C1.data(), ldc1},
			{B2.data(), ldb2},
			{C2.data(), ldc2},
			{D.data(), ldd},
			{(TypeCompute)1.0f, beta1, (TypeCompute)0.0f},
			{(TypeCompute)1.0f, beta2, (TypeCompute)0.0f},
		};

		fc_multiply_b2b_impl<Gemm>(stream, arguments);
	} else if (activation == Activation::Exponential) {
		using Gemm = OurB2bGemm<ExponentialIntermediateOp<TypeC1>, ExponentialOp<TypeC2>, config1, config2, TypeA, CutlassLayoutA, TypeB1, CutlassLayoutB1, TypeC1, CutlassLayoutC1>;
		typename Gemm::Arguments arguments{
			{M1, N1, K1},
			{M2, N2, K2},
			{A.data(), lda},
			{B1.data(), ldb1},
			{C1.data(), ldc1},
			{B2.data(), ldb2},
			{C2.data(), ldc2},
			{D.data(), ldd},
			{beta1},
			{beta2},
		};

		fc_multiply_b2b_impl<Gemm>(stream, arguments);
	} else if (activation == Activation::Sine) {
		using Gemm = OurB2bGemm<SineIntermediateOp<TypeC1>, SineOp<TypeC2>, config1, config2, TypeA, CutlassLayoutA, TypeB1, CutlassLayoutB1, TypeC1, CutlassLayoutC1>;
		typename Gemm::Arguments arguments{
			{M1, N1, K1},
			{M2, N2, K2},
			{A.data(), lda},
			{B1.data(), ldb1},
			{C1.data(), ldc1},
			{B2.data(), ldb2},
			{C2.data(), ldc2},
			{D.data(), ldd},
			typename SineIntermediateOp<TypeC1>::Params(),
			typename SineOp<TypeC2>::Params(),
		};

		fc_multiply_b2b_impl<Gemm>(stream, arguments);
	} else if (activation == Activation::ReLUTransfer) {
		using Gemm = OurB2bGemm<IntermediateOp<TypeC1>, ReLUTransferOp<TypeC2>, config1, config2, TypeA, CutlassLayoutA, TypeB1, CutlassLayoutB1, TypeC1, CutlassLayoutC1>;
		typename Gemm::Arguments arguments{
			{M1, N1, K1},
			{M2, N2, K2},
			{A.data(), lda},
			{B1.data(), ldb1},
			{C1.data(), ldc1},
			{B2.data(), ldb2},
			{C2.data(), ldc2},
			{D.data(), ldd},
			{(TypeCompute)1.0f, beta1},
			typename ReLUTransferOp<TypeC2>::Params(),
		};

		fc_multiply_b2b_impl<Gemm>(stream, arguments);
	} else {
        throw std::runtime_error{
            "Unsupported activation type in fc_multiply_b2b()"
        };
    }
}

TCNN_NAMESPACE_END
