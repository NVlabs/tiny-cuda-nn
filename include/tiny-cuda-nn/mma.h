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

/** @file   mma.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  mma instruction (tensor core) compatible matrix and vector classes that can
 *          interface with the similarly named scalar vector classes from "vec.h".
 *          Used in the tiny-cuda-nn's new JIT functionality to enable full fusion
 *          of entire forward+loss+bwd passes (or, downstream in Instant NGP, full
 *          fusion of a NeRF renderer, i.e. inline MLP queries in raymarching kernel).
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>

namespace tcnn {

// Implementation of the MMA fragment / register layout as per
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float
// Each warp-wide fragment contains 16x16 fp16 values, distributed over four registers that hold
// 8x8 warp-wide fp16 blocks each (arranged in CM or RM order). RM order corresponds to type A or
// accumulator fragments while CM corresponds to two type B fragments (each 16x8). For each individual thread in
// a warp, this amounts to two fp16 values, i.e. one 32-bit register (== 8x8 / 32 / 2, recall that 32
// is the number of threads in a warp).
template <MatrixLayout _LAYOUT>
struct alignas(16) mma_frag {
	static constexpr MatrixLayout LAYOUT = _LAYOUT;
	static constexpr MatrixLayout TRANSPOSED_LAYOUT = LAYOUT == CM ? RM : CM;

	TCNN_DEVICE static uint32_t transpose_reg(uint32_t reg) {
		asm("movmatrix.sync.trans.aligned.m8n8.b16 %0, %0;" : "+r"(reg));
		return reg;
	}

	// The four registers of a 16x16 fragment form 8x8 sub-fragments that are
	// arranged in 2x2 col-major fashion, regardless of whether the underlying
	// 8x8 blocks are represented in col- or row-major layout. This is why the
	// below `flip_layout` function leaves the register arrangement unchanged
	// whereas the `transpose` function transposes it. Under normal circumstances
	// where the 2x2 register arrangement would follow the global row-/col-major
	// setting, it would be the other way around.
	uint32_t regs[4];

	TCNN_DEVICE auto flip_layout() const {
		mma_frag<TRANSPOSED_LAYOUT> result;

		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < 4; ++i) {
			result.regs[i] = transpose_reg(regs[i]);
		}

		return result;
	}

	TCNN_DEVICE auto transpose() const {
		mma_frag<LAYOUT> result;

		TCNN_PRAGMA_UNROLL
		for (uint32_t reg_row = 0; reg_row < 2; ++reg_row) {
			TCNN_PRAGMA_UNROLL
			for (uint32_t reg_col = 0; reg_col < 2; ++reg_col) {
				result.regs[reg_row * 2 + reg_col] = transpose_reg(regs[reg_row + reg_col * 2]);
			}
		}

		return result;
	}

	TCNN_DEVICE void mma_sync_accum(const mma_frag<RM>& a, const mma_frag<CM>& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
		asm(
			"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
			"{%0, %1}, "
			"{%2, %3, %4, %5}, "
			"{%6, %7}, "
			"{%0, %1};"
			: "+r"(regs[0]), "+r"(regs[1]) :
			"r"(a.regs[0]), "r"(a.regs[1]), "r"(a.regs[2]), "r"(a.regs[3]),
			"r"(b.regs[0]), "r"(b.regs[1])
		);

		asm(
			"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
			"{%0, %1}, "
			"{%2, %3, %4, %5}, "
			"{%6, %7}, "
			"{%0, %1};"
			: "+r"(regs[2]), "+r"(regs[3]) :
			"r"(a.regs[0]), "r"(a.regs[1]), "r"(a.regs[2]), "r"(a.regs[3]),
			"r"(b.regs[2]), "r"(b.regs[3])
		);
#else
		// The above mma instructions are only available on compute capability 80 and above.
		// For earlier CCs fall back to the wmma instruction family (which is less well
		// documented but should yield equivalent perf).
		asm(
			"wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16 "
			"{%0, %1, %2, %3}, "
			"{%4, %5, %6, %7, %4, %5, %6, %7}, "
			"{%8, %9, %10, %11, %8, %9, %10, %11}, "
			"{%0, %1, %2, %3};"
			: "+r"(regs[0]), "+r"(regs[1]), "+r"(regs[2]), "+r"(regs[3]) :
			"r"(a.regs[0]), "r"(a.regs[1]), "r"(a.regs[2]), "r"(a.regs[3]),
			"r"(b.regs[0]), "r"(b.regs[1]), "r"(b.regs[2]), "r"(b.regs[3])
		);
#endif
	}
};

template <uint32_t _M, uint32_t _N, MatrixLayout _LAYOUT = CM>
struct alignas(16) mma_mat {
	static constexpr uint32_t M = _M;
	static constexpr uint32_t N = _N;
	static constexpr MatrixLayout LAYOUT = _LAYOUT;
	static constexpr MatrixLayout TRANSPOSED_LAYOUT = LAYOUT == CM ? RM : CM;

	static_assert(M % 16 == 0 && N % 16 == 0, "mma_mat: M and N must be divisible by 16.");
	static constexpr uint32_t STRIDE = LAYOUT == CM ? M : N;
	static constexpr uint32_t N_ELEMS = M * N;

	static constexpr uint32_t FRAG_STRIDE = LAYOUT == CM ? (M / 16) : (N / 16);
	static constexpr uint32_t N_FRAGS = N_ELEMS / 16 / 16;
	static constexpr uint32_t N_REGS = N_FRAGS * 4;

	// Each thread of the warp holds 4 32-bit registers (4 half2's) for each 16x16
	// __half-valued matrix fragment. The 16x16 fragments are arranged in M/16xN/16
	// blocks that follow the row-/col-major setting of the matrix. This is in contrast
	// to the 2x2 arrangements of the 8x8 subblocks of each 16x16 fragment, which are
	// always arranged in col-major fashion (and the 8x8 subblocks themselves again
	// respect the col-/row- major setting).
	__half2 regs[N_REGS];

	TCNN_DEVICE mma_frag<LAYOUT>& frag(uint32_t i) { return *(mma_frag<LAYOUT>*)&regs[i * 4]; }
	TCNN_DEVICE const mma_frag<LAYOUT>& frag(uint32_t i) const { return *(const mma_frag<LAYOUT>*)&regs[i * 4]; }

	TCNN_DEVICE static constexpr uint32_t frag_idx(uint32_t frag_row, uint32_t frag_col) {
		uint32_t x = LAYOUT == RM ? frag_col : frag_row;
		uint32_t y = LAYOUT == RM ? frag_row : frag_col;
		return x + y * FRAG_STRIDE;
	}

	TCNN_DEVICE mma_frag<LAYOUT>& frag(uint32_t frag_row, uint32_t frag_col) { return frag(frag_idx(frag_row, frag_col)); }
	TCNN_DEVICE const mma_frag<LAYOUT>& frag(uint32_t frag_row, uint32_t frag_col) const { return frag(frag_idx(frag_row, frag_col)); }

	TCNN_DEVICE mma_mat() {}

	TCNN_DEVICE static constexpr uint32_t to_linear(uint32_t lid, uint32_t reg_idx) {
		uint32_t fid = reg_idx / 4;
		uint32_t reg = reg_idx % 4;

		// The swap takes into account that the 2x2 arrangement of registers
		// within each fragment is col-major, independent of the LAYOUT setting.
		uint32_t rx = LAYOUT == CM ? (reg % 2) : (reg / 2);
		uint32_t ry = LAYOUT == RM ? (reg % 2) : (reg / 2);

		uint32_t x = (lid % 4) * 2 + rx * 8 + (fid % FRAG_STRIDE) * 16;
		uint32_t y = (lid / 4) + ry * 8 + (fid / FRAG_STRIDE) * 16;

		return x + y * STRIDE;
	}

	TCNN_DEVICE static constexpr uint32_t reg_from_xy(uint32_t x, uint32_t y) {
		uint32_t fid = x / 16 + (y / 16) * FRAG_STRIDE;

		// The swap takes into account that the 2x2 arrangement of registers
		// within each fragment is col-major, independent of the LAYOUT setting.
		uint32_t bx = LAYOUT == CM ? (x / 8) : (y / 8);
		uint32_t by = LAYOUT == RM ? (x / 8) : (y / 8);

		uint32_t bid = bx % 2 + (by % 2) * 2;
		return fid * 4 + bid;
	}

	TCNN_DEVICE static auto zero() {
		mma_mat<M, N, LAYOUT> result;
		memset(&result, 0, sizeof(result));
		return result;
	}

	TCNN_DEVICE static auto from_linear_memory(const __half* __restrict__ data) {
		mma_mat<M, N, LAYOUT> result;
		uint32_t lid = lane_id();

		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_REGS; ++i) {
			result.regs[i] = *(const __half2*)&data[to_linear(lid, i)];
		}

		return result;
	}

	TCNN_DEVICE void into_linear_memory(__half* __restrict__ data) const {
		uint32_t lid = lane_id();

		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_REGS; ++i) {
			*(__half2*)&data[to_linear(lid, i)] = regs[i];
		}
	}

	TCNN_DEVICE void sum_into_linear_global_memory_atomic(__half* __restrict__ data) {
		uint32_t lid = lane_id();

		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_REGS; ++i) {
			atomic_add_gmem_h2((__half2*)&data[to_linear(lid, i)], regs[i]);
		}
	}

	// Sum-reduce over threadblock before atomically adding to global memory.
	// This makes the reduction more efficient by reducing atomic-add contention
	// on global memory. However, this comes at the cost of requiring N_ELEMS
	// bytes of shared memory for each warp in the threadblock.
	template <uint32_t N_THREADS>
	TCNN_DEVICE void sum_into_linear_global_memory_hierarchical(__half* __restrict__ data) {
		extern __shared__ __half shmem[];
		uint32_t warp_id = threadIdx.x / 32;

		TCNN_PRAGMA_UNROLL
		for (int j = 2; j <= N_THREADS / 32; j <<= 1) {
			if (warp_id % j == j / 2) {
				into_native_memory(shmem + (warp_id / j) * N_ELEMS);
			}

			__syncthreads();

			if (warp_id % j == 0) {
				*this += from_native_memory(shmem + (warp_id / j) * N_ELEMS);
			}

			__syncthreads();
		}

		if (warp_id == 0) {
			sum_into_linear_global_memory_atomic(data);
		}
	}

	// "Native" layout refers to fragments being serialized to RAM as-is. This function
	// copies the fragments of this matrix from ram in warp-cooperative fashion. Every
	// fragment consists of 128 bits (four 32-bit registers), the maximum number a thread
	// can load in a single instruction, and so the coalesced access patterns below
	// result in optimal bandwidth.
	TCNN_DEVICE static auto from_native_memory(const __half* __restrict__ data) {
		mma_mat<M, N, LAYOUT> result;
		uint32_t lid = lane_id();

		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_FRAGS; ++i) {
			result.frag(i) = ((const mma_frag<LAYOUT>*)data)[i * 32 + lid];
		}

		return result;
	}

	// Same as above, but storing the fragments instead of loading them.
	TCNN_DEVICE void into_native_memory(__half* __restrict__ data) const {
		uint32_t lid = lane_id();

		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_FRAGS; ++i) {
			((mma_frag<LAYOUT>*)data)[i * 32 + lid] = frag(i);
		}
	}

	TCNN_DEVICE auto flip_layout() const {
		mma_mat<M, N, TRANSPOSED_LAYOUT> result;

		TCNN_PRAGMA_UNROLL
		for (uint32_t frag_row = 0; frag_row < M / 16; ++frag_row) {
			TCNN_PRAGMA_UNROLL
			for (uint32_t frag_col = 0; frag_col < N / 16; ++frag_col) {
				result.frag(frag_row, frag_col) = frag(frag_row, frag_col).flip_layout();
			}
		}

		return result;
	}

	TCNN_DEVICE auto transpose() const {
		mma_mat<N, M, LAYOUT> result;

		TCNN_PRAGMA_UNROLL
		for (uint32_t frag_row = 0; frag_row < M / 16; ++frag_row) {
			TCNN_PRAGMA_UNROLL
			for (uint32_t frag_col = 0; frag_col < N / 16; ++frag_col) {
				result.frag(frag_col, frag_row) = frag(frag_row, frag_col).transpose();
			}
		}

		return result;
	}

	TCNN_DEVICE auto& operator+=(const mma_mat<M, N, LAYOUT>& other) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_REGS; ++i) {
			regs[i] = __hadd2(regs[i], other.regs[i]);
		}

		return *this;
	}

	TCNN_DEVICE void activate(Activation act) {
		using vec_t = tvec<__half, N_ELEMS, 16>;
		*(vec_t*)this = vec_activation(act, *(vec_t*)this);
	}

	TCNN_DEVICE void activate_bwd(Activation act, const mma_mat<M, N, LAYOUT>& fwd_in) {
		using vec_t = tvec<__half, N_ELEMS, 16>;
		*(vec_t*)this = vec_activation_backward_in(act, *(vec_t*)this, *(vec_t*)&fwd_in);
	}

	// Comptime specializations of activation functions that make use of __half2 intrinsics
	// in the case of ReLU. Improves compile time by quite a bit and perf by 0-5% depending
	// on situation.
	template <Activation act, std::enable_if_t<act != Activation::None && act != Activation::ReLU, int> = 0>
	TCNN_DEVICE void activate() {
		using vec_t = tvec<__half, N_ELEMS, 16>;
		*(vec_t*)this = vec_activation<act, __half, N_ELEMS, 16>(*(vec_t*)this);
	}

	template <Activation act, std::enable_if_t<act != Activation::None && act != Activation::ReLU, int> = 0>
	TCNN_DEVICE void activate_bwd(const mma_mat<M, N, LAYOUT>& fwd_in) {
		using vec_t = tvec<__half, N_ELEMS, 16>;
		*(vec_t*)this = vec_activation_backward_in<act, __half, N_ELEMS, 16>(*(vec_t*)this, *(const vec_t*)&fwd_in);
	}

	template <Activation act, std::enable_if_t<act == Activation::None, int> = 0>
	TCNN_DEVICE void activate() {}

	template <Activation act, std::enable_if_t<act == Activation::None, int> = 0>
	TCNN_DEVICE void activate_bwd(const mma_mat<M, N, LAYOUT>& fwd_in) {}

	template <Activation act, std::enable_if_t<act == Activation::ReLU, int> = 0>
	TCNN_DEVICE void activate() {
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_REGS; ++i) {
			regs[i] = __hmax2(regs[i], __half2half2(0));
		}
	}

	template <Activation act, std::enable_if_t<act == Activation::ReLU, int> = 0>
	TCNN_DEVICE void activate_bwd(const mma_mat<M, N, LAYOUT>& fwd_in) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_REGS; ++i) {
			regs[i] = __hmul2(regs[i], __hgt2(fwd_in.regs[i], __half2half2(0)));
		}
	}

	// ----------- Vector-related methods that assume M == 32, layout == RM ------------
	// These methods interface between warp-wide per-thread vectors of size N and mma matrices
	// of size 32xN that hold the same total number of elements but have them shared per the
	// mma fragment layout collaboratively across the warp. The indexing logic below is carefully
	// tuned to #1 ensure comp-time indices into register arrays (else they'd spill into local
	// memory) and #2 to minimize the amount of runtime arithmetic (i.e. anything depending on
	// the lane ID) and shuffles. See
	// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float
	// for details of the mma fragment register layout.

	template <uint32_t N_IN, size_t A>
	TCNN_DEVICE mma_mat(const tvec<__half, N_IN, A>& vec) {
		static_assert(M == 32 && LAYOUT == RM, "mma_mat(hvec) is only permitted if M is 32 and the layout is row major.");

		uint32_t lid = lane_id();
		uint32_t xlid = lid % 4;
		uint32_t ylid = lid / 4;

		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N; i += 2) {
			__half2 v = {0, 0};
			if (i + 0 < N_IN) { v.x = vec[i + 0]; }
			if (i + 1 < N_IN) { v.y = vec[i + 1]; }

			uint32_t xi = (i / 2) % 4;

			TCNN_PRAGMA_UNROLL
			for (uint32_t j = 0; j < 4; ++j) {
				uint32_t paired_lid = j * 8 + ylid;
				uint32_t reg = reg_from_xy(i, j * 8);
				__half2 other = __shfl_sync(0xFFFFFFFF, v, paired_lid);

				if (xlid == xi) {
					regs[reg] = other;
				}
			}
		}
	}

	template <uint32_t N_OUT>
	TCNN_DEVICE hvec<N_OUT> vec() {
		static_assert(M == 32 && LAYOUT == RM, "mma_mat::vec() is only permitted if M is 32 and the layout is row major.");
		static_assert(N_OUT <= N, "mma_mat::vec(): cannot convert to vector with more elements than N.");

		hvec<N_OUT> result;

		uint32_t lid = lane_id();
		uint32_t xlid = lid % 8 * 4;
		uint32_t ylid = lid / 8;

		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N; i += 2) {
			uint32_t paired_lid = xlid + i / 2 % 4;

			TCNN_PRAGMA_UNROLL
			for (uint32_t j = 0; j < 4; ++j) {
				uint32_t reg = reg_from_xy(i, j * 8);
				__half2 other = __shfl_sync(0xFFFFFFFF, regs[reg], paired_lid);

				if (ylid == j) {
					if (i + 0 < N_OUT) { result[i + 0] = other.x; }
					if (i + 1 < N_OUT) { result[i + 1] = other.y; }
				}
			}
		}

		return result;
	}
};

template <uint32_t N>
using mma_vec = mma_mat<32, N, RM>;

template <uint32_t M, uint32_t K, uint32_t N>
TCNN_DEVICE auto operator*(const mma_mat<M, K, RM>& a, const mma_mat<K, N, CM>& b) {
	auto result = mma_mat<M, N, RM>::zero();

	TCNN_PRAGMA_UNROLL
	for (uint32_t frag_row = 0; frag_row < M / 16; ++frag_row) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t frag_col = 0; frag_col < N / 16; ++frag_col) {
			auto& r = result.frag(frag_row, frag_col);

			TCNN_PRAGMA_UNROLL
			for (uint32_t k = 0; k < K / 16; ++k) {
				const auto& af = a.frag(frag_row, k);
				const auto& bf = b.frag(k, frag_col);
				r.mma_sync_accum(af, bf);
			}
		}
	}

	return result;
}

template <uint32_t N1, uint32_t N2>
TCNN_DEVICE auto outer_product(const mma_vec<N1>& a, const mma_vec<N2>& b) {
	return a.transpose() * b.flip_layout();
}

}
