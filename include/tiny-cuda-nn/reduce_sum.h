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

/** @file   reduce_sum.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Wrapper around thrust's sum reduction to provide warning-free compilation.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <cassert>
#include <iostream>
#include <map>

TCNN_NAMESPACE_BEGIN

uint32_t reduce_sum_workspace_size(uint32_t n_elements);

template <typename T>
inline __device__ T warp_reduce(T val) {
	TCNN_PRAGMA_UNROLL
	for (int offset = warpSize/2; offset > 0; offset /= 2) {
		val += __shfl_xor_sync(0xffffffff, val, offset);
	}

	return val;
}

template <typename T, typename T_OUT, typename F>
__global__ void block_reduce(
	const uint32_t n_elements,
	const F fun,
	const T* __restrict__ input,
	T_OUT* __restrict__ output,
	const uint32_t n_blocks
) {
	const uint32_t sum_idx = blockIdx.x / n_blocks;
	const uint32_t sub_blocks_idx = blockIdx.x % n_blocks;

	const uint32_t i = threadIdx.x + sub_blocks_idx * blockDim.x;
	const uint32_t block_offset = sum_idx * n_elements;

	static __shared__ T_OUT sdata[32];

	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	using T_DECAYED = std::decay_t<T>;

	T_OUT val;
	if (std::is_same<T_DECAYED, __half>::value || std::is_same<T_DECAYED, ::half>::value) {
		if (i < n_elements) {
			::half vals[8];
			*(int4*)&vals[0] = *((int4*)input + i + block_offset);
			val = fun((T)vals[0]) + fun((T)vals[1]) + fun((T)vals[2]) + fun((T)vals[3]) + fun((T)vals[4]) + fun((T)vals[5]) + fun((T)vals[6]) + fun((T)vals[7]);
		} else {
			val = 0;
		}
	} else if (std::is_same<T_DECAYED, float>::value) {
		if (i < n_elements) {
			float4 vals = *((float4*)input + i + block_offset);
			val = fun((T)vals.x) + fun((T)vals.y) + fun((T)vals.z) + fun((T)vals.w);
		} else {
			val = 0;
		}
	} else if (std::is_same<T_DECAYED, double>::value) {
		if (i < n_elements) {
			double2 vals = *((double2*)input + i + block_offset);
			val = fun((T)vals.x) + fun((T)vals.y);
		} else {
			val = 0;
		}
	} else {
		assert(false);
		return;
	}

	val = warp_reduce(val);

	if (lane == 0) sdata[wid] = val;

	__syncthreads();

	if (wid == 0) {
		val = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane] : 0;
		val = warp_reduce(val);

		if (lane == 0) {
			atomicAdd(&output[sum_idx], val);
		}
	}
}

template <typename T, typename T_OUT, typename F>
void reduce_sum(T* device_pointer, F fun, T_OUT* workspace, uint32_t n_elements, cudaStream_t stream, uint32_t n_sums = 1) {
	const uint32_t threads = 1024;

	const uint32_t N_ELEMS_PER_LOAD = 16 / sizeof(T);

	if (n_elements % N_ELEMS_PER_LOAD != 0) {
		throw std::runtime_error{"Number of bytes to reduce_sum must be a multiple of 16."};
	}
	if (((size_t)device_pointer) % 16 != 0) {
		throw std::runtime_error{"Can only reduce_sum on 16-byte aligned memory."};
	}
	n_elements /= N_ELEMS_PER_LOAD;

	uint32_t blocks = div_round_up(n_elements, threads);
	block_reduce<T, T_OUT, F><<<blocks * n_sums, threads, 0, stream>>>(n_elements, fun, device_pointer, workspace, blocks);
}

template <typename T>
void reduce_sum(T* device_pointer, float* workspace, uint32_t n_elements, cudaStream_t stream) {
	reduce_sum(device_pointer, [] __device__ (float val) { return val; }, workspace, n_elements, stream);
}

template <typename T, typename F>
float reduce_sum(T* device_pointer, F fun, uint32_t n_elements, cudaStream_t stream) {
	auto workspace = allocate_workspace(stream, reduce_sum_workspace_size(n_elements) * sizeof(float));
	float* workspace_data = (float*)workspace.data();

	CUDA_CHECK_THROW(cudaMemsetAsync(workspace_data, 0, sizeof(float), stream));
	reduce_sum(device_pointer, fun, workspace_data, n_elements, stream);

	float sum;
	CUDA_CHECK_THROW(cudaMemcpyAsync(&sum, workspace_data, sizeof(float), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	return sum;
}

template <typename T>
float reduce_sum(T* device_pointer, uint32_t n_elements, cudaStream_t stream) {
	return reduce_sum(device_pointer, [] __device__ (float val) { return val; }, n_elements, stream);
}

template <typename T, typename F>
__global__ void block_reduce0(
	const uint32_t n_elements,
	const F fun,
	const T* __restrict__ input,
	float* __restrict__ output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

	extern __shared__ float sdata[];
	sdata[threadIdx.x] = i < n_elements ? fun(input[i]) : 0;

	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (threadIdx.x < s) {
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}

		__syncthreads();
	}

	if (threadIdx.x < 32) {
		float val = sdata[threadIdx.x];
		val = warp_reduce(val);

		if (threadIdx.x == 0) {
			output[blockIdx.x] = val;
		}
	}
}

__global__ void block_reduce1(
	const uint32_t n_elements,
	float* __restrict__ inout
);

template <typename T, typename F>
void reduce_sum_old(T* device_pointer, F fun, float* workspace, uint32_t n_elements, cudaStream_t stream) {
	linear_kernel(block_reduce0<T, F>, sizeof(float) * n_threads_linear, stream, n_elements, fun, device_pointer, workspace);

	n_elements = n_blocks_linear(n_elements);

	// If the first block reduction wasn't sufficient, keep reducing
	while (n_elements > 1) {
		linear_kernel(block_reduce1, sizeof(float) * n_threads_linear, stream, n_elements, workspace);
		n_elements = n_blocks_linear(n_elements);
	}
}

TCNN_NAMESPACE_END
