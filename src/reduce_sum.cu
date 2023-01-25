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

/** @file   reduce_sum.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  Wrapper around thrust's sum reduction to provide warning-free compilation.
 */

#include <tiny-cuda-nn/reduce_sum.h>

TCNN_NAMESPACE_BEGIN

__global__ void block_reduce1(
	const uint32_t n_elements,
	float* __restrict__ inout
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

	extern __shared__ float sdata[];
	sdata[threadIdx.x] = i < n_elements ? inout[i] : 0;

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
			inout[blockIdx.x] = val;
		}
	}
}

uint32_t reduce_sum_workspace_size(uint32_t n_elements) {
	return n_blocks_linear(n_elements);
}

TCNN_NAMESPACE_END
