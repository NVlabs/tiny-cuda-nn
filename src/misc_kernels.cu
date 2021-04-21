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

/** @file   misc_kernels.cu
 *  @author Thomas Müller & Nikolaus Binder, NVIDIA
 *  @brief  Implementation of various miscellaneous CUDA kernels
 */

#include <tiny-cuda-nn/misc_kernels.h>


TCNN_NAMESPACE_BEGIN

__global__ void relu(const uint32_t num_elements, const float* __restrict__ data_in, float* __restrict__ data_out)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_out[i] = fmaxf(0.0f, data_in[i]);
}

__global__ void relu(const uint32_t num_elements, const __half* __restrict__ data_in, __half* __restrict__ data_out)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_out[i] = (__half)fmaxf(0.0f, (float)data_in[i]);
}

__global__ void exp(const uint32_t num_elements, const float* __restrict__ data_in, float* __restrict__ data_out)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_out[i] = expf(data_in[i]);
}

__global__ void exp(const uint32_t num_elements, const __half* __restrict__ data_in, __half* __restrict__ data_out)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_out[i] = expf(data_in[i]);
}

__global__ void add(const uint32_t num_elements, const float* __restrict__ data_in_1, const float* __restrict__ data_in_2, float* __restrict__ data_out)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_out[i] = data_in_1[i] + data_in_2[i];
}

TCNN_NAMESPACE_END
