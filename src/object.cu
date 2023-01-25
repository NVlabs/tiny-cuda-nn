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

/** @file   object.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface of a TCNN object
 */

#include <tiny-cuda-nn/object.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>

TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void one_hot_batched_kernel(const uint32_t num_elements, const uint32_t width, const uint32_t one_hot_dim, T* out, float scale) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint32_t dim = i % width;
	out[i] = dim == one_hot_dim ? (T)scale : (T)0.0f;
}

template <typename T>
void one_hot_batched(cudaStream_t stream, const uint32_t num_elements, const uint32_t width, const uint32_t one_hot_dim, T* out, float scale) {
	linear_kernel(one_hot_batched_kernel<T>, 0, stream, num_elements, width, one_hot_dim, out, scale);
}

template void one_hot_batched(cudaStream_t stream, const uint32_t num_elements, const uint32_t width, const uint32_t one_hot_dim, float* out, float scale);
template void one_hot_batched(cudaStream_t stream, const uint32_t num_elements, const uint32_t width, const uint32_t one_hot_dim, __half* out, float scale);

template <typename T>
void mult(cudaStream_t stream, const uint32_t num_elements, T* inout, float factor) {
	linear_kernel(mult_scalar_kernel<T>, 0, stream, num_elements, inout, factor);
}

template void mult(cudaStream_t stream, const uint32_t num_elements, float* inout, float factor);
template void mult(cudaStream_t stream, const uint32_t num_elements, __half* inout, float factor);

template <typename T>
void trim_and_cast_from(cudaStream_t stream, const MatrixLayout layout, const uint32_t num_elements, const uint32_t input_width, const uint32_t output_width, const T* in, float* out) {
	if (layout == RM) {
		linear_kernel(cast_from<T>, 0, stream, num_elements, in, out);
	} else {
		linear_kernel(trim_and_cast<T>, 0, stream, num_elements, input_width, output_width, in, out);
	}
}

template void trim_and_cast_from(cudaStream_t stream, const MatrixLayout layout, const uint32_t num_elements, const uint32_t input_width, const uint32_t output_width, const float* in, float* out);
template void trim_and_cast_from(cudaStream_t stream, const MatrixLayout layout, const uint32_t num_elements, const uint32_t input_width, const uint32_t output_width, const __half* in, float* out);

TCNN_NAMESPACE_END
