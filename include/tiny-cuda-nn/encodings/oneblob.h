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

/** @file   oneblob.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the oneblob encoding [Mueller et al. 2019].
 *          The Gaussian kernel was replaced by a quartic kernel for performance.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/misc_kernels.h>


#include <random>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>


TCNN_NAMESPACE_BEGIN


template <typename F>
__device__ inline float one_blob_subwarp_aligned(F kernel, const float* __restrict__ data_in, const uint32_t base_input_index, const uint32_t encoded_index, const uint32_t num_bins_log2) {
	const uint32_t n_bins = 1 << num_bins_log2;
	const uint32_t bin_index = encoded_index & (n_bins - 1);
	const float x = data_in[base_input_index + (encoded_index >> num_bins_log2)];

	const float left_boundary = scalbnf(bin_index, -num_bins_log2);
	float left_cdf = kernel(left_boundary - x, n_bins) + kernel(left_boundary - x - 1.0f, n_bins) + kernel(left_boundary - x + 1.0f, n_bins);

	// OneBlob needs an evaluation for both the left and the right boundary.
	// Compute cost can be saved by computing just one boundary and shuffling the result of the next from the neighboring lane.
	// Threadblocks are arranged such that bin counts are powers of two that never span across multiple warps.
	// Note that this procedure necessitates making the OneBlob encoding wrap around (hence also the 3 kernel calls above),
	// which may not always be desired.
	// If not desired, use the slower implementation without wraparound below.
	float right_cdf = __shfl_sync(0xffffffff, left_cdf, bin_index + 1, n_bins);
	if (bin_index == n_bins - 1) {
		right_cdf += 1; // The right CDF must gain a 1 due to wrapping from right to left (it lost one (hopefully) saturated CDF)
	}

	return right_cdf - left_cdf;
}


template <typename F>
__device__ inline float one_blob(F kernel, const float* __restrict__ data_in, const uint32_t base_input_index, const uint32_t encoded_index, const uint32_t num_bins_log2) {
	const uint32_t n_bins = 1 << num_bins_log2;
	const uint32_t bin_index = encoded_index & (n_bins - 1);
	const float x = data_in[base_input_index + (encoded_index >> num_bins_log2)];

	const float left_boundary = scalbnf(bin_index, -num_bins_log2);
	const float left_cdf = kernel(left_boundary - x, n_bins);

	const float right_boundary = scalbnf(bin_index + 1, -num_bins_log2);
	const float right_cdf = kernel(right_boundary - x, n_bins);

	return right_cdf - left_cdf;
}


template <typename T>
__global__ void kernel_one_blob(
	const uint32_t num_elements,
	const uint32_t num_bins_log2,
	const uint32_t num_to_encode,
	const uint32_t num_passthrough,
	const uint32_t num_to_pad,
	const float* __restrict__ data_in,
	T* __restrict__ data_out,
	float* __restrict__ dy_dx)
{
	const uint32_t fan_in_encoded = num_to_encode;
	const uint32_t fan_in = fan_in_encoded + num_passthrough;
	const uint32_t fan_out_encoded = num_to_encode << num_bins_log2;
	const uint32_t fan_out = fan_out_encoded + num_passthrough + num_to_pad;

	const uint32_t i = blockIdx.x * blockDim.y + threadIdx.y;
	const uint32_t j = threadIdx.x;
	const uint32_t output_index = j + i * blockDim.x;
	if (output_index >= num_elements * fan_out) return;

	const uint32_t input_index = i * fan_in;

	if (j >= fan_out_encoded) {
		// A value of 1 here allows the network to learn a bias-like thing.
		data_out[output_index] = j >= (fan_out_encoded + num_passthrough) ? 1 : data_in[input_index + fan_in_encoded + j - fan_out_encoded];
	} else {
		data_out[output_index] = (T)one_blob_subwarp_aligned(quartic_cdf, data_in, input_index, j, num_bins_log2);
		if (dy_dx != nullptr) {
			// Negative sign, because the kernels are translated with their input (i.e. the input has a negative sign)
			dy_dx[i * fan_out_encoded + j] = -one_blob_subwarp_aligned(quartic_cdf_deriv, data_in, input_index, j, num_bins_log2);
		}
	}
}

template <typename T>
__global__ void kernel_one_blob_backward(
	const uint32_t num_elements,
	const uint32_t n_dims_to_encode,
	const uint32_t n_dims_to_pass_through,
	const uint32_t n_padded_output_dims,
	const uint32_t n_bins,
	const T* dL_dy,
	const float* dy_dx,
	float* dL_dx)
{
	const uint32_t fan_out = n_dims_to_encode + n_dims_to_pass_through;
	const uint32_t stride = n_padded_output_dims;

	const uint32_t outputs_per_input = n_bins;
	const uint32_t passthrough_offset = n_dims_to_encode * outputs_per_input;

	const uint32_t output_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (output_index >= num_elements) return;

	const uint32_t i = output_index / fan_out;
	const uint32_t j = output_index % fan_out;

	if (j >= n_dims_to_encode) {
		// Passthrough dimensions act as a passthrough in the backward pass, too.
		dL_dx[output_index] = dL_dy[i * stride + j - n_dims_to_encode + passthrough_offset];
	} else {
		float result = 0;
		for (int k = 0; k < outputs_per_input; ++k) {
			result += (float)dL_dy[i * stride + j * outputs_per_input + k] * dy_dx[i * passthrough_offset + j * outputs_per_input + k];
		}
		dL_dx[output_index] = result;
	}
}

template <typename T>
class OneBlobEncoding : public Encoding<T> {
public:
	OneBlobEncoding(uint32_t n_bins, uint32_t n_dims_to_encode, uint32_t n_dims_to_pass_through, uint32_t alignment)
	: m_n_bins{n_bins}, m_n_dims_to_encode{n_dims_to_encode}, m_n_dims_to_pass_through{n_dims_to_pass_through} {
		m_n_output_dims = m_n_dims_to_encode * m_n_bins + m_n_dims_to_pass_through;
		m_n_padded_output_dims = next_multiple(m_n_output_dims, alignment);
		m_n_to_pad = m_n_padded_output_dims - m_n_output_dims;

		// Make sure the number of bins is a power of 2---this is required for certain optimizations
		// in our compute kernel.
		if ((n_bins & (n_bins - 1)) != 0) {
			throw std::runtime_error{"Number of bins must be a power of 2"};
		}
	}

	void encode(
		const uint32_t num_elements,
		const float* inputs,
		T* outputs,
		cudaStream_t stream,
		float* dy_dx = nullptr,
		bool is_inference = false
	) const override {
		if (m_n_padded_output_dims == 0) {
			return;
		}

		const uint32_t num_bins_log2 = (uint32_t)std::log2(m_n_bins);

		// Since the padded number of dimensions is always divisible by 8, we can
		// always exactly divide it by 2.
		const uint32_t min_n_threads = n_threads_linear;
		const dim3 threads = { num_encoded_dims(), div_round_up(min_n_threads, num_encoded_dims()), 1 };
		const uint32_t n_threads = threads.x * threads.y;
		const dim3 blocks = { div_round_up(num_elements * num_encoded_dims(), n_threads), 1, 1 };

		kernel_one_blob<T><<<blocks, threads, 0, stream>>>(
			num_elements,
			num_bins_log2,
			m_n_dims_to_encode,
			m_n_dims_to_pass_through,
			m_n_to_pad,
			inputs,
			outputs,
			dy_dx
		);
	}

	void backward(
		cudaStream_t stream,
		const uint32_t num_elements,
		const T* dL_dy, // num_encoded_dims() x num_elements
		const float* dy_dx, // encoded output dims x num_elements
		float* dL_dx, // input dims x num_elements
		const float* inputs
	) override {
		if (m_n_padded_output_dims == 0) {
			return;
		}

		const uint32_t n_input_dims = m_n_dims_to_encode + m_n_dims_to_pass_through;
		linear_kernel(kernel_one_blob_backward<T>, 0, stream,
			num_elements * n_input_dims,
			m_n_dims_to_encode,
			m_n_dims_to_pass_through,
			m_n_padded_output_dims,
			m_n_bins,
			dL_dy,
			dy_dx,
			dL_dx
		);
	}

	uint32_t num_encoded_dims() const override {
		return m_n_padded_output_dims;
	}

	uint32_t num_forward_gradient_dims() const override {
		return m_n_dims_to_encode * m_n_bins;
	}

private:
	uint32_t m_n_bins;
	uint32_t m_n_dims_to_encode;
	uint32_t m_n_dims_to_pass_through;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_padded_output_dims;
	uint32_t m_n_to_pad;
};

TCNN_NAMESPACE_END
