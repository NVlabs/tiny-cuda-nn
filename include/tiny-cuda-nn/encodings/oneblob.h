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

/** @file   oneblob.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the oneblob encoding [Mueller et al. 2019].
 *          The Gaussian kernel was replaced by a quartic kernel for performance.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

template <typename F>
__device__ inline float one_blob_subwarp_aligned(F kernel, const float* __restrict__ data_in, const uint32_t encoded_index, const uint32_t num_bins_log2) {
	const uint32_t n_bins = 1 << num_bins_log2;
	const uint32_t bin_index = encoded_index & (n_bins - 1);
	const float x = data_in[encoded_index >> num_bins_log2];

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
__device__ inline float one_blob(F kernel, const float* __restrict__ data_in, const uint32_t encoded_index, const uint32_t num_bins_log2) {
	const uint32_t n_bins = 1 << num_bins_log2;
	const uint32_t bin_index = encoded_index & (n_bins - 1);
	const float x = data_in[encoded_index >> num_bins_log2];

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
	const uint32_t num_to_pad,
	PitchedPtr<const float> data_in,
	PitchedPtr<T> data_out
) {
	const uint32_t fan_out_encoded = num_to_encode << num_bins_log2;
	const uint32_t fan_out = fan_out_encoded + num_to_pad;

	const uint32_t i = blockIdx.x * blockDim.y + threadIdx.y;
	const uint32_t j = threadIdx.x;
	const uint32_t encoded_index = j + i * blockDim.x;
	if (encoded_index >= num_elements * fan_out) return;

	if (j >= fan_out_encoded) {
		// A value of 1 here allows the network to learn a bias-like thing.
		data_out(i)[j] = 1;
	} else {
		data_out(i)[j] = (T)one_blob_subwarp_aligned(quartic_cdf, data_in(i), j, num_bins_log2);
	}
}

template <typename T>
__global__ void kernel_one_blob_soa(
	const uint32_t num_elements,
	const uint32_t num_bins_log2,
	const uint32_t num_to_encode,
	PitchedPtr<const float> data_in,
	PitchedPtr<T> data_out
) {
	const uint32_t i = blockIdx.x * blockDim.y + threadIdx.y;
	const uint32_t j = threadIdx.x;
	const uint32_t to_encode_index = j + i * blockDim.x;
	if (to_encode_index >= num_elements * num_to_encode) return;

	const float x = data_in(i)[j];

	const uint32_t n_bins = 1 << num_bins_log2;
	T* out = (data_out.ptr + i + j * n_bins * num_elements);

	float left_cdf = quartic_cdf(-x, n_bins) + quartic_cdf(-x - 1.0f, n_bins) + quartic_cdf(-x + 1.0f, n_bins);

	for (uint32_t k = 0; k < n_bins; ++k) {
		const float right_boundary = scalbnf(k+1, -num_bins_log2);
		const float right_cdf = quartic_cdf(right_boundary - x, n_bins) + quartic_cdf(right_boundary - x - 1.0f, n_bins) + quartic_cdf(right_boundary - x + 1.0f, n_bins);

		*out = (T)(right_cdf - left_cdf);

		left_cdf = right_cdf;
		out += num_elements;
	}
}

template <typename T>
__global__ void kernel_one_blob_backward(
	const uint32_t num_elements,
	const uint32_t n_dims_to_encode,
	const uint32_t num_bins_log2,
	MatrixLayout output_layout,
	PitchedPtr<const T> dL_dy,
	PitchedPtr<const float> data_in,
	PitchedPtr<float> dL_dx)
{
	const uint32_t i = blockIdx.x * blockDim.y + threadIdx.y;
	const uint32_t j = threadIdx.x;
	const uint32_t to_encode_index = j + i * blockDim.x;
	if (to_encode_index >= num_elements * n_dims_to_encode) return;

	const float x = data_in(i)[j];

	const uint32_t n_bins = 1 << num_bins_log2;

	float result = 0;

	float left_cdf = quartic_cdf_deriv(-x, n_bins) + quartic_cdf_deriv(-x - 1.0f, n_bins) + quartic_cdf_deriv(-x + 1.0f, n_bins);

	for (uint32_t k = 0; k < n_bins; ++k) {
		const float right_boundary = scalbnf(k+1, -num_bins_log2);
		const float right_cdf = quartic_cdf_deriv(right_boundary - x, n_bins) + quartic_cdf_deriv(right_boundary - x - 1.0f, n_bins) + quartic_cdf_deriv(right_boundary - x + 1.0f, n_bins);

		float deriv = left_cdf - right_cdf;

		left_cdf = right_cdf;

		uint32_t encoded_dim = j * n_bins + k;
		const T* grad = output_layout == AoS ? &dL_dy(i)[encoded_dim] : (dL_dy.ptr + i + encoded_dim * num_elements);

		result += (float)*grad * deriv;
	}

	dL_dx(i)[j] = result;
}

template <typename T>
class OneBlobEncoding : public Encoding<T> {
public:
	OneBlobEncoding(uint32_t n_bins, uint32_t n_dims_to_encode)
	: m_n_bins{n_bins}, m_n_dims_to_encode{n_dims_to_encode} {
		m_n_padded_output_dims = m_n_output_dims = m_n_dims_to_encode * m_n_bins;

		// Make sure the number of bins is a power of 2---this is required for certain optimizations
		// in our compute kernel.
		if ((n_bins & (n_bins - 1)) != 0) {
			throw std::runtime_error{"Number of bins must be a power of 2"};
		}
	}

	void encode(
		cudaStream_t stream,
		const uint32_t num_elements,
		PitchedPtr<const float> inputs,
		PitchedPtr<T> outputs,
		float* dy_dx = nullptr,
		bool is_inference = false
	) const override {
		if (m_n_padded_output_dims == 0) {
			return;
		}

		const uint32_t num_bins_log2 = (uint32_t)std::log2(m_n_bins);

		if (m_output_layout == AoS) {
			const uint32_t min_n_threads = n_threads_linear;
			const dim3 threads = { num_encoded_dims(), div_round_up(min_n_threads, num_encoded_dims()), 1 };
			const uint32_t n_threads = threads.x * threads.y;
			const dim3 blocks = { div_round_up(num_elements * num_encoded_dims(), n_threads), 1, 1 };

			kernel_one_blob<T><<<blocks, threads, 0, stream>>>(
				num_elements,
				num_bins_log2,
				m_n_dims_to_encode,
				m_n_to_pad,
				inputs,
				outputs
			);
		} else {
			const uint32_t min_n_threads = n_threads_linear;
			const dim3 threads = { m_n_dims_to_encode, div_round_up(min_n_threads, m_n_dims_to_encode), 1 };
			const uint32_t n_threads = threads.x * threads.y;
			const dim3 blocks = { div_round_up(num_elements * m_n_dims_to_encode, n_threads), 1, 1 };

			kernel_one_blob_soa<T><<<blocks, threads, 0, stream>>>(
				num_elements,
				num_bins_log2,
				m_n_dims_to_encode,
				inputs,
				outputs
			);

			// Padding
			parallel_for_gpu(stream, num_elements * m_n_to_pad, [outputs=outputs.ptr + num_elements * m_n_dims_to_encode] __device__ (size_t i) {
				outputs[i] = (T)1.0f;
			});
		}
	}

	void backward(
		cudaStream_t stream,
		const uint32_t num_elements,
		PitchedPtr<const T> dL_dy, // Same shape as outputs
		const float* dy_dx, // encoded output dims x num_elements
		PitchedPtr<float> dL_dx, // Same shape as inputs
		PitchedPtr<const float> inputs,
		EGradientMode param_gradients_mode
	) override {
		if (m_n_padded_output_dims == 0) {
			return;
		}

		// Can't compute input gradients if insufficient info is available
		if (!dL_dx) {
			return;
		}

		const uint32_t num_bins_log2 = (uint32_t)std::log2(m_n_bins);

		const uint32_t min_n_threads = n_threads_linear;
		const dim3 threads = { m_n_dims_to_encode, div_round_up(min_n_threads, m_n_dims_to_encode), 1 };
		const uint32_t n_threads = threads.x * threads.y;
		const dim3 blocks = { div_round_up(num_elements * m_n_dims_to_encode, n_threads), 1, 1 };

		kernel_one_blob_backward<T><<<blocks, threads, 0, stream>>>(
			num_elements,
			m_n_dims_to_encode,
			num_bins_log2,
			m_output_layout,
			dL_dy,
			inputs,
			dL_dx
		);
	}

	uint32_t num_dims_to_encode() const override {
		return m_n_dims_to_encode;
	}

	uint32_t num_encoded_dims() const override {
		return m_n_padded_output_dims;
	}

	uint32_t num_forward_gradient_dims() const override {
		return 0;
	}

	void set_alignment(uint32_t alignment) override {
		alignment = lcm(alignment, min_alignment());
		m_n_padded_output_dims = next_multiple(m_n_output_dims, alignment);
		m_n_to_pad = m_n_padded_output_dims - m_n_output_dims;
	}

	uint32_t min_alignment() const override {
		return m_n_bins;
	}

	bool supports_output_layout(MatrixLayout layout) const override {
		return true;
	}

	void set_output_layout(MatrixLayout layout) override {
		m_output_layout = layout;
	}

	MatrixLayout output_layout() const override {
		return m_output_layout;
	}

	json hyperparams() const override {
		return {
			{"otype", "OneBlob"},
			{"n_bins", m_n_bins},
		};
	}

private:
	uint32_t m_n_bins;
	uint32_t m_n_dims_to_encode;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_padded_output_dims;
	uint32_t m_n_to_pad = 0;

	MatrixLayout m_output_layout = AoS;
};

TCNN_NAMESPACE_END
