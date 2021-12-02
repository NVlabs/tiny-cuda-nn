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

/** @file   nrc.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  The encoding used by the Neural Radiance Caching (NRC) paper [Müller et al. 2021].
 *          Uses a variant of Mildenhall et al.'s frequency encoding (no cosines, sines replaced
 *          by a triangle wave), concatenated with a variant of Müller et al.'s oneblob encoding
 *          (the Gaussian kernel replaced by a quartic kernel).
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/encodings/oneblob.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/misc_kernels.h>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>


TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void nrc_encoding_kernel(
	const uint32_t num_elements,
	const uint32_t num_frequencies,
	const uint32_t log2_num_oneblob_bins,
	const uint32_t num_to_oneblob_encode,
	const uint32_t num_passthrough,
	const uint32_t num_to_pad,
	PitchedPtr<const float> data_in,
	PitchedPtr<T> data_out)
{
	const uint32_t fan_in_encoded = num_to_oneblob_encode + 3;
	const uint32_t fan_out = 3 * num_frequencies + (num_to_oneblob_encode << log2_num_oneblob_bins) + num_passthrough + num_to_pad;

	const uint32_t i = blockIdx.x * blockDim.y + threadIdx.y;
	uint32_t j = threadIdx.x;
	const uint32_t encoded_index = j + i * blockDim.x;
	if (encoded_index >= num_elements * fan_out) return;

	const uint32_t encoded_offset = num_to_oneblob_encode << log2_num_oneblob_bins;
	const uint32_t passthrough_offset = encoded_offset + 3 * num_frequencies;
	const uint32_t padding_offset = passthrough_offset + num_passthrough;

	if (j >= passthrough_offset) {
		// A value of 1 here allows the network to learn a bias-like thing.
		data_out(i)[j] = j >= padding_offset ? 1 : data_in(i)[fan_in_encoded + j - passthrough_offset];
	} else if (j >= encoded_offset) {
		j -= encoded_offset;
		const uint32_t log2_frequency = j / 3;
		const uint32_t pos_dim = j % 3;

		const float x = scalbnf(data_in(i)[pos_dim], log2_frequency);

		// Small log2_frequency-based phase shift to help disambiguate locations
		const float val = x / 2 + log2_frequency * 0.25f;
		const float result = fabsf(val - floorf(val) - 0.5f) * 4 - 1;

		data_out(i)[j] = (T)result;
	} else {
		data_out(i)[j] = (T)one_blob_subwarp_aligned(quartic_cdf, data_in(i) + 3, j, log2_num_oneblob_bins);
	}
}

template <typename T>
class NrcEncoding : public Encoding<T> {
public:
	NrcEncoding(uint32_t n_frequencies, uint32_t n_oneblob_bins, uint32_t n_dims_to_encode, uint32_t n_dims_to_pass_through)
	: m_n_frequencies{n_frequencies}, m_n_oneblob_bins{n_oneblob_bins}, m_n_dims_to_oneblob_encode{n_dims_to_encode - 3}, m_n_dims_to_pass_through{n_dims_to_pass_through} {
		m_n_padded_output_dims = m_n_output_dims = 3 * m_n_frequencies + m_n_dims_to_oneblob_encode * m_n_oneblob_bins + m_n_dims_to_pass_through;

		if (n_dims_to_encode < 3) {
			throw std::runtime_error{"NrcEncoding only supports 3 or more encoded dimensions."};
		}

		// Make sure the number of bins is a power of 2---this is required for certain optimizations
		// in our compute kernel.
		if ((n_oneblob_bins & (n_oneblob_bins - 1)) != 0) {
			throw std::runtime_error{"Number of pos bins must be a power of 2"};
		}
	}

	void encode(
		cudaStream_t stream,
		const uint32_t n_elements,
		PitchedPtr<const float> inputs,
		PitchedPtr<T> outputs,
		float* dy_dx = nullptr,
		bool is_inference = false
	) const override {
		const uint32_t log2_n_oneblob_bins = (uint32_t)std::log2(m_n_oneblob_bins);

		const uint32_t min_n_threads = n_threads_linear;
		const dim3 threads = { num_encoded_dims(), div_round_up(min_n_threads, num_encoded_dims()), 1 };
		const uint32_t n_threads = threads.x * threads.y;
		const dim3 blocks = { div_round_up(n_elements * num_encoded_dims(), n_threads), 1, 1 };

		nrc_encoding_kernel<T><<<blocks, threads, 0, stream>>>(
			n_elements,
			m_n_frequencies,
			log2_n_oneblob_bins,
			m_n_dims_to_oneblob_encode,
			m_n_dims_to_pass_through,
			m_n_to_pad,
			inputs,
			outputs
		);
	}

	void backward(
		cudaStream_t stream,
		const uint32_t n_elements,
		PitchedPtr<const T> dL_dy, // Same shape as outputs
		const float* dy_dx, // encoded output dims x n_elements
		PitchedPtr<float> dL_dx, // Same shape as inputs
		PitchedPtr<const float> inputs,
		bool accumulate_param_gradients
	) override {
		if (m_n_padded_output_dims == 0) {
			return;
		}

		// Can't compute input gradients if insufficient info is available
		if (!dy_dx || !dL_dx) {
			return;
		}

		throw std::runtime_error{"NrcEncoding does not support the backward pass."};
	}

	uint32_t num_dims_to_encode() const override {
		return m_n_dims_to_oneblob_encode + 3;
	}

	uint32_t num_encoded_dims() const override {
		return m_n_padded_output_dims;
	}

	uint32_t num_forward_gradient_dims() const override {
		return 0; // Unsupported for now
	}

	void set_alignment(uint32_t alignment) override {
		alignment = std::lcm(alignment, min_alignment());
		m_n_padded_output_dims = next_multiple(m_n_output_dims, alignment);
		m_n_to_pad = m_n_padded_output_dims - m_n_output_dims;
	}

	uint32_t min_alignment() const override {
		return m_n_oneblob_bins;
	}

private:
	uint32_t m_n_frequencies;

	uint32_t m_n_oneblob_bins;
	uint32_t m_n_dims_to_oneblob_encode;
	uint32_t m_n_dims_to_pass_through;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_padded_output_dims;
	uint32_t m_n_to_pad = 0;
};

TCNN_NAMESPACE_END
