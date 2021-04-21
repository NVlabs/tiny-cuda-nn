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

/** @file   frequency.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the frequency encoding of NeRF [Mildenhall et al. 2020].
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

template <typename T>
__global__ void frequency_encoding(
	const uint32_t num_elements,
	const uint32_t n_frequencies,
	const uint32_t num_to_encode,
	const uint32_t num_passthrough,
	const uint32_t num_to_pad,
	const float* __restrict__ data_in,
	T* __restrict__ data_out,
	float* __restrict__ dy_dx)
{
	const uint32_t fan_in_encoded = num_to_encode;
	const uint32_t fan_in = fan_in_encoded + num_passthrough;
	const uint32_t fan_out_encoded = num_to_encode * n_frequencies * 2;
	const uint32_t fan_out = fan_out_encoded + num_passthrough + num_to_pad;

	const uint32_t output_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (output_index >= num_elements) return;

	const uint32_t i = output_index / fan_out;
	const uint32_t j = output_index % fan_out;
	const uint32_t input_index = i * fan_in;

	/* Layout of outputs (for each input record):
	 *     frequency-encoded input dimension 0
	 *     frequency-encoded input dimension 1
	 *     frequency-encoded input dimension ...
	 *     passthrough inputs
	 *     padding (value 0.f)
	 */
	if (j >= fan_out_encoded + num_passthrough) {
		data_out[output_index] = 0;
	} else if (j >= fan_out_encoded) {
		data_out[output_index] = data_in[input_index + fan_in_encoded + j - fan_out_encoded];
	} else {
		/* Layout of encoded features (e.g. when inputs abcd.. are XYZ positions):
		 *     sin(a.x), cos(a.x) sin(2pi a.x), cos(2pi a.x) sin(4pi a.x) ...
		 *     sin(a.y), cos(a.y) sin(2pi a.y), cos(2pi a.y) sin(4pi a.y) ...
		 *     sin(a.z), cos(a.z) sin(2pi a.z), cos(2pi a.z) sin(4pi a.z) ...
		 *     (passthrough features)
		 *     (padding)
		 */
		const uint32_t encoded_input_feature_i = j / (n_frequencies * 2);

		const uint32_t log2_frequency = (j / 2) % n_frequencies;
		assert(encoded_input_feature_i < num_to_encode);
		assert(log2_frequency < n_frequencies);

		const float phase_shift = (j % 2) * (M_PI/2);

		const float x = scalbnf(data_in[input_index + encoded_input_feature_i], log2_frequency);
		const float input = x * M_PI + phase_shift;
		data_out[output_index] = (T)__sinf(input);
		if (dy_dx != nullptr) {
			dy_dx[i * fan_out_encoded + j] = scalbnf(1.0f, log2_frequency) * M_PI * __cosf(input);
		}
	}
}

template <typename T>
__global__ void frequency_encoding_backward(
	const uint32_t num_elements,
	const uint32_t n_dims_to_encode,
	const uint32_t n_dims_to_pass_through,
	const uint32_t n_padded_output_dims,
	const uint32_t n_frequencies,
	const T* dL_dy,
	const float* dy_dx,
	float* dL_dx
) {
	const uint32_t fan_out = n_dims_to_encode + n_dims_to_pass_through;
	const uint32_t stride = n_padded_output_dims;

	const uint32_t outputs_per_input = n_frequencies * 2;
	const uint32_t fan_out_encoded = n_dims_to_encode * outputs_per_input;

	const uint32_t output_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (output_index >= num_elements) return;

	const uint32_t i = output_index / fan_out;
	const uint32_t j = output_index % fan_out;

	if (j >= n_dims_to_encode) {
		// Passthrough dimensions act as a passthrough in the backward pass, too.
		dL_dx[output_index] = dL_dy[i * stride + j - n_dims_to_encode + fan_out_encoded];
	} else {
		float result = 0;
		for (int k = 0; k < outputs_per_input; ++k) {
			result += (float)dL_dy[i * stride + j * outputs_per_input + k] * dy_dx[i * fan_out_encoded + j * outputs_per_input + k];
		}
		dL_dx[output_index] = result;
	}
}

template <typename T>
class FrequencyEncoding : public Encoding<T> {
public:
	FrequencyEncoding(uint32_t n_frequencies, uint32_t n_dims_to_encode, uint32_t n_dims_to_pass_through, uint32_t alignment)
	: m_n_frequencies{n_frequencies}, m_n_dims_to_encode{n_dims_to_encode}, m_n_dims_to_pass_through{n_dims_to_pass_through} {
		m_n_output_dims = m_n_dims_to_encode * m_n_frequencies * 2 + m_n_dims_to_pass_through;
		m_n_padded_output_dims = next_multiple(m_n_output_dims, alignment);
		m_n_to_pad = m_n_padded_output_dims - m_n_output_dims;
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

		linear_kernel(frequency_encoding<T>, 0, stream,
			num_elements * num_encoded_dims(),
			m_n_frequencies,
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
		linear_kernel(frequency_encoding_backward<T>, 0, stream,
			num_elements * n_input_dims,
			m_n_dims_to_encode,
			m_n_dims_to_pass_through,
			m_n_padded_output_dims,
			m_n_frequencies,
			dL_dy,
			dy_dx,
			dL_dx
		);
	}

	uint32_t num_encoded_dims() const override {
		return m_n_padded_output_dims;
	}

	uint32_t num_forward_gradient_dims() const override {
		return m_n_dims_to_encode * m_n_frequencies * 2;
	}

private:
	uint32_t m_n_frequencies;
	uint32_t m_n_dims_to_encode;
	uint32_t m_n_dims_to_pass_through;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_padded_output_dims;
	uint32_t m_n_to_pad;
};

TCNN_NAMESPACE_END
