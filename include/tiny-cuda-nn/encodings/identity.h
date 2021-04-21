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

/** @file   identity.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the identity encoding (output == input).
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
__global__ void identity(
	const uint32_t num_elements,
	const uint32_t num_to_encode,
	const uint32_t num_passthrough,
	const uint32_t num_to_pad,
	const float scale,
	const float offset,
	const float* __restrict__ data_in,
	T* __restrict__ data_out,
	float* __restrict__ dy_dx)
{
	const uint32_t fan_in_encoded = num_to_encode;
	const uint32_t fan_in = fan_in_encoded + num_passthrough;
	const uint32_t fan_out_encoded = num_to_encode;
	const uint32_t fan_out = num_to_encode + num_passthrough + num_to_pad;

	const uint32_t output_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (output_index >= num_elements) return;

	const uint32_t i = output_index / fan_out;
	const uint32_t j = output_index % fan_out;
	const uint32_t input_index = i * fan_in;

	if (j >= fan_out_encoded + num_passthrough) {
		data_out[output_index] = 1;
	} else if (j >= fan_out_encoded) {
		data_out[output_index] = data_in[input_index + fan_in_encoded + j - fan_out_encoded] * scale + offset;
	} else {
		data_out[output_index] = data_in[input_index + j] * scale + offset;
		if (dy_dx != nullptr) {
			dy_dx[i * fan_out_encoded + j] = 1.0f;
		}
	}
}

template <typename T>
__global__ void identity_backward(
	const uint32_t num_elements,
	const uint32_t n_dims_to_encode,
	const uint32_t n_dims_to_pass_through,
	const uint32_t n_padded_output_dims,
	const T* dL_dy,
	const float* dy_dx,
	float* dL_dx)
{
	const uint32_t fan_out = n_dims_to_encode + n_dims_to_pass_through;
	const uint32_t stride = n_padded_output_dims;

	const uint32_t output_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (output_index >= num_elements) return;

	const uint32_t i = output_index / fan_out;
	const uint32_t j = output_index % fan_out;

	// The identity encoding can simply pass through the derivative.
	dL_dx[output_index] = dL_dy[i * stride + j];
}

template <typename T>
class IdentityEncoding : public Encoding<T> {
public:
	IdentityEncoding(uint32_t n_dims_to_encode, uint32_t n_dims_to_pass_through, float scale = 1.0f, float offset = 0.0f, uint32_t alignment = 8)
	: m_n_dims_to_encode{n_dims_to_encode}, m_n_dims_to_pass_through{n_dims_to_pass_through}, m_scale{scale}, m_offset{offset} {
		m_n_output_dims = m_n_dims_to_encode + m_n_dims_to_pass_through;
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

		linear_kernel(identity<T>, 0, stream,
			num_elements * num_encoded_dims(),
			m_n_dims_to_encode,
			m_n_dims_to_pass_through,
			m_n_to_pad,
			m_scale,
			m_offset,
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
		linear_kernel(identity_backward<T>, 0, stream,
			num_elements * n_input_dims,
			m_n_dims_to_encode,
			m_n_dims_to_pass_through,
			m_n_padded_output_dims,
			dL_dy,
			dy_dx,
			dL_dx
		);
	}

	uint32_t num_encoded_dims() const override {
		return m_n_padded_output_dims;
	}

	uint32_t num_forward_gradient_dims() const override {
		return m_n_dims_to_encode;
	}

private:
	uint32_t m_n_dims_to_encode;
	uint32_t m_n_dims_to_pass_through;

	float m_scale;
	float m_offset;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_padded_output_dims;
	uint32_t m_n_to_pad;
};

TCNN_NAMESPACE_END
