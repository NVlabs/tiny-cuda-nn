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

/** @file   identity.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the identity encoding (output == input).
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

template <typename T>
__global__ void identity(
	const uint32_t num_outputs,
	const uint32_t num_to_encode,
	const uint32_t num_to_pad,
	const float scale,
	const float offset,
	MatrixView<const float> data_in,
	MatrixView<T> data_out)
{
	const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (encoded_index >= num_outputs) return;

	const uint32_t fan_out = num_to_encode + num_to_pad;
	const uint32_t i = encoded_index / fan_out;
	const uint32_t j = encoded_index - i * fan_out;

	if (j >= num_to_encode) {
		data_out(j, i) = 1;
	} else {
		data_out(j, i) = data_in(j, i) * scale + offset;
	}
}

template <typename T>
__global__ void identity_backward(
	const uint32_t num_outputs,
	const uint32_t n_dims_to_encode,
	const float scale,
	MatrixView<const T> dL_dy,
	MatrixView<float> dL_dx)
{
	const uint32_t output_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (output_index >= num_outputs) return;

	const uint32_t i = output_index / n_dims_to_encode;
	const uint32_t j = output_index - i * n_dims_to_encode;

	// The identity encoding can simply pass through the derivative.
	dL_dx(j, i) = (T)((float)dL_dy(j, i) * scale);
}

template <typename T>
class IdentityEncoding : public Encoding<T> {
public:
	IdentityEncoding(uint32_t n_dims_to_encode, float scale = 1.0f, float offset = 0.0f)
	: m_n_dims_to_encode{n_dims_to_encode}, m_scale{scale}, m_offset{offset} {
		m_n_output_dims = m_n_dims_to_encode;
	}

	std::unique_ptr<Context> forward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		if (!output || padded_output_width() == 0) {
			return std::make_unique<Context>();
		}

		linear_kernel(identity<T>, 0, stream,
			input.n() * padded_output_width(),
			m_n_dims_to_encode,
			m_n_to_pad,
			m_scale,
			m_offset,
			input.view(),
			output->view()
		);

		return std::make_unique<Context>();
	}

	void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		EGradientMode param_gradients_mode = EGradientMode::Overwrite
	) override {
		if (!dL_dinput || padded_output_width() == 0) {
			return;
		}

		linear_kernel(identity_backward<T>, 0, stream,
			input.n() * m_n_dims_to_encode,
			m_n_dims_to_encode,
			m_scale,
			dL_doutput.view(),
			dL_dinput->view()
		);
	}

	uint32_t input_width() const override {
		return m_n_dims_to_encode;
	}

	uint32_t padded_output_width() const override {
		return m_n_output_dims + m_n_to_pad;
	}

	uint32_t output_width() const override {
		return padded_output_width();
	}

	uint32_t required_input_alignment() const override {
		return 1;
	}

	void set_padded_output_width(uint32_t padded_output_width) override {
		CHECK_THROW(padded_output_width >= m_n_output_dims);
		m_n_to_pad = padded_output_width - m_n_output_dims;
	}

	uint32_t required_output_alignment() const override {
		return 1;
	}

	MatrixLayout preferred_output_layout() const override {
		return AoS;
	}

	json hyperparams() const override {
		return {
			{"otype", "Identity"},
			{"scale", m_scale},
			{"offset", m_offset},
		};
	}

private:
	uint32_t m_n_dims_to_encode;

	float m_scale;
	float m_offset;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_to_pad = 0;
};

TCNN_NAMESPACE_END
