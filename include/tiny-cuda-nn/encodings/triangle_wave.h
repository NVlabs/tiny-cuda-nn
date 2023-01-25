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

/** @file   triangle_wave.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the triangle wave encoding of NRC [Müller et al. 2021].
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
__global__ void triangle_wave_encoding(
	const uint32_t num_elements,
	const uint32_t n_frequencies,
	const uint32_t num_to_encode,
	const uint32_t num_to_pad,
	MatrixView<const float> data_in,
	MatrixView<T> data_out,
	float* __restrict__ dy_dx)
{
	const uint32_t fan_out_encoded = num_to_encode * n_frequencies;
	const uint32_t fan_out = fan_out_encoded + num_to_pad;

	const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (encoded_index >= num_elements) return;

	const uint32_t i = encoded_index / fan_out;
	const uint32_t j = encoded_index - i * fan_out;

	if (j >= fan_out_encoded) {
		data_out(j, i) = 1;
	} else {
		const uint32_t encoded_input_feature_i = j / n_frequencies;
		const int log2_frequency = j - encoded_input_feature_i * n_frequencies;

		const float x = scalbnf(data_in(encoded_input_feature_i, i), log2_frequency-1);

		// Small log2_frequency-based phase shift to help disambiguate locations
		const float val = x + log2_frequency * 0.25f;
		const float result = fabsf(val - floorf(val) - 0.5f) * 4 - 1;

		data_out(j, i) = (T)result;
		if (dy_dx != nullptr) {
			dy_dx[i * fan_out_encoded + j] = scalbnf((int)floorf(val*2.0f) % 2 == 0 ? -1.0f : 1.0f, log2_frequency+1);
		}
	}
}

template <typename T>
__global__ void triangle_wave_encoding_backward(
	const uint32_t num_elements,
	const uint32_t n_dims_to_encode,
	const uint32_t n_frequencies,
	MatrixView<const T> dL_dy,
	const float* dy_dx,
	MatrixView<float> dL_dx
) {
	const uint32_t fan_out = n_dims_to_encode;

	const uint32_t outputs_per_input = n_frequencies;
	const uint32_t fan_out_encoded = n_dims_to_encode * outputs_per_input;

	const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (encoded_index >= num_elements) return;

	const uint32_t i = encoded_index / fan_out;
	const uint32_t j = encoded_index - i * fan_out;

	float result = 0;
	for (int k = 0; k < outputs_per_input; ++k) {
		result += (float)dL_dy(j * outputs_per_input + k, i) * dy_dx[i * fan_out_encoded + j * outputs_per_input + k];
	}
	dL_dx(j, i) = result;
}

template <typename T>
class TriangleWaveEncoding : public Encoding<T> {
public:
	TriangleWaveEncoding(uint32_t n_frequencies, uint32_t n_dims_to_encode)
	: m_n_frequencies{n_frequencies}, m_n_dims_to_encode{n_dims_to_encode} {
		m_n_output_dims = m_n_dims_to_encode * m_n_frequencies;
	}

	std::unique_ptr<Context> forward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		auto forward = std::make_unique<ForwardContext>();

		if (!output || padded_output_width() == 0) {
			return forward;
		}

		if (prepare_input_gradients) {
			forward->dy_dx = GPUMatrix<float>{m_n_dims_to_encode * m_n_frequencies, input.n(), stream};
		}

		linear_kernel(triangle_wave_encoding<T>, 0, stream,
			input.n() * padded_output_width(),
			m_n_frequencies,
			m_n_dims_to_encode,
			m_n_to_pad,
			input.view(),
			output->view(),
			forward->dy_dx.data()
		);

		return forward;
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

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		linear_kernel(triangle_wave_encoding_backward<T>, 0, stream,
			input.n() * m_n_dims_to_encode,
			m_n_dims_to_encode,
			m_n_frequencies,
			dL_doutput.view(),
			forward.dy_dx.data(),
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
			{"otype", "TriangleWave"},
			{"n_frequencies", m_n_frequencies},
		};
	}

private:
	struct ForwardContext : public Context {
		GPUMatrix<float> dy_dx;
	};

	uint32_t m_n_frequencies;
	uint32_t m_n_dims_to_encode;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_to_pad = 0;
};

TCNN_NAMESPACE_END
