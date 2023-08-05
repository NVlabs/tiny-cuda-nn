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

/** @file   spherical_harmonics.h
 *  @author Alex Evans and Thomas Müller, NVIDIA
 *  @brief  Implementation of a spherical harmonics based frequency encoding.
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

namespace tcnn {

template <typename T>
__global__ void kernel_sh(
	const uint32_t num_elements,
	const uint32_t degree,
	const uint32_t num_to_pad,
	MatrixView<const float> data_in,
	MatrixView<T> data_out
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_out.advance_cols(i);

	TCNN_PRAGMA_UNROLL
	for (uint32_t j = 0; j < num_to_pad; ++j) {
		data_out(j) = (T)1.0f;
	}

	data_out.advance_rows(num_to_pad);

	sh_enc<T, MatrixView<T>>(
		degree,
		data_in(0, i) * 2.f - 1.f,
		data_in(1, i) * 2.f - 1.f,
		data_in(2, i) * 2.f - 1.f,
		data_out
	);
}

template <typename T>
__global__ void kernel_sh_backward(
	const uint32_t num_elements,
	const uint32_t degree,
	const uint32_t num_to_pad,
	MatrixView<const T> dL_dy,
	MatrixView<const float> data_in,
	MatrixView<float> dL_dx
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	dL_dy.advance(num_to_pad, i);

	vec3 d = sh_enc_grad<T, MatrixView<const T>>(
		degree,
		data_in(0, i) * 2.f - 1.f,
		data_in(1, i) * 2.f - 1.f,
		data_in(2, i) * 2.f - 1.f,
		dL_dy
	);

	// Multiplication by 2 due to the conversion
	// from [0,1]^3 to directions in [-1,1]^3.
	// See implementation in `kernel_sh`.
	dL_dx.set_col(i, 2.0f * d);
}

template <typename T>
class SphericalHarmonicsEncoding : public Encoding<T> {
public:
	SphericalHarmonicsEncoding(uint32_t degree, uint32_t n_dims_to_encode)
	: m_degree{degree}, m_n_dims_to_encode{n_dims_to_encode} {
		m_n_output_dims = degree * degree;

		if (n_dims_to_encode != 3) {
			throw std::runtime_error{"Can only encode 3D directions in spherical harmonics."};
		}

		if (m_degree <= 0) {
			throw std::runtime_error{"Spherical harmonics must have positive degree."};
		}

		if (m_degree > 8) {
			throw std::runtime_error{"Spherical harmonics are only implemented up to degree 8."};
		}
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

		linear_kernel(kernel_sh<T>, 0, stream,
			input.n(),
			m_degree,
			m_n_to_pad,
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
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override {
		if (!dL_dinput) {
			return;
		}

		linear_kernel(kernel_sh_backward<T>, 0, stream,
			input.n(),
			m_degree,
			m_n_to_pad,
			dL_doutput.view(),
			input.view(),
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
		return SoA;
	}

	json hyperparams() const override {
		return {
			{"otype", "SphericalHarmonics"},
			{"degree", m_degree},
		};
	}

private:
	uint32_t m_degree;
	uint32_t m_n_dims_to_encode;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_to_pad = 0;
};

}
