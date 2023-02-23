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

/** @file   fork.h
 *  @brief  The fork encoding takes an input and applied multiple encodings to it which are then concatenated.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void add_gradients_in_views(
	const uint32_t n_elements,
	MatrixView<const T> data_in_1,
	MatrixView<const T> data_in_2,
	const uint32_t n_rows,
	MatrixView<T> data_out
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	uint32_t row = i % n_rows; 
	uint32_t col = i / n_rows;
	data_out(row, col) = data_in_1(row, col) + data_in_2(row, col);
}

template <typename T>
__global__ void assemble(
	const uint32_t n_elements,
	MatrixView<const T> data_in,
	const uint32_t padded_dim,
	const uint32_t dim,
	MatrixView<T> data_out,
	const uint32_t offset
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	uint32_t row = i % padded_dim; 
	if(row >= dim) return;  // do nothing with the pads
	uint32_t col = i / padded_dim;
	data_out(row+offset, col) = data_in(row, col);
}

template <typename T>
__global__ void extract(
	const uint32_t n_elements,
	MatrixView<T> data_out,
	const uint32_t padded_dim,
	const uint32_t dim,
	MatrixView<const T> data_in,
	const uint32_t offset
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	uint32_t row = i % padded_dim; 
	if(row >= dim) return;  // do nothing with the pads
	uint32_t col = i / padded_dim;
	data_out(row, col) = data_in(row + offset, col);
}

template <typename T>
class ForkEncoding : public Encoding<T> {
public:
	ForkEncoding(const json& params, uint32_t n_dims_to_encode)
	: m_n_dims_to_encode{n_dims_to_encode} {
		if (!params.contains("nested") || !params["nested"].is_array()) {
			throw std::runtime_error{"Must provide an array of nested encodings to SequentialEncoding."};
		}
		const json::array_t& nested = params["nested"];
		for (size_t i = 0; i < nested.size(); ++i) {
			m_nested.emplace_back(create_encoding<T>(n_dims_to_encode, params["nested"][i]));
		}

			uint32_t dims_encoded_so_far = 0;
			for (size_t i = 0; i < m_nested.size()-1; ++i) {
				uint32_t desired_alignment = m_nested[i+1]->required_output_alignment();
				uint32_t padded_output_width_required = next_multiple(dims_encoded_so_far + m_nested[i]->output_width(), desired_alignment) - dims_encoded_so_far;

				m_nested[i]->set_padded_output_width(padded_output_width_required);

				dims_encoded_so_far += m_nested[i]->padded_output_width();
			}

		m_n_output_dims = 0;
		for(uint32_t i = 0; i < m_nested.size(); ++i) {
			m_n_output_dims += m_nested[i]->output_width();
		}
	}

	std::unique_ptr<Context> forward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		if (m_n_dims_to_encode == 0) {
			return std::make_unique<ForwardContext>();
		}

		auto forward = std::make_unique<ForwardContext>();
		uint32_t batch_size = input.n();

		forward->nested.resize(m_nested.size());
		forward->nested_outputs.resize(m_nested.size());

		uint32_t output_start = 0;

		for (size_t i = 0; i < m_nested.size(); ++i) {
			const auto& nested = m_nested[i];

			GPUMatrixDynamic<T> sliced_output;
			if(output != nullptr) {
				forward->nested_outputs[i] = GPUMatrixDynamic<T>(nested->padded_output_width(), batch_size, stream, nested->preferred_output_layout());
				// sliced_output = output->slice_rows(output_start, nested->padded_output_width());
			}
			
			forward->nested[i] = nested->forward(
				stream, 
				input.slice_rows(0, input_width()),
				output ? &(forward->nested_outputs[i]) : nullptr,
				// output ? &(sliced_output) : nullptr,
				use_inference_params,
				prepare_input_gradients
			);

			if(output != nullptr) {
				CHECK_THROW(output->rows() >= forward->nested_outputs[i].rows());
				linear_kernel(assemble<T>, 0, stream, 
					forward->nested_outputs[i].n_elements(), 
					forward->nested_outputs[i].view(), 
					nested->padded_output_width(),
					nested->output_width(),
					output->view(),
					output_start);
			}

			output_start = output_start + nested->output_width(); 
		}

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
		if (m_n_dims_to_encode == 0) {
			return;
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		if (forward.nested.size() != m_nested.size()) {
			throw std::runtime_error{"ForkEncoding::backward called with incompatible context size."};
		}

		uint32_t batch_size = input.cols();


		uint32_t output_start = 0;
		for (size_t i = 0; i < m_nested.size(); ++i) {

			const auto& nested = m_nested[i];

			tcnn::GPUMatrixDynamic<float> dL_di;
			if(dL_dinput != nullptr)
				dL_di = tcnn::GPUMatrixDynamic<float>{dL_dinput->rows(), dL_dinput->cols(), stream, dL_dinput->layout()}; 

			auto dL_do = tcnn::GPUMatrixDynamic<T>{nested->padded_output_width(), batch_size, stream, nested->preferred_output_layout()}; 

			// GPUMatrixDynamic<T> sliced_output = output.slice_rows(output_start, nested->padded_output_width());

			linear_kernel(extract<T>, 0, stream, 
				dL_do.n_elements(), 
				dL_do.view(), 
				nested->padded_output_width(),
				nested->output_width(),
				dL_doutput.view(),
				output_start);

			nested->backward(
				stream,
				*forward.nested[i],
				input.slice_rows(0, nested->input_width()),
				forward.nested_outputs[i],
				// sliced_output,
				dL_doutput.slice_rows(output_start, nested->padded_output_width()),
				dL_dinput == nullptr ? nullptr : &dL_di,
				use_inference_params,
				param_gradients_mode
			);

			// if(dL_dinput != nullptr && i == 0) {
			if(dL_dinput != nullptr) {
				// add the gradients from this nested encoding to the total gradients
				linear_kernel(add_gradients_in_views<float>, 0, stream, 
					dL_di.n_elements(), 
					dL_di.view(),
					dL_dinput->view(),
					dL_di.rows(),
					dL_dinput->view());
			}
			output_start = output_start + nested->output_width();
		}
	}

	uint32_t input_width() const override {
		return m_n_dims_to_encode;
	}

	uint32_t padded_output_width() const override {
		// return m_n_output_dims + m_n_to_pad;
		return output_width() + m_n_to_pad; 
	}

	uint32_t output_width() const override {
		// return m_n_output_dims;
		uint32_t output_width = 0;
		for(auto& nested : m_nested) {
			output_width += nested->output_width();
		}
		return output_width;
	}

	uint32_t required_input_alignment() const override {
		return 1; 
		// return m_nested.front()->required_input_alignment();
	}

	void set_padded_output_width(uint32_t padded_output_width) override {
		// for(auto& nested : m_nested) {
		// 	nested->set_padded_output_width(padded_output_width);
		// }
		// m_nested.front()->set_padded_output_width(padded_output_width);
		// CHECK_THROW(padded_output_width >= m_n_output_dims);
		m_n_to_pad = padded_output_width - output_width();
	}

	uint32_t required_output_alignment() const override {
		uint32_t alignment = 1;
		for (const auto& nested : m_nested) {
			alignment = lcm(alignment, nested->required_output_alignment());
		}
		return alignment;
	}

	MatrixLayout preferred_output_layout() const override {
		// Output layout of first nested encoding (tends to be the most significant, i.e. hash encoding)
		return m_nested.empty() ? AoS : m_nested.front()->preferred_output_layout();
	}

	void set_params_impl(T* params, T* inference_params, T* gradients) override {
		size_t offset = 0;
		for (auto& nested : m_nested) {
			nested->set_params(params + offset, inference_params + offset, gradients + offset);
			offset += nested->n_params();
		}
	}

	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
		for (auto& nested : m_nested) {
			nested->initialize_params(rnd, params_full_precision, scale);
			params_full_precision += nested->n_params();
		}
	}

	size_t n_params() const override {
		size_t total = 0;
		for (const auto& nested : m_nested) {
			total += nested->n_params();
		}
		return total;
	}

	json hyperparams() const override {
		json::array_t nested;
		for (auto& n : m_nested) {
			nested.emplace_back(n->hyperparams());
		}

		return {
			{"otype", "Fork"},
			{"nested", nested}
		};
	}

	size_t n_nested() const override {
		return m_nested.size();
	}

	const std::shared_ptr<Encoding<T>>& nested(size_t idx = 0) const {
		CHECK_THROW(idx < m_nested.size());
		return m_nested[idx];
	}

private:
	struct ForwardContext : public Context {
		std::vector<std::shared_ptr<Context>> nested;
		std::vector<GPUMatrixDynamic<T>> nested_outputs;
	};

	std::vector<std::shared_ptr<Encoding<T>>> m_nested;

	uint32_t m_n_dims_to_encode;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_to_pad = 0;
};
TCNN_NAMESPACE_END