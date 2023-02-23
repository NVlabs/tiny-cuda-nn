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

/** @file   sequential.h
 *  @brief  The sequential encoding allows serially connecting multiple encodings 
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
class SequentialEncoding : public Encoding<T> {
public:
	SequentialEncoding(const json& params, uint32_t n_dims_to_encode)
	: m_n_dims_to_encode{n_dims_to_encode} {
		if (!params.contains("nested") || !params["nested"].is_array()) {
			throw std::runtime_error{"Must provide an array of nested encodings to SequentialEncoding."};
		}
		const json::array_t& nested = params["nested"];
		for (size_t i = 0; i < nested.size(); ++i) {
			uint32_t nested_n_dims_to_encode = i == 0 ? n_dims_to_encode : m_nested.back()->output_width();
			m_nested.emplace_back(create_encoding<T>(nested_n_dims_to_encode, params["nested"][i]));
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
		forward->nested.resize(m_nested.size());
		forward->nested_outputs.resize(m_nested.size()-1);
		forward->nested_outputs_full_precision.resize(m_nested.size()-1);

		for (size_t i = 0; i < m_nested.size(); ++i) {
			const auto& nested = m_nested[i];

			bool first_layer = i == 0;
			bool last_layer = i == m_nested.size() - 1;

			if(!last_layer) {
				uint32_t padded_output_width = nested->padded_output_width();
				forward->nested_outputs[i] = GPUMatrixDynamic<T>(padded_output_width, input.n(), stream, nested->preferred_output_layout());
				forward->nested_outputs_full_precision[i] = GPUMatrixDynamic<float>(padded_output_width, input.n(), stream, nested->preferred_output_layout());
			}

			uint32_t input_width = nested->input_width();
			forward->nested[i] = nested->forward(
				stream, 
				first_layer ? input.slice_rows(0, input_width) : ((forward->nested_outputs_full_precision[i-1]).slice_rows(0, input_width)),
				last_layer ? output : &(forward->nested_outputs[i]),
				use_inference_params,
				first_layer ? prepare_input_gradients : true
			);

			if(!last_layer) {
				linear_kernel(cast_from<T>, 0, stream, forward->nested_outputs[i].n_elements(), forward->nested_outputs[i].data(), forward->nested_outputs_full_precision[i].data());
			}
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
			throw std::runtime_error{"SequentialEncoding::backward called with incompatible context size."};
		}

		GPUMatrixDynamic<T> nested_dL_doutput;
		GPUMatrixDynamic<float> tmp_dL_dinput;
		GPUMatrixDynamic<float> tmp_dL_dinput_sliced;

		for (int i = m_nested.size() - 1; i > -1 ; --i) {
			const auto& nested = m_nested[i];
			uint32_t input_width = nested->input_width();

			bool first_layer = i == 0;
			bool last_layer = i == m_nested.size() - 1;

			if(!first_layer) {
				tmp_dL_dinput = GPUMatrixDynamic<float>(m_nested[i-1]->padded_output_width(), input.n(), stream, m_nested[i-1]->preferred_output_layout());
				tmp_dL_dinput_sliced = tmp_dL_dinput.slice_rows(0, input_width); 
			}

			uint32_t padded_output_width = nested->padded_output_width();
			nested->backward(
				stream,
				*forward.nested[i],
				first_layer ? input.slice_rows(0, input_width) : forward.nested_outputs_full_precision[i-1].slice_rows(0, input_width),
				last_layer ? output.alias() : forward.nested_outputs[i].slice_rows(0, padded_output_width),
				last_layer ? dL_doutput.alias() : nested_dL_doutput.slice_rows(0, padded_output_width),
				first_layer ? dL_dinput : &tmp_dL_dinput_sliced,
				use_inference_params,
				param_gradients_mode
			);

			if(!first_layer) {
				nested_dL_doutput = GPUMatrixDynamic<T>(tmp_dL_dinput.m(), tmp_dL_dinput.n(), stream, m_nested[i-1]->preferred_output_layout());
				linear_kernel(cast<T>, 0, stream, nested_dL_doutput.n_elements(), tmp_dL_dinput.data(), nested_dL_doutput.data());
			}
		}
	}

	uint32_t input_width() const override {
		return m_n_dims_to_encode;
	}

	uint32_t padded_output_width() const override {
		return m_nested.back()->padded_output_width();
	}

	uint32_t output_width() const override {
		return m_nested.back()->output_width();
	}

	uint32_t required_input_alignment() const override {
		return m_nested.front()->required_input_alignment();
	}

	void set_padded_output_width(uint32_t padded_output_width) override {
        m_nested.back()->set_padded_output_width(padded_output_width);
	}

	uint32_t required_output_alignment() const override {
		return m_nested.back()->required_output_alignment();
	}

	MatrixLayout preferred_output_layout() const override {
		return m_nested.empty() ? AoS : m_nested.back()->preferred_output_layout();
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
			{"otype", "Sequential"},
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
		std::vector<GPUMatrixDynamic<float>> nested_outputs_full_precision;
	};


	std::vector<std::shared_ptr<Encoding<T>>> m_nested;
	uint32_t m_n_dims_to_encode;
};
TCNN_NAMESPACE_END