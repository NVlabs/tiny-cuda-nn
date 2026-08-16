/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   multi_level_lod.h
 *  @brief  Hard per-element level control for multi-level encodings.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/encodings/multi_level_interface.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tcnn {

template <typename T> class MultiLevelEncodingLoD : public Encoding<T> {
public:
	MultiLevelEncodingLoD(uint32_t n_dims_to_encode, const json& encoding) {
		if (n_dims_to_encode <= 1) {
			throw std::runtime_error{"MultiLevelEncodingLoD requires at least two input dimensions."};
		}

		const std::string lod_type = encoding.value("lod_type", "Hard");
		if (!equals_case_insensitive(lod_type, "Hard")) {
			throw std::runtime_error{"MultiLevelEncodingLoD only supports hard level control."};
		}

		const json& base_config = encoding.at("base");
		const std::string base_type = base_config.value("otype", "");
		if (!equals_case_insensitive(base_type, "Permuto")) {
			throw std::runtime_error{"MultiLevelEncodingLoD only supports a Permuto base encoding."};
		}

		std::shared_ptr<Encoding<T>> base{create_encoding<T>(n_dims_to_encode - 1, base_config, 1)};
		m_base = std::dynamic_pointer_cast<MultiLevelEncoding<T>>(base);
		if (!m_base) {
			throw std::runtime_error{"MultiLevelEncodingLoD requires a multi-level base encoding."};
		}

		m_base->set_alignment(m_base->required_output_alignment());
	}

#if !defined(TCNN_NO_FWD_BWD)
	std::unique_ptr<Context> forward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		auto forward = std::make_unique<ForwardContext>();
		const uint32_t n_elements = input.n();
		if (padded_output_width() == 0 || n_elements == 0) {
			return forward;
		}

		const uint32_t n_pos_dims = m_base->input_width();
		const auto positions = GPUMatrixDynamic<float>{input.data(), n_pos_dims, n_elements, input.layout(), input.stride()};

		forward->levels = GPUMatrixDynamic<float>{1, n_elements, stream};
		parallel_for_gpu(stream, n_elements, [input_view = input.view(), levels = forward->levels.data(), n_pos_dims] __device__(size_t i) {
			levels[i] = input_view(n_pos_dims, i);
		});

		float* previous_max_level_gpu = m_base->max_level_gpu();
		m_base->set_max_level_gpu(forward->levels.data());
		ScopeGuard restore_max_level_gpu{[this, previous_max_level_gpu] { m_base->set_max_level_gpu(previous_max_level_gpu); }};
		forward->base = m_base->forward(stream, positions, output, use_inference_params, prepare_input_gradients);
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
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override {
		const uint32_t n_elements = input.n();
		if (n_elements == 0) {
			if (param_gradients_mode == GradientMode::Overwrite && n_params() > 0) {
				CUDA_CHECK_THROW(cudaMemsetAsync(this->gradients(), 0, n_params() * sizeof(T), stream));
			}
			return;
		}

		if (padded_output_width() == 0 || (!dL_dinput && param_gradients_mode == GradientMode::Ignore)) {
			return;
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		const uint32_t n_pos_dims = m_base->input_width();
		const auto positions = GPUMatrixDynamic<float>{input.data(), n_pos_dims, n_elements, input.layout(), input.stride()};

		GPUMatrixDynamic<float> dL_dpositions;
		if (dL_dinput) {
			dL_dpositions = GPUMatrixDynamic<float>{dL_dinput->data(), n_pos_dims, n_elements, dL_dinput->layout(), dL_dinput->stride()};
		}

		float* previous_max_level_gpu = m_base->max_level_gpu();
		m_base->set_max_level_gpu(forward.levels.data());
		ScopeGuard restore_max_level_gpu{[this, previous_max_level_gpu] { m_base->set_max_level_gpu(previous_max_level_gpu); }};
		m_base->backward(
			stream, *forward.base, positions, output, dL_doutput, dL_dinput ? &dL_dpositions : nullptr, use_inference_params, param_gradients_mode
		);

		if (dL_dinput) {
			parallel_for_gpu(stream, n_elements, [gradient = dL_dinput->view(), n_pos_dims] __device__(size_t i) {
				gradient(n_pos_dims, i) = 0.0f;
			});
		}
	}

	void backward_backward_input_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<float>& dL_ddLdinput,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_ddLdoutput = nullptr,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override {
		const uint32_t n_elements = input.n();
		if (n_elements == 0) {
			if (param_gradients_mode == GradientMode::Overwrite && n_params() > 0) {
				CUDA_CHECK_THROW(cudaMemsetAsync(this->gradients(), 0, n_params() * sizeof(T), stream));
			}
			return;
		}
		if (padded_output_width() == 0 || (!dL_ddLdoutput && !dL_dinput && param_gradients_mode == GradientMode::Ignore)) {
			return;
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		const uint32_t n_pos_dims = m_base->input_width();
		const auto positions = GPUMatrixDynamic<float>{input.data(), n_pos_dims, n_elements, input.layout(), input.stride()};
		const auto dL_ddLdpositions = GPUMatrixDynamic<float>{
			dL_ddLdinput.data(), n_pos_dims, n_elements, dL_ddLdinput.layout(), dL_ddLdinput.stride()
		};

		GPUMatrixDynamic<float> dL_dpositions;
		if (dL_dinput) {
			dL_dpositions = GPUMatrixDynamic<float>{
				dL_dinput->data(), n_pos_dims, n_elements, dL_dinput->layout(), dL_dinput->stride()
			};
		}

		float* previous_max_level_gpu = m_base->max_level_gpu();
		m_base->set_max_level_gpu(forward.levels.data());
		ScopeGuard restore_max_level_gpu{[this, previous_max_level_gpu] { m_base->set_max_level_gpu(previous_max_level_gpu); }};
		m_base->backward_backward_input(
			stream,
			*forward.base,
			positions,
			dL_ddLdpositions,
			dL_doutput,
			dL_ddLdoutput,
			dL_dinput ? &dL_dpositions : nullptr,
			use_inference_params,
			param_gradients_mode
		);

		if (dL_dinput) {
			parallel_for_gpu(stream, n_elements, [gradient = dL_dinput->view(), n_pos_dims] __device__(size_t i) {
				gradient(n_pos_dims, i) = 0.0f;
			});
		}
	}
	#endif

	uint32_t input_width() const override { return m_base->input_width() + 1; }

	uint32_t padded_output_width() const override { return m_base->padded_output_width(); }

	uint32_t output_width() const override { return m_base->output_width(); }

	uint32_t required_input_alignment() const override { return 1; }

	void set_padded_output_width(uint32_t padded_output_width) override { m_base->set_padded_output_width(padded_output_width); }

	uint32_t required_output_alignment() const override { return m_base->required_output_alignment(); }

	MatrixLayout preferred_output_layout() const override { return m_base->preferred_output_layout(); }

	void set_params_impl(T* params, T* inference_params, T* gradients) override { m_base->set_params(params, inference_params, gradients); }

	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
		m_base->initialize_params(rnd, params_full_precision, scale);
	}

	size_t n_params() const override { return m_base->n_params(); }

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override { return m_base->layer_sizes(); }

	json hyperparams() const override {
		return {
			{"otype",    "MultiLevelEncodingLoD"},
			{"lod_type", "Hard"                 },
			{"base",     m_base->hyperparams()  },
		};
	}

private:
	struct ForwardContext : public Context {
		GPUMatrixDynamic<float> levels;
		std::unique_ptr<Context> base;
	};

	std::shared_ptr<MultiLevelEncoding<T>> m_base;
};

} // namespace tcnn
