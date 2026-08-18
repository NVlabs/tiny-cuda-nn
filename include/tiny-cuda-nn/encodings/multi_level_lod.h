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
 *  @brief  Hard or soft per-element level control for multi-level encodings.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/encodings/multi_level_interface.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tcnn {

template <typename T>
__global__ void kernel_lod_copy(
	const uint32_t n_elements,
	const uint32_t n_levels,
	const uint32_t n_features_per_level,
	const uint32_t padded_output_width,
	const bool apply_lod_weights,
	const float* __restrict__ levels,
	MatrixView<const T> input,
	MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	if (!apply_lod_weights) {
		for (uint32_t feature = 0; feature < padded_output_width; ++feature) {
			output(feature, i) = input(feature, i);
		}
		return;
	}

	const float level_f = levels[i] * n_levels + 1e-3f;
	const bool all_levels_active = level_f >= (float)n_levels;
	const bool no_levels_active = level_f < 0.0f;
	const int32_t level_i = all_levels_active || no_levels_active ? 0 : (int32_t)floorf(level_f);
	for (uint32_t level = 0; level < n_levels; ++level) {
		float weight = 0.0f;
		if (all_levels_active || (!no_levels_active && (int32_t)level < level_i)) {
			weight = 1.0f;
		} else if (!no_levels_active && (int32_t)level == level_i) {
			weight = level_f - level_i;
		}

		for (uint32_t feature = 0; feature < n_features_per_level; ++feature) {
			const uint32_t output_feature = level * n_features_per_level + feature;
			output(output_feature, i) = weight == 0.0f ? (T)0.0f : (T)weight * input(output_feature, i);
		}
	}

	for (uint32_t feature = n_levels * n_features_per_level; feature < padded_output_width; ++feature) {
		output(feature, i) = input(feature, i);
	}
}

template <typename T> class MultiLevelEncodingLoD : public MultiLevelEncoding<T> {
public:
	MultiLevelEncodingLoD(uint32_t n_dims_to_encode, const json& encoding) {
		if (n_dims_to_encode <= 1) {
			throw std::runtime_error{"MultiLevelEncodingLoD requires at least two input dimensions."};
		}

		const std::string lod_type = encoding.value("lod_type", "Hard");
		if (equals_case_insensitive(lod_type, "Hard") || equals_case_insensitive(lod_type, "Discontinuous")) {
			m_is_soft = false;
		} else if (equals_case_insensitive(lod_type, "Soft") || equals_case_insensitive(lod_type, "Continuous")) {
			m_is_soft = true;
		} else {
			throw std::runtime_error{"MultiLevelEncodingLoD: lod_type must be Hard, Discontinuous, Soft, or Continuous."};
		}

		const json& base_config = encoding.at("base");
		const std::string base_type = base_config.value("otype", "");
		if (equals_case_insensitive(base_type, "MultiLevelEncodingLoD")) {
			throw std::runtime_error{"MultiLevelEncodingLoD cannot wrap another MultiLevelEncodingLoD."};
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
		parallel_for_gpu(stream, n_elements, [
			input_view = input.view(),
			levels = forward->levels.data(),
			n_pos_dims,
			max_level = this->m_max_level,
			max_level_gpu = this->m_max_level_gpu
		] __device__(size_t i) {
			levels[i] = fminf(input_view(n_pos_dims, i), max_level_gpu ? max_level_gpu[i] : max_level);
		});

		GPUMatrixDynamic<T>* base_output = output;
		const MatrixLayout base_layout = m_base->preferred_output_layout();
		if (m_is_soft || (output && (!output->is_contiguous() || output->layout() != base_layout))) {
			forward->base_output = GPUMatrixDynamic<T>{
				padded_output_width(), n_elements, stream, base_layout
			};
			base_output = &forward->base_output;
		}

		{
			float* previous_max_level_gpu = m_base->max_level_gpu();
			m_base->set_max_level_gpu(forward->levels.data());
			ScopeGuard restore_max_level_gpu{[this, previous_max_level_gpu] { m_base->set_max_level_gpu(previous_max_level_gpu); }};
			forward->base = m_base->forward(stream, positions, base_output, use_inference_params, prepare_input_gradients);
		}

		if (output && forward->base_output.data()) {
			copy_with_lod_weights(
				stream, n_elements, m_is_soft, forward->levels.data(), forward->base_output.view(), output->view()
			);
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

		GPUMatrixDynamic<T> compact_dL_doutput;
		const GPUMatrixDynamic<T>* base_dL_doutput = &dL_doutput;
		const MatrixLayout base_layout = m_base->preferred_output_layout();
		if (m_is_soft || !dL_doutput.is_contiguous() || dL_doutput.layout() != base_layout) {
			compact_dL_doutput = GPUMatrixDynamic<T>{dL_doutput.m(), dL_doutput.n(), stream, base_layout};
			copy_with_lod_weights(
				stream, n_elements, m_is_soft, forward.levels.data(), dL_doutput.view(), compact_dL_doutput.view()
			);
			base_dL_doutput = &compact_dL_doutput;
		}
		const GPUMatrixDynamic<T>& base_output = forward.base_output.data() ? forward.base_output : output;

		{
			float* previous_max_level_gpu = m_base->max_level_gpu();
			m_base->set_max_level_gpu(forward.levels.data());
			ScopeGuard restore_max_level_gpu{[this, previous_max_level_gpu] { m_base->set_max_level_gpu(previous_max_level_gpu); }};
			m_base->backward(
				stream,
				*forward.base,
				positions,
				base_output,
				*base_dL_doutput,
				dL_dinput ? &dL_dpositions : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}

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

		GPUMatrixDynamic<T> compact_dL_doutput;
		const GPUMatrixDynamic<T>* base_dL_doutput = &dL_doutput;
		const MatrixLayout base_layout = m_base->preferred_output_layout();
		if (m_is_soft || !dL_doutput.is_contiguous() || dL_doutput.layout() != base_layout) {
			compact_dL_doutput = GPUMatrixDynamic<T>{dL_doutput.m(), dL_doutput.n(), stream, base_layout};
			copy_with_lod_weights(
				stream, n_elements, m_is_soft, forward.levels.data(), dL_doutput.view(), compact_dL_doutput.view()
			);
			base_dL_doutput = &compact_dL_doutput;
		}

		{
			float* previous_max_level_gpu = m_base->max_level_gpu();
			m_base->set_max_level_gpu(forward.levels.data());
			ScopeGuard restore_max_level_gpu{[this, previous_max_level_gpu] { m_base->set_max_level_gpu(previous_max_level_gpu); }};
			m_base->backward_backward_input(
				stream,
				*forward.base,
				positions,
				dL_ddLdpositions,
				*base_dL_doutput,
				dL_ddLdoutput,
				dL_dinput ? &dL_dpositions : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}

		if (m_is_soft && dL_ddLdoutput) {
			copy_with_lod_weights(
				stream, n_elements, true, forward.levels.data(), dL_ddLdoutput->view(), dL_ddLdoutput->view()
			);
		}

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

	uint32_t n_pos_dims() const override { return m_base->n_pos_dims(); }

	uint32_t n_levels() const override { return m_base->n_levels(); }

	uint32_t n_features_per_level() const override { return m_base->n_features_per_level(); }

	size_t level_n_params(uint32_t level) const override { return m_base->level_n_params(level); }

	size_t level_params_offset(uint32_t level) const override { return m_base->level_params_offset(level); }

	const ParamsOffsetTable& params_offset_table() const override { return m_base->params_offset_table(); }

	json hyperparams() const override {
		return {
			{"otype",    "MultiLevelEncodingLoD"},
			{"lod_type", m_is_soft ? "Soft" : "Hard"},
			{"base",     m_base->hyperparams()  },
		};
	}

	std::string generate_device_function(const std::string& name) const override {
		if (this->m_max_level != 1000.0f || this->m_max_level_gpu) {
			throw std::runtime_error{"MultiLevelEncodingLoD: generated forward does not support nondefault max-level state."};
		}

		const std::string base_name = fmt::format("{}_base", name);
		const uint32_t base_input_width = m_base->input_width();
		std::ostringstream body;
		body << fmt::format(
			"\tauto result = {BASE}(input.slice<0, {BASE_INPUT_WIDTH}>(), params, nullptr);\n"
			"\tconst float level_f = input[{BASE_INPUT_WIDTH}] * {N_LEVELS} + 1e-3f;\n",
			"BASE"_a = base_name,
			"BASE_INPUT_WIDTH"_a = base_input_width,
			"N_LEVELS"_a = n_levels()
		);

		if (m_is_soft) {
			body << dfmt(1, R"(
				if (level_f < 0.0f) {{
					TCNN_PRAGMA_UNROLL
					for (uint32_t output_idx = 0; output_idx < {N_LEVELS} * {N_FEATURES_PER_LEVEL}; ++output_idx) {{
						result[output_idx] = ({T})0.0f;
					}}
				}} else if (level_f < (float){N_LEVELS}) {{
					const int32_t level_i = (int32_t)floor(level_f);
					const float weight = level_f - level_i;
					TCNN_PRAGMA_UNROLL
					for (uint32_t level = 0; level < {N_LEVELS}; ++level) {{
						TCNN_PRAGMA_UNROLL
						for (uint32_t feature = 0; feature < {N_FEATURES_PER_LEVEL}; ++feature) {{
							const uint32_t output_idx = level * {N_FEATURES_PER_LEVEL} + feature;
							if ((int32_t)level > level_i || ((int32_t)level == level_i && weight == 0.0f)) {{
								result[output_idx] = ({T})0.0f;
							}} else if ((int32_t)level == level_i && weight != 1.0f) {{
								result[output_idx] *= ({T})weight;
							}}
						}}
					}}
				}}
			)",
				"N_LEVELS"_a = n_levels(),
				"N_FEATURES_PER_LEVEL"_a = n_features_per_level(),
				"T"_a = type_to_string<T>()
			);
		} else {
			body << dfmt(1, R"(
				TCNN_PRAGMA_UNROLL
				for (uint32_t level = 0; level < {N_LEVELS}; ++level) {{
					if ((float)level >= level_f) {{
						TCNN_PRAGMA_UNROLL
						for (uint32_t feature = 0; feature < {N_FEATURES_PER_LEVEL}; ++feature) {{
							result[level * {N_FEATURES_PER_LEVEL} + feature] = ({T})0.0f;
						}}
					}}
				}}
			)",
				"N_LEVELS"_a = n_levels(),
				"N_FEATURES_PER_LEVEL"_a = n_features_per_level(),
				"T"_a = type_to_string<T>()
			);
		}

		body << "\treturn result;";
		return fmt::format("{}\n\n{}", m_base->generate_device_function(base_name), this->generate_device_function_from_body(name, body.str()));
	}

	bool device_function_fwd_ctx_aligned_per_element() const override { return false; }

private:
	void copy_with_lod_weights(
		cudaStream_t stream,
		uint32_t n_elements,
		bool apply_lod_weights,
		const float* levels,
		MatrixView<const T> input,
		MatrixView<T> output
	) const {
		linear_kernel(
			kernel_lod_copy<T>,
			0,
			stream,
			n_elements,
			n_levels(),
			n_features_per_level(),
			padded_output_width(),
			apply_lod_weights,
			levels,
			input,
			output
		);
	}

	struct ForwardContext : public Context {
		GPUMatrixDynamic<float> levels;
		GPUMatrixDynamic<T> base_output;
		std::unique_ptr<Context> base;
	};

	std::shared_ptr<MultiLevelEncoding<T>> m_base;
	bool m_is_soft = false;
};

} // namespace tcnn
