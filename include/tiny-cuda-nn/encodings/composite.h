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

/** @file   composite.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  The composite encoding allows applying different, nested encodings
 *          to different dimensions of the input.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/multi_stream.h>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

template <typename T>
class CompositeEncoding : public Encoding<T> {
public:
	CompositeEncoding(const json& params, uint32_t n_dims_to_encode)
	: m_n_dims_to_encode{n_dims_to_encode} {
		if (!params.contains("nested") || !params["nested"].is_array()) {
			throw std::runtime_error{"Must provide an array of nested encodings to CompositeEncoding."};
		}

		const json::array_t& nested = params["nested"];

		uint32_t total_nested_dims_to_encode = 0;
		for (size_t i = 0; i < nested.size(); ++i) {
			total_nested_dims_to_encode += nested[i].value("n_dims_to_encode", 0);
			if (nested[i].contains("dims_to_encode_begin")) {
				total_nested_dims_to_encode = 0xFFFFFFFF;
				break;
			}
		}

		if (total_nested_dims_to_encode != 0xFFFFFFFF && total_nested_dims_to_encode > n_dims_to_encode) {
			throw std::runtime_error{"CompositeEncoding: nested encodings must not encode more dims than composite"};
		}

		uint32_t unspecified_dims_to_encode = total_nested_dims_to_encode == 0xFFFFFFFF ? 0xFFFFFFFF : (n_dims_to_encode - total_nested_dims_to_encode);
		uint32_t offset = 0;

		// Create encodings with somewhat arbitrary alignment
		for (size_t i = 0; i < nested.size(); ++i) {
			uint32_t nested_dims_to_encode;
			if (nested[i].contains("n_dims_to_encode")) {
				if (nested[i].contains("dims_to_encode_begin")) {
					offset = nested[i]["dims_to_encode_begin"];
				}

				nested_dims_to_encode = nested[i]["n_dims_to_encode"];
			} else {
				if (unspecified_dims_to_encode == 0xFFFFFFFF) {
					throw std::runtime_error{"CompositeEncoding: may only leave 'n_dims_to_encode' unspecified for a single nested encoding"};
				}
				nested_dims_to_encode = unspecified_dims_to_encode;
				unspecified_dims_to_encode = 0xFFFFFFFF;
			}

			if (nested_dims_to_encode > 0) {
				m_nested.emplace_back(create_encoding<T>(nested_dims_to_encode, nested[i], 1));
				m_dims_to_encode_begin.emplace_back(offset);
			}

			offset += nested_dims_to_encode;
		}

		// Fix alignment such that min_alignment of each individual encoding's output is ensured
		uint32_t dims_encoded_so_far = 0;
		for (size_t i = 0; i < m_nested.size()-1; ++i) {
			uint32_t desired_alignment = m_nested[i+1]->min_alignment();
			uint32_t effective_alignmen_needed = next_multiple(dims_encoded_so_far, desired_alignment) - dims_encoded_so_far;

			if (effective_alignmen_needed > 0) {
				m_nested[i]->set_alignment(effective_alignmen_needed);
			}

			dims_encoded_so_far += m_nested[i]->padded_output_width();
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

		SyncedMultiStream synced_streams{stream, m_nested.size()};

		uint32_t output_offset = 0;

		for (size_t i = 0; i < m_nested.size(); ++i) {
			const auto& nested = m_nested[i];
			uint32_t input_offset = m_dims_to_encode_begin[i];
			uint32_t input_width = nested->input_width();
			uint32_t output_width = nested->output_width();

			GPUMatrixDynamic<T> sliced_output;
			if (output) {
				sliced_output = output->slice_rows(output_offset, output_width);
			}

			forward->nested[i] = nested->forward(
				stream,
				input.slice_rows(input_offset, input_width),
				output ? &sliced_output : nullptr,
				use_inference_params,
				prepare_input_gradients
			);

			input_offset += input_width;
			output_offset += output_width;
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
			throw std::runtime_error{"CompositeEncoding::backward called with incompatible context size."};
		}

		SyncedMultiStream synced_streams{stream, m_nested.size()};

		uint32_t output_offset = 0;

		for (size_t i = 0; i < m_nested.size(); ++i) {
			const auto& nested = m_nested[i];
			uint32_t input_offset = m_dims_to_encode_begin[i];
			uint32_t input_width = nested->input_width();
			uint32_t output_width = nested->output_width();

			GPUMatrixDynamic<float> sliced_dL_dinput;
			if (dL_dinput) {
				sliced_dL_dinput = dL_dinput->slice_rows(input_offset, input_width);
			}

			nested->backward(
				synced_streams.get(i),
				*forward.nested[i],
				input.slice_rows(input_offset, input_width),
				output.slice_rows(output_offset, output_width),
				dL_doutput.slice_rows(output_offset, output_width),
				dL_dinput ? &sliced_dL_dinput : nullptr,
				use_inference_params,
				param_gradients_mode
			);

			output_offset += output_width;
		}
	}

	uint32_t input_width() const override {
		return m_n_dims_to_encode;
	}

	uint32_t padded_output_width() const override {
		uint32_t total = 0;
		for (const auto& nested : m_nested) {
			total += nested->padded_output_width();
		}
		return total;
	}

	uint32_t output_width() const override {
		uint32_t total = 0;
		for (const auto& nested : m_nested) {
			total += nested->output_width();
		}
		return total;
	}

	uint32_t required_input_alignment() const override {
		return 1;
	}

	void set_alignment(uint32_t alignment) override {
		uint32_t n_dims = padded_output_width();
		uint32_t last_n_dims = m_nested.back()->padded_output_width();

		uint32_t desired_n_dims = next_multiple(n_dims, alignment);
		m_nested.back()->set_alignment(desired_n_dims - (n_dims - last_n_dims));
	}

	uint32_t min_alignment() const override {
		return 1;
	}

	bool supports_output_layout(MatrixLayout layout) const {
		// Only supports layout if all nested encodings do
		bool result = true;
		for (const auto& nested : m_nested) {
			result &= nested->supports_output_layout(layout);
		}

		return result;
	}

	MatrixLayout preferred_output_layout() const override {
		// All encodings support AoS, so if any prefers AoS, use that.
		for (const auto& nested : m_nested) {
			if (nested->preferred_output_layout() == AoS) {
				return AoS;
			}
		}

		return SoA;
	}

	void initialize_params(pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		size_t offset = 0;
		for (auto& nested : m_nested) {
			nested->initialize_params(
				rnd,
				params_full_precision + offset,
				params + offset,
				inference_params + offset,
				backward_params + offset,
				gradients + offset,
				scale
			);
			offset += nested->n_params();
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
			{"otype", "Composite"},
			{"nested", nested}
		};
	}

private:
	struct ForwardContext : public Context {
		std::vector<std::unique_ptr<Context>> nested;
	};

	std::vector<std::unique_ptr<Encoding<T>>> m_nested;
	std::vector<uint32_t> m_dims_to_encode_begin;
	uint32_t m_n_dims_to_encode;

	MatrixLayout m_output_layout = AoS;
};

TCNN_NAMESPACE_END
