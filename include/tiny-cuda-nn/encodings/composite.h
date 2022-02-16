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
		}

		if (total_nested_dims_to_encode > n_dims_to_encode) {
			throw std::runtime_error{"CompositeEncoding:' nested encodings must not encode more dims than composite"};
		}

		uint32_t unspecified_dims_to_encode = n_dims_to_encode - total_nested_dims_to_encode;

		// Create encodings with somewhat arbitrary alignment
		for (size_t i = 0; i < nested.size(); ++i) {
			uint32_t nested_dims_to_encode;
			if (nested[i].contains("n_dims_to_encode")) {
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
			}
		}

		// Fix alignment such that min_alignment of each individual encoding's output is ensured
		uint32_t dims_encoded_so_far = 0;
		for (size_t i = 0; i < m_nested.size()-1; ++i) {
			uint32_t desired_alignment = m_nested[i+1]->min_alignment();
			uint32_t effective_alignmen_needed = next_multiple(dims_encoded_so_far, desired_alignment) - dims_encoded_so_far;

			if (effective_alignmen_needed > 0) {
				m_nested[i]->set_alignment(effective_alignmen_needed);
			}

			dims_encoded_so_far += m_nested[i]->num_encoded_dims();
		}
	}

	void encode(
		cudaStream_t stream,
		const uint32_t num_elements,
		PitchedPtr<const float> inputs,
		PitchedPtr<T> outputs,
		float* dy_dx = nullptr,
		bool is_inference = false
	) const override {
		if (m_n_dims_to_encode == 0) {
			return;
		}

		SyncedMultiStream synced_streams{stream, m_nested.size()};

		for (size_t i = 0; i < m_nested.size(); ++i) {
			const auto& nested = m_nested[i];
			nested->encode(synced_streams.get(i), num_elements, inputs, outputs, dy_dx, is_inference);

			inputs.ptr += nested->num_dims_to_encode();
			outputs.ptr += nested->num_encoded_dims();
			if (dy_dx) {
				dy_dx += nested->num_forward_gradient_dims() * num_elements;
			}
		}
	}

	void backward(
		cudaStream_t stream,
		const uint32_t num_elements,
		PitchedPtr<const T> dL_dy, // Same shape as outputs
		const float* dy_dx, // encoded output dims x num_elements
		PitchedPtr<float> dL_dx, // Same shape as inputs
		PitchedPtr<const float> inputs,
		bool accumulate_param_gradients,
		bool compute_param_gradients
	) override {
		if (m_n_dims_to_encode == 0) {
			return;
		}

		SyncedMultiStream synced_streams{stream, m_nested.size()};

		for (size_t i = 0; i < m_nested.size(); ++i) {
			const auto& nested = m_nested[i];
			nested->backward(synced_streams.get(i), num_elements, dL_dy, dy_dx, dL_dx, inputs, accumulate_param_gradients, compute_param_gradients);

			dL_dy.ptr += nested->num_encoded_dims();
			if (dy_dx) {
				dy_dx += nested->num_forward_gradient_dims() * num_elements;
			}
			if (dL_dx) {
				dL_dx.ptr += nested->num_dims_to_encode();
			}
			if (inputs) {
				inputs.ptr += nested->num_dims_to_encode();
			}
		}
	}

	uint32_t num_dims_to_encode() const override {
		return m_n_dims_to_encode;
	}

	uint32_t num_encoded_dims() const override {
		uint32_t total = 0;
		for (const auto& nested : m_nested) {
			total += nested->num_encoded_dims();
		}
		return total;
	}

	uint32_t num_forward_gradient_dims() const override {
		uint32_t total = 0;
		for (const auto& nested : m_nested) {
			total += nested->num_forward_gradient_dims();
		}
		return total;
	}

	void set_alignment(uint32_t alignment) override {
		uint32_t n_dims = num_encoded_dims();
		uint32_t last_n_dims = m_nested.back()->num_encoded_dims();

		uint32_t desired_n_dims = next_multiple(n_dims, alignment);
		m_nested.back()->set_alignment(desired_n_dims - (n_dims - last_n_dims));
	}

	uint32_t min_alignment() const override {
		return 1;
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

private:
	std::vector<std::unique_ptr<Encoding<T>>> m_nested;
	uint32_t m_n_dims_to_encode;
};

TCNN_NAMESPACE_END
