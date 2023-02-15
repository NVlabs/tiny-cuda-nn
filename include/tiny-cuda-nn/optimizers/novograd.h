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

/** @file   novograd.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the novograd optimizer: https://arxiv.org/pdf/1905.11286.pdf
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/reduce_sum.h>

#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void novo_step(
	const uint32_t n_elements,
	const float relative_weight_decay,
	const float absolute_weight_decay,
	const float loss_scale,
	const float learning_rate,
	float beta1,
	const float epsilon,
	float* __restrict__ weights_full_precision,
	T* __restrict__ weights,
	const T* __restrict__ gradients,
	float* __restrict__ first_moments,
	const float* __restrict__ per_layer_second_moment
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const float weight_fp = weights_full_precision[i];
	float gradient = (float)gradients[i] / loss_scale;

	// Optional gradient clipping
	// static const float clip_value = 1.0f;
	// gradient = fminf(fmaxf(gradient, -clip_value), clip_value);

	const float first_moment = first_moments[i] = beta1 * first_moments[i] + (1 - beta1) * gradient / (sqrtf(*per_layer_second_moment) + epsilon);

	const float decayed_weight = weight_decay(relative_weight_decay * learning_rate, absolute_weight_decay * learning_rate, weight_fp);
	const float new_weight = decayed_weight - learning_rate * first_moment;

	weights_full_precision[i] = new_weight;
	weights[i] = (T)new_weight;
}

template <typename T>
__global__ void novo_update_per_layer_second_moment(
	const uint32_t n_elements,
	const float loss_scale,
	float beta2,
	const float* __restrict__ gradient_norm,
	float* __restrict__ per_layer_second_moment
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	*per_layer_second_moment = beta2 * *per_layer_second_moment + (1 - beta2) * *gradient_norm / loss_scale / loss_scale;
}

template <typename T>
class NovogradOptimizer : public Optimizer<T> {
public:
	NovogradOptimizer(const json& params) {
		update_hyperparams(params);
	}

	void allocate(uint32_t n_weights, const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
		m_n_weights = n_weights;

		if (m_n_weights <= m_first_moments.size()) {
			return;
		}

		m_first_moments.resize(m_n_weights);
		m_first_moments.memset(0);

		size_t total_size = 0;

		m_layers.clear();
		for (const auto& pair : layer_sizes) {
			m_layers.push_back(pair.first * pair.second);
			total_size += m_layers.back();
		}

		m_per_layer_second_moments.resize(m_layers.size());
		m_per_layer_second_moments.memset(0);
	}

	void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
		++m_current_step;

		uint32_t offset = 0;
		for (size_t i = 0; i < m_layers.size(); ++i) {
			uint32_t workspace_size = reduce_sum_workspace_size(m_layers[i]);

			if (workspace_size > m_reduction_workspace.size()) {
				workspace_size *= 2;
#ifdef TCNN_VERBOSE_MEMORY_ALLOCS
				std::cout << "NOVOGRAD: resizing reduction buffer to " << workspace_size << std::endl;
#endif
				m_reduction_workspace.resize(workspace_size);
			}

			reduce_sum(gradients + offset, [] __device__ (T val) { return (float)val * (float)val; }, m_reduction_workspace.data(), m_layers[i], stream);

			// Seems wasteful to invoke a kernel for updating a single float...
			// maybe this is potential for optimization in the future.
			linear_kernel(novo_update_per_layer_second_moment<T>, 0, stream,
				1u,
				loss_scale,
				m_current_step == 1 ? 0.0f : m_beta2, // Set exact value on first step
				m_reduction_workspace.data(),
				m_per_layer_second_moments.data() + i
			);

			linear_kernel(novo_step<T>, 0, stream,
				m_layers[i],
				m_relative_weight_decay,
				m_absolute_weight_decay,
				loss_scale,
				m_base_learning_rate,
				m_current_step == 1 ? 0.0f : m_beta1, // Set exact value on first step
				m_epsilon,
				weights_full_precision + offset,
				weights + offset,
				gradients + offset,
				m_first_moments.data() + offset,
				m_per_layer_second_moments.data() + i
			);

			offset += m_layers[i];
		}
	}

	float learning_rate() const override {
		return m_base_learning_rate;
	}

	void set_learning_rate(float val) override {
		m_base_learning_rate = val;
	}

	uint32_t step() const override {
		return m_current_step;
	}

	uint32_t n_weights() const override {
		return m_n_weights;
	}

	T* custom_weights() const override {
		return nullptr;
	}

	void update_hyperparams(const json& params) override {
		if (params.contains("beta1")) {
			m_beta1 = params["beta1"];
		}

		if (params.contains("beta2")) {
			m_beta2 = params["beta2"];
		}

		if (params.contains("epsilon")) {
			m_epsilon = params["epsilon"];
		}

		if (params.contains("learning_rate")) {
			m_base_learning_rate = params["learning_rate"];
		}

		if (params.contains("relative_decay")) {
			m_relative_weight_decay = params["relative_decay"];
		}

		if (params.contains("absolute_decay")) {
			m_absolute_weight_decay = params["absolute_decay"];
		}
	}

	json hyperparams() const override {
		return {
			{"otype", "Novograd"},
			{"beta1", m_beta1},
			{"beta2", m_beta2},
			{"epsilon", m_epsilon},
			{"learning_rate", m_base_learning_rate},
			{"relative_decay", m_relative_weight_decay},
			{"absolute_decay", m_absolute_weight_decay},
		};
	}

	json serialize() const override {
		json data;
		data["current_step"] = m_current_step;
		data["base_learning_rate"] = m_base_learning_rate;
		data["first_moments_binary"] = m_first_moments;
		data["per_layer_second_moments_binary"] = m_per_layer_second_moments;
		return data;
	}

	void deserialize(const json& data) override {
		m_first_moments = data["first_moments_binary"];
		m_per_layer_second_moments = data["per_layer_second_moments_binary"];
		m_current_step = data["current_step"];
		m_base_learning_rate = data["base_learning_rate"];
	}

private:
	uint32_t m_n_weights;

	GPUMemory<float> m_first_moments;
	GPUMemory<float> m_per_layer_second_moments;

	GPUMemory<float> m_reduction_workspace;

	std::vector<uint32_t> m_layers;

	uint32_t m_current_step = 0;

	// Hyperparameters
	float m_base_learning_rate = 1e-3f;
	float m_beta1 = 0.9f;
	float m_beta2 = 0.999f;
	float m_epsilon = 1e-8f;

	float m_relative_weight_decay = 0.0f;
	float m_absolute_weight_decay = 0.0f;
};

TCNN_NAMESPACE_END
