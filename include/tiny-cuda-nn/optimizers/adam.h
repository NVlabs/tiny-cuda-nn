/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

/** @file   adam.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the adam optimizer with support for
 *          the AdaBound paper: https://openreview.net/pdf?id=Bkg3g2R9FX
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/misc_kernels.h>
#include <tiny-cuda-nn/optimizer.h>

#include <random>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>


TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void adam_step(
	const uint32_t n_elements,
	const uint32_t n_matrix_weights,
	const float relative_weight_decay,
	const float absolute_weight_decay,
	const float loss_scale,
	float base_learning_rate,
	float learning_rate,
	const float beta1,
	const float beta2,
	const float epsilon,
	const float lower_lr_bound,
	const float upper_lr_bound,
	const float l2_reg,
	float* __restrict__ weights_full_precision,
	T* __restrict__ weights,
	const T* __restrict__ gradients,
	float* __restrict__ first_moments,
	float* __restrict__ second_moments
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const float weight_fp = weights_full_precision[i];
	float gradient = (float)gradients[i] / loss_scale;


	gradient += l2_reg * weight_fp;

	const float gradient_sq = gradient * gradient;

	float first_moment = first_moments[i] = beta1 * first_moments[i] + (1 - beta1) * gradient;
	const float second_moment = second_moments[i] = beta2 * second_moments[i] + (1 - beta2) * gradient_sq;


	// Follow AdaBound paradigm
	const float effective_learning_rate = fminf(fmaxf(learning_rate / (sqrtf(second_moment) + epsilon), lower_lr_bound), upper_lr_bound);

	const float decayed_weight = weight_decay(relative_weight_decay * learning_rate, absolute_weight_decay * learning_rate, weight_fp);
	const float new_weight = decayed_weight - effective_learning_rate * first_moment;

	weights_full_precision[i] = new_weight;
	weights[i] = (T)new_weight;
}

template <typename T>
class AdamOptimizer : public Optimizer<T> {
public:
	AdamOptimizer(json params) {
		update_hyperparams(params);
	}

	void allocate(std::shared_ptr<ParametricObject<T>> target) override {
		uint32_t size = (uint32_t)target->n_params();

		m_n_weights = size;
		if (m_n_weights <= m_first_moments.get_num_elements()) {
			return;
		}

		m_first_moments.resize(size);
		m_first_moments.memset(0);

		m_second_moments.resize(size);
		m_second_moments.memset(0);

		m_n_weights_covered_by_matrices = 0;
		auto layer_sizes = target->layer_sizes();

		for (size_t i = 0; i < layer_sizes.size(); ++i) {
			m_n_weights_covered_by_matrices += layer_sizes[i].first * layer_sizes[i].second;
		}
	}

	void step(cudaStream_t stream, float loss_scale, float learning_rate, float* weights_full_precision, T* weights, const T* gradients) override {
		++m_current_step;

		m_base_learning_rate = learning_rate;
		learning_rate = m_base_learning_rate * std::sqrt(1 - std::pow(m_beta2, (float)m_current_step)) / (1 - std::pow(m_beta1, (float)m_current_step));

		float lower_lr_bound = 0;
		float upper_lr_bound = std::numeric_limits<float>::max();

		// AdaBound paper: https://openreview.net/pdf?id=Bkg3g2R9FX
		if (m_adabound) {
			lower_lr_bound = 0.1f - 0.1f / ((1 - m_beta2) * (float)step() + 1);
			upper_lr_bound = 0.1f + 0.1f / ((1 - m_beta2) * (float)step());
		}

		linear_kernel(adam_step<T>, 0, stream,
			n_weights(),
			m_n_weights_covered_by_matrices,
			m_relative_weight_decay,
			m_absolute_weight_decay,
			loss_scale,
			m_base_learning_rate,
			learning_rate,
			m_beta1,
			m_beta2,
			m_epsilon,
			lower_lr_bound,
			upper_lr_bound,
			m_l2_reg,
			weights_full_precision,
			weights,
			gradients,
			m_first_moments.data(),
			m_second_moments.data()
		);
	}

	float learning_rate() const {
		return m_base_learning_rate;
	}

	uint32_t step() const {
		return m_current_step;
	}

	uint32_t n_weights() const {
		return m_n_weights;
	}

	T* custom_weights() const {
		return nullptr;
	}

	void update_hyperparams(json params) override {
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

		if (params.contains("l2_reg")) {
			m_l2_reg = params["l2_reg"];
		}

		if (params.contains("adabound")) {
			m_adabound = params["adabound"];
		}

		if (params.contains("relative_decay")) {
			m_relative_weight_decay = params["relative_decay"];
		}

		if (params.contains("absolute_decay")) {
			m_absolute_weight_decay = params["absolute_decay"];
		}
	}

private:
	uint32_t m_n_weights;
	uint32_t m_n_weights_covered_by_matrices;

	GPUMemory<float> m_first_moments;
	GPUMemory<float> m_second_moments;

	uint32_t m_current_step = 0;

	// Hyperparameters
	float m_base_learning_rate = 1e-3f;
	float m_beta1 = 0.9f;
	float m_beta2 = 0.999f;
	float m_epsilon = 1e-8f;
	float m_l2_reg = 1e-8f;

	float m_relative_weight_decay = 0.0f;
	float m_absolute_weight_decay = 0.0f;

	bool m_adabound = false;
};

TCNN_NAMESPACE_END
