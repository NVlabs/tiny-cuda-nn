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

/** @file   sgd.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the sgd optimizer.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/optimizer.h>

#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void sgd_step(
	const uint32_t n_elements,
	const float loss_scale,
	const float learning_rate,
	const float l2_reg,
	float* __restrict__ weights_full_precision,
	T* __restrict__ weights,
	const T* __restrict__ gradients
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const float weight_fp = weights_full_precision[i];
	float gradient = (float)gradients[i] / loss_scale;

	// Optional gradient clipping
	// static const float clip_value = 1000.0f;
	// gradient = fminf(fmaxf(gradient, -clip_value), clip_value);

	gradient += l2_reg * weight_fp;

	const float new_weight = weight_fp - learning_rate * gradient;

	weights_full_precision[i] = new_weight;
	weights[i] = (T)new_weight;
}

template <typename T>
class SGDOptimizer : public Optimizer<T> {
public:
	SGDOptimizer(const json& params) {
		update_hyperparams(params);
	}

	void allocate(uint32_t n_weights, const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
		m_n_weights = n_weights;
	}

	void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
		++m_current_step;

		linear_kernel(sgd_step<T>, 0, stream,
			n_weights(),
			loss_scale,
			m_learning_rate,
			m_l2_reg,
			weights_full_precision,
			weights,
			gradients
		);
	}

	float learning_rate() const override {
		return m_learning_rate;
	}

	void set_learning_rate(float val) override {
		m_learning_rate = val;
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
		if (params.contains("learning_rate")) {
			m_learning_rate = params["learning_rate"];
		}

		if (params.contains("l2_reg")) {
			m_l2_reg = params["l2_reg"];
		}
	}

	json hyperparams() const override {
		return {
			{"otype", "SGD"},
			{"learning_rate", m_learning_rate},
			{"l2_reg", m_l2_reg},
		};
	}

	json serialize() const override {
		json data;
		data["current_step"] = m_current_step;
		data["learning_rate"] = m_learning_rate;
		return data;
	}

	void deserialize(const json& data) override {
		m_current_step = data["current_step"];
		m_learning_rate = data["learning_rate"];
	}

private:
	uint32_t m_n_weights;
	uint32_t m_current_step = 0;

	// Hyperparameters
	float m_learning_rate = 1e-3f;
	float m_l2_reg = 1e-8f;
};

TCNN_NAMESPACE_END
