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

/** @file   lion.h
 *  @author James Perlman, NVIDIA superfan
 *  @brief  CUDA Implementation of the lion optimizer: https://arxiv.org/abs/2302.06675
 *  @note   This code is based on https://github.com/google/automl/blob/master/lion/lion_pytorch.py
 *          Copyright (c) 2023, Google Research. Apache 2.0 License.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/gpu_memory_json.h>
#include <json/json.hpp>

#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void lion_step(
    const uint32_t n_weights,
    const float loss_scale,
    const float learning_rate,
    const float beta1,
    const float beta2,
    const float weight_decay,
    float* __restrict__ weights_fp,
    T* __restrict__ weights,
    const T* __restrict__ gradients,
    float* __restrict__ exp_avgs
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_weights) return;

    // Fetch inputs
    float weight = weights_fp[i];
    float gradient = (float)gradients[i] / loss_scale;
    float exp_avg = exp_avgs[i];

    // Perform stepweight decay
    weight = weight * (1.0f - learning_rate * weight_decay);

    // Weight update
    float update = exp_avg * beta1 + gradient * (1.0f - beta1);
    weight = weight - learning_rate * copysignf(1.0f, update);

    // Decay the momentum running average coefficient
    exp_avg = exp_avg * beta2 + gradient * (1.0f - beta2);

    // Assign outputs
    weights[i] = (T)weight;
    weights_fp[i] = weight;
    exp_avgs[i] = exp_avg;
}

template <typename T>
class LionOptimizer : public Optimizer<T> {
public:
	LionOptimizer(const json& params) {
		update_hyperparams(params);
	}

	void allocate(uint32_t n_weights, const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
		m_n_weights = n_weights;

		m_exponential_averages.resize(m_n_weights);
		m_exponential_averages.memset(0);
	}

	void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
		++m_current_step;

		linear_kernel(lion_step<T>, 0, stream,
			m_n_weights,
			loss_scale,
			m_base_learning_rate,
			m_beta1,
			m_beta2,
			m_weight_decay,
			weights_full_precision,
			weights,
			gradients,
			m_exponential_averages.data()
		);
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

		if (params.contains("learning_rate")) {
			m_base_learning_rate = params["learning_rate"];
		}

		if (params.contains("weight_decay")) {
			m_weight_decay = params["weight_decay"];
		}
	}

	json hyperparams() const override {
		return {
			{"otype", "Lion"},
			{"beta1", m_beta1},
			{"beta2", m_beta2},
			{"learning_rate", m_base_learning_rate},
			{"weight_decay", m_weight_decay},
		};
	}

	json serialize() const override {
		json data;
		data["current_step"] = m_current_step;
		data["base_learning_rate"] = m_base_learning_rate;
		data["exponential_averages_binary"] = m_exponential_averages;
		return data;
	}

	void deserialize(const json& data) override {
		m_exponential_averages = data["exponential_averages_binary"];
		m_current_step = data["current_step"];
		m_base_learning_rate = data["base_learning_rate"];
	}

private:
	uint32_t m_n_weights;

	GPUMemory<float> m_exponential_averages;

	uint32_t m_current_step = 0;

	// Hyperparameters
	float m_base_learning_rate = 1e-4f;
	float m_weight_decay = 0.0f;
	float m_beta1 = 0.9f;
	float m_beta2 = 0.99f;
};

TCNN_NAMESPACE_END
