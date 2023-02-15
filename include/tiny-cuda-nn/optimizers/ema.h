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

/** @file   ema.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of an ema wrapper around any optimizer.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/optimizer.h>

#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void ema_step_full_precision(
	const uint32_t n_elements,
	const float ema_decay,
	const float ema_debias_old,
	const float ema_debias_new,
	const T* __restrict__ weights,
	T* __restrict__ weights_ema,
	float* __restrict__ tmp
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	float filtered_val = ((float)tmp[i] * ema_decay * ema_debias_old + (float)weights[i] * (1 - ema_decay)) * ema_debias_new;
	tmp[i] = filtered_val;
	weights_ema[i] = (T)filtered_val;
}

template <typename T>
__global__ void ema_step_half_precision(
	const uint32_t n_elements,
	const float ema_decay,
	const float ema_debias_old,
	const float ema_debias_new,
	const T* __restrict__ weights,
	T* __restrict__ weights_ema
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	float filtered_val = ((float)weights_ema[i] * ema_decay * ema_debias_old + (float)weights[i] * (1 - ema_decay)) * ema_debias_new;
	weights_ema[i] = (T)filtered_val;
}

template <typename T>
class EmaOptimizer : public Optimizer<T> {
public:
	EmaOptimizer(const json& params) {
		m_nested.reset(create_optimizer<T>(params.value("nested", json::object())));
		update_hyperparams(params);
	}

	void allocate(uint32_t n_weights, const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
		m_nested->allocate(n_weights, layer_sizes);

		if (n_weights <= m_weights_ema.size()) {
			return;
		}

		m_weights_ema.resize(n_weights);
		m_weights_ema.memset(0);

		if (m_full_precision) {
			m_tmp.resize(n_weights);
			m_tmp.memset(0);
		}
	}

	void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
		m_nested->step(stream, loss_scale, weights_full_precision, weights, gradients);

		uint32_t current_step = m_nested->step();

		float ema_debias_old = 1 - (float)std::pow(m_ema_decay, current_step-1);
		float ema_debias_new = 1.0f / (1 - (float)std::pow(m_ema_decay, current_step));

		T* nested_custom_weights = m_nested->custom_weights();

		if (nested_custom_weights) {
			weights = nested_custom_weights;
		}

		if (m_full_precision) {
			linear_kernel(ema_step_full_precision<T>, 0, stream,
				n_weights(),
				m_ema_decay,
				ema_debias_old,
				ema_debias_new,
				weights,
				m_weights_ema.data(),
				m_tmp.data()
			);
		} else {
			linear_kernel(ema_step_half_precision<T>, 0, stream,
				n_weights(),
				m_ema_decay,
				ema_debias_old,
				ema_debias_new,
				weights,
				m_weights_ema.data()
			);
		}
	}

	float learning_rate() const override {
		return m_nested->learning_rate();
	}

	void set_learning_rate(float val) override {
		m_nested->set_learning_rate(val);
	}

	uint32_t step() const override {
		return m_nested->step();
	}

	uint32_t n_weights() const override {
		return m_nested->n_weights();
	}

	T* custom_weights() const override {
		return m_weights_ema.data();
	}

	size_t n_nested() const override {
		return 1;
	}

	const std::shared_ptr<Optimizer<T>>& nested(size_t idx) const override {
		CHECK_THROW(idx == 0);
		return m_nested;
	}

	void update_hyperparams(const json& params) override {
		if (params.contains("decay")) {
			m_ema_decay = params["decay"];
		}

		if (params.contains("full_precision")) {
			m_full_precision = params["full_precision"];
		}

		if (params.contains("nested")) {
			m_nested->update_hyperparams(params["nested"]);
		}
	}

	json hyperparams() const override {
		return {
			{"otype", "EMA"},
			{"nested", m_nested->hyperparams()},
			{"decay", m_ema_decay},
			{"full_precision", m_full_precision},
		};
	}

	json serialize() const override {
		json data;
		data["nested"] = m_nested->serialize();
		data["weights_ema_binary"] = m_weights_ema;
		return data;
	}

	void deserialize(const json& data) override {
		m_weights_ema = data["weights_ema_binary"];

		if (m_full_precision) {
			m_tmp.resize(m_weights_ema.size());
			linear_kernel(cast_from<T>, 0, nullptr, m_weights_ema.size(), m_weights_ema.data(), m_tmp.data());
		}

		m_nested->deserialize(data["nested"]);
	}

private:
	float m_ema_decay = 0.99f;
	bool m_full_precision = false;
	std::shared_ptr<Optimizer<T>> m_nested;

	GPUMemory<T> m_weights_ema;
	GPUMemory<float> m_tmp;
};

TCNN_NAMESPACE_END
