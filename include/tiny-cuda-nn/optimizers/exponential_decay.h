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

/** @file   exponential_decay.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Optimizer that performs piecewise-constant exponential learning rate decay on a nested optimizer.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/optimizer.h>

#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

template <typename T>
class ExponentialDecayOptimizer : public Optimizer<T> {
public:
	ExponentialDecayOptimizer(const json& params) {
		m_nested.reset(create_optimizer<T>(params.value("nested", json::object())));
		update_hyperparams(params);

		m_learning_rate_factor = 1.0f;
		m_base_learning_rate = m_nested->learning_rate();
	}

	void allocate(uint32_t n_weights, const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
		m_nested->allocate(n_weights, layer_sizes);
	}

	void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
		if (step() == 0) {
			m_learning_rate_factor = 1.0f;
		}

		if (step() >= m_decay_start && (step() - m_decay_start) % m_decay_interval == 0 && step() <= m_decay_end) {
			m_learning_rate_factor *= m_decay_base;
		}

		m_nested->set_learning_rate(m_base_learning_rate * m_learning_rate_factor);
		m_nested->step(stream, loss_scale, weights_full_precision, weights, gradients);
	}

	float learning_rate() const override {
		return m_base_learning_rate * m_learning_rate_factor;
	}

	void set_learning_rate(float val) override {
		m_base_learning_rate = val / m_learning_rate_factor;
		m_nested->set_learning_rate(m_base_learning_rate * m_learning_rate_factor);
	}

	uint32_t step() const override {
		return m_nested->step();
	}

	uint32_t n_weights() const override {
		return m_nested->n_weights();
	}

	T* custom_weights() const override {
		return m_nested->custom_weights();
	}

	size_t n_nested() const override {
		return 1;
	}

	const std::shared_ptr<Optimizer<T>>& nested(size_t idx) const override {
		CHECK_THROW(idx == 0);
		return m_nested;
	}

	void update_hyperparams(const json& params) override {
		if (params.contains("decay_base")) {
			m_decay_base = params["decay_base"];
		}

		if (params.contains("decay_interval")) {
			m_decay_interval = params["decay_interval"];
		}

		if (params.contains("decay_start")) {
			m_decay_start = params["decay_start"];
		}

		if (params.contains("decay_end")) {
			m_decay_end = params["decay_end"];
		}

		if (params.contains("nested")) {
			m_nested->update_hyperparams(params["nested"]);
		}
	}

	json hyperparams() const override {
		return {
			{"otype", "ExponentialDecay"},
			{"nested", m_nested->hyperparams()},
			{"decay_base", m_decay_base},
			{"decay_interval", m_decay_interval},
			{"decay_start", m_decay_start},
			{"decay_end", m_decay_end},
		};
	}

	json serialize() const override {
		json data;
		data["nested"] = m_nested->serialize();
		data["learning_rate"] = m_base_learning_rate;
		data["learning_rate_factor"] = m_learning_rate_factor;
		return data;
	}

	void deserialize(const json& data) override {
		m_base_learning_rate = data["learning_rate"];
		m_learning_rate_factor = data.value("learning_rate_factor", 1.0f);
		m_nested->deserialize(data["nested"]);
	}

private:
	std::shared_ptr<Optimizer<T>> m_nested;

	float m_learning_rate_factor = 1.0f;
	float m_base_learning_rate;

	float m_decay_base = 0.1f;
	uint32_t m_decay_interval = 10000;
	uint32_t m_decay_start = 10000;
	uint32_t m_decay_end = 10000000;
};

TCNN_NAMESPACE_END
