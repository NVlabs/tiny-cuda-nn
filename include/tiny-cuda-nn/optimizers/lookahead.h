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

/** @file   lookahead.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of a lookahead (this paper: https://arxiv.org/pdf/1907.08610.pdf) wrapper around any optimizer.
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
__global__ void lookahead_step(
	const uint32_t n_elements,
	const float alpha,
	float* __restrict__ weights_full_precision,
	T* __restrict__ weights,
	T* __restrict__ weights_lookahead
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	float new_weight = ((float)weights_lookahead[i] * (1.0f - alpha) + weights_full_precision[i] * alpha);
	weights_full_precision[i] = new_weight;
	weights_lookahead[i] = weights[i] = (T)new_weight;
}

template <typename T>
class LookaheadOptimizer : public Optimizer<T> {
public:
	LookaheadOptimizer(const json& params) {
		m_nested.reset(create_optimizer<T>(params.value("nested", json::object())));
		update_hyperparams(params);
	}

	void allocate(uint32_t n_weights, const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
		m_nested->allocate(n_weights, layer_sizes);

		if (n_weights <= m_weights_lookahead.size()) {
			return;
		}

		m_weights_lookahead.resize(n_weights);
	}

	void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
		uint32_t current_step = m_nested->step();

		if (current_step == 0) {
			CUDA_CHECK_THROW(cudaMemcpy(m_weights_lookahead.data(), weights, m_weights_lookahead.size() * sizeof(T), cudaMemcpyDeviceToDevice));
		}

		if (current_step % m_n_steps == 0) {
			linear_kernel(lookahead_step<T>, 0, stream,
				n_weights(),
				m_alpha,
				weights_full_precision,
				weights,
				m_weights_lookahead.data()
			);
		}

		m_nested->step(stream, loss_scale, weights_full_precision, weights, gradients);
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
		return m_weights_lookahead.data();
	}

	size_t n_nested() const override {
		return 1;
	}

	const std::shared_ptr<Optimizer<T>>& nested(size_t idx) const override {
		CHECK_THROW(idx == 0);
		return m_nested;
	}

	void update_hyperparams(const json& params) override {
		if (params.contains("alpha")) {
			m_alpha = params["alpha"];
		}

		if (params.contains("n_steps")) {
			m_n_steps = params["n_steps"];
		}

		if (params.contains("nested")) {
			m_nested->update_hyperparams(params["nested"]);
		}
	}

	json hyperparams() const override {
		return {
			{"otype", "Lookahead"},
			{"nested", m_nested->hyperparams()},
			{"alpha", m_alpha},
			{"n_steps", m_n_steps},
		};
	}

	json serialize() const override {
		json data;
		data["nested"] = m_nested->serialize();
		data["weights_lookahead_binary"] = m_weights_lookahead;
		return data;
	}

	void deserialize(const json& data) override {
		m_weights_lookahead = data["weights_lookahead_binary"];
		m_nested->deserialize(data["nested"]);
	}

private:
	float m_alpha = 0.5f;
	uint32_t m_n_steps = 16;
	std::shared_ptr<Optimizer<T>> m_nested;

	GPUMemory<T> m_weights_lookahead;
};

TCNN_NAMESPACE_END
