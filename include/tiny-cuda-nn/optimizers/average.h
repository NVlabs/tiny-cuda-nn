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

/** @file   average.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of a weight-average wrapper around any optimizer.
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
__global__ void average_step(
	const uint32_t n_elements,
	const uint32_t n_samples,
	const T* __restrict__ weights,
	T* __restrict__ weights_current_sample,
	T* __restrict__ weights_average
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	T weight = weights[i];
	weights_average[i] = (T)((float)weights_average[i] + ((float)weight - (float)weights_current_sample[i]) / n_samples);
	weights_current_sample[i] = weight;
}

template <typename T>
class AverageOptimizer : public Optimizer<T> {
public:
	AverageOptimizer(const json& params) {
		m_nested.reset(create_optimizer<T>(params.value("nested", json::object())));
		update_hyperparams(params);
	}

	void allocate(uint32_t n_weights, const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
		m_n_weights = n_weights;
		m_layer_sizes = layer_sizes;

		m_nested->allocate(n_weights, layer_sizes);

		m_weights_samples.resize(m_n_weights * m_n_samples);
		m_weights_samples.memset(0);

		m_weights_average.resize(m_n_weights);
		m_weights_average.memset(0);
	}

	void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
		m_nested->step(stream, loss_scale, weights_full_precision, weights, gradients);

		linear_kernel(average_step<T>, 0, stream,
			n_weights(),
			m_n_samples,
			weights,
			current_sample(),
			m_weights_average.data()
		);
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
		return m_weights_average.data();
	}

	uint32_t current_sample_idx() const {
		return step() % m_n_samples;
	}

	T* current_sample() const {
		return m_weights_samples.data() + current_sample_idx() * m_n_weights;
	}

	size_t n_nested() const override {
		return 1;
	}

	const std::shared_ptr<Optimizer<T>>& nested(size_t idx) const override {
		CHECK_THROW(idx == 0);
		return m_nested;
	}

	void update_hyperparams(const json& params) override {
		if (params.contains("n_samples")) {
			m_n_samples = params["n_samples"];
			if (m_n_weights > 0 || !m_layer_sizes.empty()) {
				allocate(m_n_weights, m_layer_sizes);
			}
		}

		if (params.contains("nested")) {
			m_nested->update_hyperparams(params["nested"]);
		}
	}

	json hyperparams() const override {
		return {
			{"otype", "Average"},
			{"nested", m_nested->hyperparams()},
			{"n_samples", m_n_samples},
		};
	}

	json serialize() const override {
		json data;
		data["nested"] = m_nested->serialize();
		data["weights_samples_binary"] = m_weights_samples;
		data["weights_average_binary"] = m_weights_average;
		return data;
	}

	void deserialize(const json& data) override {
		m_weights_samples = data["weights_samples_binary"];
		m_weights_average = data["weights_average_binary"];
		m_nested->deserialize(data["nested"]);
	}

private:
	uint32_t m_n_samples = 128;
	uint32_t m_n_weights = 0;
	std::shared_ptr<Optimizer<T>> m_nested;

	std::vector<std::pair<uint32_t, uint32_t>> m_layer_sizes;

	GPUMemory<T> m_weights_samples;
	GPUMemory<T> m_weights_average;
};

TCNN_NAMESPACE_END
