/** @file   composite.h
 *  @author Sergei Solonets, Technical University of Munich
 *  @brief  Allow using different optimizer on different parameters.
 */

#pragma once

#include <stdint.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/optimizer.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

std::vector<std::pair<uint32_t, uint32_t>> slice_weights(
	const std::vector<std::pair<uint32_t, uint32_t>>& object_layer_size,
	uint32_t offset,
	uint32_t size
) {
	uint32_t current_offset = 0;
	uint32_t current_layer = 0;
	while (current_offset < offset && current_layer < object_layer_size.size()) {
		current_layer++;
		current_offset += object_layer_size[current_layer].first * object_layer_size[current_layer].second;
	}

	if (current_layer < object_layer_size.size() && current_offset != offset) {
		throw std::runtime_error{"Invalid slice. Can't slice within a layer."};
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes;
	for (; current_layer < object_layer_size.size(); current_layer++) {
		layer_sizes.emplace_back(object_layer_size[current_layer]);
	}

	return layer_sizes;
}

template <typename T>
class CompositeOptimizer : public Optimizer<T> {
public:
	CompositeOptimizer(const json &params) {
		if (!params.contains("nested") || !params["nested"].is_array() || params["nested"].empty()) {
			throw std::runtime_error{"Must provide an array of nested encodings to CompositeOptimizer."};
		}

		m_n_weights = 0;
		for (const auto& nested : params["nested"]) {
			m_offsets.emplace_back(m_n_weights);
			m_n_weights += nested.value("n_params_to_optimize", 0);
			m_nested.emplace_back(std::shared_ptr<Optimizer<T>>{create_optimizer<T>(nested)});
			m_base_learning_rates.emplace_back(m_nested.back()->learning_rate());
		}

		m_offsets.emplace_back(m_n_weights);
		update_hyperparams(params);
	}

	void allocate(uint32_t n_weights, const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
		for (size_t i = 0; i < m_nested.size(); i++) {
			uint32_t size = m_offsets[i + 1] - m_offsets[i];
			m_nested[i]->allocate(size, slice_weights(layer_sizes, m_offsets[i], size));
			m_need_custom_weights |= m_nested[i]->custom_weights() != nullptr;
		}

		if (m_need_custom_weights) {
			m_custom_weights.resize(m_n_weights);
		}
	}

	void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
		for (size_t i = 0; i < m_nested.size(); ++i) {
			uint32_t offset = m_offsets[i];
			m_nested[i]->step(stream, loss_scale, weights_full_precision + offset, weights + offset, gradients + offset);
			if (m_need_custom_weights) {
				CUDA_CHECK_THROW(cudaMemcpyAsync(
					m_custom_weights.data() + offset,
					m_nested[i]->custom_weights() == nullptr ? weights + offset : m_nested[i]->custom_weights(),
					m_nested[i]->n_weights() * sizeof(T),
					cudaMemcpyDeviceToDevice,
					stream
				));
			}
		}
	}

	float learning_rate() const override {
		return m_learning_rate_factor;
	}

	void set_learning_rate(float val) override {
		m_learning_rate_factor = val;
		for (size_t i = 0; i < m_nested.size(); i++) {
			m_nested[i]->set_learning_rate(m_base_learning_rates[i] * m_learning_rate_factor);
		}
	}

	uint32_t step() const override {
		return m_nested[0]->step();
	}

	uint32_t n_weights() const override {
		return m_n_weights;
	}

	T* custom_weights() const override {
		return m_custom_weights.data();
	}

	void update_hyperparams(const json& params) override {
		if (params.contains("nested") && params["nested"].is_array()) {
			const json::array_t& nested = params["nested"];
			for (size_t i = 0; i < m_nested.size(); i++) {
				m_nested[i]->update_hyperparams(nested[i]);
			}
		}
	}

	size_t n_nested() const override {
		return m_nested.size();
	}

	const std::shared_ptr<Optimizer<T>>& nested(size_t idx) const override {
		CHECK_THROW(idx < m_nested.size());
		return m_nested[idx];
	}

	json hyperparams() const override {
		json::array_t nested;
		for (auto& n : m_nested) {
			nested.emplace_back(n->hyperparams());
		}

		return {{"otype", "Composite"}, {"nested", nested}};
	}

	json serialize() const override {
		json::array_t nested;
		for (auto& n : m_nested) {
			nested.emplace_back(n->serialize());
		}

		return {
			{"nested", nested},
			{"base_learning_rates", m_base_learning_rates},
			{"learning_rate_factor", m_learning_rate_factor}
		};
	}

	void deserialize(const json& data) override {
		const json::array_t& nested = data["nested"];
		for (size_t i = 0; i < m_nested.size(); i++) {
			m_nested.at(i)->deserialize(nested[i]);
		}

		data["base_learning_rates"].get_to(m_base_learning_rates);
		set_learning_rate(data["learning_rate_factor"]);
	}

private:
	std::vector<std::shared_ptr<Optimizer<T>>> m_nested;
	std::vector<float> m_base_learning_rates;
	std::vector<uint32_t> m_offsets;
	uint32_t m_n_weights;
	float m_learning_rate_factor = 1.0f;
	bool m_need_custom_weights = false;
	GPUMemory<T> m_custom_weights;
};

TCNN_NAMESPACE_END
