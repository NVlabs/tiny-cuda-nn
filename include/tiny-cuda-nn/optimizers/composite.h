/** @file   composite.h
 *  @author Sergei Solonets, Technical University of Munich
 *  @brief  Allow using different optimizer on different parameters.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/optimizer.h>

#include <stdint.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

/* This class is workaround around current architecture.
   If this PR is accepted the signature of allocate function could be changed to allocate(uint32 size).
   Apllying different regularization to encodings and mlp could be instead done using this optimizer
*/
template <typename PARAMS_T>
class ParametricObjectSlice : public ParametricObject<PARAMS_T> {
 public:
  ParametricObjectSlice(const ParametricObject<PARAMS_T>& object, uint32_t offset, uint32_t size)
      : m_offset(offset), m_size(size) {
    auto object_layer_size = object.layer_sizes();
    uint32_t current_offset = 0;
    uint32_t current_layer = 0;
    while (current_offset < offset && current_layer < object_layer_size.size()) {
      current_layer++;
      current_offset += object_layer_size[current_layer].first * object_layer_size[current_layer].second;
    }
    if (current_layer < object_layer_size.size() && current_offset != offset) {
      std::cout << current_offset << " " << offset << " " << current_layer << " " << object_layer_size.size()
                << std::endl;
      throw std::runtime_error{"Invalid slice. Can't slice within a layer"};
    }
    for (; current_layer < object_layer_size.size(); current_layer++) {
      m_layer_sizes.emplace_back(object_layer_size[current_layer]);
    }
  }
  void set_params(PARAMS_T* params, PARAMS_T* inference_params, PARAMS_T* backward_params,
                  PARAMS_T* gradients) override {
    throw std::runtime_error{"Not implemented"};
  }
  void initialize_params(pcg32& rnd, float* params_full_precision, PARAMS_T* params, PARAMS_T* inference_params,
                         PARAMS_T* backward_params, PARAMS_T* gradients, float scale = 1) override {
    throw std::runtime_error{"Not implemented"};
  }
  size_t n_params() const override { return m_size; }

  std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override { return m_layer_sizes; }
  json hyperparams() const override { return {}; }

 private:
  uint32_t m_offset;
  uint32_t m_size;
  std::vector<std::pair<uint32_t, uint32_t>> m_layer_sizes;
};

template <typename T>
class CompositeOptimizer : public Optimizer<T> {
 public:
  CompositeOptimizer(const json& params) {
    if (!params.contains("nested") || !params["nested"].is_array()) {
      throw std::runtime_error{"Must provide an array of nested encodings to CompositeOptimizer."};
    }
    const json::array_t& nested = params["nested"];
    m_n_weights = 0;
    for (size_t i = 0; i < nested.size(); ++i) {
      m_offsets.emplace_back(m_n_weights);
      m_n_weights += nested[i].value("n_params_to_optimize", 0);
      m_nested.emplace_back(std::unique_ptr<Optimizer<T>>(create_optimizer<T>(nested[i])));
    }
    m_offsets.emplace_back(m_n_weights);
    update_hyperparams(params);
  }

  void allocate(std::shared_ptr<ParametricObject<T>> target) override {
    for (int i = 0; i < m_nested.size(); i++) {
      uint32_t size = m_offsets[i + 1] - m_offsets[i];
      auto slice = std::make_shared<ParametricObjectSlice<T>>(*target, m_offsets[i], size);
      m_nested[i]->allocate(slice);
    }
    uint32_t size = (uint32_t)target->n_params();
  }

  void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, T* weights,
            const T* gradients) override {
    for (int i = 0; i < m_nested.size(); i++) {
      uint32_t offset = m_offsets[i];
      m_nested[i]->step(stream, loss_scale, weights_full_precision + offset, weights + offset, gradients + offset);
    }
  }

  float learning_rate() const override { return learning_rate_scale; }

  void set_learning_rate(float val) override {
    learning_rate_scale = val;
    for (int i = 0; i < m_nested.size(); i++) {
      m_nested[i]->set_learning_rate(m_nested[i]->learning_rate() * learning_rate_scale);
    }
  }

  uint32_t step() const override { return m_nested[0]->step(); }

  uint32_t n_weights() const override { return m_n_weights; }

  T* custom_weights() const override { return nullptr; }

  void update_hyperparams(const json& params) override {
    if (params.contains("nested") && params["nested"].is_array()) {
      const json::array_t& nested = params["nested"];
      for (int i = 0; i < m_nested.size(); i++) {
        m_nested[i]->update_hyperparams(nested[i]);
      }
    }
    if (params.contains("learning_rate_scale")) {
      learning_rate_scale = params["learning_rate_scale"];
    }
  }

  json hyperparams() const override {
    json::array_t nested;
    for (auto& n : m_nested) {
      nested.emplace_back(n->hyperparams());
    }
    return {{"otype", "Composite"}, {"nested", nested}, {"learning_rate_scale", learning_rate_scale}};
  }

  json serialize() const override {
    json::array_t nested;
    for (auto& n : m_nested) {
      nested.emplace_back(n->serialize());
    }
    return {{"nested", nested}};
  }

  void deserialize(const json& data) override {
    const json::array_t& nested = data["nested"];
    for (int i = 0; i < m_nested.size(); i++) {
      m_nested[i]->deserialize(nested[i]);
    }
    learning_rate_scale = data["learning_rate_scale"];
  }

 private:
  std::vector<std::unique_ptr<Optimizer<T>>> m_nested;
  std::vector<uint32_t> m_offsets;
  uint32_t m_n_weights;
  float learning_rate_scale = 1.0f;
};

TCNN_NAMESPACE_END
