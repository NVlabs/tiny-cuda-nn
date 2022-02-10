/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

/** @file   fully_fused_mlp.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Fully fused CUDA implementation of a multi-layer perceptron. Supports online training
 *          and simultaneous inference.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <vector>


TCNN_NAMESPACE_BEGIN

template <typename T, int WIDTH>
class FullyFusedMLP : public Network<T> {
public:
	FullyFusedMLP(uint32_t input_width, uint32_t output_width, uint32_t n_hidden_layers, bool use_feedback_alignment, Activation activation, Activation output_activation);
	~FullyFusedMLP() override;

	void inference(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<float>& output) override;
	void inference_mixed_precision(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output, bool use_inference_matrices = true) override;

	void forward(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_matrices = false, bool prepare_input_gradients = false) override;
	void forward_clear() override {
		m_forward.clear();
	}

	void backward(
		cudaStream_t stream,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_matrices = false,
		bool compute_param_gradients = true
	) override;

	void set_params(T* params, T* inference_params, T* backward_params, T* gradients) override;
	void initialize_params(pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override;

	GPUMatrix<T, RM>& input_weight_matrix(WeightUsage usage) {
		switch (usage) {
			case WeightUsage::Inference: return m_weight_matrices_inference.front();
			case WeightUsage::Forward: return m_weight_matrices.front();
			case WeightUsage::Backward: return m_weight_matrices_backward.front();
		}

		throw std::runtime_error{"Invalid weight usage."};
	}

	GPUMatrix<T, RM>& weight_matrix_at(WeightUsage usage, uint32_t idx) {
		switch (usage) {
			case WeightUsage::Inference: return m_weight_matrices_inference.at(1 + idx);
			case WeightUsage::Forward: return m_weight_matrices.at(1 + idx);
			case WeightUsage::Backward: return m_weight_matrices_backward.at(1 + idx);
		}

		throw std::runtime_error{"Invalid weight usage."};
	}

	GPUMatrix<T, RM>& output_weight_matrix(WeightUsage usage) {
		switch (usage) {
			case WeightUsage::Inference: return m_weight_matrices_inference.back();
			case WeightUsage::Forward: return m_weight_matrices.back();
			case WeightUsage::Backward: return m_weight_matrices_backward.back();
		}

		throw std::runtime_error{"Invalid weight usage."};
	}

	GPUMatrix<T, RM>& input_gradient_matrix() {
		return m_gradient_matrices.front();
	}

	GPUMatrix<T, RM>& gradient_matrix_at(uint32_t idx) {
		return m_gradient_matrices.at(1 + idx);
	}

	GPUMatrix<T, RM>& output_gradient_matrix() {
		return m_gradient_matrices.back();
	}

	size_t n_params() const override {
		return m_total_n_params;
	}

	uint32_t padded_output_width() const override {
		return m_padded_output_width;
	}

	uint32_t output_width() const override {
		return m_output_width;
	}

	uint32_t required_input_alignment() const override {
		return tensorcore_width;
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		std::vector<std::pair<uint32_t, uint32_t>> result;
		for (auto& matrix : m_weight_matrices) {
			result.emplace_back(matrix.m(), matrix.n());
		}
		return result;
	}

	uint32_t width(uint32_t layer) const override {
		return WIDTH;
	}

	uint32_t num_forward_activations() const override {
		return m_n_hidden_layers;
	}

	const T* forward_activations(uint32_t layer) const override {
		if (m_forward.hidden.size() == 0) {
			throw std::runtime_error{"Must call forward() before accessing activations."};
		}
		return m_forward.hidden.at(layer).data();
	}

private:
	void allocate_forward_buffers(cudaStream_t stream, uint32_t batch_size);

	uint32_t m_n_hidden_layers;
	uint32_t m_n_hidden_matmuls;
	uint32_t m_input_width;
	uint32_t m_network_width;
	uint32_t m_output_width;
	uint32_t m_padded_output_width;

	Activation m_activation;
	Activation m_output_activation;

	bool m_use_feedback_alignment = false;

	static const uint32_t tensorcore_width = 16;

	// Streams and events
	std::vector<cudaStream_t> m_training_splitk_streams;
	std::vector<cudaEvent_t> m_training_splitk_events;

	// Storage of forward pass data
	struct {
		std::vector<GPUMatrix<T>> hidden;
		GPUMemoryArena::Allocation alloc;

		void clear() {
			hidden.clear();
			alloc = {};
		}
	} m_forward;

	// Storage of params
	std::vector<GPUMatrix<T, RM>> m_weight_matrices;
	std::vector<GPUMatrix<T, RM>> m_weight_matrices_inference;
	std::vector<GPUMatrix<T, RM>> m_weight_matrices_backward;
	size_t m_total_n_params;

	std::vector<GPUMatrix<float, RM>> m_weight_matrices_full_precision;

	std::vector<GPUMatrix<T, RM>> m_gradient_matrices;
};

TCNN_NAMESPACE_END
