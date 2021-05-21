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

/** @file   fully_fused_mlp.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Fully fused CUDA implementation of a multi-layer perceptron. Supports online training
 *          and simultaneous inference.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/cuda_graph.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/misc_kernels.h>

#include <tiny-cuda-nn/cutlass_matmul_interface.h>

#include <cutlass/half.h>

#include <array>
#include <iostream>
#include <memory>


TCNN_NAMESPACE_BEGIN

template <typename T, int WIDTH>
class FullyFusedMLP : public Network<T> {
public:
	FullyFusedMLP(uint32_t input_width, uint32_t output_width, uint32_t n_hidden_layers, bool use_feedback_alignment, Activation activation, Activation output_activation);
	~FullyFusedMLP() override;

	void inference(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& input, GPUMatrix<float, MatrixLayout::ColumnMajor>& output) override;
	void inference_mixed_precision(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& input, GPUMatrix<T, MatrixLayout::ColumnMajor>& output, MatrixLayout output_layout = MatrixLayout::ColumnMajor) override;

	void forward(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& input, GPUMatrix<T, MatrixLayout::ColumnMajor>& output, MatrixLayout output_layout = MatrixLayout::ColumnMajor, bool use_inference_matrices = false) override;

	void backward(
		cudaStream_t stream,
		const GPUMatrix<T, MatrixLayout::ColumnMajor>& input,
		const GPUMatrix<T, MatrixLayout::ColumnMajor>& output,
		const GPUMatrix<T, MatrixLayout::ColumnMajor>& dL_doutput,
		GPUMatrix<T, MatrixLayout::ColumnMajor>* dL_dinput = nullptr,
		MatrixLayout output_layout = MatrixLayout::ColumnMajor,
		bool use_inference_matrices = false,
		bool compute_param_gradients = true
	) override;

	void initialize_params(std::mt19937& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override;

	GPUMatrix<T, MatrixLayout::RowMajor>& input_weight_matrix(WeightUsage usage) {
		switch (usage) {
			case WeightUsage::Inference: return m_weight_matrices_inference.front();
			case WeightUsage::Forward: return m_weight_matrices.front();
			case WeightUsage::Backward: return m_weight_matrices_backward.front();
		}

		throw std::runtime_error{"Invalid weight usage."};
	}

	GPUMatrix<T, MatrixLayout::RowMajor>& weight_matrix_at(WeightUsage usage, uint32_t idx) {
		switch (usage) {
			case WeightUsage::Inference: return m_weight_matrices_inference.at(1 + idx);
			case WeightUsage::Forward: return m_weight_matrices.at(1 + idx);
			case WeightUsage::Backward: return m_weight_matrices_backward.at(1 + idx);
		}

		throw std::runtime_error{"Invalid weight usage."};
	}

	GPUMatrix<T, MatrixLayout::RowMajor>& output_weight_matrix(WeightUsage usage) {
		switch (usage) {
			case WeightUsage::Inference: return m_weight_matrices_inference.back();
			case WeightUsage::Forward: return m_weight_matrices.back();
			case WeightUsage::Backward: return m_weight_matrices_backward.back();
		}

		throw std::runtime_error{"Invalid weight usage."};
	}

	GPUMatrix<T, MatrixLayout::RowMajor>& input_gradient_matrix() {
		return m_gradient_matrices.front();
	}

	GPUMatrix<T, MatrixLayout::RowMajor>& gradient_matrix_at(uint32_t idx) {
		return m_gradient_matrices.at(1 + idx);
	}

	GPUMatrix<T, MatrixLayout::RowMajor>& output_gradient_matrix() {
		return m_gradient_matrices.back();
	}

	void set_output_activation_param(float value) {
		m_output_activation_param = value;
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

private:
	void allocate_inference_buffers(uint32_t batch_size);

	void allocate_forward_buffers(uint32_t batch_size);

	void allocate_backward_buffers(uint32_t batch_size);

	uint32_t m_n_hidden_layers;
	uint32_t m_n_hidden_matmuls;
	uint32_t m_input_width;
	uint32_t m_network_width;
	uint32_t m_output_width;
	uint32_t m_padded_output_width;

	Activation m_activation;
	Activation m_output_activation;
	float m_output_activation_param = 0;

	bool m_use_feedback_alignment = false;

	static const uint32_t tensorcore_width = 16;

	// Streams and events
	std::vector<cudaStream_t> m_training_splitk_streams;
	std::vector<cudaEvent_t> m_training_splitk_events;

	// Storage of inference temporary data
	GPUMemory<char> m_inference_buffer;
	std::array<GPUMatrix<T, MatrixLayout::ColumnMajor>, 2> m_inference_tmp;
	GPUMatrix<T, MatrixLayout::ColumnMajor> m_inference_output_tmp;

	// Storage of forward pass data
	GPUMemory<char> m_forward_buffer = GPUMemory<char>(0);
	std::vector<GPUMatrix<T, MatrixLayout::ColumnMajor>> m_forward_tmp;

	// Storage of backward pass data
	GPUMemory<char> m_backward_buffer = GPUMemory<char>(0);
	std::vector<GPUMatrix<T, MatrixLayout::ColumnMajor>> m_backward_tmp;
	GPUMatrix<T, MatrixLayout::ColumnMajor> m_backward_output_tmp;

	// Storage of params
	std::vector<GPUMatrix<T, MatrixLayout::RowMajor>> m_weight_matrices;
	std::vector<GPUMatrix<T, MatrixLayout::RowMajor>> m_weight_matrices_inference;
	std::vector<GPUMatrix<T, MatrixLayout::RowMajor>> m_weight_matrices_backward;
	size_t m_total_n_params;

	std::vector<GPUMatrix<float, MatrixLayout::RowMajor>> m_weight_matrices_full_precision;

	std::vector<GPUMatrix<T, MatrixLayout::RowMajor>> m_gradient_matrices;
};

TCNN_NAMESPACE_END
