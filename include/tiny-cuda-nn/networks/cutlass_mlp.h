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

/** @file   cutlass_mlp.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  CUDA/CUTLASS implementation of a multi-layer perceptron. Supports online training
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

template <typename T>
class CutlassMLP : public Network<T> {
public:
	using type_t = T;

	CutlassMLP(
		uint32_t input_width, uint32_t network_width, uint32_t output_width, uint32_t n_hidden_layers,
		Activation activation, Activation output_activation
	);
	~CutlassMLP() override;

	void inference(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& input, GPUMatrix<float, MatrixLayout::ColumnMajor>& output) override;
	void inference_mixed_precision(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& input, GPUMatrix<T, MatrixLayout::ColumnMajor>& output, MatrixLayout output_layout = MatrixLayout::ColumnMajor) override;

	void forward(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& input, GPUMatrix<T, MatrixLayout::ColumnMajor>& output, MatrixLayout output_layout = MatrixLayout::ColumnMajor, bool use_inference_matrices = false) override;

	void compute_activation_transfer(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& values, GPUMatrix<T, MatrixLayout::ColumnMajor>& gradients);

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

	void initialize_params(float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override;

	GPUMatrix<T, MatrixLayout::RowMajor>& input_weight_matrix(bool inference) {
		auto& weight_matrices = inference ? m_weight_matrices_inference : m_weight_matrices;
		return weight_matrices.front();
	}

	GPUMatrix<T, MatrixLayout::RowMajor>& weight_matrix_at(bool inference, uint32_t idx) {
		auto& weight_matrices = inference ? m_weight_matrices_inference : m_weight_matrices;
		return weight_matrices.at(1 + idx);
	}

	GPUMatrix<T, MatrixLayout::RowMajor>& output_weight_matrix(bool inference) {
		auto& weight_matrices = inference ? m_weight_matrices_inference : m_weight_matrices;
		return weight_matrices.back();
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

	template <MatrixLayout output_layout>
	void compute_activation(cudaStream_t stream, Activation activation, const GPUMatrix<T, output_layout>& input, GPUMatrix<T, output_layout>& output) {
		if (input.n() != output.n() || input.m() != output.m()) {
			throw std::runtime_error(std::string("Input and output don't have matching size: ") + std::to_string(input.n()) + "!=" + std::to_string(output.n()));
		}

		const uint32_t width = input.m();

		switch (activation) {
			case Activation::None: break;
			case Activation::Exponential: exp<T><<<n_blocks_linear(input.n_elements()), n_threads_linear, 0, stream>>>(input.n_elements(), input.data(), output.data()); break;
			case Activation::ReLU: relu<T><<<n_blocks_linear(input.n_elements()), n_threads_linear, 0, stream>>>(input.n_elements(), input.data(), output.data()); break;
			case Activation::Sine: sin<T><<<n_blocks_linear(input.n_elements()), n_threads_linear, 0, stream>>>(input.n_elements(), input.data(), output.data()); break;
			default: throw std::runtime_error{"Unsupported activation."};
		}
	}

	template <typename CutlassLayer, MatrixLayout input_layout, MatrixLayout output_layout>
	bool compute_layer(
		cudaStream_t stream,
		bool is_inference,
		Activation activation,
		const GPUMatrix<T, MatrixLayout::RowMajor>& weights,
		const GPUMatrix<T, input_layout>& input,
		GPUMatrix<T, output_layout>& output,
		GPUMatrix<T, output_layout>& activation_output,
		T activation_param = (T)0
	) {
		bool can_fuse_activation = true;
		if (!is_inference) {
			// Never disallow fusing if the caller passes the same output and activation_output buffers... in that case,
			// invertibility of the activation function may be ignored.
			can_fuse_activation &= activation == Activation::None || activation == Activation::ReLU || &output == &activation_output;
		}

		if (can_fuse_activation) {
			switch (activation) {
				case Activation::None: fc_multiply<Activation::None, CutlassLayer>(stream, weights, input, output, activation_param); break;
				case Activation::Exponential: fc_multiply<Activation::Exponential, CutlassLayer>(stream, weights, input, output, activation_param); break;
				case Activation::ReLU: fc_multiply<Activation::ReLU, CutlassLayer>(stream, weights, input, output, activation_param); break;
				case Activation::Sine: fc_multiply<Activation::Sine, CutlassLayer>(stream, weights, input, output, activation_param); break;
				default: throw std::runtime_error{"Unsupported activation."};
			}
		} else {
			fc_multiply<Activation::None, CutlassLayer>(stream, weights, input, output);
			compute_activation(stream, activation, output, activation_output);
		}

		return can_fuse_activation;
	}

	template <typename CutlassLayer, MatrixLayout input_layout, MatrixLayout output_layout>
	bool compute_inference_layer(
		cudaStream_t stream,
		Activation activation,
		const GPUMatrix<T, MatrixLayout::RowMajor>& weights,
		const GPUMatrix<T, input_layout>& input,
		GPUMatrix<T, output_layout>& output,
		T activation_param = (T)0
	) {
		return compute_layer<CutlassLayer>(stream, true, activation, weights, input, output, output, activation_param);
	}

	uint32_t m_n_hidden_layers;
	uint32_t m_n_hidden_matmuls;
	uint32_t m_input_width;
	uint32_t m_network_width;
	uint32_t m_output_width;
	uint32_t m_padded_output_width;

	Activation m_activation;
	Activation m_output_activation;
	float m_output_activation_param = 0;

	static const uint32_t tensorcore_width = 8;

	// Streams and events
	std::vector<cudaStream_t> m_training_splitk_streams;
	std::vector<cudaEvent_t> m_training_splitk_events;

	// Graphs
	CudaGraph m_inference_graph;

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
	size_t m_total_n_params;

	std::vector<GPUMatrix<float, MatrixLayout::RowMajor>> m_weight_matrices_full_precision;

	std::vector<GPUMatrix<T, MatrixLayout::RowMajor>> m_gradient_matrices;
};

TCNN_NAMESPACE_END
