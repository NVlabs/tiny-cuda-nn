/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

namespace tcnn {

template <typename T, uint32_t WIDTH>
class FullyFusedMLP : public Network<T> {
public:
	FullyFusedMLP(uint32_t input_width, uint32_t output_width, uint32_t n_hidden_layers, Activation activation, Activation output_activation);

#if !defined(TCNN_NO_FWD_BWD)
	void inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) override;

	std::unique_ptr<Context> forward_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override;

	void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override;
#endif // !defined(TCNN_NO_FWD_BWD)

	void set_params_impl(T* params, T* inference_params, T* gradients) override;
	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override;

	GPUMatrix<T, RM>& input_weight_matrix(bool inference) {
		auto& weight_matrices = inference ? m_weight_matrices_inference : m_weight_matrices;
		return weight_matrices.front();
	}

	GPUMatrix<T, RM>& weight_matrix_at(bool inference, uint32_t idx) {
		auto& weight_matrices = inference ? m_weight_matrices_inference : m_weight_matrices;
		return weight_matrices.at(1 + idx);
	}

	GPUMatrix<T, RM>& output_weight_matrix(bool inference) {
		auto& weight_matrices = inference ? m_weight_matrices_inference : m_weight_matrices;
		return weight_matrices.back();
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

	uint32_t input_width() const override {
		return m_input_width;
	}

	uint32_t padded_output_width() const override {
		return m_padded_output_width;
	}

	uint32_t output_width() const override {
		return m_output_width;
	}

	static uint32_t REQUIRED_ALIGNMENT() {
		return 16; // Uses 16x16x16 tensor ops
	}

	uint32_t required_input_alignment() const override {
		return REQUIRED_ALIGNMENT();
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

	std::pair<const T*, MatrixLayout> forward_activations(const Context& ctx, uint32_t layer) const override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		return {forward.hidden.at(layer).data(), CM};
	}

	json hyperparams() const override {
		return {
			{"otype", "FullyFusedMLP"},
			{"activation", to_string(m_activation)},
			{"output_activation", to_string(m_output_activation)},
			{"n_neurons", m_network_width},
			{"n_hidden_layers", m_n_hidden_layers},
		};
	}

	std::string generate_device_function(const std::string& name) const override {
		return this->generate_device_function_from_body(
			name,
			generate_mlp_device_code<T>(
				m_input_width,
				WIDTH,
				m_padded_output_width,
				m_output_width,
				m_n_hidden_layers,
				m_activation,
				m_output_activation
			)
		);
	}

	std::string generate_backward_device_function(const std::string& name, uint32_t n_threads) const override {
		return this->generate_backward_device_function_from_body(
			name,
			generate_backward_mlp_device_code<T>(
				n_threads,
				m_input_width,
				WIDTH,
				m_padded_output_width,
				m_output_width,
				m_n_hidden_layers,
				m_activation,
				m_output_activation
			)
		);
	}

	uint32_t device_function_fwd_ctx_bytes() const override {
		return mlp_device_code_fwd_ctx_bytes<T>(
			m_input_width,
			WIDTH,
			m_padded_output_width,
			m_output_width,
			m_n_hidden_layers,
			m_activation,
			m_output_activation
		);
	}

	bool device_function_fwd_ctx_aligned_per_element() const override {
		return false;
	}

	uint32_t backward_device_function_shmem_bytes(uint32_t n_threads, GradientMode param_gradients_mode) const override {
		return backward_mlp_device_code_shmem_bytes<T>(n_threads, param_gradients_mode, m_input_width, WIDTH, m_padded_output_width);
	}

	void convert_params_to_jit_layout(cudaStream_t stream, bool use_inference_params) override {
		if (!m_convert_params_to_jit_layout_kernel) {
			m_convert_params_to_jit_layout_kernel = generate_mlp_convert_params_to_jit_layout_kernel<T>(
				m_input_width, m_network_width, m_padded_output_width, m_n_hidden_layers
			);
		}

		m_convert_params_to_jit_layout_kernel->launch(m_n_hidden_layers + 1, WARP_SIZE, 0, stream, use_inference_params ? this->inference_params() : this->params());
	}

	void convert_params_from_jit_layout(cudaStream_t stream, bool use_inference_params) override {
		if (!m_convert_params_from_jit_layout_kernel) {
			m_convert_params_from_jit_layout_kernel = generate_mlp_convert_params_from_jit_layout_kernel<T>(
				m_input_width, m_network_width, m_padded_output_width, m_n_hidden_layers
			);
		}

		m_convert_params_from_jit_layout_kernel->launch(m_n_hidden_layers + 1, WARP_SIZE, 0, stream, use_inference_params ? this->inference_params() : this->params());
	}

private:
	std::unique_ptr<CudaRtcKernel> m_convert_params_to_jit_layout_kernel;
	std::unique_ptr<CudaRtcKernel> m_convert_params_from_jit_layout_kernel;

	struct ForwardContext : public Context {
		std::vector<GPUMatrix<T>> hidden;
		GPUMemoryArena::Allocation alloc;
	};

	std::unique_ptr<ForwardContext> allocate_forward_buffers(cudaStream_t stream, uint32_t batch_size);

	uint32_t m_n_hidden_layers;
	uint32_t m_n_hidden_matmuls;
	uint32_t m_input_width;
	uint32_t m_network_width;
	uint32_t m_output_width;
	uint32_t m_padded_output_width;

	Activation m_activation;
	Activation m_output_activation;

	// Storage of params
	std::vector<GPUMatrix<T, RM>> m_weight_matrices;
	std::vector<GPUMatrix<T, RM>> m_weight_matrices_inference;
	size_t m_total_n_params;

	std::vector<GPUMatrix<T, RM>> m_gradient_matrices;
};

}
