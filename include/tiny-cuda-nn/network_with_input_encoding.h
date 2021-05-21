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

/** @file   model_with_input_encoding.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  A model that includes its encoding
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>


TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void extract_dimension_pos_neg(const uint32_t num_elements, const uint32_t dim, const uint32_t fan_in, const uint32_t fan_out, const T* __restrict__ encoded, float* __restrict__ output) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint32_t elem_idx = i / fan_out;
	const uint32_t dim_idx = i % fan_out;

	if (dim_idx == 0) {
		output[i] = fmaxf(-(float)encoded[elem_idx * fan_in + dim], 0.0f);
	} else if (dim_idx == 1) {
		output[i] = fmaxf((float)encoded[elem_idx * fan_in + dim], 0.0f);
	} else {
		output[i] = 0;
	}
}

template <typename T>
__global__ void one_hot_batched(const uint32_t num_elements, const uint32_t width, const uint32_t one_hot_dim, T* out) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint32_t dim = i % width;
	out[i] = dim == one_hot_dim ? (T)1.0f : (T)0.0f;
}

template <typename T>
class NetworkWithInputEncoding : public DifferentiableObject<float, T, T> {
public:
	NetworkWithInputEncoding(uint32_t n_dims_to_encode, uint32_t n_dims_to_pass_through, uint32_t n_output_dims, json encoding, json network) {
		// TODO: make this more automatic
		uint32_t alignment = 8;
		if (network.contains("otype") && (equals_case_insensitive(network["otype"], "FullyFusedMLP") || equals_case_insensitive(network["otype"], "MegakernelMLP"))) {
			alignment = 16;
		}

		m_encoding.reset(create_encoding<T>(
			n_dims_to_encode,
			n_dims_to_pass_through,
			encoding,
			alignment
		));

		network["n_input_dims"] = m_encoding->num_encoded_dims();
		network["n_output_dims"] = n_output_dims;
		m_network.reset(create_network<T>(network));

		std::cout << "Created NetworkWithInputEncoding with dimensionality Encoding(" << n_dims_to_encode << "," << n_dims_to_pass_through << ")->Network(" << m_encoding->num_encoded_dims() << ")->Output(" << n_output_dims << ")." << std::endl;
	}

	virtual ~NetworkWithInputEncoding() { }

	void inference(cudaStream_t stream, const GPUMatrix<float, MatrixLayout::ColumnMajor>& input, GPUMatrix<float, MatrixLayout::ColumnMajor>& output) override {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_inference_network_input.n() != batch_size) {
			allocate_inference_buffers(batch_size);
		}

		m_encoding->encode(input.n(), input.data(), m_inference_network_input.data(), stream, nullptr, true);
		m_network->inference(stream, m_inference_network_input, output);
	}

	void visualize_encoding(cudaStream_t stream, uint32_t dimension, const GPUMatrix<float, MatrixLayout::ColumnMajor>& input, GPUMatrix<float, MatrixLayout::ColumnMajor>& output) {
		dimension = std::min(dimension, m_encoding->num_encoded_dims()-1);

		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_inference_network_input.n() != batch_size) {
			allocate_inference_buffers(batch_size);
		}

		m_encoding->encode(input.n(), input.data(), m_inference_network_input.data(), stream, nullptr, true);
		extract_dimension_pos_neg<T><<<n_blocks_linear(output.n_elements()), n_threads_linear, 0, stream>>>(output.n_elements(), dimension, m_inference_network_input.rows(), output.rows(), m_inference_network_input.data(), output.data());
	}

	void forward(cudaStream_t stream, const GPUMatrix<float, MatrixLayout::ColumnMajor>& input, GPUMatrix<T, MatrixLayout::ColumnMajor>& output, MatrixLayout output_layout, bool use_inference_matrices) override {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_forward_network_input.n() != batch_size) {
			allocate_forward_buffers(batch_size);
		}

		m_encoding->encode(input.n(), input.data(), m_forward_network_input.data(), stream, nullptr, false);
		m_network->forward(stream, m_forward_network_input, output, output_layout, use_inference_matrices);
	}

	void backward(
		cudaStream_t stream,
		const GPUMatrix<float, MatrixLayout::ColumnMajor>& input,
		const GPUMatrix<T, MatrixLayout::ColumnMajor>& output,
		const GPUMatrix<T, MatrixLayout::ColumnMajor>& dL_doutput,
		GPUMatrix<T, MatrixLayout::ColumnMajor>* dL_dinput,
		MatrixLayout output_layout,
		bool use_inference_matrices,
		bool compute_param_gradients
	) override {
		if (dL_dinput != nullptr) {
			throw std::runtime_error{
				"NetworkWithInputEncoding::backward currently does not support computing loss gradient w.r.t. the input."
			};
		}

		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_backward_dL_dnetwork_input.n() != batch_size) {
			allocate_backward_buffers(batch_size);
		}

		GPUMatrix<T, MatrixLayout::ColumnMajor>* dL_dnetwork_input = nullptr;
		if (m_encoding->n_params() > 0) {
			dL_dnetwork_input = &m_backward_dL_dnetwork_input;
		}

		m_network->backward(stream, m_forward_network_input, output, dL_doutput, dL_dnetwork_input, output_layout, use_inference_matrices, compute_param_gradients);
		if (dL_dnetwork_input) {
			m_encoding->backward(stream, input.n(), dL_dnetwork_input->data(), nullptr, nullptr, input.data());
		}
	}

	void input_gradient(
		cudaStream_t stream,
		uint32_t dim,
		const GPUMatrix<float, MatrixLayout::ColumnMajor>& input,
		GPUMatrix<float, MatrixLayout::ColumnMajor>& d_dinput
	) {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_input_gradient_output.n() != batch_size) {
			allocate_input_gradient_buffers(batch_size);
		}

		m_encoding->encode(input.n(), input.data(), m_input_gradient_network_input.data(), stream, m_input_gradient_encoding_forward_gradient.data(), true);
		m_network->forward(
			stream,
			m_input_gradient_network_input,
			m_input_gradient_output,
			MatrixLayout::ColumnMajor,
			true /* Use inference weights */
		);

		if (dim >= m_network->padded_output_width()) {
			throw std::runtime_error{"Invalid dimension to compute the input gradient for."};
		}

		// Set "loss gradient" at network outputs to 1 at the chosen dimension and 0 elsewhere.
		one_hot_batched<T><<<n_blocks_linear(m_input_gradient_output.n_elements()), n_threads_linear, 0, stream>>>(m_input_gradient_output.n_elements(), m_network->padded_output_width(), dim, m_input_gradient_d_doutput.data());

		m_network->backward(
			stream,
			m_input_gradient_network_input,
			m_input_gradient_output,
			m_input_gradient_d_doutput,
			&m_input_gradient_d_dnetwork_input,
			MatrixLayout::ColumnMajor,
			true /* Use inference weights */,
			false /* Don't compute parameter gradients. We don't wanna optimize with this. */
		);
		m_encoding->backward(stream, input.n(), m_input_gradient_d_dnetwork_input.data(), m_input_gradient_encoding_forward_gradient.data(), d_dinput.data(), input.data());
	}

	void initialize_params(std::mt19937& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		size_t offset = 0;
		m_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_network->n_params();

		m_encoding->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_encoding->n_params();
	}

	size_t n_params() const override {
		return m_encoding->n_params() + m_network->n_params();
	}

	uint32_t padded_output_width() const override {
		return m_network->padded_output_width();
	}

	uint32_t output_width() const override {
		return m_network->output_width();
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required due to encoding
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		return m_network->layer_sizes();
	}

private:
	void allocate_inference_buffers(uint32_t batch_size) {
		m_inference_network_input.set_size(m_encoding->num_encoded_dims(), batch_size);

		GPUMatrixBase::allocate_shared_memory(
			m_inference_buffer,
			{
				&m_inference_network_input,
			}
		);
	}

	void allocate_forward_buffers(uint32_t batch_size) {
		m_forward_network_input.set_size(m_encoding->num_encoded_dims(), batch_size);

		GPUMatrixBase::allocate_shared_memory(
			m_forward_buffer,
			{
				&m_forward_network_input,
			}
		);
	}

	void allocate_backward_buffers(uint32_t batch_size) {
		m_backward_dL_dnetwork_input.set_size(m_encoding->num_encoded_dims(), batch_size);

		GPUMatrixBase::allocate_shared_memory(
			m_backward_buffer,
			{
				&m_backward_dL_dnetwork_input,
			}
		);
	}

	void allocate_input_gradient_buffers(uint32_t batch_size) {
		m_input_gradient_encoding_forward_gradient.set_size(m_encoding->num_forward_gradient_dims(), batch_size);
		m_input_gradient_network_input.set_size(m_encoding->num_encoded_dims(), batch_size);
		m_input_gradient_d_dnetwork_input.set_size(m_encoding->num_encoded_dims(), batch_size);
		m_input_gradient_d_doutput.set_size(m_network->padded_output_width(), batch_size);
		m_input_gradient_output.set_size(m_network->padded_output_width(), batch_size);

		GPUMatrixBase::allocate_shared_memory(
			m_input_gradient_buffer,
			{
				&m_input_gradient_encoding_forward_gradient,
				&m_input_gradient_network_input,
				&m_input_gradient_d_dnetwork_input,
				&m_input_gradient_d_doutput,
				&m_input_gradient_output,
			}
		);
	}

	std::unique_ptr<Network<T>> m_network;
	std::unique_ptr<Encoding<T>> m_encoding;

	// Temporary buffers to hold inference data
	GPUMemory<char> m_inference_buffer;
	GPUMatrix<T, MatrixLayout::ColumnMajor> m_inference_network_input;

	// Temporary buffers to hold forward data
	GPUMemory<char> m_forward_buffer;
	GPUMatrix<T, MatrixLayout::ColumnMajor> m_forward_network_input;

	// Temporary buffers to hold backward data
	GPUMemory<char> m_backward_buffer;
	GPUMatrix<T, MatrixLayout::ColumnMajor> m_backward_dL_dnetwork_input;

	// Temporary buffers to input gradient buffers
	GPUMemory<char> m_input_gradient_buffer;
	GPUMatrix<float, MatrixLayout::ColumnMajor> m_input_gradient_encoding_forward_gradient;
	GPUMatrix<T, MatrixLayout::ColumnMajor> m_input_gradient_network_input;
	GPUMatrix<T, MatrixLayout::ColumnMajor> m_input_gradient_d_dnetwork_input;
	GPUMatrix<T, MatrixLayout::ColumnMajor> m_input_gradient_d_doutput;
	GPUMatrix<T, MatrixLayout::ColumnMajor> m_input_gradient_output;
};

TCNN_NAMESPACE_END
