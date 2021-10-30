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

/** @file   network_with_input_encoding.h
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
	} else if (dim_idx == 2) {
		output[i] = 0;
	} else {
		output[i] = 1;
	}
}

template <typename T>
__global__ void one_hot_batched(const uint32_t num_elements, const uint32_t width, const uint32_t one_hot_dim, T* out, float scale) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint32_t dim = i % width;
	out[i] = dim == one_hot_dim ? (T)scale : (T)0.0f;
}

__global__ void mult(const uint32_t num_elements, float* __restrict__ inout, float factor) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	inout[i] *= factor;
}

template <typename T>
class NetworkWithInputEncoding : public DifferentiableObject<float, T, T> {
public:
	NetworkWithInputEncoding(Encoding<T>* encoding, uint32_t n_output_dims, const json& network) : m_encoding{encoding} {
		json local_network_config = network;
		local_network_config["n_input_dims"] = m_encoding->num_encoded_dims();
		local_network_config["n_output_dims"] = n_output_dims;
		m_network.reset(create_network<T>(local_network_config));
	}

	NetworkWithInputEncoding(uint32_t n_dims_to_encode, uint32_t n_dims_to_pass_through, uint32_t n_output_dims, const json& encoding, const json& network)
	: NetworkWithInputEncoding(create_encoding<T>(
		n_dims_to_encode,
		n_dims_to_pass_through,
		encoding,
		network.contains("otype") && (equals_case_insensitive(network["otype"], "FullyFusedMLP") || equals_case_insensitive(network["otype"], "MegakernelMLP")) ? 16u : 8u
	), n_output_dims, network) { }

	virtual ~NetworkWithInputEncoding() { }

	void inference(cudaStream_t stream, const GPUMatrix<float>& input, GPUMatrix<float>& output) override {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_inference_network_input.n() != batch_size) {
			allocate_inference_buffers(batch_size);
		}

		m_encoding->encode(input.n(), input.data(), m_inference_network_input.data(), stream, nullptr, true);
		m_network->inference(stream, m_inference_network_input, output);
	}

	void inference_mixed_precision(cudaStream_t stream, const GPUMatrix<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_matrices = true) {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_inference_network_input.n() != batch_size) {
			allocate_inference_buffers(batch_size);
		}

		m_encoding->encode(input.n(), input.data(), m_inference_network_input.data(), stream, nullptr, use_inference_matrices);
		m_network->inference_mixed_precision(stream, m_inference_network_input, output, use_inference_matrices);
	}

	void visualize_activation(cudaStream_t stream, uint32_t layer, uint32_t dimension, const GPUMatrix<float>& input, GPUMatrix<float>& output) {
		layer = std::min(layer, num_forward_activations()-1);
		dimension = std::min(dimension, width(layer)-1);

		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_forward_network_input.n() != batch_size) {
			allocate_forward_buffers(batch_size);
		}

		m_encoding->encode(input.n(), input.data(), m_forward_network_input.data(), stream, nullptr, false);
		m_network->forward(stream, m_forward_network_input, m_forward_network_output, false);
		extract_dimension_pos_neg<T><<<n_blocks_linear(output.n_elements()), n_threads_linear, 0, stream>>>(output.n_elements(), dimension, width(layer), output.rows(), forward_activations(layer), output.data());
	}

	uint32_t num_encoded_dims() const {
		return m_encoding->num_encoded_dims();
	}

	void forward(cudaStream_t stream, const GPUMatrix<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_matrices = false, bool prepare_input_gradients = false) override {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_forward_network_input.n() != batch_size) {
			allocate_forward_buffers(batch_size);
		}

		m_encoding->encode(input.n(), input.data(), m_forward_network_input.data(), stream, prepare_input_gradients ? m_forward_encoding_forward_gradient.data() : nullptr, use_inference_matrices);
		m_network->forward(stream, m_forward_network_input, output, use_inference_matrices, prepare_input_gradients);
	}

	void backward(
		cudaStream_t stream,
		const GPUMatrix<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrix<float>* dL_dinput = nullptr,
		bool use_inference_matrices = false,
		bool compute_param_gradients = true
	) override {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_backward_dL_dnetwork_input.n() != batch_size) {
			allocate_backward_buffers(batch_size);
		}

		GPUMatrix<T>* dL_dnetwork_input = nullptr;
		if (m_encoding->n_params() > 0 || dL_dinput) {
			dL_dnetwork_input = &m_backward_dL_dnetwork_input;
		}

		m_network->backward(stream, m_forward_network_input, output, dL_doutput, dL_dnetwork_input, use_inference_matrices, compute_param_gradients);
		if (dL_dnetwork_input) {
			m_encoding->backward(stream, input.n(), dL_dnetwork_input->data(), dL_dinput ? m_forward_encoding_forward_gradient.data() : nullptr, dL_dinput ? dL_dinput->data() : nullptr, input.data());
		}
	}

	void input_gradient(
		cudaStream_t stream,
		uint32_t dim,
		const GPUMatrix<float>& input,
		GPUMatrix<float>& d_dinput,
		float backprop_scale = 128.0f // Prevents underflows during half-precision backprop. Same reason for loss_scale to exist.
	) {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_input_gradient_output.n() != batch_size) {
			allocate_input_gradient_buffers(batch_size);
		}

		if (dim >= padded_output_width()) {
			throw std::runtime_error{"Invalid dimension to compute the input gradient for."};
		}

		// Set "loss gradient" at network outputs to 1 at the chosen dimension and 0 elsewhere.
		linear_kernel(one_hot_batched<T>, 0, stream,
			m_input_gradient_output.n_elements(), padded_output_width(), dim, m_input_gradient_d_doutput.data(), backprop_scale
		);

		forward(stream, input, m_input_gradient_output, true /* inference matrices */, true /* prep forward buffers for input gradients */);
		backward(stream, input, m_input_gradient_output, m_input_gradient_d_doutput, &d_dinput, true /* inference matrices */, false /* no param gradients */);

		linear_kernel(mult, 0, stream,
			d_dinput.n_elements(), d_dinput.data(), 1.0f / backprop_scale
		);
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

	uint32_t width(uint32_t layer) const {
		return layer == 0 ? m_encoding->num_encoded_dims() : m_network->width();
	}

	uint32_t num_forward_activations() const {
		return m_network->num_forward_activations() + 1;
	}

	const T* forward_activations(uint32_t layer) const {
		return layer == 0 ? m_forward_network_input.data() : m_network->forward_activations(layer - 1);
	}

	const Encoding<T>* encoding() const {
		return m_encoding.get();
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
		m_forward_network_output.set_size(m_network->padded_output_width(), batch_size);
		m_forward_encoding_forward_gradient.set_size(m_encoding->num_forward_gradient_dims(), batch_size);

		GPUMatrixBase::allocate_shared_memory(
			m_forward_buffer,
			{
				&m_forward_network_input,
				&m_forward_network_output,
				&m_forward_encoding_forward_gradient,
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
		m_input_gradient_d_doutput.set_size(m_network->padded_output_width(), batch_size);
		m_input_gradient_output.set_size(m_network->padded_output_width(), batch_size);

		GPUMatrixBase::allocate_shared_memory(
			m_input_gradient_buffer,
			{
				&m_input_gradient_encoding_forward_gradient,
				&m_input_gradient_network_input,
				&m_input_gradient_d_doutput,
				&m_input_gradient_output,
			}
		);
	}

private:
	std::unique_ptr<Network<T>> m_network;
	std::unique_ptr<Encoding<T>> m_encoding;

	// Temporary buffers to hold inference data
	GPUMemory<char> m_inference_buffer;
	GPUMatrix<T> m_inference_network_input;

	// Temporary buffers to hold forward data
	GPUMemory<char> m_forward_buffer;
	GPUMatrix<T> m_forward_network_input;
	GPUMatrix<T> m_forward_network_output; // Only needed when visualizing
	GPUMatrix<float> m_forward_encoding_forward_gradient; // Only needed when computing input gradients

	// Temporary buffers to hold backward data
	GPUMemory<char> m_backward_buffer;
	GPUMatrix<T> m_backward_dL_dnetwork_input;

	// Temporary buffers to input gradient buffers
	GPUMemory<char> m_input_gradient_buffer;
	GPUMatrix<float> m_input_gradient_encoding_forward_gradient;
	GPUMatrix<T> m_input_gradient_network_input;
	GPUMatrix<T> m_input_gradient_d_doutput;
	GPUMatrix<T> m_input_gradient_output;
};

TCNN_NAMESPACE_END
