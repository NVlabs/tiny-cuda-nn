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

/** @file   network.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface of a neural network implementation
 */

#pragma once

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/object.h>

namespace tcnn {

template <typename T>
void extract_dimension_pos_neg(cudaStream_t stream, const uint32_t num_elements, const uint32_t dim, const uint32_t fan_in, const uint32_t fan_out, const T* encoded, MatrixLayout layout, float* output);

template <typename T, typename PARAMS_T=T>
class Network : public DifferentiableObject<T, PARAMS_T, PARAMS_T> {
public:
	virtual ~Network() { }

	void visualize_activation(cudaStream_t stream, uint32_t layer, uint32_t dimension, const GPUMatrix<T>& input, GPUMatrix<float>& output) {
		layer = std::min(layer, num_forward_activations()-1);
		dimension = std::min(dimension, width(layer)-1);

		auto ctx = this->forward(stream, input);
		auto vals = forward_activations(*ctx, layer);
		extract_dimension_pos_neg<PARAMS_T>(stream, output.n_elements(), dimension, width(layer), output.rows(), vals.first, vals.second, output.data());
	}

	virtual uint32_t width(uint32_t layer) const = 0;
	virtual uint32_t num_forward_activations() const = 0;
	virtual std::pair<const PARAMS_T*, MatrixLayout> forward_activations(const Context& ctx, uint32_t layer) const = 0;
};

template <typename T>
Network<T, T>* create_network(const json& network);

template <typename T>
std::unique_ptr<Network<T>> default_network(uint32_t n_input_dims, uint32_t n_output_dims, const std::string& name) {
	return std::unique_ptr<Network<T>>{create_network<T>({{"otype", name}, {"n_input_dims", n_input_dims}, {"n_output_dims", n_output_dims}})};
}

std::vector<std::string> builtin_networks();

std::string select_network(const json& network);
uint32_t minimum_alignment(const json& network);

template <typename T>
void activation_gpu(cudaStream_t stream, const uint32_t num_elements, const Activation act, const T* in, T* out) {
	static constexpr uint32_t ACTIVATION_VECTOR_SIZE = 16u / sizeof(T);
	if (num_elements % ACTIVATION_VECTOR_SIZE != 0) {
		throw std::runtime_error{fmt::format("activation_gpu: number of elements must be a multiple of {}", ACTIVATION_VECTOR_SIZE)};
	}

	// Activation::None is a noop
	if (act == Activation::None && in == out) {
		return;
	}

	linear_kernel(kernel_activation<T, ACTIVATION_VECTOR_SIZE>, 0, stream, div_round_up(num_elements, ACTIVATION_VECTOR_SIZE), act, in, out);
}

template <typename T>
void activation_gpu(cudaStream_t stream, Activation activation, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output) {
	if (input.n() != output.n() || input.m() != output.m()) {
		throw std::runtime_error{fmt::format("Input and output don't have matching size: {} != {}", input.n(), output.n())};
	}

	activation_gpu(stream, input.n_elements(), activation, input.data(), output.data());
}

template <typename T>
void activation_backward_gpu(cudaStream_t stream, const uint32_t num_elements, const Activation act, const T* __restrict__ values, const T* gradients_out, T* gradients_in) {
	static constexpr uint32_t ACTIVATION_VECTOR_SIZE = 16u / sizeof(T);
	if (num_elements % ACTIVATION_VECTOR_SIZE != 0) {
		throw std::runtime_error{fmt::format("activation_backward_gpu: number of elements must be a multiple of {}", ACTIVATION_VECTOR_SIZE)};
	}

	// Activation transfer is a noop for Activation::None
	if (act == Activation::None && gradients_out == gradients_in) {
		return;
	}

	linear_kernel(kernel_activation_backward<T, ACTIVATION_VECTOR_SIZE>, 0, stream, div_round_up(num_elements, ACTIVATION_VECTOR_SIZE), act, values, gradients_out, gradients_in);
}

template <typename T>
void activation_backward_gpu(cudaStream_t stream, Activation activation, const GPUMatrixDynamic<T>& values, GPUMatrixDynamic<T>& gradients) {
	if (values.n() != gradients.n() || values.m() != gradients.m()) {
		throw std::runtime_error{fmt::format("Values and gradients don't have matching size: {} != {}", values.n(), gradients.n())};
	}

	activation_backward_gpu(stream, values.n_elements(), activation, values.data(), gradients.data(), gradients.data());
}

template <typename T>
void activation_backward_output_gpu(cudaStream_t stream, const uint32_t num_elements, const Activation act, const T* __restrict__ output_values, const T* gradients_out, T* gradients_in) {
	static constexpr uint32_t ACTIVATION_VECTOR_SIZE = 16u / sizeof(T);
	if (num_elements % ACTIVATION_VECTOR_SIZE != 0) {
		throw std::runtime_error{fmt::format("activation_backward_output_gpu: number of elements must be a multiple of {}", ACTIVATION_VECTOR_SIZE)};
	}

	// Activation transfer is a noop for Activation::None
	if (act == Activation::None && gradients_out == gradients_in) {
		return;
	}

	linear_kernel(kernel_activation_backward_output<T, ACTIVATION_VECTOR_SIZE>, 0, stream, div_round_up(num_elements, ACTIVATION_VECTOR_SIZE), act, output_values, gradients_out, gradients_in);
}

}
