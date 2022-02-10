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

/** @file   network.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface of a neural network implementation
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>


TCNN_NAMESPACE_BEGIN

enum class WeightUsage {
	Inference,
	Forward,
	Backward,
};

template <typename T>
void extract_dimension_pos_neg(cudaStream_t stream, const uint32_t num_elements, const uint32_t dim, const uint32_t fan_in, const uint32_t fan_out, const T* encoded, float* output);

template <typename T, typename PARAMS_T=T>
class Network : public DifferentiableObject<T, PARAMS_T, PARAMS_T> {
public:
	virtual ~Network() { }

	virtual void inference_mixed_precision(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<PARAMS_T>& output, bool use_inference_matrices = true) = 0;
	void inference_mixed_precision(const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<PARAMS_T>& output, bool use_inference_matrices = true) {
		inference_mixed_precision(nullptr, input, output, use_inference_matrices);
	}

	void visualize_activation(cudaStream_t stream, uint32_t layer, uint32_t dimension, const GPUMatrix<T>& input, GPUMatrix<float>& output) {
		layer = std::min(layer, num_forward_activations()-1);
		dimension = std::min(dimension, width(layer)-1);

		this->forward(stream, input);
		extract_dimension_pos_neg<PARAMS_T>(stream, output.n_elements(), dimension, width(layer), output.rows(), forward_activations(layer), output.data());
		this->forward_clear();
	}

	virtual uint32_t width(uint32_t layer) const = 0;
	virtual uint32_t num_forward_activations() const = 0;
	virtual const PARAMS_T* forward_activations(uint32_t layer) const = 0;
};

template <typename T>
Network<T, T>* create_network(const json& network);

TCNN_NAMESPACE_END
