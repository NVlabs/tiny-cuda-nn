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

/** @file   object.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Abstract interface of objects in the tiny-cuda-nn framework
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>

#include <json/json.hpp>

#include <pcg32/pcg32.h>


TCNN_NAMESPACE_BEGIN

using json = nlohmann::json;

template<typename T>
class GPUMatrixDynamic;

template<typename T, MatrixLayout _layout = MatrixLayout::ColumnMajor>
class GPUMatrix;

class Object {
public:
	virtual ~Object() { }
};

class ObjectWithMutableHyperparams : public Object {
public:
	virtual ~ObjectWithMutableHyperparams() { }

	virtual void update_hyperparams(const json& params) = 0;
};

template <typename PARAMS_T>
class ParametricObject : public Object {
public:
	virtual ~ParametricObject() { }

	virtual void initialize_params(pcg32& rnd, float* params_full_precision, PARAMS_T* params, PARAMS_T* inference_params, PARAMS_T* backward_params, PARAMS_T* gradients, float scale = 1) = 0;
	virtual size_t n_params() const = 0;

	virtual std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const = 0;
};

template <typename T>
void one_hot_batched(cudaStream_t stream, const uint32_t num_elements, const uint32_t width, const uint32_t one_hot_dim, T* out, float scale);

template <typename T>
void mult(cudaStream_t stream, const uint32_t num_elements, T* inout, float factor);

template <typename T, typename PARAMS_T, typename COMPUTE_T=T>
class DifferentiableObject : public ParametricObject<PARAMS_T> {
public:
	virtual ~DifferentiableObject() { }

	virtual void inference(cudaStream_t stream, const GPUMatrix<T>& input, GPUMatrix<float>& output) = 0;
	void inference(const GPUMatrix<T>& input, GPUMatrix<float>& output) {
		inference(nullptr, input, output);
	}

	virtual void forward(
		cudaStream_t stream,
		const GPUMatrix<T>& input,
		GPUMatrixDynamic<COMPUTE_T>* output = nullptr,
		bool use_inference_matrices = false,
		bool prepare_input_gradients = false
	) = 0;
	void forward(
		const GPUMatrix<T>& input,
		GPUMatrixDynamic<COMPUTE_T>* output = nullptr,
		bool use_inference_matrices = false,
		bool prepare_input_gradients = false
	) {
		forward(nullptr, input, output, use_inference_matrices, prepare_input_gradients);
	}

	virtual void backward(
		cudaStream_t stream,
		const GPUMatrix<T>& input,
		const GPUMatrixDynamic<COMPUTE_T>& output,
		const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
		GPUMatrix<T>* dL_dinput = nullptr,
		bool use_inference_matrices = false,
		bool compute_param_gradients = true
	) = 0;
	void backward(
		const GPUMatrix<T>& input,
		const GPUMatrixDynamic<COMPUTE_T>& output,
		const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
		GPUMatrix<T>* dL_dinput = nullptr,
		bool use_inference_matrices = false,
		bool compute_param_gradients = true
	) {
		backward(nullptr, input, output, dL_doutput, dL_dinput, use_inference_matrices, compute_param_gradients);
	}

	void input_gradient(
		cudaStream_t stream,
		uint32_t dim,
		const GPUMatrix<T>& input,
		GPUMatrix<T>& d_dinput,
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
		one_hot_batched(stream, m_input_gradient_output.n_elements(), padded_output_width(), dim, m_input_gradient_d_doutput.data(), backprop_scale);

		forward(stream, input, &m_input_gradient_output, true /* inference matrices */, true /* prep forward buffers for input gradients */);
		backward(stream, input, m_input_gradient_output, m_input_gradient_d_doutput, &d_dinput, true /* inference matrices */, false /* no param gradients */);

		mult(stream, d_dinput.n_elements(), d_dinput.data(), 1.0f / backprop_scale);
	}

	virtual uint32_t padded_output_width() const = 0;
	virtual uint32_t output_width() const = 0;

	virtual uint32_t required_input_alignment() const = 0;

private:
	void allocate_input_gradient_buffers(uint32_t batch_size) {
		m_input_gradient_d_doutput.set_size(padded_output_width(), batch_size);
		m_input_gradient_output.set_size(padded_output_width(), batch_size);

		GPUMatrixBase::allocate_shared_memory(
			m_input_gradient_buffer,
			{
				&m_input_gradient_d_doutput,
				&m_input_gradient_output,
			}
		);
	}

	// Temporary buffers for computing input gradients.
	// (Lazily allocated on demand.)
	GPUMemory<char> m_input_gradient_buffer;
	GPUMatrix<COMPUTE_T> m_input_gradient_d_doutput;
	GPUMatrix<COMPUTE_T> m_input_gradient_output;
};

TCNN_NAMESPACE_END
