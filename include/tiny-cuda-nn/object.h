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

/** @file   object.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Abstract interface of objects in the tiny-cuda-nn framework
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>

#include <json/json.hpp>

#include <pcg32/pcg32.h>

#include <memory>

TCNN_NAMESPACE_BEGIN

using json = nlohmann::json;

class Object {
public:
	virtual ~Object() { }

	virtual json hyperparams() const = 0;

	std::string name() const {
		return hyperparams().value("otype", "<Unknown>");
	}
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

	virtual void set_params(PARAMS_T* params, PARAMS_T* inference_params, PARAMS_T* backward_params, PARAMS_T* gradients) = 0;
	virtual void initialize_params(pcg32& rnd, float* params_full_precision, PARAMS_T* params, PARAMS_T* inference_params, PARAMS_T* backward_params, PARAMS_T* gradients, float scale = 1) = 0;
	virtual size_t n_params() const = 0;

	virtual std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const = 0;
};

template <typename T>
void one_hot_batched(cudaStream_t stream, const uint32_t num_elements, const uint32_t width, const uint32_t one_hot_dim, T* out, float scale);

template <typename T>
void mult(cudaStream_t stream, const uint32_t num_elements, T* inout, float factor);

enum class EGradientMode {
	Ignore,
	Overwrite,
	Accumulate,
};

template <typename T, typename PARAMS_T, typename COMPUTE_T=T>
class DifferentiableObject : public ParametricObject<PARAMS_T> {
public:
	virtual ~DifferentiableObject() { }

	virtual void inference(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<float>& output) = 0;
	void inference(const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<float>& output) {
		inference(nullptr, input, output);
	}

	virtual std::unique_ptr<Context> forward(
		cudaStream_t stream,
		const GPUMatrixDynamic<T>& input,
		GPUMatrixDynamic<COMPUTE_T>* output = nullptr,
		bool use_inference_matrices = false,
		bool prepare_input_gradients = false
	) = 0;
	std::unique_ptr<Context> forward(
		const GPUMatrixDynamic<T>& input,
		GPUMatrixDynamic<COMPUTE_T>* output = nullptr,
		bool use_inference_matrices = false,
		bool prepare_input_gradients = false
	) {
		return forward(nullptr, input, output, use_inference_matrices, prepare_input_gradients);
	}

	virtual void backward(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<COMPUTE_T>& output,
		const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_matrices = false,
		EGradientMode param_gradients_mode = EGradientMode::Overwrite
	) = 0;
	void backward(
		const Context& ctx,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<COMPUTE_T>& output,
		const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_matrices = false,
		EGradientMode param_gradients_mode = EGradientMode::Overwrite
	) {
		backward(nullptr, input, output, dL_doutput, dL_dinput, use_inference_matrices, param_gradients_mode);
	}

	void input_gradient(
		cudaStream_t stream,
		uint32_t dim,
		const GPUMatrix<T>& input,
		GPUMatrix<T>& d_dinput,
		float backprop_scale = 128.0f // Prevents underflows during half-precision backprop. Same reason for loss_scale to exist.
	) {
		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		GPUMatrix<COMPUTE_T> d_doutput = {padded_output_width(), batch_size, stream};
		GPUMatrix<COMPUTE_T> output = {padded_output_width(), batch_size, stream};

		if (dim >= padded_output_width()) {
			throw std::runtime_error{"Invalid dimension to compute the input gradient for."};
		}

		// Set "loss gradient" at network outputs to 1 at the chosen dimension and 0 elsewhere.
		one_hot_batched(stream, output.n_elements(), padded_output_width(), dim, d_doutput.data(), backprop_scale);

		auto ctx = forward(stream, input, &output, true /* inference matrices */, true /* prep forward buffers for input gradients */);
		backward(stream, *ctx, input, output, d_doutput, &d_dinput, true /* inference matrices */, EGradientMode::Ignore);

		mult(stream, d_dinput.n_elements(), d_dinput.data(), 1.0f / backprop_scale);
	}

	virtual uint32_t padded_output_width() const = 0;
	virtual uint32_t output_width() const = 0;

	virtual uint32_t required_input_alignment() const = 0;
};

TCNN_NAMESPACE_END
