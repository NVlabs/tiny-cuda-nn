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
#include <tiny-cuda-nn/matrix_layout.h>

#include <json/json.hpp>

#include <random>


TCNN_NAMESPACE_BEGIN

using json = nlohmann::json;

template<typename T, MatrixLayout _layout>
class GPUMatrix;

class Object {
public:
	virtual ~Object() { }
};

class ObjectWithMutableHyperparams : public Object {
public:
	virtual ~ObjectWithMutableHyperparams() { }

	virtual void update_hyperparams(json params) = 0;
};

template <typename PARAMS_T>
class ParametricObject : public Object {
public:
	virtual ~ParametricObject() { }

	virtual void initialize_params(std::mt19937& rnd, float* params_full_precision, PARAMS_T* params, PARAMS_T* inference_params, PARAMS_T* backward_params, PARAMS_T* gradients, float scale = 1) = 0;
	virtual size_t n_params() const = 0;

	virtual std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const = 0;
};

template <typename T, typename PARAMS_T, typename COMPUTE_T=T>
class DifferentiableObject : public ParametricObject<PARAMS_T> {
public:
	virtual ~DifferentiableObject() { }

	virtual void inference(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& input, GPUMatrix<float, MatrixLayout::ColumnMajor>& output) = 0;
	virtual void forward(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& input, GPUMatrix<COMPUTE_T, MatrixLayout::ColumnMajor>& output, MatrixLayout output_layout = MatrixLayout::ColumnMajor, bool use_inference_matrices = false) = 0;
	virtual void backward(
		cudaStream_t stream,
		const GPUMatrix<T, MatrixLayout::ColumnMajor>& input,
		const GPUMatrix<COMPUTE_T, MatrixLayout::ColumnMajor>& output,
		const GPUMatrix<COMPUTE_T, MatrixLayout::ColumnMajor>& dL_doutput,
		GPUMatrix<COMPUTE_T, MatrixLayout::ColumnMajor>* dL_dinput = nullptr,
		MatrixLayout output_layout = MatrixLayout::ColumnMajor,
		bool use_inference_matrices = false,
		bool compute_param_gradients = true
	) = 0;

	virtual uint32_t padded_output_width() const = 0;
	virtual uint32_t output_width() const = 0;

	virtual uint32_t required_input_alignment() const = 0;
};

TCNN_NAMESPACE_END
