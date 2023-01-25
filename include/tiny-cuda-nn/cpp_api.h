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

/** @file   cpp_api.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  API to be consumed by cpp (non-CUDA) programs.
 */

#pragma once

#include <json/json.hpp>

#include <cuda_runtime.h>

#include <memory>
#include <string>

namespace tcnn {
	struct Context {
		Context() = default;
		virtual ~Context() {}
		Context(const Context&) = delete;
		Context& operator=(const Context&) = delete;
		Context(Context&&) = delete;
		Context& operator=(Context&&) = delete;
	};
}

namespace tcnn { namespace cpp {

using json = nlohmann::json;

uint32_t batch_size_granularity();

int cuda_device();
void set_cuda_device(int device);

void free_temporary_memory();

bool has_networks();

enum EPrecision {
	Fp32,
	Fp16,
};

EPrecision preferred_precision();

struct Context {
	std::unique_ptr<tcnn::Context> ctx;
};

class Module {
public:
	Module(EPrecision param_precision, EPrecision output_precision) : m_param_precision{param_precision}, m_output_precision{output_precision} {}
	virtual ~Module() {}

	virtual void inference(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params) = 0;
	virtual Context forward(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params, bool prepare_input_gradients) = 0;
	virtual void backward(cudaStream_t stream, const Context& ctx, uint32_t n_elements, float* dL_dinput, const void* dL_doutput, void* dL_dparams, const float* input, const void* output, const void* params) = 0;
	virtual void backward_backward_input(cudaStream_t stream, const Context& ctx, uint32_t n_elements, const float* dL_ddLdinput, const float* input, const void* dL_doutput, void* dL_dparams, void* dL_ddLdoutput, float* dL_dinput, const void* params) = 0;

	virtual uint32_t n_input_dims() const = 0;

	virtual size_t n_params() const = 0;
	EPrecision param_precision() const {
		return m_param_precision;
	}

	virtual void initialize_params(size_t seed, float* params_full_precision, float scale = 1.0f) = 0;

	virtual uint32_t n_output_dims() const = 0;
	EPrecision output_precision() const {
		return m_output_precision;
	}

	virtual json hyperparams() const = 0;
	virtual std::string name() const = 0;

private:
	EPrecision m_param_precision;
	EPrecision m_output_precision;
};

Module* create_network_with_input_encoding(uint32_t n_input_dims, uint32_t n_output_dims, const json& encoding, const json& network);
Module* create_network(uint32_t n_input_dims, uint32_t n_output_dims, const json& network);
Module* create_encoding(uint32_t n_input_dims, const json& encoding, EPrecision requested_precision);

}}
