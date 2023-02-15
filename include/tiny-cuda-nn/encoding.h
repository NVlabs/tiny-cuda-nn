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

/** @file   encoding.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface for input encodings
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>

#include <stdint.h>

TCNN_NAMESPACE_BEGIN

enum class InterpolationType {
	Nearest,
	Linear,
	Smoothstep,
};

InterpolationType string_to_interpolation_type(const std::string& interpolation_type);

std::string to_string(InterpolationType interpolation_type);

enum class ReductionType {
	Concatenation,
	Sum,
	Product,
};

ReductionType string_to_reduction_type(const std::string& reduction_type);

std::string to_string(ReductionType reduction_type);

template <typename T>
class Encoding : public DifferentiableObject<float, T, T> {
public:
	virtual ~Encoding() { }

	void inference_mixed_precision_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>& output,
		bool use_inference_params = true
	) override {
		this->forward(stream, input, &output, use_inference_params, false);
	}

	virtual void set_padded_output_width(uint32_t padded_output_width) = 0;
	virtual uint32_t required_output_alignment() const = 0;

	virtual MatrixLayout preferred_output_layout() const = 0;

	virtual size_t n_nested() const { return 0; }
	virtual const std::shared_ptr<Encoding<T>>& nested(size_t idx = 0) const {
		throw std::runtime_error{"Encoding does not support nesting."};
	}

	// By default, an encoding has no parameters
	void set_params_impl(T* params, T* inference_params, T* gradients) override { }
	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override { }
	size_t n_params() const override { return 0; }

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override { return {}; }

	void set_alignment(uint32_t alignment) {
		this->set_padded_output_width(next_multiple(this->output_width(), lcm(alignment, this->required_output_alignment())));
	}
};

template <typename T>
Encoding<T>* create_encoding(uint32_t n_dims_to_encode, const json& params, uint32_t alignment = 8);

template <typename T>
void register_encoding(const std::string& name, const std::function<Encoding<T>*(uint32_t, const json&)>& factory);

TCNN_NAMESPACE_END
