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

/** @file   loss.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  API for loss functions
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/object.h>

namespace tcnn {

template <typename T>
class Loss : public ObjectWithMutableHyperparams {
public:
	virtual void evaluate(
		cudaStream_t stream,
		const float loss_scale,
		const GPUMatrix<T>& prediction,
		const GPUMatrix<float>& target,
		GPUMatrix<float>& values,
		GPUMatrix<T>& gradients,
		const GPUMatrix<float>* data_pdf = nullptr
	) const = 0;

	void evaluate(
		const float loss_scale,
		const GPUMatrix<T>& prediction,
		const GPUMatrix<float>& target,
		GPUMatrix<float>& values,
		GPUMatrix<T>& gradients,
		const GPUMatrix<float>* data_pdf = nullptr
	) const {
		evaluate(nullptr, loss_scale, prediction, target, values, gradients, data_pdf);
	}

	virtual std::string generate_device_function(const std::string& name, uint32_t n_dims) const {
		throw std::runtime_error{fmt::format("{}: device code generation is not supported.", this->name())};
	}

	std::string generate_device_function_from_body(const std::string& name, uint32_t n_dims, const std::string& body) const {
		return dfmt(0, R"(
				__device__ auto {NAME}(
					const uint32_t n_elements,
					const float loss_scale,
					const tvec<{T}, {N_DIMS}>& prediction,
					const vec<{N_DIMS}>& target,
					const vec<{N_DIMS}>& pdf,
					vec<{N_DIMS}>* value = nullptr
				) -> tvec<{T}, {N_DIMS}> {{
				{BODY}
				}}
			)",
			"NAME"_a = name,
			"T"_a = type_to_string<T>(),
			"N_DIMS"_a = n_dims,
			"BODY"_a = body
		);
	}
};

template <typename T>
Loss<T>* create_loss(const json& params);

template <typename T>
std::unique_ptr<Loss<T>> default_loss(const std::string& name) {
	return std::unique_ptr<Loss<T>>{create_loss<T>({{"otype", name}})};
}

std::vector<std::string> builtin_losses();

}
