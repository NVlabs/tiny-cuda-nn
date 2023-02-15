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

/** @file   optimizer.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  API for optimizers
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>

#include <stdint.h>

TCNN_NAMESPACE_BEGIN

template <typename T>
class Optimizer : public ObjectWithMutableHyperparams {
public:
	virtual ~Optimizer() {}

	virtual void allocate(uint32_t n_weights, const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes = {}) = 0;
	void allocate(const std::shared_ptr<ParametricObject<T>>& target) {
		allocate(target->n_params(), target->layer_sizes());
	};

	virtual void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) = 0;
	virtual float learning_rate() const = 0;
	virtual void set_learning_rate(float val) = 0;
	virtual uint32_t step() const = 0;
	virtual uint32_t n_weights() const = 0;
	virtual T* custom_weights() const = 0;

	virtual size_t n_nested() const { return 0; }
	virtual const std::shared_ptr<Optimizer<T>>& nested(size_t idx = 0) const {
		throw std::runtime_error{"Optimizer does not support nesting."};
	}

	virtual json serialize() const { return {}; }
	virtual void deserialize(const json& data) { }
};

template <typename T>
Optimizer<T>* create_optimizer(const json& params);

TCNN_NAMESPACE_END
