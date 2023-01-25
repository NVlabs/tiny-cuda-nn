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

/** @file   optimizer.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface of optimizers that can be used to train models.
 */

#include <tiny-cuda-nn/optimizer.h>

#include <tiny-cuda-nn/optimizers/adam.h>
#include <tiny-cuda-nn/optimizers/average.h>
#include <tiny-cuda-nn/optimizers/batched.h>
#include <tiny-cuda-nn/optimizers/composite.h>
#include <tiny-cuda-nn/optimizers/ema.h>
#include <tiny-cuda-nn/optimizers/exponential_decay.h>
#include <tiny-cuda-nn/optimizers/lookahead.h>
#include <tiny-cuda-nn/optimizers/novograd.h>
#include <tiny-cuda-nn/optimizers/sgd.h>

#ifdef TCNN_SHAMPOO
#include <tiny-cuda-nn/optimizers/shampoo.h>
#endif


TCNN_NAMESPACE_BEGIN

template <typename T>
Optimizer<T>* create_optimizer(const json& optimizer) {
	std::string optimizer_type = optimizer.value("otype", "Adam");

	if (equals_case_insensitive(optimizer_type, "Adam")) {
		return new AdamOptimizer<T>{optimizer};
	} else if (equals_case_insensitive(optimizer_type, "Average")) {
		return new AverageOptimizer<T>{optimizer};
	} else if (equals_case_insensitive(optimizer_type, "Batched")) {
		return new BatchedOptimizer<T>{optimizer};
	} else if (equals_case_insensitive(optimizer_type, "Composite")) {
		return new CompositeOptimizer<T>{optimizer};
	} else if (equals_case_insensitive(optimizer_type, "Ema")) {
		return new EmaOptimizer<T>{optimizer};
	} else if (equals_case_insensitive(optimizer_type, "ExponentialDecay")) {
		return new ExponentialDecayOptimizer<T>{optimizer};
	} else if (equals_case_insensitive(optimizer_type, "Lookahead")) {
		return new LookaheadOptimizer<T>{optimizer};
	} else if (equals_case_insensitive(optimizer_type, "Novograd")) {
		return new NovogradOptimizer<T>{optimizer};
	} else if (equals_case_insensitive(optimizer_type, "SGD")) {
		return new SGDOptimizer<T>{optimizer};
	} else if (equals_case_insensitive(optimizer_type, "Shampoo")) {
#ifdef TCNN_SHAMPOO
		return new ShampooOptimizer<T>{optimizer};
#else
		throw std::runtime_error{"Cannot create `ShampooOptimizer` because tiny-cuda-nn was not compiled with cuBLAS and cuSolver."};
#endif
	} else {
		throw std::runtime_error{fmt::format("Invalid optimizer type: {}", optimizer_type)};
	}
}

template Optimizer<float>* create_optimizer(const json& optimizer);
template Optimizer<__half>* create_optimizer(const json& optimizer);

TCNN_NAMESPACE_END
