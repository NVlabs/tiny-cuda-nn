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

/** @file   loss.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface for loss functions that models can be trained to minimize
 */

#include <tiny-cuda-nn/loss.h>

#include <tiny-cuda-nn/losses/mape.h>
#include <tiny-cuda-nn/losses/smape.h>
#include <tiny-cuda-nn/losses/l1.h>
#include <tiny-cuda-nn/losses/l2.h>
#include <tiny-cuda-nn/losses/relative_l1.h>
#include <tiny-cuda-nn/losses/relative_l2.h>
#include <tiny-cuda-nn/losses/relative_l2_luminance.h>
#include <tiny-cuda-nn/losses/cross_entropy.h>
#include <tiny-cuda-nn/losses/variance_is.h>

namespace tcnn {

template <typename T>
void register_loss(ci_hashmap<std::function<Loss<T>*(const json&)>>& factories, const std::string& name, const std::function<Loss<T>*(const json&)>& factory) {
	if (factories.count(name) > 0) {
		throw std::runtime_error{fmt::format("Can not register loss '{}' twice.", name)};
	}

	factories.insert({name, factory});
}

template <typename T>
auto register_builtin_losses() {
	ci_hashmap<std::function<Loss<T>*(const json&)>> factories;

	register_loss<T>(factories, "L2", [](const json& loss) { return new L2Loss<T>{}; });
	register_loss<T>(factories, "RelativeL2", [](const json& loss) { return new RelativeL2Loss<T>{}; });
	register_loss<T>(factories, "RelativeL2Luminance", [](const json& loss) { return new RelativeL2LuminanceLoss<T>{}; });
	register_loss<T>(factories, "L1", [](const json& loss) { return new L1Loss<T>{}; });
	register_loss<T>(factories, "RelativeL1", [](const json& loss) { return new RelativeL1Loss<T>{}; });
	register_loss<T>(factories, "Mape", [](const json& loss) { return new MapeLoss<T>{}; });
	register_loss<T>(factories, "Smape", [](const json& loss) { return new SmapeLoss<T>{}; });
	register_loss<T>(factories, "CrossEntropy", [](const json& loss) { return new CrossEntropyLoss<T>{}; });
	register_loss<T>(factories, "Variance", [](const json& loss) { return new VarianceIsLoss<T>{}; });

	return factories;
}

template <typename T>
auto& loss_factories() {
	static ci_hashmap<std::function<Loss<T>*(const json&)>> factories = register_builtin_losses<T>();
	return factories;
}

template <typename T>
void register_loss(const std::string& name, const std::function<Loss<T>*(const json&)>& factory) {
	register_loss(loss_factories<T>(), name, factory);
}

template void register_loss<float>(const std::string& name, const std::function<Loss<float>*(const json&)>& factory);
template void register_loss<__half>(const std::string& name, const std::function<Loss<__half>*(const json&)>& factory);

template <typename T>
Loss<T>* create_loss(const json& loss) {
	std::string name = loss.value("otype", "RelativeL2");

	if (loss_factories<T>().count(name) == 0) {
		throw std::runtime_error{fmt::format("Loss '{}' not found", name)};
	}

	return loss_factories<T>().at(name)(loss);
}

template Loss<float>* create_loss(const json& loss);
template Loss<__half>* create_loss(const json& loss);

std::vector<std::string> builtin_losses() {
	std::vector<std::string> result;
	for (const auto& kv : loss_factories<float>()) {
		result.emplace_back(kv.first);
	}

	return result;
}

}
