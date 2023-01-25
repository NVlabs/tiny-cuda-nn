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

TCNN_NAMESPACE_BEGIN

template <typename T>
Loss<T>* create_loss(const json& loss) {
	std::string loss_type = loss.value("otype", "RelativeL2");

	if (equals_case_insensitive(loss_type, "L2")) {
		return new L2Loss<T>{};
	} else if (equals_case_insensitive(loss_type, "RelativeL2")) {
		return new RelativeL2Loss<T>{};
	} else if (equals_case_insensitive(loss_type, "RelativeL2Luminance")) {
		return new RelativeL2LuminanceLoss<T>{};
	} else if (equals_case_insensitive(loss_type, "L1")) {
		return new L1Loss<T>{};
	} else if (equals_case_insensitive(loss_type, "RelativeL1")) {
		return new RelativeL1Loss<T>{};
	} else if (equals_case_insensitive(loss_type, "Mape")) {
		return new MapeLoss<T>{};
	} else if (equals_case_insensitive(loss_type, "Smape")) {
		return new SmapeLoss<T>{};
	} else if (equals_case_insensitive(loss_type, "CrossEntropy")) {
		return new CrossEntropyLoss<T>{};
	} else {
		throw std::runtime_error{fmt::format("Invalid loss type: {}", loss_type)};
	}
}

template Loss<float>* create_loss(const json& loss);
template Loss<__half>* create_loss(const json& loss);

TCNN_NAMESPACE_END
