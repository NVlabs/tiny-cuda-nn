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

/** @file   config.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  API to create everything needed for using the framework from
 *          a single json config.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

TCNN_NAMESPACE_BEGIN

struct TrainableModel {
	std::shared_ptr<Loss<network_precision_t>> loss;
	std::shared_ptr<Optimizer<network_precision_t>> optimizer;
	std::shared_ptr<NetworkWithInputEncoding<network_precision_t>> network;
	std::shared_ptr<Trainer<float, network_precision_t, network_precision_t>> trainer;
};

inline TrainableModel create_from_config(
	uint32_t n_input_dims,
	uint32_t n_output_dims,
	json config
) {
	std::shared_ptr<Loss<network_precision_t>> loss{create_loss<network_precision_t>(config.value("loss", json::object()))};
	std::shared_ptr<Optimizer<network_precision_t>> optimizer{create_optimizer<network_precision_t>(config.value("optimizer", json::object()))};
	auto network = std::make_shared<NetworkWithInputEncoding<network_precision_t>>(n_input_dims, n_output_dims, config.value("encoding", json::object()), config.value("network", json::object()));
	auto trainer = std::make_shared<Trainer<float, network_precision_t, network_precision_t>>(network, optimizer, loss);
	return {loss, optimizer, network, trainer};
}

TCNN_NAMESPACE_END
