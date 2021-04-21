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

/** @file   network.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface of a neural network implementation
 */

#include <tiny-cuda-nn/network.h>

#include <tiny-cuda-nn/networks/cutlass_mlp.h>
#include <tiny-cuda-nn/networks/cutlass_resnet.h>
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>

#include <cutlass/half.h>


TCNN_NAMESPACE_BEGIN

Activation string_to_activation(std::string activation_name) {
	if (equals_case_insensitive(activation_name, "None")) {
		return Activation::None;
	} else if (equals_case_insensitive(activation_name, "ReLU")) {
		return Activation::ReLU;
	} else if (equals_case_insensitive(activation_name, "Exponential")) {
		return Activation::Exponential;
	} else if (equals_case_insensitive(activation_name, "Sigmoid")) {
		return Activation::Sigmoid;
	} else if (equals_case_insensitive(activation_name, "Sine")) {
		return Activation::Sine;
	}

	throw std::runtime_error{std::string{"Invalid activation name: "} + activation_name};
}

template <typename T>
Network<T>* create_network(json network) {
	std::string network_type = network.value("otype", "MLP");

	if (equals_case_insensitive(network_type, "MegakernelMLP") || equals_case_insensitive(network_type, "FullyFusedMLP")) {
		if (!std::is_same_v<network_precision_t, cutlass::half_t>) {
			throw std::runtime_error{"FullyFusedMLP can only be used if the network precision is set to cutlass::half_t."};
		} else {
			uint32_t n_neurons = network.value("n_neurons", 128u);
			if (n_neurons == 128) {
				return new FullyFusedMLP<T, 128>{
					network["n_input_dims"],
					network["n_output_dims"],
					network.value("n_hidden_layers", 5u),
					network.value("feedback_alignment", false),
					string_to_activation(network.value("activation", "ReLU")),
					string_to_activation(network.value("output_activation", "None")),
				};
			} else if (n_neurons == 64) {
				return new FullyFusedMLP<T, 64>{
					network["n_input_dims"],
					network["n_output_dims"],
					network.value("n_hidden_layers", 5u),
					network.value("feedback_alignment", false),
					string_to_activation(network.value("activation", "ReLU")),
					string_to_activation(network.value("output_activation", "None")),
				};
			} else if (n_neurons == 32) {
				return new FullyFusedMLP<T, 32>{
					network["n_input_dims"],
					network["n_output_dims"],
					network.value("n_hidden_layers", 5u),
					network.value("feedback_alignment", false),
					string_to_activation(network.value("activation", "ReLU")),
					string_to_activation(network.value("output_activation", "None")),
				};
			} else {
				throw std::runtime_error{std::string{"FullyFusedMLP only supports 32, 64, and 128 neurons, but got: "} + std::to_string(n_neurons)};
			}
		}
	} else if (equals_case_insensitive(network_type, "MLP") || equals_case_insensitive(network_type, "CutlassMLP")) {
		return new CutlassMLP<T>{
			network["n_input_dims"],
			network.value("n_neurons", 128u),
			network["n_output_dims"],
			network.value("n_hidden_layers", 5u),
			string_to_activation(network.value("activation", "ReLU")),
			string_to_activation(network.value("output_activation", "None")),
		};
	} else if (equals_case_insensitive(network_type, "ResNet") || equals_case_insensitive(network_type, "CutlassResNet")) {
		return new CutlassResNet<T, Activation::None, Activation::None>{
			network["n_input_dims"],
			network.value("n_neurons", 128u),
			network["n_output_dims"],
			network.value("n_blocks", 2u),
			network.value("n_matrices_per_block", 2u),
		};
	}

	throw std::runtime_error{std::string{"Invalid network type: "} + network_type};
}

template Network<network_precision_t>* create_network(json network);

TCNN_NAMESPACE_END
