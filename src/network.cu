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

/** @file   network.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface of a neural network implementation
 */

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/network.h>

#include <tiny-cuda-nn/networks/cutlass_mlp.h>

#if TCNN_MIN_GPU_ARCH > 70
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>
#endif

TCNN_NAMESPACE_BEGIN

Activation string_to_activation(const std::string& activation_name) {
	if (equals_case_insensitive(activation_name, "None")) {
		return Activation::None;
	} else if (equals_case_insensitive(activation_name, "ReLU")) {
		return Activation::ReLU;
	} else if (equals_case_insensitive(activation_name, "LeakyReLU")) {
		return Activation::LeakyReLU;
	} else if (equals_case_insensitive(activation_name, "Exponential")) {
		return Activation::Exponential;
	} else if (equals_case_insensitive(activation_name, "Sigmoid")) {
		return Activation::Sigmoid;
	} else if (equals_case_insensitive(activation_name, "Sine")) {
		return Activation::Sine;
	} else if (equals_case_insensitive(activation_name, "Squareplus")) {
		return Activation::Squareplus;
	} else if (equals_case_insensitive(activation_name, "Softplus")) {
		return Activation::Softplus;
	} else if (equals_case_insensitive(activation_name, "Tanh")) {
		return Activation::Tanh;
	}

	throw std::runtime_error{fmt::format("Invalid activation name: {}", activation_name)};
}

std::string to_string(Activation activation) {
	switch (activation) {
		case Activation::None: return "None";
		case Activation::ReLU: return "ReLU";
		case Activation::LeakyReLU: return "LeakyReLU";
		case Activation::Exponential: return "Exponential";
		case Activation::Sigmoid: return "Sigmoid";
		case Activation::Sine: return "Sine";
		case Activation::Squareplus: return "Squareplus";
		case Activation::Softplus: return "Softplus";
		case Activation::Tanh: return "Tanh";
		default: throw std::runtime_error{"Invalid activation."};
	}
}

template <typename T>
void extract_dimension_pos_neg(cudaStream_t stream, const uint32_t num_elements, const uint32_t dim, const uint32_t fan_in, const uint32_t fan_out, const T* encoded, MatrixLayout layout, float* output) {
	linear_kernel(extract_dimension_pos_neg_kernel<T>, 0, stream, num_elements, dim, fan_in, fan_out, encoded, layout, output);
}

template void extract_dimension_pos_neg(cudaStream_t stream, const uint32_t num_elements, const uint32_t dim, const uint32_t fan_in, const uint32_t fan_out, const network_precision_t* encoded, MatrixLayout layout, float* output);

std::string select_network(const json& network) {
	std::string otype = network.value("otype", "MLP");
	bool want_fully_fused_mlp = equals_case_insensitive(otype, "MegakernelMLP") || equals_case_insensitive(otype, "FullyFusedMLP");
	bool want_cutlass_mlp = equals_case_insensitive(otype, "MLP") || equals_case_insensitive(otype, "CutlassMLP");

	// If the GPU architecture is insufficient for
	if (MIN_GPU_ARCH <= 70 || std::is_same<network_precision_t, float>::value) {
		if (want_fully_fused_mlp && MIN_GPU_ARCH <= 70) {
			std::cout
				<< "Warning: FullyFusedMLP is not supported for the selected architecture " << MIN_GPU_ARCH << ". "
				<< "Falling back to CutlassMLP. For maximum performance, raise the target GPU architecture to 75+."
				<< std::endl;
		}

		want_cutlass_mlp |= want_fully_fused_mlp;
		want_fully_fused_mlp = false;
	}

	if (want_fully_fused_mlp) {
		return "FullyFusedMLP";
	} else if (want_cutlass_mlp) {
		return "CutlassMLP";
	} else {
		return otype;
	}
}

uint32_t minimum_alignment(const json& network) {
	std::string network_type = select_network(network);

	if (equals_case_insensitive(network_type, "FullyFusedMLP")) {
#if TCNN_MIN_GPU_ARCH > 70
		uint32_t n_neurons = network.value("n_neurons", 128u);
		switch (n_neurons) {
			case  16: return FullyFusedMLP<network_precision_t,  16>::REQUIRED_ALIGNMENT();
			case  32: return FullyFusedMLP<network_precision_t,  32>::REQUIRED_ALIGNMENT();
			case  64: return FullyFusedMLP<network_precision_t,  64>::REQUIRED_ALIGNMENT();
			case 128: return FullyFusedMLP<network_precision_t, 128>::REQUIRED_ALIGNMENT();
			default: throw std::runtime_error{fmt::format("FullyFusedMLP only supports 16, 32, 64, and 128 neurons, but got {}. Use CutlassMLP instead if this is a requirement.", n_neurons)};
		}
#else
		throw std::runtime_error{"FullyFusedMLP was not compiled due to insufficient GPU arch of <=70."};
#endif
	} else {
		return CutlassMLP<network_precision_t>::REQUIRED_ALIGNMENT();
	}
}

template <typename T>
Network<T>* create_network(const json& network) {
	std::string network_type = select_network(network);

	if (equals_case_insensitive(network_type, "FullyFusedMLP")) {
		if (!std::is_same<network_precision_t, __half>::value) {
			throw std::runtime_error{"FullyFusedMLP can only be used if the network precision is set to __half."};
		} else {
#if TCNN_MIN_GPU_ARCH > 70
#  define TCNN_FULLY_FUSED_PARAMS \
	network["n_input_dims"], \
	network["n_output_dims"], \
	network.value("n_hidden_layers", 5u), \
	string_to_activation(network.value("activation", "ReLU")), \
	string_to_activation(network.value("output_activation", "None")),

			uint32_t n_neurons = network.value("n_neurons", 128u);
			switch (n_neurons) {
				case  16: return new FullyFusedMLP<T,  16>{TCNN_FULLY_FUSED_PARAMS};
				case  32: return new FullyFusedMLP<T,  32>{TCNN_FULLY_FUSED_PARAMS};
				case  64: return new FullyFusedMLP<T,  64>{TCNN_FULLY_FUSED_PARAMS};
				case 128: return new FullyFusedMLP<T, 128>{TCNN_FULLY_FUSED_PARAMS};
				default: throw std::runtime_error{fmt::format("FullyFusedMLP only supports 16, 32, 64, and 128 neurons, but got {}. Use CutlassMLP instead if this is a requirement.", n_neurons)};
			}
#  undef TCNN_FULLY_FUSED_PARAMS
#else //TCNN_MIN_GPU_ARCH > 70
			throw std::runtime_error{"FullyFusedMLP was not compiled due to insufficient GPU arch of <=70."};
#endif //TCNN_MIN_GPU_ARCH > 70
		}
	} else if (equals_case_insensitive(network_type, "CutlassMLP")) {
		return new CutlassMLP<T>{
			network["n_input_dims"],
			network.value("n_neurons", 128u),
			network["n_output_dims"],
			network.value("n_hidden_layers", 5u),
			string_to_activation(network.value("activation", "ReLU")),
			string_to_activation(network.value("output_activation", "None")),
		};
	}

	throw std::runtime_error{fmt::format("Invalid network type: {}", network_type)};
}

template Network<network_precision_t>* create_network(const json& network);

TCNN_NAMESPACE_END
