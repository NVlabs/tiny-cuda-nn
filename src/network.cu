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

/** @file   network.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface of a neural network implementation
 */

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/network.h>

#include <tiny-cuda-nn/networks/cutlass_mlp.h>

#if TCNN_MIN_GPU_ARCH > 70
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>
#include <tiny-cuda-nn/mma.h>
#endif


namespace tcnn {

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
			log_warning(
				"FullyFusedMLP is not supported for the selected architecture {}. Falling back to CutlassMLP. "
				"For maximum performance, raise the target GPU architecture to 75+.",
				MIN_GPU_ARCH
			);
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

std::vector<std::string> builtin_networks() {
	return {
		"FullyFusedMLP",
		"CutlassMLP",
	};
}

template <>
std::unique_ptr<CudaRtcKernel> generate_mlp_convert_params_to_jit_layout_kernel<float>(uint32_t input_width, uint32_t width, uint32_t padded_output_width, uint32_t n_hidden) {
	throw std::runtime_error{"FP32 MLP device code generation not supported."};
}

template <>
std::unique_ptr<CudaRtcKernel> generate_mlp_convert_params_to_jit_layout_kernel<__half>(uint32_t input_width, uint32_t width, uint32_t padded_output_width, uint32_t n_hidden) {
	return std::make_unique<CudaRtcKernel>("mlp_convert_params_to_jit_layout", dfmt(0, R"(
			__global__ void mlp_convert_params_to_jit_layout({T}* __restrict__ params) {{
				if ({N_HIDDEN} == 0) {{
					auto mat = mma_mat<{N_DIMS_IN}, {N_DIMS_OUT}>::from_linear_memory(params);
					mat.into_native_memory(params);
					return;
				}}

				if (blockIdx.x == 0) {{
					auto first_mat = mma_mat<{N_DIMS_IN}, {N_DIMS_HIDDEN}>::from_linear_memory(params);
					first_mat.into_native_memory(params);
				}} else if (blockIdx.x == 1) {{
					params += {N_DIMS_IN} * {N_DIMS_HIDDEN} + ({N_HIDDEN} - 1) * {N_DIMS_HIDDEN} * {N_DIMS_HIDDEN};
					auto last_mat = mma_mat<{N_DIMS_HIDDEN}, {N_DIMS_OUT}>::from_linear_memory(params);
					last_mat.into_native_memory(params);
				}} else {{
					params += {N_DIMS_IN} * {N_DIMS_HIDDEN} + (blockIdx.x - 2) * {N_DIMS_HIDDEN} * {N_DIMS_HIDDEN};
					auto hidden_mat = mma_mat<{N_DIMS_HIDDEN}, {N_DIMS_HIDDEN}>::from_linear_memory(params);
					hidden_mat.into_native_memory(params);
				}}
			}}
		)",
		"T"_a = type_to_string<__half>(),
		"N_DIMS_IN"_a = input_width,
		"N_DIMS_HIDDEN"_a = width,
		"N_DIMS_OUT"_a = padded_output_width,
		"N_HIDDEN"_a = n_hidden
	));
}

template <>
std::unique_ptr<CudaRtcKernel> generate_mlp_convert_params_from_jit_layout_kernel<float>(uint32_t input_width, uint32_t width, uint32_t padded_output_width, uint32_t n_hidden) {
	throw std::runtime_error{"FP32 MLP device code generation not supported."};
}

template <>
std::unique_ptr<CudaRtcKernel> generate_mlp_convert_params_from_jit_layout_kernel<__half>(uint32_t input_width, uint32_t width, uint32_t padded_output_width, uint32_t n_hidden) {
	return std::make_unique<CudaRtcKernel>("mlp_convert_params_from_jit_layout", dfmt(0, R"(
			__global__ void mlp_convert_params_from_jit_layout({T}* __restrict__ params) {{
				if ({N_HIDDEN} == 0) {{
					auto mat = mma_mat<{N_DIMS_IN}, {N_DIMS_OUT}>::from_native_memory(params);
					mat.into_linear_memory(params);
					return;
				}}

				if (blockIdx.x == 0) {{
					auto first_mat = mma_mat<{N_DIMS_IN}, {N_DIMS_HIDDEN}>::from_native_memory(params);
					first_mat.into_linear_memory(params);
				}} else if (blockIdx.x == 1) {{
					params += {N_DIMS_IN} * {N_DIMS_HIDDEN} + ({N_HIDDEN} - 1) * {N_DIMS_HIDDEN} * {N_DIMS_HIDDEN};
					auto last_mat = mma_mat<{N_DIMS_HIDDEN}, {N_DIMS_OUT}>::from_native_memory(params);
					last_mat.into_linear_memory(params);
				}} else {{
					params += {N_DIMS_IN} * {N_DIMS_HIDDEN} + (blockIdx.x - 2) * {N_DIMS_HIDDEN} * {N_DIMS_HIDDEN};
					auto hidden_mat = mma_mat<{N_DIMS_HIDDEN}, {N_DIMS_HIDDEN}>::from_native_memory(params);
					hidden_mat.into_linear_memory(params);
				}}
			}}
		)",
		"T"_a = type_to_string<__half>(),
		"N_DIMS_IN"_a = input_width,
		"N_DIMS_HIDDEN"_a = width,
		"N_DIMS_OUT"_a = padded_output_width,
		"N_HIDDEN"_a = n_hidden
	));
}

template <>
std::string generate_mlp_device_code<float>(uint32_t input_width, uint32_t width, uint32_t padded_output_width, uint32_t output_width, uint32_t n_hidden, Activation activation, Activation output_activation) {
	throw std::runtime_error{"FP32 MLP device code generation not supported."};
}

template <>
std::string generate_mlp_device_code<__half>(uint32_t input_width, uint32_t width, uint32_t padded_output_width, uint32_t output_width, uint32_t n_hidden, Activation activation, Activation output_activation) {
	std::string output_activation_body = output_activation == Activation::None ? "" : dfmt(1, R"(
			if (fwd_ctx) {{
				out.into_native_memory(({T}*)fwd_ctx);
			}}

			out.activate<Activation::{ACT_OUT}>();
		)",
		"T"_a = type_to_string<__half>(),
		"ACT_OUT"_a = to_string(output_activation)
	);

	if (n_hidden == 0) {
		return dfmt(1, R"(
				mma_vec<{N_DIMS_IN}> in{{input}};

				if (fwd_ctx) {{
					in.into_native_memory(({T}*)fwd_ctx);
					fwd_ctx += in.M * in.N * sizeof({T});
				}}

				auto mat = mma_mat<{N_DIMS_IN}, {N_PADDED_DIMS_OUT}>::from_native_memory(params);
				auto out = in * mat;
				{OUTPUT_ACTIVATION_BODY}

				return out.vec<{N_DIMS_OUT}>();
			)",
			"T"_a = type_to_string<__half>(),
			"N_DIMS_IN"_a = input_width,
			"N_PADDED_DIMS_OUT"_a = padded_output_width,
			"N_DIMS_OUT"_a = output_width,
			"OUTPUT_ACTIVATION_BODY"_a = output_activation_body
		);
	}

	return dfmt(1, R"(
			mma_vec<{N_DIMS_IN}> in{{input}};
			if (fwd_ctx) {{
				in.into_native_memory(({T}*)fwd_ctx);
				fwd_ctx += 32 * sizeof({T}) * {N_DIMS_IN};
			}}

			auto first_mat = mma_mat<{N_DIMS_IN}, {N_DIMS_HIDDEN}>::from_native_memory(params);
			params += {N_DIMS_IN} * {N_DIMS_HIDDEN};

			auto hidden = in * first_mat;
			hidden.activate<Activation::{ACT}>();

			if (fwd_ctx) {{
				hidden.into_native_memory(({T}*)fwd_ctx);
				fwd_ctx += 32 * sizeof({T}) * {N_DIMS_HIDDEN};
			}}

			TCNN_PRAGMA_UNROLL
			for (uint32_t i = 0; i < {N_HIDDEN_MATMULS}; ++i) {{
				auto hidden_mat = mma_mat<{N_DIMS_HIDDEN}, {N_DIMS_HIDDEN}>::from_native_memory(params);
				params += {N_DIMS_HIDDEN} * {N_DIMS_HIDDEN};
				hidden = hidden * hidden_mat;
				hidden.activate<Activation::{ACT}>();

				if (fwd_ctx) {{
					hidden.into_native_memory(({T}*)fwd_ctx);
					fwd_ctx += 32 * sizeof({T}) * {N_DIMS_HIDDEN};
				}}
			}}

			auto last_mat = mma_mat<{N_DIMS_HIDDEN}, {N_PADDED_DIMS_OUT}>::from_native_memory(params);
			auto out = hidden * last_mat;
			{OUTPUT_ACTIVATION_BODY}

			return out.vec<{N_DIMS_OUT}>();
		)",
		"T"_a = type_to_string<__half>(),
		"N_DIMS_IN"_a = input_width,
		"N_DIMS_HIDDEN"_a = width,
		"N_PADDED_DIMS_OUT"_a = padded_output_width,
		"N_DIMS_OUT"_a = output_width,
		"N_HIDDEN_MATMULS"_a = n_hidden-1,
		"ACT"_a = to_string(activation),
		"OUTPUT_ACTIVATION_BODY"_a = output_activation_body
	);
}

template <>
std::string generate_backward_mlp_device_code<float>(uint32_t n_threads, uint32_t input_width, uint32_t width, uint32_t padded_output_width, uint32_t output_width, uint32_t n_hidden, Activation activation, Activation output_activation) {
	throw std::runtime_error{"FP32 MLP device code generation not supported."};
}

template <>
std::string generate_backward_mlp_device_code<__half>(uint32_t n_threads, uint32_t input_width, uint32_t width, uint32_t padded_output_width, uint32_t output_width, uint32_t n_hidden, Activation activation, Activation output_activation) {
	std::string output_activation_body = output_activation == Activation::None ? "" : dfmt(1, R"(
			auto out = mma_vec<{N_PADDED_DIMS_OUT}>::from_native_memory(({T}*)fwd_ctx + 32 * ({N_DIMS_IN} + {N_HIDDEN} * {N_DIMS_HIDDEN}));
			out_grad.activate_bwd<Activation::{ACT_OUT}>(out);
		)",
		"T"_a = type_to_string<__half>(),
		"N_DIMS_IN"_a = input_width,
		"N_HIDDEN"_a = n_hidden,
		"N_DIMS_HIDDEN"_a = width,
		"N_PADDED_DIMS_OUT"_a = padded_output_width,
		"ACT_OUT"_a = to_string(output_activation)
	);

	if (n_hidden == 0) {
		return dfmt(1, R"(
				mma_vec<{N_PADDED_DIMS_OUT}> out_grad{{dL_dy}};
				{OUTPUT_ACTIVATION_BODY}

				auto in = mma_vec<{N_DIMS_IN}>::from_native_memory(({T}*)fwd_ctx);
				if (dL_dparams) {{
					outer_product(out_grad, in).sum_into_linear_global_memory_hierarchical<{N_THREADS}>(dL_dparams);
				}}

				if (!dL_dx) {{
					return;
				}}

				auto mat = mma_mat<{N_DIMS_IN}, {N_PADDED_DIMS_OUT}>::from_native_memory(params);
				auto in_grad = out_grad * mat.transpose();
				*dL_dx = in_grad.vec<{N_DIMS_IN}>();
			)",
			"T"_a = type_to_string<__half>(),
			"N_DIMS_IN"_a = input_width,
			"N_PADDED_DIMS_OUT"_a = padded_output_width,
			"N_THREADS"_a = n_threads,
			"OUTPUT_ACTIVATION_BODY"_a = output_activation_body
		);
	}

	return dfmt(1, R"(
			mma_vec<{N_PADDED_DIMS_OUT}> out_grad{{dL_dy}};
			{OUTPUT_ACTIVATION_BODY}

			auto hidden = mma_vec<{N_DIMS_HIDDEN}>::from_native_memory(({T}*)fwd_ctx + 32 * ({N_DIMS_IN} + {N_HIDDEN_MATMULS} * {N_DIMS_HIDDEN}));
			if (dL_dparams) {{
				outer_product(out_grad, hidden).sum_into_linear_global_memory_hierarchical<{N_THREADS}>(dL_dparams + {N_DIMS_HIDDEN} * ({N_DIMS_IN} + {N_HIDDEN_MATMULS} * {N_DIMS_HIDDEN}));
			}}

			auto out_mat = mma_mat<{N_DIMS_HIDDEN}, {N_PADDED_DIMS_OUT}>::from_native_memory(params + {N_DIMS_HIDDEN} * ({N_DIMS_IN} + {N_HIDDEN_MATMULS} * {N_DIMS_HIDDEN}));
			auto hidden_grad = out_grad * out_mat.transpose();
			hidden_grad.activate_bwd<Activation::{ACT}>(hidden);

			TCNN_PRAGMA_UNROLL
			for (int i = {N_HIDDEN_MATMULS}-1; i >= 0; --i) {{
				hidden = mma_vec<{N_DIMS_HIDDEN}>::from_native_memory(({T}*)fwd_ctx + 32 * ({N_DIMS_IN} + i * {N_DIMS_HIDDEN}));
				if (dL_dparams) {{
					outer_product(hidden_grad, hidden).sum_into_linear_global_memory_hierarchical<{N_THREADS}>(dL_dparams + {N_DIMS_HIDDEN} * ({N_DIMS_IN} + i * {N_DIMS_HIDDEN}));
				}}

				auto hidden_mat = mma_mat<{N_DIMS_HIDDEN}, {N_DIMS_HIDDEN}>::from_native_memory(params + {N_DIMS_HIDDEN} * ({N_DIMS_IN} + i * {N_DIMS_HIDDEN}));
				hidden_grad = hidden_grad * hidden_mat.transpose();
				hidden_grad.activate_bwd<Activation::{ACT}>(hidden);
			}}

			auto in = mma_vec<{N_DIMS_IN}>::from_native_memory(({T}*)fwd_ctx);
			if (dL_dparams) {{
				outer_product(hidden_grad, in).sum_into_linear_global_memory_hierarchical<{N_THREADS}>(dL_dparams);
			}}

			if (!dL_dx) {{
				return;
			}}

			auto in_mat = mma_mat<{N_DIMS_IN}, {N_DIMS_HIDDEN}>::from_native_memory(params);
			auto in_grad = hidden_grad * in_mat.transpose();
			*dL_dx = in_grad.vec<{N_DIMS_IN}>();
		)",
		"T"_a = type_to_string<__half>(),
		"N_DIMS_IN"_a = input_width,
		"N_DIMS_OUT"_a = output_width,
		"N_PADDED_DIMS_OUT"_a = padded_output_width,
		"N_HIDDEN_MATMULS"_a = n_hidden-1,
		"N_DIMS_HIDDEN"_a = width,
		"N_THREADS"_a = n_threads,
		"ACT"_a = to_string(activation),
		"OUTPUT_ACTIVATION_BODY"_a = output_activation_body
	);
}

}
