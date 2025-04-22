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

/** @file   object.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface of a TCNN object
 */

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/object.h>

namespace tcnn {

template <typename T>
__global__ void one_hot_batched_kernel(const uint32_t num_elements, const uint32_t width, const uint32_t one_hot_dim, T* out, float scale) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint32_t dim = i % width;
	out[i] = dim == one_hot_dim ? (T)scale : (T)0.0f;
}

template <typename T>
void one_hot_batched(cudaStream_t stream, const uint32_t num_elements, const uint32_t width, const uint32_t one_hot_dim, T* out, float scale) {
	linear_kernel(one_hot_batched_kernel<T>, 0, stream, num_elements, width, one_hot_dim, out, scale);
}

template void one_hot_batched(cudaStream_t stream, const uint32_t num_elements, const uint32_t width, const uint32_t one_hot_dim, float* out, float scale);
template void one_hot_batched(cudaStream_t stream, const uint32_t num_elements, const uint32_t width, const uint32_t one_hot_dim, __half* out, float scale);

template <typename T>
void mult(cudaStream_t stream, const uint32_t num_elements, T* inout, float factor) {
	linear_kernel(mult_scalar_kernel<T>, 0, stream, num_elements, inout, factor);
}

template void mult(cudaStream_t stream, const uint32_t num_elements, float* inout, float factor);
template void mult(cudaStream_t stream, const uint32_t num_elements, __half* inout, float factor);

template <typename T>
void trim_and_cast_from(cudaStream_t stream, const MatrixLayout layout, const uint32_t num_elements, const uint32_t input_width, const uint32_t output_width, const T* in, float* out) {
	if (layout == RM) {
		linear_kernel(cast_from<T>, 0, stream, num_elements, in, out);
	} else {
		linear_kernel(trim_and_cast<T>, 0, stream, num_elements, input_width, output_width, in, out);
	}
}

template void trim_and_cast_from(cudaStream_t stream, const MatrixLayout layout, const uint32_t num_elements, const uint32_t input_width, const uint32_t output_width, const float* in, float* out);
template void trim_and_cast_from(cudaStream_t stream, const MatrixLayout layout, const uint32_t num_elements, const uint32_t input_width, const uint32_t output_width, const __half* in, float* out);

std::unique_ptr<CudaRtcKernel> generate_kernel(
	const std::string& kernel_name,
	const std::string& device_function,
	const std::string& T,
	const std::string& PARAMS_T,
	const std::string& COMPUTE_T,
	uint32_t n_dims_in,
	uint32_t n_fwd_ctx_bytes
) {
	return std::make_unique<CudaRtcKernel>(kernel_name, dfmt(0, R"(
			{DEVICE_FUNCTION}

			__global__ void {KERNEL_NAME}(const uint32_t num_elements, MatrixView<const {T}> data_in, MatrixView<{COMPUTE_T}> data_out, const {PARAMS_T}* __restrict__ params{FWD_CTX_PARAM}) {{
				{FWD_CTX_ADVANCE}const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

				auto input = data_in.col<{N_DIMS_IN}>(i);
				auto output = eval_model(input, params, {FWD_CTX_ARG});
				if (data_out) {{
					data_out.set_col(i, output);
				}}
			}}
		)",
		"DEVICE_FUNCTION"_a = device_function,
		"KERNEL_NAME"_a = kernel_name,
		"T"_a = T,
		"PARAMS_T"_a = PARAMS_T,
		"COMPUTE_T"_a = COMPUTE_T,
		"N_DIMS_IN"_a = n_dims_in,
		"FWD_CTX_PARAM"_a = n_fwd_ctx_bytes ? ", uint8_t* __restrict__ fwd_ctx" : "",
		"FWD_CTX_ADVANCE"_a = n_fwd_ctx_bytes ? fmt::format("fwd_ctx += ((threadIdx.x / WARP_SIZE) * WARP_SIZE + blockIdx.x * blockDim.x) * {};\n", n_fwd_ctx_bytes) : "",
		"FWD_CTX_ARG"_a = n_fwd_ctx_bytes ? "fwd_ctx" : "nullptr"
	));
}

std::unique_ptr<CudaRtcKernel> generate_backward_kernel(
	const std::string& kernel_name,
	const std::string& device_function,
	const std::string& T,
	const std::string& PARAMS_T,
	const std::string& COMPUTE_T,
	uint32_t n_dims_in,
	uint32_t n_dims_out,
	uint32_t n_fwd_ctx_bytes
) {
	return std::make_unique<CudaRtcKernel>(kernel_name, dfmt(0, R"(
			{DEVICE_FUNCTION}

			__global__ void {KERNEL_NAME}(const uint32_t num_elements, MatrixView<const {COMPUTE_T}> data_dL_dy, MatrixView<{T}> data_dL_dx, const {PARAMS_T}* __restrict__ params, const uint8_t* __restrict__ fwd_ctx, {PARAMS_T}* __restrict__ dL_dparams) {{
				fwd_ctx += ((threadIdx.x / WARP_SIZE) * WARP_SIZE + blockIdx.x * blockDim.x) * {N_FWD_CTX_BYTES};
				const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

				auto dL_dy = data_dL_dy.col<{N_DIMS_OUT}>(i);

				auto dL_dx = tvec<{T}, {N_DIMS_IN}>::zero();
				backward_eval_model(dL_dy, params, fwd_ctx, dL_dparams, data_dL_dx ? &dL_dx : nullptr);
				if (data_dL_dx) {{
					data_dL_dx.set_col(i, dL_dx);
				}}
			}}
		)",
		"DEVICE_FUNCTION"_a = device_function,
		"KERNEL_NAME"_a = kernel_name,
		"T"_a = T,
		"PARAMS_T"_a = PARAMS_T,
		"COMPUTE_T"_a = COMPUTE_T,
		"N_DIMS_IN"_a = n_dims_in,
		"N_DIMS_OUT"_a = n_dims_out,
		"N_FWD_CTX_BYTES"_a = n_fwd_ctx_bytes
	));
}

std::unique_ptr<CudaRtcKernel> generate_backward_backward_input_kernel(
	const std::string& kernel_name,
	const std::string& device_function,
	const std::string& T,
	const std::string& PARAMS_T,
	const std::string& COMPUTE_T,
	uint32_t n_dims_in,
	uint32_t n_dims_out,
	uint32_t n_fwd_ctx_bytes
) {
	return std::make_unique<CudaRtcKernel>(kernel_name, dfmt(0, R"(
			{DEVICE_FUNCTION}

			__global__ void {KERNEL_NAME}(const uint32_t num_elements, MatrixView<const {T}> data_dL_ddLdx, MatrixView<const {COMPUTE_T}> data_dL_dy, MatrixView<{T}> data_dL_dx, MatrixView<{COMPUTE_T}> data_dL_ddLdy, const {PARAMS_T}* __restrict__ params, const uint8_t* __restrict__ fwd_ctx, {PARAMS_T}* __restrict__ dL_dparams) {{
				fwd_ctx += ((threadIdx.x / WARP_SIZE) * WARP_SIZE + blockIdx.x * blockDim.x) * {N_FWD_CTX_BYTES};
				const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

				auto dL_ddLdx = data_dL_ddLdx.col<{N_DIMS_IN}>(i);
				auto dL_dy = data_dL_dy.col<{N_DIMS_OUT}>(i);

				auto dL_dx = tvec<{T}, {N_DIMS_IN}>::zero();
				auto dL_ddLdy = tvec<{COMPUTE_T}, {N_DIMS_OUT}>::zero();

				backward_backward_input_eval_model(dL_ddLdx, dL_dy, params, fwd_ctx, dL_dparams, data_dL_dx ? &dL_dx : nullptr, data_dL_ddLdy ? &dL_ddLdy : nullptr); 
				if (data_dL_dx) {{
					data_dL_dx.set_col(i, dL_dx); 
				}}
				if (data_dL_ddLdy) {{
					data_dL_ddLdy.set_col(i, dL_ddLdy); 
				}}
			}}
		)", 
		"DEVICE_FUNCTION"_a = device_function,
		"KERNEL_NAME"_a = kernel_name,
		"T"_a = T,
		"PARAMS_T"_a = PARAMS_T,
		"COMPUTE_T"_a = COMPUTE_T,
		"N_DIMS_IN"_a = n_dims_in,
		"N_DIMS_OUT"_a = n_dims_out,
		"N_FWD_CTX_BYTES"_a = n_fwd_ctx_bytes
	)); 
}

}
