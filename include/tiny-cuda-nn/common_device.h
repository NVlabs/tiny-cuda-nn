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

/** @file   common_device.h
 *  @author Thomas Müller & Nikolaus Binder, NVIDIA
 *  @brief  Implementation of various miscellaneous CUDA kernels and
            device functions.
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#define _USE_MATH_DEFINES
#include <cmath>

#include <cuda_fp16.h>

#include <cassert>
#include <cstdint>
#include <cstdio>

#include <tiny-cuda-nn/gpu_matrix.h>

TCNN_NAMESPACE_BEGIN

static constexpr float PI = 3.14159265358979323846f;
static constexpr float SQRT2 = 1.41421356237309504880f;

__host__ __device__ inline float logistic(const float x) {
	return 1.0f / (1.0f + expf(-x));
}

__host__ __device__ inline float logit(const float x) {
	return -logf(1.0f / (fminf(fmaxf(x, 1e-9f), 1.0f - 1e-9f)) - 1.0f);
}

template <typename V>
struct VectorFragment {
	static const uint32_t num_elements = V::N;
	V x;
};

template <typename T>
__host__ __device__ T relu(T val) {
	return (T)max((float)val, 0.0f);
}

template <>
inline __host__ __device__ __half relu(__half val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	return __hmax(val, (__half)0.0f);
#else
	return (__half)relu<float>((float)val);
#endif
}

static constexpr float K_ACT = 10.0f;

template <typename T, typename fragment_t>
__host__ __device__ void warp_activation(Activation activation, const fragment_t& frag, fragment_t& result) {
	switch (activation) {
		case Activation::ReLU:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = relu((T)frag.x[t]);
			}
			return;
		case Activation::LeakyReLU:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)((T)frag.x[t] > (T)0.0f ? 1.0f : 0.01f);
			}
			return;
		case Activation::Exponential:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = (T)(expf((float)frag.x[t]));
			}
			return;
		case Activation::Sine:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = (T)(sinf((float)frag.x[t]));
			}
			return;
		case Activation::Sigmoid:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = (T)(logistic((float)frag.x[t]));
			}
			return;
		case Activation::Squareplus:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				float x = (float)frag.x[t] * K_ACT;
				result.x[t] = (T)(0.5f * (x + sqrtf(x * x + 4)) / K_ACT);
			}
			return;
		case Activation::Softplus:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = (T)(logf(expf((float)frag.x[t] * K_ACT) + 1.0f) / K_ACT);
			}
			return;
		case Activation::Tanh:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = (T)(tanhf((float)frag.x[t]));
			}
			return;
		case Activation::None: result = frag; return;
		default:
			// Unsupported activation
			// assert(false); // Commented out due to isolated strange side-effects on Windows
			return;
	}
}

template <typename T, typename fragment_t>
__host__ __device__ fragment_t warp_activation(Activation activation, const fragment_t& frag) {
	fragment_t result;
	warp_activation<T>(activation, frag, result);
	return result;
}

template <typename T, typename fragment_t, typename forward_fragment_t>
__host__ __device__ void warp_activation_backward_in(Activation activation, const fragment_t& frag, const forward_fragment_t& forward_frag_in, fragment_t& result) {
	switch (activation) {
		case Activation::ReLU:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)(forward_frag_in.x[t] > (T)0.0f);
			}
			return;
		case Activation::LeakyReLU:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)(forward_frag_in.x[t] > (T)0.0f ? 1.0f : 0.01f);
			}
			return;
		case Activation::Exponential:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)(expf(forward_frag_in.x[t]));
			}
			return;
		case Activation::Sine:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)(cosf(forward_frag_in.x[t]));
			}
			return;
		case Activation::Sigmoid:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				float x = logistic(forward_frag_in.x[t]);
				result.x[t] = frag.x[t] * (T)(x * (1.0f - x));
			}
			return;
		case Activation::Squareplus:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				float x = (float)forward_frag_in.x[t] * K_ACT;
				float y = 0.5f * (x + sqrtf(x * x + 4));
				result.x[t] = frag.x[t] * (T)(y * y / (y * y + 1));
			}
			return;
		case Activation::Softplus:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				float tmp = expf((float)forward_frag_in.x[t] * K_ACT);
				result.x[t] = frag.x[t] * (T)(tmp / (tmp + 1));
			}
			return;
		case Activation::Tanh:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				float x = tanhf(forward_frag_in.x[t]);
				result.x[t] = frag.x[t] * (T)(1.0f - x * x);
			}
			return;
		case Activation::None: result = frag; return;
		default:
			// Unsupported activation
			// assert(false); // Commented out due to isolated strange side-effects on Windows
			return;
	}
}

template <typename T, typename fragment_t, typename forward_fragment_t>
__host__ __device__ fragment_t warp_activation_backward_in(Activation activation, const fragment_t& frag, const forward_fragment_t& forward_frag_in) {
	fragment_t result;
	warp_activation_backward_in<T>(activation, frag, forward_frag_in, result);
	return result;
}

template <typename T, typename fragment_t, typename forward_fragment_t>
__host__ __device__ void warp_activation_backward(Activation activation, const fragment_t& frag, const forward_fragment_t& forward_frag, fragment_t& result) {
	switch (activation) {
		case Activation::ReLU:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)(forward_frag.x[t] > (T)0.0f);
			}
			return;
		case Activation::LeakyReLU:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)(forward_frag.x[t] > (T)0.0f ? 1.0f : 0.01f);
			}
			return;
		case Activation::Exponential:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * forward_frag.x[t];
			}
			return;
		case Activation::Sine:
			// Sine requires stored pre-activations, which we don't have. We only
			// write out the post-activations.
			// assert(false); // Commented out due to isolated strange side-effects on Windows
			return;
		case Activation::Sigmoid:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)(forward_frag.x[t] * (T)(1.0f - (float)forward_frag.x[t]));
			}
			return;
		case Activation::Squareplus:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				float y = (float)forward_frag.x[t] * K_ACT;
				result.x[t] = frag.x[t] * (T)(y * y / (y * y + 1));
			}
			return;
		case Activation::Softplus:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)(1.0f - expf(-(float)forward_frag.x[t] * K_ACT));
			}
			return;
		case Activation::Tanh:
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)(1.0f - ((float)forward_frag.x[t] * (float)forward_frag.x[t]));
			}
			return;
		case Activation::None: result = frag; return;
		default:
			// Unsupported activation
			// assert(false); // Commented out due to isolated strange side-effects on Windows
			return;
	}
}

template <typename T, typename fragment_t, typename forward_fragment_t>
__host__ __device__ fragment_t warp_activation_backward(Activation activation, const fragment_t& frag, const forward_fragment_t& forward_frag) {
	fragment_t result;
	warp_activation_backward<T>(activation, frag, forward_frag, result);
	return result;
}

template <typename T, uint32_t N>
using vector_fragment_t = VectorFragment<vector_t<T, N>>;

template <typename T, uint32_t N=1>
__global__ void kernel_activation(const uint32_t num_elements, const Activation act, const T* in, T* out) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	auto frag = ((vector_fragment_t<T, N>*)in)[i];
	warp_activation<T>(act, frag, frag);
	((vector_fragment_t<T, N>*)out)[i] = frag;
}

// Transfer functions corresponding to activations; version without biases
template <typename T, uint32_t N=1>
__global__ void kernel_activation_backward(const uint32_t num_elements, const Activation act, const T* __restrict__ values, const T* gradients_out, T* gradients_in) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	auto frag_forward_in = ((vector_fragment_t<T, N>*)values)[i];
	auto frag = ((vector_fragment_t<T, N>*)gradients_out)[i];
	warp_activation_backward_in<T>(act, frag, frag_forward_in, frag);

	((vector_fragment_t<T, N>*)gradients_in)[i] = frag;
}

// Transfer functions corresponding to activations, given _output_ values. Only works if the activation is invertible
template <typename T, uint32_t N=1>
__global__ void kernel_activation_backward_output(const uint32_t num_elements, const Activation act, const T* __restrict__ output_values, const T* gradients_out, T* gradients_in) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	auto frag_forward_out = ((vector_fragment_t<T, N>*)output_values)[i];
	auto frag = ((vector_fragment_t<T, N>*)gradients_out)[i];
	warp_activation_backward<T>(act, frag, frag_forward_out, frag);

	((vector_fragment_t<T, N>*)gradients_in)[i] = frag;
}

template <typename T>
void activation_gpu(cudaStream_t stream, const uint32_t num_elements, const Activation act, const T* in, T* out) {
	static constexpr uint32_t ACTIVATION_VECTOR_SIZE = 16u / sizeof(T);
	if (num_elements % ACTIVATION_VECTOR_SIZE != 0) {
		throw std::runtime_error{fmt::format("activation_gpu: number of elements must be a multiple of {}", ACTIVATION_VECTOR_SIZE)};
	}

	// Activation::None is a noop
	if (act == Activation::None && in == out) {
		return;
	}

	linear_kernel(kernel_activation<T, ACTIVATION_VECTOR_SIZE>, 0, stream, div_round_up(num_elements, ACTIVATION_VECTOR_SIZE), act, in, out);
}

template <typename T>
void activation_gpu(cudaStream_t stream, Activation activation, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output) {
	if (input.n() != output.n() || input.m() != output.m()) {
		throw std::runtime_error{fmt::format("Input and output don't have matching size: {} != {}", input.n(), output.n())};
	}

	activation_gpu(stream, input.n_elements(), activation, input.data(), output.data());
}

template <typename T>
void activation_backward_gpu(cudaStream_t stream, const uint32_t num_elements, const Activation act, const T* __restrict__ values, const T* gradients_out, T* gradients_in) {
	static constexpr uint32_t ACTIVATION_VECTOR_SIZE = 16u / sizeof(T);
	if (num_elements % ACTIVATION_VECTOR_SIZE != 0) {
		throw std::runtime_error{fmt::format("activation_backward_gpu: number of elements must be a multiple of {}", ACTIVATION_VECTOR_SIZE)};
	}

	// Activation transfer is a noop for Activation::None
	if (act == Activation::None && gradients_out == gradients_in) {
		return;
	}

	linear_kernel(kernel_activation_backward<T, ACTIVATION_VECTOR_SIZE>, 0, stream, div_round_up(num_elements, ACTIVATION_VECTOR_SIZE), act, values, gradients_out, gradients_in);
}

template <typename T>
void activation_backward_gpu(cudaStream_t stream, Activation activation, const GPUMatrixDynamic<T>& values, GPUMatrixDynamic<T>& gradients) {
	if (values.n() != gradients.n() || values.m() != gradients.m()) {
		throw std::runtime_error{fmt::format("Values and gradients don't have matching size: {} != {}", values.n(), gradients.n())};
	}

	activation_backward_gpu(stream, values.n_elements(), activation, values.data(), gradients.data(), gradients.data());
}

template <typename T>
void activation_backward_output_gpu(cudaStream_t stream, const uint32_t num_elements, const Activation act, const T* __restrict__ output_values, const T* gradients_out, T* gradients_in) {
	static constexpr uint32_t ACTIVATION_VECTOR_SIZE = 16u / sizeof(T);
	if (num_elements % ACTIVATION_VECTOR_SIZE != 0) {
		throw std::runtime_error{fmt::format("activation_backward_output_gpu: number of elements must be a multiple of {}", ACTIVATION_VECTOR_SIZE)};
	}

	// Activation transfer is a noop for Activation::None
	if (act == Activation::None && gradients_out == gradients_in) {
		return;
	}

	linear_kernel(kernel_activation_backward_output<T, ACTIVATION_VECTOR_SIZE>, 0, stream, div_round_up(num_elements, ACTIVATION_VECTOR_SIZE), act, output_values, gradients_out, gradients_in);
}

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__host__ __device__ inline uint32_t expand_bits(uint32_t v) {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__host__ __device__ inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
	uint32_t xx = expand_bits(x);
	uint32_t yy = expand_bits(y);
	uint32_t zz = expand_bits(z);
	return xx | (yy << 1) | (zz << 2);
}

__host__ __device__ inline uint32_t morton3D_invert(uint32_t x) {
	x = x               & 0x49249249;
	x = (x | (x >> 2))  & 0xc30c30c3;
	x = (x | (x >> 4))  & 0x0f00f00f;
	x = (x | (x >> 8))  & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

__host__ __device__ inline uint64_t expand_bits(uint64_t w)  {
	w &=                0x00000000001fffff;
	w = (w | w << 32) & 0x001f00000000ffff;
	w = (w | w << 16) & 0x001f0000ff0000ff;
	w = (w | w <<  8) & 0x010f00f00f00f00f;
	w = (w | w <<  4) & 0x10c30c30c30c30c3;
	w = (w | w <<  2) & 0x1249249249249249;
	return w;
}

__host__ __device__ inline uint64_t morton3D_64bit(uint32_t x, uint32_t y, uint32_t z)  {
	return ((expand_bits((uint64_t)x)) | (expand_bits((uint64_t)y) << 1) | (expand_bits((uint64_t)z) << 2));
}

__device__ inline float smoothstep(float val) {
	return val*val*(3.0f - 2.0f * val);
}

__device__ inline float smoothstep_derivative(float val) {
	return 6*val*(1.0f - val);
}

__device__ inline float smoothstep_2nd_derivative(float val) {
	return 6.0f - 12.0f * val;
}

__device__ inline float identity_fun(float val) {
	return val;
}

__device__ inline float identity_derivative(float val) {
	return 1.0f;
}

__device__ inline float identity_2nd_derivative(float val) {
	return 0.0f;
}

template <typename F, typename FPRIME, typename FPRIMEPRIME>
__device__ inline void pos_fract(const float input, float* pos, float* pos_derivative, float* pos_2nd_derivative, uint32_t* pos_grid, float scale, F interpolation_fun, FPRIME interpolation_fun_derivative, FPRIMEPRIME interpolation_fun_2nd_derivative) {
	// The offset of 0.5 causes different scales to be staggered with respect to each other, thus
	// preventing spurious alignment of fractional coordinates upon integer scales (or powers thereof).
	// This is mentioned in Appendix A of the "Instant Neural Graphics Primitives" paper.
	// The offset can cause wraparound indexing in dense grids, which didn't negatively impact
	// the approximation quality in any of our tests.
	*pos = input * scale + 0.5f;
	int tmp = floorf(*pos);
	*pos_grid = (uint32_t)tmp;
	*pos -= (float)tmp;
	*pos_2nd_derivative = interpolation_fun_2nd_derivative(*pos);
	*pos_derivative = interpolation_fun_derivative(*pos);
	*pos = interpolation_fun(*pos);
}

template <typename F, typename FPRIME>
__device__ inline void pos_fract(const float input, float* pos, float* pos_derivative, uint32_t* pos_grid, float scale, F interpolation_fun, FPRIME interpolation_fun_derivative) {
	// The offset of 0.5 causes different scales to be staggered with respect to each other, thus
	// preventing spurious alignment of fractional coordinates upon integer scales (or powers thereof).
	// This is mentioned in Appendix A of the "Instant Neural Graphics Primitives" paper.
	// The offset can cause wraparound indexing in dense grids, which didn't negatively impact
	// the approximation quality in any of our tests.
	*pos = input * scale + 0.5f;
	int tmp = floorf(*pos);
	*pos_grid = (uint32_t)tmp;
	*pos -= (float)tmp;
	*pos_derivative = interpolation_fun_derivative(*pos);
	*pos = interpolation_fun(*pos);
}

template <typename F>
__device__ inline void pos_fract(const float input, float* pos, uint32_t* pos_grid, float scale, F interpolation_fun) {
	// The offset of 0.5 causes different scales to be staggered with respect to each other, thus
	// preventing spurious alignment of fractional coordinates upon integer scales (or powers thereof).
	// This is mentioned in Appendix A of the "Instant Neural Graphics Primitives" paper.
	// The offset can cause wraparound indexing in dense grids, which didn't negatively impact
	// the approximation quality in any of our tests.
	*pos = input * scale + 0.5f;
	int tmp = floorf(*pos);
	*pos_grid = (uint32_t)tmp;
	*pos -= (float)tmp;
	*pos = interpolation_fun(*pos);
}

__device__ inline float weight_decay(float relative_weight_decay, float absolute_weight_decay, float weight) {
	// Relative weight decay is closely related to l2 regularization, whereas absolute weight decay corresponds to l1 regularization
	return (1 - relative_weight_decay) * weight - copysignf(absolute_weight_decay, weight);
}

__device__ inline float gaussian_cdf(const float x, const float inv_radius) {
	return normcdff(x * inv_radius);
}

__device__ inline float gaussian_cdf_approx(const float x, const float inv_radius) {
	static constexpr float MAGIC_SIGMOID_FACTOR = 1.12f / SQRT2;
	return logistic(MAGIC_SIGMOID_FACTOR * x * inv_radius);
}

__device__ inline float gaussian_cdf_approx_derivative(const float result, const float inv_radius) {
	static constexpr float MAGIC_SIGMOID_FACTOR = 1.12f / SQRT2;
	return result * (1 - result) * MAGIC_SIGMOID_FACTOR * inv_radius;
}

__device__ inline float gaussian_pdf(const float x, const float inv_radius) {
	return inv_radius * rsqrtf(2.0f * PI) * expf(-0.5f * (x * x * inv_radius * inv_radius));
}

__device__ inline float gaussian_pdf_max_1(const float x, const float inv_radius) {
	return expf(-0.5f * (x * x * inv_radius * inv_radius));
}

__device__ inline float tent(const float x, const float inv_radius) {
	return fmaxf(1.0f - fabsf(x * inv_radius), 0.0f);
}

__device__ inline float tent_cdf(const float x, const float inv_radius) {
	return fmaxf(0.0f, fminf(1.0f, x * inv_radius + 0.5f));
}

__device__ inline float quartic(const float x, const float inv_radius) {
	const float u = x * inv_radius;
	const float tmp = fmaxf(1 - u*u, 0.0f);
	return ((float)15 / 16) * tmp * tmp;
}

__device__ inline float quartic_cdf_deriv(const float x, const float inv_radius) {
	return quartic(x, inv_radius) * inv_radius;
}

__device__ inline float quartic_cdf(const float x, const float inv_radius) {
	const float u = x * inv_radius;
	const float u2 = u * u;
	const float u4 = u2 * u2;
	return fmaxf(0.0f, fminf(1.0f, ((float)15 / 16) * u * (1 - ((float)2 / 3) * u2 + ((float)1 / 5) * u4) + 0.5f));
}

__device__ inline uint32_t permute(uint32_t num, uint32_t size) {
	const uint32_t A = 1434869437; // Large prime number
	const uint32_t B = 2097192037;
	return ((uint64_t)num * A + B) % size;
}

template <typename T>
__global__ void shuffle(const uint32_t n_elements, const uint32_t stride, const uint32_t seed, const T* __restrict__ in, T* __restrict__ out) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements * stride) return;

	const uint32_t elem_id = i / stride;
	const uint32_t member_id = i % stride;

	out[i] = in[permute(elem_id + seed, n_elements) * stride + member_id];
}

template <typename T>
__global__ void fill_rollover(const uint32_t n_elements, const uint32_t stride, const uint32_t* n_input_elements_ptr, T* inout) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t n_input_elements = *n_input_elements_ptr;

	if (i < (n_input_elements * stride) || i >= (n_elements * stride) || n_input_elements == 0) return;

	T result = inout[i % (n_input_elements * stride)];
	inout[i] = result;
}

template <typename T>
__global__ void fill_rollover_and_rescale(const uint32_t n_elements, const uint32_t stride, const uint32_t* n_input_elements_ptr, T* inout) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t n_input_elements = *n_input_elements_ptr;

	if (i < (n_input_elements * stride) || i >= (n_elements * stride) || n_input_elements == 0) return;

	T result = inout[i % (n_input_elements * stride)];
	result = (T)((float)result * n_input_elements / n_elements);
	inout[i] = result;
}

template <typename T1, typename T2, typename T3>
__global__ void add(const uint32_t num_elements, const T1* data_in_1, const T2* data_in_2, T3* data_out) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_out[i] = (T3)((float)data_in_1[i] + (float)data_in_2[i]);
}

template <typename T>
__global__ void add(const uint32_t num_elements, const T* __restrict__ data_in, T* __restrict__ data_in_out) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_in_out[i] = data_in[i] + data_in_out[i];
}

template <typename T>
__global__ void trim(const uint32_t num_elements, const uint32_t stride, const uint32_t dims, const T* __restrict__ data_in, T* __restrict__ data_out) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	uint32_t idx = i % dims;
	uint32_t elem = i / dims;

	data_out[i] = data_in[elem * stride + idx];
}

template <typename T>
__global__ void trim_and_cast(const uint32_t num_elements, const uint32_t stride, const uint32_t dims, const T* __restrict__ data_in, float* __restrict__ data_out) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	uint32_t idx = i % dims;
	uint32_t elem = i / dims;

	data_out[i] = (float)data_in[elem * stride + idx];
}

template <typename T>
__global__ void cast(const uint32_t num_elements, const float* __restrict__ full_precision, T* __restrict__ target) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	target[i] = (T)full_precision[i];
}

template <typename T>
__global__ void cast_from(const uint32_t num_elements, const T* __restrict__ precision, float* __restrict__ full_precision) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	full_precision[i] = (float)precision[i];
}

template <typename T>
__global__ void extract_dimension_pos_neg_kernel(const uint32_t num_elements, const uint32_t dim, const uint32_t fan_in, const uint32_t fan_out, const T* __restrict__ encoded, const MatrixLayout layout, float* __restrict__ output) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint32_t elem_idx = i / fan_out;
	const uint32_t dim_idx = i % fan_out;

	const uint32_t encoded_idx = layout == MatrixLayout::AoS ? (elem_idx * fan_in + dim) : (elem_idx + dim * num_elements / fan_out);

	if (fan_out == 1) {
		output[i] = (float)encoded[encoded_idx];
		return;
	}

	if (dim_idx == 0) {
		output[i] = fmaxf(-(float)encoded[encoded_idx], 0.0f);
	} else if (dim_idx == 1) {
		output[i] = fmaxf((float)encoded[encoded_idx], 0.0f);
	} else if (dim_idx == 2) {
		output[i] = 0;
	} else {
		output[i] = 1;
	}
}

template <typename T>
__global__ void mult_scalar_kernel(const uint32_t num_elements, T* __restrict__ inout, float factor) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	inout[i] = (T)((float)inout[i] * factor);
}

template <typename T>
__global__ void mult_kernel(const uint32_t num_elements, const T* factor1, const T* factor2, T* result) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	result[i] = factor1[i] * factor2[i];
}

TCNN_NAMESPACE_END
