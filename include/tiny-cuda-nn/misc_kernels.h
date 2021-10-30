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

/** @file   misc_kernels.h
 *  @author Thomas Müller & Nikolaus Binder, NVIDIA
 *  @brief  Implementation of various miscellaneous CUDA kernels
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#define _USE_MATH_DEFINES
#include <cmath>

#include <cuda_fp16.h>

#include <cassert>
#include <cstdint>
#include <cstdio>

#include <tiny-cuda-nn/activations.h>

#ifndef M_PI
#include <corecrt_math_defines.h>
#endif


TCNN_NAMESPACE_BEGIN

enum InterpolationType {
	Linear,
	Smoothstep,
};

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ inline uint32_t expand_bits(uint32_t v) {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
	uint32_t xx = expand_bits(x);
	uint32_t yy = expand_bits(y);
	uint32_t zz = expand_bits(z);
	return xx * 4 + yy * 2 + zz;
}

__device__ inline uint32_t morton3D_invert(uint32_t x) {
	x = x               & 0x49249249;
	x = (x | (x >> 2))  & 0xc30c30c3;
	x = (x | (x >> 4))  & 0x0f00f00f;
	x = (x | (x >> 8))  & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

__device__ inline float smoothstep(float val) {
	return val*val*(3.0f - 2.0f * val);
}

__device__ inline float smoothstep_derivative(float val) {
	return 6*val*(1.0f - val);
}

__device__ inline float identity_fun(float val) {
	return val;
}

__device__ inline float identity_derivative(float val) {
	return 1;
}

template <typename F, typename FPRIME>
__device__ inline void pos_fract(const float input, float* pos, float* pos_derivative, uint32_t* pos_grid, float scale, F interpolation_fun, FPRIME interpolation_fun_derivative) {
	*pos = input * scale + 0.5f;
	int tmp = __float2int_rd(*pos);
	*pos_grid = (uint32_t)tmp;
	*pos -= __int2float_rd(tmp);
	*pos_derivative = interpolation_fun_derivative(*pos);
	*pos = interpolation_fun(*pos);
}

template <typename F>
__device__ inline void pos_fract(const float input, float* pos, uint32_t* pos_grid, float scale, F interpolation_fun) {
	*pos = input * scale + 0.5f;
	int tmp = __float2int_rd(*pos);
	*pos_grid = (uint32_t)tmp;
	*pos -= __int2float_rd(tmp);
	*pos = interpolation_fun(*pos);
}

__device__ inline float weight_decay(float relative_weight_decay, float absolute_weight_decay, float weight) {
	// Relative weight decay is closely related to l2 regularization, whereas absolute weight decay corresponds to l1 regularization
	return (1 - relative_weight_decay) * weight - copysignf(absolute_weight_decay, weight);
}

__device__ inline float logistic(const float x) {
	return __frcp_rn(1.0f + __expf(-x));
}

__device__ inline float logit(const float x) {
	return -__logf(__frcp_rn(fminf(fmaxf(x, 1e-8f), 1.0f - 1e-8f)) - 1.0f);
}

__device__ inline float gaussian_cdf(const float x, const float inv_radius) {
	return normcdff(x * inv_radius);
}

__device__ inline float gaussian_cdf_approx(const float x, const float inv_radius) {
	static constexpr float MAGIC_SIGMOID_FACTOR = 1.12f / M_SQRT2;
	return logistic(MAGIC_SIGMOID_FACTOR * x * inv_radius);
}

__device__ inline float gaussian_cdf_approx_derivative(const float result, const float inv_radius) {
	static constexpr float MAGIC_SIGMOID_FACTOR = 1.12f / M_SQRT2;
	return result * (1 - result) * MAGIC_SIGMOID_FACTOR * inv_radius;
}

__device__ inline float gaussian_pdf(const float x, const float inv_radius) {
	return inv_radius * rsqrtf(2.0f * M_PI) * expf(-0.5f * (x * x * inv_radius * inv_radius));
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
    const uint32_t A = 10002659; // Large prime number
    const uint32_t B = 4234151;
    return (num * A + B) % size;
}

template <typename T>
__global__ void shuffle(const uint32_t n_elements, const uint32_t stride, const uint32_t seed, const T* __restrict__ in, T* __restrict__ out) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements * stride) return;

    const uint32_t elem_id = i / stride;
    const uint32_t member_id = i % stride;

    out[i] = in[permute(elem_id ^ seed, n_elements) * stride + member_id];
}

template <typename T>
__global__ void fill_rollover(const uint32_t n_elements, const uint32_t stride, const uint32_t* n_input_elements_ptr, T* inout) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t n_input_elements = *n_input_elements_ptr;

    if (i < (n_input_elements * stride) || i >= (n_elements * stride) || n_input_elements == 0) return;

    inout[i] = inout[i % (n_input_elements * stride)];
}

template <typename T>
__global__ void relu(const uint32_t num_elements, const uint32_t width, const T* data_in, T* data_out, const T* biases) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint32_t bias_idx = i % width;

	data_out[i] = fmaxf(0.0f, data_in[i] + biases[bias_idx]);
}

template <typename T>
__global__ void exp(const uint32_t num_elements, const uint32_t width, const T* data_in, T* data_out, const T* biases) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint32_t bias_idx = i % width;

	data_out[i] = expf(data_in[i] + biases[bias_idx]);
}

template <typename T>
__global__ void sin(const uint32_t num_elements, const uint32_t width, const T* data_in, T* data_out, const T* biases)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint32_t bias_idx = i % width;

	data_out[i] = __sinf(data_in[i] + biases[bias_idx]);
}

template <typename T>
__global__ void relu(const uint32_t num_elements, const T* data_in, T* data_out) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_out[i] = fmaxf(0.0f, data_in[i]);
}

template <typename T>
__global__ void exp(const uint32_t num_elements, const T* data_in, T* data_out) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_out[i] = expf(data_in[i]);
}

template <typename T>
__global__ void sin(const uint32_t num_elements, const T* data_in, T* data_out)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_out[i] = __sinf(data_in[i]);
}

template <typename T1, typename T2, typename T3>
__global__ void add(const uint32_t num_elements, const T1* data_in_1, const T2* data_in_2, T3* data_out) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_out[i] = (T3)((float)data_in_1[i] + (float)data_in_2[i]);
}

template <typename T>
__global__ void add(const uint32_t num_elements, const T* __restrict__ data_in, T* __restrict__ data_in_out)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_in_out[i] = data_in[i] + data_in_out[i];
}

template <typename T>
__global__ void relu(const uint32_t num_elements, T* __restrict__ data_in_out)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_in_out[i] = fmaxf(0.0f, data_in_out[i]);
}

template <typename T>
__global__ void exp(const uint32_t num_elements, T* __restrict__ data_in_out)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_in_out[i] = expf(data_in_out[i]);
}

template <typename T>
__global__ void sin(const uint32_t num_elements, T* __restrict__ data_in_out)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_in_out[i] = sinf(data_in_out[i]);
}

// Transfer functions corresponding to activations; version without biases
template <typename T>
__global__ void relu_transfer(const uint32_t num_elements, const T* __restrict__ values, T* __restrict__ gradients)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	gradients[i] = values[i] > (T)0 ? gradients[i] : (T)0;
}

template <typename T>
__global__ void exp_transfer(const uint32_t num_elements, const T* __restrict__ values, T* __restrict__ gradients)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	gradients[i] = (T)((float)gradients[i] * expf((float)values[i]));
}

template <typename T>
__global__ void sin_transfer(const uint32_t num_elements, const T* __restrict__ values, T* __restrict__ gradients)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	gradients[i] = (T)((float)gradients[i] * cosf((float)values[i]));
}

// Transfer functions corresponding to activations; version with biases
template <typename T>
__global__ void relu_transfer(const uint32_t num_elements, const uint32_t width, const T* __restrict__ values, T* __restrict__ gradients, const T* biases)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint32_t bias_idx = i % width;
	const float bias = biases[bias_idx];

	gradients[i] = ((float)values[i] + bias) > 0.0f ? gradients[i] : (T)0;
}

template <typename T>
__global__ void exp_transfer(const uint32_t num_elements, const uint32_t width, const T* __restrict__ values, T* __restrict__ gradients, const T* biases)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint32_t bias_idx = i % width;
	const float bias = biases[bias_idx];

	gradients[i] = (T)((float)gradients[i] * expf((float)values[i] + bias));
}

template <typename T>
__global__ void sin_transfer(const uint32_t num_elements, const uint32_t width, const T* __restrict__ values, T* __restrict__ gradients, const T* biases)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint32_t bias_idx = i % width;
	const float bias = biases[bias_idx];

	gradients[i] = (T)((float)gradients[i] * cosf((float)values[i] + bias));
}

// Transfer functions corresponding to activations, given _output_ values. Only works if the activation is invertible
template <typename T>
__global__ void relu_transfer_output(const uint32_t num_elements, const T* __restrict__ output_values, const T* __restrict__ gradients_out, T* __restrict__ gradients_in)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	gradients_in[i] = output_values[i] > (T)0 ? gradients_out[i] : (T)0;
}

template <typename T>
__global__ void exp_transfer_output(const uint32_t num_elements, const T* __restrict__ output_values, const T* __restrict__ gradients_out, T* __restrict__ gradients_in)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	gradients_in[i] = (T)((float)gradients_out[i] * (float)output_values[i]);

	// L2 regularization for too small values to prevent vanishing gradients
	// gradients_in[i] += fminf(logf(output_values[i] + 1e-8f) + 4.0f, 0.0f);
}

template <typename T>
__global__ void logistic_transfer_output(const uint32_t num_elements, const T* __restrict__ output_values, const T* __restrict__ gradients_out, T* __restrict__ gradients_in)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	gradients_in[i] = (T)((float)gradients_out[i] * (float)output_values[i] * (1.0f - (float)output_values[i]));

	// L2 regularization for too small values to prevent vanishing gradients
	// gradients_in[i] += fminf(logf(output_values[i] + 1e-8f) + 4.0f, 0.0f);
}

template <typename T>
__global__ void trim(const uint32_t num_elements, const uint32_t stride, const uint32_t dims, const T* __restrict__ data_in, T* __restrict__ data_out)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	uint32_t idx = i % dims;
	uint32_t elem = i / dims;

	data_out[i] = data_in[elem * stride + idx];
}

template <typename T>
__global__ void trim_and_cast(const uint32_t num_elements, const uint32_t stride, const uint32_t dims, const T* __restrict__ data_in, float* __restrict__ data_out)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	uint32_t idx = i % dims;
	uint32_t elem = i / dims;

	data_out[i] = (float)data_in[elem * stride + idx];
}

template <typename T>
__global__ void cast(const uint32_t num_elements, const float* __restrict__ full_precision, T* __restrict__ target)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	target[i] = (T)full_precision[i];
}

template <typename T>
__global__ void cast_from(const uint32_t num_elements, const T* __restrict__ precision, float* __restrict__ full_precision)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	full_precision[i] = (float)precision[i];
}

TCNN_NAMESPACE_END
