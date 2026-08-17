/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   permuto.h
 *  @author Jianfei Guo, NVIDIA
 *  @brief  Trainable hierarchy of N-D permutohedral lattice grids of floating point values.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/encodings/multi_level_interface.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/random.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

namespace tcnn {

static constexpr uint32_t PERMUTO_MAX_N_LEVELS = 32;

inline constexpr TCNN_HOST_DEVICE bool permuto_level_is_inactive(uint32_t level, float max_level) { return level >= max_level + 1e-3f; }

template <uint32_t N_DIMS>
__device__ __forceinline__ void permuto_prepare(
	uint32_t element,
	uint32_t level,
	float base_scale,
	float log2_per_level_scale,
	pcg32 rng,
	MatrixView<const float> positions_in,
	float* __restrict__ scales_per_dim,
	float* __restrict__ elevated,
	int* __restrict__ rem0,
	int* __restrict__ rank
) {
	float pos[N_DIMS];
	float shifts_per_dim[N_DIMS];
	const float scale = base_scale * exp2f(level * log2_per_level_scale);
	rng.advance(level * N_DIMS);
	TCNN_PRAGMA_UNROLL
	for (uint32_t dim = 0; dim < N_DIMS; ++dim) {
		pos[dim] = positions_in(dim, element);
		scales_per_dim[dim] = scale * rsqrtf((dim + 1) * (dim + 2));
		shifts_per_dim[dim] = fmaf(rng.next_float(), 10.0f, -5.0f);
	}

	float sum = 0;
	TCNN_PRAGMA_UNROLL
	for (int dim = N_DIMS; dim > 0; --dim) {
		const float cf = (pos[dim - 1] + shifts_per_dim[dim - 1]) * scales_per_dim[dim - 1];
		elevated[dim] = sum - (float)dim * cf;
		sum += cf;
	}
	elevated[0] = sum;

	int rem0_sum = 0;
	TCNN_PRAGMA_UNROLL
	for (uint32_t dim = 0; dim <= N_DIMS; ++dim) {
		const float v = elevated[dim] * (1.0f / (N_DIMS + 1));
		const float up = ceil(v) * (N_DIMS + 1);
		const float down = floor(v) * (N_DIMS + 1);
		rem0[dim] = up - elevated[dim] < elevated[dim] - down ? (int)up : (int)down;
		rem0_sum += rem0[dim];
	}
	rem0_sum /= (int)(N_DIMS + 1);

	TCNN_PRAGMA_UNROLL
	for (uint32_t dim = 0; dim < N_DIMS; ++dim) {
		const float di = elevated[dim] - rem0[dim];
		for (uint32_t other_dim = dim + 1; other_dim <= N_DIMS; ++other_dim) {
			if (di < elevated[other_dim] - rem0[other_dim]) {
				++rank[dim];
			} else {
				++rank[other_dim];
			}
		}
	}

	TCNN_PRAGMA_UNROLL
	for (uint32_t dim = 0; dim <= N_DIMS; ++dim) {
		rank[dim] += rem0_sum;
		if (rank[dim] < 0) {
			rank[dim] += (int)(N_DIMS + 1);
			rem0[dim] += (int)(N_DIMS + 1);
		} else if (rank[dim] > (int)N_DIMS) {
			rank[dim] -= (int)(N_DIMS + 1);
			rem0[dim] -= (int)(N_DIMS + 1);
		}
	}
}

template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL>
__global__ void kernel_permuto(
	const uint32_t num_elements,
	const uint32_t num_grid_features,
	const ParamsOffsetTable offset_table,
	const float base_scale,
	const float log2_per_level_scale,
	pcg32 rng,
	float max_level,
	const float* __restrict__ max_level_gpu,
	const T* __restrict__ grid,
	MatrixView<const float> positions_in,
	MatrixView<T> encoded_positions
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elements) {
		return;
	}

	const uint32_t level = blockIdx.y; // <- the level is the same for all threads

	if (max_level_gpu) {
		max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
	} else {
		max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;
	}

	if (permuto_level_is_inactive(level, max_level)) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
			encoded_positions(level * N_FEATURES_PER_LEVEL + f, i) = (T)0.0f;
		}
		return;
	}

	grid += offset_table.data[level] * N_FEATURES_PER_LEVEL;
	const uint32_t hashmap_size = offset_table.data[level + 1] - offset_table.data[level];

	float scales_per_dim[N_POS_DIMS];
	float elevated[N_POS_DIMS + 1];
	int rem0[N_POS_DIMS + 1];
	int rank[N_POS_DIMS + 1]{0};
	permuto_prepare<N_POS_DIMS>(i, level, base_scale, log2_per_level_scale, rng, positions_in, scales_per_dim, elevated, rem0, rank);

	//---- Compute the barycentric coordinates (p.10 in [Adams etal 2010])
	float barycentric[N_POS_DIMS + 2]{0};
	TCNN_PRAGMA_UNROLL
	for (uint32_t dim = 0; dim <= N_POS_DIMS; ++dim) {
		const float delta = (elevated[dim] - rem0[dim]) * (1.0f / (N_POS_DIMS + 1));
		barycentric[(int)N_POS_DIMS - rank[dim]] += delta;
		barycentric[(int)(N_POS_DIMS + 1) - rank[dim]] -= delta;
	}
	barycentric[0] += 1.0f + barycentric[N_POS_DIMS + 1];

	//---- Prepare
	tvec<T, N_FEATURES_PER_LEVEL, PARAMS_ALIGNED ? sizeof(T) * N_FEATURES_PER_LEVEL : sizeof(T)> result((T)0.0f);

	//---- Interpolate the values to calculate encoded
	uvec<N_POS_DIMS> key;
	TCNN_PRAGMA_UNROLL
	for (uint32_t k = 0; k <= N_POS_DIMS; ++k) { // For each remainder-k vertex: for k \in {0,1,...,d}
		// Compute the coordinates of the remainder-k vertex explicitly
		// (all but the last coordinate - it's redundant because they sum to zero)
		TCNN_PRAGMA_UNROLL
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			key[dim] = rem0[dim] + (int)k;
			if (rank[dim] > (int)(N_POS_DIMS - k)) {
				key[dim] -= (int)(N_POS_DIMS + 1);
			}
		}

		// Accumulate vertex's feature value by the barycentric weight
		const float weight = barycentric[k];
		const uint32_t index = (base_convert_hash<N_POS_DIMS>(key) % hashmap_size) * N_FEATURES_PER_LEVEL;
		const auto value = *(tvec < T, N_FEATURES_PER_LEVEL, PARAMS_ALIGNED ? sizeof(T) * N_FEATURES_PER_LEVEL : sizeof(T) > *)&grid[index];
		result = fma((T)weight, value, result);
	}

	TCNN_PRAGMA_UNROLL
	for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
		encoded_positions(level * N_FEATURES_PER_LEVEL + f, i) = result[f];
	}
}

template <typename T, typename GRAD_T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t N_FEATURES_PER_THREAD>
__global__ void kernel_permuto_backward(
	const uint32_t num_elements,
	const uint32_t num_grid_features,
	const ParamsOffsetTable offset_table,
	const float base_scale,
	const float log2_per_level_scale,
	pcg32 rng,
	float max_level,
	const float* __restrict__ max_level_gpu,
	// Inputs
	MatrixView<const T> dL_dy,
	MatrixView<const float> positions_in,
	// Outputs
	GRAD_T* __restrict__ grid_gradient
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
	if (i >= num_elements) {
		return;
	}

	const uint32_t level = blockIdx.y; // <- the level is the same for all threads.
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

	if (max_level_gpu) {
		max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
	} else {
		max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;
	}

	if (permuto_level_is_inactive(level, max_level)) {
		return;
	}

	grid_gradient += offset_table.data[level] * N_FEATURES_PER_LEVEL;
	const uint32_t hashmap_size = offset_table.data[level + 1] - offset_table.data[level];

	float scales_per_dim[N_POS_DIMS];
	float elevated[N_POS_DIMS + 1];
	int rem0[N_POS_DIMS + 1];
	int rank[N_POS_DIMS + 1]{0};
	permuto_prepare<N_POS_DIMS>(i, level, base_scale, log2_per_level_scale, rng, positions_in, scales_per_dim, elevated, rem0, rank);

	//---- Compute the barycentric coordinates (p.10 in [Adams etal 2010])
	float barycentric[N_POS_DIMS + 2]{0};
	TCNN_PRAGMA_UNROLL
	for (uint32_t dim = 0; dim <= N_POS_DIMS; ++dim) {
		const float delta = (elevated[dim] - rem0[dim]) * (1.0f / (N_POS_DIMS + 1));
		barycentric[(int)N_POS_DIMS - rank[dim]] += delta;
		barycentric[(int)(N_POS_DIMS + 1) - rank[dim]] -= delta;
	}
	barycentric[0] += 1.0f + barycentric[N_POS_DIMS + 1];

	// Force using float to ensure numerical stability
	tvec<float, N_FEATURES_PER_THREAD> grad;
	TCNN_PRAGMA_UNROLL
	for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
		grad[f] = (float)dL_dy(level * N_FEATURES_PER_LEVEL + feature + f, i);
	}

	//---- Calculate grid_gradient
	uvec<N_POS_DIMS> key;
	TCNN_PRAGMA_UNROLL
	for (uint32_t k = 0; k <= N_POS_DIMS; ++k) { // For each remainder-k vertex: for k \in {0,1,...,d}
		// Compute the coordinates of the remainder-k vertex explicitly
		// (all but the last coordinate - it's redundant because they sum to zero)
		TCNN_PRAGMA_UNROLL
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			key[dim] = rem0[dim] + (int)k;
			if (rank[dim] > (int)(N_POS_DIMS - k)) {
				key[dim] -= (int)(N_POS_DIMS + 1);
			}
		}

		// Accumulate vertex's gradients by the barycentric weight
		const float weight = barycentric[k];
		const uint32_t index = (base_convert_hash<N_POS_DIMS>(key) % hashmap_size) * N_FEATURES_PER_LEVEL + feature;
		atomic_add_gmem(grid_gradient + index, tvec<GRAD_T, N_FEATURES_PER_THREAD>(weight * grad));
	}
}

template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t N_FEATURES_PER_THREAD>
__global__ void kernel_permuto_backward_input(
	const uint32_t num_elements,
	const uint32_t num_grid_features,
	const ParamsOffsetTable offset_table,
	const float base_scale,
	const float log2_per_level_scale,
	pcg32 rng,
	float max_level,
	const float* __restrict__ max_level_gpu,
	const uint32_t max_input_grad_dims, // If we want to limit the computation to first-n dims
	// Inputs
	const T* __restrict__ grid,
	MatrixView<const T> dL_dy,
	MatrixView<const float> positions_in,
	// Outputs
	MatrixView<float> dL_dx
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
	if (i >= num_elements) {
		return;
	}

	const uint32_t level = blockIdx.y; // <- the level is the same for all threads.
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

	if (max_level_gpu) {
		max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
	} else {
		max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;
	}

	if (permuto_level_is_inactive(level, max_level)) {
		return;
	}

	grid += offset_table.data[level] * N_FEATURES_PER_LEVEL;
	const uint32_t hashmap_size = offset_table.data[level + 1] - offset_table.data[level];

	float scales_per_dim[N_POS_DIMS];
	float elevated[N_POS_DIMS + 1];
	int rem0[N_POS_DIMS + 1];
	int rank[N_POS_DIMS + 1]{0};
	permuto_prepare<N_POS_DIMS>(i, level, base_scale, log2_per_level_scale, rng, positions_in, scales_per_dim, elevated, rem0, rank);

	// Force using float to ensure numerical stability
	tvec<float, N_FEATURES_PER_THREAD> grad;
	TCNN_PRAGMA_UNROLL
	for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
		grad[f] = (float)dL_dy(level * N_FEATURES_PER_LEVEL + feature + f, i);
	}

	//---- Calculate dL_dx
	// The upstream gradient dL/dy differentiates the loss with respect to the encoded value.
	// If we require positions grad we want to obtain dL/dx
	// dL/dx = dL/dy * dy/dB * dB/dE * dE/dx
	// We need dy/dB which is the derivative of the sliced value wrt to the barycentric coords
	// We need dB/dE which is the derivative of the barycentric wrt to the elevated value
	// We need dE/dx which is the derivative of the elevated wrt to the position in xyz

	// dL/dB = dL/dy * dy/dB
	uvec<N_POS_DIMS> key;
	float dL_dbarycentric[N_POS_DIMS + 2]{0};
	TCNN_PRAGMA_UNROLL
	for (uint32_t k = 0; k <= N_POS_DIMS; ++k) { // For each remainder-k vertex: for k \in {0,1,...,d}
		// Compute the coordinates of the remainder-k vertex explicitly
		// (all but the last coordinate - it's redundant because they sum to zero)
		TCNN_PRAGMA_UNROLL
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			key[dim] = rem0[dim] + (int)k;
			if (rank[dim] > (int)(N_POS_DIMS - k)) {
				key[dim] -= (int)(N_POS_DIMS + 1);
			}
		}

		// Add to dL_d_barycentric
		const uint32_t index = (base_convert_hash<N_POS_DIMS>(key) % hashmap_size) * N_FEATURES_PER_LEVEL + feature;
		const auto val = *(tvec < T, N_FEATURES_PER_THREAD, PARAMS_ALIGNED ? sizeof(T) * N_FEATURES_PER_THREAD : sizeof(T) > *)&grid[index];
		TCNN_PRAGMA_UNROLL
		for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
			dL_dbarycentric[k] += (float)val[f] * (float)grad[f];
		}
	}
	dL_dbarycentric[N_POS_DIMS + 1] += dL_dbarycentric[0];

	// dL/dE = dL/dB * dB/dE
	float dL_delevated[N_POS_DIMS + 1]{0};
	TCNN_PRAGMA_UNROLL
	for (uint32_t dim = 0; dim <= N_POS_DIMS; ++dim) {
		dL_delevated[dim] += dL_dbarycentric[(int)N_POS_DIMS - rank[dim]] * (1.0f / (N_POS_DIMS + 1));
		dL_delevated[dim] -= dL_dbarycentric[(int)(N_POS_DIMS + 1) - rank[dim]] * (1.0f / (N_POS_DIMS + 1));
	}

	// dL/dx = dL/dE * dE/dx
	tvec<float, N_POS_DIMS> dL_dx_local(0.0f);
	TCNN_PRAGMA_UNROLL
	for (uint32_t dim = 0; dim < max_input_grad_dims; ++dim) {
		float dL_dx_dim = 0;
		TCNN_PRAGMA_UNROLL
		for (uint32_t other_dim = 0; other_dim <= dim; ++other_dim) {
			dL_dx_dim += dL_delevated[other_dim] * scales_per_dim[dim];
		}
		dL_dx_dim -= dL_delevated[dim + 1] * scales_per_dim[dim] * (dim + 1);
		dL_dx_local[dim] = dL_dx_dim;
	}

	//---- Finish
	// Should be atomic, since different levels of dL_dy can backward to the same input.
	TCNN_PRAGMA_UNROLL
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		atomicAdd(&dL_dx(dim, i), dL_dx_local[dim]);
	}
}

template <typename T, typename GRAD_T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t N_FEATURES_PER_THREAD>
__global__ void kernel_permuto_backward_backward_input(
	const uint32_t num_elements,
	const uint32_t num_grid_features,
	const ParamsOffsetTable offset_table,
	const float base_scale,
	const float log2_per_level_scale,
	pcg32 rng,
	float max_level,
	const float* __restrict__ max_level_gpu,
	const uint32_t max_input_grad_dims,
	MatrixView<const float> dL_ddLdx,
	MatrixView<const float> positions_in,
	MatrixView<const T> dL_dy,
	const T* __restrict__ grid,
	GRAD_T* __restrict__ grid_gradient,
	MatrixView<T> dL_ddLdy
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
	if (i >= num_elements) {
		return;
	}

	const uint32_t level = blockIdx.y;
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

	if (max_level_gpu) {
		max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
	} else {
		max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;
	}

	if (permuto_level_is_inactive(level, max_level)) {
		if (dL_ddLdy.data) {
			TCNN_PRAGMA_UNROLL
			for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
				dL_ddLdy(level * N_FEATURES_PER_LEVEL + feature + f, i) = (T)0.0f;
			}
		}
		return;
	}

	grid += offset_table.data[level] * N_FEATURES_PER_LEVEL;
	if (grid_gradient) {
		grid_gradient += offset_table.data[level] * N_FEATURES_PER_LEVEL;
	}
	const uint32_t hashmap_size = offset_table.data[level + 1] - offset_table.data[level];

	float scales_per_dim[N_POS_DIMS];
	float elevated[N_POS_DIMS + 1];
	int rem0[N_POS_DIMS + 1];
	int rank[N_POS_DIMS + 1]{0};
	permuto_prepare<N_POS_DIMS>(i, level, base_scale, log2_per_level_scale, rng, positions_in, scales_per_dim, elevated, rem0, rank);

	float dL_delevated[N_POS_DIMS + 1]{0};
	TCNN_PRAGMA_UNROLL
	for (uint32_t dim = 0; dim < max_input_grad_dims; ++dim) {
		const float grad = dL_ddLdx(dim, i) * scales_per_dim[dim];
		TCNN_PRAGMA_UNROLL
		for (uint32_t other_dim = 0; other_dim <= dim; ++other_dim) {
			dL_delevated[other_dim] += grad;
		}
		dL_delevated[dim + 1] -= grad * (dim + 1);
	}

	float dL_dbarycentric[N_POS_DIMS + 2]{0};
	TCNN_PRAGMA_UNROLL
	for (uint32_t dim = 0; dim <= N_POS_DIMS; ++dim) {
		const float grad = dL_delevated[dim] * (1.0f / (N_POS_DIMS + 1));
		dL_dbarycentric[(int)N_POS_DIMS - rank[dim]] += grad;
		dL_dbarycentric[(int)(N_POS_DIMS + 1) - rank[dim]] -= grad;
	}
	dL_dbarycentric[0] += dL_dbarycentric[N_POS_DIMS + 1];

	tvec<float, N_FEATURES_PER_THREAD> dL_ddLdy_local(0.0f);
	tvec<float, N_FEATURES_PER_THREAD> dL_dy_local;
	TCNN_PRAGMA_UNROLL
	for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
		dL_dy_local[f] = (float)dL_dy(level * N_FEATURES_PER_LEVEL + feature + f, i);
	}

	uvec<N_POS_DIMS> key;
	TCNN_PRAGMA_UNROLL
	for (uint32_t k = 0; k <= N_POS_DIMS; ++k) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			key[dim] = rem0[dim] + (int)k;
			if (rank[dim] > (int)(N_POS_DIMS - k)) {
				key[dim] -= (int)(N_POS_DIMS + 1);
			}
		}

		const uint32_t index = (base_convert_hash<N_POS_DIMS>(key) % hashmap_size) * N_FEATURES_PER_LEVEL + feature;
		if (dL_ddLdy.data) {
			const auto value = *(tvec<T, N_FEATURES_PER_THREAD, PARAMS_ALIGNED ? sizeof(T) * N_FEATURES_PER_THREAD : sizeof(T)>*)&grid[index];
			TCNN_PRAGMA_UNROLL
			for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
				dL_ddLdy_local[f] = fmaf(dL_dbarycentric[k], (float)value[f], dL_ddLdy_local[f]);
			}
		}

		if (grid_gradient) {
			atomic_add_gmem(grid_gradient + index, tvec<GRAD_T, N_FEATURES_PER_THREAD>(dL_dbarycentric[k] * dL_dy_local));
		}
	}

	if (dL_ddLdy.data) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
			dL_ddLdy(level * N_FEATURES_PER_LEVEL + feature + f, i) = (T)dL_ddLdy_local[f];
		}
	}
}

template <typename T, uint32_t N_POS_DIMS = 3, uint32_t N_FEATURES_PER_LEVEL = 2>
class PermutoEncodingTemplated : public MultiLevelEncoding<T> {
public:
#if TCNN_MIN_GPU_ARCH >= 62 || TCNN_MIN_GPU_ARCH == 60
	// The GPUs that we tested this on do not have an efficient 1D fp16
	// atomicAdd feature. Thus, we accumulate gradients at fp32 if we're
	// forced to use 1D atomicAdds. As soon as 2D or higher is possible,
	// we can make use the efficient atomicAdd(half2) function.
	using grad_t = std::conditional_t<N_FEATURES_PER_LEVEL == 1, float, T>;
#else
	// atomicAdd(__half2) is only supported with compute capability 60 and above.
	// Since atomicAdd(__half) is relatively slow / doesn't exist for low compute
	// capabilities, accumulate in fp32 instead.
	using grad_t = float;
#endif

	PermutoEncodingTemplated(
		uint32_t n_features, uint32_t log2_hashmap_size, float base_scale, float per_level_scale, uint32_t max_input_grad_dims, uint32_t seed = 1337
	) :
		m_n_features{n_features},
		m_rng{seed},
		m_seed{seed},
		m_log2_hashmap_size{log2_hashmap_size},
		m_max_input_grad_dims{max_input_grad_dims},
		m_per_level_scale{per_level_scale},
		m_base_scale{base_scale} {
		if (n_features == 0 || n_features % N_FEATURES_PER_LEVEL != 0) {
			throw std::runtime_error{fmt::format(
				"PermutoEncoding: n_features={} must be a nonzero multiple of N_FEATURES_PER_LEVEL={}", n_features, N_FEATURES_PER_LEVEL
			)};
		}

		m_n_levels = n_features / N_FEATURES_PER_LEVEL;
		if (m_n_levels > PERMUTO_MAX_N_LEVELS) {
			throw std::runtime_error{
				fmt::format("PermutoEncoding: m_n_levels={} must be at most PERMUTO_MAX_N_LEVELS={}", m_n_levels, PERMUTO_MAX_N_LEVELS)
			};
		}

		if (log2_hashmap_size >= std::numeric_limits<uint32_t>::digits) {
			throw std::runtime_error{fmt::format(
				"PermutoEncoding: log2_hashmap_size={} must be less than {}", log2_hashmap_size, std::numeric_limits<uint32_t>::digits
			)};
		}
		if (max_input_grad_dims > N_POS_DIMS) {
			throw std::runtime_error{"PermutoEncoding: max_input_grad_dims exceeds its input width."};
		}
		if (!std::isfinite(base_scale)) {
			throw std::runtime_error{"PermutoEncoding: base_scale must be finite."};
		}
		if (!std::isfinite(per_level_scale) || per_level_scale <= 0.0f) {
			throw std::runtime_error{"PermutoEncoding: per_level_scale must be positive and finite."};
		}

		const uint64_t params_in_level = uint64_t{1} << log2_hashmap_size;
		const uint64_t n_params = params_in_level * m_n_levels * N_FEATURES_PER_LEVEL;
		if (n_params > std::numeric_limits<uint32_t>::max()) {
			throw std::runtime_error{fmt::format(
				"PermutoEncoding: parameter count={} exceeds the supported maximum={}", n_params, std::numeric_limits<uint32_t>::max()
			)};
		}

		const float log2_per_level_scale = std::log2(per_level_scale);
		constexpr double max_abs_position = 1.0;
		constexpr double max_abs_lattice_shift = 5.0;
		const double rounded_coordinate_budget = static_cast<double>(std::numeric_limits<int>::max()) / (2.0 * (N_POS_DIMS + 1));
		const double max_abs_scale = (rounded_coordinate_budget - (N_POS_DIMS + 1)) /
			(N_POS_DIMS * (max_abs_position + max_abs_lattice_shift));

		uint64_t offset = 0;
		for (uint32_t level = 0; level < m_n_levels; ++level) {
			m_offset_table.data[level] = static_cast<uint32_t>(offset);
			offset += params_in_level;

			const float scale = base_scale * exp2f(level * log2_per_level_scale);
			if (!std::isfinite(scale) || std::abs(static_cast<double>(scale)) > max_abs_scale) {
				throw std::runtime_error{
					fmt::format("PermutoEncoding: scale at level {} is outside the normalized-input coordinate range.", level)
				};
			}
			log_debug("PermutoEncoding at level {}: scale={} params_in_level={}", level, scale, params_in_level);
		}

		m_offset_table.data[m_n_levels] = static_cast<uint32_t>(offset);
		m_offset_table.size = m_n_levels + 1;
		m_n_params = static_cast<uint32_t>(n_params);
	}

#if !defined(TCNN_NO_FWD_BWD)
	std::unique_ptr<Context> forward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		(void)prepare_input_gradients;
		auto forward = std::make_unique<Context>();

		const uint32_t num_elements = input.n();
		if (!output || num_elements == 0) {
			return forward;
		}

		SyncedMultiStream synced_streams{stream, m_n_to_pad > 0 ? 2u : 1u};

		// Take care of padding on the auxiliary stream
		if (m_n_to_pad > 0) {
			if (output->layout() == AoS) {
				parallel_for_gpu_aos(
					synced_streams.get(1),
					num_elements,
					m_n_to_pad,
					[n_features = m_n_features, out = output->pitched_ptr()] __device__(size_t elem, size_t dim) {
						out(elem)[n_features + dim] = 0;
					}
				);
			} else {
				parallel_for_gpu(
					synced_streams.get(1),
					num_elements * m_n_to_pad,
					[out = output->view(), n_features = m_n_features, num_elements] __device__(size_t i) {
						const uint32_t element = static_cast<uint32_t>(i % num_elements);
						const uint32_t padding_row = n_features + static_cast<uint32_t>(i / num_elements);
						out(padding_row, element) = (T)0;
					}
				);
			}
		}

		// Idea: each block only takes care of _one_ hash level (but may iterate over multiple input elements).
		// This way, only one level of the hashmap needs to fit into caches at a time (and it is reused for consecutive
		// elements) until it is time to process the next level.

		static constexpr uint32_t N_THREADS_HASHGRID = 512;
		const dim3 blocks_hashgrid = {div_round_up(num_elements, N_THREADS_HASHGRID), m_n_levels, 1};

		MatrixView<T> encoded_positions_soa = output->view();
		GPUMemoryArena::Allocation workspace;
		if (output->layout() == AoS) {
			workspace = allocate_workspace(synced_streams.get(0), num_elements * m_n_features * sizeof(T));
			encoded_positions_soa = {reinterpret_cast<T*>(workspace.data()), num_elements, 1u};
		}

		kernel_permuto<T, N_POS_DIMS, N_FEATURES_PER_LEVEL><<<blocks_hashgrid, N_THREADS_HASHGRID, 0, synced_streams.get(0)>>>(
			num_elements,
			m_n_features,
			m_offset_table,
			m_base_scale,
			std::log2(m_per_level_scale),
			m_rng,
			this->m_max_level,
			this->m_max_level_gpu,
			use_inference_params ? this->inference_params() : this->params(),
			input.view(),
			encoded_positions_soa
		);

		if (output->layout() == AoS) {
			// Transpose result (was stored row major due to coalescing)
			const dim3 threads_transpose = transpose_threads();
			const uint32_t blocks_transpose = div_round_up(num_elements, threads_transpose.y);
			transpose_encoded_position<T><<<blocks_transpose, threads_transpose, 0, synced_streams.get(0)>>>(
				num_elements, encoded_positions_soa.data, output->pitched_ptr()
			);
		}

		return forward;
	}

	void backward_impl(
		cudaStream_t stream,
		const Context&,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>&,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override {
		const uint32_t num_elements = input.n();
		if (num_elements == 0) {
			if (param_gradients_mode == GradientMode::Overwrite) {
				CUDA_CHECK_THROW(cudaMemsetAsync(this->gradients(), 0, n_params() * sizeof(T), stream));
			}
			return;
		}
		if (!dL_dinput && param_gradients_mode == GradientMode::Ignore) {
			return;
		}

		MatrixView<const T> dL_dy_soa = dL_doutput.view();

		GPUMemoryArena::Allocation workspace;
		if (dL_doutput.layout() == AoS) {
			workspace = allocate_workspace(stream, num_elements * m_n_features * sizeof(T));

			// Transpose dL_dy. Use the buffer previously occupied by the encoded positions
			const dim3 threads_transpose = transpose_threads();
			const uint32_t blocks_transpose = div_round_up(num_elements, threads_transpose.y);
			transpose_gradients<T>
				<<<blocks_transpose, threads_transpose, 0, stream>>>(num_elements, (T*)workspace.data(), dL_doutput.pitched_ptr());

			dL_dy_soa = {reinterpret_cast<const T*>(workspace.data()), num_elements, 1u};
		}

		if (param_gradients_mode != GradientMode::Ignore) {
			// We accumulate gradients with grad_t precision, which, for performance reasons, is not always T.
			// If not, accumulate in a temporary buffer and cast later.
			grad_t* grid_gradient = nullptr;
			GPUMemoryArena::Allocation grid_gradient_tmp;
			constexpr bool uses_temporary_gradient = !std::is_same<grad_t, T>::value;

			if (uses_temporary_gradient) {
				grid_gradient_tmp = allocate_workspace(stream, m_n_params * sizeof(grad_t));
				grid_gradient = (grad_t*)grid_gradient_tmp.data();
			} else {
				grid_gradient = (grad_t*)this->gradients();
			}

			if (param_gradients_mode == GradientMode::Overwrite || uses_temporary_gradient) {
				CUDA_CHECK_THROW(cudaMemsetAsync(grid_gradient, 0, n_params() * sizeof(grad_t), stream));
			}

			static constexpr uint32_t N_THREADS_HASHGRID = 256;
			static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);
			const dim3 blocks_hashgrid = {
				div_round_up(num_elements * N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD, N_THREADS_HASHGRID), m_n_levels, 1
			};
			kernel_permuto_backward<T, grad_t, N_POS_DIMS, N_FEATURES_PER_LEVEL, N_FEATURES_PER_THREAD>
				<<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(
					num_elements,
					m_n_features,
					m_offset_table,
					m_base_scale,
					std::log2(m_per_level_scale),
					m_rng,
					this->m_max_level,
					this->m_max_level_gpu,
					dL_dy_soa,
					input.view(),
					grid_gradient
				);

			if (uses_temporary_gradient) {
				const bool accumulate = param_gradients_mode == GradientMode::Accumulate;
				parallel_for_gpu(stream, n_params(), [grad = this->gradients(), grad_tmp = grid_gradient, accumulate] __device__(size_t i) {
					const grad_t value = grad_tmp[i] + (accumulate ? (grad_t)grad[i] : (grad_t)0);
					grad[i] = (T)value;
				});
			}
		}

		if (dL_dinput) {
			parallel_for_gpu(stream, num_elements, [grad = dL_dinput->view()] __device__(size_t i) {
				TCNN_PRAGMA_UNROLL
				for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
					grad(dim, i) = 0;
				}
			});

			static constexpr uint32_t N_THREADS_HASHGRID = 256;
			static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);
			const dim3 blocks_hashgrid = {
				div_round_up(num_elements * N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD, N_THREADS_HASHGRID), m_n_levels, 1
			};
			kernel_permuto_backward_input<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, N_FEATURES_PER_THREAD>
				<<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(
					num_elements,
					m_n_features,
					m_offset_table,
					m_base_scale,
					std::log2(m_per_level_scale),
					m_rng,
					this->m_max_level,
					this->m_max_level_gpu,
					m_max_input_grad_dims,
					use_inference_params ? this->inference_params() : this->params(),
					dL_dy_soa,
					input.view(),
					dL_dinput->view()
				);
		}
	}

	void backward_backward_input_impl(
		cudaStream_t stream,
		const Context&,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<float>& dL_ddLdinput,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_ddLdoutput = nullptr,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override {
		const uint32_t num_elements = input.n();
		if (num_elements == 0) {
			if (param_gradients_mode == GradientMode::Overwrite && n_params() > 0) {
				CUDA_CHECK_THROW(cudaMemsetAsync(this->gradients(), 0, n_params() * sizeof(T), stream));
			}
			return;
		}
		if (padded_output_width() == 0 || (!dL_ddLdoutput && !dL_dinput && param_gradients_mode == GradientMode::Ignore)) {
			return;
		}

		if (dL_ddLdoutput && m_n_to_pad > 0) {
			parallel_for_gpu(stream, num_elements * m_n_to_pad, [out = dL_ddLdoutput->view(), n_features = m_n_features, num_elements] __device__(size_t idx) {
				const uint32_t element = (uint32_t)(idx % num_elements);
				const uint32_t feature = n_features + (uint32_t)(idx / num_elements);
				out(feature, element) = (T)0.0f;
			});
		}

		MatrixView<const T> dL_dy_soa = dL_doutput.view();
		GPUMemoryArena::Allocation dL_dy_workspace;
		if (dL_doutput.layout() == AoS) {
			dL_dy_workspace = allocate_workspace(stream, num_elements * m_n_features * sizeof(T));
			const dim3 threads_transpose = transpose_threads();
			const uint32_t blocks_transpose = div_round_up(num_elements, threads_transpose.y);
			transpose_gradients<T><<<blocks_transpose, threads_transpose, 0, stream>>>(
				num_elements, (T*)dL_dy_workspace.data(), dL_doutput.pitched_ptr()
			);
			dL_dy_soa = {reinterpret_cast<const T*>(dL_dy_workspace.data()), num_elements, 1u};
		}

		grad_t* grid_gradient = nullptr;
		GPUMemoryArena::Allocation grid_gradient_tmp;
		constexpr bool uses_temporary_gradient = !std::is_same<grad_t, T>::value;
		if (param_gradients_mode != GradientMode::Ignore) {
			if (uses_temporary_gradient) {
				grid_gradient_tmp = allocate_workspace(stream, m_n_params * sizeof(grad_t));
				grid_gradient = (grad_t*)grid_gradient_tmp.data();
			} else {
				grid_gradient = (grad_t*)this->gradients();
			}

			if (param_gradients_mode == GradientMode::Overwrite || uses_temporary_gradient) {
				CUDA_CHECK_THROW(cudaMemsetAsync(grid_gradient, 0, n_params() * sizeof(grad_t), stream));
			}
		}

		if (grid_gradient || dL_ddLdoutput) {
			static constexpr uint32_t N_THREADS_HASHGRID = 256;
			static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);
			const dim3 blocks_hashgrid = {
				div_round_up(num_elements * N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD, N_THREADS_HASHGRID), m_n_levels, 1
			};
			kernel_permuto_backward_backward_input<T, grad_t, N_POS_DIMS, N_FEATURES_PER_LEVEL, N_FEATURES_PER_THREAD>
				<<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(
					num_elements,
					m_n_features,
					m_offset_table,
					m_base_scale,
					std::log2(m_per_level_scale),
					m_rng,
					this->m_max_level,
					this->m_max_level_gpu,
					m_max_input_grad_dims,
					dL_ddLdinput.view(),
					input.view(),
					dL_dy_soa,
					use_inference_params ? this->inference_params() : this->params(),
					grid_gradient,
					dL_ddLdoutput ? dL_ddLdoutput->view() : MatrixView<T>{}
				);
		}

		if (uses_temporary_gradient && grid_gradient) {
			const bool accumulate = param_gradients_mode == GradientMode::Accumulate;
			parallel_for_gpu(stream, n_params(), [grad = this->gradients(), grad_tmp = grid_gradient, accumulate] __device__(size_t i) {
				grad[i] = (T)(grad_tmp[i] + (accumulate ? (grad_t)grad[i] : (grad_t)0));
			});
		}

		if (dL_dinput) {
			parallel_for_gpu(stream, num_elements, [gradient = dL_dinput->view()] __device__(size_t i) {
				TCNN_PRAGMA_UNROLL
				for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
					gradient(dim, i) = 0.0f;
				}
			});
		}
	}

	#endif // !defined(TCNN_NO_FWD_BWD)

	uint32_t input_width() const override { return N_POS_DIMS; }

	uint32_t padded_output_width() const override { return m_n_features + m_n_to_pad; }

	uint32_t output_width() const override { return padded_output_width(); }

	uint32_t required_input_alignment() const override { return 1; }

	void set_padded_output_width(uint32_t padded_output_width) override {
		CHECK_THROW(padded_output_width >= m_n_features);
		m_n_to_pad = padded_output_width - m_n_features;
	}

	uint32_t required_output_alignment() const override { return N_FEATURES_PER_LEVEL; }

	MatrixLayout preferred_output_layout() const override { return SoA; }

	void set_params_impl(T* params, T* inference_params, T* gradients) override {}

	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
		// Initialize the hashgrid from the GPU, because the number of parameters can be quite large.
		generate_random_uniform<float>(rnd, n_params(), params_full_precision, -1e-4f * scale, 1e-4f * scale);
	}

	size_t n_params() const override { return m_n_params; }

	size_t level_n_params(uint32_t level) const override { return level_params_offset(level + 1) - level_params_offset(level); }

	size_t level_params_offset(uint32_t level) const override {
		if (level >= m_offset_table.size) {
			throw std::runtime_error{"Out of bounds params offset request."};
		}

		return m_offset_table.data[level];
	}

	const ParamsOffsetTable& params_offset_table() const override { return m_offset_table; }

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		// Even though we have parameters, they can't really be considered a "layer".
		// So we return an empty array here.
		return {};
	}

	uint32_t n_pos_dims() const override { return N_POS_DIMS; }

	uint32_t n_features_per_level() const override { return N_FEATURES_PER_LEVEL; }

	json hyperparams() const override {
		std::vector<float> scales_table(m_n_levels * N_POS_DIMS);
		for (uint32_t i = 0; i < m_n_levels; ++i) {
			const float scale = m_base_scale * exp2f(i * std::log2(m_per_level_scale));
			for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
				scales_table[i * N_POS_DIMS + dim] = scale * rsqrtf((dim + 1) * (dim + 2));
			}
		}

		std::vector<float> shifts_table(m_n_levels * N_POS_DIMS);
		for (uint32_t i = 0; i < m_n_levels; ++i) {
			pcg32 rng_tmp = m_rng;
			rng_tmp.advance(i * N_POS_DIMS); // Different level should draw differently
			for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
				shifts_table[i * N_POS_DIMS + dim] = (rng_tmp.next_float() - 0.5f) * 10.0f; // [-5, 5)
			}
		}
		return {
			{"otype",                "Permuto"            },
			{"n_levels",             m_n_levels           },
			{"n_features_per_level", N_FEATURES_PER_LEVEL },
			{"base_scale",           m_base_scale         },
			{"per_level_scale",      m_per_level_scale    },
			{"log2_hashmap_size",    m_log2_hashmap_size  },
			{"max_input_grad_dims",  m_max_input_grad_dims},
			{"seed",                 m_seed               },
			{"scales_table",         scales_table         },
			{"shifts_table",         shifts_table         }
		};
	}

private:
	dim3 transpose_threads() const {
		const uint32_t width = m_n_levels * N_FEATURES_PER_LEVEL;
		return {width, std::min(8u, 1024u / width), 1};
	}

	ParamsOffsetTable m_offset_table;

	uint32_t m_n_features;
	uint32_t m_n_levels;
	uint32_t m_n_params;

	pcg32 m_rng;
	uint32_t m_seed;
	uint32_t m_log2_hashmap_size;
	uint32_t m_max_input_grad_dims;

	uint32_t m_n_to_pad = 0;

	float m_per_level_scale;
	float m_base_scale;
};

struct PermutoFactoryConfig {
	uint32_t n_features;
	uint32_t log2_hashmap_size;
	float base_scale;
	float per_level_scale;
	uint32_t max_input_grad_dims;
	uint32_t seed;
};

template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL>
std::unique_ptr<MultiLevelEncoding<T>> make_permuto_encoding(const PermutoFactoryConfig& config) {
	return std::make_unique<PermutoEncodingTemplated<T, N_POS_DIMS, N_FEATURES_PER_LEVEL>>(
		config.n_features,
		config.log2_hashmap_size,
		config.base_scale,
		config.per_level_scale,
		config.max_input_grad_dims,
		config.seed
	);
}

template <typename T, uint32_t N_FEATURES_PER_LEVEL>
std::unique_ptr<MultiLevelEncoding<T>> dispatch_permuto_dimension(uint32_t n_dims_to_encode, const PermutoFactoryConfig& config) {
	switch (n_dims_to_encode) {
		case 1: return make_permuto_encoding<T, 1, N_FEATURES_PER_LEVEL>(config);
		case 2: return make_permuto_encoding<T, 2, N_FEATURES_PER_LEVEL>(config);
		case 3: return make_permuto_encoding<T, 3, N_FEATURES_PER_LEVEL>(config);
		case 4: return make_permuto_encoding<T, 4, N_FEATURES_PER_LEVEL>(config);
		case 5: return make_permuto_encoding<T, 5, N_FEATURES_PER_LEVEL>(config);
		case 6: return make_permuto_encoding<T, 6, N_FEATURES_PER_LEVEL>(config);
		case 7: return make_permuto_encoding<T, 7, N_FEATURES_PER_LEVEL>(config);
		case 8: return make_permuto_encoding<T, 8, N_FEATURES_PER_LEVEL>(config);
		case 9: return make_permuto_encoding<T, 9, N_FEATURES_PER_LEVEL>(config);
		case 10: return make_permuto_encoding<T, 10, N_FEATURES_PER_LEVEL>(config);
		case 12: return make_permuto_encoding<T, 12, N_FEATURES_PER_LEVEL>(config);
		case 16: return make_permuto_encoding<T, 16, N_FEATURES_PER_LEVEL>(config);
		case 24: return make_permuto_encoding<T, 24, N_FEATURES_PER_LEVEL>(config);
		default: throw std::runtime_error{"PermutoEncoding: input dimensions must be one of 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, or 24."};
	}
}

template <typename T>
std::unique_ptr<MultiLevelEncoding<T>> dispatch_permuto_features(
	uint32_t n_dims_to_encode, uint32_t n_features_per_level, const PermutoFactoryConfig& config
) {
	switch (n_features_per_level) {
		case 1: return dispatch_permuto_dimension<T, 1>(n_dims_to_encode, config);
		case 2: return dispatch_permuto_dimension<T, 2>(n_dims_to_encode, config);
		case 4: return dispatch_permuto_dimension<T, 4>(n_dims_to_encode, config);
		case 8: return dispatch_permuto_dimension<T, 8>(n_dims_to_encode, config);
		default: throw std::runtime_error{"PermutoEncoding: n_features_per_level must be 1, 2, 4, or 8."};
	}
}

template <typename T> MultiLevelEncoding<T>* create_permuto_encoding(uint32_t n_dims_to_encode, const json& encoding) {
	if (!encoding.is_object()) {
		throw std::runtime_error{"PermutoEncoding configuration must be an object."};
	}

	const auto read_nonnegative_integer = [&encoding](const char* key, uint64_t default_value) -> uint64_t {
		const auto value = encoding.find(key);
		if (value == encoding.end()) {
			return default_value;
		}
		if (!value->is_number_integer()) {
			throw std::runtime_error{fmt::format("PermutoEncoding: {} must be an integer.", key)};
		}
		if (value->is_number_unsigned()) {
			return value->get<uint64_t>();
		}
		const int64_t signed_value = value->get<int64_t>();
		if (signed_value < 0) {
			throw std::runtime_error{fmt::format("PermutoEncoding: {} must be nonnegative.", key)};
		}
		return static_cast<uint64_t>(signed_value);
	};

	const uint64_t n_features_per_level = read_nonnegative_integer("n_features_per_level", 2);
	if (n_features_per_level != 1 && n_features_per_level != 2 && n_features_per_level != 4 && n_features_per_level != 8) {
		throw std::runtime_error{"PermutoEncoding: n_features_per_level must be 1, 2, 4, or 8."};
	}

	const uint64_t log2_hashmap_size = read_nonnegative_integer("log2_hashmap_size", 19);
	if (log2_hashmap_size >= std::numeric_limits<uint32_t>::digits) {
		throw std::runtime_error{"PermutoEncoding log2_hashmap_size is out of range."};
	}

	const bool has_n_features = encoding.contains("n_features");
	const bool has_n_grid_features = encoding.contains("n_grid_features");
	if (has_n_features && has_n_grid_features) {
		throw std::runtime_error{"PermutoEncoding: n_features and n_grid_features are mutually exclusive."};
	}
	if ((has_n_features || has_n_grid_features) && encoding.contains("n_levels")) {
		throw std::runtime_error{"PermutoEncoding: total feature aliases and n_levels are mutually exclusive."};
	}

	uint64_t n_levels = 16;
	if (has_n_features || has_n_grid_features) {
		const char* key = has_n_features ? "n_features" : "n_grid_features";
		const uint64_t total_features = read_nonnegative_integer(key, 0);
		if (total_features == 0) {
			throw std::runtime_error{fmt::format("PermutoEncoding: {} must be positive.", key)};
		}
		if (total_features % n_features_per_level != 0) {
			throw std::runtime_error{fmt::format("PermutoEncoding: {} must be divisible by n_features_per_level.", key)};
		}
		n_levels = total_features / n_features_per_level;
	} else {
		n_levels = read_nonnegative_integer("n_levels", 16);
	}
	if (n_levels == 0 || n_levels > PERMUTO_MAX_N_LEVELS) {
		throw std::runtime_error{"PermutoEncoding n_levels is out of range."};
	}

	const uint64_t max_input_grad_dims = read_nonnegative_integer("max_input_grad_dims", n_dims_to_encode);
	if (max_input_grad_dims > n_dims_to_encode) {
		throw std::runtime_error{"PermutoEncoding max_input_grad_dims exceeds its input width."};
	}

	const uint64_t seed = read_nonnegative_integer("seed", 1337);
	if (seed > std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error{"PermutoEncoding seed is out of range."};
	}

	const auto base_scale_value = encoding.find("base_scale");
	if (base_scale_value != encoding.end() && !base_scale_value->is_number()) {
		throw std::runtime_error{"PermutoEncoding: base_scale must be numeric."};
	}
	const double base_scale_wide = base_scale_value == encoding.end() ? 16.0 : base_scale_value->get<double>();
	if (!std::isfinite(base_scale_wide) || std::abs(base_scale_wide) > std::numeric_limits<float>::max()) {
		throw std::runtime_error{"PermutoEncoding: base_scale must be finite and representable as float."};
	}
	const float base_scale = static_cast<float>(base_scale_wide);

	const auto per_level_scale_value = encoding.find("per_level_scale");
	if (per_level_scale_value != encoding.end() && !per_level_scale_value->is_number()) {
		throw std::runtime_error{"PermutoEncoding: per_level_scale must be numeric."};
	}
	const double per_level_scale_wide = per_level_scale_value == encoding.end() ? 2.0 : per_level_scale_value->get<double>();
	if (!std::isfinite(per_level_scale_wide) || per_level_scale_wide <= 0.0 || per_level_scale_wide > std::numeric_limits<float>::max()) {
		throw std::runtime_error{"PermutoEncoding: per_level_scale must be positive, finite, and representable as float."};
	}
	const float per_level_scale = static_cast<float>(per_level_scale_wide);

	const auto interpolation_value = encoding.find("interpolation");
	if (interpolation_value != encoding.end() && !interpolation_value->is_string()) {
		throw std::runtime_error{"PermutoEncoding: interpolation must be a string."};
	}
	const std::string interpolation = encoding.value("interpolation", "Linear");
	if (!equals_case_insensitive(interpolation, "Linear")) {
		throw std::runtime_error{"PermutoEncoding requires linear interpolation."};
	}

	const PermutoFactoryConfig config{
		static_cast<uint32_t>(n_features_per_level * n_levels),
		static_cast<uint32_t>(log2_hashmap_size),
		base_scale,
		per_level_scale,
		static_cast<uint32_t>(max_input_grad_dims),
		static_cast<uint32_t>(seed),
	};
	auto result = dispatch_permuto_features<T>(n_dims_to_encode, static_cast<uint32_t>(n_features_per_level), config);

	const json derived = result->hyperparams();
	for (const char* key : {"scales_table", "shifts_table"}) {
		if (!encoding.contains(key)) {
			continue;
		}

		const json& supplied = encoding.at(key);
		const auto expected = derived.at(key).get<std::vector<float>>();
		if (!supplied.is_array() || supplied.size() != expected.size()) {
			throw std::runtime_error{fmt::format("PermutoEncoding: {} must contain exactly {} values.", key, expected.size())};
		}

		for (size_t i = 0; i < expected.size(); ++i) {
			if (!supplied.at(i).is_number()) {
				throw std::runtime_error{fmt::format("PermutoEncoding: {}[{}] must be numeric.", key, i)};
			}
			const double actual = supplied.at(i).get<double>();
			const double reference = static_cast<double>(expected[i]);
			const double tolerance = 8.0 * std::numeric_limits<float>::epsilon() * std::max(1.0, std::abs(reference));
			if (!std::isfinite(actual) || std::abs(actual - reference) > tolerance) {
				throw std::runtime_error{fmt::format("PermutoEncoding: {}[{}] does not match the derived lattice.", key, i)};
			}
		}
	}

	return result.release();
}

} // namespace tcnn
