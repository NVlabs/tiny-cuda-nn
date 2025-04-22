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

/** @file   multi_level_interface.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Base class / abstract interface for controlling and querying
            general aspects of all multi-level encodings.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>

#include <cstdint>

namespace tcnn {

template <typename T>
__global__ void extract_position(
	const uint32_t num_elements,
	PitchedPtr<const float> data_in,
	T* __restrict__ output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint32_t dim_idx = threadIdx.y;

	output[i + dim_idx * num_elements] = (T)data_in(i)[dim_idx];
}

template <typename T>
__global__ void transpose_encoded_position(
	const uint32_t n_elements,
	const T* __restrict__ encoded_positions,
	PitchedPtr<T> output
) {
	const uint32_t i = threadIdx.y + blockIdx.x * blockDim.y;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i;
	const uint32_t dim_idx = threadIdx.x;

	output(elem_idx)[dim_idx] = encoded_positions[elem_idx + n_elements * dim_idx];
}

template <typename T>
__global__ void transpose_gradients(
	const uint32_t n_elements,
	T* __restrict__ transposed_dL_dy,
	PitchedPtr<const T> dL_dy
) {
	const uint32_t i = threadIdx.y + blockIdx.x * blockDim.y;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i;
	const uint32_t dim_idx = threadIdx.x;

	transposed_dL_dy[elem_idx + n_elements * dim_idx] = dL_dy(elem_idx)[dim_idx];
}

static constexpr uint32_t MAX_N_LEVELS = 128;
struct ParamsOffsetTable {
	uint32_t data[MAX_N_LEVELS+1] = {};
	uint32_t size = 0;
};

template <typename T>
class MultiLevelEncoding : public Encoding<T> {
public:
	virtual uint32_t n_pos_dims() const = 0;
	virtual uint32_t n_features_per_level() const = 0;

	virtual size_t level_n_params(uint32_t level) const = 0;
	virtual size_t level_params_offset(uint32_t level) const = 0;

	virtual const ParamsOffsetTable& params_offset_table() const = 0;

	float max_level() const {
		return m_max_level;
	}

	void set_max_level(float value) {
		m_max_level = value;
	}

	float* max_level_gpu() const {
		return m_max_level_gpu;
	}

	void set_max_level_gpu(float* value) {
		m_max_level_gpu = value;
	}

protected:
	// Disables lookups of finer levels than this.
	// The default value of 1000 effectively disables the feature
	float m_max_level = 1000.f;

	// If this pointer is non-null, it is expected to point to per-element m_max_level
	float* m_max_level_gpu = nullptr;
};

}
