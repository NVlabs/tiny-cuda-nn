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

/** @file   gpu_memory_json.h
 *  @author Nikolaus Binder and Thomas Müller, NVIDIA
 *  @brief  binding between GPUMemory and JSON librariy
 */

#pragma once

#include <json/json.hpp>

TCNN_NAMESPACE_BEGIN

inline nlohmann::json::binary_t gpu_memory_to_json_binary(const void* gpu_data, size_t n_bytes) {
	nlohmann::json::binary_t data_cpu;
	data_cpu.resize(n_bytes);
	CUDA_CHECK_THROW(cudaMemcpy(data_cpu.data(), gpu_data, n_bytes, cudaMemcpyDeviceToHost));
	return data_cpu;
}

inline void json_binary_to_gpu_memory(const nlohmann::json::binary_t& cpu_data, void* gpu_data, size_t n_bytes) {
	CUDA_CHECK_THROW(cudaMemcpy(gpu_data, cpu_data.data(), n_bytes, cudaMemcpyHostToDevice));
}

template <typename T>
inline void to_json(nlohmann::json& j, const GPUMemory<T>& gpu_data) {
	j = gpu_memory_to_json_binary(gpu_data.data(), gpu_data.get_bytes());
}

template <typename T>
inline void from_json(const nlohmann::json& j, GPUMemory<T>& gpu_data) {
	if (j.is_binary()) {
		const nlohmann::json::binary_t& cpu_data = j.get_binary();
		gpu_data.resize(cpu_data.size()/sizeof(T));
		json_binary_to_gpu_memory(cpu_data, gpu_data.data(), gpu_data.get_bytes());
	} else if (j.is_object()) {
		// https://json.nlohmann.me/features/binary_values/#json
		json::array_t arr = j["bytes"];
		nlohmann::json::binary_t cpu_data;
		cpu_data.resize(arr.size());
		for(size_t i = 0; i < arr.size(); ++i) {
			cpu_data[i] = (uint8_t)arr[i];
		}
		gpu_data.resize(cpu_data.size()/sizeof(T));
		json_binary_to_gpu_memory(cpu_data, gpu_data.data(), gpu_data.get_bytes());
	} else {
		throw std::runtime_error("Invalid json type: must be either binary or object");
	}
}

TCNN_NAMESPACE_END
