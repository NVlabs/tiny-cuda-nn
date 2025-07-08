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

/** @file   common_host.cu
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Common utilities that are needed by pretty much every component of this framework.
 */

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>

#include <cuda.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <unordered_map>

namespace tcnn {

static_assert(
	__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2),
	"tiny-cuda-nn requires at least CUDA 10.2"
);

std::function<void(LogSeverity, const std::string&)> g_log_callback = [](LogSeverity severity, const std::string& msg) {
	switch (severity) {
		case LogSeverity::Warning: std::cerr << fmt::format("tiny-cuda-nn warning: {}\n", msg); break;
		case LogSeverity::Error: std::cerr << fmt::format("tiny-cuda-nn error: {}\n", msg); break;
		default: break;
	}

	if (verbose()) {
		switch (severity) {
			case LogSeverity::Debug: std::cerr << fmt::format("tiny-cuda-nn debug: {}\n", msg); break;
			case LogSeverity::Info: std::cerr << fmt::format("tiny-cuda-nn info: {}\n", msg); break;
			case LogSeverity::Success: std::cerr << fmt::format("tiny-cuda-nn success: {}\n", msg); break;
			default: break;
		}
	}
};

const std::function<void(LogSeverity, const std::string&)>& log_callback() { return g_log_callback; }
void set_log_callback(const std::function<void(LogSeverity, const std::string&)>& cb) { g_log_callback = cb; }

bool g_verbose = false;
bool verbose() { return g_verbose; }
void set_verbose(bool verbose) { g_verbose = verbose; }

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

GridType string_to_grid_type(const std::string& grid_type) {
	if (equals_case_insensitive(grid_type, "Hash")) {
		return GridType::Hash;
	} else if (equals_case_insensitive(grid_type, "Dense")) {
		return GridType::Dense;
	} else if (equals_case_insensitive(grid_type, "Tiled") || equals_case_insensitive(grid_type, "Tile")) {
		return GridType::Tiled;
	}

	throw std::runtime_error{fmt::format("Invalid grid type: {}", grid_type)};
}

std::string to_string(GridType grid_type) {
	switch (grid_type) {
		case GridType::Hash: return "Hash";
		case GridType::Dense: return "Dense";
		case GridType::Tiled: return "Tiled";
		default: throw std::runtime_error{"Invalid grid type."};
	}
}

HashType string_to_hash_type(const std::string& hash_type) {
	if (equals_case_insensitive(hash_type, "Prime")) {
		return HashType::Prime;
	} else if (equals_case_insensitive(hash_type, "CoherentPrime")) {
		return HashType::CoherentPrime;
	} else if (equals_case_insensitive(hash_type, "ReversedPrime")) {
		return HashType::ReversedPrime;
	} else if (equals_case_insensitive(hash_type, "Rng")) {
		return HashType::Rng;
	} else if (equals_case_insensitive(hash_type, "BaseConvert")) {
		return HashType::BaseConvert;
	}

	throw std::runtime_error{fmt::format("Invalid hash type: {}", hash_type)};
}

std::string to_string(HashType hash_type) {
	switch (hash_type) {
		case HashType::Prime: return "Prime";
		case HashType::CoherentPrime: return "CoherentPrime";
		case HashType::ReversedPrime: return "ReversedPrime";
		case HashType::Rng: return "Rng";
		case HashType::BaseConvert: return "BaseConvert";
		default: throw std::runtime_error{"Invalid hash type."};
	}
}

InterpolationType string_to_interpolation_type(const std::string& interpolation_type) {
	if (equals_case_insensitive(interpolation_type, "Nearest")) {
		return InterpolationType::Nearest;
	} else if (equals_case_insensitive(interpolation_type, "Linear")) {
		return InterpolationType::Linear;
	} else if (equals_case_insensitive(interpolation_type, "Smoothstep")) {
		return InterpolationType::Smoothstep;
	}

	throw std::runtime_error{fmt::format("Invalid interpolation type: {}", interpolation_type)};
}

std::string to_string(InterpolationType interpolation_type) {
	switch (interpolation_type) {
		case InterpolationType::Nearest: return "Nearest";
		case InterpolationType::Linear: return "Linear";
		case InterpolationType::Smoothstep: return "Smoothstep";
		default: throw std::runtime_error{"Invalid interpolation type."};
	}
}

ReductionType string_to_reduction_type(const std::string& reduction_type) {
	if (equals_case_insensitive(reduction_type, "Concatenation")) {
		return ReductionType::Concatenation;
	} else if (equals_case_insensitive(reduction_type, "Sum")) {
		return ReductionType::Sum;
	} else if (equals_case_insensitive(reduction_type, "Product")) {
		return ReductionType::Product;
	}

	throw std::runtime_error{fmt::format("Invalid reduction type: {}", reduction_type)};
}

std::string to_string(ReductionType reduction_type) {
	switch (reduction_type) {
		case ReductionType::Concatenation: return "Concatenation";
		case ReductionType::Sum: return "Sum";
		case ReductionType::Product: return "Product";
		default: throw std::runtime_error{"Invalid reduction type."};
	}
}

int cuda_runtime_version() {
	int version;
	CUDA_CHECK_THROW(cudaRuntimeGetVersion(&version));
	return version;
}

int cuda_device() {
	int device;
	CUDA_CHECK_THROW(cudaGetDevice(&device));
	return device;
}

void set_cuda_device(int device) {
	CUDA_CHECK_THROW(cudaSetDevice(device));
}

int cuda_device_count() {
	int device_count;
	CUDA_CHECK_THROW(cudaGetDeviceCount(&device_count));
	return device_count;
}

bool cuda_supports_virtual_memory(int device) {
	int supports_vmm;
	CU_CHECK_THROW(cuDeviceGetAttribute(&supports_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device));
	return supports_vmm != 0;
}

std::unordered_map<int, cudaDeviceProp>& cuda_device_properties() {
	static auto* cuda_device_props = new std::unordered_map<int, cudaDeviceProp>{};
	return *cuda_device_props;
}

const cudaDeviceProp& cuda_get_device_properties(int device) {
	if (cuda_device_properties().count(device) == 0) {
		auto& props = cuda_device_properties()[device];
		CUDA_CHECK_THROW(cudaGetDeviceProperties(&props, device));
	}

	return cuda_device_properties().at(device);
}

std::string cuda_device_name(int device) {
	return cuda_get_device_properties(device).name;
}

uint32_t cuda_compute_capability(int device) {
	const auto& props = cuda_get_device_properties(device);
	return props.major * 10 + props.minor;
}

uint32_t cuda_max_supported_compute_capability() {
	int cuda_version = cuda_runtime_version();
	if (cuda_version < 11000) {
		return 75;
	} else if (cuda_version < 11010) {
		return 80;
	} else if (cuda_version < 11080) {
		return 86;
	} else {
		return 90;
	}
}

uint32_t cuda_supported_compute_capability(int device) {
	return std::min(cuda_compute_capability(device), cuda_max_supported_compute_capability());
}

size_t cuda_max_shmem(int device) {
	return cuda_get_device_properties(device).sharedMemPerBlockOptin;
}

uint32_t cuda_max_registers(int device) {
	return (uint32_t)cuda_get_device_properties(device).regsPerBlock;
}

size_t cuda_memory_granularity(int device) {
	size_t granularity;
	CUmemAllocationProp prop = {};
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = 0;
	CUresult granularity_result = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
	if (granularity_result == CUDA_ERROR_NOT_SUPPORTED) {
		return 1;
	}
	CU_CHECK_THROW(granularity_result);
	return granularity;
}

MemoryInfo cuda_memory_info() {
	MemoryInfo info;
	CUDA_CHECK_THROW(cudaMemGetInfo(&info.free, &info.total));
	info.used = info.total - info.free;
	return info;
}

std::string generate_device_code_preamble() {
	return dfmt(0, R"(
		#include <tiny-cuda-nn/common_device.h>
		#include <tiny-cuda-nn/mma.h>

		using namespace tcnn;
	)");
}

std::string to_snake_case(const std::string& str) {
	std::stringstream result;
	result << (char)std::tolower(str[0]);
	for (uint32_t i = 1; i < str.length(); ++i) {
		if (std::isupper(str[i])) {
			result << "_" << (char)std::tolower(str[i]);
		} else {
			result << str[i];
		}
	}
	return result.str();
}

std::vector<std::string> split(const std::string& text, const std::string& delim) {
	std::vector<std::string> result;
	size_t begin = 0;
	while (true) {
		size_t end = text.find_first_of(delim, begin);
		if (end == std::string::npos) {
			result.emplace_back(text.substr(begin));
			return result;
		} else {
			result.emplace_back(text.substr(begin, end - begin));
			begin = end + 1;
		}
	}

	return result;
}

std::string to_lower(std::string str) {
	std::transform(std::begin(str), std::end(str), std::begin(str), [](unsigned char c) { return (char)std::tolower(c); });
	return str;
}

std::string to_upper(std::string str) {
	std::transform(std::begin(str), std::end(str), std::begin(str), [](unsigned char c) { return (char)std::toupper(c); });
	return str;
}

template <> std::string type_to_string<bool>() { return "bool"; }
template <> std::string type_to_string<int>() { return "int"; }
template <> std::string type_to_string<char>() { return "char"; }
template <> std::string type_to_string<uint8_t>() { return "uint8_t"; }
template <> std::string type_to_string<uint16_t>() { return "uint16_t"; }
template <> std::string type_to_string<uint32_t>() { return "uint32_t"; }
template <> std::string type_to_string<double>() { return "double"; }
template <> std::string type_to_string<float>() { return "float"; }
template <> std::string type_to_string<__half>() { return "__half"; }

}
