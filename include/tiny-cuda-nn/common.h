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

/** @file   common.h
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Common utilities that are needed by pretty much every component of this framework.
 */


#pragma once

// A macro is used such that external tools won't end up indenting entire files,
// resulting in wasted horizontal space.
#define TCNN_NAMESPACE_BEGIN namespace tcnn {
#define TCNN_NAMESPACE_END }

#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>


TCNN_NAMESPACE_BEGIN

// using network_precision_t = float;
using network_precision_t = __half;

// #define TCNN_VERBOSE_MEMORY_ALLOCS

//////////////////
// Misc helpers //
//////////////////

std::string to_lower(std::string str);
std::string to_upper(std::string str);
inline bool equals_case_insensitive(const std::string& str1, const std::string& str2) {
	return to_lower(str1) == to_lower(str2);
}

//////////////////////////////////////
// CUDA ERROR HANDLING (EXCEPTIONS) //
//////////////////////////////////////

/// Checks the result of a cuXXXXXX call and throws an error on failure
#define CU_CHECK_THROW(x)                                                                          \
	do {                                                                                           \
		CUresult result = x;                                                                       \
		if (result != CUDA_SUCCESS) {                                                              \
			const char *msg;                                                                       \
			cuGetErrorName(result, &msg);                                                          \
			throw std::runtime_error(std::string("CUDA Error: " #x " failed with error ") + msg);  \
		}                                                                                          \
	} while(0)

/// Checks the result of a cudaXXXXXX call and throws an error on failure
#define CUDA_CHECK_THROW(x)                                                                                               \
	do {                                                                                                                  \
		cudaError_t result = x;                                                                                           \
		if (result != cudaSuccess)                                                                                        \
			throw std::runtime_error(std::string("CUDA Error: " #x " failed with error ") + cudaGetErrorString(result));  \
	} while(0)

/// Checks the result of a cudaXXXXXX call and prints an error on failure
#define CUDA_CHECK_PRINT(x)                                                                                   \
	do {                                                                                                      \
		cudaError_t result = x;                                                                               \
		if (result != cudaSuccess)                                                                            \
			std::cout << "CUDA Error: " #x " failed with error " << cudaGetErrorString(result) << std::endl;  \
	} while(0)


#define CURAND_CHECK_THROW(x)                                                      \
	do {                                                                           \
		curandStatus_t result = x;                                                 \
		if (result != CURAND_STATUS_SUCCESS)                                       \
			throw std::runtime_error(std::string("CURAND Error: " #x " failed"));  \
	} while(0)


#define CUBLAS_CHECK_THROW(x)                                                                                           \
	do {                                                                                                                \
		cublasStatus_t result = x;                                                                                      \
		if (result != CUBLAS_STATUS_SUCCESS)                                                                            \
			throw std::runtime_error(std::string("CUBLAS Error: " #x " failed with error ") + cublasGetError(result));  \
	} while(0)

#define CUSOLVER_CHECK_THROW(x)                                                                                             \
	do {                                                                                                                    \
		cusolverStatus_t result = x;                                                                                        \
		if (result != CUBLAS_STATUS_SUCCESS)                                                                                \
			throw std::runtime_error(std::string("CUSOLVER Error: " #x " failed with error ") + cusolverGetError(result));  \
	} while(0)


inline std::string cusolverGetError(cusolverStatus_t error) {
	switch (error) {
		case CUSOLVER_STATUS_SUCCESS:
			return "CUSOLVER_SUCCESS";

		case CUSOLVER_STATUS_NOT_INITIALIZED:
			return "CUSOLVER_STATUS_NOT_INITIALIZED";

		case CUSOLVER_STATUS_ALLOC_FAILED:
			return "CUSOLVER_STATUS_ALLOC_FAILED";

		case CUSOLVER_STATUS_INVALID_VALUE:
			return "CUSOLVER_STATUS_INVALID_VALUE";

		case CUSOLVER_STATUS_ARCH_MISMATCH:
			return "CUSOLVER_STATUS_ARCH_MISMATCH";

		case CUSOLVER_STATUS_EXECUTION_FAILED:
			return "CUSOLVER_STATUS_EXECUTION_FAILED";

		case CUSOLVER_STATUS_INTERNAL_ERROR:
			return "CUSOLVER_STATUS_INTERNAL_ERROR";

		case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
	}

	return "<unknown>";
}

inline std::string cublasGetError(cublasStatus_t error) {
	switch (error) {
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";

		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";

		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";

		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";

		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";

		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";

		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";

		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";

		case CUBLAS_STATUS_NOT_SUPPORTED:
			return "CUBLAS_STATUS_NOT_SUPPORTED";
	}

	return "<unknown>";
}


////////////////////
// Kernel helpers //
////////////////////

#ifdef __NVCC__
#define TCNN_HOST_DEVICE __host__ __device__
#else
#define TCNN_HOST_DEVICE
#endif

template <typename T>
TCNN_HOST_DEVICE T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename T>
TCNN_HOST_DEVICE T next_multiple(T val, T divisor) {
	return div_round_up(val, divisor) * divisor;
}

constexpr uint32_t n_threads_linear = 128;

template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements) {
	return div_round_up((uint32_t)n_elements, n_threads_linear);
}

#ifdef __NVCC__
template <typename K, typename T, typename ... Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args) {
	if (n_elements <= 0) {
		return;
	}
	kernel<<<n_blocks_linear(n_elements), n_threads_linear, shmem_size, stream>>>((uint32_t)n_elements, args...);
}
#endif

inline std::string bytes_to_string(size_t bytes) {
	std::array<std::string, 7> suffixes = {{ "B", "KB", "MB", "GB", "TB", "PB", "EB" }};

	double count = (double)bytes;
	uint32_t i = 0;
	for (; i < suffixes.size() && count >= 1024; ++i) {
		count /= 1024;
	}

	std::ostringstream oss;
	oss.precision(3);
	oss << count << " " << suffixes[i];
	return oss.str();
}

template <uint32_t N_FLOATS>
using vector_fullp_t = std::array<float, N_FLOATS>;

template <uint32_t N_HALFS>
using vector_halfp_t = std::array<__half, N_HALFS>;

template <typename T, uint32_t N_ELEMS>
using vector_t = std::conditional_t<std::is_same_v<T, float>, vector_fullp_t<N_ELEMS>, vector_halfp_t<N_ELEMS>>;

TCNN_NAMESPACE_END
