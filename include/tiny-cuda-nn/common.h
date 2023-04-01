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

/** @file   common.h
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Common utilities that are needed by pretty much every component of this framework.
 */

#pragma once

// A macro is used such that external tools won't end up indenting entire files,
// resulting in wasted horizontal space.
#define TCNN_NAMESPACE_BEGIN namespace tcnn {
#define TCNN_NAMESPACE_END }


#include <tiny-cuda-nn/cpp_api.h>

#include <fmt/core.h>

#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_fp16.h>

TCNN_NAMESPACE_BEGIN

static constexpr uint32_t MIN_GPU_ARCH = TCNN_MIN_GPU_ARCH;

#define TCNN_HALF_PRECISION (!(TCNN_MIN_GPU_ARCH == 61 || TCNN_MIN_GPU_ARCH <= 52))

// TCNN has the following behavior depending on GPU arch.
// Refer to the first row of the table at the following URL for information about
// when to pick fp16 versus fp32 precision for maximum performance.
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions__throughput-native-arithmetic-instructions
//
//  GPU Arch | FullyFusedMLP supported | CUTLASS SmArch supported |                 Precision
// ----------|-------------------------|--------------------------|--------------------------
//     80-90 |                     yes |                       80 |                    __half
//        75 |                     yes |                       75 |                    __half
//        70 |                      no |                       70 |                    __half
// 53-60, 62 |                      no |                       70 |  __half (no tensor cores)
//  <=52, 61 |                      no |                       70 |   float (no tensor cores)

#if TCNN_HALF_PRECISION
using network_precision_t = __half;
#else
using network_precision_t = float;
#endif

// Optionally: set the precision to `float` to disable tensor cores and debug potential
//             problems with mixed-precision training.
// using network_precision_t = float;

// #define TCNN_VERBOSE_MEMORY_ALLOCS

enum class Activation {
	ReLU,
	LeakyReLU,
	Exponential,
	Sine,
	Sigmoid,
	Squareplus,
	Softplus,
	Tanh,
	None,
};

//////////////////
// Misc helpers //
//////////////////

int cuda_device();
void set_cuda_device(int device);
int cuda_device_count();

bool cuda_supports_virtual_memory(int device);
inline bool cuda_supports_virtual_memory() {
	return cuda_supports_virtual_memory(cuda_device());
}

std::string cuda_device_name(int device);
inline std::string cuda_device_name() {
	return cuda_device_name(cuda_device());
}

uint32_t cuda_compute_capability(int device);
inline uint32_t cuda_compute_capability() {
	return cuda_compute_capability(cuda_device());
}

size_t cuda_memory_granularity(int device);
inline size_t cuda_memory_granularity() {
	return cuda_memory_granularity(cuda_device());
}

struct MemoryInfo {
	size_t total;
	size_t free;
	size_t used;
};

MemoryInfo cuda_memory_info();

std::string to_lower(std::string str);
std::string to_upper(std::string str);
inline bool equals_case_insensitive(const std::string& str1, const std::string& str2) {
	return to_lower(str1) == to_lower(str2);
}

template <typename T>
std::string type_to_string();

inline bool is_pot(uint32_t num, uint32_t* log2 = nullptr) {
	if (log2) *log2 = 0;
	if (num > 0) {
		while (num % 2 == 0) {
			num /= 2;
			if (log2) ++*log2;
		}
		if (num == 1) {
			return true;
		}
	}

	return false;
}

inline uint32_t powi(uint32_t base, uint32_t exponent) {
	uint32_t result = 1;
	for (uint32_t i = 0; i < exponent; ++i) {
		result *= base;
	}

	return result;
}

class ScopeGuard {
public:
	ScopeGuard() = default;
	ScopeGuard(const std::function<void()>& callback) : m_callback{callback} {}
	ScopeGuard(std::function<void()>&& callback) : m_callback{std::move(callback)} {}
	ScopeGuard& operator=(const ScopeGuard& other) = delete;
	ScopeGuard(const ScopeGuard& other) = delete;
	ScopeGuard& operator=(ScopeGuard&& other) { std::swap(m_callback, other.m_callback); return *this; }
	ScopeGuard(ScopeGuard&& other) { *this = std::move(other); }
	~ScopeGuard() { if (m_callback) { m_callback(); } }

	void disarm() {
		m_callback = {};
	}
private:
	std::function<void()> m_callback;
};

//////////////////////////////////////
// CUDA ERROR HANDLING (EXCEPTIONS) //
//////////////////////////////////////

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)

#define CHECK_THROW(x) \
	do { if (!(x)) throw std::runtime_error(std::string(FILE_LINE " check failed " #x)); } while(0)

/// Checks the result of a cuXXXXXX call and throws an error on failure
#define CU_CHECK_THROW(x)                                                                          \
	do {                                                                                           \
		CUresult result = x;                                                                       \
		if (result != CUDA_SUCCESS) {                                                              \
			const char *msg;                                                                       \
			cuGetErrorName(result, &msg);                                                          \
			throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + msg);   \
		}                                                                                          \
	} while(0)

/// Checks the result of a cuXXXXXX call and prints an error on failure
#define CU_CHECK_PRINT(x)                                                                          \
	do {                                                                                           \
		CUresult result = x;                                                                       \
		if (result != CUDA_SUCCESS) {                                                              \
			const char *msg;                                                                       \
			cuGetErrorName(result, &msg);                                                          \
			std::cout << FILE_LINE " " #x " failed with error " << msg << std::endl;               \
		}                                                                                          \
	} while(0)

/// Checks the result of a cudaXXXXXX call and throws an error on failure
#define CUDA_CHECK_THROW(x)                                                                                               \
	do {                                                                                                                  \
		cudaError_t result = x;                                                                                           \
		if (result != cudaSuccess)                                                                                        \
			throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + cudaGetErrorString(result));  \
	} while(0)

/// Checks the result of a cudaXXXXXX call and prints an error on failure
#define CUDA_CHECK_PRINT(x)                                                                                   \
	do {                                                                                                      \
		cudaError_t result = x;                                                                               \
		if (result != cudaSuccess)                                                                            \
			std::cout << FILE_LINE " " #x " failed with error " << cudaGetErrorString(result) << std::endl;  \
	} while(0)

#if defined(__CUDA_ARCH__)
	#if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
		#define TCNN_PRAGMA_UNROLL _Pragma("unroll")
		#define TCNN_PRAGMA_NO_UNROLL _Pragma("unroll 1")
	#else
		#define TCNN_PRAGMA_UNROLL #pragma unroll
		#define TCNN_PRAGMA_NO_UNROLL #pragma unroll 1
	#endif
#else
	#define TCNN_PRAGMA_UNROLL
	#define TCNN_PRAGMA_NO_UNROLL
#endif


////////////////////
// Kernel helpers //
////////////////////

#ifdef __NVCC__
#  ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#    pragma nv_diag_suppress = unsigned_compare_with_zero
#  else
#    pragma diag_suppress = unsigned_compare_with_zero
#  endif
#endif

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define TCNN_HOST_DEVICE __host__ __device__
#define TCNN_DEVICE __device__
#define TCNN_HOST __host__
#else
#define TCNN_HOST_DEVICE
#define TCNN_DEVICE
#define TCNN_HOST
#endif

#if defined(__CUDA_ARCH__)
static_assert(__CUDA_ARCH__ >= MIN_GPU_ARCH * 10, "MIN_GPU_ARCH=" STR(TCNN_MIN_GPU_ARCH) "0 must bound __CUDA_ARCH__=" STR(__CUDA_ARCH__) " from below, but doesn't.");
#endif

template <typename T>
TCNN_HOST_DEVICE T clamp(T val, T lower, T upper) {
	return val < lower ? lower : (upper < val ? upper : val);
}

template <typename T>
TCNN_HOST_DEVICE void host_device_swap(T& a, T& b) {
	T c(a); a=b; b=c;
}

template <typename T>
TCNN_HOST_DEVICE T gcd(T a, T b) {
	while (a != 0) {
		b %= a;
		host_device_swap(a, b);
	}
	return b;
}

template <typename T>
TCNN_HOST_DEVICE T lcm(T a, T b) {
	T tmp = gcd(a, b);
	return tmp ? (a / tmp) * b : 0;
}

template <typename T>
TCNN_HOST_DEVICE T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename T>
TCNN_HOST_DEVICE T next_multiple(T val, T divisor) {
	return div_round_up(val, divisor) * divisor;
}

template <typename T>
TCNN_HOST_DEVICE T previous_multiple(T val, T divisor) {
	return (val / divisor) * divisor;
}

template <typename T>
constexpr TCNN_HOST_DEVICE bool is_pot(T val) {
	return (val & (val - 1)) == 0;
}

inline constexpr TCNN_HOST_DEVICE uint32_t next_pot(uint32_t v) {
	--v;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	return v+1;
}

constexpr uint32_t batch_size_granularity = 128;

constexpr uint32_t n_threads_linear = 128;

template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements) {
	return (uint32_t)div_round_up(n_elements, (T)n_threads_linear);
}

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
template <typename K, typename T, typename ... Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args) {
	if (n_elements <= 0) {
		return;
	}
	kernel<<<n_blocks_linear(n_elements), n_threads_linear, shmem_size, stream>>>(n_elements, args...);
}

template <typename F>
__global__ void parallel_for_kernel(const size_t n_elements, F fun) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	fun(i);
}

template <typename F>
inline void parallel_for_gpu(uint32_t shmem_size, cudaStream_t stream, size_t n_elements, F&& fun) {
	if (n_elements <= 0) {
		return;
	}
	parallel_for_kernel<F><<<n_blocks_linear(n_elements), n_threads_linear, shmem_size, stream>>>(n_elements, fun);
}

template <typename F>
inline void parallel_for_gpu(cudaStream_t stream, size_t n_elements, F&& fun) {
	parallel_for_gpu(0, stream, n_elements, std::forward<F>(fun));
}

template <typename F>
inline void parallel_for_gpu(size_t n_elements, F&& fun) {
	parallel_for_gpu(nullptr, n_elements, std::forward<F>(fun));
}

template <typename F>
__global__ void parallel_for_aos_kernel(const size_t n_elements, const uint32_t n_dims, F fun) {
	const size_t dim = threadIdx.x;
	const size_t elem = threadIdx.y + blockIdx.x * blockDim.y;
	if (dim >= n_dims) return;
	if (elem >= n_elements) return;

	fun(elem, dim);
}

template <typename F>
inline void parallel_for_gpu_aos(uint32_t shmem_size, cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
	if (n_elements <= 0 || n_dims <= 0) {
		return;
	}

	const dim3 threads = { n_dims, div_round_up(n_threads_linear, n_dims), 1 };
	const size_t n_threads = threads.x * threads.y;
	const dim3 blocks = { (uint32_t)div_round_up(n_elements * n_dims, n_threads), 1, 1 };

	parallel_for_aos_kernel<<<blocks, threads, shmem_size, stream>>>(
		n_elements, n_dims, fun
	);
}

template <typename F>
inline void parallel_for_gpu_aos(cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
	parallel_for_gpu_aos(0, stream, n_elements, n_dims, std::forward<F>(fun));
}

template <typename F>
inline void parallel_for_gpu_aos(size_t n_elements, uint32_t n_dims, F&& fun) {
	parallel_for_gpu_aos(nullptr, n_elements, n_dims, std::forward<F>(fun));
}

template <typename F>
__global__ void parallel_for_soa_kernel(const size_t n_elements, const uint32_t n_dims, F fun) {
	const size_t elem = threadIdx.x + blockIdx.x * blockDim.x;
	const size_t dim = blockIdx.y;
	if (elem >= n_elements) return;
	if (dim >= n_dims) return;

	fun(elem, dim);
}

template <typename F>
inline void parallel_for_gpu_soa(uint32_t shmem_size, cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
	if (n_elements <= 0 || n_dims <= 0) {
		return;
	}

	const dim3 blocks = { n_blocks_linear(n_elements), n_dims, 1 };

	parallel_for_soa_kernel<<<n_blocks_linear(n_elements), n_threads_linear, shmem_size, stream>>>(
		n_elements, n_dims, fun
	);
}

template <typename F>
inline void parallel_for_gpu_soa(cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
	parallel_for_gpu_soa(0, stream, n_elements, n_dims, std::forward<F>(fun));
}

template <typename F>
inline void parallel_for_gpu_soa(size_t n_elements, uint32_t n_dims, F&& fun) {
	parallel_for_gpu_soa(nullptr, n_elements, n_dims, std::forward<F>(fun));
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

template <typename T, uint32_t N_DIMS, bool ALIGNED=false>
struct alignas(ALIGNED ? next_pot(sizeof(T) * N_DIMS) : 0) vector_t {
	vector_t() = default;

	TCNN_HOST_DEVICE vector_t(T scalar) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_DIMS; ++i) {
			data[i] = scalar;
		}
	}

	template <typename U, uint32_t N, bool A>
	TCNN_HOST_DEVICE vector_t(const vector_t<U, N, A>& other) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < min(N_DIMS, N); ++i) {
			data[i] = (T)other[i];
		}
	}

	TCNN_HOST_DEVICE T& operator[](uint32_t idx) {
		return data[idx];
	}

	TCNN_HOST_DEVICE T operator[](uint32_t idx) const {
		return data[idx];
	}

	TCNN_HOST_DEVICE T& operator()(uint32_t idx) {
		return data[idx];
	}

	TCNN_HOST_DEVICE T operator()(uint32_t idx) const {
		return data[idx];
	}

	TCNN_HOST_DEVICE static constexpr uint32_t size() {
		return N_DIMS;
	}

	T data[N_DIMS];
	static constexpr uint32_t N = N_DIMS;
};

template <typename T, uint32_t N, bool A>
TCNN_HOST_DEVICE vector_t<T, N, A> operator*(T s, const vector_t<T, N, A>& v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // __hmul2 is only supported with compute capability 60 and above
	if (std::is_same<T, __half>::value && A && N % 2 == 0) {
		vector_t<T, N, A> result;
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N/2; ++i) {
			*(__half2*)&result.data[i*2] = __hmul2({s, s}, *(__half2*)&v.data[i*2]);
		}
		return result;
	} else
#endif
	{
		vector_t<T, N, A> result;
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N; ++i) {
			result[i] = s * v[i];
		}
		return result;
	}
}

template <typename T, uint32_t N, bool A>
TCNN_HOST_DEVICE vector_t<T, N, A> operator+(const vector_t<T, N, A>& v1, const vector_t<T, N, A>& v2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // __hadd2 is only supported with compute capability 60 and above
	if (std::is_same<T, __half>::value && A && N % 2 == 0) {
		vector_t<T, N, A> result;
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N/2; ++i) {
			*(__half2*)&result.data[i*2] = __hadd2(*(__half2*)&v1.data[i*2], *(__half2*)&v2.data[i*2]);
		}
		return result;
	} else
#endif
	{
		vector_t<T, N, A> result;
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N; ++i) {
			result[i] = v1[i] + v2[i];
		}
		return result;
	}
}

template <typename T, uint32_t N, bool A>
TCNN_HOST_DEVICE vector_t<T, N, A> fma(T s, const vector_t<T, N, A>& v1, const vector_t<T, N, A>& v2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // __hfma2 is only supported with compute capability 60 and above
	if (std::is_same<T, __half>::value && A && N % 2 == 0) {
		vector_t<T, N, A> result;
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N/2; ++i) {
			*(__half2*)&result.data[i*2] = __hfma2({s, s}, *(__half2*)&v1.data[i*2], *(__half2*)&v2.data[i*2]);
		}
		return result;
	} else
#endif
	{
		vector_t<T, N, A> result;
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N; ++i) {
			result[i] = fmaf(s, v1[i], v2[i]);
		}
		return result;
	}
}

template <typename T, uint32_t N, bool A>
TCNN_HOST_DEVICE vector_t<T, N, A> mix(const vector_t<T, N, A>& v1, const vector_t<T, N, A>& v2, T t) {
	return fma(((T)1.0f - t), v1, t * v2);
}

template <uint32_t N>
using vecf = vector_t<float, N>;

template <uint32_t N>
using vech = vector_t<__half, N>;

template <uint32_t N>
using vec = vector_t<network_precision_t, N>;

template <uint32_t N>
using avecf = vector_t<float, N, true>;

template <uint32_t N>
using avech = vector_t<__half, N, true>;

template <uint32_t N>
using avec = vector_t<network_precision_t, N, true>;

template <uint32_t N_FLOATS>
using vector_fullp_t = vector_t<float, N_FLOATS>;

template <uint32_t N_HALFS>
using vector_halfp_t = vector_t<__half, N_HALFS>;

template <typename T>
struct PitchedPtr {
	TCNN_HOST_DEVICE PitchedPtr() : ptr{nullptr}, stride_in_bytes{sizeof(T)} {}
	TCNN_HOST_DEVICE PitchedPtr(T* ptr, size_t stride_in_elements, size_t offset = 0, size_t extra_stride_bytes = 0) : ptr{ptr + offset}, stride_in_bytes{(uint32_t)(stride_in_elements * sizeof(T) + extra_stride_bytes)} {}

	template <typename U>
	TCNN_HOST_DEVICE explicit PitchedPtr(PitchedPtr<U> other) : ptr{(T*)other.ptr}, stride_in_bytes{other.stride_in_bytes} {}

	TCNN_HOST_DEVICE T* operator()(uint32_t y) const {
		return (T*)((const char*)ptr + y * stride_in_bytes);
	}

	TCNN_HOST_DEVICE void operator+=(uint32_t y) {
		ptr = (T*)((const char*)ptr + y * stride_in_bytes);
	}

	TCNN_HOST_DEVICE void operator-=(uint32_t y) {
		ptr = (T*)((const char*)ptr - y * stride_in_bytes);
	}

	TCNN_HOST_DEVICE explicit operator bool() const {
		return ptr;
	}

	T* ptr;
	uint32_t stride_in_bytes;
};

TCNN_NAMESPACE_END
