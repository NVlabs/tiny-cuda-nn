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

/** @file   common_host.h
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Common utilities that are needed by pretty much every component of this framework.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/cpp_api.h>

#include <fmt/format.h>

#include <array>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tcnn {

using namespace fmt::literals;

enum class LogSeverity {
	Info,
	Debug,
	Warning,
	Error,
	Success,
};

const std::function<void(LogSeverity, const std::string&)>& log_callback();
void set_log_callback(const std::function<void(LogSeverity, const std::string&)>& callback);

template <typename... Ts>
void log(LogSeverity severity, const std::string& msg, Ts&&... args) {
	log_callback()(severity, fmt::format(msg, std::forward<Ts>(args)...));
}

template <typename... Ts> void log_info(const std::string& msg, Ts&&... args) { log(LogSeverity::Info, msg, std::forward<Ts>(args)...); }
template <typename... Ts> void log_debug(const std::string& msg, Ts&&... args) { log(LogSeverity::Debug, msg, std::forward<Ts>(args)...); }
template <typename... Ts> void log_warning(const std::string& msg, Ts&&... args) { log(LogSeverity::Warning, msg, std::forward<Ts>(args)...); }
template <typename... Ts> void log_error(const std::string& msg, Ts&&... args) { log(LogSeverity::Error, msg, std::forward<Ts>(args)...); }
template <typename... Ts> void log_success(const std::string& msg, Ts&&... args) { log(LogSeverity::Success, msg, std::forward<Ts>(args)...); }

bool verbose();
void set_verbose(bool verbose);

#define CHECK_THROW(x) \
	do { if (!(x)) throw std::runtime_error{FILE_LINE " check failed: " #x}; } while(0)

/// Checks the result of a cuXXXXXX call and throws an error on failure
#define CU_CHECK_THROW(x) \
	do { \
		CUresult _result = x; \
		if (_result != CUDA_SUCCESS) { \
			const char *msg; \
			cuGetErrorName(_result, &msg); \
			throw std::runtime_error{fmt::format(FILE_LINE " " #x " failed: {}", msg)}; \
		} \
	} while(0)

/// Checks the result of a cuXXXXXX call and prints an error on failure
#define CU_CHECK_PRINT(x) \
	do { \
		CUresult _result = x; \
		if (_result != CUDA_SUCCESS) { \
			const char *msg; \
			cuGetErrorName(_result, &msg); \
			log_error(FILE_LINE " " #x " failed: {}", msg); \
		} \
	} while(0)

/// Checks the result of a cudaXXXXXX call and throws an error on failure
#define CUDA_CHECK_THROW(x) \
	do { \
		cudaError_t _result = x; \
		if (_result != cudaSuccess) \
			throw std::runtime_error{fmt::format(FILE_LINE " " #x " failed: {}", cudaGetErrorString(_result))}; \
	} while(0)

/// Checks the result of a cudaXXXXXX call and prints an error on failure
#define CUDA_CHECK_PRINT(x) \
	do { \
		cudaError_t _result = x; \
		if (_result != cudaSuccess) \
			log_error(FILE_LINE " " #x " failed: {}", cudaGetErrorString(_result)); \
	} while(0)

/// Checks the result of optixXXXXXX call and throws an error on failure
#define OPTIX_CHECK_THROW(x)                                                                                 \
	do {                                                                                                     \
		OptixResult _result = x;                                                                                 \
		if (_result != OPTIX_SUCCESS) {                                                                          \
			throw std::runtime_error(std::string("Optix call '" #x "' failed."));                            \
		}                                                                                                    \
	} while(0)

/// Checks the result of a optixXXXXXX call and throws an error with a log message on failure
#define OPTIX_CHECK_THROW_LOG(x)                                                                                                                          \
	do {                                                                                                                                                  \
		OptixResult _result = x;                                                                                                                              \
		const size_t sizeof_log_returned = sizeof_log;                                                                                                    \
		sizeof_log = sizeof( log ); /* reset sizeof_log for future calls */                                                                               \
		if (_result != OPTIX_SUCCESS) {                                                                                                                       \
			throw std::runtime_error(std::string("Optix call '" #x "' failed. Log:\n") + log + (sizeof_log_returned == sizeof_log ? "" : "<truncated>")); \
		}                                                                                                                                                 \
	} while(0)

//////////////////////////////
// Enum<->string conversion //
//////////////////////////////

Activation string_to_activation(const std::string& activation_name);
std::string to_string(Activation activation);

GridType string_to_grid_type(const std::string& grid_type);
std::string to_string(GridType grid_type);

HashType string_to_hash_type(const std::string& hash_type);
std::string to_string(HashType hash_type);

InterpolationType string_to_interpolation_type(const std::string& interpolation_type);
std::string to_string(InterpolationType interpolation_type);

ReductionType string_to_reduction_type(const std::string& reduction_type);
std::string to_string(ReductionType reduction_type);

//////////////////
// Misc helpers //
//////////////////

int cuda_runtime_version();
inline std::string cuda_runtime_version_string() {
	int v = cuda_runtime_version();
	return fmt::format("{}.{}", v / 1000, (v % 100) / 10);
}

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

uint32_t cuda_max_supported_compute_capability();
uint32_t cuda_supported_compute_capability(int device);
inline uint32_t cuda_supported_compute_capability() {
	return cuda_supported_compute_capability(cuda_device());
}

size_t cuda_max_shmem(int device);
inline size_t cuda_max_shmem() {
	return cuda_max_shmem(cuda_device());
}

uint32_t cuda_max_registers(int device);
inline uint32_t cuda_max_registers() {
	return cuda_max_registers(cuda_device());
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

// Hash helpers taken from https://stackoverflow.com/a/50978188
template <typename T>
T xorshift(T n, int i) {
	return n ^ (n >> i);
}

inline uint32_t distribute(uint32_t n) {
	uint32_t p = 0x55555555ul; // pattern of alternating 0 and 1
	uint32_t c = 3423571495ul; // random uneven integer constant;
	return c * xorshift(p * xorshift(n, 16), 16);
}

inline uint64_t distribute(uint64_t n) {
	uint64_t p = 0x5555555555555555ull; // pattern of alternating 0 and 1
	uint64_t c = 17316035218449499591ull;// random uneven integer constant;
	return c * xorshift(p * xorshift(n, 32), 32);
}

template <typename T, typename S>
constexpr typename std::enable_if<std::is_unsigned<T>::value, T>::type rotl(const T n, const S i) {
	const T m = (std::numeric_limits<T>::digits - 1);
	const T c = i & m;
	return (n << c) | (n >> (((T)0 - c) & m)); // this is usually recognized by the compiler to mean rotation
}

template <typename T>
size_t hash_combine(std::size_t seed, const T& v) {
	return rotl(seed, std::numeric_limits<size_t>::digits / 3) ^ distribute(std::hash<T>{}(v));
}

std::string generate_device_code_preamble();

std::string to_snake_case(const std::string& str);

std::vector<std::string> split(const std::string& text, const std::string& delim);

template <typename T>
std::string join(const T& components, const std::string& delim) {
	std::ostringstream s;
	for (const auto& component : components) {
		if (&components[0] != &component) {
			s << delim;
		}
		s << component;
	}

	return s.str();
}

template <typename... Ts>
std::string dfmt(uint32_t indent, const std::string& format, Ts&&... args) {
	// Trim empty lines at the beginning and end of format string.
	// Also re-indent the format string `indent` deep.
	uint32_t input_indent = std::numeric_limits<uint32_t>::max();
	uint32_t n_empty_leading = 0, n_empty_trailing = 0;
	bool leading = true;

	std::vector<std::string> lines = split(format, "\n");
	for (const auto& line : lines) {
		bool empty = true;
		uint32_t line_indent = 0;
		for (uint32_t i = 0; i < line.length(); ++i) {
			if (empty && line[i] == '\t') {
				line_indent = i+1;
			} else {
				empty = false;
				break;
			}
		}

		if (empty) {
			if (leading) {
				++n_empty_leading;
			}
			++n_empty_trailing;
			continue;
		}

		n_empty_trailing = 0;

		leading = false;
		input_indent = std::min(input_indent, line_indent);
	}

	if (input_indent == std::numeric_limits<uint32_t>::max()) {
		return "";
	}

	lines.erase(lines.end() - n_empty_trailing, lines.end());
	lines.erase(lines.begin(), lines.begin() + n_empty_leading);

	for (auto& line : lines) {
		if (line.length() >= input_indent) {
			line = line.substr(input_indent);
			line = line.insert(0, indent, '\t');
		}
	}

	return fmt::format(join(lines, "\n"), std::forward<Ts>(args)...);
}

std::string to_lower(std::string str);
std::string to_upper(std::string str);
inline bool equals_case_insensitive(const std::string& str1, const std::string& str2) {
	return to_lower(str1) == to_lower(str2);
}

struct CaseInsensitiveHash { size_t operator()(const std::string& v) const { return std::hash<std::string>{}(to_lower(v)); }};
struct CaseInsensitiveEqual { bool operator()(const std::string& l, const std::string& r) const { return equals_case_insensitive(l, r); }};

template <typename T>
using ci_hashmap = std::unordered_map<std::string, T, CaseInsensitiveHash, CaseInsensitiveEqual>;


template <typename T>
std::string type_to_string();

template <typename T, uint32_t N, size_t A>
std::string to_string(const tvec<T, N, A>& v) {
	return fmt::format("tvec<{}, {}, {}>({})", type_to_string<T>(), N, A, join(v, ", "));
}

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

template <typename T>
class Lazy {
public:
	template <typename F>
	T& get(F&& generator) {
		if (!m_val) {
			m_val = generator();
		}

		return m_val;
	}

private:
	T m_val;
};

#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
template <typename K, typename T, typename ... Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args) {
	if (n_elements <= 0) {
		return;
	}
	kernel<<<n_blocks_linear(n_elements), N_THREADS_LINEAR, shmem_size, stream>>>(n_elements, args...);
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
	parallel_for_kernel<F><<<n_blocks_linear(n_elements), N_THREADS_LINEAR, shmem_size, stream>>>(n_elements, fun);
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

	const dim3 threads = { n_dims, div_round_up(N_THREADS_LINEAR, n_dims), 1 };
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

	parallel_for_soa_kernel<<<n_blocks_linear(n_elements), N_THREADS_LINEAR, shmem_size, stream>>>(
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

}
