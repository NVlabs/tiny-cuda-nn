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

/** @file   rtc_kernel.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Helper class for compiling CUDA kernels at run time using
 *          the NVRTC API.
 */

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/rtc_kernel.h>

#ifdef TCNN_CMRC
#include <cmrc/cmrc.hpp>
CMRC_DECLARE(tcnn);
#endif

#include <chrono>
#include <fstream>
#include <sstream>

namespace tcnn {
std::string& rtc_cache_dir() {
	static std::string cache_dir = "";
	return cache_dir;
}

void rtc_set_cache_dir(const std::string& dir) {
	rtc_cache_dir() = dir;
}

std::string& rtc_include_dir() {
	static std::string header_dir = "";
	return header_dir;
}

void rtc_set_include_dir(const std::string& dir) {
	rtc_include_dir() = dir;
}
}

#ifndef TCNN_RTC
namespace tcnn {

bool supports_jit_fusion(int device) {
	return false;
}

CudaRtcKernel::CudaRtcKernel(const std::string& name, const std::string& kernel_code, const std::vector<std::pair<std::string, const char*>>& extra_includes) {
	throw std::runtime_error{"tiny-cuda-nn was not compiled with runtime compilation (RTC) support."};
}

CudaRtcKernel::~CudaRtcKernel() {}
void CudaRtcKernel::clear() {}

}
#else
#include <nvrtc.h>

/// Checks the result of a cudaXXXXXX call and throws an error on failure
#define NVRTC_CHECK_THROW(x)                                                                                             \
	do {                                                                                                                 \
		nvrtcResult _result = x;                                                                                          \
		if (_result != NVRTC_SUCCESS)                                                                                     \
			throw std::runtime_error{std::string(FILE_LINE " " #x " failed with error ") + nvrtcGetErrorString(_result)}; \
	} while(0)

namespace tcnn {

bool supports_jit_fusion(int device) {
	// Technically, RTC is supported on earlier architectures and CUDA versions
	// as well, but the way we fuse MLPs doesn't work on those. So only permit
	// JIT operation on Turing and higher.
	return TCNN_HALF_PRECISION && cuda_supported_compute_capability(device) >= 75 && cuda_runtime_version() >= 11080;
}

#ifdef TCNN_CMRC
std::vector<std::pair<std::string, const char*>> all_files(const cmrc::embedded_filesystem& fs, const std::string& dir) {
	std::vector<std::pair<std::string, const char*>> result;
	for (auto&& entry : fs.iterate_directory(dir)) {
		auto fn = dir.empty() ? entry.filename() : fmt::format("{}/{}", dir, entry.filename());
		if (entry.is_file()) {
			result.emplace_back(fn, fs.open(fn).begin());
		} else if (entry.is_directory()) {
			auto files_in_dir = all_files(fs, fn);
			result.insert(std::end(result), std::begin(files_in_dir), std::end(files_in_dir));
		}
	}

	return result;
}
#endif

// If a cache dir is provided, compilation artifacts will be cached in there and re-loaded upon program restart. Improves the user experience.
CudaRtcKernel::CudaRtcKernel(const std::string& name, const std::string& kernel_code, const std::vector<std::pair<std::string, const char*>>& extra_includes) {
	if (!supports_jit_fusion()) {
		throw std::runtime_error{fmt::format(
			"JIT: unsupported. Must use CUDA 11.8 and a GPU with compute capability 75 or higher, but got CUDA {} and compute capability {}.", cuda_runtime_version(), cuda_compute_capability()
		)};
	}

	ScopeGuard cleanup_guard{[&]() { clear(); }};

	auto start_time = std::chrono::steady_clock::now();

	uint32_t cc = cuda_supported_compute_capability();

	// Strangely, compute capabilities above 90 compile to slower MLP code, even on Blackwell GPUs.
	// For now, until we figure out how to fix this, we therefore cap the compute capability at 90.
	cc = min(cc, 90u);

	std::vector<std::string> opts = {
		fmt::format("--gpu-architecture=compute_{}", cc),
		fmt::format("-DTCNN_MIN_GPU_ARCH={}", cc),
		"--std=c++14",
#ifdef TCNN_RTC_USE_FAST_MATH
		"--use_fast_math",
#endif
		"--extra-device-vectorization",
	};

	if (!rtc_include_dir().empty()) {
		opts.emplace_back(fmt::format("-I{}", rtc_include_dir()));
	}

	std::string complete_code = dfmt(0, R"(
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

			/** @file   {KERNEL_NAME}.cu
			 *  @author Thomas Müller, NVIDIA
			 *  @brief  Automatically generated kernel {KERNEL_NAME}
			 */

			/* Compiler options
			{OPTS}
			*/

			// NVRTC does not come with the C++ standard library out of the box and
			// it would be troublesome to bundle it or require users to have it installed
			// in readily available paths. So we instead include a minimal custom
			// implementation of just those function of std:: that we require.
			#include <tiny-cuda-nn/ministd.h>
			{PREAMBLE}

			{KERNEL_CODE}
		)",
		"KERNEL_NAME"_a = name,
		"PREAMBLE"_a = generate_device_code_preamble(),
		"OPTS"_a = join(opts, "\n"),
		"KERNEL_CODE"_a = kernel_code
	);

	size_t code_hash = hash_combine(0, complete_code);

	std::vector<const char*> headers = {
		// First define headers that we wish to ignore (because they won't be available
		// at runtime). Instead, we will include tiny-cuda-nn/ministd.h, which implements
		// the small subset of the STL that we actually need.
		"algorithm", "cassert", "cmath", "cstddef", "cstdint", "cstdio", "cuda.h", "limits", "type_traits", "initializer_list",
	};
	std::vector<const char*> headers_content(headers.size(), "");

	// Next, we add all headers that come bundled with tcnn as well as those we received in the constructor.
	// We combine each header's hash with that of our code to obtain a unique fingerprint that can be used
	// for caching.
	std::vector<std::pair<std::string, const char*>> includes;
#if TCNN_CMRC
	includes = all_files(cmrc::tcnn::get_filesystem());
#endif
	includes.insert(std::end(includes), std::begin(extra_includes), std::end(extra_includes));
	for (const auto& entry : includes) {
		headers.emplace_back(entry.first.c_str());
		headers_content.emplace_back(entry.second);
		code_hash = hash_combine(code_hash, std::string{entry.second});
	}

	std::string code_hash_string = fmt::format("{:016x}", code_hash);
	std::string filename = fmt::format("{}.{}.cu", name, code_hash_string);

	std::string cache_dir = rtc_cache_dir();
	bool use_cache = !cache_dir.empty();

	std::string cached_code_filename = fmt::format("{}/{}", cache_dir, filename);
	std::string cached_ptx_filename = fmt::format("{}/{}.{}.ptx", cache_dir, name, code_hash_string);

	std::vector<char> ptx;
	std::string lowered_kernel_name;

	if (use_cache) {
		{
			std::ifstream f{cached_code_filename};
			if (!f) {
				// Dump source code to disk for easier debugging & caching.
				std::ofstream of{cached_code_filename};
				of << complete_code;
			}
		}

		// Check if we've cached the resulting PTX last time around and load it if so.
		std::ifstream f{cached_ptx_filename};
		if (f) {
			// The first line of the cached PTX contains a comment of the form
			//  `//lowered_kernel_name=<value>` which we need to parse.
			std::string first_line;
			f >> first_line;
			auto s = split(first_line, "=");
			if (s.size() == 2 && s[0].find("lowered_kernel_name") != std::string::npos) {
				lowered_kernel_name = s[1];
				f.seekg(0, std::ios::end);
				size_t size = f.tellg();
				ptx.resize(size, '\0');
				f.seekg(0);
				f.read(ptx.data(), size);
			}
		}
	}

	// If we haven't loaded PTX from cache, compile the program
	bool cached_ptx = !ptx.empty();
	if (!cached_ptx) {
		nvrtcProgram prog;
		NVRTC_CHECK_THROW(nvrtcCreateProgram(&prog, complete_code.c_str(), filename.c_str(), headers.size(), headers_content.data(), headers.data()));

		NVRTC_CHECK_THROW(nvrtcAddNameExpression(prog, name.c_str()));

		std::vector<const char*> opts_c_str;
		for (const auto& opt : opts) {
			opts_c_str.emplace_back(opt.c_str());
		}

		nvrtcResult compile_result = nvrtcCompileProgram(prog, opts_c_str.size(), opts_c_str.data());
		if (compile_result != NVRTC_SUCCESS) {
			size_t log_size;
			NVRTC_CHECK_THROW(nvrtcGetProgramLogSize(prog, &log_size));
			std::vector<char> log(log_size+1, '\0');
			NVRTC_CHECK_THROW(nvrtcGetProgramLog(prog, log.data()));
			throw std::runtime_error{fmt::format("JIT: compiling {} failed:\n{}", filename, log.data())};
		}

		const char* lowered_kernel_name_cstr;
		NVRTC_CHECK_THROW(nvrtcGetLoweredName(prog, name.c_str(), &lowered_kernel_name_cstr));
		lowered_kernel_name = lowered_kernel_name_cstr;

		size_t ptx_size;
		NVRTC_CHECK_THROW(nvrtcGetPTXSize(prog, &ptx_size));
		ptx.resize(ptx_size, '\0');
		NVRTC_CHECK_THROW(nvrtcGetPTX(prog, ptx.data()));

		std::string lowered_kernel_name_comment = fmt::format("//lowered_kernel_name={}\n", lowered_kernel_name);
		ptx.insert(std::begin(ptx), std::begin(lowered_kernel_name_comment), std::end(lowered_kernel_name_comment));

		NVRTC_CHECK_THROW(nvrtcDestroyProgram(&prog));

		if (use_cache) {
			std::ofstream f{cached_ptx_filename};
			if (f) {
				f.write(ptx.data(), ptx.size());
			}
		}
	}

	CU_CHECK_THROW(cuModuleLoadDataEx(&m_module, ptx.data(), 0, nullptr, nullptr));
	CU_CHECK_THROW(cuModuleGetFunction(&m_kernel, m_module, lowered_kernel_name.c_str()));

	float compilation_duration_seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - start_time).count();
	if (!cached_ptx || compilation_duration_seconds > 0.1f) { // Don't spam the user if the kernel is loaded from cache very fast.
		log_success("JIT: compiled {} in {:.02f}s (cached_ptx={})", filename, compilation_duration_seconds, cached_ptx);
	}

	cleanup_guard.disarm();
}

CudaRtcKernel::~CudaRtcKernel() {
	clear();
}

void CudaRtcKernel::clear() {
	if (m_module) {
		cuModuleUnload(m_module);
		m_module = {};
	}
}

void CudaRtcKernel::set(CUfunction_attribute attrib, int value) {
	CU_CHECK_THROW(cuFuncSetAttribute(m_kernel, attrib, value));
}
}
#endif
