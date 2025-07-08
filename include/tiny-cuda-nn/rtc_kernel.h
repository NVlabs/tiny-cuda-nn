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
 */

#pragma once

#include <tiny-cuda-nn/common_host.h>

#include <cuda.h>

#ifdef TCNN_CMRC
namespace cmrc { class embedded_filesystem; }
namespace tcnn { std::vector<std::pair<std::string, const char*>> all_files(const cmrc::embedded_filesystem& fs, const std::string& dir = ""); }
#endif

namespace tcnn {

bool supports_jit_fusion(int device);
inline bool supports_jit_fusion() {
	return supports_jit_fusion(cuda_device());
}

std::string& rtc_cache_dir();
void rtc_set_cache_dir(const std::string& dir);

std::string& rtc_include_dir();
void rtc_set_include_dir(const std::string& dir);

class CudaRtcKernel {
public:
	CudaRtcKernel(const std::string& name, const std::string& kernel_code, const std::vector<std::pair<std::string, const char*>>& extra_includes = {});
	~CudaRtcKernel();

	void clear();

	void set(CUfunction_attribute attr, int value);

	template <typename ... Types>
	void launch(dim3 blocks, dim3 threads, uint32_t shmem_size, cudaStream_t stream, Types&&... args) {
		if (blocks.x * blocks.y * blocks.z == 0 || threads.x * threads.y * threads.z == 0) {
			return;
		}

		const void* args_array[sizeof...(Types)] = { &args... };

		// CUDA docs state that one has to opt-in for larger amounts of shmem than 48KiB == 49'152B
		if (shmem_size > 49152) {
			set(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_size);
		}

		CU_CHECK_THROW(cuLaunchKernel(m_kernel, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, shmem_size, stream, (void**)args_array, nullptr));
	}

	template <typename ... Types>
	void launch(uint32_t blocks, uint32_t threads, uint32_t shmem_size, cudaStream_t stream, Types&&... args) {
		launch(dim3(blocks, 1, 1), dim3(threads, 1, 1), shmem_size, stream, std::forward<Types>(args)...);
	}

private:
	CUmodule m_module = {};
	CUfunction m_kernel = {};
};

}
