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

/** @file   gpu_memory.h
 *  @author Nikolaus Binder and Thomas Müller, NVIDIA
 *  @brief  Managed memory on the GPU. Like a std::vector, memory is alocated  either explicitly (resize/enlarge)
 *          or implicitly (resize_and_copy_from_host etc). Memory is always and automatically released in the destructor.
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <json/json.hpp>

#include <atomic>
#include <cuda.h>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>


TCNN_NAMESPACE_BEGIN

#define DEBUG_GUARD_SIZE 0

static std::atomic<size_t> s_total_n_bytes_allocated{0};

inline size_t total_n_bytes_allocated() {
	return s_total_n_bytes_allocated;
}

/// Managed memory on the Device
template<class T> class GPUMemory {
protected:
	T*       m_data      = nullptr; ///< pointer to the actual data
	size_t   m_size      = 0; ///< size of the array (number of elements)
	size_t   m_allocation_n_bytes = 0;
	bool m_compressed = false;
	cudaStream_t m_stream = nullptr;

	CUmemGenericAllocationHandle m_alloc_handle;

public:
	/** @name Constructors/destructor
	 *  @{
	 */
	/// Default constructor (does not allocate anything)
	GPUMemory() {}

	GPUMemory<T>& operator=(GPUMemory<T>&& other) {
		m_size = other.m_size;
		m_allocation_n_bytes = other.m_allocation_n_bytes;
		m_data = other.m_data;
		m_compressed = other.m_compressed;
		m_stream = other.m_stream;
		m_alloc_handle = other.m_alloc_handle;

		other.m_size = 0; other.m_data = nullptr; other.m_stream = nullptr;
		return *this;
	}

	/// Move constructor
	GPUMemory(GPUMemory<T>&& other) {
		*this = std::move(other);
	}

	/// Copy constructor (data is actually being duplicated)
	explicit GPUMemory(const GPUMemory<T> &other) {
		copy_from_device(other);
	}

	void check_guards() const {
#if DEBUG_GUARD_SIZE > 0
		if (!m_data)
			return;
		if (m_compressed)
			return;
		uint8_t buf[DEBUG_GUARD_SIZE];
		const uint8_t *rawptr=(const uint8_t *)m_data;
		cudaMemcpy(buf, rawptr-DEBUG_GUARD_SIZE, DEBUG_GUARD_SIZE, cudaMemcpyDeviceToHost);
		for (int i=0;i<DEBUG_GUARD_SIZE;++i) if (buf[i]!=0xff) {
			printf("TRASH BEFORE BLOCK offset %d data %p, read 0x%02x expected 0xff!\n", i, m_data, buf[i] );
			break;
		}
		cudaMemcpy(buf, rawptr+m_size*sizeof(T), DEBUG_GUARD_SIZE, cudaMemcpyDeviceToHost);
		for (int i=0;i<DEBUG_GUARD_SIZE;++i) if (buf[i]!=0xfe) {
			printf("TRASH AFTER BLOCK offset %d data %p, read 0x%02x expected 0xfe!\n", i, m_data, buf[i] );
			break;
		}
#endif
	}

	void allocate_memory(size_t n_bytes) {
		if (n_bytes == 0) {
			return;
		}

#ifdef TCNN_VERBOSE_MEMORY_ALLOCS
		std::cout << "GPUMemory: Allocating " << bytes_to_string(n_bytes) << "." << std::endl;
#endif

		if (!m_compressed) {
			uint8_t *rawptr = nullptr;
			if (m_stream) {
				CUDA_CHECK_THROW(cudaMallocAsync(&rawptr, n_bytes+DEBUG_GUARD_SIZE*2, m_stream));
#if DEBUG_GUARD_SIZE > 0
				CUDA_CHECK_THROW(cudaMemsetAsync(rawptr , 0xff, DEBUG_GUARD_SIZE, m_stream));
				CUDA_CHECK_THROW(cudaMemsetAsync(rawptr+n_bytes+DEBUG_GUARD_SIZE , 0xfe, DEBUG_GUARD_SIZE, m_stream));
#endif
			} else {
				CUDA_CHECK_THROW(cudaMalloc(&rawptr, n_bytes+DEBUG_GUARD_SIZE*2));
#if DEBUG_GUARD_SIZE > 0
				CUDA_CHECK_THROW(cudaMemset(rawptr , 0xff, DEBUG_GUARD_SIZE));
				CUDA_CHECK_THROW(cudaMemset(rawptr+n_bytes+DEBUG_GUARD_SIZE , 0xfe, DEBUG_GUARD_SIZE));
#endif
			}
			if (rawptr) rawptr+=DEBUG_GUARD_SIZE;
			m_data=(T*)(rawptr);
			s_total_n_bytes_allocated += n_bytes;
			return;
		}

		if (m_stream) {
			throw std::runtime_error{"GPUMemory does not support async compressed memory."};
		}

		int device = 0;

		CUmemAllocationProp prop = {};
		prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id = device;
		prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

		size_t granularity;
		CU_CHECK_THROW(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

		// Ensure size matches granularity requirements for the allocation
		size_t padded_n_bytes = next_multiple(n_bytes, granularity);

		// Allocate physical memory
		CU_CHECK_THROW(cuMemCreate(&m_alloc_handle, padded_n_bytes, &prop, 0));

		CUmemAllocationProp alloc_prop = {};
		cuMemGetAllocationPropertiesFromHandle(&alloc_prop, m_alloc_handle);
		if (alloc_prop.allocFlags.compressionType != CU_MEM_ALLOCATION_COMP_GENERIC) {
			std::cout << "WARNING: requested compressed memory, but did not obtain it." << std::endl;
		}
#ifdef TCNN_VERBOSE_MEMORY_ALLOCS
		else {
			std::cout << "SUCCESS: got compressed memory." << std::endl;
		}
#endif

		CUdeviceptr ptr;
		// `ptr` holds the returned start of virtual address range reserved.
		CU_CHECK_THROW(cuMemAddressReserve(&ptr, padded_n_bytes, 0, 0, 0));
		CU_CHECK_THROW(cuMemMap(ptr, padded_n_bytes, 0, m_alloc_handle, 0));

		CUmemAccessDesc accessDesc = {};
		accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		accessDesc.location.id = device;
		accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

		// Make the address accessible
		CU_CHECK_THROW(cuMemSetAccess(ptr, padded_n_bytes, &accessDesc, 1));

		m_data = (T*)ptr;
		m_allocation_n_bytes = padded_n_bytes;

		s_total_n_bytes_allocated += n_bytes;
	}

	void free_memory() {
		if (!m_data) {
			return;
		}

		if (!m_compressed) {
			uint8_t *rawptr = (uint8_t*)m_data;
			if (rawptr) rawptr-=DEBUG_GUARD_SIZE;
			if (m_stream) {
				CUDA_CHECK_THROW(cudaFreeAsync(rawptr, m_stream));
			} else {
				CUDA_CHECK_THROW(cudaFree(rawptr));
			}

			s_total_n_bytes_allocated -= get_bytes();
		} else {
			if (m_stream) {
				throw std::runtime_error{"GPUMemory does not support async compressed memory."};
			}

			CU_CHECK_THROW(cuMemUnmap((CUdeviceptr)m_data, m_allocation_n_bytes));
			CU_CHECK_THROW(cuMemRelease(m_alloc_handle));
			CU_CHECK_THROW(cuMemAddressFree((CUdeviceptr)m_data, m_allocation_n_bytes));

			s_total_n_bytes_allocated -= get_bytes();
		}

		m_data = nullptr;
	}

	/// Allocates memory for size items of type T
	GPUMemory(const size_t size, bool compress) : m_compressed(compress) {
		resize(size);
	}

	/// Allocates memory for size items of type T
	GPUMemory(const size_t size, cudaStream_t stream = nullptr) : m_stream(stream) {
		resize(size);
	}

	/// Frees memory again
	virtual ~GPUMemory() {
		try {
			if (m_data && m_size > 0) {
				free_memory();
				m_size = 0;
			}
		} catch (std::runtime_error error) {
			// Don't need to report on memory-free problems when the driver is shutting down.
			if (std::string{error.what()}.find("driver shutting down") == std::string::npos) {
				fprintf(stderr, "Could not free memory: %s\n", error.what());
			}
		}
	}
	/** @} */

	/** @name Resizing/enlargement
	 *  @{
	 */
	/// Resizes the array to the exact new size, even if it is already larger
	void resize(const size_t size) {
		if (m_size != size) {
			if (m_size) {
				try {
					free_memory();
				} catch (std::runtime_error error) {
					throw std::runtime_error(std::string("Could not free memory: ") + error.what());
				}
			}

			if (size > 0) {
				try {
					allocate_memory(size * sizeof(T));
				} catch (std::runtime_error error) {
					throw std::runtime_error(std::string("Could not allocate memory: ") + error.what());
				}
			}

			m_size = size;
		}
	}

	/// Enlarges the array if its size is smaller
	void enlarge(const size_t size) {
		if (size > m_size) {
			resize(size);
		}
	}
	/** @} */

	/** @name Memset
	 *  @{
	 */
	/// Sets the memory of the first num_elements to value
	void memset(const int value, const size_t num_elements, const size_t offset = 0) {
		if (num_elements + offset > m_size) {
			throw std::runtime_error("Could not set memory: Number of elements larger than allocated memory");
		}

		try {
			if (m_stream) {
				CUDA_CHECK_THROW(cudaMemsetAsync(m_data + offset, value, num_elements * sizeof(T), m_stream));
			} else {
				CUDA_CHECK_THROW(cudaMemset(m_data + offset, value, num_elements * sizeof(T)));
			}
		} catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not set memory: ") + error.what());
		}
	}

	/// Sets the memory of the all elements to value
	void memset(const int value) {
		memset(value, m_size);
	}
	/** @} */

	/** @name Copy operations
	 *  @{
	 */
	/// Copy data of num_elements from the raw pointer on the host
	void copy_from_host(const T* host_data, const size_t num_elements) {
		try {
			if (m_stream) {
				CUDA_CHECK_THROW(cudaMemcpyAsync(data(), host_data, num_elements * sizeof(T), cudaMemcpyHostToDevice, m_stream));
				CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream));
			} else {
				CUDA_CHECK_THROW(cudaMemcpy(data(), host_data, num_elements * sizeof(T), cudaMemcpyHostToDevice));
			}
		} catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not copy from host: ") + error.what());
		}
	}

	/// Copy num_elements from the host vector
	void copy_from_host(const std::vector<T>& data, const size_t num_elements) {
		if (data.size() < num_elements) {
			throw std::runtime_error(std::string("Trying to copy ") + std::to_string(num_elements) + std::string(" elements, but vector size is only ") + std::to_string(data.size()));
		}
		copy_from_host(data.data(), num_elements);
	}

	/// Copies data from the raw host pointer to fill the entire array
	void copy_from_host(const T* data) {
		copy_from_host(data, m_size);
	}

	/// Copies num_elements of data from the raw host pointer after enlarging the array so that everything fits in
	void enlarge_and_copy_from_host(const T* data, const size_t num_elements) {
		enlarge(num_elements);
		copy_from_host(data, num_elements);
	}

	/// Copies num_elements from the host vector after enlarging the array so that everything fits in
	void enlarge_and_copy_from_host(const std::vector<T>& data, const size_t num_elements) {
		enlarge_and_copy_from_host(data.data(), num_elements);
	}

	/// Copies the entire host vector after enlarging the array so that everything fits in
	void enlarge_and_copy_from_host(const std::vector<T>& data) {
		enlarge_and_copy_from_host(data.data(), data.size());
	}

	/// Copies num_elements of data from the raw host pointer after resizing the array
	void resize_and_copy_from_host(const T* data, const size_t num_elements) {
		resize(num_elements);
		copy_from_host(data, num_elements);
	}

	/// Copies num_elements from the host vector after resizing the array
	void resize_and_copy_from_host(const std::vector<T>& data, const size_t num_elements) {
		resize_and_copy_from_host(data.data(), num_elements);
	}

	/// Copies the entire host vector after resizing the array
	void resize_and_copy_from_host(const std::vector<T>& data) {
		resize_and_copy_from_host(data.data(), data.size());
	}

	/// Copies the entire host vector to the device. Fails if there is not enough space available.
	void copy_from_host(const std::vector<T>& data) {
		if (data.size() < m_size) {
			throw std::runtime_error(std::string("Trying to copy ") + std::to_string(m_size) + std::string(" elements, but vector size is only ") + std::to_string(data.size()));
		}
		copy_from_host(data.data(), m_size);
	}

	/// Copies num_elements of data from the raw host pointer to the device. Fails if there is not enough space available.
	void copy_to_host(T* host_data, const size_t num_elements) const {
		if (num_elements > m_size) {
			throw std::runtime_error(std::string("Trying to copy ") + std::to_string(num_elements) + std::string(" elements, but vector size is only ") + std::to_string(m_size));
		}
		try {
			if (m_stream) {
				CUDA_CHECK_THROW(cudaMemcpyAsync(host_data, data(), num_elements * sizeof(T), cudaMemcpyDeviceToHost, m_stream));
				CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream));
			} else {
				CUDA_CHECK_THROW(cudaMemcpy(host_data, data(), num_elements * sizeof(T), cudaMemcpyDeviceToHost));
			}
		} catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not copy to host: ") + error.what());
		}
	}

	/// Copies num_elements from the device to a vector on the host
	void copy_to_host(std::vector<T>& data, const size_t num_elements) const {
		if (data.size() < num_elements) {
			throw std::runtime_error(std::string("Trying to copy ") + std::to_string(num_elements) + std::string(" elements, but vector size is only ") + std::to_string(data.size()));
		}
		copy_to_host(data.data(), num_elements);
	}

	/// Copies num_elements from the device to a raw pointer on the host
	void copy_to_host(T* data) const {
		copy_to_host(data, m_size);
	}

	/// Copies all elements from the device to a vector on the host
	void copy_to_host(std::vector<T>& data) const {
		if (data.size() < m_size) {
			throw std::runtime_error(std::string("Trying to copy ") + std::to_string(m_size) + std::string(" elements, but vector size is only ") + std::to_string(data.size()));
		}
		copy_to_host(data.data(), m_size);
	}

	/// Copies data from another device array to this one, automatically resizing it
	void copy_from_device(const GPUMemory<T> &other) {
		m_compressed = other.m_compressed;
		m_stream = other.m_stream;

		if (m_size != other.m_size) {
			resize(other.m_size);
		}

		try {
			if (m_stream) {
				CUDA_CHECK_THROW(cudaMemcpyAsync(m_data, other.m_data, m_size * sizeof(T), cudaMemcpyDeviceToDevice, m_stream));
			} else {
				CUDA_CHECK_THROW(cudaMemcpy(m_data, other.m_data, m_size * sizeof(T), cudaMemcpyDeviceToDevice));
			}
		} catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not copy from device: ") + error.what());
		}
	}

	/// Copies size elements from another device array to this one, automatically resizing it
	void copy_from_device(const GPUMemory<T> &other, const size_t size) {
		m_compressed = other.m_compressed;
		m_stream = other.m_stream;

		if (m_size < size) {
			resize(size);
		}

		try {
			if (m_stream) {
				CUDA_CHECK_THROW(cudaMemcpyAsync(m_data, other.m_data, size * sizeof(T), cudaMemcpyDeviceToDevice, m_stream));
			} else {
				CUDA_CHECK_THROW(cudaMemcpy(m_data, other.m_data, size * sizeof(T), cudaMemcpyDeviceToDevice));
			}
		} catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not copy from device: ") + error.what());
		}
	}
	/** @} */

	/** @name Data/size access
	 *  @{
	 */
	/// Returns a raw pointer of the data on the device
	inline T* data() const {
		check_guards();
		return m_data;
	}

	/// Returns the number of elements in the array
	size_t get_num_elements() const {
		return m_size;
	}

	/// Returns the number of bytes in the array
	size_t get_bytes() const {
		return m_size * sizeof(T);
	}
	/** @} */
};

using json = nlohmann::json;

inline json::binary_t gpu_memory_to_json_binary(const void* gpu_data, size_t n_bytes) {
	json::binary_t data_cpu;
	data_cpu.resize(n_bytes);
	CUDA_CHECK_THROW(cudaMemcpy(data_cpu.data(), gpu_data, n_bytes, cudaMemcpyDeviceToHost));
	return data_cpu;
}

inline void json_binary_to_gpu_memory(const json::binary_t& cpu_data, void* gpu_data, size_t n_bytes) {
	CUDA_CHECK_THROW(cudaMemcpy(gpu_data, cpu_data.data(), n_bytes, cudaMemcpyHostToDevice));
}

template <typename T>
json::binary_t gpu_memory_to_json_binary(const GPUMemory<T>& gpu_data) {
	return gpu_memory_to_json_binary(gpu_data.data(), gpu_data.get_bytes());
}

template <typename T>
void json_binary_to_gpu_memory(const json::binary_t& cpu_data, GPUMemory<T>& gpu_data) {
	gpu_data.resize(cpu_data.size()/sizeof(T));
	json_binary_to_gpu_memory(cpu_data, gpu_data.data(), gpu_data.get_bytes());
}

TCNN_NAMESPACE_END
