/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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
#include <atomic>
#include <cuda.h>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>


TCNN_NAMESPACE_BEGIN

#define DEBUG_GUARD_SIZE 0

inline std::atomic<size_t>& total_n_bytes_allocated() {
	static std::atomic<size_t> s_total_n_bytes_allocated{0};
	return s_total_n_bytes_allocated;
}

/// Managed memory on the Device
template<class T>
class GPUMemory {
private:
	T* m_data = nullptr;
	size_t m_size = 0; // Number of elements
	bool m_owned = true;

public:
	GPUMemory() {}

	GPUMemory<T>& operator=(GPUMemory<T>&& other) {
		std::swap(m_data, other.m_data);
		std::swap(m_size, other.m_size);
		return *this;
	}

	GPUMemory(GPUMemory<T>&& other) {
		*this = std::move(other);
	}

	TCNN_HOST_DEVICE GPUMemory(const GPUMemory<T> &other) : m_data{other.m_data}, m_size{other.m_size}, m_owned{false} {}

	void check_guards() const {
#if DEBUG_GUARD_SIZE > 0
		if (!m_data)
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

		uint8_t *rawptr = nullptr;
		CUDA_CHECK_THROW(cudaMalloc(&rawptr, n_bytes+DEBUG_GUARD_SIZE*2));
#if DEBUG_GUARD_SIZE > 0
		CUDA_CHECK_THROW(cudaMemset(rawptr , 0xff, DEBUG_GUARD_SIZE));
		CUDA_CHECK_THROW(cudaMemset(rawptr+n_bytes+DEBUG_GUARD_SIZE , 0xfe, DEBUG_GUARD_SIZE));
#endif
		if (rawptr) rawptr+=DEBUG_GUARD_SIZE;
		m_data=(T*)(rawptr);
		total_n_bytes_allocated() += n_bytes;
	}

	void free_memory() {
		if (!m_data) {
			return;
		}

		uint8_t *rawptr = (uint8_t*)m_data;
		if (rawptr) rawptr-=DEBUG_GUARD_SIZE;
		CUDA_CHECK_THROW(cudaFree(rawptr));

		total_n_bytes_allocated() -= get_bytes();

		m_data = nullptr;
	}

	/// Allocates memory for size items of type T
	GPUMemory(const size_t size) {
		resize(size);
	}

	/// Frees memory again
	TCNN_HOST_DEVICE ~GPUMemory() {
#ifndef __CUDA_ARCH__
		if (!m_owned) {
			return;
		}

		try {
			if (m_data) {
				free_memory();
				m_size = 0;
			}
		} catch (std::runtime_error error) {
			// Don't need to report on memory-free problems when the driver is shutting down.
			if (std::string{error.what()}.find("driver shutting down") == std::string::npos) {
				fprintf(stderr, "Could not free memory: %s\n", error.what());
			}
		}
#endif
	}

	/** @name Resizing/enlargement
	 *  @{
	 */
	/// Resizes the array to the exact new size, even if it is already larger
	void resize(const size_t size) {
		if (!m_owned) {
			throw std::runtime_error("Cannot resize non-owned memory.");
		}

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
			CUDA_CHECK_THROW(cudaMemset(m_data + offset, value, num_elements * sizeof(T)));
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
			CUDA_CHECK_THROW(cudaMemcpy(data(), host_data, num_elements * sizeof(T), cudaMemcpyHostToDevice));
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
			CUDA_CHECK_THROW(cudaMemcpy(host_data, data(), num_elements * sizeof(T), cudaMemcpyDeviceToHost));
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
		if (m_size != other.m_size) {
			resize(other.m_size);
		}

		try {
			CUDA_CHECK_THROW(cudaMemcpy(m_data, other.m_data, m_size * sizeof(T), cudaMemcpyDeviceToDevice));
		} catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not copy from device: ") + error.what());
		}
	}

	/// Copies size elements from another device array to this one, automatically resizing it
	void copy_from_device(const GPUMemory<T> &other, const size_t size) {
		if (m_size < size) {
			resize(size);
		}

		try {
			CUDA_CHECK_THROW(cudaMemcpy(m_data, other.m_data, size * sizeof(T), cudaMemcpyDeviceToDevice));
		} catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not copy from device: ") + error.what());
		}
	}

	// Created an (owned) copy of the data
	GPUMemory<T> copy() const {
		GPUMemory<T> result{m_size};
		result.copy_from_device(*this);
		return result;
	}

	T* data() const {
		check_guards();
		return m_data;
	}

	TCNN_HOST_DEVICE T& operator[](size_t idx) const {
#ifdef DEBUG_BUFFER_OVERRUN
		if (idx > m_size) {
			printf("WARNING: buffer overrun of %p at idx %zu\n", idx);
		}
#endif
		return m_data[idx];
	}

	TCNN_HOST_DEVICE T& operator[](uint32_t idx) const {
#ifdef DEBUG_BUFFER_OVERRUN
		if (idx > m_size) {
			printf("WARNING: buffer overrun of %p at idx %u\n", idx);
		}
#endif
		return m_data[idx];
	}

	size_t get_num_elements() const {
		return m_size;
	}

	size_t size() const {
		return get_num_elements();
	}

	size_t get_bytes() const {
		return m_size * sizeof(T);
	}

	size_t bytes() const {
		return get_bytes();
	}
};

TCNN_NAMESPACE_END
