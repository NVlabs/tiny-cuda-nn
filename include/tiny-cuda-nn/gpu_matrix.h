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

/** @file   gpu_matrix.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Matrix whose data resides in GPU (CUDA) memory
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/matrix_layout.h>

#include <pcg32/pcg32.h>

#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>


TCNN_NAMESPACE_BEGIN

template<typename T>
class GPUMatrixDynamic;

template<typename T, MatrixLayout _layout>
class GPUMatrix;

class GPUMatrixBase {
public:
	virtual ~GPUMatrixBase() {}

	virtual size_t n_bytes() const = 0;
	virtual void set_data(void* data) = 0;

	static void allocate_shared_memory(GPUMemory<char>& memory, const std::vector<GPUMatrixBase*>& matrices) {
		size_t total_n_bytes = 0;
		for (auto* matrix : matrices) {
			total_n_bytes += matrix->n_bytes();
		}

		if (memory.bytes() < total_n_bytes) {
#ifdef TCNN_VERBOSE_MEMORY_ALLOCS
			std::cout << "GPUMatrix: Allocating " << bytes_to_string(total_n_bytes) << " shared among " << matrices.size() << " matrices." << std::endl;
#endif
			memory.resize(total_n_bytes);
		}

		size_t offset = 0;
		for (auto* matrix : matrices) {
			matrix->set_data(memory.data() + offset);
			offset += matrix->n_bytes();
		}
	}

	template <typename T>
	static void allocate_shared_memory(GPUMemory<char>& memory, std::vector<GPUMatrixDynamic<T>>& matrices);

	template <typename T, MatrixLayout layout>
	static void allocate_shared_memory(GPUMemory<char>& memory, std::vector<GPUMatrix<T, layout>>& matrices);
};

template <typename T>
class GPUMatrixDynamic : public GPUMatrixBase {
public:
	using Type = T;

	// Owning its memory
	GPUMatrixDynamic(uint32_t m, uint32_t n, MatrixLayout layout = CM)
	: m_owned_data{m * n}, m_rows{m}, m_cols{n}, m_layout{layout} {
		m_data = m_owned_data.data();
	}

	// Pointing to external memory
	explicit GPUMatrixDynamic(T* data, uint32_t m, uint32_t n, MatrixLayout layout = CM)
	: m_data{data}, m_rows{m}, m_cols{n}, m_layout{layout} {
	}

	GPUMatrixDynamic() : GPUMatrixDynamic(nullptr, 0, 0) {}

	GPUMatrixDynamic(GPUMatrixDynamic<T>&& other) : m_data{other.m_data}, m_rows{other.m_rows}, m_cols{other.m_cols}, m_layout{other.m_layout}, m_owned_data{std::move(other.m_owned_data)} { }
	explicit GPUMatrixDynamic(const GPUMatrixDynamic<T>& other) : m_data{other.m_data}, m_rows{other.m_rows}, m_cols{other.m_cols}, m_layout{other.m_layout}, m_owned_data{other.m_owned_data.copy()} {
		// If we just copied over some owned data, then we want to point to our copy
		if (m_owned_data.data()) {
			m_data = m_owned_data.data();
		}
	}

	virtual ~GPUMatrixDynamic() {}

	void set_data(void* data) override { m_data = (T*)data; }
	void set_size(uint32_t rows, uint32_t cols) {
		m_rows = rows;
		m_cols = cols;
	}

	uint32_t rows() const { return m_rows; }
	uint32_t fan_out() const { return m_rows; }
	uint32_t m() const { return m_rows; }

	uint32_t cols() const { return m_cols; }
	uint32_t fan_in() const { return m_cols; }
	uint32_t n() const { return m_cols; }

	uint32_t n_elements() const { return m_rows * m_cols; }
	size_t n_bytes() const override { return n_elements() * sizeof(T); }

	MatrixLayout layout() const { return m_layout; }
	MatrixLayout transposed_layout() const { return m_layout == RM ? CM : RM; }

	T* data() { return m_data; }
	const T* data() const { return m_data; }

	void memset(int value) {
		CUDA_CHECK_THROW(cudaMemset(m_data, value, n_elements() * sizeof(T)));
	}

	void memset_async(cudaStream_t stream, int value) {
		CUDA_CHECK_THROW(cudaMemsetAsync(m_data, value, n_elements() * sizeof(T), stream));
	}

	// Various initializations
	void initialize_xavier_uniform(pcg32& rnd, float scale = 1) {
		if (!data()) {
			throw std::runtime_error("Matrix has no allocated data");
		}

		// Define probability distribution
		scale *= std::sqrt(6.0f / (float)(fan_in() + fan_out()));

		// Sample initialized values
		std::vector<T> new_data(n_elements());

		for (size_t i = 0; i < new_data.size(); ++i) {
			new_data[i] = (T)(rnd.next_float() * 2.0f * scale - scale);
		}

		CUDA_CHECK_THROW(cudaMemcpy(data(), new_data.data(), n_elements() * sizeof(T), cudaMemcpyHostToDevice));
	}

	void initialize_fa_uniform_forward(pcg32& rnd, float scale = 1) {
		if (!data()) {
			throw std::runtime_error("Matrix has no allocated data");
		}

		// Define probability distribution
		scale *= std::sqrt(1.0f / (float)fan_in());

		// Sample initialized values
		std::vector<T> new_data(n_elements());

		for (size_t i = 0; i < new_data.size(); ++i) {
			new_data[i] = (T)(rnd.next_float() * 2.0f * scale - scale);
		}

		CUDA_CHECK_THROW(cudaMemcpy(data(), new_data.data(), n_elements() * sizeof(T), cudaMemcpyHostToDevice));
	}

	void initialize_fa_uniform_backward(pcg32& rnd, float scale = 1) {
		if (!data()) {
			throw std::runtime_error("Matrix has no allocated data");
		}

		// Define probability distribution
		scale *= std::sqrt(1.0f / (float)fan_out());

		// Sample initialized values
		std::vector<T> new_data(n_elements());

		for (size_t i = 0; i < new_data.size(); ++i) {
			new_data[i] = (T)(rnd.next_float() * 2.0f * scale - scale);
		}

		CUDA_CHECK_THROW(cudaMemcpy(data(), new_data.data(), n_elements() * sizeof(T), cudaMemcpyHostToDevice));
	}

	void initialize_siren_uniform(pcg32& rnd, float scale = 1) {
		if (!data()) {
			throw std::runtime_error("Matrix has no allocated data");
		}

		// Define probability distribution
		scale *= std::sqrt(6.0f / (float)fan_in());

		// Sample initialized values
		std::vector<T> new_data(n_elements());

		for (size_t i = 0; i < new_data.size(); ++i) {
			new_data[i] = (T)(rnd.next_float() * 2.0f * scale - scale);
		}

		CUDA_CHECK_THROW(cudaMemcpy(data(), new_data.data(), n_elements() * sizeof(T), cudaMemcpyHostToDevice));
	}

	void initialize_siren_uniform_first(pcg32& rnd, float scale = 1) {
		if (!data()) {
			throw std::runtime_error("Matrix has no allocated data");
		}

		// Define probability distribution

		// The 30 in the first layer comes from https://vsitzmann.github.io/siren/
		scale *= 30.0f / (float)fan_in();

		// Sample initialized values
		std::vector<T> new_data(n_elements());

		for (size_t i = 0; i < new_data.size(); ++i) {
			new_data[i] = (T)(rnd.next_float() * 2.0f * scale - scale);
		}

		CUDA_CHECK_THROW(cudaMemcpy(data(), new_data.data(), n_elements() * sizeof(T), cudaMemcpyHostToDevice));
	}

	void initialize_constant(float val) {
		if (!data()) {
			throw std::runtime_error("Matrix has no allocated data");
		}

		std::vector<T> new_data(n_elements(), (T)val);
		CUDA_CHECK_THROW(cudaMemcpy(data(), new_data.data(), n_elements() * sizeof(T), cudaMemcpyHostToDevice));
	}

	void initialize_diagonal(float val = 1) {
		if (!data()) {
			throw std::runtime_error("Matrix has no allocated data");
		}

		if (n() != m()) {
			throw std::runtime_error("Can only perform diagonal initialization if the matrix is square");
		}

		std::vector<T> new_data(n_elements(), (T)0);
		for (uint32_t i = 0; i < n(); ++i) {
			new_data[i + i*n()] = (T)val;
		}

		CUDA_CHECK_THROW(cudaMemcpy(data(), new_data.data(), n_elements() * sizeof(T), cudaMemcpyHostToDevice));
	}

	const GPUMatrixDynamic<T> transposed() const {
		return std::move(GPUMatrixDynamic<T>(m_data, n(), m(), transposed_layout()));
	}

	GPUMatrixDynamic<T> transposed() {
		return std::move(GPUMatrixDynamic<T>(m_data, n(), m(), transposed_layout()));
	}

	const GPUMatrixDynamic<T> with_opposite_layout() const {
		return std::move(GPUMatrixDynamic<T>(m_data, m(), n(), transposed_layout()));
	}

	GPUMatrixDynamic<T> with_opposite_layout() {
		return std::move(GPUMatrixDynamic<T>(m_data, m(), n(), transposed_layout()));
	}

	virtual void set_layout(MatrixLayout layout) {
		m_layout = layout;
	}

private:
	T* m_data;
	uint32_t m_rows, m_cols;
	MatrixLayout m_layout;
	GPUMemory<T> m_owned_data;
};

template <typename T, MatrixLayout _layout = MatrixLayout::ColumnMajor>
class GPUMatrix : public GPUMatrixDynamic<T> {
public:
	static const MatrixLayout static_layout = _layout;
	static const MatrixLayout static_transposed_layout = _layout == RM ? CM : RM;

	// Owning its memory
	GPUMatrix(uint32_t m, uint32_t n)
	: GPUMatrixDynamic<T>{m, n, static_layout} { }

	// Pointing to external memory
	explicit GPUMatrix(T* data, uint32_t m, uint32_t n)
	: GPUMatrixDynamic<T>{data, m, n, static_layout} { }

	GPUMatrix() : GPUMatrix(nullptr, 0, 0) {}

	GPUMatrix(GPUMatrixDynamic<T>&& other)
	: GPUMatrixDynamic<T>{other} {
		if (static_layout != this->layout()) {
			throw std::runtime_error{"GPUMatrix must be constructed from a GPUMatrixDynamic with matching layout."};
		}
	}

	// Only copy by reference. This is to prevent accidental deep copies of owned data.
	explicit GPUMatrix(const GPUMatrixDynamic<T>& other)
	: GPUMatrixDynamic<T>{const_cast<T*>(other.data()), other.rows(), other.cols(), other.layout()} {
		if (static_layout != this->layout()) {
			throw std::runtime_error{"GPUMatrix must be constructed from a GPUMatrixDynamic with matching layout."};
		}
	}

	virtual ~GPUMatrix() {}

	void set_layout(MatrixLayout layout) override {
		throw std::runtime_error{"Cannot set the layout of a GPUMatrix at runtime (it is determined by the type system). Use GPUMatrixDynamic instead."};
	}

	const GPUMatrix<T, static_transposed_layout> transposed() const {
		return std::move(GPUMatrix<T, static_transposed_layout>(const_cast<T*>(this->data()), this->n(), this->m()));
	}

	GPUMatrix<T, static_transposed_layout> transposed() {
		return std::move(GPUMatrix<T, static_transposed_layout>(this->data(), this->n(), this->m()));
	}

	const GPUMatrix<T, static_transposed_layout> with_opposite_layout() const {
		return std::move(GPUMatrix<T, static_transposed_layout>(const_cast<T*>(this->data()), this->m(), this->n()));
	}

	GPUMatrix<T, static_transposed_layout> with_opposite_layout() {
		return std::move(GPUMatrix<T, static_transposed_layout>(this->data(), this->m(), this->n()));
	}
};

template <typename T>
void GPUMatrixBase::allocate_shared_memory(GPUMemory<char>& memory, std::vector<GPUMatrixDynamic<T>>& matrices) {
	std::vector<GPUMatrixBase*> matrix_pointers;
	for (auto& matrix : matrices) {
		matrix_pointers.emplace_back(&matrix);
	}
	allocate_shared_memory(memory, matrix_pointers);
}

template <typename T, MatrixLayout layout>
void GPUMatrixBase::allocate_shared_memory(GPUMemory<char>& memory, std::vector<GPUMatrix<T, layout>>& matrices) {
	std::vector<GPUMatrixBase*> matrix_pointers;
	for (auto& matrix : matrices) {
		matrix_pointers.emplace_back(&matrix);
	}
	allocate_shared_memory(memory, matrix_pointers);
}

TCNN_NAMESPACE_END
