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

/** @file   shampoo.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the Shampoo optimizer [Gupta et al. 2018].
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/cuda_graph.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/reduce_sum.h>

#include <mma.h>
#include <cublas_v2.h>

#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

inline std::string cublasGetError(cublasStatus_t error) {
	switch (error) {
		case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
		case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
		default: return "<unknown>";
	}
}

#define CUBLAS_CHECK_THROW(x)                                                                                           \
	do {                                                                                                                \
		cublasStatus_t result = x;                                                                                      \
		if (result != CUBLAS_STATUS_SUCCESS)                                                                            \
			throw std::runtime_error(std::string("CUBLAS Error: " #x " failed with error ") + cublasGetError(result));  \
	} while(0)

template <typename T>
__global__ void subtract(
	const uint32_t n_elements,
	const T* arg1,
	const T* arg2,
	T* out,
	float scale = 1
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	out[i] = (arg1[i] - arg2[i]) * (T)scale;
}

template <typename T>
__global__ void set_identity(
	const uint32_t n_elements,
	const uint32_t M,
	T* __restrict__ output,
	float scale,
	const uint32_t n_matrices = 1
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t n_elements_matrix = M*M;

	const uint32_t idx = i % n_elements_matrix;

	const uint32_t row = idx / M;
	const uint32_t col = idx % M;

	const uint32_t idx_T = col * M + row;

	output[i] = idx == idx_T ? (T)scale : (T)0;
}

template <typename T, typename F>
__global__ void set_identity(
	const uint32_t n_elements,
	const uint32_t M,
	T* __restrict__ output,
	T* scale,
	F fun,
	const uint32_t n_matrices = 1
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t n_elements_matrix = M*M;

	const uint32_t matrix_idx = i / n_elements_matrix;
	const uint32_t idx = i % n_elements_matrix;

	const uint32_t row = idx / M;
	const uint32_t col = idx % M;

	const uint32_t idx_T = col * M + row;

	output[i] = idx == idx_T ? fun(scale[matrix_idx]) : (T)0;
}

template <typename T>
__global__ void set_matrix(
	const uint32_t n_elements,
	T* output,
	const T* input,
	float scale,
	const uint32_t n_matrices = 1
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	output[i] = input[i] * (T)scale;
}

template <typename T, typename F>
__global__ void set_matrix(
	const uint32_t n_elements,
	const uint32_t n_elements_matrix,
	T* output,
	const T* input,
	T* scale,
	F fun,
	const uint32_t n_matrices = 1
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t matrix_idx = i / n_elements_matrix;

	output[i] = input[i] * fun(scale[matrix_idx]);
}


template <typename T>
__global__ void shampoo_momentum_update_batched(
	const uint32_t n_elements,
	const float loss_scale,
	const float alpha1,
	const float beta1,
	const float alpha2,
	const float beta2,
	const float epsilon,
	const float l2,
	const float* __restrict__ weights_full_precision,
	const T* __restrict__ gradients,
	float* __restrict__ first_moments,
	float* __restrict__ second_moments,
	float* __restrict__ momentum
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const float gradient = (float)gradients[i] / loss_scale + l2 * weights_full_precision[i];
	float first_moment = first_moments[i] = beta1 * first_moments[i] + alpha1 * gradient;

	const float gradient_sq = gradient * gradient;
	float second_moment = second_moments[i] = beta2 * second_moments[i] + alpha2 * gradient_sq;

	momentum[i] = first_moment / (sqrtf(second_moment) + epsilon);
}


__global__ void shampoo_symmetrize_batched(
	const uint32_t M,
	const uint32_t n_matrices,
	const float identity_strength,
	const float* __restrict__ input,
	float* __restrict__ output
) {
	const uint32_t n_elements_per_matrix = M*M;
	const uint32_t n_elements = n_elements_per_matrix * n_matrices;
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t idx = i % n_elements_per_matrix;
	const uint32_t matrix_idx = i / n_elements_per_matrix;

	const uint32_t row = idx / M;
	const uint32_t col = idx % M;

	const uint32_t idx_T = col * M + row;

	float val = 0.5 * (input[i] + input[idx_T + matrix_idx * n_elements_per_matrix]) * (1 - identity_strength);
	if (idx == idx_T) {
		val += identity_strength;
	}

	output[i] = val;
}


template <typename T>
__global__ void shampoo_step_batched(
	const uint32_t M,
	const uint32_t N,
	const uint32_t n_matrices,
	const float relative_weight_decay,
	const float absolute_weight_decay,
	float learning_rate,
	const float* __restrict__ shampoo_norm,
	const float* __restrict__ adam_norm,
	const float* __restrict__ update_buffer,
	float* __restrict__ weights_full_precision,
	T* __restrict__ weights
) {
	const uint32_t n_elements_per_matrix = M*N;
	const uint32_t n_elements = n_elements_per_matrix*n_matrices;
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t idx = i % n_elements_per_matrix;
	const uint32_t matrix_idx = i / n_elements_per_matrix;

	const uint32_t base_idx = matrix_idx * n_elements_per_matrix;

	const float adam_norm_local = adam_norm[matrix_idx];
	const float shampoo_norm_local = shampoo_norm[matrix_idx];

	learning_rate *= sqrtf(adam_norm_local) / sqrtf(shampoo_norm_local);

	const uint32_t row = idx / N;
	const uint32_t col = idx % N;

	const float decayed_weight = weight_decay(relative_weight_decay * learning_rate, absolute_weight_decay * learning_rate, weights_full_precision[i]);
	const float new_weight = decayed_weight - learning_rate * update_buffer[col * M + row + base_idx];

	weights_full_precision[i] = new_weight;
	weights[i] = (T)new_weight;
}


template <typename T>
__global__ void shampoo_step_remaining(
	const uint32_t n_elements,
	const float relative_weight_decay,
	const float absolute_weight_decay,
	float learning_rate,
	const float* __restrict__ momentum,
	float* __restrict__ weights_full_precision,
	T* __restrict__ weights
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const float decayed_weight = weight_decay(relative_weight_decay * learning_rate, absolute_weight_decay * learning_rate, weights_full_precision[i]);
	const float new_weight = decayed_weight - learning_rate * momentum[i];

	weights_full_precision[i] = new_weight;
	weights[i] = (T)new_weight;
}


template <typename T>
class ShampooOptimizer : public Optimizer<T> {
public:
	// using ROOT_TYPE = double;
	using ROOT_TYPE = float;

	ShampooOptimizer(const json& params) {
		update_hyperparams(params);

		CUBLAS_CHECK_THROW(cublasCreate(&m_cublas));
		CUDA_CHECK_THROW(cudaEventCreate(&m_global_event));

		cublasSetPointerMode(m_cublas, CUBLAS_POINTER_MODE_DEVICE);
	}

	~ShampooOptimizer() {
		// cublasDestroy(m_cublas);

		for (size_t i = 0; i < m_streams.size(); ++i) {
			CUDA_CHECK_PRINT(cudaEventDestroy(m_events[i]));
			CUDA_CHECK_PRINT(cudaStreamDestroy(m_streams[i]));
		}

		cudaEventDestroy(m_global_event);
	}

	std::pair<float, float> debiased_alpha_beta(float decay) const {
		float alpha = 1 - decay;
		float beta = decay;

		// de-biasing
		float debias = 1 - std::pow(decay, (float)m_current_step+1);
		alpha /= debias;
		beta *= (1 - std::pow(decay, (float)m_current_step)) / debias;

		return {alpha, beta};
	}

	void allocate(uint32_t n_weights, const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
		m_n_weights = n_weights;

		if (m_n_weights <= m_first_moments.size()) {
			return;
		}

		m_coefficients.resize(1024); // Conservatively too big
		m_coefficients.memset(0);

		float* it = m_coefficients.data();

		m_one = it++;
		m_zero = it++;
		m_alpha_1 = it++;
		m_beta_1 = it++;
		m_alpha_2 = it++;
		m_beta_2 = it++;
		m_alpha_3 = it++;
		m_beta_3 = it++;
		m_alpha_shampoo_coef = it++;
		m_beta_shampoo_coef = it++;

		m_coefficients_root.enlarge_and_copy_from_host({1.0, 0.0});
		m_one_root = m_coefficients_root.data() + 0;
		m_zero_root = m_coefficients_root.data() + 1;

		m_first_moments.resize(m_n_weights);
		m_first_moments.memset(0);

		m_second_moments.resize(m_n_weights);
		m_second_moments.memset(0);

		m_momentum.resize(m_n_weights);
		m_momentum.memset(0);

		m_shampoo_momentum.resize(m_n_weights);
		m_shampoo_momentum.memset(0);

		uint32_t total_M = 0;
		uint32_t total_MM = 0;
		uint32_t total_N = 0;
		uint32_t total_NN = 0;
		uint32_t total_MN = 0;

		std::vector<GPUMatrixBase*> matrices;

		m_matrix_batches.clear();

		std::pair<uint32_t, uint32_t> current_size = layer_sizes.front();
		size_t current_idx = 0;

		for (size_t i = 0; i < layer_sizes.size(); ++i) {
			const auto& pair = layer_sizes[i];

			m_L.emplace_back(nullptr, pair.first, pair.first);
			m_L_root.emplace_back(nullptr, pair.first, pair.first);
			m_R.emplace_back(nullptr, pair.second, pair.second);
			m_R_root.emplace_back(nullptr, pair.second, pair.second);

			total_M += pair.first;
			total_MM += pair.first * pair.first;
			total_N += pair.second;
			total_NN += pair.second * pair.second;
			total_MN += pair.first * pair.second;

			if (current_size.first != pair.first || current_size.second != pair.second) {
				m_matrix_batches.emplace_back(current_idx, i);
				current_idx = i;
				current_size = pair;
			}
		}
		m_n_weights_covered_by_matrices = total_MN;
		m_matrix_batches.emplace_back(current_idx, layer_sizes.size());

		GPUMatrixBase::allocate_shared_memory(m_L_buffer, m_L);
		GPUMatrixBase::allocate_shared_memory(m_L_root_buffer, m_L_root);
		GPUMatrixBase::allocate_shared_memory(m_R_buffer, m_R);
		GPUMatrixBase::allocate_shared_memory(m_R_root_buffer, m_R_root);

		m_L_buffer.memset(0);
		m_R_buffer.memset(0);

		m_gradient_tmp.resize(total_MN);

		std::vector<float> ones(layer_sizes.size(), 1.0f);
		m_sqr1_tmp.enlarge_and_copy_from_host(ones);
		m_sqr2_tmp.enlarge_and_copy_from_host(ones);

		m_shampoo_update.resize(total_MN);
		m_adam_update.resize(total_MN);

		m_inverse_pth_root_buffers.resize(m_matrix_batches.size());

		for (size_t i = 0; i < m_streams.size(); ++i) {
			CUDA_CHECK_PRINT(cudaEventDestroy(m_events[i]));
			CUDA_CHECK_PRINT(cudaStreamDestroy(m_streams[i]));
		}

		m_streams.resize(m_matrix_batches.size() * 3);
		m_events.resize(m_matrix_batches.size() * 3);
		m_cublas_workspaces.resize(m_matrix_batches.size() * 3);

		for (size_t i = 0; i < m_streams.size(); ++i) {
			CUDA_CHECK_THROW(cudaStreamCreate(&m_streams[i]));
			CUDA_CHECK_THROW(cudaEventCreate(&m_events[i]));

			m_cublas_workspaces[i].resize(16 * 1024 * 1024); // 16 MiB
		}
	}

	template <typename ROOT_TYPE>
	void inverse_pth_root_batched(cudaStream_t stream, uint32_t M, float* data, GPUMemory<ROOT_TYPE>& tmp, uint32_t n_matrices, uint32_t idx) {
		CUBLAS_CHECK_THROW(cublasSetStream(m_cublas, stream));

		uint32_t n_elements = M*M;
		uint32_t workspace_size = n_elements * 6 * n_matrices;
		if (tmp.size() < workspace_size) {
			tmp.resize(workspace_size * 2);
		}

		ROOT_TYPE* Xk;
		if (std::is_same<ROOT_TYPE, float>::value) {
			Xk = (ROOT_TYPE*)data;
		} else {
			Xk = tmp.data() + n_elements * n_matrices * 5;
			linear_kernel(cast<ROOT_TYPE>, 0, stream, n_elements*n_matrices, data, Xk);
		}

		ROOT_TYPE* Mk      = tmp.data() + n_elements * n_matrices * 0;
		ROOT_TYPE* tmp1    = tmp.data() + n_elements * n_matrices * 1;
		ROOT_TYPE* tmp2    = tmp.data() + n_elements * n_matrices * 2;
		ROOT_TYPE* I5      = tmp.data() + n_elements * n_matrices * 3;
		ROOT_TYPE* sum_tmp = tmp.data() + n_elements * n_matrices * 4;

		cudaDataType_t dataType;
		cublasComputeType_t computeType;

		if (std::is_same<ROOT_TYPE, float>::value) {
			dataType = CUDA_R_32F;
			computeType = CUBLAS_COMPUTE_32F;
		} else if (std::is_same<ROOT_TYPE, double>::value) {
			dataType = CUDA_R_64F;
			computeType = CUBLAS_COMPUTE_64F;
		}

		// Compute c following section 3.2 of the paper http://eprints.ma.man.ac.uk/637/1/covered/MIMS_ep2005_9.pdf
		{
			// To upper bound the spectral radius of A, the authors propose using the matrix norm.
			// We get a tighter upper bound at the cost of 2 matrix multiplications by using
			//   lim k->inf (|A^k|^{1/k}) = spectral_radius(A)
			// k=4 seems to give a reasonable amount of numerical stability and accuracy.

			// A^2
			CUBLAS_CHECK_THROW(cublasGemmStridedBatchedEx(
				m_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
				M, M, M,
				m_one_root,
				Xk, dataType, M, n_elements,
				Xk, dataType, M, n_elements,
				m_zero_root,
				tmp1, dataType, M, n_elements,
				n_matrices,
				computeType,
				CUBLAS_GEMM_DEFAULT
			));

			// A^4
			CUBLAS_CHECK_THROW(cublasGemmStridedBatchedEx(
				m_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
				M, M, M,
				m_one_root,
				tmp1, dataType, M, n_elements,
				tmp1, dataType, M, n_elements,
				m_zero_root,
				tmp2, dataType, M, n_elements,
				n_matrices,
				computeType,
				CUBLAS_GEMM_DEFAULT
			));

			// Squared norm + 4th root of that
			cudaMemsetAsync(sum_tmp, 0, n_matrices * sizeof(ROOT_TYPE), stream);
			reduce_sum(tmp2, [] __device__ (ROOT_TYPE val) { return val * val; }, sum_tmp, n_elements, stream, n_matrices);
		}

		set_matrix<<<n_blocks_linear(n_elements*n_matrices), n_threads_linear, 0, stream>>>(n_elements*n_matrices, n_elements, Mk, Xk, sum_tmp, [] __device__ (ROOT_TYPE c) {
			return std::sqrt((ROOT_TYPE)2.0) / std::pow(c, (ROOT_TYPE)(0.5 * 0.25));
		}, n_matrices);
		set_identity<<<n_blocks_linear(n_elements*n_matrices), n_threads_linear, 0, stream>>>(n_elements*n_matrices, M, Xk, sum_tmp, [] __device__ (ROOT_TYPE c) {
			return std::pow(std::sqrt((ROOT_TYPE)2.0) / std::pow(c, (ROOT_TYPE)(0.5 * 0.25)), (ROOT_TYPE)0.25);
		}, n_matrices);
		linear_kernel(set_identity<ROOT_TYPE>, 0, stream, n_elements*n_matrices, M, I5, 5.0f, n_matrices);

		// The iterations have the form
		// X_{k+1} = X_k ( (5 I - M_k) / 4 ) , where X_0 = 1/c * I
		// M_{k+1} = ( (5 I - M_k) / 4 )^4 M_k , where M_0 = 1/c^p * A

		// tmp1 = (5I - Mk) / 4
		linear_kernel(subtract<ROOT_TYPE>, 0, stream, n_elements*n_matrices, I5, Mk, tmp1, 0.25f);

		// Xk+1 (one indirect copy to prevent the need for in-place operations)
		linear_kernel(set_matrix<ROOT_TYPE>, 0, stream, n_elements*n_matrices, tmp2, Xk, 1.0f, n_matrices);
		CUBLAS_CHECK_THROW(cublasGemmStridedBatchedEx(
			m_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
			M, M, M,
			m_one_root,
			tmp2, dataType, M, n_elements,
			tmp1, dataType, M, n_elements,
			m_zero_root,
			Xk, dataType, M, n_elements,
			n_matrices,
			computeType,
			CUBLAS_GEMM_DEFAULT
		));

		// Only check every couple of iterations whether we're converging...
		// The check is expensive due to the CPU synchronization.
		// 10 iterations appear to be sufficient most of the time.
		static const uint32_t CHECK_INTERVAL = 5;

		std::vector<ROOT_TYPE> delta(n_matrices, std::numeric_limits<ROOT_TYPE>::infinity());
		ROOT_TYPE epsilon = (ROOT_TYPE)1e-20;

		int i = 0;
		while (true) {
			for (int j = 0; j < CHECK_INTERVAL; ++j, ++i) {
				// tmp1^2
				CUBLAS_CHECK_THROW(cublasGemmStridedBatchedEx(
					m_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
					M, M, M,
					m_one_root,
					tmp1, dataType, M, n_elements,
					tmp1, dataType, M, n_elements,
					m_zero_root,
					tmp2, dataType, M, n_elements,
					n_matrices,
					computeType,
					CUBLAS_GEMM_DEFAULT
				));

				// tmp2^2
				CUBLAS_CHECK_THROW(cublasGemmStridedBatchedEx(
					m_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
					M, M, M,
					m_one_root,
					tmp2, dataType, M, n_elements,
					tmp2, dataType, M, n_elements,
					m_zero_root,
					tmp1, dataType, M, n_elements,
					n_matrices,
					computeType,
					CUBLAS_GEMM_DEFAULT
				));

				// Mk+1 (one indirect copy to prevent the need for in-place operations)
				linear_kernel(set_matrix<ROOT_TYPE>, 0, stream, n_elements*n_matrices, tmp2, Mk, 1.0f, n_matrices);
				CUBLAS_CHECK_THROW(cublasGemmStridedBatchedEx(
					m_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
					M, M, M,
					m_one_root,
					tmp1, dataType, M, n_elements,
					tmp2, dataType, M, n_elements,
					m_zero_root,
					Mk, dataType, M, n_elements,
					n_matrices,
					computeType,
					CUBLAS_GEMM_DEFAULT
				));

				// tmp1 = (5I - Mk) / 4
				linear_kernel(subtract<ROOT_TYPE>, 0, stream, n_elements*n_matrices, I5, Mk, tmp1, 0.25f);

				// Xk+1 (one indirect copy to prevent the need for in-place operations)
				linear_kernel(set_matrix<ROOT_TYPE>, 0, stream, n_elements*n_matrices, tmp2, Xk, 1.0f, n_matrices);
				CUBLAS_CHECK_THROW(cublasGemmStridedBatchedEx(
					m_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
					M, M, M,
					m_one_root,
					tmp2, dataType, M, n_elements,
					tmp1, dataType, M, n_elements,
					m_zero_root,
					Xk, dataType, M, n_elements,
					n_matrices,
					computeType,
					CUBLAS_GEMM_DEFAULT
				));
			}

			linear_kernel(subtract<ROOT_TYPE>, 0, stream, n_elements*n_matrices, Xk, tmp2, tmp2, 1.0f);

			CUDA_CHECK_THROW(cudaMemsetAsync(sum_tmp, 0, n_matrices * sizeof(ROOT_TYPE), stream));
			reduce_sum(tmp2, [] __device__ (ROOT_TYPE val) { return val * val; }, sum_tmp, n_elements, stream, n_matrices);

			CUDA_CHECK_THROW(cudaMemcpyAsync(delta.data(), sum_tmp, n_matrices * sizeof(ROOT_TYPE), cudaMemcpyDeviceToHost, stream));

			if (std::any_of(std::begin(delta), std::end(delta), [](ROOT_TYPE v) { return !std::isfinite(v); })) {
				std::cout << "Failed to converge! " << delta[0] << std::endl;
				break;
			} else if (std::all_of(std::begin(delta), std::end(delta), [epsilon](ROOT_TYPE v) { return v < epsilon; })) {
				// Converged after i steps.
				break;
			}
		}

		if (!std::is_same<ROOT_TYPE, float>::value) {
			linear_kernel(cast_from<ROOT_TYPE>, 0, stream, n_elements, Xk, data);
		}
	}

	void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
		auto alpha_beta_1 = debiased_alpha_beta(m_beta1);
		auto alpha_beta_2 = debiased_alpha_beta(m_beta2);
		auto alpha_beta_3 = debiased_alpha_beta(m_beta3);
		auto alpha_beta_shampoo = debiased_alpha_beta(m_beta_shampoo);

		if (!m_cg_on_momentum) {
			alpha_beta_3.first /= loss_scale * loss_scale;
		}

		++m_current_step;

		std::vector<float> coefs;
		coefs.push_back(1.0f);
		coefs.push_back(0.0f);
		coefs.push_back(alpha_beta_1.first);
		coefs.push_back(alpha_beta_1.second);
		coefs.push_back(alpha_beta_2.first);
		coefs.push_back(alpha_beta_2.second);
		coefs.push_back(alpha_beta_3.first);
		coefs.push_back(alpha_beta_3.second);
		coefs.push_back(alpha_beta_shampoo.first);
		coefs.push_back(alpha_beta_shampoo.second);

		{
			// CUDA graph capture only if not the first optimization step (in which some synchronous work needs to happen)
			auto capture_guard = (m_current_step-1 == 0) ? ScopeGuard{} : m_graph.capture_guard(stream);

			linear_kernel(shampoo_momentum_update_batched<T>, 0, stream,
				m_n_weights,
				loss_scale,
				alpha_beta_1.first, alpha_beta_1.second,
				alpha_beta_2.first, alpha_beta_2.second,
				m_epsilon,
				m_l2_reg, weights_full_precision, gradients,
				m_first_moments.data(), m_second_moments.data(), m_momentum.data()
			);

			CUDA_CHECK_THROW(cudaMemcpyAsync(m_coefficients.data(), coefs.data(), coefs.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

			if (m_frobenius_normalization) {
				cudaMemsetAsync(m_sqr1_tmp.data(), 0, m_sqr1_tmp.size() * sizeof(float), stream);
				cudaMemsetAsync(m_sqr2_tmp.data(), 0, m_sqr2_tmp.size() * sizeof(float), stream);
			}

			cudaEventRecord(m_global_event, stream);

			size_t offset_MN = 0;
			size_t offset_M = 0;
			size_t offset_MM = 0;
			size_t offset_N = 0;
			size_t offset_NN = 0;
			for (size_t j = 0; j < m_matrix_batches.size(); ++j) {
				auto interval = m_matrix_batches[j];

				cudaStream_t update_stream = m_streams[j*3+0];
				cudaStream_t L_stream = m_streams[j*3+1];
				cudaStream_t R_stream = m_streams[j*3+2];
				cudaEvent_t update_event = m_events[j*3+0];
				cudaEvent_t L_event = m_events[j*3+1];
				cudaEvent_t R_event = m_events[j*3+2];

				GPUMemory<char>& update_workspace = m_cublas_workspaces[j*3+0];
				GPUMemory<char>& L_workspace = m_cublas_workspaces[j*3+1];
				GPUMemory<char>& R_workspace = m_cublas_workspaces[j*3+2];

				cudaStreamWaitEvent(update_stream, m_global_event, 0);
				cudaStreamWaitEvent(L_stream, m_global_event, 0);
				cudaStreamWaitEvent(R_stream, m_global_event, 0);

				// const GPUMatrix<T, RM> gradient_matrix(gradients + offset_MN, m_L[i].n(), m_R[i].n());

				uint32_t n_matrices = (uint32_t)(interval.second - interval.first);

				uint32_t M = m_L[interval.first].n();
				uint32_t N = m_R[interval.first].n();

				uint32_t gradient_stride = M * N;
				uint32_t L_stride = M * M;
				uint32_t R_stride = N * N;

				float* L_begin = m_L[interval.first].data();
				float* R_begin = m_R[interval.first].data();

				float* L_root_begin = m_L_root[interval.first].data();
				float* R_root_begin = m_R_root[interval.first].data();

				cudaDataType_t data_type = (m_cg_on_momentum || std::is_same<T, float>::value) ? CUDA_R_32F : CUDA_R_16F;
				void* gradient_pointer = m_cg_on_momentum ? (void*)(m_momentum.data() + offset_MN) : (void*)(gradients + offset_MN);
				cublasComputeType_t compute_type = m_cg_on_momentum ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

				// m_L[i] = m_beta2 * m_L[i] + (1 - m_beta2) * gradient_matrix * gradient_matrix.transpose();
				CUBLAS_CHECK_THROW(cublasSetStream(m_cublas, L_stream));
				CUBLAS_CHECK_THROW(cublasSetWorkspace(m_cublas, L_workspace.data(), L_workspace.size()));
				CUBLAS_CHECK_THROW(cublasGemmStridedBatchedEx(
					m_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
					M, M, N,
					m_alpha_3,
					gradient_pointer, data_type, N, gradient_stride,
					gradient_pointer, data_type, N, gradient_stride,
					m_beta_3,
					L_begin, CUDA_R_32F, M, L_stride,
					n_matrices,
					compute_type,
					CUBLAS_GEMM_DEFAULT_TENSOR_OP
				));

				cudaEventRecord(L_event, L_stream);

				// m_R[i] = m_beta2 * m_R[i] + (1 - m_beta2) * gradient_matrix.transpose() * gradient_matrix;
				CUBLAS_CHECK_THROW(cublasSetStream(m_cublas, R_stream));
				CUBLAS_CHECK_THROW(cublasSetWorkspace(m_cublas, R_workspace.data(), R_workspace.size()));
				CUBLAS_CHECK_THROW(cublasGemmStridedBatchedEx(
					m_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
					N, N, M,
					m_alpha_3,
					gradient_pointer, data_type, N, gradient_stride,
					gradient_pointer, data_type, N, gradient_stride,
					m_beta_3,
					R_begin, CUDA_R_32F, N, R_stride,
					n_matrices,
					compute_type,
					CUBLAS_GEMM_DEFAULT_TENSOR_OP
				));

				cudaEventRecord(R_event, R_stream);

				// ======================
				// Update step
				// ======================

				// Must wait until after the first step for the L and R matrix roots to get initialized.
				if (m_current_step-1 > 0) {
					CUBLAS_CHECK_THROW(cublasSetStream(m_cublas, update_stream));
					CUBLAS_CHECK_THROW(cublasSetWorkspace(m_cublas, update_workspace.data(), update_workspace.size()));

					// gradient_matrix = L_root * gradients
					CUBLAS_CHECK_THROW(cublasGemmStridedBatchedEx(
						m_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
						M, N, M,
						m_one,
						L_root_begin, CUDA_R_32F, M, L_stride,
						m_momentum.data() + offset_MN, CUDA_R_32F, N, gradient_stride,
						m_zero,
						m_gradient_tmp.data() + offset_MN, CUDA_R_32F, M, gradient_stride,
						n_matrices,
						CUBLAS_COMPUTE_32F_FAST_TF32,
						CUBLAS_GEMM_DEFAULT_TENSOR_OP
					));

					// gradient_matrix = gradient_matrix * R_root
					CUBLAS_CHECK_THROW(cublasGemmStridedBatchedEx(
						m_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
						M, N, N,
						m_alpha_shampoo_coef,
						m_gradient_tmp.data() + offset_MN, CUDA_R_32F, M, gradient_stride,
						R_root_begin, CUDA_R_32F, N, R_stride,
						m_beta_shampoo_coef,
						m_shampoo_momentum.data() + offset_MN, CUDA_R_32F, M, gradient_stride,
						n_matrices,
						CUBLAS_COMPUTE_32F_FAST_TF32,
						CUBLAS_GEMM_DEFAULT_TENSOR_OP
					));

					if (m_frobenius_normalization) {
						reduce_sum(m_shampoo_momentum.data() + offset_MN, [] __device__ (float val) { return val * val; }, m_sqr1_tmp.data() + interval.first, M*N, update_stream, n_matrices);
						reduce_sum(m_momentum.data() + offset_MN, [] __device__ (float val) { return val * val; }, m_sqr2_tmp.data() + interval.first, M*N, update_stream, n_matrices);
					}

					shampoo_step_batched<T><<<n_blocks_linear(M*N*n_matrices), n_threads_linear, 0, update_stream>>>(
						M, N, n_matrices,
						m_relative_weight_decay,
						m_absolute_weight_decay,
						m_base_learning_rate,
						m_sqr1_tmp.data() + interval.first,
						m_sqr2_tmp.data() + interval.first,
						m_shampoo_momentum.data() + offset_MN,
						weights_full_precision + offset_MN,
						weights + offset_MN
					);
				}

				cudaEventRecord(update_event, update_stream);

				for (size_t i = interval.first; i < interval.second; ++i) {
					offset_MN += M*N;
					offset_M += M;
					offset_MM += M*M;
					offset_N += N;
					offset_NN += N*N;
				}
			}

			for (auto& event : m_events) {
				cudaStreamWaitEvent(stream, event, 0);
			}
		}

		const uint32_t update_interval = m_current_step < 100 ? 10 : 200;
		const uint32_t single_update_interval = update_interval / (uint32_t)m_matrix_batches.size();

		std::vector<size_t> to_update;
		if (m_current_step-1 == 0) {
			for (size_t i = 0; i < m_matrix_batches.size(); ++i) {
				to_update.push_back(i);
			}
		} else if (m_current_step % single_update_interval == 0) {
			to_update.push_back((m_current_step / single_update_interval) % m_matrix_batches.size());
		}

		for (size_t j : to_update) {
			auto interval = m_matrix_batches[j];
			uint32_t n_matrices = (uint32_t)(interval.second - interval.first);

			uint32_t M = m_L[interval.first].n();
			uint32_t N = m_R[interval.first].n();

			shampoo_symmetrize_batched<<<n_blocks_linear(M*M*n_matrices), n_threads_linear, 0, stream>>>(M, n_matrices, m_identity_strength, m_L[interval.first].data(), m_L_root[interval.first].data());
			shampoo_symmetrize_batched<<<n_blocks_linear(N*N*n_matrices), n_threads_linear, 0, stream>>>(N, n_matrices, m_identity_strength, m_R[interval.first].data(), m_R_root[interval.first].data());

			inverse_pth_root_batched(stream, M, m_L_root[interval.first].data(), m_inverse_pth_root_buffers[0], n_matrices, (uint32_t)j*2+0);
			inverse_pth_root_batched(stream, N, m_R_root[interval.first].data(), m_inverse_pth_root_buffers[0], n_matrices, (uint32_t)j*2+1);
		}

		// Do an adam update on all the params that were not covered by the layers
		const uint32_t n_remaining_weights = m_n_weights - m_n_weights_covered_by_matrices;
		if (n_remaining_weights > 0) {
			linear_kernel(shampoo_step_remaining<T>, 0, stream,
				n_remaining_weights,
				m_relative_weight_decay,
				m_absolute_weight_decay,
				m_base_learning_rate,
				m_momentum.data() + m_n_weights_covered_by_matrices,
				weights_full_precision + m_n_weights_covered_by_matrices,
				weights + m_n_weights_covered_by_matrices
			);
		}
	}

	float learning_rate() const override {
		return m_base_learning_rate;
	}

	void set_learning_rate(float val) override {
		m_base_learning_rate = val;
	}

	uint32_t step() const override {
		return m_current_step;
	}

	uint32_t n_weights() const override {
		return m_n_weights;
	}

	T* custom_weights() const override {
		return nullptr;
	}

	void update_hyperparams(const json& params) override {
		if (params.contains("beta1")) {
			m_beta1 = params["beta1"];
		}

		if (params.contains("beta2")) {
			m_beta2 = params["beta2"];
		}

		if (params.contains("beta3")) {
			m_beta3 = params["beta3"];
		}

		if (params.contains("beta_shampoo")) {
			m_beta_shampoo = params["beta_shampoo"];
		}

		if (params.contains("epsilon")) {
			m_epsilon = params["epsilon"];
		}

		if (params.contains("identity")) {
			m_identity_strength = params["identity"];
		}

		if (params.contains("learning_rate")) {
			m_base_learning_rate = params["learning_rate"];
		}

		if (params.contains("cg_on_momentum")) {
			m_cg_on_momentum = params["cg_on_momentum"];
		}

		if (params.contains("frobenius_normalization")) {
			m_frobenius_normalization = params["frobenius_normalization"];
		}

		if (params.contains("l2_reg")) {
			m_l2_reg = params["l2_reg"];
		}

		if (params.contains("relative_decay")) {
			m_relative_weight_decay = params["relative_decay"];
		}

		if (params.contains("absolute_decay")) {
			m_absolute_weight_decay = params["absolute_decay"];
		}

		// m_graph.reset();
	}

	json hyperparams() const override {
		return {
			{"otype", "Shampoo"},
			{"beta1", m_beta1},
			{"beta2", m_beta2},
			{"beta3", m_beta3},
			{"beta_shampoo", m_beta_shampoo},
			{"epsilon", m_epsilon},
			{"identity", m_identity_strength},
			{"learning_rate", m_base_learning_rate},
			{"cg_on_momentum", m_cg_on_momentum},
			{"frobenius_normalization", m_frobenius_normalization},
			{"l2_reg", m_l2_reg},
			{"relative_decay", m_relative_weight_decay},
			{"absolute_decay", m_absolute_weight_decay},
		};
	}

	json serialize() const override {
		throw std::runtime_error{"The Shampoo optimizer does not yet support serialization."};
	}

	void deserialize(const json& data) override {
		throw std::runtime_error{"The Shampoo optimizer does not yet support deserialization."};
	}

private:
	uint32_t m_n_weights;
	uint32_t m_n_weights_covered_by_matrices = 0;

	CudaGraph m_graph;

	GPUMemory<float> m_coefficients;

	float* m_one;
	float* m_zero;
	float* m_alpha_1;
	float* m_beta_1;
	float* m_alpha_2;
	float* m_beta_2;
	float* m_alpha_3;
	float* m_beta_3;
	float* m_alpha_shampoo_coef;
	float* m_beta_shampoo_coef;

	GPUMemory<ROOT_TYPE> m_coefficients_root;

	ROOT_TYPE* m_one_root;
	ROOT_TYPE* m_zero_root;

	GPUMemory<float> m_first_moments;
	GPUMemory<float> m_second_moments;
	GPUMemory<float> m_momentum;
	GPUMemory<float> m_shampoo_momentum;

	GPUMemory<char> m_L_buffer;
	GPUMemory<char> m_R_buffer;
	std::vector<GPUMatrix<float>> m_L;
	std::vector<GPUMatrix<float>> m_R;

	GPUMemory<char> m_L_root_buffer;
	GPUMemory<char> m_R_root_buffer;
	std::vector<GPUMatrix<float>> m_L_root;
	std::vector<GPUMatrix<float>> m_R_root;

	GPUMemory<float> m_gradient_tmp;
	GPUMemory<float> m_sqr1_tmp;
	GPUMemory<float> m_sqr2_tmp;

	std::vector<GPUMemory<ROOT_TYPE>> m_inverse_pth_root_buffers;

	std::vector<float> m_shampoo_update;
	std::vector<float> m_adam_update;

	std::vector<std::pair<size_t, size_t>> m_matrix_batches;

	std::vector<GPUMatrix<T, RM>> m_gradient_matrices;

	uint32_t m_current_step = 0;

	// Hyperparameters
	float m_base_learning_rate = 1e-3f;
	float m_beta1 = 0.9f;
	float m_beta2 = 0.99f;
	float m_beta3 = 0.9f;
	float m_beta_shampoo = 0.9f;
	float m_epsilon = 1e-8f;
	float m_identity_strength = 0.01f;
	float m_l2_reg = 1e-5f;

	float m_relative_weight_decay = 0.0f;
	float m_absolute_weight_decay = 0.0f;

	bool m_cg_on_momentum = true;
	bool m_frobenius_normalization = true;

	cublasHandle_t m_cublas;

	std::vector<cudaStream_t> m_streams;
	std::vector<cudaEvent_t> m_events;
	std::vector<GPUMemory<char>> m_cublas_workspaces;

	cudaEvent_t m_global_event;
};

TCNN_NAMESPACE_END
