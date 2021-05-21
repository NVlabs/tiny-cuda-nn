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

/** @file   trainer.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Class that performs training of a differentiable cuda object, given an optimizer and a loss.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>

#include <tiny-cuda-nn/cuda_graph.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/loss.h>

#include <tiny-cuda-nn/misc_kernels.h>
#include <tiny-cuda-nn/reduce_sum.h>

#include <iostream>


TCNN_NAMESPACE_BEGIN

template <typename T, typename PARAMS_T, typename COMPUTE_T=T>
class Trainer : public ObjectWithMutableHyperparams {
public:
	Trainer(std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> model, std::shared_ptr<Optimizer<PARAMS_T>> optimizer, std::shared_ptr<Loss<COMPUTE_T>> loss, uint32_t seed = 1337, float perturbation_sigma = 0)
	: m_model{model}, m_optimizer{optimizer}, m_loss{loss}, m_perturbation_sigma{perturbation_sigma} {
		initialize_params(seed);

		CURAND_CHECK_THROW(curandCreateGenerator(&m_curand, CURAND_RNG_PSEUDO_DEFAULT));
		CURAND_CHECK_THROW(curandSetPseudoRandomGeneratorSeed(m_curand, 1337ULL));
	}

	virtual ~Trainer() {
		curandDestroyGenerator(m_curand);
	}

	void initialize_params(uint32_t seed) {
		size_t n_params = m_model->n_params();
		std::cout << "Trainer: Initializing " << n_params << " params and resetting training." << std::endl;

		m_params_buffer.resize(sizeof(PARAMS_T) * n_params * 3 + sizeof(float) * n_params * 1);
		m_params_buffer.memset(0);

		m_params_full_precision = (float*)(m_params_buffer.data());
		m_params                = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params);
		m_params_backward       = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params + sizeof(PARAMS_T) * n_params);
		m_param_gradients       = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params + sizeof(PARAMS_T) * n_params * 2);

		// Allocate auxiliary optimizer buffers
		m_optimizer->allocate(m_model);

		// Use the optimizer's custom params for inference, if they exist.
		PARAMS_T* params_inference = m_optimizer->custom_weights();
		if (params_inference == nullptr) {
			params_inference = m_params;
		}

		std::seed_seq seq{seed};
		std::vector<uint32_t> seeds(1);
		seq.generate(std::begin(seeds), std::end(seeds));
		std::mt19937 rnd(seeds.front());

		m_model->initialize_params(
			rnd,
			m_params_full_precision,
			m_params,
			params_inference,
			m_params_backward,
			m_param_gradients
		);

		// initialize_params is only expected to initialize m_params_full_precision. Cast and copy these over!
		linear_kernel(cast<PARAMS_T>, 0, nullptr, n_params, m_params_full_precision, m_params);
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	void allocate_training_buffers(uint32_t padded_output_width, uint32_t batch_size) {
		m_perturbation.set_size(padded_output_width, batch_size);
		m_perturbed_training_prediction_tmp.set_size(padded_output_width, batch_size);
		m_training_prediction_tmp.set_size(padded_output_width, batch_size);
		m_training_loss_gradient_tmp.set_size(padded_output_width, batch_size);
		m_training_loss_tmp.set_size(padded_output_width, batch_size);

		GPUMatrixBase::allocate_shared_memory(
			m_training_buffer,
			{
				&m_perturbation,
				&m_perturbed_training_prediction_tmp,
				&m_training_prediction_tmp,
				&m_training_loss_gradient_tmp,
				&m_training_loss_tmp,
			}
		);
	}

	void training_step(
		cudaStream_t stream,
		const GPUMatrix<T, MatrixLayout::ColumnMajor>& input,
		const GPUMatrix<float, MatrixLayout::ColumnMajor>& target,
		float* loss_value = nullptr,
		const GPUMatrix<float, MatrixLayout::ColumnMajor>* data_pdf = nullptr,
		const GPUMatrix<float, MatrixLayout::ColumnMajor>* data_factor = nullptr
	) {
		if (input.n() != target.n()) {
			throw std::runtime_error(std::string("Input and target don't have matching batch size ") + std::to_string(input.n()) + "!=" + std::to_string(target.n()));
		}

		uint32_t padded_output_width = m_model->padded_output_width();
		uint32_t output_width = m_model->output_width();

		if (target.m() != output_width) {
			throw std::runtime_error(std::string("Target does not have the correct number of dimensions ") + std::to_string(target.m()) + "!=" + std::to_string(output_width));
		}

		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		bool did_allocate = false;
		if (m_training_prediction_tmp.n() != batch_size) {
			allocate_training_buffers(padded_output_width, batch_size);
			did_allocate = true;
		}

		static const float loss_scale = 128;

		m_graph.capture_and_execute(stream, did_allocate, [&]() {
			m_model->forward(stream, input, m_training_prediction_tmp);

			if (m_perturbation_sigma > 0) {
				const uint32_t n_elements = m_perturbation.n_elements();
				CURAND_CHECK_THROW(curandSetStream(m_curand, stream));
				CURAND_CHECK_THROW(curandGenerateNormal(m_curand, m_perturbation.data(), n_elements, 0.0f, m_perturbation_sigma));

				add<<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_training_prediction_tmp.data(), m_perturbation.data(), m_perturbed_training_prediction_tmp.data());
			}

			auto& loss_input = m_perturbation_sigma > 0 ? m_perturbed_training_prediction_tmp : m_training_prediction_tmp;

			m_loss->evaluate(
				stream,
				padded_output_width,
				output_width,
				loss_scale,
				loss_input,
				target,
				m_training_loss_tmp,
				m_training_loss_gradient_tmp,
				data_pdf,
				data_factor
			);

			m_model->backward(stream, input, m_training_prediction_tmp, m_training_loss_gradient_tmp);
		});

		m_optimizer->step(stream, loss_scale, m_optimizer->learning_rate(), m_params_full_precision, m_params, m_param_gradients);

		if (loss_value) {
			*loss_value = reduce_sum(m_training_loss_tmp.data(), m_training_loss_tmp.n_elements(), stream);
		}
	}

	void update_hyperparams(json params) override {
		m_optimizer->update_hyperparams(params.value("optimizer", json::object()));
		m_loss->update_hyperparams(params.value("loss", json::object()));
	}

private:
	std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> m_model;
	std::shared_ptr<Optimizer<PARAMS_T>> m_optimizer;
	std::shared_ptr<Loss<COMPUTE_T>> m_loss;

	CudaGraph m_graph;

	GPUMemory<char> m_params_buffer;

	float* m_params_full_precision;
	PARAMS_T* m_params;
	PARAMS_T* m_params_backward; // Used for wonky things like feedback alignment
	PARAMS_T* m_param_gradients;

	float m_perturbation_sigma;
	curandGenerator_t m_curand;

	GPUMemory<char> m_training_buffer;

	GPUMatrix<float, MatrixLayout::ColumnMajor> m_perturbation;
	GPUMatrix<COMPUTE_T, MatrixLayout::ColumnMajor> m_perturbed_training_prediction_tmp;
	GPUMatrix<COMPUTE_T, MatrixLayout::ColumnMajor> m_training_prediction_tmp;
	GPUMatrix<COMPUTE_T, MatrixLayout::ColumnMajor> m_training_loss_gradient_tmp;
	GPUMatrix<float, MatrixLayout::ColumnMajor> m_training_loss_tmp;
};

TCNN_NAMESPACE_END
