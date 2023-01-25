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

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/reduce_sum.h>
#include <tiny-cuda-nn/gpu_memory_json.h>

#include <iostream>
#include <random>

TCNN_NAMESPACE_BEGIN

template <typename T, typename PARAMS_T, typename COMPUTE_T=T>
class Trainer : public ObjectWithMutableHyperparams {
public:
	Trainer(std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> model, std::shared_ptr<Optimizer<PARAMS_T>> optimizer, std::shared_ptr<Loss<COMPUTE_T>> loss, uint32_t seed = 1337, float perturbation_sigma = 0)
	: m_model{model}, m_optimizer{optimizer}, m_loss{loss}, m_perturbation_sigma{perturbation_sigma} {
		std::seed_seq seq{seed};
		std::vector<uint32_t> seeds(2);
		seq.generate(std::begin(seeds), std::end(seeds));
		m_rng = pcg32{seeds.front()};
		initialize_params();
	}

	virtual ~Trainer() {}

	void set_loss(std::shared_ptr<Loss<COMPUTE_T>> loss) {
		if (!loss) {
			throw std::runtime_error{"Trainer: may not set loss to nullptr"};
		}
		m_loss = loss;
	}

	void initialize_params() {
		size_t n_params = m_model->n_params();
#ifdef TCNN_VERBOSE_MEMORY_ALLOCS
		std::cout << "Trainer: Initializing " << n_params << " params and resetting training." << std::endl;
#endif

		// Allocate auxiliary optimizer buffers
		m_optimizer->allocate(m_model);

		m_params_buffer.resize(sizeof(PARAMS_T) * n_params * 2 + sizeof(float) * n_params * 1);
		m_params_buffer.memset(0);

		reset_param_pointers();

		m_model->initialize_params(m_rng, m_params_full_precision);

		// initialize_params is only expected to initialize m_params_full_precision. Cast and copy these over!
		parallel_for_gpu(n_params, [params_fp=m_params_full_precision, params=m_params] __device__ (size_t i) {
			params[i] = (PARAMS_T)params_fp[i];
		});
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	struct ForwardContext : public Context {
		GPUMatrix<COMPUTE_T> perturbed_output;
		GPUMatrix<COMPUTE_T> output;
		GPUMatrix<COMPUTE_T> dL_doutput;
		GPUMatrix<float> L;
		std::unique_ptr<Context> model_ctx;
	};

	std::unique_ptr<ForwardContext> forward(cudaStream_t stream, const float loss_scale, const GPUMatrixDynamic<T>& input, const GPUMatrix<float>& target, const GPUMatrix<float>* data_pdf = nullptr) {
		CHECK_THROW(input.n() == target.n());

		const uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->output = GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size, stream};
		forward->model_ctx = m_model->forward(stream, input, &forward->output);

		if (m_perturbation_sigma > 0) {
			GPUMatrix<float> perturbation{m_model->padded_output_width(), batch_size, stream};
			forward->perturbed_output = GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size, stream};

			const uint32_t n_elements = perturbation.n_elements();
			generate_random_logistic<float>(stream, m_rng, n_elements, perturbation.data(), 0.0f, m_perturbation_sigma);
			add<<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, forward->output.data(), perturbation.data(), forward->perturbed_output.data());
		}

		auto& loss_input = m_perturbation_sigma > 0 ? forward->perturbed_output : forward->output;

		forward->dL_doutput = GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size, stream};
		forward->L = GPUMatrix<float>{m_model->padded_output_width(), batch_size, stream};

		m_loss->evaluate(
			stream,
			m_model->padded_output_width(),
			m_model->output_width(),
			loss_scale,
			loss_input,
			target,
			forward->L,
			forward->dL_doutput,
			data_pdf
		);

		return forward;
	}

	std::unique_ptr<ForwardContext> forward(const float loss_scale, const GPUMatrixDynamic<T>& input, const GPUMatrix<float>& target, const GPUMatrix<float>* data_pdf = nullptr) {
		return forward(nullptr, loss_scale, input, target, data_pdf);
	}

	void backward(cudaStream_t stream, const ForwardContext& ctx, const GPUMatrixDynamic<T>& input) {
		m_model->backward(stream, *ctx.model_ctx, input, ctx.output, ctx.dL_doutput);
	}

	void backward(const ForwardContext& ctx, const GPUMatrixDynamic<T>& input) {
		backward(nullptr, ctx, input);
	}

	void optimizer_step(cudaStream_t stream, float loss_scale) {
		m_optimizer->step(stream, loss_scale, m_params_full_precision, m_params, m_param_gradients);
	}

	void optimizer_step(float loss_scale) {
		optimizer_step(nullptr, loss_scale);
	}

	std::unique_ptr<ForwardContext> training_step(
		cudaStream_t stream,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrix<float>& target,
		const GPUMatrix<float>* data_pdf = nullptr,
		bool run_optimizer = true
	) {
		CHECK_THROW(input.n() == target.n());
		CHECK_THROW(target.m() == m_model->output_width());

		static const float loss_scale = 128;

		std::unique_ptr<ForwardContext> result;

		// Execute forward and backward in a CUDA graph for maximum performance.
		{
			auto capture_guard = m_graph.capture_guard(stream);
			result = forward(stream, loss_scale, input, target, data_pdf);
			backward(stream, *result, input);
		}

		if (run_optimizer) {
			optimizer_step(stream, loss_scale);
		}
		return result;
	}

	std::unique_ptr<ForwardContext> training_step(
		const GPUMatrixDynamic<T>& input,
		const GPUMatrix<float>& target,
		const GPUMatrix<float>* data_pdf = nullptr,
		bool run_optimizer = true
	) {
		return training_step(nullptr, input, target, data_pdf, run_optimizer);
	}

	float loss(cudaStream_t stream, const ForwardContext& ctx) const {
		return reduce_sum(ctx.L.data(), ctx.L.n_elements(), stream);
	}

	float loss(const ForwardContext& ctx) const {
		return loss(nullptr, ctx);
	}

	void update_hyperparams(const json& params) override {
		m_optimizer->update_hyperparams(params.value("optimizer", json::object()));
		m_loss->update_hyperparams(params.value("loss", json::object()));
	}

	json hyperparams() const override {
		return {
			{"otype", "Trainer"},
			{"optimizer", m_optimizer->hyperparams()},
			{"loss", m_loss->hyperparams()},
		};
	}

	float* params_full_precision() const {
		return m_params_full_precision;
	}

	PARAMS_T* params() const {
		return m_params;
	}

	PARAMS_T* params_inference() const {
		return m_params_inference;
	}

	PARAMS_T* param_gradients() const {
		return m_param_gradients;
	}

	void set_params_full_precision(const float* params, size_t n_params, bool device_ptr = false) {
		if (n_params != m_model->n_params()) {
			throw std::runtime_error{"Can't set fp params because buffer has the wrong size."};
		}
		CUDA_CHECK_THROW(cudaMemcpy(m_params_full_precision, params, sizeof(float)*n_params, device_ptr ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));

		parallel_for_gpu(n_params, [params_fp=m_params_full_precision, params_inference=m_params_inference] __device__ (size_t i) {
			params_inference[i] = (PARAMS_T)params_fp[i];
		});

		CUDA_CHECK_THROW(cudaMemcpy(m_params, m_params_inference, sizeof(PARAMS_T)*n_params, cudaMemcpyDeviceToDevice));
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	void set_params(const PARAMS_T* params, size_t n_params, bool device_ptr = false) {
		if (n_params != m_model->n_params()) {
			throw std::runtime_error{"Can't set params because buffer has the wrong size."};
		}

		CUDA_CHECK_THROW(cudaMemcpy(m_params_inference, params, sizeof(PARAMS_T)*n_params, device_ptr ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
		CUDA_CHECK_THROW(cudaMemcpy(m_params, m_params_inference, sizeof(PARAMS_T)*n_params, cudaMemcpyDeviceToDevice));

		parallel_for_gpu(n_params, [params_fp=m_params_full_precision, params_inference=m_params_inference] __device__ (size_t i) {
			params_fp[i] = (float)params_inference[i];
		});

		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> model() {
		return m_model;
	}

	json serialize(bool serialize_optimizer = false) {
		size_t n_params = m_model->n_params();

		json data;
		data["n_params"] = n_params;
		data["params_type"] = type_to_string<PARAMS_T>();
		data["params_binary"] = gpu_memory_to_json_binary(m_params_inference, sizeof(PARAMS_T)*n_params);

		if (serialize_optimizer) {
			data["optimizer"] = m_optimizer->serialize();
		}

		return data;
	}

	void deserialize(const json& data) {
		std::string type = data.value("params_type", type_to_string<PARAMS_T>());
		if (type == "float") {
			GPUMemory<float> params = data["params_binary"];
			set_params_full_precision(params.data(), params.size(), true);
		} else if (type == "__half") {
			GPUMemory<__half> params_hp = data["params_binary"];
			size_t n_params = params_hp.size();

			GPUMemory<PARAMS_T> params(n_params);
			parallel_for_gpu(n_params, [params=params.data(), params_hp=params_hp.data()] __device__ (size_t i) {
				params[i] = (PARAMS_T)params_hp[i];
			});

			set_params(params.data(), params.size(), true);
		} else {
			throw std::runtime_error{"Trainer: snapshot parameters must be of type float of __half"};
		}

		if (data.contains("optimizer")) {
			m_optimizer->deserialize(data["optimizer"]);
		}

		reset_param_pointers();
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	void set_param_gradients_pointer(PARAMS_T* gradients) {
		reset_param_pointers();
		m_model->set_params(m_params, m_params_inference, gradients);
	}

	void reset_param_pointers() {
		size_t n_params = m_model->n_params();

		m_params_full_precision = (float*)(m_params_buffer.data());
		m_params                = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params);
		m_param_gradients       = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params + sizeof(PARAMS_T) * n_params);

		// Use the optimizer's custom params for inference, if they exist.
		m_params_inference = m_optimizer ? m_optimizer->custom_weights() : nullptr;
		if (m_params_inference == nullptr) {
			m_params_inference = m_params;
		}

		m_model->set_params(m_params, m_params_inference, m_param_gradients);
	}

	size_t n_params() const {
		return m_model->n_params();
	}

private:
	std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> m_model;
	std::shared_ptr<Optimizer<PARAMS_T>> m_optimizer;
	std::shared_ptr<Loss<COMPUTE_T>> m_loss;

	CudaGraph m_graph;

	GPUMemory<char> m_params_buffer;

	float* m_params_full_precision = nullptr;
	PARAMS_T* m_params_inference = nullptr;
	PARAMS_T* m_params = nullptr;
	PARAMS_T* m_param_gradients = nullptr;

	float m_perturbation_sigma;

	std::unique_ptr<Context> m_training_ctx;

	pcg32 m_rng;
};

TCNN_NAMESPACE_END
