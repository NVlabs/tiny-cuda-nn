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

/** @file   trainer.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Class that performs training of a differentiable cuda object, given an optimizer and a loss.
 */

#pragma once

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/cuda_graph.h>
#include <tiny-cuda-nn/gpu_memory_json.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/object.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/reduce_sum.h>
#include <tiny-cuda-nn/rtc_kernel.h>

#include <random>

namespace tcnn {

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
		log_debug("Trainer: initializing {} params and resetting training.", n_params);

		// Allocate auxiliary optimizer buffers
		m_optimizer->allocate(m_model);

		m_params_buffer.resize(sizeof(PARAMS_T) * n_params * 2 + sizeof(float) * n_params * 1);
		m_params_buffer.memset(0);

		reset_param_pointers();

		m_model->initialize_params(m_rng, m_params_full_precision);

		// initialize_params is only expected to initialize m_params_full_precision. Cast and copy these over!
		set_params_full_precision(m_params_full_precision, n_params, true);

		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	struct ForwardContext : public Context {
		GPUMatrix<COMPUTE_T> perturbed_output;
		GPUMatrix<COMPUTE_T> output;
		GPUMatrix<COMPUTE_T> dL_doutput;
		GPUMatrix<float> L;
		std::unique_ptr<Context> model_ctx;
	};

	std::unique_ptr<ForwardContext> forward(
		cudaStream_t stream,
		const float loss_scale,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrix<float>& target,
		const GPUMatrix<float>* data_pdf = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false,
		const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr
	) {
		const uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->output = GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size, stream};
		forward->model_ctx = m_model->forward(stream, input, &forward->output, use_inference_params, prepare_input_gradients);

		if (m_perturbation_sigma > 0) {
			GPUMatrix<float> perturbation{m_model->padded_output_width(), batch_size, stream};
			forward->perturbed_output = GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size, stream};

			const uint32_t n_elements = perturbation.n_elements();
			generate_random_logistic<float>(stream, m_rng, n_elements, perturbation.data(), 0.0f, m_perturbation_sigma);
			add<<<n_blocks_linear(n_elements), N_THREADS_LINEAR, 0, stream>>>(n_elements, forward->output.data(), perturbation.data(), forward->perturbed_output.data());
		}

		auto& loss_input = m_perturbation_sigma > 0 ? forward->perturbed_output : forward->output;

		forward->L = GPUMatrix<float>{m_model->padded_output_width(), batch_size, stream};

		if (external_dL_dy) {
			CHECK_THROW(external_dL_dy->m() == m_model->padded_output_width());
			CHECK_THROW(external_dL_dy->n() == batch_size);

			forward->dL_doutput = GPUMatrix<COMPUTE_T>{external_dL_dy->data(), m_model->padded_output_width(), batch_size};
		} else {
			CHECK_THROW(input.n() == target.n());
			CHECK_THROW(m_model->output_width() == target.m());

			forward->dL_doutput = GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size, stream};
			m_loss->evaluate(stream, loss_scale, loss_input, target, forward->L, forward->dL_doutput, data_pdf);
		}

		return forward;
	}

	std::unique_ptr<ForwardContext> forward(const float loss_scale, const GPUMatrixDynamic<T>& input, const GPUMatrix<float>& target, const GPUMatrix<float>* data_pdf = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false, const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr) {
		return forward(nullptr, loss_scale, input, target, data_pdf, use_inference_params, prepare_input_gradients, external_dL_dy);
	}

	void backward(cudaStream_t stream, const ForwardContext& ctx, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* dL_dinput = nullptr, bool use_inference_params = false, GradientMode param_gradients_mode = GradientMode::Overwrite) {
		m_model->backward(stream, *ctx.model_ctx, input, ctx.output, ctx.dL_doutput, dL_dinput, use_inference_params, param_gradients_mode);
	}

	void backward(const ForwardContext& ctx, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* dL_dinput = nullptr, bool use_inference_params = false, GradientMode param_gradients_mode = GradientMode::Overwrite) {
		backward(nullptr, ctx, input, dL_dinput, use_inference_params, param_gradients_mode);
	}

	void optimizer_step(cudaStream_t stream, float loss_scale) {
		m_optimizer->step(stream, loss_scale, m_params_full_precision, m_params, m_param_gradients);
	}

	void optimizer_step(float loss_scale) {
		optimizer_step(nullptr, loss_scale);
	}

	std::unique_ptr<CudaRtcKernel> generate_fused_fwd_bwd_kernel(const std::string& name, uint32_t n_threads, bool fwd_ctx_in_shmem, uint32_t non_fwd_ctx_shmem_bytes, bool has_external_dL_dy) {
		std::string forward_name = fmt::format("{}_forward", name);
		std::string backward_name = fmt::format("{}_backward", name);
		std::string loss_name = fmt::format("{}_loss", name);

		std::string loss_body = has_external_dL_dy ? dfmt(0, R"(
				auto dL_dy = external_dL_dy.col<{N_DIMS_OUT}>(i);
			)",
			"LOSS"_a = loss_name,
			"N_DIMS_OUT"_a = m_model->output_width(),
			"COMPUTE_T"_a = type_to_string<COMPUTE_T>()
		) : dfmt(1, R"(
				auto target = data_target.col<{N_DIMS_OUT}>(i);
				vec<{N_DIMS_OUT}> loss;

				auto pdf = data_pdf ? data_pdf.col<{N_DIMS_OUT}>(i) : vec<{N_DIMS_OUT}>::ones();
				auto dL_dy = {LOSS}(
					{N_DIMS_OUT} * n_elements,
					loss_scale,
					out,
					target,
					pdf,
					&loss
				);

				if (data_loss) {{
					data_loss.set_col(i, loss);
				}}
			)",
			"LOSS"_a = loss_name,
			"N_DIMS_OUT"_a = m_model->output_width()
		);

		return std::make_unique<CudaRtcKernel>(name, dfmt(0, R"(
				{FORWARD_DEVICE_FUNCTION}

				{BACKWARD_DEVICE_FUNCTION}

				{LOSS_DEVICE_FUNCTION}

				__global__ void {KERNEL_NAME}(
					const uint32_t n_elements,
					const float loss_scale,
					MatrixView<const {T}> data_in,
					MatrixView<const float> data_target,
					MatrixView<const float> data_pdf,
					MatrixView<const {COMPUTE_T}> external_dL_dy,
					MatrixView<{T}> data_dL_dx,
					MatrixView<float> data_loss,
					const {PARAMS_T}* __restrict__ params,
					uint8_t* __restrict__ fwd_ctx_gmem,
					{PARAMS_T}* __restrict__ dL_dparams
				) {{
					{FWD_CTX_GMEM}{FWD_CTX_SHMEM}
					fwd_ctx += previous_multiple(threadIdx.x, WARP_SIZE) * {N_FWD_CTX_BYTES};

					// Here, fwd_ctx is aligned to each _warp_, i.e. every warp
					// has {N_FWD_CTX_BYTES} bytes of context memory at its
					// disposal, starting at `fwd_ctx`, and it can order its
					// accesses of this memory however it wishes.

					const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

					auto out = {FORWARD}(data_in.col<{N_DIMS_IN}>(i), params, fwd_ctx);

					{LOSS_BODY}

					auto dL_dx = tvec<{T}, {N_DIMS_IN}>::zero();
					{BACKWARD}(dL_dy, params, fwd_ctx, dL_dparams, data_dL_dx ? &dL_dx : nullptr);
					if (data_dL_dx) {{
						data_dL_dx.set_col(i, dL_dx);
					}}
				}}
			)",
			"FORWARD_DEVICE_FUNCTION"_a = m_model->generate_device_function(forward_name),
			"BACKWARD_DEVICE_FUNCTION"_a = m_model->generate_backward_device_function(backward_name, n_threads),
			"LOSS_DEVICE_FUNCTION"_a = has_external_dL_dy ? "" : m_loss->generate_device_function(loss_name, m_model->output_width()),
			"KERNEL_NAME"_a = name,
			"T"_a = type_to_string<T>(),
			"PARAMS_T"_a = type_to_string<PARAMS_T>(),
			"COMPUTE_T"_a = type_to_string<COMPUTE_T>(),
			"FWD_CTX_GMEM"_a = fwd_ctx_in_shmem ? "" : fmt::format("uint8_t* __restrict__ fwd_ctx = fwd_ctx_gmem + blockIdx.x * blockDim.x * {};", m_model->device_function_fwd_ctx_bytes()),
			"FWD_CTX_SHMEM"_a = fwd_ctx_in_shmem ? fmt::format("extern __shared__ uint8_t fwd_ctx_shmem[]; uint8_t* fwd_ctx = (uint8_t*)fwd_ctx_shmem + {};", non_fwd_ctx_shmem_bytes) : "",
			"N_FWD_CTX_BYTES"_a = m_model->device_function_fwd_ctx_bytes(),
			"FORWARD"_a = forward_name,
			"BACKWARD"_a = backward_name,
			"LOSS_BODY"_a = loss_body,
			"N_DIMS_IN"_a = m_model->input_width()
		));
	}

	std::unique_ptr<ForwardContext> training_step(
		cudaStream_t stream,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrix<float>& target,
		const GPUMatrix<float>* data_pdf = nullptr,
		bool run_optimizer = true,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite,
		const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr
	) {
		const float loss_scale = default_loss_scale<PARAMS_T>();

		if (m_model->jit_fusion()) {
			std::string kernel_name = fmt::format("training_step_{}", to_snake_case(m_model->name()));
			try {
				// Automatically figure out whether our forward context fits into shmem while still utilizing
				// a reasonable (4 or more) number of warps. If not, try to find the maximum number of warps
				// that can handle just the backward reduction fitting into shmem.
				bool fwd_ctx_in_shmem = true;
				uint32_t n_warps = 16, n_threads, bwd_reduction_shmem_bytes, shmem_bytes;
				do {
					n_warps /= 2;
					if (n_warps < 4 && fwd_ctx_in_shmem) {
						n_warps = 8;
						fwd_ctx_in_shmem = false;
					}

					n_threads = WARP_SIZE * n_warps;
					bwd_reduction_shmem_bytes = m_model->backward_device_function_shmem_bytes(n_threads, param_gradients_mode);
					shmem_bytes = bwd_reduction_shmem_bytes + (fwd_ctx_in_shmem ? m_model->device_function_fwd_ctx_bytes() * n_threads : 0);
				} while (n_warps > 1 && shmem_bytes > cuda_max_shmem());

				if (shmem_bytes > cuda_max_shmem()) {
					throw std::runtime_error{"Not enough shmem."};
				}

				if (!m_jit_fused_kernel) {
					log_debug("Training JIT: ctx_in_shmem={} shmem={} n_warps={}", fwd_ctx_in_shmem, shmem_bytes, n_warps);
					m_jit_fused_kernel = generate_fused_fwd_bwd_kernel(kernel_name, n_threads, fwd_ctx_in_shmem, bwd_reduction_shmem_bytes, external_dL_dy);
				}

				if (m_jit_fused_kernel) {
					std::unique_ptr<ForwardContext> ctx;
					ctx = std::make_unique<ForwardContext>();
					ctx->L = GPUMatrix<float>{m_model->output_width(), input.n(), stream};

					// Only allocate fwd_ctx in global memory if it doesn't fit in shared memory as indicated by `fwd_ctx_in_shmem`
					GPUMemoryArena::Allocation fwd_ctx_gmem;
					if (!fwd_ctx_in_shmem) {
						fwd_ctx_gmem = allocate_workspace(stream, m_model->device_function_fwd_ctx_bytes() * input.n());
					}

					if (param_gradients_mode == GradientMode::Overwrite) {
						CUDA_CHECK_THROW(cudaMemsetAsync(m_param_gradients, 0, sizeof(PARAMS_T) * m_model->n_params(), stream));
					}

					{
						auto jit_guard = m_model->jit_guard(stream, use_inference_params);

						CHECK_THROW(BATCH_SIZE_GRANULARITY % n_threads == 0);
						m_jit_fused_kernel->launch(
							n_blocks_linear(input.n(), n_threads), n_threads, shmem_bytes, stream,
							input.n(),
							loss_scale,
							input.view(),
							target.view(),
							data_pdf ? data_pdf->view() : MatrixView<float>{},
							external_dL_dy ? external_dL_dy->view() : MatrixView<COMPUTE_T>{},
							dL_dinput ? dL_dinput->view() : MatrixView<T>{},
							ctx->L.view(),
							use_inference_params ? m_params_inference : m_params,
							fwd_ctx_gmem.data(), // nullptr if `fwd_ctx_in_shmem == true`
							param_gradients_mode == GradientMode::Ignore ? nullptr : m_param_gradients
						);
					}

					if (run_optimizer) {
						CHECK_THROW(!m_model->in_jit_guard());
						optimizer_step(stream, loss_scale);
					}

					return ctx;
				}
			} catch (const std::runtime_error& e) {
				m_model->set_jit_fusion(false);
				log_warning("{}\nFailed to JIT-compile `{}`. Disabling JIT.", e.what(), kernel_name);
			}
		}

		std::unique_ptr<ForwardContext> ctx;
		{
			// Execute forward and backward in a CUDA graph for maximum performance.
			auto capture_guard = m_graph.capture_guard(stream);
			ctx = forward(stream, loss_scale, input, target, data_pdf, use_inference_params, dL_dinput, external_dL_dy);
			backward(stream, *ctx, input, dL_dinput, use_inference_params, param_gradients_mode);
		}

		if (run_optimizer) {
			optimizer_step(stream, loss_scale);
		}

		return ctx;
	}

	std::unique_ptr<ForwardContext> training_step(
		const GPUMatrixDynamic<T>& input,
		const GPUMatrix<float>& target,
		const GPUMatrix<float>* data_pdf = nullptr,
		bool run_optimizer = true,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite,
		const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr
	) {
		return training_step(nullptr, input, target, data_pdf, run_optimizer, dL_dinput, use_inference_params, param_gradients_mode, external_dL_dy);
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

	std::unique_ptr<CudaRtcKernel> m_jit_fused_kernel;
};

}
