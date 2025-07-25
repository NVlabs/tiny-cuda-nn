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

/** @file   object.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Abstract interface of objects in the tiny-cuda-nn framework
 */

#pragma once

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/rtc_kernel.h>

#include <json/json.hpp>

#include <pcg32/pcg32.h>

#include <memory>

namespace tcnn {

using json = nlohmann::json;

class Object {
public:
	virtual ~Object() { }

	virtual json hyperparams() const = 0;

	std::string name() const {
		return hyperparams().value("otype", "<Unknown>");
	}
};

class ObjectWithMutableHyperparams : public Object {
public:
	virtual ~ObjectWithMutableHyperparams() { }

	virtual void update_hyperparams(const json& params) = 0;
};

template <typename PARAMS_T>
class ParametricObject : public Object {
public:
	virtual ~ParametricObject() { }

	virtual void set_params_impl(PARAMS_T* params, PARAMS_T* inference_params, PARAMS_T* gradients) = 0;

	// Must be called prior to inference or training use of the parametric object. Each parameter
	// must point to GPU memory that holds `n_params()` elements.
	// `params` and `inference_params` may point to the same memory.
	void set_params(PARAMS_T* params, PARAMS_T* inference_params, PARAMS_T* gradients) {
		m_params = params;
		m_inference_params = inference_params;
		m_gradients = gradients;

		set_params_impl(params, inference_params, gradients);
	}

	// Initializes the GPU memory addressed by `params_full_precision`. Must hold `n_params()` elements.
	virtual void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) = 0;

	virtual size_t n_params() const = 0;

	PARAMS_T* params() const {
		return m_params;
	}

	PARAMS_T* inference_params() const {
		return m_inference_params;
	}

	PARAMS_T* gradients() const {
		return m_gradients;
	}

	virtual std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const = 0;

private:
	PARAMS_T* m_params = nullptr;
	PARAMS_T* m_inference_params = nullptr;
	PARAMS_T* m_gradients = nullptr;
};

template <typename T>
void one_hot_batched(cudaStream_t stream, const uint32_t num_elements, const uint32_t width, const uint32_t one_hot_dim, T* out, float scale);

template <typename T>
void mult(cudaStream_t stream, const uint32_t num_elements, T* inout, float factor);

template <typename T>
void trim_and_cast_from(cudaStream_t stream, const MatrixLayout layout, const uint32_t num_elements, const uint32_t input_width, const uint32_t output_width, const T* in, float* out);

enum class GradientMode {
	Ignore,
	Overwrite,
	Accumulate,
};

std::unique_ptr<CudaRtcKernel> generate_kernel(
	const std::string& kernel_name,
	const std::string& device_function,
	const std::string& T,
	const std::string& PARAMS_T,
	const std::string& COMPUTE_T,
	uint32_t n_dims_in,
	uint32_t n_fwd_ctx_bytes = 0
);

std::unique_ptr<CudaRtcKernel> generate_backward_kernel(
	const std::string& kernel_name,
	const std::string& device_function,
	const std::string& T,
	const std::string& PARAMS_T,
	const std::string& COMPUTE_T,
	uint32_t n_dims_in,
	uint32_t n_dims_out,
	uint32_t n_fwd_ctx_bytes
);

std::unique_ptr<CudaRtcKernel> generate_backward_backward_input_kernel(
	const std::string& kernel_name,
	const std::string& device_function,
	const std::string& T,
	const std::string& PARAMS_T,
	const std::string& COMPUTE_T,
	uint32_t n_dims_in,
	uint32_t n_dims_out,
	uint32_t n_fwd_ctx_bytes
);

template <typename T, typename PARAMS_T, typename COMPUTE_T=T>
class DifferentiableObject : public ParametricObject<PARAMS_T> {
public:
	virtual ~DifferentiableObject() { }

	virtual void inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<COMPUTE_T>& output, bool use_inference_params = true)
#if defined(TCNN_NO_FWD_BWD)
	{ throw std::runtime_error{"tiny-cuda-nn was compiled without forward / backward support. You must call `set_jit_fusion(true)` on each model before using it."}; }
#else
	= 0;
#endif

	void inference_mixed_precision(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<COMPUTE_T>& output, bool use_inference_params = true) {
		CHECK_THROW(input.m() == input_width());
		CHECK_THROW(output.m() == padded_output_width());
		CHECK_THROW(input.n() % BATCH_SIZE_GRANULARITY == 0);
		CHECK_THROW(input.n() == output.n());

		if (this->n_params() > 0) {
			if (use_inference_params) {
				CHECK_THROW(this->inference_params() != nullptr);
			} else {
				CHECK_THROW(this->params() != nullptr);
			}
		}

		CudaRtcKernel* jit_kernel = !m_jit_fusion ? nullptr : m_jit_fused_inference_mp_kernel.get([this]() {
			auto name = fmt::format("inference_mp_{}", to_snake_case(this->name()));
			try {
				return generate_kernel(
					name,
					this->generate_device_function("eval_model"),
					type_to_string<T>(),
					type_to_string<PARAMS_T>(),
					type_to_string<COMPUTE_T>(),
					input_width(),
					0 // fwd_ctx_shmem
				);
			} catch (const std::runtime_error& e) {
				m_jit_fusion = false;
				log_warning("{}\nFailed to JIT-compile `{}`. Disabling JIT.", e.what(), name);
				return std::unique_ptr<CudaRtcKernel>{};
			}
		}).get();

		if (jit_kernel) {
			auto g = jit_guard(stream, use_inference_params);
			jit_kernel->launch(
				n_blocks_linear(input.n()), N_THREADS_LINEAR, 0, stream,
				input.n(), input.view(), output.view(), use_inference_params ? this->inference_params() : this->params()
			);
			return;
		}

		inference_mixed_precision_impl(stream, input, output, use_inference_params);
	}
	void inference_mixed_precision(const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<COMPUTE_T>& output, bool use_inference_params = true) {
		inference_mixed_precision(nullptr, input, output, use_inference_params);
	}

	void inference(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<float>& output, bool use_inference_params = true) {
		CHECK_THROW(input.m() == input_width());
		CHECK_THROW(output.m() == output_width());
		CHECK_THROW(input.n() % BATCH_SIZE_GRANULARITY == 0);
		CHECK_THROW(input.n() == output.n());

		if (this->n_params() > 0) {
			if (use_inference_params) {
				CHECK_THROW(this->inference_params() != nullptr);
			} else {
				CHECK_THROW(this->params() != nullptr);
			}
		}

		CudaRtcKernel* jit_kernel = !m_jit_fusion ? nullptr : m_jit_fused_inference_kernel.get([this]() {
			auto name = fmt::format("inference_{}", to_snake_case(this->name()));
			try {
				return generate_kernel(
					name,
					this->generate_device_function("eval_model"),
					type_to_string<T>(),
					type_to_string<PARAMS_T>(),
					type_to_string<float>(),
					input_width(),
					0 // fwd_ctx_shmem
				);
			} catch (const std::runtime_error& e) {
				m_jit_fusion = false;
				log_warning("{}\nFailed to JIT-compile `{}`. Disabling JIT.", e.what(), name);
				return std::unique_ptr<CudaRtcKernel>{};
			}
		}).get();

		if (jit_kernel) {
			auto g = jit_guard(stream, use_inference_params);
			jit_kernel->launch(
				n_blocks_linear(input.n()), N_THREADS_LINEAR, 0, stream,
				input.n(), input.view(), output.view(), use_inference_params ? this->inference_params() : this->params()
			);
			return;
		}

		GPUMatrixDynamic<COMPUTE_T> inference_output_tmp;
		if (std::is_same<COMPUTE_T, float>::value && padded_output_width() == output_width()) {
			inference_output_tmp = GPUMatrixDynamic<COMPUTE_T>{(COMPUTE_T*)output.data(), output.m(), output.n(), output.layout()};
		} else {
			inference_output_tmp = GPUMatrixDynamic<COMPUTE_T>{padded_output_width(), output.n(), stream, output.layout()};
		}

		inference_mixed_precision(stream, input, inference_output_tmp, use_inference_params);

		if (std::is_same<COMPUTE_T, float>::value && padded_output_width() == output_width()) {
			return;
		}

		const uint32_t n_elements = (uint32_t)output.n_elements();
		trim_and_cast_from(stream, output.layout(), n_elements, padded_output_width(), output_width(), inference_output_tmp.data(), output.data());
	}
	void inference(const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<float>& output, bool use_inference_params = true) {
		inference(nullptr, input, output, use_inference_params);
	}

	virtual std::unique_ptr<Context> forward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<T>& input,
		GPUMatrixDynamic<COMPUTE_T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	)
#if defined(TCNN_NO_FWD_BWD)
	{ throw std::runtime_error{"tiny-cuda-nn was compiled without forward / backward support. You must call `set_jit_fusion(true)` on each model before using it."}; }
#else
	= 0;
#endif
	std::unique_ptr<Context> forward(
		cudaStream_t stream,
		const GPUMatrixDynamic<T>& input,
		GPUMatrixDynamic<COMPUTE_T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) {
		CHECK_THROW(input.m() == input_width());
		CHECK_THROW(!output || output->m() == padded_output_width());
		CHECK_THROW(input.n() % BATCH_SIZE_GRANULARITY == 0);
		CHECK_THROW(!output || input.n() == output->n());

		if (this->n_params() > 0) {
			if (use_inference_params) {
				CHECK_THROW(this->inference_params() != nullptr);
			} else {
				CHECK_THROW(this->params() != nullptr);
			}
		}

		CudaRtcKernel* jit_kernel = !m_jit_fusion ? nullptr : m_jit_fused_forward_kernel.get([this]() {
			auto name = fmt::format("forward_{}", to_snake_case(this->name()));
			try {
				return generate_kernel(
					name,
					this->generate_device_function("eval_model"),
					type_to_string<T>(),
					type_to_string<PARAMS_T>(),
					type_to_string<COMPUTE_T>(),
					input_width(),
					device_function_fwd_ctx_bytes()
				);
			} catch (const std::runtime_error& e) {
				m_jit_fusion = false;
				log_warning("{}\nFailed to JIT-compile `{}`. Disabling JIT.", e.what(), name);
				return std::unique_ptr<CudaRtcKernel>{};
			}
		}).get();

		if (jit_kernel) {
			auto forward = std::make_unique<JitForwardContext>();
			forward->data = allocate_workspace(stream, device_function_fwd_ctx_bytes() * input.n());

			auto g = jit_guard(stream, use_inference_params);
			jit_kernel->launch(
				n_blocks_linear(input.n()), N_THREADS_LINEAR, 0, stream,
				input.n(), input.view(), output ? output->view() : MatrixView<COMPUTE_T>{}, use_inference_params ? this->inference_params() : this->params(), forward->data.data()
			);

			return forward;
		}

		return forward_impl(stream, input, output, use_inference_params, prepare_input_gradients);
	}
	std::unique_ptr<Context> forward(
		const GPUMatrixDynamic<T>& input,
		GPUMatrixDynamic<COMPUTE_T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) {
		return forward(nullptr, input, output, use_inference_params, prepare_input_gradients);
	}

	virtual void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<COMPUTE_T>& output,
		const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	)
#if defined(TCNN_NO_FWD_BWD)
	{ throw std::runtime_error{"tiny-cuda-nn was compiled without forward / backward support. You must call `set_jit_fusion(true)` on each model before using it."}; }
#else
	= 0;
#endif
	void backward(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<COMPUTE_T>& output,
		const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) {
		// Width
		CHECK_THROW(input.m() == input_width());
		CHECK_THROW(output.m() == padded_output_width());
		CHECK_THROW(dL_doutput.m() == padded_output_width());
		CHECK_THROW(!dL_dinput || dL_dinput->m() == input_width());

		// Batch size
		CHECK_THROW(input.n() % BATCH_SIZE_GRANULARITY == 0);
		CHECK_THROW(input.n() == output.n());
		CHECK_THROW(input.n() == dL_doutput.n());
		CHECK_THROW(!dL_dinput || input.n() == dL_dinput->n());

		// Param & gradient memory must have been set via `set_params(...)`
		if (this->n_params() > 0) {
			if (use_inference_params) {
				CHECK_THROW(this->inference_params() != nullptr);
			} else {
				CHECK_THROW(this->params() != nullptr);
			}

			if (param_gradients_mode != GradientMode::Ignore) {
				CHECK_THROW(this->gradients() != nullptr);
			}
		}

		// Start by considering 256 threads as the reduction of parameter gradients in global memory
		// is slower than doing it in shmem, favoring larger thread blocks.
		uint32_t n_threads = 256 * 2, shmem_bytes;
		do {
			n_threads /= 2;
			shmem_bytes = backward_device_function_shmem_bytes(n_threads, param_gradients_mode);
		} while (n_threads > 32 && shmem_bytes > cuda_max_shmem());

		CudaRtcKernel* jit_kernel = !m_jit_fusion ? nullptr : m_jit_fused_backward_kernel.get([this, n_threads, shmem_bytes]() {
			auto name = fmt::format("backward_{}", to_snake_case(this->name()));
			try {
				if (shmem_bytes > cuda_max_shmem()) {
					throw std::runtime_error{"Not enough shmem."};
				}

				return generate_backward_kernel(
					name,
					this->generate_backward_device_function("backward_eval_model", n_threads),
					type_to_string<T>(),
					type_to_string<PARAMS_T>(),
					type_to_string<COMPUTE_T>(),
					input_width(),
					output_width(),
					device_function_fwd_ctx_bytes()
				);
			} catch (const std::runtime_error& e) {
				m_jit_fusion = false;
				log_warning("{}\nFailed to JIT-compile `{}`. Disabling JIT.", e.what(), name);
				return std::unique_ptr<CudaRtcKernel>{};
			}
		}).get();

		if (jit_kernel) {
			const auto& forward = dynamic_cast<const JitForwardContext&>(ctx);

			if (param_gradients_mode == GradientMode::Overwrite) {
				CUDA_CHECK_THROW(cudaMemsetAsync(this->gradients(), 0, sizeof(PARAMS_T) * this->n_params(), stream));
			}

			auto g = jit_guard(stream, use_inference_params);
			jit_kernel->launch(
				div_round_up(input.n(), n_threads), n_threads, shmem_bytes, stream,
				input.n(),
				dL_doutput.view(),
				dL_dinput ? dL_dinput->view() : MatrixView<T>{},
				use_inference_params ? this->inference_params() : this->params(),
				forward.data.data(),
				param_gradients_mode == GradientMode::Ignore ? nullptr : this->gradients()
			);

			return;
		}

		backward_impl(stream, ctx, input, output, dL_doutput, dL_dinput, use_inference_params, param_gradients_mode);
	}
	void backward(
		const Context& ctx,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<COMPUTE_T>& output,
		const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) {
		backward(nullptr, ctx, input, output, dL_doutput, dL_dinput, use_inference_params, param_gradients_mode);
	}

	virtual void backward_backward_input_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<T>& dL_ddLdinput,
		const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
		GPUMatrixDynamic<COMPUTE_T>* dL_ddLdoutput = nullptr,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	)
#if defined(TCNN_NO_FWD_BWD)
	{ throw std::runtime_error{"tiny-cuda-nn was compiled without forward / backward support."}; }
#else
	{ throw std::runtime_error{fmt::format("{} does not support double backward.", this->name())}; }
#endif
	void backward_backward_input(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<T>& dL_ddLdinput,
		const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
		GPUMatrixDynamic<COMPUTE_T>* dL_ddLdoutput = nullptr,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) {
		// Width
		CHECK_THROW(input.m() == input_width());
		CHECK_THROW(dL_ddLdinput.m() == input_width());
		CHECK_THROW(dL_doutput.m() == padded_output_width());
		CHECK_THROW(!dL_ddLdoutput || dL_ddLdoutput->m() == padded_output_width());
		CHECK_THROW(!dL_dinput || dL_dinput->m() == input_width());

		// Batch size
		CHECK_THROW(input.n() % BATCH_SIZE_GRANULARITY == 0);
		CHECK_THROW(input.n() == dL_ddLdinput.n());
		CHECK_THROW(input.n() == dL_doutput.n());
		CHECK_THROW(!dL_ddLdoutput || input.n() == dL_ddLdoutput->n());
		CHECK_THROW(!dL_dinput || input.n() == dL_dinput->n());

		// Param & gradient memory must have been set via `set_params(...)`
		if (this->n_params() > 0) {
			if (use_inference_params) {
				CHECK_THROW(this->inference_params() != nullptr);
			} else {
				CHECK_THROW(this->params() != nullptr);
			}

			if (param_gradients_mode != GradientMode::Ignore) {
				CHECK_THROW(this->gradients() != nullptr);
			}
		}

		// Start by considering 256 threads as the reduction of parameter gradients in global memory
		// is slower than doing it in shmem, favoring larger thread blocks.
		uint32_t n_threads = 256 * 2, shmem_bytes;
		do {
			n_threads /= 2;
			shmem_bytes = backward_device_function_shmem_bytes(n_threads, param_gradients_mode);
		} while (n_threads > 32 && shmem_bytes > cuda_max_shmem());

		CudaRtcKernel* jit_kernel = !m_jit_fusion ? nullptr : m_jit_fused_backward_backward_input_kernel.get([this, n_threads, shmem_bytes]() {
			auto name = fmt::format("backward_backward_input_{}", to_snake_case(this->name()));
			try {
				if (shmem_bytes > cuda_max_shmem()) {
					throw std::runtime_error{"Not enough shmem."};
				}

				return generate_backward_backward_input_kernel(
					name,
					this->generate_backward_backward_input_device_function("backward_backward_input_eval_model", n_threads),
					type_to_string<T>(),
					type_to_string<PARAMS_T>(),
					type_to_string<COMPUTE_T>(),
					input_width(),
					output_width(),
					device_function_fwd_ctx_bytes()
				);
			} catch (const std::runtime_error& e) {
				m_jit_fusion = false;
				log_warning("{}\nFailed to JIT-compile `{}`. Disabling JIT.", e.what(), name);
				return std::unique_ptr<CudaRtcKernel>{};
			}
		}).get();

		if (jit_kernel) {
			const auto& forward = dynamic_cast<const JitForwardContext&>(ctx);

			if (param_gradients_mode == GradientMode::Overwrite) {
				CUDA_CHECK_THROW(cudaMemsetAsync(this->gradients(), 0, sizeof(PARAMS_T) * this->n_params(), stream));
			}

			auto g = jit_guard(stream, use_inference_params);
			jit_kernel->launch(
				div_round_up(input.n(), n_threads), n_threads, shmem_bytes, stream,
				input.n(),
				dL_ddLdinput.view(),
				dL_doutput.view(),
				dL_dinput ? dL_dinput->view() : MatrixView<T>{},
				dL_ddLdoutput ? dL_ddLdoutput->view() : MatrixView<COMPUTE_T>{},
				use_inference_params ? this->inference_params() : this->params(),
				forward.data.data(),
				param_gradients_mode == GradientMode::Ignore ? nullptr : this->gradients()
			);

			return;
		}

		backward_backward_input_impl(stream, ctx, input, dL_ddLdinput, dL_doutput, dL_ddLdoutput, dL_dinput, use_inference_params, param_gradients_mode);
	}
	void backward_backward_input(
		const Context& ctx,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<T>& dL_ddLdinput,
		const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
		GPUMatrixDynamic<COMPUTE_T>* dL_ddLdoutput = nullptr,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) {
		backward_backward_input(nullptr, ctx, input, dL_ddLdinput, dL_doutput, dL_ddLdoutput, dL_dinput, use_inference_params, param_gradients_mode);
	}

	void input_gradient(
		cudaStream_t stream,
		uint32_t dim,
		const GPUMatrix<T>& input,
		GPUMatrix<T>& d_dinput,
		float backprop_scale = default_loss_scale<PARAMS_T>() // Prevents underflows during half-precision backprop. Same reason for loss_scale to exist.
	) {
		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		GPUMatrix<COMPUTE_T> d_doutput = {padded_output_width(), batch_size, stream};
		GPUMatrix<COMPUTE_T> output = {padded_output_width(), batch_size, stream};

		if (dim >= padded_output_width()) {
			throw std::runtime_error{"Invalid dimension to compute the input gradient for."};
		}

		// Set "loss gradient" at network outputs to 1 at the chosen dimension and 0 elsewhere.
		one_hot_batched(stream, output.n_elements(), padded_output_width(), dim, d_doutput.data(), backprop_scale);

		auto ctx = forward(stream, input, &output, true /* inference matrices */, true /* prep forward buffers for input gradients */);
		backward(stream, *ctx, input, output, d_doutput, &d_dinput, true /* inference matrices */, GradientMode::Ignore);

		mult(stream, d_dinput.n_elements(), d_dinput.data(), 1.0f / backprop_scale);
	}

	virtual uint32_t input_width() const = 0;

	virtual uint32_t padded_output_width() const = 0;
	virtual uint32_t output_width() const = 0;

	virtual uint32_t required_input_alignment() const = 0;

	virtual std::string generate_device_function(const std::string& name) const {
		throw std::runtime_error{fmt::format("{}: device code generation is not supported.", this->name())};
	}

	virtual std::string generate_backward_device_function(const std::string& name, uint32_t n_threads) const {
		throw std::runtime_error{fmt::format("{}: backward device code generation is not supported.", this->name())};
	}

	virtual std::string generate_backward_backward_input_device_function(const std::string& name, uint32_t n_threads) const {
		throw std::runtime_error{fmt::format("{}: backward backward input device code generation is not supported.", this->name())};
	}

	virtual uint32_t device_function_fwd_ctx_bytes() const {
		throw std::runtime_error{fmt::format("{}: forward device code context size is not implemented.", this->name())};
	}

	virtual bool device_function_fwd_ctx_aligned_per_element() const {
		return true;
	}

	virtual uint32_t backward_device_function_shmem_bytes(uint32_t n_threads, GradientMode param_gradients_mode) const {
		return 0;
	}

	std::string generate_vec_in() const {
		return fmt::format("tvec<{}, {}>", type_to_string<T>(), input_width());
	}

	std::string generate_vec(uint32_t width) const {
		return fmt::format("tvec<{}, {}>", type_to_string<COMPUTE_T>(), width);
	}

	std::string generate_vec_padded_out() const {
		return generate_vec(padded_output_width());
	}

	std::string generate_vec_out() const {
		return generate_vec(output_width());
	}

	std::string generate_device_function_from_body(const std::string& name, const std::string& body) const {
		return dfmt(0, R"(
				__device__ auto {NAME}(const {VEC_IN}& input, const {PARAMS_T}* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> {VEC_OUT} {{
				{FWD_CONTEXT_OFFSET}
				{BODY}
				}}
			)",
			"NAME"_a = name,
			"VEC_IN"_a = generate_vec_in(),
			"PARAMS_T"_a = type_to_string<PARAMS_T>(),
			"VEC_OUT"_a = generate_vec_out(),
			"FWD_CONTEXT_OFFSET"_a = device_function_fwd_ctx_aligned_per_element() ? fmt::format("	if (fwd_ctx) fwd_ctx += lane_id() * {};", device_function_fwd_ctx_bytes()) : "",
			"BODY"_a = body
		);
	}

	std::string generate_backward_device_function_from_body(const std::string& name, const std::string& body) const {
		return dfmt(0, R"(
				__device__ void {NAME}(const {VEC_OUT}& dL_dy, const {PARAMS_T}* __restrict__ params, const uint8_t* __restrict__ fwd_ctx, {PARAMS_T}* __restrict__ dL_dparams = nullptr, {VEC_IN}* __restrict__ dL_dx = nullptr) {{
				{FWD_CONTEXT_OFFSET}
				{BODY}
				}}
			)",
			"NAME"_a = name,
			"VEC_IN"_a = generate_vec_in(),
			"PARAMS_T"_a = type_to_string<PARAMS_T>(),
			"VEC_OUT"_a = generate_vec_out(),
			"FWD_CONTEXT_OFFSET"_a = device_function_fwd_ctx_aligned_per_element() ? fmt::format("	fwd_ctx += lane_id() * {};", device_function_fwd_ctx_bytes()) : "",
			"BODY"_a = body
		);
	}

	std::string generate_backward_backward_input_device_function_from_body(const std::string& name, const std::string body) const {
		return dfmt(0, R"(
				__device__ void {NAME}(const {VEC_IN}& dL_ddLdx, const {VEC_OUT}& dL_dy, const {PARAMS_T}* __restrict__ params, const uint8_t* __restrict__ fwd_ctx, {PARAMS_T}* __restrict__ dL_dparams = nullptr, {VEC_IN}* __restrict__ dL_dx = nullptr, {VEC_OUT}* __restrict__ dL_ddLdy = nullptr) {{
				{FWD_CONTEXT_OFFSET}
				{BODY}
				}}
			)",
			"NAME"_a = name,
			"VEC_IN"_a = generate_vec_in(),
			"PARAMS_T"_a = type_to_string<PARAMS_T>(),
			"VEC_OUT"_a = generate_vec_out(),
			"FWD_CONTEXT_OFFSET"_a = device_function_fwd_ctx_aligned_per_element() ? fmt::format("	fwd_ctx += lane_id() * {};", device_function_fwd_ctx_bytes()) : "",
			"BODY"_a = body
		);
	}

	bool jit_fusion() const {
		return m_jit_fusion;
	}

	void set_jit_fusion(bool val) {
		m_jit_fusion = val;
	}

	virtual void convert_params_to_jit_layout(cudaStream_t stream, bool use_inference_params) {}
	virtual void convert_params_from_jit_layout(cudaStream_t stream, bool use_inference_params) {}

	ScopeGuard jit_guard(cudaStream_t stream, bool use_inference_params) {
		if (!m_jit_fusion || m_in_jit_guard) {
			// Permits nesting of jit guards to avoid too frequent
			// back-and-forth conversions.
			return {};
		}

		try {
			convert_params_to_jit_layout(stream, use_inference_params);
		} catch (const std::runtime_error& e) {
			m_jit_fusion = false;
			log_warning("{}\nFailed to JIT-compile parameter conversion. Disabling JIT.", e.what());
			return {};
		}

		m_in_jit_guard = true;
		return {[this, stream, use_inference_params]() {
			m_in_jit_guard = false;
			convert_params_from_jit_layout(stream, use_inference_params);
		}};
	}

	bool in_jit_guard() const {
		return m_in_jit_guard;
	}

private:
	struct JitForwardContext : public Context {
		GPUMemoryArena::Allocation data;
	};

	Lazy<std::unique_ptr<CudaRtcKernel>> m_jit_fused_inference_mp_kernel;
	Lazy<std::unique_ptr<CudaRtcKernel>> m_jit_fused_inference_kernel;
	Lazy<std::unique_ptr<CudaRtcKernel>> m_jit_fused_forward_kernel;
	Lazy<std::unique_ptr<CudaRtcKernel>> m_jit_fused_backward_kernel;
	Lazy<std::unique_ptr<CudaRtcKernel>> m_jit_fused_backward_backward_input_kernel;

	bool m_jit_fusion = false;
	bool m_in_jit_guard = false;
};

}
