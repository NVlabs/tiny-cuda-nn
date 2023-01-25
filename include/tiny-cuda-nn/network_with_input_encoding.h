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

/** @file   network_with_input_encoding.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  A model that includes its encoding
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>

TCNN_NAMESPACE_BEGIN

template <typename T>
class NetworkWithInputEncoding : public Network<float, T> {
public:
	NetworkWithInputEncoding(std::shared_ptr<Encoding<T>> encoding, uint32_t n_output_dims, const json& network) : m_encoding{encoding} {
		encoding->set_alignment(minimum_alignment(network));

		json local_network_config = network;
		local_network_config["n_input_dims"] = m_encoding->padded_output_width();
		local_network_config["n_output_dims"] = n_output_dims;
		m_network.reset(create_network<T>(local_network_config));
	}

	NetworkWithInputEncoding(uint32_t n_dims_to_encode, uint32_t n_output_dims, const json& encoding, const json& network)
	: NetworkWithInputEncoding{std::shared_ptr<Encoding<T>>{create_encoding<T>(n_dims_to_encode, encoding)}, n_output_dims, network} { }

	virtual ~NetworkWithInputEncoding() { }

	void inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {
		GPUMatrixDynamic<T> network_input = {m_encoding->padded_output_width(), input.n(), stream, m_encoding->preferred_output_layout()};
		m_encoding->inference_mixed_precision(stream, input, network_input, use_inference_params);
		m_network->inference_mixed_precision(stream, network_input, output, use_inference_params);
	}

	uint32_t num_encoded_dims() const {
		return m_encoding->padded_output_width();
	}

	std::unique_ptr<Context> forward_impl(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->network_input = GPUMatrixDynamic<T>{m_encoding->padded_output_width(), input.n(), stream, m_encoding->preferred_output_layout()};
		forward->encoding_ctx = m_encoding->forward(stream, input, &forward->network_input, use_inference_params, prepare_input_gradients);
		forward->network_ctx = m_network->forward(stream, forward->network_input, output, use_inference_params, true);

		return forward;
	}

	void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		EGradientMode param_gradients_mode = EGradientMode::Overwrite
	) override {
		GPUMatrixDynamic<T> dL_dnetwork_input;
		if (m_encoding->n_params() > 0 || dL_dinput) {
			dL_dnetwork_input = {m_encoding->padded_output_width(), input.n(), stream, m_encoding->preferred_output_layout()};
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		m_network->backward(stream, *forward.network_ctx, forward.network_input, output, dL_doutput, dL_dnetwork_input.data() ? &dL_dnetwork_input : nullptr, use_inference_params, param_gradients_mode);
		if (dL_dnetwork_input.data()) {
			m_encoding->backward(
				stream,
				*forward.encoding_ctx,
				input,
				forward.network_input,
				dL_dnetwork_input,
				dL_dinput,
				use_inference_params,
				param_gradients_mode
			);
		}
	}

	void set_params_impl(T* params, T* inference_params, T* gradients) override {
		size_t offset = 0;
		m_network->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_network->n_params();

		m_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_encoding->n_params();
	}

	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
		m_network->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_network->n_params();

		m_encoding->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_encoding->n_params();
	}

	size_t n_params() const override {
		return m_encoding->n_params() + m_network->n_params();
	}

	uint32_t padded_output_width() const override {
		return m_network->padded_output_width();
	}

	uint32_t output_width() const override {
		return m_network->output_width();
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required due to encoding
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		return m_network->layer_sizes();
	}

	uint32_t width(uint32_t layer) const override {
		return layer == 0 ? m_encoding->padded_output_width() : m_network->width(layer - 1);
	}

	uint32_t num_forward_activations() const override {
		return m_network->num_forward_activations() + 1;
	}

	std::pair<const T*, MatrixLayout> forward_activations(const Context& ctx, uint32_t layer) const override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		return layer == 0 ? std::make_pair<const T*, MatrixLayout>(forward.network_input.data(), m_encoding->preferred_output_layout()) : m_network->forward_activations(*forward.network_ctx, layer - 1);
	}

	uint32_t input_width() const override {
		return m_encoding->input_width();
	}

	const std::shared_ptr<Encoding<T>>& encoding() const {
		return m_encoding;
	}

	json hyperparams() const override {
		return {
			{"otype", "NetworkWithInputEncoding"},
			{"encoding", m_encoding->hyperparams()},
			{"network", m_network->hyperparams()},
		};
	}

private:
	std::unique_ptr<Network<T>> m_network;
	std::shared_ptr<Encoding<T>> m_encoding;

	struct ForwardContext : public Context {
		GPUMatrixDynamic<T> network_input;
		std::unique_ptr<Context> encoding_ctx;
		std::unique_ptr<Context> network_ctx;
	};
};

TCNN_NAMESPACE_END
