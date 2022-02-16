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

/** @file   cpp_api.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  API to be consumed by cpp (non-CUDA) programs.
 */

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/cpp_api.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

namespace tcnn { namespace cpp {

template <typename T>
constexpr EPrecision precision() {
	return std::is_same<T, float>::value ? EPrecision::Fp32 : EPrecision::Fp16;
}

class NetworkWithInputEncoding : public Module {
public:
	NetworkWithInputEncoding(uint32_t n_input_dims, uint32_t n_output_dims, const json& encoding, const json& network)
	: Module{precision<network_precision_t>(), precision<network_precision_t>()}, m_network{std::make_shared<tcnn::NetworkWithInputEncoding<network_precision_t>>(n_input_dims, n_output_dims, encoding, network)}
	{}

	void inference(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params) override {
		m_network->set_params((network_precision_t*)params, (network_precision_t*)params, nullptr, nullptr);

		GPUMatrix<float, MatrixLayout::ColumnMajor> input_matrix((float*)input, m_network->input_width(), n_elements);
		GPUMatrix<network_precision_t, MatrixLayout::ColumnMajor> output_matrix((network_precision_t*)output, m_network->padded_output_width(), n_elements);

		// Run on our own custom stream to ensure CUDA graph capture is possible.
		// (Significant possible speedup.)
		SyncedMultiStream synced_stream{stream, 2};
		m_network->inference_mixed_precision(synced_stream.get(1), input_matrix, output_matrix);
	}

	Context forward(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params, bool prepare_input_gradients) override {
		m_network->set_params((network_precision_t*)params, (network_precision_t*)params, nullptr, nullptr);

		GPUMatrix<float, MatrixLayout::ColumnMajor> input_matrix((float*)input, m_network->input_width(), n_elements);
		GPUMatrix<network_precision_t, MatrixLayout::ColumnMajor> output_matrix((network_precision_t*)output, m_network->padded_output_width(), n_elements);

		// Run on our own custom stream to ensure CUDA graph capture is possible.
		// (Significant possible speedup.)
		SyncedMultiStream synced_stream{stream, 2};
		return { m_network->forward(synced_stream.get(1), input_matrix, &output_matrix, false, prepare_input_gradients) };
	}

	void backward(cudaStream_t stream, const Context& ctx, uint32_t n_elements, float* dL_dinput, const void* dL_doutput, void* dL_dparams, const float* input, const void* output, const void* params) override {
		m_network->set_params((network_precision_t*)params, (network_precision_t*)params, (network_precision_t*)params, (network_precision_t*)dL_dparams);

		GPUMatrix<float, MatrixLayout::ColumnMajor> input_matrix((float*)input, m_network->input_width(), n_elements);
		GPUMatrix<float, MatrixLayout::ColumnMajor> dL_dinput_matrix(dL_dinput, m_network->input_width(), n_elements);

		GPUMatrix<network_precision_t, MatrixLayout::ColumnMajor> output_matrix((network_precision_t*)output, m_network->padded_output_width(), n_elements);
		GPUMatrix<network_precision_t, MatrixLayout::ColumnMajor> dL_doutput_matrix((network_precision_t*)dL_doutput, m_network->padded_output_width(), n_elements);

		// Run on our own custom stream to ensure CUDA graph capture is possible.
		// (Significant possible speedup.)
		SyncedMultiStream synced_stream{stream, 2};
		m_network->backward(synced_stream.get(1), *ctx.ctx, input_matrix, output_matrix, dL_doutput_matrix, dL_dinput ? &dL_dinput_matrix : nullptr);
	}

	uint32_t n_input_dims() const override {
		return m_network->input_width();
	}

	size_t n_params() const override {
		return m_network->n_params();
	}

	void initialize_params(size_t seed, float* params_full_precision) override {
		pcg32 rng{seed};
		m_network->initialize_params(rng, params_full_precision, nullptr, nullptr, nullptr, nullptr);
	}

	uint32_t n_output_dims() const override {
		return m_network->padded_output_width();
	}

private:
	std::shared_ptr<tcnn::NetworkWithInputEncoding<network_precision_t>> m_network;
};

template <typename T>
class Encoding : public Module {
public:
	Encoding(uint32_t n_input_dims, const json& encoding)
	: Module{precision<T>(), precision<T>()}, m_encoding{tcnn::create_encoding<T>(n_input_dims, encoding, 0)}
	{}

	void inference(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params) override {
		m_encoding->set_params((T*)params, (T*)params, nullptr, nullptr);

		PitchedPtr<const float> pitched_input(input, m_encoding->num_dims_to_encode());
		PitchedPtr<T> pitched_output((T*)output, m_encoding->num_encoded_dims());

		m_encoding->encode(stream, n_elements, pitched_input, pitched_output, nullptr, true);
	}

	Context forward(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params, bool prepare_input_gradients) override {
		m_encoding->set_params((T*)params, (T*)params, nullptr, nullptr);

		PitchedPtr<const float> pitched_input(input, m_encoding->num_dims_to_encode());
		PitchedPtr<T> pitched_output((T*)output, m_encoding->num_encoded_dims());

		auto forward = std::make_unique<ForwardContext>();
		if (prepare_input_gradients) {
			forward->dy_dx = prepare_input_gradients ? GPUMatrix<float>{m_encoding->num_forward_gradient_dims(), n_elements, stream} : GPUMatrix<float>{};
		}

		m_encoding->encode(stream, n_elements, pitched_input, pitched_output, forward->dy_dx.data(), false);

		return { std::move(forward) };
	}

	void backward(cudaStream_t stream, const Context& ctx, uint32_t n_elements, float* dL_dinput, const void* dL_doutput, void* dL_dparams, const float* input, const void*, const void* params) override {
		m_encoding->set_params((T*)params, (T*)params, (T*)params, (T*)dL_dparams);

		PitchedPtr<const float> pitched_input(input, m_encoding->num_dims_to_encode());
		PitchedPtr<float> pitched_dL_dinput(dL_dinput, m_encoding->num_dims_to_encode());
		PitchedPtr<const T> pitched_dL_doutput((T*)dL_doutput, m_encoding->num_encoded_dims());

		const auto& forward = dynamic_cast<const ForwardContext&>(*ctx.ctx);
		if (dL_dinput && !forward.dy_dx.data()) {
			throw std::runtime_error{"Encoding: forward(prepare_input_gradients) must be called before backward(dL_dinput)"};
		}

		m_encoding->backward(stream, n_elements, pitched_dL_doutput, forward.dy_dx.data(), pitched_dL_dinput, pitched_input);
	}

	uint32_t n_input_dims() const override {
		return m_encoding->num_dims_to_encode();
	}

	size_t n_params() const override {
		return m_encoding->n_params();
	}

	void initialize_params(size_t seed, float* params_full_precision) override {
		pcg32 rng{seed};
		m_encoding->initialize_params(rng, params_full_precision, nullptr, nullptr, nullptr, nullptr);
	}

	uint32_t n_output_dims() const override {
		return m_encoding->num_encoded_dims();
	}

private:
	struct ForwardContext : public tcnn::Context {
		GPUMatrix<float> dy_dx;
	};

	std::shared_ptr<tcnn::Encoding<T>> m_encoding;
};

Module* create_encoding(uint32_t n_input_dims, const json& encoding, EPrecision requested_precision) {
	if (requested_precision == EPrecision::Fp32) {
		return new Encoding<float>{n_input_dims, encoding};
	}
#if TCNN_HALF_PRECISION
	return new Encoding<__half>{n_input_dims, encoding};
#else
	throw std::runtime_error{"TCNN was not compiled with half-precision support."};
#endif
}

Module* create_network_with_input_encoding(uint32_t n_input_dims, uint32_t n_output_dims, const json& encoding, const json& network) {
	return new NetworkWithInputEncoding{n_input_dims, n_output_dims, encoding, network};
}

Module* create_network(uint32_t n_input_dims, uint32_t n_output_dims, const json& network) {
	return create_network_with_input_encoding(n_input_dims, n_output_dims, {{"otype", "Identity"}}, network);
}

EPrecision preferred_precision() {
	return precision<network_precision_t>();
}

}}
