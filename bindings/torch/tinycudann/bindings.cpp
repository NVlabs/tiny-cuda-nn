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

/** @file   torch_bindings.cu
 *  @author Thomas Müller, Jacob Munkberg, Jon Hasselgren, Or Perel, NVIDIA
 */

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef snprintf
#undef snprintf
#endif

#include <json/json.hpp>

#include <pybind11_json/pybind11_json.hpp>

#include <tiny-cuda-nn/cpp_api.h>

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CHECK_THROW(x) \
	do { if (!(x)) throw std::runtime_error(std::string(FILE_LINE " check failed " #x)); } while(0)

c10::ScalarType torch_type(tcnn::cpp::EPrecision precision) {
	switch (precision) {
		case tcnn::cpp::EPrecision::Fp32: return torch::kFloat32;
		case tcnn::cpp::EPrecision::Fp16: return torch::kHalf;
		default: throw std::runtime_error{"Unknown precision tcnn->torch"};
	}
}

void* void_data_ptr(torch::Tensor& tensor) {
	switch (tensor.scalar_type()) {
		case torch::kFloat32: return tensor.data_ptr<float>();
		case torch::kHalf: return tensor.data_ptr<torch::Half>();
		default: throw std::runtime_error{"Unknown precision torch->void"};
	}
}

#define CHECK_INPUT(x) CHECK_THROW(x.device().is_cuda()); CHECK_THROW(x.is_contiguous())

class Module {
public:
	Module(tcnn::cpp::Module* module) : m_module{module} {}

	std::tuple<tcnn::cpp::Context, torch::Tensor> fwd(torch::Tensor input, torch::Tensor params) {
		CHECK_INPUT(input);
		CHECK_INPUT(params);

		// Types
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10_param_precision());

		// Sizes
		CHECK_THROW(input.size(1) == n_input_dims());
		CHECK_THROW(params.size(0) == n_params());

		// Device
		at::Device device = input.device();
		CHECK_THROW(device == params.device());

		const at::cuda::CUDAGuard device_guard{device};
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		uint32_t batch_size = input.size(0);

		torch::Tensor output = torch::empty({ batch_size, n_output_dims() }, torch::TensorOptions().dtype(c10_output_precision()).device(device));

		tcnn::cpp::Context ctx;
		if (!input.requires_grad() && !params.requires_grad()) {
			m_module->inference(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params));
		} else {
			ctx = m_module->forward(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params), input.requires_grad());
		}

		return { std::move(ctx), output };
	}

	std::tuple<torch::Tensor, torch::Tensor> bwd(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor output, torch::Tensor dL_doutput) {
		if (!ctx.ctx) {
			throw std::runtime_error{"Module::bwd: called with invalid context. fwd likely (mistakenly) ran in inference mode."};
		}

		CHECK_INPUT(input);
		CHECK_INPUT(params);
		CHECK_INPUT(output);
		CHECK_INPUT(dL_doutput);

		// Types
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10_param_precision());
		CHECK_THROW(output.scalar_type() == c10_output_precision());
		CHECK_THROW(dL_doutput.scalar_type() == c10_output_precision());

		// Sizes
		CHECK_THROW(input.size(1) == n_input_dims());
		CHECK_THROW(output.size(1) == n_output_dims());
		CHECK_THROW(params.size(0) == n_params());
		CHECK_THROW(output.size(0) == input.size(0));
		CHECK_THROW(dL_doutput.size(0) == input.size(0));

		// Device
		at::Device device = input.device();
		CHECK_THROW(device == params.device());
		CHECK_THROW(device == output.device());
		CHECK_THROW(device == dL_doutput.device());

		const at::cuda::CUDAGuard device_guard{device};
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		uint32_t batch_size = input.size(0);

		torch::Tensor dL_dinput;
		if (input.requires_grad()) {
			dL_dinput = torch::empty({ batch_size, input.size(1) }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		}

		torch::Tensor dL_dparams;
		if (params.requires_grad()) {
			dL_dparams = torch::empty({ n_params() }, torch::TensorOptions().dtype(c10_param_precision()).device(device));
		}

		if (input.requires_grad() || params.requires_grad()) {
			m_module->backward(
				stream,
				ctx,
				batch_size,
				input.requires_grad() ? dL_dinput.data_ptr<float>() : nullptr,
				void_data_ptr(dL_doutput),
				params.requires_grad() ? void_data_ptr(dL_dparams) : nullptr,
				input.data_ptr<float>(),
				void_data_ptr(output),
				void_data_ptr(params)
			);
		}

		return { dL_dinput, dL_dparams };
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bwd_bwd_input(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor dL_ddLdinput, torch::Tensor dL_doutput) {
		// from: dL_ddLdinput
		// to:   dL_ddLdoutput, dL_dparams, dL_dinput

		if (!ctx.ctx) {
			throw std::runtime_error{"Module::bwd_bwd_input: called with invalid context. fwd likely (mistakenly) ran in inference mode."};
		}

		CHECK_INPUT(input);
		CHECK_INPUT(params);
		CHECK_INPUT(dL_ddLdinput);
		CHECK_INPUT(dL_doutput);

		// Types
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(dL_ddLdinput.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10_param_precision());
		CHECK_THROW(dL_doutput.scalar_type() == c10_output_precision());

		// Sizes
		CHECK_THROW(input.size(1) == n_input_dims());
		CHECK_THROW(dL_doutput.size(1) == n_output_dims());
		CHECK_THROW(dL_ddLdinput.size(1) == n_input_dims());
		CHECK_THROW(params.size(0) == n_params());
		CHECK_THROW(dL_doutput.size(0) == input.size(0));
		CHECK_THROW(dL_ddLdinput.size(0) == input.size(0));

		// Device
		at::Device device = input.device();
		CHECK_THROW(device == params.device());
		CHECK_THROW(device == dL_ddLdinput.device());
		CHECK_THROW(device == dL_doutput.device());

		const at::cuda::CUDAGuard device_guard{device};
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		uint32_t batch_size = input.size(0);

		torch::Tensor dL_ddLdoutput;
		if (dL_doutput.requires_grad()) {
			dL_ddLdoutput = torch::zeros({ batch_size, n_output_dims() }, torch::TensorOptions().dtype(c10_output_precision()).device(device));
		}

		torch::Tensor dL_dparams;
		if (params.requires_grad()) {
			dL_dparams = torch::zeros({ n_params() }, torch::TensorOptions().dtype(c10_param_precision()).device(device));
		}

		torch::Tensor dL_dinput;
		if (input.requires_grad()) {
			dL_dinput = torch::zeros({ batch_size, n_input_dims() }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		}

		if (dL_doutput.requires_grad() || params.requires_grad()) {
			m_module->backward_backward_input(
				stream,
				ctx,
				batch_size,
				dL_ddLdinput.data_ptr<float>(),
				input.data_ptr<float>(),
				(params.requires_grad() || input.requires_grad() ) ? void_data_ptr(dL_doutput) : nullptr,
				params.requires_grad() ? void_data_ptr(dL_dparams) : nullptr,
				dL_doutput.requires_grad() ? void_data_ptr(dL_ddLdoutput) : nullptr,
				input.requires_grad() ? dL_dinput.data_ptr<float>() : nullptr,
				void_data_ptr(params)
			);
		}

		return {dL_ddLdoutput, dL_dparams, dL_dinput};
	}

	torch::Tensor initial_params(size_t seed) {
		torch::Tensor output = torch::zeros({ n_params() }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
		m_module->initialize_params(seed, output.data_ptr<float>());
		return output;
	}

	uint32_t n_input_dims() const {
		return m_module->n_input_dims();
	}

	uint32_t n_params() const {
		return (uint32_t)m_module->n_params();
	}

	tcnn::cpp::EPrecision param_precision() const {
		return m_module->param_precision();
	}

	c10::ScalarType c10_param_precision() const {
		return torch_type(param_precision());
	}

	uint32_t n_output_dims() const {
		return m_module->n_output_dims();
	}

	tcnn::cpp::EPrecision output_precision() const {
		return m_module->output_precision();
	}

	c10::ScalarType c10_output_precision() const {
		return torch_type(output_precision());
	}

	nlohmann::json hyperparams() const {
		return m_module->hyperparams();
	}

	std::string name() const {
		return m_module->name();
	}

private:
	std::unique_ptr<tcnn::cpp::Module> m_module;
};

#if !defined(TCNN_NO_NETWORKS)
Module create_network_with_input_encoding(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json& encoding, const nlohmann::json& network) {
	return Module{tcnn::cpp::create_network_with_input_encoding(n_input_dims, n_output_dims, encoding, network)};
}

Module create_network(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json& network) {
	return Module{tcnn::cpp::create_network(n_input_dims, n_output_dims, network)};
}
#endif

Module create_encoding(uint32_t n_input_dims, const nlohmann::json& encoding, tcnn::cpp::EPrecision requested_precision) {
	return Module{tcnn::cpp::create_encoding(n_input_dims, encoding, requested_precision)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	py::enum_<tcnn::cpp::EPrecision>(m, "Precision")
		.value("Fp32", tcnn::cpp::EPrecision::Fp32)
		.value("Fp16", tcnn::cpp::EPrecision::Fp16)
		.export_values()
		;

	m.def("batch_size_granularity", &tcnn::cpp::batch_size_granularity);
	m.def("free_temporary_memory", &tcnn::cpp::free_temporary_memory);
	m.def("has_networks", &tcnn::cpp::has_networks);
	m.def("preferred_precision", &tcnn::cpp::preferred_precision);

	// Encapsulates an abstract context of an operation
	// (commonly the forward pass) to be passed on to other
	// operations (commonly the backward pass).
	py::class_<tcnn::cpp::Context>(m, "Context");

	// The python bindings expose TCNN's C++ API through
	// a single "Module" class that can act as the encoding,
	// the neural network, or a combined encoding + network
	// under the hood. The bindings don't need to concern
	// themselves with these implementation details, though.
	py::class_<Module>(m, "Module")
		.def("fwd", &Module::fwd)
		.def("bwd", &Module::bwd)
		.def("bwd_bwd_input", &Module::bwd_bwd_input)
		.def("initial_params", &Module::initial_params)
		.def("n_input_dims", &Module::n_input_dims)
		.def("n_params", &Module::n_params)
		.def("param_precision", &Module::param_precision)
		.def("n_output_dims", &Module::n_output_dims)
		.def("output_precision", &Module::output_precision)
		.def("hyperparams", &Module::hyperparams)
		.def("name", &Module::name)
		;

#if !defined(TCNN_NO_NETWORKS)
	m.def("create_network_with_input_encoding", &create_network_with_input_encoding);
	m.def("create_network", &create_network);
#endif

	m.def("create_encoding", &create_encoding);
}
