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

#ifdef snprintf
#undef snprintf
#endif

#include <json/json.hpp>

#include <pybind11_json/pybind11_json.hpp>

#include <tiny-cuda-nn/cpp_api.h>

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

class Module {
public:
	Module(tcnn::cpp::Module* module) : m_module{module} {}

	// Helper constructor to create a NetworkWithInputEncoding module
	Module(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json& encoding, const nlohmann::json& network)
	: Module{tcnn::cpp::create_network_with_input_encoding(n_input_dims, n_output_dims, encoding, network)} {}

	// Helper constructor to create a Network module
	Module(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json& network)
	: Module{tcnn::cpp::create_network(n_input_dims, n_output_dims, network)} {}

	// Helper constructor to create a Encoding module
	Module(uint32_t n_input_dims, const nlohmann::json& encoding)
	: Module{tcnn::cpp::create_encoding(n_input_dims, encoding)} {}

	torch::Tensor fwd(torch::Tensor input, torch::Tensor params) {
		// Check for correct types
		if (input.scalar_type() != torch::kFloat32) {
			throw std::runtime_error{"Module::fwd: invalid input type"};
		}

		if (params.scalar_type() != c10_param_precision()) {
			throw std::runtime_error{"Module::fwd: invalid param type"};
		}

		// Check for correct sizes
		if (input.size(1) != n_input_dims()) {
			throw std::runtime_error{"Module::fwd: invalid number of input dimensions"};
		}

		if (params.size(0) != n_params()) {
			throw std::runtime_error{"Module::fwd: invalid number of params"};
		}

		m_fwd_stream = at::cuda::getCurrentCUDAStream();

		uint32_t batch_size = input.size(0);
		torch::Tensor output = torch::empty({ batch_size, n_output_dims() }, torch::TensorOptions().dtype(c10_output_precision()).device(torch::kCUDA));

		if (!input.requires_grad() && !params.requires_grad()) {
			m_module->inference(m_fwd_stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params));
		} else {
			m_module->forward(m_fwd_stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params), input.requires_grad());
			m_fwd_was_called = true;
		}

		return output;
	}

	std::tuple<torch::Tensor, torch::Tensor> bwd(torch::Tensor input, torch::Tensor params, torch::Tensor output, torch::Tensor dL_doutput) {
		if (!m_fwd_was_called) {
			throw std::runtime_error{"Must call Module::fwd before Module::bwd"};
		}

		// Check for correct types
		if (input.scalar_type() != torch::kFloat32) {
			throw std::runtime_error{"Module::bwd: invalid input type"};
		}

		if (params.scalar_type() != c10_param_precision()) {
			throw std::runtime_error{"Module::bwd: invalid param type"};
		}

		if (output.scalar_type() != c10_output_precision()) {
			throw std::runtime_error{"Module::bwd: invalid output type"};
		}

		if (dL_doutput.scalar_type() != c10_output_precision()) {
			throw std::runtime_error{"Module::bwd: invalid output gradient type"};
		}

		// Check for correct sizes
		if (input.size(1) != n_input_dims()) {
			throw std::runtime_error{"Module::bwd: invalid number of input dimensions"};
		}

		if (output.size(1) != n_output_dims()) {
			throw std::runtime_error{"Module::bwd: invalid number of output dimensions"};
		}

		if (params.size(0) != n_params()) {
			throw std::runtime_error{"Module::bwd: invalid number of params"};
		}

		uint32_t batch_size = input.size(0);
		if (output.size(0) != batch_size || dL_doutput.size(0) != batch_size) {
			throw std::runtime_error{"Module::bwd: batch size mismatch"};
		}

		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		if (stream != m_fwd_stream) {
			// TODO: this entire problem can be worked around by creating our own stream
			//       and syncing it with whichever streams are provided by torch. Let's
			//       hold off with that for now and fix it if fwd/bwd calls ever happen
			//       from disparate streams.
			throw std::runtime_error{"Module::bwd must be called with the same CUDA stream as Module::fwd"};
		}

		torch::Tensor dL_dinput;
		if (input.requires_grad()) {
			dL_dinput = torch::empty({ batch_size, input.size(1) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
		}
		torch::Tensor dL_dparams = torch::empty({ n_params() }, torch::TensorOptions().dtype(c10_param_precision()).device(torch::kCUDA));

		m_module->backward(stream, batch_size, input.requires_grad() ? dL_dinput.data_ptr<float>() : nullptr, void_data_ptr(dL_doutput), void_data_ptr(dL_dparams), input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params));
		m_fwd_was_called = false;

		return std::tuple<torch::Tensor, torch::Tensor>(dL_dinput, dL_dparams);
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

private:
	std::unique_ptr<tcnn::cpp::Module> m_module;

	cudaStream_t m_fwd_stream = nullptr;
	bool m_fwd_was_called = false;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	py::enum_<tcnn::cpp::EPrecision>(m, "Precision")
		.value("Fp32", tcnn::cpp::EPrecision::Fp32)
		.value("Fp16", tcnn::cpp::EPrecision::Fp16)
		.export_values()
		;

	// The python bindings expose TCNN's C++ API through
	// a single "Module" class that can act as the encoding,
	// the neural network, or a combined encoding + network
	// under the hood. The bindings don't need to concern
	// themselves with these implementation details, though.
	pybind11::class_<Module>(m, "Module")
		.def(
			pybind11::init<uint32_t, uint32_t, const nlohmann::json&, const nlohmann::json&>(),
			"Constructor for Encoding+Network combo",
			py::arg("n_input_dims"), py::arg("n_output_dims"), py::arg("encoding_config"), py::arg("network_config")
		)
		.def(
			pybind11::init<uint32_t, uint32_t, const nlohmann::json&>(),
			"Constructor for just the Network",
			py::arg("n_input_dims"), py::arg("n_output_dims"), py::arg("network_config")
		)
		.def(
			pybind11::init<uint32_t, const nlohmann::json&>(),
			"Constructor for just the Encoding",
			py::arg("n_input_dims"), py::arg("encoding_config")
		)
		.def("fwd", &Module::fwd)
		.def("bwd", &Module::bwd)
		.def("initial_params", &Module::initial_params)
		.def("n_input_dims", &Module::n_input_dims)
		.def("n_params", &Module::n_params)
		.def("param_precision", &Module::param_precision)
		.def("n_output_dims", &Module::n_output_dims)
		.def("output_precision", &Module::output_precision)
		;
}
