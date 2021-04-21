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

/** @file   cutlass_mlp.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  CUTLASS implementation of an optimized multi-layer perceptron. Supports online training
 *          and simultaneous inference.
 */

#include <tiny-cuda-nn/networks/cutlass_mlp.h>

#include <tiny-cuda-nn/cutlass_matmul.h>


TCNN_NAMESPACE_BEGIN

void cutlass_free_workspace(cudaStream_t stream) {
	free_workspace(stream);
}

template <typename T>
CutlassMLP<T>::CutlassMLP(
	uint32_t input_width,
	uint32_t network_width,
	uint32_t output_width,
	uint32_t n_hidden_layers,
	Activation activation,
	Activation output_activation
) :
m_input_width{input_width},
m_network_width{network_width},
m_output_width{output_width},
m_n_hidden_layers{n_hidden_layers},
m_activation{activation},
m_output_activation{output_activation}
{
	m_padded_output_width = next_multiple(m_output_width, tensorcore_width);

	if (n_hidden_layers > 0) {
		m_n_hidden_matmuls = n_hidden_layers-1;
	} else {
		m_n_hidden_matmuls = 0;
	}

	// Create matrices related to weights
	if (n_hidden_layers == 0) {
		m_weight_matrices.emplace_back(nullptr, m_padded_output_width, m_input_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_padded_output_width, m_input_width);
		m_weight_matrices_full_precision.emplace_back(nullptr, m_padded_output_width, m_input_width);
		m_gradient_matrices.emplace_back(nullptr, m_padded_output_width, m_input_width);
	} else {
		m_weight_matrices.emplace_back(nullptr, m_network_width, m_input_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_input_width);
		m_weight_matrices_full_precision.emplace_back(nullptr, m_network_width, m_input_width);
		m_gradient_matrices.emplace_back(nullptr, m_network_width, m_input_width);

		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
			m_weight_matrices.emplace_back(nullptr, m_network_width, m_network_width);
			m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_network_width);
			m_weight_matrices_full_precision.emplace_back(nullptr, m_network_width, m_network_width);
			m_gradient_matrices.emplace_back(nullptr, m_network_width, m_network_width);
		}

		m_weight_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_padded_output_width, m_network_width);
		m_weight_matrices_full_precision.emplace_back(nullptr, m_padded_output_width, m_network_width);
		m_gradient_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);
	}

	// Determine total number of memory entries and set it
	m_total_n_params = 0;
	for (const auto& m : m_weight_matrices) {
		m_total_n_params += m.n_elements();
	}

	// Buffers to keep data from the forward and backward pass
	m_forward_tmp.resize(m_n_hidden_layers * 2);
	m_backward_tmp.resize(m_n_hidden_layers * 2);

	// 1 stream per matrix.
	m_training_splitk_streams.resize(m_n_hidden_layers + 1);
	m_training_splitk_events.resize(m_n_hidden_layers + 1);

	for (size_t i = 0; i < m_training_splitk_streams.size(); ++i) {
		CUDA_CHECK_THROW(cudaStreamCreate(&m_training_splitk_streams[i]));
		CUDA_CHECK_THROW(cudaEventCreate(&m_training_splitk_events[i]));
	}
}

template <typename T>
CutlassMLP<T>::~CutlassMLP() {
	for (size_t i = 0; i < m_training_splitk_streams.size(); ++i) {
		free_workspace(m_training_splitk_streams[i]);

		CUDA_CHECK_PRINT(cudaEventDestroy(m_training_splitk_events[i]));
		CUDA_CHECK_PRINT(cudaStreamDestroy(m_training_splitk_streams[i]));
	}
}

template <typename T, typename arch, Activation activation>
std::enable_if_t<std::is_same<arch, cutlass::arch::Sm75>::value && std::is_same<cutlass::half_t, T>::value> fused_2_inference(
	cudaStream_t stream,
	const GPUMatrix<T, MatrixLayout::ColumnMajor>& input,
	const GPUMatrix<T, MatrixLayout::RowMajor>& weights1,
	const GPUMatrix<T, MatrixLayout::RowMajor>& weights2,
	GPUMatrix<T, MatrixLayout::ColumnMajor>& output
) {
	const auto transposed_input = input.transposed();
	auto transposed_output = output.transposed();

	switch (weights1.n()) {
		case 64:
			fc_multiply_b2b<activation, FullLayerB2b64, FullLayerB2b64>(
				stream,
				transposed_input,
				weights1.transposed(),
				transposed_output,
				weights2.transposed(),
				transposed_output,
				transposed_output,
				(TypeCompute)0,
				(TypeCompute)0
			);
			break;
		case 128:
			fc_multiply_b2b<activation, FullLayerB2b128, FullLayerB2b128>(
				stream,
				transposed_input,
				weights1.transposed(),
				transposed_output,
				weights2.transposed(),
				transposed_output,
				transposed_output,
				(TypeCompute)0,
				(TypeCompute)0
			);
			break;
		default:
			throw std::runtime_error{"Invalid layer size (must be 64, 128, or 256)."};
	}
}

template <typename T, typename arch, Activation activation>
std::enable_if_t<!(std::is_same<arch, cutlass::arch::Sm75>::value && std::is_same<cutlass::half_t, T>::value)> fused_2_inference(
	cudaStream_t,
	const GPUMatrix<T, MatrixLayout::ColumnMajor>&,
	const GPUMatrix<T, MatrixLayout::RowMajor>&,
	const GPUMatrix<T, MatrixLayout::RowMajor>&,
	GPUMatrix<T, MatrixLayout::ColumnMajor>&
) {
	// Dummy implementation for successful compilation when Sm75 is not available
}

template <typename T>
void CutlassMLP<T>::inference(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& input, GPUMatrix<float, MatrixLayout::ColumnMajor>& output) {
	inference_mixed_precision(stream, input, m_inference_output_tmp);

	const uint32_t n_elements = (uint32_t)output.n_elements();
	trim_and_cast<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_padded_output_width, m_output_width, m_inference_output_tmp.data(), output.data());
}

template <typename T>
void CutlassMLP<T>::inference_mixed_precision(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& input, GPUMatrix<T, MatrixLayout::ColumnMajor>& output, MatrixLayout output_layout) {
	// Various error checks
	if (input.m() != m_input_width) {
		throw std::runtime_error(std::string("Input has incorrect width: ") + std::to_string(input.m()) + "!=" + std::to_string(m_input_width));
	}

	if (&output != &m_inference_output_tmp && output.m() != m_padded_output_width) {
		throw std::runtime_error(std::string("Output has incorrect width: ") + std::to_string(output.m()) + "!=" + std::to_string(m_output_width));
	}

	if (&output != &m_inference_output_tmp && input.n() != output.n()) {
		throw std::runtime_error(std::string("Input and output don't have matching batch size: ") + std::to_string(input.n()) + "!=" + std::to_string(output.n()));
	}

	bool did_reallocate = false;

	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();
	if (m_inference_output_tmp.n() != batch_size) {
		allocate_inference_buffers(batch_size);
		did_reallocate = true;
	}

	// If there are no hidden layers, the network is just a simple matmul.
	if (m_n_hidden_layers == 0) {
		if (output_layout == MatrixLayout::ColumnMajor) {
			compute_inference_layer<LastLayer>(stream, m_output_activation, input_weight_matrix(true), input, output, (T)m_output_activation_param);
		} else {
			auto rm_output{output.with_opposite_layout()};
			compute_inference_layer<LastLayer>(stream, m_output_activation, input_weight_matrix(true), input, rm_output, (T)m_output_activation_param);
		}
		return;
	}

	m_inference_graph.capture_and_execute(stream, did_reallocate, [&]() {
		// We can troubleshoot matmul fusion once MLPs actually become interesting.
		const bool can_fuse_pairs =
		std::is_same<SmArch, cutlass::arch::Sm75>::value &&
		std::is_same<T, cutlass::half_t>::value &&
		(m_network_width == 128 || m_network_width == 64);

		// Run the actual network
		{
			uint32_t tmp_idx = 0;

			// Input layer
			compute_inference_layer<FullLayer>(stream, m_activation, input_weight_matrix(true), input, m_inference_tmp[tmp_idx++ % 2]);

			// Hidden layers
			for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
				if (can_fuse_pairs && i < m_n_hidden_matmuls - 1) {
					switch (m_activation) {
						case Activation::None: fused_2_inference<T, SmArch, Activation::None>(
							stream,
							m_inference_tmp[(tmp_idx + 1) % 2],
							weight_matrix_at(true, i),
							weight_matrix_at(true, i + 1),
							m_inference_tmp[tmp_idx % 2]
						); break;

						case Activation::ReLU: fused_2_inference<T, SmArch, Activation::ReLU>(
							stream,
							m_inference_tmp[(tmp_idx + 1) % 2],
							weight_matrix_at(true, i),
							weight_matrix_at(true, i + 1),
							m_inference_tmp[tmp_idx % 2]
						); break;

						case Activation::Sine: fused_2_inference<T, SmArch, Activation::Sine>(
							stream,
							m_inference_tmp[(tmp_idx + 1) % 2],
							weight_matrix_at(true, i),
							weight_matrix_at(true, i + 1),
							m_inference_tmp[tmp_idx % 2]
						); break;

						case Activation::Exponential: fused_2_inference<T, SmArch, Activation::Exponential>(
							stream,
							m_inference_tmp[(tmp_idx + 1) % 2],
							weight_matrix_at(true, i),
							weight_matrix_at(true, i + 1),
							m_inference_tmp[tmp_idx % 2]
						); break;

						default: throw std::runtime_error{"Unsupported activation."};
					}
					++i;
					++tmp_idx;
					continue;
				}

				compute_inference_layer<FullLayer>(stream, m_activation, weight_matrix_at(true, i), m_inference_tmp[(tmp_idx + 1) % 2], m_inference_tmp[tmp_idx % 2]);
				++tmp_idx;
			}

			// Output
			if (output_layout == MatrixLayout::ColumnMajor) {
				compute_inference_layer<LastLayer>(stream, m_output_activation, output_weight_matrix(true), m_inference_tmp[(tmp_idx + 1) % 2], output, (T)m_output_activation_param);
			} else {
				auto rm_output{output.with_opposite_layout()};
				compute_inference_layer<LastLayer>(stream, m_output_activation, output_weight_matrix(true), m_inference_tmp[(tmp_idx + 1) % 2], rm_output, (T)m_output_activation_param);
			}
		}
	});
}

template <typename T>
void CutlassMLP<T>::forward(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& input, GPUMatrix<T, MatrixLayout::ColumnMajor>& output, MatrixLayout output_layout, bool use_inference_matrices) {
	// Various error checks
	if (input.m() != m_input_width) {
		throw std::runtime_error(std::string("Input has incorrect width: ") + std::to_string(input.m()) + "!=" + std::to_string(m_input_width));
	}

	if (output.m() != m_padded_output_width) {
		throw std::runtime_error(std::string("Output has incorrect width (must be padded): ") + std::to_string(output.m()) + "!=" + std::to_string(m_padded_output_width));
	}

	if (input.n() != output.n()) {
		throw std::runtime_error(std::string("Input and output don't have matching batch size: ") + std::to_string(input.n()) + "!=" + std::to_string(output.n()));
	}

	// If there are no hidden layers, the network is just a simple matmul. No tmp buffers required
	if (m_n_hidden_layers == 0) {
		if (output_layout == MatrixLayout::ColumnMajor) {
			compute_layer<LastLayer>(stream, false, m_output_activation, input_weight_matrix(use_inference_matrices), input, output, output, (T)m_output_activation_param);
		} else {
			auto rm_output{output.with_opposite_layout()};
			compute_layer<LastLayer>(stream, false, m_output_activation, input_weight_matrix(use_inference_matrices), input, rm_output, rm_output, (T)m_output_activation_param);
		}
		return;
	}

	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();
	if (m_forward_tmp.front().n() != batch_size) {
		allocate_forward_buffers(batch_size);
	}

	// Run the actual network
	uint32_t tmp_idx = 0;

	bool fused = compute_layer<FullLayer>(stream, false, m_activation, input_weight_matrix(use_inference_matrices), input, m_forward_tmp.at(tmp_idx), m_forward_tmp.at(tmp_idx+1));
	tmp_idx += fused ? 1 : 2;

	// layers
	for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
		fused = compute_layer<FullLayer>(stream, false, m_activation, weight_matrix_at(use_inference_matrices, i), m_forward_tmp.at(tmp_idx-1), m_forward_tmp.at(tmp_idx), m_forward_tmp.at(tmp_idx+1));
		tmp_idx += fused ? 1 : 2;
	}

	// Output
	if (output_layout == MatrixLayout::ColumnMajor) {
		compute_layer<LastLayer>(stream, false, m_output_activation, output_weight_matrix(use_inference_matrices), m_forward_tmp.at(tmp_idx-1), output, output, (T)m_output_activation_param);
	} else {
		auto rm_output{output.with_opposite_layout()};
		compute_layer<LastLayer>(stream, false, m_output_activation, output_weight_matrix(use_inference_matrices), m_forward_tmp.at(tmp_idx-1), rm_output, rm_output, (T)m_output_activation_param);
	}
}

template <typename T>
void CutlassMLP<T>::compute_activation_transfer(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& values, GPUMatrix<T, MatrixLayout::ColumnMajor>& gradients) {
	if (values.n() != gradients.n() || values.m() != gradients.m()) {
		throw std::runtime_error(std::string("Values and gradients don't have matching size: ") + std::to_string(values.n()) + "!=" + std::to_string(gradients.n()));
	}

	const uint32_t width = values.m();
	const uint32_t n_elements = (uint32_t)values.n_elements();

	switch (m_activation) {
		case Activation::None: break;
		case Activation::Exponential: exp_transfer<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, values.data(), gradients.data()); break;
		case Activation::ReLU: relu_transfer<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, values.data(), gradients.data()); break;
		case Activation::Sine: sin_transfer<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, values.data(), gradients.data()); break;
		default: throw std::runtime_error{"Unsupported activation."};
	}
}

template <typename T>
void CutlassMLP<T>::backward(
	cudaStream_t stream,
	const GPUMatrix<T, MatrixLayout::ColumnMajor>& input,
	const GPUMatrix<T, MatrixLayout::ColumnMajor>& output,
	const GPUMatrix<T, MatrixLayout::ColumnMajor>& dL_doutput,
	GPUMatrix<T, MatrixLayout::ColumnMajor>* dL_dinput,
	MatrixLayout output_layout,
	bool use_inference_matrices,
	bool compute_param_gradients
) {
	if (dL_doutput.m() != m_padded_output_width) {
		throw std::runtime_error(std::string("Output gradients have incorrect width (must be padded): ") + std::to_string(dL_doutput.m()) + "!=" + std::to_string(m_padded_output_width));
	}

	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = dL_doutput.n();
	if (m_backward_output_tmp.n() != batch_size) {
		allocate_backward_buffers(batch_size);
	}

	bool can_fuse_activation = m_activation == Activation::ReLU || m_activation == Activation::None;
	Activation transfer_activation = m_activation == Activation::None ? Activation::None : Activation::ReLUTransfer;

	// Compute transfer of output activation in-place... it's treated specially for performance reasons
	const uint32_t n_elements = (uint32_t)dL_doutput.n_elements();
	switch (m_output_activation) {
		case Activation::None: break;
		case Activation::Exponential: exp_transfer_output<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, output.data(), dL_doutput.data(), m_backward_output_tmp.data()); break;
		case Activation::ReLU: relu_transfer_output<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, output.data(), dL_doutput.data(), m_backward_output_tmp.data()); break;
		default: throw std::runtime_error{"Unsupported output activation."};
	}

	// Backprop
	// - weight_gradient.T = activation * output_gradient.T
	// - input_gradient = weights.T * output_gradient
	// - RELU: pre_activation_gradinet = post_activation_gradient if val > 0 else 0

	{
		// T normalization = (T)(1.0f / batch_size);
		T normalization = (T)(1.0f);

		int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);

		const GPUMatrix<T, MatrixLayout::ColumnMajor>& tmp_dL_doutput = m_output_activation == Activation::None ? dL_doutput : m_backward_output_tmp;
		auto rm_tmp_dL_doutput{tmp_dL_doutput.with_opposite_layout()};

		// If there are no hidden layers, the network is just a simple matmul
		if (m_n_hidden_layers == 0) {
			if (compute_param_gradients) {
				cudaEventRecord(m_training_splitk_events.at(0), stream);
				cudaStreamWaitEvent(m_training_splitk_streams.at(0), m_training_splitk_events.at(0), 0);

				// Compute weight gradients
				if (output_layout == MatrixLayout::ColumnMajor) {
					fc_multiply_split_k<Activation::None, LastLayerK>(m_training_splitk_streams.at(0), tmp_dL_doutput, input.transposed(), input_gradient_matrix(), split_k_factor, normalization);
				} else {
					fc_multiply_split_k<Activation::None, LastLayerK>(m_training_splitk_streams.at(0), rm_tmp_dL_doutput, input.transposed(), input_gradient_matrix(), split_k_factor, normalization);
				}

				cudaEventRecord(m_training_splitk_events.at(0), m_training_splitk_streams.at(0));
			}

			if (dL_dinput) {
				if (output_layout == MatrixLayout::ColumnMajor) {
					fc_multiply<Activation::None, FullLayer>(stream, input_weight_matrix(use_inference_matrices).transposed(), tmp_dL_doutput, *dL_dinput);
				} else {
					fc_multiply<Activation::None, FullLayer>(stream, input_weight_matrix(use_inference_matrices).transposed(), rm_tmp_dL_doutput, *dL_dinput);
				}
			}

			if (compute_param_gradients) {
				cudaStreamWaitEvent(stream, m_training_splitk_events.at(0), 0);
			}
			return;
		}

		uint32_t tmp_idx = (can_fuse_activation ? (m_n_hidden_matmuls+1) : ((m_n_hidden_matmuls+1) * 2)) - 1;
		uint32_t backward_tmp_idx = 0;

		if (compute_param_gradients) {
			// Output layer
			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
			cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);

			// Compute weight gradients
			if (output_layout == MatrixLayout::ColumnMajor) {
				fc_multiply_split_k<Activation::None, LastLayerK>(m_training_splitk_streams.at(backward_tmp_idx), tmp_dL_doutput, m_forward_tmp.at(tmp_idx).transposed(), output_gradient_matrix(), split_k_factor, normalization);
			} else {
				fc_multiply_split_k<Activation::None, LastLayerK>(m_training_splitk_streams.at(backward_tmp_idx), rm_tmp_dL_doutput, m_forward_tmp.at(tmp_idx).transposed(), output_gradient_matrix(), split_k_factor, normalization);
			}

			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));
		}

		if (!can_fuse_activation) {
			if (output_layout == MatrixLayout::ColumnMajor) {
				fc_multiply<Activation::None, FullLayer>(stream, output_weight_matrix(use_inference_matrices).transposed(), tmp_dL_doutput, m_backward_tmp.at(backward_tmp_idx));
			} else {
				fc_multiply<Activation::None, FullLayer>(stream, output_weight_matrix(use_inference_matrices).transposed(), rm_tmp_dL_doutput, m_backward_tmp.at(backward_tmp_idx));
			}
			compute_activation_transfer(stream, m_forward_tmp.at(tmp_idx-1), m_backward_tmp.at(backward_tmp_idx));
		} else {
			switch (transfer_activation) {
				case Activation::None:
					if (output_layout == MatrixLayout::ColumnMajor) {
						fc_multiply<Activation::None, FullLayer>(stream, output_weight_matrix(use_inference_matrices).transposed(), tmp_dL_doutput, m_forward_tmp.at(tmp_idx), m_backward_tmp.at(backward_tmp_idx));
					} else {
						fc_multiply<Activation::None, FullLayer>(stream, output_weight_matrix(use_inference_matrices).transposed(), rm_tmp_dL_doutput, m_forward_tmp.at(tmp_idx), m_backward_tmp.at(backward_tmp_idx));
					}
					break;
				case Activation::ReLUTransfer:
					if (output_layout == MatrixLayout::ColumnMajor) {
						fc_multiply<Activation::ReLUTransfer, FullLayer>(stream, output_weight_matrix(use_inference_matrices).transposed(), tmp_dL_doutput, m_forward_tmp.at(tmp_idx), m_backward_tmp.at(backward_tmp_idx));
					} else {
						fc_multiply<Activation::ReLUTransfer, FullLayer>(stream, output_weight_matrix(use_inference_matrices).transposed(), rm_tmp_dL_doutput, m_forward_tmp.at(tmp_idx), m_backward_tmp.at(backward_tmp_idx));
					}
					break;
				default: throw std::runtime_error{"Unsupported activation transfer."};
			};
		}

		tmp_idx -= can_fuse_activation ? 1 : 2;
		++backward_tmp_idx;

		// layers
		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
			uint32_t matrix_idx = m_n_hidden_matmuls - i - 1;

			if (compute_param_gradients) {
				cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
				cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);
				fc_multiply_split_k<Activation::None, FullLayerK>(m_training_splitk_streams.at(backward_tmp_idx), m_backward_tmp.at(backward_tmp_idx-1), m_forward_tmp.at(tmp_idx).transposed(), gradient_matrix_at(matrix_idx), split_k_factor, normalization);
				cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));
			}

			if (!can_fuse_activation) {
				fc_multiply<Activation::None, FullLayer>(stream, weight_matrix_at(use_inference_matrices, matrix_idx).transposed(), m_backward_tmp.at(backward_tmp_idx-1), m_backward_tmp.at(backward_tmp_idx));
				compute_activation_transfer(stream, m_forward_tmp.at(tmp_idx-1), m_backward_tmp.at(backward_tmp_idx));
			} else {
				switch (transfer_activation) {
					case Activation::None: fc_multiply<Activation::None, FullLayer>(stream, weight_matrix_at(use_inference_matrices, matrix_idx).transposed(), m_backward_tmp.at(backward_tmp_idx-1), m_forward_tmp.at(tmp_idx), m_backward_tmp.at(backward_tmp_idx)); break;
					case Activation::ReLUTransfer: fc_multiply<Activation::ReLUTransfer, FullLayer>(stream, weight_matrix_at(use_inference_matrices, matrix_idx).transposed(), m_backward_tmp.at(backward_tmp_idx-1), m_forward_tmp.at(tmp_idx), m_backward_tmp.at(backward_tmp_idx)); break;
					default: throw std::runtime_error{"Unsupported activation transfer."};
				};
			}

			tmp_idx -= can_fuse_activation ? 1 : 2;
			++backward_tmp_idx;
		}

		if (compute_param_gradients) {
			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
			cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);
			fc_multiply_split_k<Activation::None, FullLayerK>(m_training_splitk_streams.at(backward_tmp_idx), m_backward_tmp.at(backward_tmp_idx-1), input.transposed(), input_gradient_matrix(), split_k_factor, normalization);
			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));
		}

		// If requested, compute sensitivity of loss w.r.t. inputs
		if (dL_dinput) {
			// optimization opportunity to only compute sensitivity w.r.t selected SUBSET of inputs. Useful for NFs, where conditional dims stay the same.
			fc_multiply<Activation::None, FullLayer>(stream, input_weight_matrix(use_inference_matrices).transposed(), m_backward_tmp.at(backward_tmp_idx-1), *dL_dinput);
		}
	}

	if (compute_param_gradients) {
		// All the per-layer split-k matrix multiplications summing over
		// the batch are computed in parallel streams to the actual
		// backpropagation. Here, we need to wait for all of these to complete.
		for (auto& event : m_training_splitk_events) {
			cudaStreamWaitEvent(stream, event, 0);
		}
	}
}

template <typename T>
void CutlassMLP<T>::allocate_inference_buffers(uint32_t batch_size) {
	m_inference_tmp[0].set_size(m_network_width, batch_size);
	m_inference_tmp[1].set_size(m_network_width, batch_size);
	m_inference_output_tmp.set_size(m_padded_output_width, batch_size);

	GPUMatrixBase::allocate_shared_memory(
		m_inference_buffer,
		{
			&m_inference_tmp[0],
			&m_inference_tmp[1],
			&m_inference_output_tmp,
		}
	);
}

template <typename T>
void CutlassMLP<T>::allocate_forward_buffers(uint32_t batch_size) {
	for (size_t i = 0; i < m_forward_tmp.size(); ++i) {
		m_forward_tmp[i].set_size(m_network_width, batch_size);
	}

	GPUMatrixBase::allocate_shared_memory(m_forward_buffer, m_forward_tmp);
}

template <typename T>
void CutlassMLP<T>::allocate_backward_buffers(uint32_t batch_size) {
	std::vector<GPUMatrixBase*> matrix_pointers = {&m_backward_output_tmp};

	m_backward_output_tmp.set_size(m_padded_output_width, batch_size);
	for (uint32_t i = 0; i < (uint32_t)m_backward_tmp.size(); ++i) {
		m_backward_tmp[i].set_size(m_network_width, batch_size);
		matrix_pointers.emplace_back(&m_backward_tmp[i]);
	}

	GPUMatrixBase::allocate_shared_memory(m_backward_buffer, matrix_pointers);
}

template <typename T>
void CutlassMLP<T>::initialize_params(float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale) {
	std::cout << "CutlassMLP: initializing " << m_total_n_params << " params" << std::endl;

	size_t current_pos = 0;
	for (size_t i = 0; i < m_weight_matrices.size(); ++i) {
		m_weight_matrices[i].set_data(params + current_pos);
		m_weight_matrices_inference[i].set_data(inference_params + current_pos);
		m_weight_matrices_full_precision[i].set_data(params_full_precision + current_pos);
		m_gradient_matrices[i].set_data(gradients + current_pos);

		current_pos += m_weight_matrices[i].n_elements();
	}

	// Initialize params
	for (size_t i = 0; i < m_weight_matrices_full_precision.size(); ++i) {
		if (m_activation == Activation::Sine) {
			if (i == 0) {
				m_weight_matrices_full_precision[i].initialize_siren_uniform_first(scale);
			} else {
				m_weight_matrices_full_precision[i].initialize_siren_uniform(scale);
			}
		} else {
			m_weight_matrices_full_precision[i].initialize_xavier_uniform(scale);
		}
	}
}

// Explicitly instantiate CutlassMLP classes.
template class CutlassMLP<network_precision_t>;

TCNN_NAMESPACE_END
