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

/** @file   cutlass_resnet.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  CUTLASS implementation of an optimized CutlassResNet. Supports online training
 *          and simultaneous inference.
 */

#include <tiny-cuda-nn/networks/cutlass_resnet.h>

#include <tiny-cuda-nn/cutlass_matmul.h>


TCNN_NAMESPACE_BEGIN

template <typename T, Activation input_activation, Activation output_activation>
CutlassResNet<T, input_activation, output_activation>::CutlassResNet(uint32_t input_width, uint32_t network_width, uint32_t output_width, uint32_t n_blocks, uint32_t n_matrices_per_block)
:
m_input_width{input_width},
m_network_width{network_width},
m_output_width{output_width},
m_n_blocks{n_blocks},
m_n_matrices_per_block{n_matrices_per_block}
{
	m_padded_output_width = next_multiple(m_output_width, tensorcore_width);

	// Create matrices related to weights
	m_weight_matrices.emplace_back(nullptr, network_width, input_width);
	m_weight_matrices_inference.emplace_back(nullptr, network_width, input_width);
	m_weight_matrices_full_precision.emplace_back(nullptr, network_width, input_width);
	m_gradient_matrices.emplace_back(nullptr, network_width, input_width);

	for (uint32_t i = 0; i < n_blocks * n_matrices_per_block; ++i) {
		m_weight_matrices.emplace_back(nullptr, network_width, network_width);
		m_weight_matrices_inference.emplace_back(nullptr, network_width, network_width);
		m_weight_matrices_full_precision.emplace_back(nullptr, network_width, network_width);
		m_gradient_matrices.emplace_back(nullptr, network_width, network_width);
	}

	m_weight_matrices.emplace_back(nullptr, m_padded_output_width, network_width);
	m_weight_matrices_inference.emplace_back(nullptr, m_padded_output_width, network_width);
	m_weight_matrices_full_precision.emplace_back(nullptr, m_padded_output_width, network_width);
	m_gradient_matrices.emplace_back(nullptr, m_padded_output_width, network_width);

	// Determine total number of memory entries and set it
	m_total_n_params = 0;
	for (const auto& m : m_weight_matrices) {
		m_total_n_params += m.n_elements();
	}


	// Buffers to keep data from the forward pass
	m_forward_tmp.resize(m_n_blocks * n_matrices_per_block + 1);
	m_backward_tmp.resize(m_n_blocks * n_matrices_per_block + 1);

	// Streams & events. Null for now to avoid clashes with external cuda calls

	// 1 fewer stream and event than the number of matrices, because the last
	// split-k matmul can use the regular training stream.
	m_training_splitk_streams.resize(m_weight_matrices.size());
	m_training_splitk_events.resize(m_weight_matrices.size());

	for (size_t i = 0; i < m_training_splitk_streams.size(); ++i) {
		CUDA_CHECK_THROW(cudaStreamCreate(&m_training_splitk_streams[i]));
		CUDA_CHECK_THROW(cudaEventCreate(&m_training_splitk_events[i]));
	}
}

template <typename T, Activation input_activation, Activation output_activation>
CutlassResNet<T, input_activation, output_activation>::~CutlassResNet() {
	for (size_t i = 0; i < m_training_splitk_streams.size(); ++i) {
		free_workspace(m_training_splitk_streams[i]);

		CUDA_CHECK_PRINT(cudaEventDestroy(m_training_splitk_events[i]));
		CUDA_CHECK_PRINT(cudaStreamDestroy(m_training_splitk_streams[i]));
	}
}

template <typename T, typename arch>
std::enable_if_t<std::is_same<arch, cutlass::arch::Sm75>::value && std::is_same<cutlass::half_t, T>::value> residual_block_2_inference(
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
			fc_multiply_b2b<Activation::None, FullLayerB2bPreReLU64, FullLayerB2bPreReLU64>(
				stream,
				transposed_input,
				weights1.transposed(),
				transposed_output,
				weights2.transposed(),
				transposed_input,
				transposed_output,
				(TypeCompute)0,
				(TypeCompute)1
			);
			break;
		case 128:
			fc_multiply_b2b<Activation::None, FullLayerB2bPreReLU128, FullLayerB2bPreReLU128>(
				stream,
				transposed_input,
				weights1.transposed(),
				transposed_output,
				weights2.transposed(),
				transposed_input,
				transposed_output,
				(TypeCompute)0,
				(TypeCompute)1
			);
			break;
		default:
			throw std::runtime_error{"Invalid layer size (must be 64, 128, or 256)."};
	}
}

template <typename T, typename arch>
std::enable_if_t<!(std::is_same<arch, cutlass::arch::Sm75>::value && std::is_same<cutlass::half_t, T>::value)> residual_block_2_inference(
	cudaStream_t,
	const GPUMatrix<T, MatrixLayout::ColumnMajor>&,
	const GPUMatrix<T, MatrixLayout::RowMajor>&,
	const GPUMatrix<T, MatrixLayout::RowMajor>&,
	GPUMatrix<T, MatrixLayout::ColumnMajor>&
) {
	// Dummy implementation for successful compilation when Sm75 is not available
}

template <typename T, Activation input_activation, Activation output_activation>
void CutlassResNet<T, input_activation, output_activation>::inference(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& input, GPUMatrix<float, MatrixLayout::ColumnMajor>& output) {
	inference_mixed_precision(stream, input, m_inference_output_tmp);

	const uint32_t n_elements = (uint32_t)output.n_elements();
	trim_and_cast<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_padded_output_width, m_output_width, m_inference_output_tmp.data(), output.data());
}

template <typename T, Activation input_activation, Activation output_activation>
void CutlassResNet<T, input_activation, output_activation>::inference_mixed_precision(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& input, GPUMatrix<T, MatrixLayout::ColumnMajor>& output, MatrixLayout output_layout) {
	// Various error checks
	if (input.m() != m_input_width) {
		throw std::runtime_error(std::string("Input has incorrect width: ") + std::to_string(input.m()) + "!=" + std::to_string(m_input_width));
	}

	if (&output != &m_inference_output_tmp && output.m() != m_output_width) {
		throw std::runtime_error(std::string("Output has incorrect width: ") + std::to_string(output.m()) + "!=" + std::to_string(m_output_width));
	}

	if (&output != &m_inference_output_tmp && input.n() != output.n()) {
		throw std::runtime_error(std::string("Input and output don't have matching batch size: ") + std::to_string(input.n()) + "!=" + std::to_string(output.n()));
	}

	// Make sure our teporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();
	if (m_inference_linear_tmp.n() != batch_size) {
		allocate_inference_buffers(batch_size);
	}

	const bool can_fuse_residual_block =
		std::is_same<SmArch, cutlass::arch::Sm75>::value &&
		std::is_same<cutlass::half_t, T>::value &&
		m_n_matrices_per_block == 2 &&
		(m_network_width == 128 || m_network_width == 64);

	// Run the actual network
	{
		// Input
		fc_multiply<input_activation, FullLayer>(stream, input_weight_matrix(true), input, m_inference_linear_tmp, m_inference_linear_tmp);

		// Res blocks
		for (uint32_t i = 0; i < m_n_blocks; ++i) {
			// Compute a residual block using a _single_ fused back-to-back matrix multiplication when applicable.
			if (can_fuse_residual_block) {
				residual_block_2_inference<T, SmArch>(
					stream,
					i == 0 ? m_inference_linear_tmp : m_inference_residual_tmp[i % 2],
					weight_matrix_at(true, i, 0),
					weight_matrix_at(true, i, 1),
					m_inference_residual_tmp[(i + 1) % 2]
				);;

				continue;
			}

			fc_multiply<Activation::None, FullLayerPreReLU>(stream, weight_matrix_at(true, i, 0), m_inference_linear_tmp, m_inference_residual_tmp[0]);

			for (uint32_t matrix_idx = 1; matrix_idx < m_n_matrices_per_block - 1; ++matrix_idx) {
				fc_multiply<Activation::None, FullLayerPreReLU>(stream, weight_matrix_at(true, i, matrix_idx), m_inference_residual_tmp[(matrix_idx+1) % 2], m_inference_residual_tmp[matrix_idx % 2]);
			}

			// In case there's just 1 matrix per block, the remaining addition must be done manually
			if (m_n_matrices_per_block == 1) {
				const uint32_t n_elements = (uint32_t)m_inference_residual_tmp.front().n_elements();
				add<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_inference_residual_tmp.front().data(), m_inference_linear_tmp.data());
			} else {
				uint32_t matrix_idx = m_n_matrices_per_block - 1;
				fc_multiply<Activation::None, FullLayerPreReLU>(
					stream,
					weight_matrix_at(true, i, matrix_idx),
					m_inference_residual_tmp[(matrix_idx+1) % 2],
					m_inference_linear_tmp,
					m_inference_linear_tmp,
					(TypeCompute)1
				); // beta==1 sums up the residual and linear parts
			}
		}

		auto& output_matrix = can_fuse_residual_block ? m_inference_residual_tmp[m_n_blocks % 2] : m_inference_linear_tmp;

		// Output
		if (output_layout == MatrixLayout::ColumnMajor) {
			fc_multiply<output_activation, LastLayer>(stream, output_weight_matrix(true), output_matrix, output, (T)m_output_activation_param);
		} else {
			auto rm_output{output.with_opposite_layout()};
			fc_multiply<output_activation, LastLayer>(stream, output_weight_matrix(true), output_matrix, rm_output, (T)m_output_activation_param);
		}
	}
}

template <typename T, Activation input_activation, Activation output_activation>
void CutlassResNet<T, input_activation, output_activation>::forward(cudaStream_t stream, const GPUMatrix<T, MatrixLayout::ColumnMajor>& input, GPUMatrix<T, MatrixLayout::ColumnMajor>& output, MatrixLayout output_layout, bool use_inference_matrices) {
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

	// Make sure our teporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();
	if (m_forward_tmp.front().n() != batch_size) {
		allocate_forward_buffers(batch_size);
	}

	const uint32_t n_elements = (uint32_t)m_forward_tmp.front().n_elements();

	// Run the actual network
	{
		auto& input_target = input_activation_value == Activation::None ? m_forward_tmp.front() : m_forward_input_tmp;
		fc_multiply<Activation::None, FullLayer>(stream, input_weight_matrix(use_inference_matrices), input, input_target);

		switch (input_activation_value) {
			case Activation::None: break;
			case Activation::Exponential: exp<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_forward_input_tmp.data(), m_forward_tmp.front().data()); break;
			case Activation::ReLU: relu<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_forward_input_tmp.data(), m_forward_tmp.front().data()); break;
			case Activation::Sine: sin<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_forward_input_tmp.data(), m_forward_tmp.front().data()); break;
			default: throw std::runtime_error{"Unsupported input activation."};
		}

		// Res blocks
		for (uint32_t i = 0; i < m_n_blocks; ++i) {
			uint32_t idx = i * m_n_matrices_per_block + 1;

			if (m_n_matrices_per_block == 1) {
				fc_multiply<Activation::None, FullLayerPreReLU>(stream, weight_matrix_at(use_inference_matrices, i, 0), m_forward_tmp.at(idx-1), m_forward_tmp.at(idx));
				add<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_forward_tmp.at(idx-1).data(), m_forward_tmp.at(idx).data());
			} else {
				fc_multiply<Activation::ReLU, FullLayerPreReLU>(stream, weight_matrix_at(use_inference_matrices, i, 0), m_forward_tmp.at(idx-1), m_forward_tmp.at(idx));

				for (uint32_t matrix_idx = 1; matrix_idx < m_n_matrices_per_block - 1; ++matrix_idx) {
					uint32_t fwd_idx = idx + matrix_idx;
					fc_multiply<Activation::ReLU, FullLayer>(stream, weight_matrix_at(use_inference_matrices, i, matrix_idx), m_forward_tmp.at(fwd_idx-1), m_forward_tmp.at(fwd_idx));
				}

				uint32_t matrix_idx = m_n_matrices_per_block - 1;
				uint32_t fwd_idx = idx + matrix_idx;
				fc_multiply<Activation::None, FullLayer>(
					stream,
					weight_matrix_at(use_inference_matrices, i, matrix_idx),
					m_forward_tmp.at(fwd_idx-1),
					m_forward_tmp.at(idx-1),
					m_forward_tmp.at(fwd_idx),
					(TypeCompute)1
				); // beta==1 sums up the residual and linear parts
			}

			// Retroactively apply ReLU to input. It's needed for backprop later.
			// We schedule it to the appropriate splitk stream, because only the later splitk operation depends on
			// the ReLU'd values to be present
			relu<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, m_training_splitk_streams.at(idx-1)>>>(n_elements, m_forward_tmp.at(idx-1).data());
		}

		// Output
		if (output_layout == MatrixLayout::ColumnMajor) {
			fc_multiply<output_activation, LastLayer>(stream, output_weight_matrix(use_inference_matrices), m_forward_tmp.back(), output, (T)m_output_activation_param);
		} else {
			auto rm_output{output.with_opposite_layout()};
			fc_multiply<output_activation, LastLayer>(stream, output_weight_matrix(use_inference_matrices), m_forward_tmp.back(), rm_output, (T)m_output_activation_param);
		}
	}
}

template <typename T, Activation input_activation, Activation output_activation>
void CutlassResNet<T, input_activation, output_activation>::backward(
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

	// Make sure our teporary buffers have the correct size for the given batch size
	uint32_t batch_size = dL_doutput.n();
	if (m_backward_tmp.front().n() != batch_size) {
		allocate_backward_buffers(batch_size);
	}

	// Compute transfer of output activation in-place... it's treated specially for performance reasons
	{
		const uint32_t n_elements = (uint32_t)dL_doutput.n_elements();
		switch (output_activation_value) {
			case Activation::None: break;
			case Activation::Exponential: exp_transfer_output<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, output.data(), dL_doutput.data(), m_backward_output_tmp.data()); break;
			case Activation::ReLU: relu_transfer_output<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, output.data(), dL_doutput.data(), m_backward_output_tmp.data()); break;
			default: throw std::runtime_error{"Unsupported output activation."};
		}
	}

	// Backprop
	// - weight_gradient.T = input_activation * output_gradient.T
	// - input_gradient = weights.T * output_gradient
	// - RELU: pre_activation_gradinet = post_activation_gradient if val > 0 else 0

	{
		// T normalization = (T)(1.0f / batch_size);
		T normalization = (T)(1.0f);

		const uint32_t n_elements = (uint32_t)m_backward_tmp.front().n_elements();

		int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);

		const GPUMatrix<T, MatrixLayout::ColumnMajor>& tmp_dL_doutput = output_activation_value == Activation::None ? dL_doutput : m_backward_output_tmp;
		auto rm_tmp_dL_doutput{tmp_dL_doutput.with_opposite_layout()};

		if (compute_param_gradients) {
			// Output layer
			cudaEventRecord(m_training_splitk_events.back(), stream);
			cudaStreamWaitEvent(m_training_splitk_streams.back(), m_training_splitk_events.back(), 0);
			if (output_layout == MatrixLayout::ColumnMajor) {
				fc_multiply_split_k<Activation::None, LastLayerK>(m_training_splitk_streams.back(), tmp_dL_doutput, m_forward_tmp.back().transposed(), output_gradient_matrix(), split_k_factor, normalization);
			} else {
				fc_multiply_split_k<Activation::None, LastLayerK>(m_training_splitk_streams.back(), rm_tmp_dL_doutput, m_forward_tmp.back().transposed(), output_gradient_matrix(), split_k_factor, normalization);
			}
			cudaEventRecord(m_training_splitk_events.back(), m_training_splitk_streams.back());
		}

		if (output_layout == MatrixLayout::ColumnMajor) {
			fc_multiply<Activation::None, FullLayer>(stream, output_weight_matrix(use_inference_matrices).transposed(), tmp_dL_doutput, m_backward_tmp.back());
		} else {
			fc_multiply<Activation::None, FullLayer>(stream, output_weight_matrix(use_inference_matrices).transposed(), rm_tmp_dL_doutput, m_backward_tmp.back());
		}

		// Res blocks
		for (uint32_t i = 0; i < m_n_blocks; ++i) {
			uint32_t block_idx = m_n_blocks - i - 1;
			uint32_t idx = block_idx * m_n_matrices_per_block + 1;

			for (uint32_t j = 0; j < m_n_matrices_per_block; ++j) {
				uint32_t matrix_idx = m_n_matrices_per_block - 1 - j;
				uint32_t fwd_idx = idx + matrix_idx;

				if (compute_param_gradients) {
					cudaEventRecord(m_training_splitk_events.at(fwd_idx), stream);
					cudaStreamWaitEvent(m_training_splitk_streams.at(fwd_idx), m_training_splitk_events.at(fwd_idx), 0);
					fc_multiply_split_k<Activation::None, FullLayerK>(m_training_splitk_streams.at(fwd_idx), m_backward_tmp.at(fwd_idx), m_forward_tmp.at(fwd_idx-1).transposed(), gradient_matrix_at(block_idx, matrix_idx), split_k_factor, normalization);
					cudaEventRecord(m_training_splitk_events.at(fwd_idx), m_training_splitk_streams.at(fwd_idx));
				}

				fc_multiply<Activation::ReLUTransfer, FullLayer>(stream, weight_matrix_at(use_inference_matrices, block_idx, matrix_idx).transposed(), m_backward_tmp.at(fwd_idx), m_forward_tmp.at(fwd_idx-1), m_backward_tmp.at(fwd_idx-1));
			}

			add<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_backward_tmp.at(idx+m_n_matrices_per_block-1).data(), m_backward_tmp.at(idx-1).data());
		}

		switch (input_activation_value) {
			case Activation::None: break;
			case Activation::ReLU: relu_transfer<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_forward_input_tmp.data(), m_backward_tmp.front().data()); break;
			case Activation::Exponential: exp_transfer<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_forward_input_tmp.data(), m_backward_tmp.front().data()); break;
			case Activation::Sine: sin_transfer<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_forward_input_tmp.data(), m_backward_tmp.front().data()); break;
			default: throw std::runtime_error{"Invalid input activation"};
		};

		if (compute_param_gradients) {
			cudaEventRecord(m_training_splitk_events.front(), stream);
			cudaStreamWaitEvent(m_training_splitk_streams.front(), m_training_splitk_events.front(), 0);
			fc_multiply_split_k<Activation::None, FullLayerK>(m_training_splitk_streams.front(), m_backward_tmp.front(), input.transposed(), input_gradient_matrix(), split_k_factor, normalization);
			cudaEventRecord(m_training_splitk_events.front(), m_training_splitk_streams.front());
		}

		// If requested, compute sensitivity of loss w.r.t. inputs
		if (dL_dinput) {
			// TODO: optimization opportunity to only compute sensitivity w.r.t selected SUBSET of inputs. Useful for NFs, where conditional dims stay the same.
			fc_multiply<Activation::None, FullLayer>(stream, input_weight_matrix(use_inference_matrices).transposed(), m_backward_tmp.front(), *dL_dinput);
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

template <typename T, Activation input_activation, Activation output_activation>
void CutlassResNet<T, input_activation, output_activation>::allocate_inference_buffers(uint32_t batch_size) {
	m_inference_linear_tmp.set_size(m_network_width, batch_size);
	m_inference_residual_tmp[0].set_size(m_network_width, batch_size);
	m_inference_residual_tmp[1].set_size(m_network_width, batch_size);
	m_inference_output_tmp.set_size(m_padded_output_width, batch_size);

	GPUMatrixBase::allocate_shared_memory(
		m_inference_buffer,
		{
			&m_inference_linear_tmp,
			&m_inference_residual_tmp[0],
			&m_inference_residual_tmp[1],
			&m_inference_output_tmp,
		}
	);
}

template <typename T, Activation input_activation, Activation output_activation>
void CutlassResNet<T, input_activation, output_activation>::allocate_forward_buffers(uint32_t batch_size) {
	std::vector<GPUMatrixBase*> matrix_pointers = {&m_forward_input_tmp};

	m_forward_input_tmp.set_size(m_network_width, batch_size);
	for (uint32_t i = 0; i < (uint32_t)m_forward_tmp.size(); ++i) {
		m_forward_tmp[i].set_size(m_network_width, batch_size);
		matrix_pointers.emplace_back(&m_forward_tmp[i]);
	}

	GPUMatrixBase::allocate_shared_memory(m_forward_buffer, matrix_pointers);
}

template <typename T, Activation input_activation, Activation output_activation>
void CutlassResNet<T, input_activation, output_activation>::allocate_backward_buffers(uint32_t batch_size) {
	std::vector<GPUMatrixBase*> matrix_pointers = {&m_backward_output_tmp};

	m_backward_output_tmp.set_size(m_padded_output_width, batch_size);
	for (uint32_t i = 0; i < (uint32_t)m_backward_tmp.size(); ++i) {
		m_backward_tmp[i].set_size(m_network_width, batch_size);
		matrix_pointers.emplace_back(&m_backward_tmp[i]);
	}

	GPUMatrixBase::allocate_shared_memory(m_backward_buffer, matrix_pointers);
}

template <typename T, Activation input_activation, Activation output_activation>
void CutlassResNet<T, input_activation, output_activation>::initialize_params(std::mt19937& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale) {
	std::cout << "CutlassResNet: initializing " << m_total_n_params << " params" << std::endl;

	size_t current_pos = 0;
	for (size_t i = 0; i < m_weight_matrices.size(); ++i) {
		m_weight_matrices[i].set_data(params + current_pos);
		m_weight_matrices_inference[i].set_data(inference_params + current_pos);
		m_weight_matrices_full_precision[i].set_data(params_full_precision + current_pos);
		m_gradient_matrices[i].set_data(gradients + current_pos);
		current_pos += m_weight_matrices[i].n_elements();
	}

	// Initialize the params
	for (size_t i = 0; i < m_weight_matrices_full_precision.size(); ++i) {
		if (i == 0 && input_activation_value == Activation::Sine) {
			m_weight_matrices_full_precision[i].initialize_siren_uniform_first(rnd, scale);
		} else {
			m_weight_matrices_full_precision[i].initialize_xavier_uniform(rnd, scale);
		}
	}
}

// Explicitly instantiate resnet classes.
template class CutlassResNet<network_precision_t, Activation::None, Activation::Exponential>;
template class CutlassResNet<network_precision_t, Activation::None, Activation::None>;

TCNN_NAMESPACE_END
