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

/** @file   cutlass_resnet.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  CUTLASS implementation of an optimized fully connected network with
            residual connections. Supports online training and simultaneous inference.
 */

#include <tiny-cuda-nn/networks/cutlass_resnet.h>

#include <tiny-cuda-nn/cutlass_matmul.h>

TCNN_NAMESPACE_BEGIN

template <typename T, Activation input_activation>
CutlassResNet<T, input_activation>::CutlassResNet(
	uint32_t input_width,
	uint32_t network_width,
	uint32_t output_width,
	uint32_t n_blocks,
	uint32_t n_matrices_per_block,
	Activation output_activation
) :
m_input_width{input_width},
m_network_width{network_width},
m_output_width{output_width},
m_n_blocks{n_blocks},
m_n_matrices_per_block{n_matrices_per_block},
m_output_activation{output_activation}
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

template <typename T, Activation input_activation>
CutlassResNet<T, input_activation>::~CutlassResNet() {
	for (size_t i = 0; i < m_training_splitk_streams.size(); ++i) {
		free_gpu_memory_arena(m_training_splitk_streams[i]);

		CUDA_CHECK_PRINT(cudaEventDestroy(m_training_splitk_events[i]));
		CUDA_CHECK_PRINT(cudaStreamDestroy(m_training_splitk_streams[i]));
	}
}

template <typename T, Activation input_activation>
void CutlassResNet<T, input_activation>::inference(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<float>& output) {
	GPUMatrixDynamic<T> inference_output_tmp{m_padded_output_width, output.n(), stream, output.layout()};
	inference_mixed_precision(stream, input, inference_output_tmp);

	const uint32_t n_elements = (uint32_t)output.n_elements();
	if (output.layout() == RM) {
		// If the layout is row major, trimming away excess dimensions amounts to simply discarding the tail of the buffer.
		cast_from<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, inference_output_tmp.data(), output.data());
	} else {
		trim_and_cast<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_padded_output_width, m_output_width, inference_output_tmp.data(), output.data());
	}
}

template <typename T, Activation input_activation>
void CutlassResNet<T, input_activation>::inference_mixed_precision(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output, bool use_inference_matrices) {
	// Various error checks
	if (input.m() != m_input_width) {
		throw std::runtime_error(std::string("Input has incorrect width: ") + std::to_string(input.m()) + "!=" + std::to_string(m_input_width));
	}

	if (output.m() != m_padded_output_width) {
		throw std::runtime_error(std::string("Output has incorrect width: ") + std::to_string(output.m()) + "!=" + std::to_string(m_output_width));
	}

	if (input.n() != output.n()) {
		throw std::runtime_error(std::string("Input and output don't have matching batch size: ") + std::to_string(input.n()) + "!=" + std::to_string(output.n()));
	}

	uint32_t batch_size = input.n();

	GPUMatrix<T> inference_residual_tmp[2] = {
		GPUMatrix<T>{m_network_width, batch_size, stream},
		GPUMatrix<T>{m_network_width, batch_size, stream},
	};
	GPUMatrix<T> inference_linear_tmp{m_network_width, batch_size, stream};

	// Run the actual network
	{
		// Input
		fc_multiply<FullLayer>(stream, input_weight_matrix(use_inference_matrices), input, inference_linear_tmp, inference_linear_tmp, input_activation);

		// Res blocks
		for (uint32_t i = 0; i < m_n_blocks; ++i) {
			fc_multiply<FullLayerPreReLU>(stream, weight_matrix_at(use_inference_matrices, i, 0), inference_linear_tmp, inference_residual_tmp[0]);

			for (uint32_t matrix_idx = 1; matrix_idx < m_n_matrices_per_block - 1; ++matrix_idx) {
				fc_multiply<FullLayerPreReLU>(stream, weight_matrix_at(use_inference_matrices, i, matrix_idx), inference_residual_tmp[(matrix_idx+1) % 2], inference_residual_tmp[matrix_idx % 2]);
			}

			// In case there's just 1 matrix per block, the remaining addition must be done manually
			if (m_n_matrices_per_block == 1) {
				const uint32_t n_elements = (uint32_t)inference_residual_tmp[0].n_elements();
				add<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, inference_residual_tmp[0].data(), inference_linear_tmp.data());
			} else {
				uint32_t matrix_idx = m_n_matrices_per_block - 1;
				fc_multiply<FullLayerPreReLU>(
					stream,
					weight_matrix_at(use_inference_matrices, i, matrix_idx),
					inference_residual_tmp[(matrix_idx+1) % 2],
					inference_linear_tmp,
					inference_linear_tmp,
					Activation::None,
					false, // no transfer
					true // sums up the residual and linear parts
				);
			}
		}

		// Output
		fc_multiply<LastLayer>(stream, output_weight_matrix(use_inference_matrices), inference_linear_tmp, output, m_output_activation);
	}
}

template <typename T, Activation input_activation>
std::unique_ptr<Context> CutlassResNet<T, input_activation>::forward(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* output, bool use_inference_matrices, bool prepare_input_gradients) {
	// Various error checks
	if (input.m() != m_input_width) {
		throw std::runtime_error(std::string("Input has incorrect width: ") + std::to_string(input.m()) + "!=" + std::to_string(m_input_width));
	}

	if (output && output->m() != m_padded_output_width) {
		throw std::runtime_error(std::string("Output has incorrect width (must be padded): ") + std::to_string(output->m()) + "!=" + std::to_string(m_padded_output_width));
	}

	if (output && input.n() != output->n()) {
		throw std::runtime_error(std::string("Input and output don't have matching batch size: ") + std::to_string(input.n()) + "!=" + std::to_string(output->n()));
	}

	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();
	auto forward = allocate_forward_buffers(stream, batch_size);

	const uint32_t n_elements = (uint32_t)forward->hidden.front().n_elements();

	// Run the actual network
	{
		auto& input_target = input_activation_value == Activation::None ? forward->hidden.front() : forward->input;
		fc_multiply<FullLayer>(stream, input_weight_matrix(use_inference_matrices), input, input_target);
		activation_gpu(stream, input_activation_value, input_target, forward->hidden.front());

		// Res blocks
		for (uint32_t i = 0; i < m_n_blocks; ++i) {
			uint32_t idx = i * m_n_matrices_per_block + 1;

			if (m_n_matrices_per_block == 1) {
				fc_multiply<FullLayerPreReLU>(stream, weight_matrix_at(use_inference_matrices, i, 0), forward->hidden.at(idx-1), forward->hidden.at(idx));
				add<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, forward->hidden.at(idx-1).data(), forward->hidden.at(idx).data());
			} else {
				fc_multiply<FullLayerPreReLU>(stream, weight_matrix_at(use_inference_matrices, i, 0), forward->hidden.at(idx-1), forward->hidden.at(idx), Activation::ReLU);

				for (uint32_t matrix_idx = 1; matrix_idx < m_n_matrices_per_block - 1; ++matrix_idx) {
					uint32_t fwd_idx = idx + matrix_idx;
					fc_multiply<FullLayer>(stream, weight_matrix_at(use_inference_matrices, i, matrix_idx), forward->hidden.at(fwd_idx-1), forward->hidden.at(fwd_idx), Activation::ReLU);
				}

				uint32_t matrix_idx = m_n_matrices_per_block - 1;
				uint32_t fwd_idx = idx + matrix_idx;
				fc_multiply<FullLayer>(
					stream,
					weight_matrix_at(use_inference_matrices, i, matrix_idx),
					forward->hidden.at(fwd_idx-1),
					forward->hidden.at(idx-1),
					forward->hidden.at(fwd_idx),
					Activation::None,
					false, // no transfer
					true // sums up the residual and linear parts
				);
			}

			// Retroactively apply ReLU to input. It's needed for backprop later.
			// We schedule it to the appropriate splitk stream, because only the later splitk operation depends on
			// the ReLU'd values to be present
			activation_gpu(m_training_splitk_streams.at(idx-1), Activation::ReLU, forward->hidden.at(idx-1), forward->hidden.at(idx-1));
		}

		if (output) {
			fc_multiply<LastLayer>(stream, output_weight_matrix(use_inference_matrices), forward->hidden.back(), *output, m_output_activation);
		}
	}

	return forward;
}

template <typename T, Activation input_activation>
void CutlassResNet<T, input_activation>::backward(
	cudaStream_t stream,
	const Context& ctx,
	const GPUMatrixDynamic<T>& input,
	const GPUMatrixDynamic<T>& output,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrixDynamic<T>* dL_dinput,
	bool use_inference_matrices,
	bool compute_param_gradients
) {
	if (dL_doutput.m() != m_padded_output_width) {
		throw std::runtime_error(std::string("Output gradients have incorrect width (must be padded): ") + std::to_string(dL_doutput.m()) + "!=" + std::to_string(m_padded_output_width));
	}

	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = dL_doutput.n();

	std::vector<GPUMatrix<T>> backward_tmp(num_forward_activations());
	for (uint32_t i = 0; i < num_forward_activations(); ++i) {
		backward_tmp[i] = GPUMatrix<T>{m_network_width, batch_size, stream};
	}

	// Compute transfer of output activation in-place... it's treated specially for performance reasons
	GPUMatrixDynamic<T> backward_output_tmp;
	if (m_output_activation != Activation::None) {
		backward_output_tmp = {m_padded_output_width, batch_size, stream, dL_doutput.layout()};
		activation_backward_output_gpu(stream, dL_doutput.n_elements(), m_output_activation, output.data(), dL_doutput.data(), backward_output_tmp.data());
	}

	// Backprop
	// - weight_gradient.T = input_activation * output_gradient.T
	// - input_gradient = weights.T * output_gradient
	// - RELU: pre_activation_gradinet = post_activation_gradient if val > 0 else 0

	{
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		const uint32_t n_elements = (uint32_t)backward_tmp.front().n_elements();

		int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);

		const GPUMatrixDynamic<T>& tmp_dL_doutput = m_output_activation == Activation::None ? dL_doutput : backward_output_tmp;

		if (compute_param_gradients) {
			// Output layer
			cudaEventRecord(m_training_splitk_events.back(), stream);
			cudaStreamWaitEvent(m_training_splitk_streams.back(), m_training_splitk_events.back(), 0);
			fc_multiply_split_k<LastLayerK>(m_training_splitk_streams.back(), tmp_dL_doutput, forward.hidden.back().transposed(), output_gradient_matrix(), split_k_factor);
			cudaEventRecord(m_training_splitk_events.back(), m_training_splitk_streams.back());
		}

		fc_multiply<FullLayer>(stream, output_weight_matrix(use_inference_matrices).transposed(), tmp_dL_doutput, backward_tmp.back());

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
					fc_multiply_split_k<FullLayerK>(m_training_splitk_streams.at(fwd_idx), backward_tmp.at(fwd_idx), forward.hidden.at(fwd_idx-1).transposed(), gradient_matrix_at(block_idx, matrix_idx), split_k_factor);
					cudaEventRecord(m_training_splitk_events.at(fwd_idx), m_training_splitk_streams.at(fwd_idx));
				}

				fc_multiply<FullLayer>(stream, weight_matrix_at(use_inference_matrices, block_idx, matrix_idx).transposed(), backward_tmp.at(fwd_idx), forward.hidden.at(fwd_idx-1), backward_tmp.at(fwd_idx-1), Activation::ReLU, true);
			}

			add<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, backward_tmp.at(idx+m_n_matrices_per_block-1).data(), backward_tmp.at(idx-1).data());
		}

		activation_backward_gpu(stream, input_activation_value, forward.input, backward_tmp.front());

		if (compute_param_gradients) {
			cudaEventRecord(m_training_splitk_events.front(), stream);
			cudaStreamWaitEvent(m_training_splitk_streams.front(), m_training_splitk_events.front(), 0);
			fc_multiply_split_k<FullLayerK>(m_training_splitk_streams.front(), backward_tmp.front(), input.transposed(), input_gradient_matrix(), split_k_factor);
			cudaEventRecord(m_training_splitk_events.front(), m_training_splitk_streams.front());
		}

		// If requested, compute sensitivity of loss w.r.t. inputs
		if (dL_dinput) {
			// TODO: optimization opportunity to only compute sensitivity w.r.t selected SUBSET of inputs. Useful for NFs, where conditional dims stay the same.
			fc_multiply<FullLayer>(stream, input_weight_matrix(use_inference_matrices).transposed(), backward_tmp.front(), *dL_dinput);
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

	forward_clear();
}

template <typename T, Activation input_activation>
std::unique_ptr<CutlassResNet<T, input_activation>::ForwardContext> CutlassResNet<T, input_activation>::allocate_forward_buffers(cudaStream_t stream, uint32_t batch_size) {
	auto forward = std::make_unique<ForwardContext>();

	forward->input = GPUMatrix<T>{m_network_width, batch_size, stream};
	forward->hidden.resize(num_forward_activations());
	for (uint32_t i = 0; i < num_forward_activations(); ++i) {
		forward->hidden[i] = GPUMatrix<T>{m_network_width, batch_size, stream};
	}

	return forward;
}

template <typename T, Activation input_activation>
void CutlassResNet<T, input_activation>::set_params(T* params, T* inference_params, T* backward_params, T* gradients) {
	size_t current_pos = 0;
	for (size_t i = 0; i < m_weight_matrices.size(); ++i) {
		m_weight_matrices[i].set_data(params + current_pos);
		m_weight_matrices_inference[i].set_data(inference_params + current_pos);
		m_gradient_matrices[i].set_data(gradients + current_pos);
		current_pos += m_weight_matrices[i].n_elements();
	}
}

template <typename T, Activation input_activation>
void CutlassResNet<T, input_activation>::initialize_params(pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale) {
	set_params(params, inference_params, backward_params, gradients);

	size_t current_pos = 0;
	for (size_t i = 0; i < m_weight_matrices_full_precision.size(); ++i) {
		m_weight_matrices_full_precision[i].set_data(params_full_precision + current_pos);
		current_pos += m_weight_matrices_full_precision[i].n_elements();

		if (i == 0 && input_activation_value == Activation::Sine) {
			m_weight_matrices_full_precision[i].initialize_siren_uniform_first(rnd, scale);
		} else {
			m_weight_matrices_full_precision[i].initialize_xavier_uniform(rnd, scale);
		}
	}
}

// Explicitly instantiate resnet classes.
template class CutlassResNet<network_precision_t, Activation::None>;

TCNN_NAMESPACE_END
