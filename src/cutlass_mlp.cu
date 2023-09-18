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

/** @file   cutlass_mlp.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  CUTLASS implementation of an optimized multi-layer perceptron. Supports online training
 *          and simultaneous inference.
 */

#include <tiny-cuda-nn/networks/cutlass_mlp.h>

#include <tiny-cuda-nn/cutlass_matmul.h>
#include <tiny-cuda-nn/multi_stream.h>

namespace tcnn {

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
m_output_activation{output_activation},
m_can_fuse_activation{activation != Activation::Sine}
{
	m_padded_output_width = next_multiple(m_output_width, REQUIRED_ALIGNMENT());

	if (m_n_hidden_layers > 0) {
		m_n_hidden_matmuls = m_n_hidden_layers-1;
	} else {
		m_n_hidden_matmuls = 0;
	}

	// Create matrices related to weights
	if (m_n_hidden_layers == 0) {
		m_weight_matrices.emplace_back(nullptr, m_padded_output_width, m_input_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_padded_output_width, m_input_width);
		m_gradient_matrices.emplace_back(nullptr, m_padded_output_width, m_input_width);
	} else {
		m_weight_matrices.emplace_back(nullptr, m_network_width, m_input_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_input_width);
		m_gradient_matrices.emplace_back(nullptr, m_network_width, m_input_width);

		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
			m_weight_matrices.emplace_back(nullptr, m_network_width, m_network_width);
			m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_network_width);
			m_gradient_matrices.emplace_back(nullptr, m_network_width, m_network_width);
		}

		m_weight_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_padded_output_width, m_network_width);
		m_gradient_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);
	}

	// Determine total number of memory entries and set it
	m_total_n_params = 0;
	for (const auto& m : m_weight_matrices) {
		m_total_n_params += m.n_elements();
	}
}

template <typename CutlassLayer, typename T>
bool compute_layer(
	cudaStream_t stream,
	bool is_inference,
	Activation activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& input,
	GPUMatrixDynamic<T>& output,
	GPUMatrixDynamic<T>& activation_output
) {
	bool can_fuse_activation = true;
	if (!is_inference) {
		// Never disallow fusing if the caller passes the same output and activation_output buffers... in that case,
		// invertibility of the activation function may be ignored.
		can_fuse_activation &= activation != Activation::Sine || &output == &activation_output;
	}

	if (can_fuse_activation) {
		fc_multiply<CutlassLayer>(stream, weights, input, output, activation);
	} else {
		fc_multiply<CutlassLayer>(stream, weights, input, output);
		activation_gpu(stream, activation, output, activation_output);
	}

	return can_fuse_activation;
}

template <typename CutlassLayer, typename T>
bool compute_fc_layer(
	cudaStream_t stream,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& input,
	GPUMatrixDynamic<T>& p
) {
	// compute for forward values before activation
	fc_multiply<CutlassLayer>(stream, weights, input, p);

	return true;
}

template <typename CutlassLayer, typename T>
bool compute_inference_layer(
	cudaStream_t stream,
	Activation activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& input,
	GPUMatrixDynamic<T>& output
) {
	return compute_layer<CutlassLayer>(stream, true, activation, weights, input, output, output);
}

template <typename T>
void CutlassMLP<T>::inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output, bool use_inference_params) {
	// If there are no hidden layers, the network is just a simple matmul.
	if (m_n_hidden_layers == 0) {
		compute_inference_layer<LastLayer>(stream, m_output_activation, input_weight_matrix(use_inference_params), input, output);
		return;
	}

	uint32_t batch_size = input.n();
	GPUMatrix<T> inference_tmp[2] = {
		GPUMatrix<T>{m_network_width, batch_size, stream},
		GPUMatrix<T>{m_network_width, batch_size, stream},
	};

	// Run the actual network
	{
		uint32_t tmp_idx = 0;

		// Input layer
		compute_inference_layer<FullLayer>(stream, m_activation, input_weight_matrix(use_inference_params), input, inference_tmp[tmp_idx++ % 2]);

		// Hidden layers
		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
			compute_inference_layer<FullLayer>(stream, m_activation, weight_matrix_at(use_inference_params, i), inference_tmp[(tmp_idx + 1) % 2], inference_tmp[tmp_idx % 2]);
			++tmp_idx;
		}

		// Output
		compute_inference_layer<LastLayer>(stream, m_output_activation, output_weight_matrix(use_inference_params), inference_tmp[(tmp_idx + 1) % 2], output);
	}
}

template <typename T>
std::unique_ptr<Context> CutlassMLP<T>::forward_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* output, bool use_inference_params, bool prepare_input_gradients) {
	// If there are no hidden layers, the network is just a simple matmul. No tmp buffers required
	if (m_n_hidden_layers == 0) {
		if (output) {
			compute_layer<LastLayer>(stream, false, m_output_activation, input_weight_matrix(use_inference_params), input, *output, *output);
		}
		return std::make_unique<ForwardContext>(); // Nothing to save -- empty context
	}

	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();
	auto forward = allocate_forward_buffers(stream, batch_size);

	// Run the actual network
	uint32_t tmp_idx = 0;

	bool fused = compute_layer<FullLayer>(
		stream,
		false,
		m_activation,
		input_weight_matrix(use_inference_params),
		input,
		forward->hidden.at(tmp_idx),
		m_can_fuse_activation ? forward->hidden.at(tmp_idx) : forward->hidden.at(tmp_idx+1)
	);
	tmp_idx += fused ? 1 : 2;

	// layers
	for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
		fused = compute_layer<FullLayer>(
			stream,
			false,
			m_activation,
			weight_matrix_at(use_inference_params, i),
			forward->hidden.at(tmp_idx-1),
			forward->hidden.at(tmp_idx),
			m_can_fuse_activation ? forward->hidden.at(tmp_idx) : forward->hidden.at(tmp_idx+1)
		);
		tmp_idx += fused ? 1 : 2;
	}

	if (output) {
		compute_layer<LastLayer>(stream, false, m_output_activation, output_weight_matrix(use_inference_params), forward->hidden.at(tmp_idx-1), *output, *output);
	}

	return forward;
}

template <typename T>
void CutlassMLP<T>::backward_impl(
	cudaStream_t stream,
	const Context& ctx,
	const GPUMatrixDynamic<T>& input,
	const GPUMatrixDynamic<T>& output,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrixDynamic<T>* dL_dinput,
	bool use_inference_params,
	GradientMode param_gradients_mode
) {
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
	// - weight_gradient.T = activation * output_gradient.T
	// - input_gradient = weights.T * output_gradient
	// - RELU: pre_activation_gradinet = post_activation_gradient if val > 0 else 0

	const float param_gradient_beta = param_gradients_mode == GradientMode::Accumulate ? 1.0f : 0.0f;

	std::vector<SyncedMultiStream> multi_streams;

	const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

	int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);

	const GPUMatrixDynamic<T>& tmp_dL_doutput = m_output_activation == Activation::None ? dL_doutput : backward_output_tmp;

	// If there are no hidden layers, the network is just a simple matmul
	if (m_n_hidden_layers == 0) {
		if (param_gradients_mode != GradientMode::Ignore) {
			multi_streams.emplace_back(stream, 2);
			fc_multiply_split_k<LastLayerK>(multi_streams.back().get(1), tmp_dL_doutput, input.transposed(), input_gradient_matrix(), split_k_factor, param_gradient_beta);
		}

		if (dL_dinput) {
			fc_multiply<FullLayer>(stream, input_weight_matrix(use_inference_params).transposed(), tmp_dL_doutput, *dL_dinput);
		}

		return;
	}

	uint32_t tmp_idx = (m_can_fuse_activation ? (m_n_hidden_matmuls+1) : ((m_n_hidden_matmuls+1) * 2)) - 1;
	uint32_t backward_tmp_idx = 0;

	// Output layer
	if (param_gradients_mode != GradientMode::Ignore) {
		multi_streams.emplace_back(stream, 2);
		fc_multiply_split_k<LastLayerK>(multi_streams.back().get(1), tmp_dL_doutput, forward.hidden.at(tmp_idx).transposed(), output_gradient_matrix(), split_k_factor, param_gradient_beta);

	}

	if (!m_can_fuse_activation) {
		fc_multiply<FullLayer>(stream, output_weight_matrix(use_inference_params).transposed(), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx));
		activation_backward_gpu(stream, m_activation, forward.hidden.at(tmp_idx-1), backward_tmp.at(backward_tmp_idx));
	} else {
		fc_multiply<FullLayer>(stream, output_weight_matrix(use_inference_params).transposed(), tmp_dL_doutput, forward.hidden.at(tmp_idx), backward_tmp.at(backward_tmp_idx), m_activation, true);
	}

	tmp_idx -= m_can_fuse_activation ? 1 : 2;
	++backward_tmp_idx;

	// layers
	for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
		uint32_t matrix_idx = m_n_hidden_matmuls - i - 1;

		if (param_gradients_mode != GradientMode::Ignore) {
			multi_streams.emplace_back(stream, 2);
			fc_multiply_split_k<FullLayerK>(multi_streams.back().get(1), backward_tmp.at(backward_tmp_idx-1), forward.hidden.at(tmp_idx).transposed(), gradient_matrix_at(matrix_idx), split_k_factor, param_gradient_beta);
		}

		if (!m_can_fuse_activation) {
			fc_multiply<FullLayer>(stream, weight_matrix_at(use_inference_params, matrix_idx).transposed(), backward_tmp.at(backward_tmp_idx-1), backward_tmp.at(backward_tmp_idx));
			activation_backward_gpu(stream, m_activation, forward.hidden.at(tmp_idx-1), backward_tmp.at(backward_tmp_idx));
		} else {
			fc_multiply<FullLayer>(stream, weight_matrix_at(use_inference_params, matrix_idx).transposed(), backward_tmp.at(backward_tmp_idx-1), forward.hidden.at(tmp_idx), backward_tmp.at(backward_tmp_idx), m_activation, true);
		}

		tmp_idx -= m_can_fuse_activation ? 1 : 2;
		++backward_tmp_idx;
	}

	if (param_gradients_mode != GradientMode::Ignore) {
		multi_streams.emplace_back(stream, 2);
		fc_multiply_split_k<FullLayerK>(multi_streams.back().get(1), backward_tmp.at(backward_tmp_idx-1), input.transposed(), input_gradient_matrix(), split_k_factor, param_gradient_beta);
	}

	// If requested, compute sensitivity of loss w.r.t. inputs
	if (dL_dinput) {
		// optimization opportunity to only compute sensitivity w.r.t selected SUBSET of inputs. Useful for NFs, where conditional dims stay the same.
		fc_multiply<FullLayer>(stream, input_weight_matrix(use_inference_params).transposed(), backward_tmp.at(backward_tmp_idx-1), *dL_dinput);
	}
}

// ======================= backward_backward_input_impl =======================
// compute 2nd order dact
template <typename T>
__global__ void compute_activation_backward_backward(uint32_t n_elements, Activation activation, T* p, T* res) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	switch (activation) {
		case Activation::Softplus:
			float K_ACT = 10.0;
			float tmp = (float)p[i] * K_ACT;//
			if (tmp > 10.0) {
				tmp = 10.0;
			} else if (tmp < -15.0) {
				tmp = -15.0;
			}

			float exp_tmp = expf(tmp);
			float pow_tmp = (exp_tmp + 1.0) * (exp_tmp + 1.0);
			float ddoutputdp_dp = exp_tmp / pow_tmp * K_ACT;
			res[i] = (T)ddoutputdp_dp;
			return;

		case Activation::ReLU:
			res[i] = 0.0;
			return;

		default:
			// ERROR: this activation currently is not supported
			res[i] = 0.0;
			return;
	}

	return;
}

// activation Softplus 2nd order derivative in 1D: doutputdp_2
template <typename T>
__global__ void compute_ddoutputdp_dp_dLdoutput(uint32_t n_elements, T* dL_doutput, T* ddoutputdp_dp, T* doutputdp_2) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	doutputdp_2[i] = ddoutputdp_dp[i] * dL_doutput[i];

	return;
}

// fuse the process of computing ddoutputdp_dp_2
template <typename T>
__global__ void fuse_ddoutputdp_dp(uint32_t n_elements, T* dL_doutput, T* w_x_dL2_ddL1dinput, T* ddoutputdp_dp, T* result) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;
	
	T dL1doutput_x_w_x_dL2_ddL1dinput = dL_doutput[i] * w_x_dL2_ddL1dinput[i];
	result[i] = ddoutputdp_dp[i] * dL1doutput_x_w_x_dL2_ddL1dinput;

	return;
}

// element-wise add back to dL_dinput
template <typename T>
__global__ void element_wise_add(uint32_t n_elements, T* tmp, T* res) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	res[i] += tmp[i];
}

// element-wise copy from an CM tmp to RM dL_dinput
// row: the tmp rows
// col: the tmp cols
template <typename T>
__global__ void element_wise_copy_CM_RM(uint32_t n_elements, uint32_t row, uint32_t col, T* tmp, T* res) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	uint32_t idx_i = i % row;
	uint32_t idx_j = i / row;
	uint32_t i_RM = idx_i * col + idx_j;
	res[i_RM] = tmp[i];
}

// compute d_doutputdp2_dinput
template <typename T>
__global__ void multiply_w_RM(uint32_t n_elements, uint32_t w_row, uint32_t w_col, T* weights, T* doutputdp_2, T* result) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	uint32_t idx_doutputdp = i % w_row;
	T tmp = doutputdp_2[idx_doutputdp];
	uint32_t i_RM = (i % w_row) * w_col + i / w_row;
	result[i] = weights[i_RM] * tmp;
}

// compute dL2_ddL1doutput: activation Softplus/ReLU derivative, doutputdp and then multiply dL2_ddL1doutput_tmp
template <typename CutlassLayer, typename T>
bool compute_dL2_ddL1doutput(
	cudaStream_t stream,
	bool is_inference,
	Activation activation,
	GPUMatrix<T, RM>& weights,
	const GPUMatrix<T, CM>& p,
	const GPUMatrixDynamic<T>& dL_ddLdinput,
	GPUMatrixDynamic<T>& dL2_ddL1doutput,
	GradientMode param_gradients_mode
) {
	// dL2_ddL1doutput = weight x dL_ddLdinput where x is matrix multiply
	if (weights.layout() == CM) {
		fc_multiply<CutlassLayer>(stream, weights.cm(), dL_ddLdinput, dL2_ddL1doutput);
	} else {
		fc_multiply<CutlassLayer>(stream, weights.rm(), dL_ddLdinput, dL2_ddL1doutput);
	}

	// if activation is None, dL2_ddL1doutput = weight x dL2_ddL1dinput
	if (activation == Activation::None) {
		return true;
	}

	// dL2_ddL1doutput = dL2_ddL1doutput · doutputdp, where p is linear output, · is dot product
	activation_backward_gpu(stream, activation, p, dL2_ddL1doutput);
	
	return true;
}

template <typename CutlassLayer, typename T>
bool compute_dL2dw(
	cudaStream_t stream,
	bool is_inference,
	Activation activation,
	GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& input,
	const GPUMatrixDynamic<T>& p, // const GPUMatrix<T, CM>& p
	const GPUMatrixDynamic<T>& dL_dp,
	const GPUMatrixDynamic<T>& dL_doutput,
	const GPUMatrixDynamic<T>& dL_ddLdinput, // dL2_ddL1dinput
	const GPUMatrixDynamic<T>& ddoutputdp_dp, // ddoutputdp_dp
	GPUMatrixDynamic<T>& dL2_ddL1doutput, // pointer better
	GPUMatrix<T, RM>& weight_gradient, // pointer better
	GradientMode param_gradients_mode
) {
	// dL2dw_1 = ddL1dinput_dw x dL2_ddL1dinput	
	uint32_t batch_size = dL_ddLdinput.n();
	int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);
	const float param_gradient_beta = param_gradients_mode == GradientMode::Accumulate ? 1.0f : 0.0f;

	//cudaStream_t stream_dL2dw_1;
	if (param_gradients_mode != GradientMode::Ignore) {
		fc_multiply_split_k<FullLayerK>(stream, dL_dp, dL_ddLdinput.transposed(), weight_gradient, split_k_factor, param_gradient_beta);
	}

	// when activation is None, don't have to compute dL2dw_2
	if (activation == Activation::None) {
		return true;
	}

	//dL2dw_2 = torch.matmul(torch.transpose(ddoutputdp_dp_2, 0, 1), input)
	GPUMatrixDynamic<T> w_x_dL2_ddL1dinput = {weights.m(), dL_ddLdinput.n(), stream};

	if (weights.layout() == CM) {
		fc_multiply<CutlassLayer>(stream, weights.cm(), dL_ddLdinput, w_x_dL2_ddL1dinput);
	} else {
		fc_multiply<CutlassLayer>(stream, weights.rm(), dL_ddLdinput, w_x_dL2_ddL1dinput);
	}

	// fuse kernels to compute ddoutputdp_dp_2
	GPUMatrixDynamic<T> ddoutputdp_dp_2_fuse = {p.rows(), p.cols(), stream};
	linear_kernel(fuse_ddoutputdp_dp<T>, 0, stream, ddoutputdp_dp_2_fuse.n_elements(), dL_doutput.data(), w_x_dL2_ddL1dinput.data(), ddoutputdp_dp.data(), ddoutputdp_dp_2_fuse.data());

	if (param_gradients_mode != GradientMode::Ignore) {
		fc_multiply_split_k<FullLayerK>(stream, ddoutputdp_dp_2_fuse, input.transposed(), weight_gradient, split_k_factor, 1.0);
	}

	return true;
}

template <typename CutlassLayer, typename T>
bool compute_dL2dinput(
	cudaStream_t stream,
	bool is_inference,
	Activation activation,
	GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& p,
	const GPUMatrixDynamic<T>& dL_doutput,
	const GPUMatrixDynamic<T>& dL_ddLdinput, // dL2_ddL1dinput
	const GPUMatrixDynamic<T>& ddoutputdp_dp, // ddoutputdp_dp
	GPUMatrixDynamic<T>& dL_dinput,
	GradientMode param_gradients_mode
) {
	// no dL2dinput when activation is None
	if (activation == Activation::None) {
		return true;
	}

	// compute weights x dL2_ddL1dinput in advance
	GPUMatrixDynamic<T> w_x_dL2ddL1dinput = {weights.rows(), dL_ddLdinput.cols(), stream};

	if (weights.layout() == CM) {
		fc_multiply<CutlassLayer>(stream, weights.cm(), dL_ddLdinput, w_x_dL2ddL1dinput);
	} else {
		fc_multiply<CutlassLayer>(stream, weights.rm(), dL_ddLdinput, w_x_dL2ddL1dinput);
	}
	
	// doutputdp_2 = (p.rows(), batch_size)
	GPUMatrixDynamic<T> doutputdp_2 = {p.rows(), p.cols(), stream};

	// compute doutputdp_2 in 1 dimension (1, 64) x batch_size, and multiply dL_doutput
	linear_kernel(compute_ddoutputdp_dp_dLdoutput<T>, 0, stream, doutputdp_2.n_elements(), dL_doutput.data(), ddoutputdp_dp.data(), doutputdp_2.data());

	GPUMatrixDynamic<T> ddoutputdp_dinput = {weights.rows(), weights.cols(), stream};	
	// ddoutputdp_dinput[i, j] = tmp * w[i, j] where tmp = doutputdp_2_sum[i, 0]
	linear_kernel(multiply_w_RM<T>, 0, stream, weights.n_elements(), weights.rows(), weights.cols(), weights.data(), doutputdp_2.data(), ddoutputdp_dinput.data());

	// dL_dinput = ddoutputdp_dinput_xw x dL2d_dL1dinput
	if (ddoutputdp_dinput.transposed().layout() == CM) {
		fc_multiply<CutlassLayer>(stream, ddoutputdp_dinput.transposed().cm(), w_x_dL2ddL1dinput, dL_dinput);
	} else {
		fc_multiply<CutlassLayer>(stream, ddoutputdp_dinput.transposed().rm(), w_x_dL2ddL1dinput, dL_dinput);
	}

	return true;
}

// prepare variables needed for backward temporary
template <typename T>
bool CutlassMLP<T>::prepare_backward_variables(
	cudaStream_t stream,
	const std::vector<GPUMatrix<T>>& output, // const GPUMatrix<T, CM>& p
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrixDynamic<T>& backward_output_tmp,
	std::vector<GPUMatrix<T>>& dL1dp,
	std::vector<GPUMatrix<T>>& dL1doutput,
	bool use_inference_params
) {
	// compute dL1dp and dL1doutput
	uint32_t batch_size = dL_doutput.n();
	uint32_t bwd_tmp_idx = m_n_hidden_matmuls + 1 - 1;
	uint32_t bwd_dL1dp_idx = 0;
	
	// compute dL1dp and dL1dinput of output layer
	const GPUMatrixDynamic<T>& tmp_dL_doutput = m_output_activation == Activation::None ? dL_doutput : backward_output_tmp;

	// directly compute dL1dp_i-1
	fc_multiply<FullLayer>(stream, output_weight_matrix(use_inference_params).transposed(), tmp_dL_doutput, output.at(bwd_tmp_idx), dL1dp.at(bwd_dL1dp_idx), m_activation, true);
	// extra computing once to save dL1doutput of each layer
	fc_multiply<FullLayer>(stream, output_weight_matrix(use_inference_params).transposed(), tmp_dL_doutput, dL1doutput.at(bwd_dL1dp_idx));

	bwd_tmp_idx -= m_can_fuse_activation ? 1 : 2;
	++bwd_dL1dp_idx;

	// layers
	for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
		uint32_t matrix_idx = m_n_hidden_matmuls - i - 1;

		fc_multiply<FullLayer>(stream, weight_matrix_at(use_inference_params, matrix_idx).transposed(), dL1dp.at(bwd_dL1dp_idx-1), output.at(bwd_tmp_idx), dL1dp.at(bwd_dL1dp_idx), m_activation, true);
		// extra computing once to save dL1doutput of each layer
		fc_multiply<FullLayer>(stream, weight_matrix_at(use_inference_params, matrix_idx).transposed(), dL1dp.at(bwd_dL1dp_idx-1), dL1doutput.at(bwd_dL1dp_idx));

		bwd_tmp_idx -= m_can_fuse_activation ? 1 : 2;
		++bwd_dL1dp_idx;
	}

	return true;
}

template <typename T>
void CutlassMLP<T>::backward_backward_input_impl(
	cudaStream_t stream,
	const Context& ctx,
	const GPUMatrixDynamic<T>& input,
	const GPUMatrixDynamic<T>& dL_ddLdinput,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrixDynamic<T>* dL_ddLdoutput,
	GPUMatrixDynamic<T>* dL_dinput,
	bool use_inference_params,
	GradientMode param_gradients_mode
) {
	uint32_t batch_size = dL_doutput.n();

	// dL2_ddL1dinput vector of each layer
	std::vector<GPUMatrix<T>> dL2_ddL1doutput(num_forward_activations());
	for (uint32_t i = 0; i < num_forward_activations(); ++i) {
		dL2_ddL1doutput[i] = GPUMatrix<T>{m_network_width, batch_size, stream};
	}
	
	// dL2dinput vector of each layer
	std::vector<GPUMatrixDynamic<T>> dL2dinput;
	if (m_n_hidden_layers == 0) {
		dL2dinput.resize(1);
		dL2dinput[0] = GPUMatrixDynamic<T>{m_input_width, batch_size, stream};
	} else {
		dL2dinput.resize(m_n_hidden_matmuls + 2);
		dL2dinput[0] = GPUMatrixDynamic<T>{m_input_width, batch_size, stream};
		for (uint32_t i = 0; i < m_n_hidden_matmuls+1; ++i) {
			dL2dinput[i+1] = GPUMatrixDynamic<T>{m_network_width, batch_size, stream};
		}
	}
	// NOTE: this code is for removing NaN initialization of dL2dinput[last_layer], must keep this code
	CUDA_CHECK_THROW(cudaMemsetAsync(dL2dinput[m_n_hidden_matmuls+1].data(), 0, dL2dinput[m_n_hidden_matmuls+1].n_elements() * sizeof(T), stream));

	// 2nd order derivative of activation for each layer
	std::vector<GPUMatrixDynamic<T>> ddoutputdp_dp;
	if (m_n_hidden_layers == 0) {
		ddoutputdp_dp.resize(1);
		ddoutputdp_dp[0] = GPUMatrixDynamic<T>{m_padded_output_width, batch_size, stream};
	} else {
		ddoutputdp_dp.resize(m_n_hidden_matmuls + 2);
		for (uint32_t i = 0; i < m_n_hidden_matmuls+1; ++i) {
			ddoutputdp_dp[i] = GPUMatrixDynamic<T>{m_network_width, batch_size, stream};
		}
		ddoutputdp_dp[m_n_hidden_matmuls+1] = GPUMatrixDynamic<T>{m_padded_output_width, batch_size, stream};
	}
	CUDA_CHECK_THROW(cudaMemsetAsync(ddoutputdp_dp[m_n_hidden_matmuls+1].data(), 0, ddoutputdp_dp[m_n_hidden_matmuls+1].n_elements() * sizeof(T), stream));

	// declare variables for fc_output, aka p, the result right after linear layer
	std::vector<GPUMatrix<T>> fc_output(num_forward_activations());
	for (uint32_t i = 0; i < num_forward_activations(); ++i) {
		fc_output[i] = GPUMatrix<T>{m_network_width, batch_size, stream}; // GPUMatrix<T>{m_network_width, batch_size, stream};
	}
	GPUMatrix<T> fc_last_output(m_padded_output_width, batch_size, stream); // p of output layer

	// declare variables for dL1dp and dL1doutput
	GPUMatrixDynamic<T> backward_output_tmp; // dL1dp of input layer
	std::vector<GPUMatrix<T>> dL1dp; // dL1dp: reverse order of all layers except for input layer
	std::vector<GPUMatrix<T>> dL1doutput; // dL1doutput of each layer: reverse order

	// initialization for dL1dp, dL1doutput
	dL1dp.resize(num_forward_activations());
	dL1doutput.resize(num_forward_activations());
	for (uint32_t i = 0; i < num_forward_activations(); ++i) {
		dL1dp[i] = GPUMatrix<T>{m_network_width, batch_size, stream};
		dL1doutput[i] = GPUMatrix<T>{m_network_width, batch_size, stream};
	}
	
	// prepare temporary variables for 2nd order derivative
	const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

	// multi-stream to compute
	{
		std::vector<SyncedMultiStream> multi_streams_pre;
		multi_streams_pre.emplace_back(stream, 2);

		// compute dL1dp and dL1doutput
		bool bwd_prep = prepare_backward_variables (
			multi_streams_pre.back().get(1),
			forward.hidden,
			dL_doutput,
			backward_output_tmp,
			dL1dp,
			dL1doutput,
			use_inference_params
		);

		// compute fc_output(p) values
		uint32_t forward_idx = 0;
		multi_streams_pre.emplace_back(stream, 2);
		bool is_returned = compute_fc_layer<FullLayer>(
			multi_streams_pre.back().get(1), //stream,
			input_weight_matrix(use_inference_params),
			input,
			fc_output.at(forward_idx) // p: linear output
		);

		// compute ddoutputdp_dp of input layer
		linear_kernel(compute_activation_backward_backward<T>, 0, 
			multi_streams_pre.back().get(1), 
			fc_output.at(forward_idx).n_elements(), 
			m_activation, 
			fc_output.at(forward_idx).data(),
			ddoutputdp_dp[forward_idx].data()
		);
		forward_idx++;

		for (uint32_t i = 0; i < m_n_hidden_matmuls; i++) {
			multi_streams_pre.emplace_back(stream, 2);
			is_returned = compute_fc_layer<FullLayer>(
				multi_streams_pre.back().get(1), //stream,
				weight_matrix_at(use_inference_params, i),
				forward.hidden.at(i), // input
				fc_output.at(forward_idx)
			);
			linear_kernel(compute_activation_backward_backward<T>, 0, 
				multi_streams_pre.back().get(1), 
				fc_output.at(forward_idx).n_elements(), 
				m_activation, 
				fc_output.at(forward_idx).data(), 
				ddoutputdp_dp[forward_idx].data()
			);
			forward_idx++;
		}
		// output layer
		multi_streams_pre.emplace_back(stream, 2);
		is_returned = compute_fc_layer<FullLayer>(
			multi_streams_pre.back().get(1), //stream,
			output_weight_matrix(use_inference_params),
			forward.hidden.at(m_n_hidden_matmuls),
			fc_last_output // p value
		);
		if (m_output_activation != Activation::None) {
			linear_kernel(compute_activation_backward_backward<T>, 0, 
				multi_streams_pre.back().get(1), 
				fc_last_output.n_elements(), 
				m_output_activation, 
				fc_last_output.data(), 
				ddoutputdp_dp[forward_idx].data()
			);
		}
		forward_idx++;
	}

	{ // 2nd order derivative computation: local definition for multi-stream
		// init for backward_backward computing
		std::vector<SyncedMultiStream> multi_streams;
		uint32_t tmp_idx = 0, bwd_idx = 0, bwd_bwd_idx = 0;

		// input layer
		// dL2dw for input layer
		if (param_gradients_mode != GradientMode::Ignore) {
			multi_streams.emplace_back(stream, 2);
			bool return_tmp_dL2dw = compute_dL2dw<FullLayer, T>(
				multi_streams.back().get(1),
				false,
				m_activation,
				input_weight_matrix(use_inference_params),
				input, // input
				fc_output.at(tmp_idx), // p
				dL1dp.at(m_n_hidden_matmuls), // dL1dp
				dL1doutput.at(m_n_hidden_matmuls), // dL1doutput
				dL_ddLdinput, // dL2_ddL1dinput
				ddoutputdp_dp.at(0), // ddoutputdp_dp
				dL2_ddL1doutput.at(bwd_bwd_idx), // dL2_ddL1doutput
				input_gradient_matrix(), // gradient matrix
				param_gradients_mode
			);
		}

		// 2nd order to dL2dinput of the 1st layer
		if (dL_dinput) {
			bool return_tmp = compute_dL2dinput<FullLayer, T>(
				stream,
				false,
				m_activation,
				input_weight_matrix(use_inference_params), // weights
				fc_output.at(0), // p
				dL1doutput.at(m_n_hidden_matmuls), // dL_doutput
				dL_ddLdinput,
				ddoutputdp_dp.at(0), // ddoutputdp_dp
				dL2dinput.at(0), // *dL_dinput,
				param_gradients_mode
			);
		}

		bool return_tmp_dL2_ddL1doutput = compute_dL2_ddL1doutput<FullLayer, T>(
			stream,
			false,
			m_activation,
			input_weight_matrix(use_inference_params),
			fc_output.at(tmp_idx),
			dL_ddLdinput,
			dL2_ddL1doutput.at(bwd_bwd_idx),
			param_gradients_mode
		);

		// TODO: hidden_layer == 0
		tmp_idx ++;
		bwd_bwd_idx++;
		// 2nd order derivative to dL2dinput and dL2dw
		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {		
			// 2nd order impact to dL2dw
			multi_streams.emplace_back(stream, 2);
			bool return_hidden_dL2dw = compute_dL2dw<FullLayer, T>(
				multi_streams.back().get(1), // stream
				false,
				m_activation,
				weight_matrix_at(use_inference_params, i),
				forward.hidden.at(tmp_idx-1), // input
				fc_output.at(tmp_idx), // p
				dL1dp.at(m_n_hidden_matmuls - bwd_idx - 1), // dL1dp
				dL1doutput.at(m_n_hidden_matmuls - bwd_idx - 1), // dL1doutput
				dL2_ddL1doutput.at(bwd_bwd_idx-1), // dL2d_dL1dinput
				ddoutputdp_dp.at(i+1), // ddoutputdp_dp
				dL2_ddL1doutput.at(bwd_bwd_idx), // dL2_ddL1doutput
				gradient_matrix_at(i), // gradient matrix
				param_gradients_mode
			);

			//multi_streams.emplace_back(stream, 2);
			bool return_hidden_dL2dinput = compute_dL2dinput<FullLayer, T>(
				stream, //multi_streams.back().get(3), // stream,
				false,
				m_activation,
				weight_matrix_at(use_inference_params, i), // weights
				fc_output.at(i+1), // p
				dL1doutput.at(m_n_hidden_matmuls - i - 1), // dL_doutput
				dL2_ddL1doutput.at(bwd_bwd_idx-1), // dL2_ddL1dinput of current layer
				ddoutputdp_dp.at(i+1), // ddoutputdp_dp
				dL2dinput.at(i+1), // dL2dinput of current layer
				param_gradients_mode
			);

			bool return_hidden_dL2_ddL1doutput = compute_dL2_ddL1doutput<FullLayer, T>(
				stream, //multi_streams.back().get(2), //stream,
				false, // is_inference
				m_activation,
				weight_matrix_at(use_inference_params, i),
				fc_output.at(tmp_idx), // p,
				dL2_ddL1doutput.at(bwd_bwd_idx-1), // dL2_ddL1dinput
				dL2_ddL1doutput.at(bwd_bwd_idx), // dL2_ddL1doutput
				param_gradients_mode
			);

			tmp_idx++; // tmp_idx += can_fused ? 1 : 2;
			bwd_bwd_idx++;
			bwd_idx++;
		}

		// Output layer weight
		multi_streams.emplace_back(stream, 2);
		bool return_output_dL2dw = compute_dL2dw<FullLayer, T>(
			multi_streams.back().get(1), //stream, 
			false,
			m_output_activation,
			output_weight_matrix(use_inference_params),
			forward.hidden.at(tmp_idx-1), // input
			fc_last_output, // p
			m_output_activation == Activation::None ? dL_doutput : backward_output_tmp, // dL1dp
			dL_doutput, // dL1doutput
			dL2_ddL1doutput.at(bwd_bwd_idx-1), // dL2d_dL1dinput
			ddoutputdp_dp.at(m_n_hidden_matmuls+1), // ddoutputdp_dp
			*dL_ddLdoutput, // dL2_ddL1doutput
			output_gradient_matrix(),
			param_gradients_mode
		);

		// Output layer dL2dinput
		bool return_output_dL2dinput = compute_dL2dinput<FullLayer, T>(
			stream,
			false, // is_inference
			m_output_activation,
			output_weight_matrix(use_inference_params), // weights
			fc_last_output, // p
			m_output_activation == Activation::None ? dL_doutput : backward_output_tmp, // dL_doutput
			dL2_ddL1doutput.at(bwd_bwd_idx-1), // dL2_ddL1dinput of current layer
			ddoutputdp_dp.at(m_n_hidden_matmuls+1), // ddoutputdp_dp
			dL2dinput.at(m_n_hidden_matmuls+1), // dL2dinput of current layer
			param_gradients_mode
		);		

		if (dL_ddLdoutput) { // if dL_ddLdoutput is not nullptr
			bool return_output_dL2_ddL1doutput = compute_dL2_ddL1doutput<FullLayer, T>(
				stream,
				false, // is_inference
				m_output_activation,
				output_weight_matrix(use_inference_params),
				fc_last_output, // p
				dL2_ddL1doutput.at(bwd_bwd_idx-1), // dL2d_dL1dinput
				*dL_ddLdoutput, // dL2_ddL1doutput
				param_gradients_mode
			);
		}
	}
	
	// 1st order backward of dL2dinput and dL2dw in reverse order
	std::vector<SyncedMultiStream> multi_streams_1st;
	multi_streams_1st.emplace_back(stream, 2);
	int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);
	const float param_gradient_beta = 1.0f; // param_gradients_mode == GradientMode::Accumulate

	for (uint32_t i = m_n_hidden_layers; i > 0; i--) {

		GPUMatrixDynamic<T> dL2dp = GPUMatrix<T>{fc_output.at(i-1).rows(), fc_output.at(i-1).cols(), stream};
		activation_backward_gpu(stream, dL2dp.n_elements(), m_activation, fc_output.at(i-1).data(), dL2dinput.at(i).data(), dL2dp.data());

		// compute 1st order dL2dw
		if (param_gradients_mode != GradientMode::Ignore) {
			if (i - 1) {
				fc_multiply_split_k<FullLayerK>(multi_streams_1st.back().get(1), dL2dp, forward.hidden.at(i-2).transposed(), gradient_matrix_at(i-2), split_k_factor, param_gradient_beta);
			} else { // i - 1 == 0: input layer
				fc_multiply_split_k<FullLayerK>(multi_streams_1st.back().get(1), dL2dp, input.transposed(), input_gradient_matrix(), split_k_factor, param_gradient_beta);
			}
		}
		
		if (i - 1) {
			GPUMatrixDynamic<T> dL_dinput_1st_order = GPUMatrix<T>{dL2dinput.at(i-1).rows(), dL2dinput.at(i-1).cols(), stream};
			// weight_matrix_at(use_inference_params, i-2).transposed().layout() == CM
			fc_multiply<FullLayer>(stream, weight_matrix_at(use_inference_params, i-2).transposed().cm(), dL2dp, dL_dinput_1st_order);
			linear_kernel(element_wise_add<T>, 0, stream, dL2dinput.at(i-1).n() * dL2dinput.at(i-1).m(), dL_dinput_1st_order.data(), dL2dinput.at(i-1).data());
		} else if (i - 1 == 0 && dL_dinput) { // input layer
			GPUMatrixDynamic<T> dL_dinput_1st_order = GPUMatrix<T>{dL2dinput.at(i-1).rows(), dL2dinput.at(i-1).cols(), stream};
			// input_weight_matrix(use_inference_params).transposed().layout() == CM
			fc_multiply<FullLayer>(stream, input_weight_matrix(use_inference_params).transposed().cm(), dL2dp, dL_dinput_1st_order);
			linear_kernel(element_wise_add<T>, 0, stream, dL2dinput.at(i-1).n() * dL2dinput.at(i-1).m(), dL_dinput_1st_order.data(), dL2dinput.at(i-1).data());
		}
	}

	if (dL_dinput) {
		// sync up back to the gradient and *dL_dinput
		linear_kernel(element_wise_copy_CM_RM<T>, 0, stream, dL_dinput->rows() * dL_dinput->cols(), dL2dinput.at(0).rows(), dL2dinput.at(0).cols(), dL2dinput.at(0).data(), dL_dinput->data());
	}
}

template <typename T>
std::unique_ptr<typename CutlassMLP<T>::ForwardContext> CutlassMLP<T>::allocate_forward_buffers(cudaStream_t stream, uint32_t batch_size) {
	auto forward = std::make_unique<ForwardContext>();

	forward->hidden.resize(num_forward_activations());
	for (uint32_t i = 0; i < num_forward_activations(); ++i) {
		forward->hidden[i] = GPUMatrix<T>{m_network_width, batch_size, stream};
	}

	return forward;
}

template <typename T>
void CutlassMLP<T>::set_params_impl(T* params, T* inference_params, T* gradients) {
	size_t current_pos = 0;
	for (size_t i = 0; i < m_weight_matrices.size(); ++i) {
		m_weight_matrices[i].set_data_unsafe(params + current_pos);
		m_weight_matrices_inference[i].set_data_unsafe(inference_params + current_pos);
		m_gradient_matrices[i].set_data_unsafe(gradients + current_pos);
		current_pos += m_weight_matrices[i].n_elements();
	}
}

template <typename T>
void CutlassMLP<T>::initialize_params(pcg32& rnd, float* params_full_precision, float scale) {
	// Construct weight matrices
	std::vector<GPUMatrix<float, RM>> weight_matrices_full_precision;

	if (m_n_hidden_layers == 0) {
		weight_matrices_full_precision.emplace_back(params_full_precision, m_padded_output_width, m_input_width);
	} else {
		weight_matrices_full_precision.emplace_back(params_full_precision, m_network_width, m_input_width);
		params_full_precision += weight_matrices_full_precision.back().n_elements();

		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
			weight_matrices_full_precision.emplace_back(params_full_precision, m_network_width, m_network_width);
			params_full_precision += weight_matrices_full_precision.back().n_elements();
		}

		weight_matrices_full_precision.emplace_back(params_full_precision, m_padded_output_width, m_network_width);
	}

	// Initialize matrices
	for (size_t i = 0; i < weight_matrices_full_precision.size(); ++i) {
		if (m_activation == Activation::Sine) {
			if (i == 0) {
				weight_matrices_full_precision[i].initialize_siren_uniform_first(rnd, scale);
			} else {
				weight_matrices_full_precision[i].initialize_siren_uniform(rnd, scale);
			}
		} else {
			weight_matrices_full_precision[i].initialize_xavier_uniform(rnd, scale);
		}
	}
}

// Explicitly instantiate CutlassMLP classes.
template class CutlassMLP<network_precision_t>;

}
