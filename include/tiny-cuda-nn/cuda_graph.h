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

/** @file   cuda_graph.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of a CUDA graph capture/update with subsequent execution
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <cuda.h>



TCNN_NAMESPACE_BEGIN

class CudaGraph {
public:
	~CudaGraph() {
		reset();
	}

	template <typename F>
	void capture_and_execute(cudaStream_t stream, bool skip_capture, F fun) {
		cudaStreamCaptureStatus captureStatus;
		CUDA_CHECK_THROW(cudaStreamIsCapturing(stream, &captureStatus));
		skip_capture |= captureStatus == cudaStreamCaptureStatusActive; // If the caller is already capturing, no need for a nested capture.
		if (!skip_capture) {
			CUDA_CHECK_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
		}

		fun();

		if (!skip_capture) {
			if (m_graph) {
				CUDA_CHECK_THROW(cudaGraphDestroy(m_graph));
				m_graph = nullptr;
			}
			CUDA_CHECK_THROW(cudaStreamEndCapture(stream, &m_graph));

			cudaGraphExecUpdateResult update_result;
			cudaGraphNode_t error_node;
			if (m_graph_instance) {
				CUDA_CHECK_THROW(cudaGraphExecUpdate(m_graph_instance, m_graph, &error_node, &update_result));
			}

			if (!m_graph_instance || update_result != cudaGraphExecUpdateSuccess) {
				if (m_graph_instance) {
					CUDA_CHECK_THROW(cudaGraphExecDestroy(m_graph_instance));
				}
				CUDA_CHECK_THROW(cudaGraphInstantiate(&m_graph_instance, m_graph, NULL, NULL, 0));
			}

			CUDA_CHECK_THROW(cudaGraphLaunch(m_graph_instance, stream));
		}
	}

	void reset() {
		if (m_graph) {
			CUDA_CHECK_PRINT(cudaGraphDestroy(m_graph));
			m_graph = nullptr;
		}

		if (m_graph_instance) {
			CUDA_CHECK_PRINT(cudaGraphExecDestroy(m_graph_instance));
			m_graph_instance = nullptr;
		}
	}

private:
	cudaGraph_t m_graph = nullptr;
	cudaGraphExec_t m_graph_instance = nullptr;
};

TCNN_NAMESPACE_END
