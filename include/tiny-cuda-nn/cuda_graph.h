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

/** @file   cuda_graph.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of a CUDA graph capture/update with subsequent execution
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <cuda.h>

#include <deque>
#include <functional>

TCNN_NAMESPACE_BEGIN

class CudaGraph;

inline std::deque<CudaGraph*>& current_captures() {
	static thread_local std::deque<CudaGraph*> s_current_captures;
	return s_current_captures;
}

inline CudaGraph* current_capture() {
	return current_captures().empty() ? nullptr : current_captures().front();
}

class CudaGraph {
public:
	~CudaGraph() {
		try {
			reset();
		} catch (std::runtime_error error) {
			// Don't need to report on destruction problems when the driver is shutting down.
			if (std::string{error.what()}.find("driver shutting down") == std::string::npos) {
				std::cerr << "Could not destroy cuda graph: " << error.what() << std::endl;
			}
		}
	}

	ScopeGuard capture_guard(cudaStream_t stream) {
		// Can't capture on the global stream
		if (stream == nullptr || stream == cudaStreamLegacy) {
			return {};
		}

		// If the caller is already capturing, no need for a nested capture.
		cudaStreamCaptureStatus capture_status;
		CUDA_CHECK_THROW(cudaStreamIsCapturing(stream, &capture_status));
		if (capture_status != cudaStreamCaptureStatusNone) {
			return {};
		}

		cudaError_t capture_result = cudaStreamIsCapturing(cudaStreamLegacy, &capture_status);
		if (capture_result == cudaErrorStreamCaptureImplicit) {
			return {};
		}

		CUDA_CHECK_THROW(capture_result);
		if (capture_status != cudaStreamCaptureStatusNone) {
			return {};
		}

		// Start capturing
		if (m_graph) {
			CUDA_CHECK_THROW(cudaGraphDestroy(m_graph));
			m_graph = nullptr;
		}

		CUDA_CHECK_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
		current_captures().push_back(this);

		// Stop capturing again once the returned object goes out of scope
		return ScopeGuard{[this, stream]() {
			CUDA_CHECK_THROW(cudaStreamEndCapture(stream, &m_graph));

			if (current_captures().back() != this) {
				throw std::runtime_error{"CudaGraph: must end captures in reverse order of creation."};
			}
			current_captures().pop_back();

			if (m_synchronize_when_capture_done) {
				CUDA_CHECK_THROW(cudaDeviceSynchronize());
				m_synchronize_when_capture_done = false;
			}

			// Capture failed for some reason. Reset state and don't execute anything.
			// A corresponding exception is likely already in flight.
			if (!m_graph) {
				if (m_graph_instance) {
					CUDA_CHECK_THROW(cudaGraphExecDestroy(m_graph_instance));
				}

				m_graph = nullptr;
				m_graph_instance = nullptr;
				return;
			}

			// If we previously created a graph instance, try to update it with the newly captured graph.
			// This is cheaper than creating a new instance from scratch (and may involve just updating
			// pointers rather than changing the topology of the graph.)
			if (m_graph_instance) {
#if CUDA_VERSION >= 12000
				cudaGraphExecUpdateResultInfo update_result;
				CUDA_CHECK_THROW(cudaGraphExecUpdate(m_graph_instance, m_graph, &update_result));

				// If the update failed, reset graph instance. We will create a new one next.
				if (update_result.result != cudaGraphExecUpdateSuccess) {
					CUDA_CHECK_THROW(cudaGraphExecDestroy(m_graph_instance));
					m_graph_instance = nullptr;
				}
#else
				cudaGraphExecUpdateResult update_result;
				cudaGraphNode_t error_node;
				CUDA_CHECK_THROW(cudaGraphExecUpdate(m_graph_instance, m_graph, &error_node, &update_result));

				// If the update failed, reset graph instance. We will create a new one next.
				if (update_result != cudaGraphExecUpdateSuccess) {
					CUDA_CHECK_THROW(cudaGraphExecDestroy(m_graph_instance));
					m_graph_instance = nullptr;
				}
#endif
			}

			if (!m_graph_instance) {
				CUDA_CHECK_THROW(cudaGraphInstantiate(&m_graph_instance, m_graph, NULL, NULL, 0));
			}

			CUDA_CHECK_THROW(cudaGraphLaunch(m_graph_instance, stream));
		}};
	}

	void reset() {
		if (m_graph) {
			CUDA_CHECK_THROW(cudaGraphDestroy(m_graph));
			m_graph = nullptr;
		}

		if (m_graph_instance) {
			CUDA_CHECK_THROW(cudaGraphExecDestroy(m_graph_instance));
			m_graph_instance = nullptr;
		}
	}

	void schedule_synchronize() {
		m_synchronize_when_capture_done = true;
	}

private:
	cudaGraph_t m_graph = nullptr;
	cudaGraphExec_t m_graph_instance = nullptr;

	bool m_synchronize_when_capture_done = false;
};

TCNN_NAMESPACE_END
