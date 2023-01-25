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

/** @file   multi_stream.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Helper class for parallelizing workload across multiple streams.
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <stack>

TCNN_NAMESPACE_BEGIN

void free_multi_streams(cudaStream_t parent_stream);

// Synchronization helpers
struct StreamAndEvent {
public:
	StreamAndEvent() {
		CUDA_CHECK_THROW(cudaStreamCreate(&m_stream));
		CUDA_CHECK_THROW(cudaEventCreate(&m_event));
	}

	~StreamAndEvent() {
		if (m_stream) {
			free_multi_streams(m_stream);
			free_gpu_memory_arena(m_stream);
			cudaStreamDestroy(m_stream);
		}

		if (m_event) {
			cudaEventDestroy(m_event);
		}
	}

	// Only allow moving of these guys. No copying.
	StreamAndEvent& operator=(const StreamAndEvent&) = delete;
	StreamAndEvent(const StreamAndEvent&) = delete;
	StreamAndEvent& operator=(StreamAndEvent&& other) {
		std::swap(m_stream, other.m_stream);
		std::swap(m_event, other.m_event);
		return *this;
	}

	StreamAndEvent(StreamAndEvent&& other) {
		*this = std::move(other);
	}

	void wait_for(cudaEvent_t event) {
		CUDA_CHECK_THROW(cudaStreamWaitEvent(m_stream, event, 0));
	}

	void wait_for(cudaStream_t stream) {
		CUDA_CHECK_THROW(cudaEventRecord(m_event, stream));
		wait_for(m_event);
	}

	void signal(cudaStream_t stream) {
		CUDA_CHECK_THROW(cudaEventRecord(m_event, m_stream));
		CUDA_CHECK_THROW(cudaStreamWaitEvent(stream, m_event, 0));
	}

	cudaStream_t get() {
		return m_stream;
	}

private:
	cudaStream_t m_stream = {};
	cudaEvent_t m_event = {};
};

struct MultiStream {
public:
	MultiStream() {
		CUDA_CHECK_THROW(cudaEventCreate(&m_event));
	}

	~MultiStream() {
		cudaEventDestroy(m_event);
	}

	MultiStream& operator=(const MultiStream&) = delete;
	MultiStream(const MultiStream&) = delete;
	MultiStream& operator=(MultiStream&&) = delete;
	MultiStream(MultiStream&&) = delete;

	void signal(cudaStream_t outer_stream) {
		for (size_t i = 0; i < m_n_streams; ++i) {
			m_streams[i].signal(outer_stream);
		}
	}

	void wait_for(cudaStream_t stream) {
		if (m_n_streams == 0) {
			return;
		}

		CUDA_CHECK_THROW(cudaEventRecord(m_event, stream));
		for (size_t i = 0; i < m_n_streams; ++i) {
			m_streams[i].wait_for(m_event);
		}
	}

	void resize(size_t n_streams) {
		if (n_streams > m_streams.size()) {
			m_streams.resize(n_streams);
		}
		m_n_streams = n_streams;
	}

	cudaStream_t get(size_t idx) {
		if (idx >= m_n_streams) {
			throw std::runtime_error{fmt::format("MultiStream: invalid stream index requested: {}/{}", idx, m_n_streams)};
		}
		return m_streams.at(idx).get();
	}

private:
	std::vector<StreamAndEvent> m_streams;
	// May be less than m_streams.size()!
	// The user may only need to sync fewer than that.
	size_t m_n_streams = 0;
	cudaEvent_t m_event;
};

inline std::unordered_map<cudaStream_t, std::stack<std::shared_ptr<MultiStream>>>& stream_multi_streams() {
	static auto* stream_multi_streams = new std::unordered_map<cudaStream_t, std::stack<std::shared_ptr<MultiStream>>>{};
	return *stream_multi_streams;
}

inline std::unordered_map<int, std::stack<std::shared_ptr<MultiStream>>>& global_multi_streams() {
	static auto* global_multi_streams = new std::unordered_map<int, std::stack<std::shared_ptr<MultiStream>>>{};
	return *global_multi_streams;
}

inline std::stack<std::shared_ptr<MultiStream>>& get_multi_stream_stack(cudaStream_t parent_stream) {
	return parent_stream ? stream_multi_streams()[parent_stream] : global_multi_streams()[cuda_device()];
}

inline void free_multi_streams(cudaStream_t parent_stream) {
	CHECK_THROW(parent_stream);

	// Copy the multi stream shared_ptr's into a separate variable,
	// such that their destruction happens after unordered_map::erase(...)
	// is already finished. This alleviates potential non-reentrancy problems.
	auto multi_streams = stream_multi_streams()[parent_stream];
	stream_multi_streams().erase(parent_stream);
}

inline std::shared_ptr<MultiStream> reserve_multi_stream(cudaStream_t parent_stream, size_t n_streams) {
	auto& stack = get_multi_stream_stack(parent_stream);
	if (stack.empty()) {
		stack.push(std::make_shared<MultiStream>());
	}
	auto result = stack.top();
	stack.pop();

	result->resize(n_streams);
	return result;
}

inline void return_multi_stream(cudaStream_t parent_stream, std::shared_ptr<MultiStream> multi_stream) {
	if (parent_stream ? (stream_multi_streams().count(parent_stream) == 0) : (global_multi_streams().count(cuda_device()) == 0)) {
		throw std::runtime_error{"Attempted to return multi stream to the wrong parent stream."};
	}

	auto& stack = get_multi_stream_stack(parent_stream);
	stack.push(multi_stream);
}

// RAII wrapper around MultiStream
struct SyncedMultiStream {
public:
	SyncedMultiStream() = default;
	SyncedMultiStream(cudaStream_t stream, size_t n_streams) : m_main_stream{stream}, m_n_streams{n_streams} {
		if (m_n_streams == 0) {
			throw std::runtime_error{"SyncedMultiStream: must request at least one stream"};
		} else if (m_n_streams == 1) {
			return;
		}

		m_multi_stream = reserve_multi_stream(m_main_stream, m_n_streams-1);
		m_multi_stream->wait_for(m_main_stream);
	}

	~SyncedMultiStream() {
		if (m_multi_stream) {
			m_multi_stream->signal(m_main_stream);
			return_multi_stream(m_main_stream, m_multi_stream);
		}
	}

	// Only allow moving of these guys. No copying.
	SyncedMultiStream& operator=(const SyncedMultiStream& other) = delete;
	SyncedMultiStream(const SyncedMultiStream&) = delete;

	SyncedMultiStream& operator=(SyncedMultiStream&& other) {
		std::swap(m_multi_stream, other.m_multi_stream);
		std::swap(m_main_stream, other.m_main_stream);
		std::swap(m_n_streams, other.m_n_streams);
		return *this;
	}

	SyncedMultiStream(SyncedMultiStream&& other) {
		*this = std::move(other);
	}

	cudaStream_t get(size_t idx) {
		if (m_n_streams == 0) {
			throw std::runtime_error{"SyncedMultiStream: must have at least one stream"};
		}

		if (idx == 0) {
			return m_main_stream;
		} else {
			if (!m_multi_stream) {
				throw std::runtime_error{"SyncedMultiStream: invalid multistream"};
			}

			return m_multi_stream->get(idx-1);
		}
	}

private:
	std::shared_ptr<MultiStream> m_multi_stream = nullptr;
	cudaStream_t m_main_stream = nullptr;
	size_t m_n_streams = 0;
};

TCNN_NAMESPACE_END
