/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

/** @file   vec_json.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Conversion between tcnn's vector / matrix / quaternion types
 *          and nlohmann::json.
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <json/json.hpp>

namespace tcnn {

template <typename T, uint32_t N, uint32_t M>
void to_json(nlohmann::json& j, const tmat<T, N, M>& mat) {
	for (int row = 0; row < M; ++row) {
		nlohmann::json column = nlohmann::json::array();
		for (int col = 0; col < N; ++col) {
			column.push_back(mat[col][row]);
		}
		j.push_back(column);
	}
}

template <typename T, uint32_t N, uint32_t M>
void from_json(const nlohmann::json& j, tmat<T, N, M>& mat) {
	for (std::size_t row = 0; row < M; ++row) {
		const auto& jrow = j.at(row);
		for (std::size_t col = 0; col < N; ++col) {
			const auto& value = jrow.at(col);
			mat[col][row] = value.get<T>();
		}
	}
}

template <typename T, uint32_t N>
void to_json(nlohmann::json& j, const tvec<T, N>& v) {
	for (uint32_t i = 0; i < N; ++i) {
		j.push_back(v[i]);
	}
}

template <typename T, uint32_t N>
void from_json(const nlohmann::json& j, tvec<T, N>& v) {
	for (uint32_t i = 0; i < N; ++i) {
		v[i] = j.at(i).get<T>();
	}
}

template <typename T>
void to_json(nlohmann::json& j, const tquat<T>& q) {
	j.push_back(q.x);
	j.push_back(q.y);
	j.push_back(q.z);
	j.push_back(q.w);
}

template <typename T>
void from_json(const nlohmann::json& j, tquat<T>& q) {
	q.x = j.at(0).get<T>();
	q.y = j.at(1).get<T>();
	q.z = j.at(2).get<T>();
	q.w = j.at(3).get<T>();
}

}
