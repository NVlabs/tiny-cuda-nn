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

/** @file   test_grid.cu
 *  @author Pierre Wilmot and Thomas Müller, NVIDIA
 *  @brief  Test basic aspects of GridEncoding
 */

#include "test_common.h"

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/encodings/multi_level_interface.h>

using namespace tcnn;

TEST_CASE("GridEncoding sanity checks", "[encoding]") {
	tcnn_test_setup();

	const char* config = R"({
		"otype": "Grid",
		"base_resolution": 32,
		"log2_hashmap_size": 16,
		"n_features_per_level": 2,
		"n_levels": 20,
		"otype": "HashGrid",
		"per_level_scale": 1.5
	})";
	nlohmann::json config_json = nlohmann::json::parse(config);
	std::unique_ptr<MultiLevelEncoding<float>> g{dynamic_cast<MultiLevelEncoding<float>*>(create_encoding<float>(3, config_json))};

	REQUIRE(g);

	REQUIRE(g->n_pos_dims() == 3);
	REQUIRE(g->n_features_per_level() == 2);
	REQUIRE(g->padded_output_width() == 20 /* levels */ * 2 /* nb_fetures_per_level */);

	// Level 0 is a dense layer, 3 dimensions with resilution 32
	REQUIRE(g->level_n_params(0) == 32 * 32 * 32);
	REQUIRE(g->level_params_offset(0) == 0);
	// Level 1 is a hash layer as 48 * 48 * 48 > 2 ^ log2_hashmap_size (65536)
	REQUIRE(g->level_n_params(1) == 65536);
	REQUIRE(g->level_params_offset(1) == 32 * 32 * 32);
	// Level 2 is a hash layer as 72 * 72 * 72 > 2 ^ log2_hashmap_size (65536)
	REQUIRE(g->level_n_params(2) == 65536);
	REQUIRE(g->level_params_offset(2) == 32 * 32 * 32 + 65536);

	// Parameters are NOT an encapsulated member of GridEncoding.
	// We need to allocate them manually and set them.
	size_t n_params = g->n_params();
	REQUIRE(n_params == 2555904);
	GPUMemory<char> params_buffer;
	params_buffer.resize(sizeof(float) * n_params);
	float* params = (float*)(params_buffer.data());
	float* inference_params = params;
	float* gradients = nullptr;
	// Using the same values for params and inference params in this test, not setting the rest.
	g->set_params(params, inference_params, gradients);

	unsigned int batch_size = BATCH_SIZE_GRANULARITY;
	GPUMatrix<float> input(g->n_pos_dims(), batch_size);
	GPUMatrix<float> output(g->padded_output_width(), batch_size);
	input.memset(0);

	REQUIRE(input.n_elements() == 3 /* dimensions */ * batch_size);
	REQUIRE(output.n_elements() == 2 /* feture per level */ * 20 /* levels*/ * batch_size);

	std::unique_ptr<Context> c = g->forward(input, &output);

	std::vector<float> result_host(output.n_elements());
	CUDA_CHECK_THROW(cudaMemcpy(result_host.data(), output.data(), output.n_bytes(), cudaMemcpyDeviceToHost));
}

TEST_CASE("DenseGrid with a single level", "[encoding]") {
	tcnn_test_setup();

	// Without an explicit per_level_scale, dense grids derive one from n_levels. This must stay finite for n_levels=1.
	const char* config = R"({
		"otype": "DenseGrid",
		"n_levels": 1,
		"n_features_per_level": 2,
		"base_resolution": 16
	})";
	nlohmann::json config_json = nlohmann::json::parse(config);
	std::unique_ptr<MultiLevelEncoding<float>> g{dynamic_cast<MultiLevelEncoding<float>*>(create_encoding<float>(3, config_json))};

	REQUIRE(g);
	REQUIRE(g->level_n_params(0) == 16 * 16 * 16);
	REQUIRE(g->n_params() == 16 * 16 * 16 * 2);

	std::vector<float> params_host(g->n_params());
	for (size_t i = 0; i < params_host.size(); ++i) {
		params_host[i] = (float)(i % 7) * 0.25f - 0.5f;
	}

	GPUMemory<float> params(params_host.size());
	params.copy_from_host(params_host);
	g->set_params(params.data(), params.data(), nullptr);

	const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	std::vector<float> input_host(3 * batch_size);
	for (size_t i = 0; i < input_host.size(); ++i) {
		input_host[i] = (float)((i * 37) % 101) / 101.0f;
	}

	GPUMatrix<float> input(3, batch_size);
	CUDA_CHECK_THROW(cudaMemcpy(input.data(), input_host.data(), input.n_bytes(), cudaMemcpyHostToDevice));
	GPUMatrix<float> output(g->padded_output_width(), batch_size);
	g->forward(input, &output);

	std::vector<float> output_host(output.n_elements());
	CUDA_CHECK_THROW(cudaMemcpy(output_host.data(), output.data(), output.n_bytes(), cudaMemcpyDeviceToHost));
	size_t n_nonfinite = 0;
	for (float v : output_host) {
		n_nonfinite += !std::isfinite(v);
	}
	REQUIRE(n_nonfinite == 0);
}
