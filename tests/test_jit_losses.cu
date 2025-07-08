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

/** @file   test_jit_losses.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  Test that losses produce the same value whether JIT fusion is
            active or not.
 */

#include "test_common.h"

#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/random.h>

using namespace tcnn;

TEMPLATE_TEST_CASE("Losses match with and without JIT", "[loss][jit]", network_precision_t, float) {
	using T = TestType;

	if (!supports_jit_fusion()) {
		SUCCEED("GPU target does not support JIT.");
	}

	tcnn_test_setup();

	for (const auto& loss_name : builtin_losses()) {
		SECTION(fmt::format("{} matches with JIT version", loss_name)) {
			const float loss_scale = default_loss_scale<T>();
			const uint32_t batch_size = 32 * BATCH_SIZE_GRANULARITY;
			const uint32_t n_dims = 6; // arbitrary choice
			const uint32_t n_dims_padded = 16; // common value due to tensor core width

			pcg32 rng{1337};

			GPUMatrix<T> prediction{n_dims_padded, batch_size};
			GPUMatrix<float> target{n_dims, batch_size};
			GPUMatrix<float> pdf{n_dims, batch_size};

			// Compute the loss on random, positive values. Most losses
			// also work on negative values, but some (cross-entropy and
			// variance) require positive values.
			prediction.initialize_uniform(rng, 0.01f, 5.0f);
			target.initialize_uniform(rng, 0.01f, 5.0f);
			pdf.initialize_uniform(rng, 0.01f, 1.99f); // average of 1

			GPUMatrix<float> values{n_dims_padded, batch_size};
			GPUMatrix<T> gradients{n_dims_padded, batch_size};

			auto loss = default_loss<T>(loss_name);
			REQUIRE_THAT(loss_name, Catch::Matchers::Equals(loss->name()));

			std::string kernel_name = to_snake_case(loss_name);
			CudaRtcKernel jit_kernel{kernel_name, dfmt(0, R"(
					{DEVICE_FUNCTION}

					__global__ void {KERNEL_NAME}(const uint32_t n_elements, MatrixView<const {T}> data_prediction, MatrixView<const float> data_target, MatrixView<float> data_values, MatrixView<{T}> data_gradients, MatrixView<float> data_pdf) {{
						const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

						auto pred = data_prediction.col<{N_DIMS}>(i);
						auto targ = data_target.col<{N_DIMS}>(i);

						vec<{N_DIMS}> val, pdf = {{1.0f}};
						if (data_pdf) {{ pdf = data_pdf.col<{N_DIMS}>(i); }}
						auto grad = eval_loss(n_elements * {N_DIMS}, (float){LOSS_SCALE}, pred, targ, pdf, &val);

						data_values.set_col(i, vec<{N_DIMS_PADDED}>(val));
						data_gradients.set_col(i, tvec<{T}, {N_DIMS_PADDED}>(grad));
					}}
				)",
				"DEVICE_FUNCTION"_a = loss->generate_device_function("eval_loss", n_dims),
				"KERNEL_NAME"_a = kernel_name,
				"T"_a = type_to_string<T>(),
				"N_DIMS"_a = n_dims,
				"N_DIMS_PADDED"_a = n_dims_padded,
				"LOSS_SCALE"_a = loss_scale
			)};

			SECTION("No PDF") {
				loss->evaluate(default_loss_scale<T>(), prediction, target, values, gradients);
				auto v1 = values.to_cpu_vector();
				auto g1 = gradients.to_cpu_vector();

				jit_kernel.launch(n_blocks_linear(batch_size, N_THREADS_LINEAR), N_THREADS_LINEAR, 0, nullptr, batch_size, prediction.view(), target.view(), values.view(), gradients.view(), MatrixView<float>{});
				auto v2 = values.to_cpu_vector();
				auto g2 = gradients.to_cpu_vector();

				REQUIRE(v1.size() == g1.size());
				vector_match_rae("Values", v1, v2, 1e-3);
				vector_match_rae("Gradients", g1, g2, 1e-3);
			}

			SECTION("PDF") {
				loss->evaluate(default_loss_scale<T>(), prediction, target, values, gradients, &pdf);
				auto v1 = values.to_cpu_vector();
				auto g1 = gradients.to_cpu_vector();

				jit_kernel.launch(n_blocks_linear(batch_size, N_THREADS_LINEAR), N_THREADS_LINEAR, 0, nullptr, batch_size, prediction.view(), target.view(), values.view(), gradients.view(), pdf.view());
				auto v2 = values.to_cpu_vector();
				auto g2 = gradients.to_cpu_vector();

				REQUIRE(v1.size() == g1.size());
				vector_match_rae("Values", v1, v2, 1e-3, 0.99);
				vector_match_rae("Gradients", g1, g2, 1e-3, 0.99);
			}
		}
	}
}
