#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "tiny-cuda-nn/encodings/grid.h"

TEST_CASE("3D All 0", "GRID")
{
  const char *config = R"({"base_resolution":32,"log2_hashmap_size":16,"n_features_per_level":2,"n_levels":20,"otype":"HashGrid","per_level_scale":1.5})";
  nlohmann::json config_json = nlohmann::json::parse(config);
  tcnn::GridEncoding<float> *g = tcnn::create_grid_encoding<float>(3, config_json);

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
  tcnn::GPUMemory<char> params_buffer;
  params_buffer.resize(sizeof(float) * n_params);
  float *params = (float*)(params_buffer.data());
  float *inference_params = params;
  float *backward_params = nullptr;
  float *gradients = nullptr;
  // Using the same values for params and inference params in this test, not setting the rest.
  g->set_params(params, inference_params, backward_params, gradients);

  unsigned int batch_size = 1;  
  tcnn::GPUMatrix<float> input(g->n_pos_dims(), batch_size);
  tcnn::GPUMatrix<float> output(g->padded_output_width(), batch_size);
  input.memset(0);

  REQUIRE(input.n_elements() == 3 /* dimensions */ * 1 /* batch size */);
  REQUIRE(output.n_elements() == 2 /* feture per level */ * 20 /* levels*/ * 1 /* batch size */);
  
  std::unique_ptr<tcnn::Context> c = g->forward(input, &output);

  std::vector<float> result_host(output.n_elements());
  CUDA_CHECK_THROW(cudaMemcpy(result_host.data(), output.data(), output.n_bytes(), cudaMemcpyDeviceToHost));
}
