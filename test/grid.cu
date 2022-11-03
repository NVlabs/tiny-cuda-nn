#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "tiny-cuda-nn/encodings/grid.h"

TEST_CASE("3D All 0", "GRID")
{
  const char *config = R"({"base_resolution":32,"log2_hashmap_size":16,"n_features_per_level":2,"n_levels":20,"otype":"HashGrid","per_level_scale":1.5})";
  nlohmann::json config_json = nlohmann::json::parse(config);
  tcnn::GridEncoding<float> *g = tcnn::create_grid_encoding<float>(3, config_json);

  // Current codebase only accept multiple of tcnn::batch_size_granularity
  unsigned int batch_size = tcnn::batch_size_granularity;
  REQUIRE(batch_size == 128);
  
  tcnn::GPUMatrix<float> input(g->n_pos_dims(), batch_size);
  tcnn::GPUMatrix<float> output(g->padded_output_width(), batch_size);
  input.memset(0);

  REQUIRE(input.n_elements() == 3 /* dimensions */ * 128 /* batch size */);
  REQUIRE(output.n_elements() == 2 /* feture per level */ * 20 /* levels*/ * 128 /* batch size */);
  
  std::unique_ptr<tcnn::Context> c = g->forward(input, &output);

  std::vector<float> result_host(output.n_elements());
  CUDA_CHECK_THROW(cudaMemcpy(result_host.data(), output.data(), output.n_bytes(), cudaMemcpyDeviceToHost));
}
