#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "tiny-cuda-nn/encodings/grid.h"

TEST_CASE("3D All 0", "GRID")
{
  const char *config = R"({"base_resolution":32,"log2_hashmap_size":16,"n_features_per_level":2,"n_levels":20,"otype":"HashGrid","per_level_scale":1.5})";
  nlohmann::json config_json = nlohmann::json::parse(config);
  tcnn::GridEncoding<float> *g = tcnn::create_grid_encoding<float>(3, config_json);

  tcnn::GPUMatrix<float> input(g->n_pos_dims(), 1);
  tcnn::GPUMatrix<float> output(g->padded_output_width(), 1);
  input.memset(0);
  
  std::unique_ptr<tcnn::Context> c = g->forward(input, &output);

  std::vector<float> result_host(output.n_elements());
  CUDA_CHECK_THROW(cudaMemcpy(result_host.data(), output.data(), output.n_bytes(), cudaMemcpyDeviceToHost));
}
