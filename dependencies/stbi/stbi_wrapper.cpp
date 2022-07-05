/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "stbi_wrapper.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <fmt/core.h>

#include <cmath>
#include <stdexcept>
#include <vector>

void save_stbi(const uint8_t* data, int width, int height, int n_channels, const char* outfilename) {
	if (stbi_write_jpg(outfilename, width, height, n_channels, data, 100) == 0) {
		throw std::runtime_error{fmt::format("Failed to write image {}", outfilename)};
	}
}

void save_stbi(const float* data, int width, int height, int n_channels, const char* outfilename) {
	if (stbi_write_hdr(outfilename, width, height, n_channels, data) == 0) {
		throw std::runtime_error{fmt::format("Failed to write image {}", outfilename)};
	}
}

float* load_stbi(int* width, int* height, const char* filename) {
	int n_channels = 4;
	float* data = stbi_loadf(filename, width, height, &n_channels, n_channels);
	if (!data) {
		throw std::runtime_error{std::string{stbi_failure_reason()}};
	}
	return data;
}
