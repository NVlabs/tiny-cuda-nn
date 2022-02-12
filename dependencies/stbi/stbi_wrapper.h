/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cstdint>

void save_stbi(const uint8_t* data, int width, int height, int n_channels, const char* outfilename);
void save_stbi(const float* data, int width, int height, int n_channels, const char* outfilename);
float* load_stbi(int* width, int* height, const char* filename);
