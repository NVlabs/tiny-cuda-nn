/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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
 *//*
 */

#include "tinyexr_wrapper.h"

#define TINYEXR_IMPLEMENTATION
#include <tinyexr/tinyexr.h>

void save_exr(const float* data, int width, int height, int nChannels, int channelStride, const char* outfilename) {
	EXRHeader header;
	InitEXRHeader(&header);

	EXRImage image;
	InitEXRImage(&image);

	image.num_channels = nChannels;

	std::vector<std::vector<float>> images(nChannels);
	std::vector<float*> image_ptr(nChannels);
	for (int i = 0; i < nChannels; ++i) {
		images[i].resize((size_t)width * (size_t)height);
	}

	for (int i = 0; i < nChannels; ++i) {
		image_ptr[i] = images[nChannels - i - 1].data();
	}

	for (size_t i = 0; i < (size_t)width * height; i++) {
		for (int c = 0; c < nChannels; ++c) {
			images[c][i] = data[channelStride*i+c];
		}
	}

	image.images = (unsigned char**)image_ptr.data();
	image.width = width;
	image.height = height;

	header.num_channels = nChannels;
	header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
	// Must be (A)BGR order, since most of EXR viewers expect this channel order.
	strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
	if (nChannels > 1) {
		strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
	}
	if (nChannels > 2) {
		strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';
	}
	if (nChannels > 3) {
		strncpy(header.channels[3].name, "A", 255); header.channels[3].name[strlen("A")] = '\0';
	}

	header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
	header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
	for (int i = 0; i < header.num_channels; i++) {
		header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
		header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
	}

	const char* err = NULL; // or nullptr in C++11 or later.
	int ret = SaveEXRImageToFile(&image, &header, outfilename, &err);
	if (ret != TINYEXR_SUCCESS) {
		std::string error_message = std::string("Failed to save EXR image: ") + err;
		FreeEXRErrorMessage(err); // free's buffer for an error message
		throw std::runtime_error(error_message);
	}
	printf("Saved exr file. [ %s ] \n", outfilename);

	free(header.channels);
	free(header.pixel_types);
	free(header.requested_pixel_types);
}


void load_exr(float** data, int* width, int* height, const char* filename) {
	const char* err = nullptr;

	int ret = LoadEXR(data, width, height, filename, &err);

	if (ret != TINYEXR_SUCCESS) {
		if (err) {
			std::string error_message = std::string("Failed to load EXR image: ") + err;
			FreeEXRErrorMessage(err);
			throw std::runtime_error(error_message);
		} else {
			throw std::runtime_error("Failed to load EXR image");
		}
	}
}
