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
 */

/** @file   encoding.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface for input encodings
 */

#include <tiny-cuda-nn/encoding.h>

#include <tiny-cuda-nn/encodings/composite.h>
#include <tiny-cuda-nn/encodings/frequency.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/encodings/identity.h>
#include <tiny-cuda-nn/encodings/oneblob.h>
#include <tiny-cuda-nn/encodings/spherical_harmonics.h>
#include <tiny-cuda-nn/encodings/triangle_wave.h>


TCNN_NAMESPACE_BEGIN

InterpolationType string_to_interpolation_type(const std::string& interpolation_type) {
	if (equals_case_insensitive(interpolation_type, "Nearest")) {
		return InterpolationType::Nearest;
	} else if (equals_case_insensitive(interpolation_type, "Linear")) {
		return InterpolationType::Linear;
	} else if (equals_case_insensitive(interpolation_type, "Smoothstep")) {
		return InterpolationType::Smoothstep;
	}

	throw std::runtime_error{fmt::format("Invalid interpolation type: {}", interpolation_type)};
}

std::string to_string(InterpolationType interpolation_type) {
	switch (interpolation_type) {
		case InterpolationType::Nearest: return "Nearest";
		case InterpolationType::Linear: return "Linear";
		case InterpolationType::Smoothstep: return "Smoothstep";
		default: throw std::runtime_error{"Invalid interpolation type."};
	}
}

ReductionType string_to_reduction_type(const std::string& reduction_type) {
	if (equals_case_insensitive(reduction_type, "Concatenation")) {
		return ReductionType::Concatenation;
	} else if (equals_case_insensitive(reduction_type, "Sum")) {
		return ReductionType::Sum;
	} else if (equals_case_insensitive(reduction_type, "Product")) {
		return ReductionType::Product;
	}

	throw std::runtime_error{fmt::format("Invalid reduction type: {}", reduction_type)};
}

std::string to_string(ReductionType reduction_type) {
	switch (reduction_type) {
		case ReductionType::Concatenation: return "Concatenation";
		case ReductionType::Sum: return "Sum";
		case ReductionType::Product: return "Product";
		default: throw std::runtime_error{"Invalid reduction type."};
	}
}

template <typename T>
Encoding<T>* create_encoding(uint32_t n_dims_to_encode, const json& encoding, uint32_t alignment) {
	std::string encoding_type = encoding.value("otype", "OneBlob");

	Encoding<T>* result;

	if (equals_case_insensitive(encoding_type, "Composite")) {
		result = new CompositeEncoding<T>{
			encoding,
			n_dims_to_encode,
		};
	} else if (equals_case_insensitive(encoding_type, "Identity")) {
		result = new IdentityEncoding<T>{
			n_dims_to_encode,
			encoding.value("scale", 1.0f),
			encoding.value("offset", 0.0f),
		};
	} else if (equals_case_insensitive(encoding_type, "Frequency")) {
		result = new FrequencyEncoding<T>{
			encoding.value("n_frequencies", 12u),
			n_dims_to_encode,
		};
	} else if (equals_case_insensitive(encoding_type, "TriangleWave")) {
		result = new TriangleWaveEncoding<T>{
			encoding.value("n_frequencies", 12u),
			n_dims_to_encode,
		};
	} else if (equals_case_insensitive(encoding_type, "SphericalHarmonics")) {
		result = new SphericalHarmonicsEncoding<T>{
			encoding.value("degree", 4u),
			n_dims_to_encode,
		};
	} else if (equals_case_insensitive(encoding_type, "OneBlob")) {
		result = new OneBlobEncoding<T>{encoding.value("n_bins", 16u), n_dims_to_encode};
	} else if (equals_case_insensitive(encoding_type, "OneBlobFrequency") || equals_case_insensitive(encoding_type, "NRC")) {
		json nrc_composite = {
			{"otype", "Composite"},
			{"nested", {
				{
					{"n_dims_to_encode", 3},
					{"otype", "TriangleWave"},
					{"n_frequencies", encoding.value("n_frequencies", 12u)},
				}, {
					{"n_dims_to_encode", 5},
					{"otype", "OneBlob"},
					{"n_bins", encoding.value("n_bins", 4u)},
				}, {
					{"otype", "Identity"},
				},
			}},
		};

		result = new CompositeEncoding<T>{
			nrc_composite,
			n_dims_to_encode,
		};
	} else if (
		equals_case_insensitive(encoding_type, "Grid") ||
		equals_case_insensitive(encoding_type, "HashGrid") ||
		equals_case_insensitive(encoding_type, "TiledGrid") ||
		equals_case_insensitive(encoding_type, "DenseGrid")
	) {
		result = create_grid_encoding<T>(n_dims_to_encode, encoding);
	} else {
		throw std::runtime_error{fmt::format("Invalid encoding type: {}", encoding_type)};
	}

	if (alignment > 0) {
		result->set_alignment(alignment);
	}
	return result;
}

#if TCNN_HALF_PRECISION
template Encoding<__half>* create_encoding(uint32_t n_dims_to_encode, const json& encoding, uint32_t alignment);
#endif
template Encoding<float>* create_encoding(uint32_t n_dims_to_encode, const json& encoding, uint32_t alignment);

TCNN_NAMESPACE_END
