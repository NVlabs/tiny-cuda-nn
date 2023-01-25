/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <tiny-cuda-nn/encodings/empty.h>
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
void register_builtin_encodings() {
	register_encoding<T>("Composite", [](uint32_t n_dims_to_encode, const json& encoding) {
		return new CompositeEncoding<T>{encoding, n_dims_to_encode};
	});

	register_encoding<T>("Empty", [](uint32_t n_dims_to_encode, const json& encoding) {
		return new EmptyEncoding<T>{n_dims_to_encode};
	});

	register_encoding<T>("Frequency", [](uint32_t n_dims_to_encode, const json& encoding) {
		return new FrequencyEncoding<T>{encoding.value("n_frequencies", 12u), n_dims_to_encode};
	});

	auto grid_factory = [](uint32_t n_dims_to_encode, const json& encoding) {
		return create_grid_encoding<T>(n_dims_to_encode, encoding);
	};
	register_encoding<T>("Grid", grid_factory);
	register_encoding<T>("HashGrid", grid_factory);
	register_encoding<T>("TiledGrid", grid_factory);
	register_encoding<T>("DenseGrid", grid_factory);

	register_encoding<T>("Identity", [](uint32_t n_dims_to_encode, const json& encoding) {
		return new IdentityEncoding<T>{n_dims_to_encode, encoding.value("scale", 1.0f), encoding.value("offset", 0.0f)};
	});

	register_encoding<T>("OneBlob", [](uint32_t n_dims_to_encode, const json& encoding) {
		return new OneBlobEncoding<T>{encoding.value("n_bins", 16u), n_dims_to_encode};
	});

	register_encoding<T>("SphericalHarmonics", [](uint32_t n_dims_to_encode, const json& encoding) {
		return new SphericalHarmonicsEncoding<T>{encoding.value("degree", 4u), n_dims_to_encode};
	});

	register_encoding<T>("TriangleWave", [](uint32_t n_dims_to_encode, const json& encoding) {
		return new TriangleWaveEncoding<T>{encoding.value("n_frequencies", 12u), n_dims_to_encode};
	});

	auto nrc_factory = [](uint32_t n_dims_to_encode, const json& encoding) {
		return new CompositeEncoding<T>{
			{
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
			},
			n_dims_to_encode,
		};
	};
	register_encoding<T>("OneBlobFrequency", nrc_factory);
	register_encoding<T>("NRC", nrc_factory);
}

template <typename T>
auto& encoding_factories() {
	static std::unordered_map<std::string, std::function<Encoding<T>*(uint32_t, const json&)>> factories;
	return factories;
}

template <typename T>
void register_encoding(const std::string& name, const std::function<Encoding<T>*(uint32_t, const json&)>& factory) {
	if (encoding_factories<T>().count(to_lower(name)) > 0) {
		throw std::runtime_error{fmt::format("Can not register encoding '{}' twice.", name)};
	}

	encoding_factories<T>().insert({to_lower(name), factory});
}

template <typename T>
Encoding<T>* create_encoding(uint32_t n_dims_to_encode, const json& encoding, uint32_t alignment) {
	// Calls register_builtin_encodings<T>() on first invocation of create_encoding<T>(...)
	// in a thread-safe manner and ensures all concurrent calls progress further only
	// once register_builtin_encodings<T>() terminated. See the C++ documentation on static
	// local variables for more information.
	static struct Init { Init() { register_builtin_encodings<T>(); } } init;

	std::string name = encoding.value("otype", "OneBlob");

	if (encoding_factories<T>().count(to_lower(name)) == 0) {
		throw std::runtime_error{fmt::format("Encoding '{}' not found", name)};
	}

	Encoding<T>* result = encoding_factories<T>().at(to_lower(name))(n_dims_to_encode, encoding);
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
