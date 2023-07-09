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


namespace tcnn {

template <typename T>
void register_encoding(ci_hashmap<std::function<Encoding<T>*(uint32_t, const json&)>>& factories, const std::string& name, const std::function<Encoding<T>*(uint32_t, const json&)>& factory) {
	if (factories.count(name) > 0) {
		throw std::runtime_error{fmt::format("Can not register encoding '{}' twice.", name)};
	}

	factories.insert({name, factory});
}

template <typename T>
auto register_builtin_encodings() {
	ci_hashmap<std::function<Encoding<T>*(uint32_t, const json&)>> factories;

	register_encoding<T>(factories, "Composite", [](uint32_t n_dims_to_encode, const json& encoding) {
		return new CompositeEncoding<T>{encoding, n_dims_to_encode};
	});

	register_encoding<T>(factories, "Empty", [](uint32_t n_dims_to_encode, const json& encoding) {
		return new EmptyEncoding<T>{n_dims_to_encode};
	});

	register_encoding<T>(factories, "Frequency", [](uint32_t n_dims_to_encode, const json& encoding) {
		return new FrequencyEncoding<T>{encoding.value("n_frequencies", 12u), n_dims_to_encode};
	});

	auto grid_factory = [](uint32_t n_dims_to_encode, const json& encoding) {
		return create_grid_encoding<T>(n_dims_to_encode, encoding);
	};
	register_encoding<T>(factories, "Grid", grid_factory);
	register_encoding<T>(factories, "HashGrid", grid_factory);
	register_encoding<T>(factories, "TiledGrid", grid_factory);
	register_encoding<T>(factories, "DenseGrid", grid_factory);

	register_encoding<T>(factories, "Identity", [](uint32_t n_dims_to_encode, const json& encoding) {
		return new IdentityEncoding<T>{n_dims_to_encode, encoding.value("scale", 1.0f), encoding.value("offset", 0.0f)};
	});

	register_encoding<T>(factories, "OneBlob", [](uint32_t n_dims_to_encode, const json& encoding) {
		return new OneBlobEncoding<T>{encoding.value("n_bins", 16u), n_dims_to_encode};
	});

	register_encoding<T>(factories, "SphericalHarmonics", [](uint32_t n_dims_to_encode, const json& encoding) {
		return new SphericalHarmonicsEncoding<T>{encoding.value("degree", 4u), n_dims_to_encode};
	});

	register_encoding<T>(factories, "TriangleWave", [](uint32_t n_dims_to_encode, const json& encoding) {
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
	register_encoding<T>(factories, "OneBlobFrequency", nrc_factory);
	register_encoding<T>(factories, "NRC", nrc_factory);

	return factories;
}

template <typename T>
auto& encoding_factories() {
	static ci_hashmap<std::function<Encoding<T>*(uint32_t, const json&)>> factories = register_builtin_encodings<T>();
	return factories;
}

template <typename T>
void register_encoding(const std::string& name, const std::function<Encoding<T>*(uint32_t, const json&)>& factory) {
	register_encoding(encoding_factories<T>(), name, factory);
}

template <typename T>
Encoding<T>* create_encoding(uint32_t n_dims_to_encode, const json& encoding, uint32_t alignment) {
	std::string name = encoding.value("otype", "OneBlob");

	if (encoding_factories<T>().count(name) == 0) {
		throw std::runtime_error{fmt::format("Encoding '{}' not found", name)};
	}

	Encoding<T>* result = encoding_factories<T>().at(name)(n_dims_to_encode, encoding);
	if (alignment > 0) {
		result->set_alignment(alignment);
	}

	return result;
}

#if TCNN_HALF_PRECISION
template Encoding<__half>* create_encoding(uint32_t n_dims_to_encode, const json& encoding, uint32_t alignment);
#endif
template Encoding<float>* create_encoding(uint32_t n_dims_to_encode, const json& encoding, uint32_t alignment);

std::vector<std::string> builtin_encodings() {
	std::vector<std::string> result;
	for (const auto& kv : encoding_factories<float>()) {
		result.emplace_back(kv.first);
	}

	return result;
}

}
