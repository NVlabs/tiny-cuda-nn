/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

/** @file   encoding.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface for input encodings
 */

#include <tiny-cuda-nn/encoding.h>

#include <tiny-cuda-nn/encodings/frequency.h>
#include <tiny-cuda-nn/encodings/identity.h>
#include <tiny-cuda-nn/encodings/oneblob.h>
#include <tiny-cuda-nn/encodings/nrc.h>


#include <cutlass/half.h>


TCNN_NAMESPACE_BEGIN

template <typename T>
Encoding<T>* create_encoding(uint32_t n_dims_to_encode, uint32_t n_dims_to_pass_through, const json& encoding, uint32_t alignment) {
	std::string encoding_type = encoding.value("otype", "OneBlob");

	if (equals_case_insensitive(encoding_type, "Identity")) {
		return new IdentityEncoding<T>{
			n_dims_to_encode,
			n_dims_to_pass_through,
			encoding.value("scale", 1.0f),
			encoding.value("offset", 0.0f),
			alignment,
		};
	} else if (equals_case_insensitive(encoding_type, "Frequency")) {
		return new FrequencyEncoding<T>{
			encoding.value("n_frequencies", 12u),
			n_dims_to_encode,
			n_dims_to_pass_through,
			alignment,
		};
	} else if (equals_case_insensitive(encoding_type, "OneBlob")) {
		return new OneBlobEncoding<T>{encoding.value("n_bins", 16u), n_dims_to_encode, n_dims_to_pass_through, alignment, encoding.value("n_trailing_dims_to_ignore", 0u)};
	} else if (equals_case_insensitive(encoding_type, "OneBlobFrequency") || equals_case_insensitive(encoding_type, "NRC")) {
		return new NrcEncoding<T>{
			encoding.value("n_frequencies", 12u),
			encoding.value("n_bins", 4u),
			n_dims_to_encode,
			n_dims_to_pass_through,
			alignment,
		};
	} else {
		throw std::runtime_error{std::string{"Invalid encoding type: "} + encoding_type};
	}
}

template Encoding<float>* create_encoding(uint32_t n_dims_to_encode, uint32_t n_dims_to_pass_through, const json& encoding, uint32_t alignment);
template Encoding<__half>* create_encoding(uint32_t n_dims_to_encode, uint32_t n_dims_to_pass_through, const json& encoding, uint32_t alignment);

TCNN_NAMESPACE_END
