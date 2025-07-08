/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

/** @file   ministd.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Tiny implementation of useful functions and types typically found
 *          in the STL. For CUDA RTC, we implement a subset of those outselves
 *          to avoid having to bundle the STL with our release.
 */

#pragma once

#ifdef __CUDACC__
#  ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#    pragma nv_diag_suppress = unsigned_compare_with_zero
#  else
#    pragma diag_suppress = unsigned_compare_with_zero
#  endif
#  include <cuda_fp16.h>
#endif

#ifndef assert
#define assert(x)
#endif

#if defined(_WIN32) or defined(_WIN64)
using int64_t = long long; using uint64_t = unsigned long long;
#else
using int64_t = long;      using uint64_t = unsigned long;
#endif
using int32_t = int;       using uint32_t = unsigned int;
using int16_t = short;     using uint16_t = unsigned short;
using int8_t = char;       using uint8_t = unsigned char;
using ptrdiff_t = int64_t; using size_t = uint64_t;

namespace std {
	using int64_t = ::int64_t;     using uint64_t = ::uint64_t;
	using int32_t = ::int32_t;     using uint32_t = ::uint32_t;
	using int16_t = ::int16_t;     using uint16_t = ::uint16_t;
	using int8_t = ::int8_t;       using uint8_t = ::uint8_t;
	using ptrdiff_t = ::ptrdiff_t; using size_t = ::size_t;

	template <typename T> struct remove_const          { using type = T; };
	template <typename T> struct remove_const<const T> { using type = T; };
	template <typename T> using remove_const_t = typename remove_const<T>::type;

	template <typename T, typename U> struct is_same { static const bool value = false; };
	template <typename T> struct is_same<T, T> { static const bool value = true; };

	template <bool B, typename T, typename U> struct conditional { using type = void; };
	template <typename T, typename U> struct conditional<true, T, U> { using type = T; };
	template <typename T, typename U> struct conditional<false, T, U> { using type = U; };
	template <bool B, typename T, typename U> using conditional_t = typename conditional<B, T, U>::type;

	template <bool B, class T = void> struct enable_if {};
	template <class T> struct enable_if<true, T> { typedef T type; };
	template <bool B, class T = void> using enable_if_t = typename enable_if<B, T>::type;

	template <class T, T v>
	struct integral_constant {
		static constexpr T value = v;
		using value_type = T;
		using type = integral_constant;
		constexpr __host__ __device__ operator value_type() const noexcept { return value; }
		constexpr __host__ __device__ value_type operator()() const noexcept { return value; }
	};

	using true_type = integral_constant<bool, true>;
	using false_type = integral_constant<bool, false>;

	template <typename T> struct is_unsigned : std::integral_constant<bool, T(0) < T(-1)> {};
	template <typename T> struct is_signed : std::integral_constant<bool, T(-1) < T(0)> {};

	template <typename T> struct make_signed { static_assert(is_signed<T>::value, "Unsupported type"); using type = T; };
	template <> struct make_signed<unsigned char> { using type = char; };
	template <> struct make_signed<unsigned short> { using type = short; };
	template <> struct make_signed<unsigned int> { using type = int; };
	template <> struct make_signed<unsigned long> { using type = long; };
	template <> struct make_signed<unsigned long long> { using type = long long; };
	template <typename T> using make_signed_t = typename make_signed<T>::type;

	template <typename T> struct make_unsigned { static_assert(is_unsigned<T>::value, "Unsupported type"); using type = T; };
	template <> struct make_unsigned<char> { using type = unsigned char; };
	template <> struct make_unsigned<short> { using type = unsigned short; };
	template <> struct make_unsigned<int> { using type = unsigned int; };
	template <> struct make_unsigned<long> { using type = unsigned long; };
	template <> struct make_unsigned<long long> { using type = unsigned long long; };
	template <typename T> using make_unsigned_t = typename make_unsigned<T>::type;

	template <typename T>
	struct numeric_limits {};

	template <>
	struct numeric_limits<double> {
		static constexpr __host__ __device__ double infinity() { return __longlong_as_double(0x7ff0000000000000ULL); }
		static constexpr __host__ __device__ double min() { return __longlong_as_double(0x0000000000000001ULL); }
		static constexpr __host__ __device__ double max() { return __longlong_as_double(0x7ff0000000000000ULL - 1ULL); }
		static constexpr __host__ __device__ double epsilon() { return 2.2204460492503131e-016; }
	};

	template <>
	struct numeric_limits<float> {
		static constexpr __host__ __device__ float infinity() { return __int_as_float(0x7f800000U); }
		static constexpr __host__ __device__ float min() { return __int_as_float(0x00000001U); }
		static constexpr __host__ __device__ float max() { return __int_as_float(0x7f7fffffU); }
		static constexpr __host__ __device__ float epsilon() { return 1.192092896e-07F; }
	};

#ifdef __CUDACC__
	template <>
	struct numeric_limits<__half> {
		static constexpr __host__ __device__ __half infinity() { return __short_as_half(0x7c00U); }
		static constexpr __host__ __device__ __half min() { return __short_as_half(0x0001U); }
		static constexpr __host__ __device__ __half max() { return __short_as_half(0x7bffU); }
		static constexpr __host__ __device__ __half epsilon() { return __short_as_half(0x3c01) - __short_as_half(0x3c00); }
	};
#endif

	// Useful math functions. CUDA already comes with those, so we simply forward.
	template <typename T> __host__ __device__ T min(T a, T b) { return ::min(a, b); }
	template <typename T> __host__ __device__ T max(T a, T b) { return ::max(a, b); }
	template <typename T> __host__ __device__ T clamp(T a, T b, T c) { return tcnn::clamp(a, b, c); }
	template <typename T> __host__ __device__ T copysign(T a, T b) { return ::copysign(a, b); }
	template <typename T> __host__ __device__ T sign(T a) { return ::copysign((T)1, a); }
	template <typename T> __host__ __device__ T mix(T a, T b, T c) { return a * ((T)1 - c) + b * c; }
	template <typename T> __host__ __device__ T floor(T a) { return ::floor(a); }
	template <typename T> __host__ __device__ T ceil(T a) { return ::ceil(a); }
	template <typename T> __host__ __device__ T abs(T a) { return ::abs(a); }
	template <typename T> __host__ __device__ T distance(T a, T b) { return ::abs(a - b); }
	template <typename T> __host__ __device__ T sin(T a) { return ::sin(a); }
	template <typename T> __host__ __device__ T asin(T a) { return ::asin(a); }
	template <typename T> __host__ __device__ T cos(T a) { return ::cos(a); }
	template <typename T> __host__ __device__ T acos(T a) { return ::acos(a); }
	template <typename T> __host__ __device__ T tan(T a) { return ::tan(a); }
	template <typename T> __host__ __device__ T atan(T a) { return ::atan(a); }
	template <typename T> __host__ __device__ T sqrt(T a) { return ::sqrt(a); }
	template <typename T> __host__ __device__ T exp(T a) { return ::exp(a); }
	template <typename T> __host__ __device__ T log(T a) { return ::log(a); }
	template <typename T> __host__ __device__ T exp2(T a) { return ::exp2(a); }
	template <typename T> __host__ __device__ T log2(T a) { return ::log2(a); }
	template <typename T> __host__ __device__ T pow(T a, T b) { return ::pow(a, b); }
	template <typename T> __host__ __device__ T isfinite(T a) { return ::isfinite(a); }
}
