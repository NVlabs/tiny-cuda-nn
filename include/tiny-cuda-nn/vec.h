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

/** @file   vec.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Tiny vector / matrix / quaternion implementation.
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace tcnn {

template <class...> struct conjunction : std::true_type {};
template <class B1> struct conjunction<B1> : B1 {};
template <class B1, class... Bn> struct conjunction<B1, Bn...> : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

template <uint32_t N, typename T, typename... Ts>
using enable_if_size_and_type_match_t = std::enable_if_t<sizeof...(Ts) == N && conjunction<std::is_same<Ts, T>...>::value>;

#define TVEC_BODY \
	using underlying_type = T; \
 \
	tvec() = default; \
 \
	TCNN_HOST_DEVICE tvec(T scalar) { \
		TCNN_PRAGMA_UNROLL \
		for (uint32_t i = 0; i < N; ++i) { \
			(*this)[i] = scalar; \
		} \
	} \
 \
	TCNN_HOST_DEVICE static constexpr tvec<T, N> ones() { return tvec<T, N>((T)1); } \
	TCNN_HOST_DEVICE static constexpr tvec<T, N> zero() { return tvec<T, N>((T)0); } \
 \
	TCNN_HOST_DEVICE tvec(const T* coeffs) { \
		TCNN_PRAGMA_UNROLL \
		for (uint32_t i = 0; i < N; ++i) { \
			(*this)[i] = coeffs[i]; \
		} \
	} \
 \
	template <typename U, uint32_t M, size_t A> \
	TCNN_HOST_DEVICE tvec(const tvec<U, M, A>& other) { \
		TCNN_PRAGMA_UNROLL \
		for (uint32_t i = 0; i < N; ++i) { \
			(*this)[i] = i < M ? (T)other[i] : (T)0; \
		} \
	} \
 \
	TCNN_HOST_DEVICE void to_array(T* coeffs) const { \
		TCNN_PRAGMA_UNROLL \
		for (uint32_t i = 0; i < N; ++i) { \
			coeffs[i] = (*this)[i]; \
		} \
	} \
 \
 	TCNN_HOST_DEVICE T* data() { return (T*)this; } \
	TCNN_HOST_DEVICE const T* data() const { return (const T*)this; } \
 \
	TCNN_HOST_DEVICE T& operator[](uint32_t idx) { return ((T*)this)[idx]; } \
	TCNN_HOST_DEVICE const T& operator[](uint32_t idx) const { return ((T*)this)[idx]; } \
	TCNN_HOST_DEVICE T& operator()(uint32_t idx) { return ((T*)this)[idx]; } \
	TCNN_HOST_DEVICE const T& operator()(uint32_t idx) const { return ((T*)this)[idx]; } \
 \
	template <uint32_t OFFSET, uint32_t M> \
	TCNN_HOST_DEVICE tvec<T, M>& slice() { \
		static_assert(OFFSET + M <= N, "Slice must be part of the vector."); \
		return *(tvec<T, M>*)(data() + OFFSET); \
	} \
 \
	template <uint32_t OFFSET, uint32_t M> \
	TCNN_HOST_DEVICE const tvec<T, M>& slice() const { \
		static_assert(OFFSET + M <= N, "Slice must be part of the vector."); \
		return *(tvec<T, M>*)(data() + OFFSET); \
	} \
 \
	TCNN_HOST_DEVICE tvec<T, 2>& xy() { return slice<0, 2>(); } \
	TCNN_HOST_DEVICE const tvec<T, 2>& xy() const { return slice<0, 2>(); } \
	TCNN_HOST_DEVICE tvec<T, 2>& yz() { return slice<1, 2>(); } \
	TCNN_HOST_DEVICE const tvec<T, 2>& yz() const { return slice<1, 2>(); } \
	TCNN_HOST_DEVICE tvec<T, 3>& xyz() { return slice<0, 3>(); } \
	TCNN_HOST_DEVICE const tvec<T, 3>& xyz() const { return slice<0, 3>(); } \
	TCNN_HOST_DEVICE tvec<T, 3>& rgb() { return slice<0, 3>(); } \
	TCNN_HOST_DEVICE const tvec<T, 3>& rgb() const { return slice<0, 3>(); } \
	TCNN_HOST_DEVICE tvec<T, 4>& xyzw() { return slice<0, 4>(); } \
	TCNN_HOST_DEVICE const tvec<T, 4>& rgba() const { return slice<0, 4>(); } \
 \
	TCNN_HOST_DEVICE static constexpr uint32_t size() { return N; }

template <typename T, uint32_t N, size_t ALIGNMENT=sizeof(T)>
struct alignas(ALIGNMENT) tvec {
	TVEC_BODY
	T elems[N];

	template <typename... Ts, typename = enable_if_size_and_type_match_t<N, T, Ts...>>
	TCNN_HOST_DEVICE tvec(Ts... coeffs) : elems{coeffs...} {}
};

template <typename T, size_t ALIGNMENT>
struct alignas(ALIGNMENT) tvec<T, 1, ALIGNMENT> {
	static constexpr uint32_t N = 1;
	TVEC_BODY
	union { T x, r; };
};

template <typename T, size_t ALIGNMENT>
struct alignas(ALIGNMENT) tvec<T, 2, ALIGNMENT> {
	static constexpr uint32_t N = 2;
	TVEC_BODY
	union { T x, r; };
	union { T y, g; };

	TCNN_HOST_DEVICE tvec(T a, T b) : x{a}, y{b} {}
};

template <typename T, size_t ALIGNMENT>
struct alignas(ALIGNMENT) tvec<T, 3, ALIGNMENT> {
	static constexpr uint32_t N = 3;
	TVEC_BODY
	union { T x, r; };
	union { T y, g; };
	union { T z, b; };

	TCNN_HOST_DEVICE tvec(T a, T b, T c) : x{a}, y{b}, z{c} {}
	template <size_t A> TCNN_HOST_DEVICE tvec(const tvec<T, 2, A>& a, T b) : x{a.x}, y{a.y}, z{b} {}
	template <size_t A> TCNN_HOST_DEVICE tvec(T a, const tvec<T, 2, A>& b) : x{a}, y{b.x}, z{b.y} {}
};

template <typename T, size_t ALIGNMENT>
struct alignas(ALIGNMENT) tvec<T, 4, ALIGNMENT> {
	static constexpr uint32_t N = 4;
	TVEC_BODY
	union { T x, r; };
	union { T y, g; };
	union { T z, b; };
	union { T w, a; };

	TCNN_HOST_DEVICE tvec(T a, T b, T c, T d) : x{a}, y{b}, z{c}, w{d} {}
	template <size_t A> TCNN_HOST_DEVICE tvec(const tvec<T, 3, A>& a, T b) : x{a.x}, y{a.y}, z{a.z}, w{b} {}
	template <size_t A1, size_t A2> TCNN_HOST_DEVICE tvec(const tvec<T, 2, A1>& a, const tvec<T, 2, A2>& b) : x{a.x}, y{a.y}, z{b.x}, w{b.y} {}
	template <size_t A> TCNN_HOST_DEVICE tvec(const tvec<T, 2, A>& a, T b, T c) : x{a.x}, y{a.y}, z{b}, w{c} {}
	template <size_t A> TCNN_HOST_DEVICE tvec(T a, const tvec<T, 2, A>& b, T c) : x{a}, y{b.x}, z{b.y}, w{c} {}
	template <size_t A> TCNN_HOST_DEVICE tvec(T a, T b, const tvec<T, 2, A>& c) : x{a}, y{b}, z{c.x}, w{c.y} {}
	template <size_t A> TCNN_HOST_DEVICE tvec(T a, const tvec<T, 3, A>& b) : x{a}, y{b.x}, z{b.y}, w{b.z} {}
};

#undef TVEC_BODY

// Import external cwise functions into ngp namespace to avoid
// name resolution problems related to the vector-values versions defined below.
template <typename T> TCNN_HOST_DEVICE T min(T a, T b) { return std::min(a, b); }
template <typename T> TCNN_HOST_DEVICE T max(T a, T b) { return std::max(a, b); }
template <typename T> TCNN_HOST_DEVICE T clamp(T a, T b, T c) { return a < b ? b : (c < a ? c : a); }
template <typename T> TCNN_HOST_DEVICE T copysign(T a, T b) { return std::copysign(a, b); }
template <typename T> TCNN_HOST_DEVICE T sign(T a) { return std::copysign((T)1, a); }
template <typename T> TCNN_HOST_DEVICE T mix(T a, T b, T c) { return a * ((T)1 - c) + b * c; }
template <typename T> TCNN_HOST_DEVICE T floor(T a) { return std::floor(a); }
template <typename T> TCNN_HOST_DEVICE T ceil(T a) { return std::ceil(a); }
template <typename T> TCNN_HOST_DEVICE T abs(T a) { return std::abs(a); }
template <typename T> TCNN_HOST_DEVICE T distance(T a, T b) { return std::abs(a - b); }
template <typename T> TCNN_HOST_DEVICE T sin(T a) { return std::sin(a); }
template <typename T> TCNN_HOST_DEVICE T asin(T a) { return std::asin(a); }
template <typename T> TCNN_HOST_DEVICE T cos(T a) { return std::cos(a); }
template <typename T> TCNN_HOST_DEVICE T acos(T a) { return std::acos(a); }
template <typename T> TCNN_HOST_DEVICE T tan(T a) { return std::tan(a); }
template <typename T> TCNN_HOST_DEVICE T atan(T a) { return std::atan(a); }
template <typename T> TCNN_HOST_DEVICE T sqrt(T a) { return std::sqrt(a); }
template <typename T> TCNN_HOST_DEVICE T exp(T a) { return std::exp(a); }
template <typename T> TCNN_HOST_DEVICE T log(T a) { return std::log(a); }
template <typename T> TCNN_HOST_DEVICE T exp2(T a) { return std::exp2(a); }
template <typename T> TCNN_HOST_DEVICE T log2(T a) { return std::log2(a); }
template <typename T> TCNN_HOST_DEVICE T pow(T a, T b) { return std::pow(a, b); }
template <typename T> TCNN_HOST_DEVICE T isfinite(T a) {
#if defined(__CUDA_ARCH__)
	return ::isfinite(a);
#else
	return std::isfinite(a);
#endif
}

inline TCNN_HOST_DEVICE float fma(float a, float b, float c) { return fmaf(a, b, c); }
#ifdef __CUDACC__
inline TCNN_DEVICE __half fma(__half a, __half b, __half c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
	return __hfma(a, b, c);
#else
	return fmaf(a, b, c);
#endif
}
#endif

#define TVEC tvec<T, N, A>
#define BVEC tvec<bool, N, A>

#define CWISE_OP(operation, type_result, expr, ...) \
template <typename T, uint32_t N, size_t A> \
TCNN_HOST_DEVICE type_result operation(__VA_ARGS__) { \
	type_result result; \
	TCNN_PRAGMA_UNROLL \
	for (uint32_t i = 0; i < N; ++i) { \
		result[i] = expr; \
	} \
	return result; \
}

CWISE_OP(operator+, TVEC, a[i] + b[i], const TVEC& a, const TVEC& b)
CWISE_OP(operator+, TVEC, a + b[i], T a, const TVEC& b)
CWISE_OP(operator+, TVEC, a[i] + b, const TVEC& a, T b)

CWISE_OP(operator-, TVEC, a[i] - b[i], const TVEC& a, const TVEC& b)
CWISE_OP(operator-, TVEC, a - b[i], T a, const TVEC& b)
CWISE_OP(operator-, TVEC, a[i] - b, const TVEC& a, T b)

CWISE_OP(operator*, TVEC, a[i] * b[i], const TVEC& a, const TVEC& b)
CWISE_OP(operator*, TVEC, a * b[i], T a, const TVEC& b)
CWISE_OP(operator*, TVEC, a[i] * b, const TVEC& a, T b)

CWISE_OP(operator/, TVEC, a[i] / b[i], const TVEC& a, const TVEC& b)
CWISE_OP(operator/, TVEC, a / b[i], T a, const TVEC& b)
CWISE_OP(operator/, TVEC, a[i] / b, const TVEC& a, T b)

CWISE_OP(fma, TVEC, fma(a[i], b[i], c[i]), const TVEC& a, const TVEC& b, const TVEC& c)
CWISE_OP(fma, TVEC, fma(a[i], b[i], c), const TVEC& a, const TVEC& b, T c)
CWISE_OP(fma, TVEC, fma(a[i], b, c[i]), const TVEC& a, T b, const TVEC& c)
CWISE_OP(fma, TVEC, fma(a[i], b, c), const TVEC& a, T b, T c)
CWISE_OP(fma, TVEC, fma(a, b[i], c[i]), T a, const TVEC& b, const TVEC& c)
CWISE_OP(fma, TVEC, fma(a, b[i], c), T a, const TVEC& b, T c)
CWISE_OP(fma, TVEC, fma(a, b, c[i]), T a, T b, const TVEC& c)

CWISE_OP(min, TVEC, min(a[i], b[i]), const TVEC& a, const TVEC& b)
CWISE_OP(min, TVEC, min(a[i], b), const TVEC& a, T b)
CWISE_OP(min, TVEC, min(a, b[i]), T a, const TVEC& b)

CWISE_OP(max, TVEC, max(a[i], b[i]), const TVEC& a, const TVEC& b)
CWISE_OP(max, TVEC, max(a[i], b), const TVEC& a, T b)
CWISE_OP(max, TVEC, max(a, b[i]), T a, const TVEC& b)

CWISE_OP(clamp, TVEC, clamp(a[i], b[i], c[i]), const TVEC& a, const TVEC& b, const TVEC& c)
CWISE_OP(clamp, TVEC, clamp(a[i], b[i], c), const TVEC& a, const TVEC& b, T c)
CWISE_OP(clamp, TVEC, clamp(a[i], b, c[i]), const TVEC& a, T b, const TVEC& c)
CWISE_OP(clamp, TVEC, clamp(a[i], b, c), const TVEC& a, T b, T c)

CWISE_OP(copysign, TVEC, copysign(a[i], b[i]), const TVEC& a, const TVEC& b)
CWISE_OP(copysign, TVEC, copysign(a[i], b), const TVEC& a, T b)
CWISE_OP(copysign, TVEC, copysign(a, b[i]), T a, const TVEC& b)

CWISE_OP(sign, TVEC, sign(a[i]), const TVEC& a)

CWISE_OP(mix, TVEC, a[i] * ((T)1 - c[i]) + b[i] * c[i], const TVEC& a, const TVEC& b, const TVEC& c)
CWISE_OP(mix, TVEC, a[i] * ((T)1 - c) + b[i] * c, const TVEC& a, const TVEC& b, T c)

CWISE_OP(operator-, TVEC, -a[i], const TVEC& a)
CWISE_OP(floor, TVEC, floor(a[i]), const TVEC& a)
CWISE_OP(ceil, TVEC, ceil(a[i]), const TVEC& a)
CWISE_OP(abs, TVEC, abs(a[i]), const TVEC& a)
CWISE_OP(sin, TVEC, sin(a[i]), const TVEC& a)
CWISE_OP(asin, TVEC, asin(a[i]), const TVEC& a)
CWISE_OP(cos, TVEC, cos(a[i]), const TVEC& a)
CWISE_OP(acos, TVEC, acos(a[i]), const TVEC& a)
CWISE_OP(tan, TVEC, tan(a[i]), const TVEC& a)
CWISE_OP(atan, TVEC, atan(a[i]), const TVEC& a)
CWISE_OP(sqrt, TVEC, sqrt(a[i]), const TVEC& a)
CWISE_OP(exp, TVEC, exp(a[i]), const TVEC& a)
CWISE_OP(log, TVEC, log(a[i]), const TVEC& a)
CWISE_OP(exp2, TVEC, exp2(a[i]), const TVEC& a)
CWISE_OP(log2, TVEC, log2(a[i]), const TVEC& a)
CWISE_OP(pow, TVEC, pow(a[i], b), const TVEC& a, T b)
CWISE_OP(pow, TVEC, pow(a[i], b[i]), const TVEC& a, const TVEC& b)

CWISE_OP(isfinite, BVEC, isfinite(a[i]), const TVEC& a)

#if defined(__CUDACC__)
inline TCNN_DEVICE void atomic_add_gmem_float(float* addr, float in) {
#if TCNN_MIN_GPU_ARCH >= 70
	int in_int = *((int*)&in);
	asm ("red.relaxed.gpu.global.add.f32 [%0], %1;" :: "l"(addr), "r"(in_int));
#else
	atomicAdd(addr, in);
#endif
}

template <typename T, uint32_t N, size_t A>
TCNN_DEVICE void atomic_add(T* dst, const tvec<T, N, A>& a) {
	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N; ++i) {
		atomicAdd(dst + i, a[i]);
	}
}

template <uint32_t N, size_t A>
TCNN_DEVICE void atomic_add_gmem(float* dst, const tvec<float, N, A>& a) {
	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N; ++i) {
		atomic_add_gmem_float(dst + i, a[i]);
	}
}

#if TCNN_MIN_GPU_ARCH >= 60
inline TCNN_DEVICE void atomic_add_gmem_h2(half2* addr, half2 in) {
#if TCNN_MIN_GPU_ARCH >= 70
	int in_int = *((int*)&in);
	asm ("red.relaxed.gpu.global.add.noftz.f16x2 [%0], %1;" :: "l"(addr), "r"(in_int));
#else
	atomicAdd(addr, in);
#endif
}

template <uint32_t N, size_t A, typename = std::enable_if_t<N % 2 == 0>>
TCNN_DEVICE void atomic_add(__half* dst, const tvec<__half, N, A>& a) {
	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N; i += 2) {
		atomicAdd((__half2*)(dst + i), __half2(a[i], a[i+1]));
	}
}

template <uint32_t N, size_t A, typename = std::enable_if_t<N % 2 == 0>>
TCNN_DEVICE void atomic_add_gmem(__half* dst, const tvec<__half, N, A>& a) {
	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N; i += 2) {
		atomic_add_gmem_h2((__half2*)(dst + i), __half2(a[i], a[i+1]));
	}
}
#endif
#endif

#undef CWISE_OP

// __half2 specializations for aligned vectors with 2*N fp16 coefficients.
#if defined(__CUDACC__) && TCNN_MIN_GPU_ARCH >= 60

#define HVEC tvec<__half, N, A>
#define HALF_CWISE_OP(operation, type_result, expr, ...) \
template <uint32_t N, size_t A, typename = std::enable_if_t<N % 2 == 0 && A % sizeof(__half2) == 0>> \
TCNN_DEVICE type_result operation(__VA_ARGS__) { \
	type_result result; \
	TCNN_PRAGMA_UNROLL \
	for (uint32_t i = 0; i < N; i += 2) { \
		*(__half2*)&result[i] = expr; \
	} \
	return result; \
}

HALF_CWISE_OP(fma, HVEC, __hfma2(*(__half2*)&a[i], *(__half2*)&b[i], *(__half2*)&c[i]), const HVEC& a, const HVEC& b, const HVEC& c)
HALF_CWISE_OP(fma, HVEC, __hfma2(*(__half2*)&a[i], *(__half2*)&b[i], __half2half2(c)),  const HVEC& a, const HVEC& b, __half c)
HALF_CWISE_OP(fma, HVEC, __hfma2(*(__half2*)&a[i], __half2half2(b),  *(__half2*)&c[i]), const HVEC& a, __half b, const HVEC& c)
HALF_CWISE_OP(fma, HVEC, __hfma2(*(__half2*)&a[i], __half2half2(b),  __half2half2(c)),  const HVEC& a, __half b, __half c)
HALF_CWISE_OP(fma, HVEC, __hfma2(__half2half2(a),  *(__half2*)&b[i], *(__half2*)&c[i]), __half a, const HVEC& b, const HVEC& c)
HALF_CWISE_OP(fma, HVEC, __hfma2(__half2half2(a),  *(__half2*)&b[i], __half2half2(c)),  __half a, const HVEC& b, __half c)
HALF_CWISE_OP(fma, HVEC, __hfma2(__half2half2(a),  __half2half2(b),  *(__half2*)&c[i]), __half a, __half b, const HVEC& c)

HALF_CWISE_OP(operator+, HVEC, __hadd2(*(__half2*)&a[i], *(__half2*)&b[i]), const HVEC& a, const HVEC& b)
HALF_CWISE_OP(operator+, HVEC, __hadd2(__half2half2(a), *(__half2*)&b[i]), __half a, const HVEC& b)
HALF_CWISE_OP(operator+, HVEC, __hadd2(*(__half2*)&a[i], __half2half2(b)), const HVEC& a, __half b)

HALF_CWISE_OP(operator-, HVEC, __hsub2(*(__half2*)&a[i], *(__half2*)&b[i]), const HVEC& a, const HVEC& b)
HALF_CWISE_OP(operator-, HVEC, __hsub2(__half2half2(a), *(__half2*)&b[i]), __half a, const HVEC& b)
HALF_CWISE_OP(operator-, HVEC, __hsub2(*(__half2*)&a[i], __half2half2(b)), const HVEC& a, __half b)

HALF_CWISE_OP(operator*, HVEC, __hmul2(*(__half2*)&a[i], *(__half2*)&b[i]), const HVEC& a, const HVEC& b)
HALF_CWISE_OP(operator*, HVEC, __hmul2(__half2half2(a), *(__half2*)&b[i]), __half a, const HVEC& b)
HALF_CWISE_OP(operator*, HVEC, __hmul2(*(__half2*)&a[i], __half2half2(b)), const HVEC& a, __half b)

HALF_CWISE_OP(operator/, HVEC, __h2div(*(__half2*)&a[i], *(__half2*)&b[i]), const HVEC& a, const HVEC& b)
HALF_CWISE_OP(operator/, HVEC, __h2div(*(__half2*)&a[i], __half2half2(b)), const HVEC& a, __half b)

#endif

#define INPLACE_OP(operation, type_b, expr) \
template <typename T, uint32_t N, size_t A> \
TCNN_HOST_DEVICE TVEC& operation(TVEC& a, type_b b) { \
	TCNN_PRAGMA_UNROLL \
	for (uint32_t i = 0; i < N; ++i) { \
		expr; \
	} \
	return a; \
}

INPLACE_OP(operator*=, const TVEC&, a[i] *= b[i])
INPLACE_OP(operator/=, const TVEC&, a[i] /= b[i])
INPLACE_OP(operator+=, const TVEC&, a[i] += b[i])
INPLACE_OP(operator-=, const TVEC&, a[i] -= b[i])

INPLACE_OP(operator*=, T, a[i] *= b)
INPLACE_OP(operator/=, T, a[i] /= b)

#undef INPLACE_OP

#define REDUCTION_OP(operation, type_result, init, expr, ...) \
template <typename T, uint32_t N, size_t A> \
TCNN_HOST_DEVICE type_result operation(__VA_ARGS__) { \
	type_result result = init; \
	TCNN_PRAGMA_UNROLL \
	for (uint32_t i = 0; i < N; ++i) { \
		expr; \
	} \
	return result; \
}

REDUCTION_OP(dot,     T, (T)0, result += a[i] * b[i], const TVEC& a, const TVEC& b)
REDUCTION_OP(sum,     T, (T)0, result += a[i], const TVEC& a)
REDUCTION_OP(mean,    T, (T)0, result += a[i] / (T)N, const TVEC& a)
REDUCTION_OP(product, T, (T)1, result *= a[i], const TVEC& a)
REDUCTION_OP(min,     T, (T)std::numeric_limits<T>::infinity(), result = min(result, a[i]), const TVEC& a)
REDUCTION_OP(max,     T, (T)-std::numeric_limits<T>::infinity(), result = max(result, a[i]), const TVEC& a)
REDUCTION_OP(length2, T, (T)0, result += a[i] * a[i], const TVEC& a)

REDUCTION_OP(operator==, bool, true,  result &= a[i] == b[i], const TVEC& a, const TVEC& b)
REDUCTION_OP(operator!=, bool, false, result |= a[i] != b[i], const TVEC& a, const TVEC& b)

#undef REDUCTION_OP

#define BOOL_REDUCTION_OP(operation, type_result, init, expr, ...) \
template <uint32_t N, size_t A> \
TCNN_HOST_DEVICE type_result operation(__VA_ARGS__) { \
	type_result result = init; \
	TCNN_PRAGMA_UNROLL \
	for (uint32_t i = 0; i < N; ++i) { \
		expr; \
	} \
	return result; \
}

BOOL_REDUCTION_OP(all, bool, true, result &= a[i], const BVEC& a)
BOOL_REDUCTION_OP(any, bool, false, result |= a[i], const BVEC& a)

#undef BOOL_REDUCTION_OP

template <typename T, uint32_t N, size_t A>
TCNN_HOST_DEVICE T length(const TVEC& a) {
	return std::sqrt(length2(a));
}

template <typename T, uint32_t N, size_t A>
TCNN_HOST_DEVICE T distance(const TVEC& a, const TVEC& b) {
	return length(a - b);
}

template <typename T, uint32_t N, size_t A>
TCNN_HOST_DEVICE TVEC normalize(const TVEC& v) {
	T len = length(v);
	if (len <= (T)0) {
		TVEC result{(T)0};
		result[0] = (T)1;
		return result;
	}
	return v / len;
}

template <typename T, uint32_t N, size_t A>
TCNN_HOST_DEVICE TVEC cross(const TVEC& a, const TVEC& b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x,
	};
}

template <typename T, uint32_t N, size_t A>
TCNN_HOST_DEVICE TVEC faceforward(const TVEC& n, const TVEC& i, const TVEC& nref) {
	return n * -copysign((T)1, dot(i, nref));
}

#undef TVEC
#undef BVEC

#define DEF_NON_TEMPLATED_VECTOR_TYPES(name, T) \
template <uint32_t N> using name = tvec<T, N>; \
template <uint32_t N> using a##name = tvec<T, N, sizeof(T)*N>; \
using name##1 = name<1>; \
using name##2 = name<2>; \
using name##3 = name<3>; \
using name##4 = name<4>;

DEF_NON_TEMPLATED_VECTOR_TYPES(bvec, bool)
DEF_NON_TEMPLATED_VECTOR_TYPES(vec, float)
DEF_NON_TEMPLATED_VECTOR_TYPES(dvec, double)
DEF_NON_TEMPLATED_VECTOR_TYPES(ivec, int)
DEF_NON_TEMPLATED_VECTOR_TYPES(uvec, uint32_t)
DEF_NON_TEMPLATED_VECTOR_TYPES(u16vec, uint16_t)
#if defined(__CUDACC__)
DEF_NON_TEMPLATED_VECTOR_TYPES(hvec, __half)
#endif

#if defined(__CUDACC__)
inline TCNN_HOST_DEVICE float4 to_float4(const vec4& x) { return {x.x, x.y, x.z, x.w}; }
inline TCNN_HOST_DEVICE float3 to_float3(const vec3& x) { return {x.x, x.y, x.z}; }
inline TCNN_HOST_DEVICE float2 to_float2(const vec2& x) { return {x.x, x.y}; }
inline TCNN_HOST_DEVICE vec4 to_vec4(const float4& x) { return {x.x, x.y, x.z, x.w}; }
inline TCNN_HOST_DEVICE vec3 to_vec3(const float3& x) { return {x.x, x.y, x.z}; }
inline TCNN_HOST_DEVICE vec2 to_vec2(const float2& x) { return {x.x, x.y}; }
#endif

template <typename T, uint32_t N, uint32_t M>
struct tmat {
	tmat() = default;

	TCNN_HOST_DEVICE tmat(T scalar) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N; ++i) {
			TCNN_PRAGMA_UNROLL
			for (uint32_t j = 0; j < M; ++j) {
				m[i][j] = i == j ? scalar : (T)0;
			}
		}
	}

	TCNN_HOST_DEVICE static constexpr tmat<T, N, M> identity() { return tmat<T, N, M>((T)1); }
	TCNN_HOST_DEVICE static constexpr tmat<T, N, M> zero() { return tmat<T, N, M>((T)0); }

	template <typename... Ts, typename = enable_if_size_and_type_match_t<N*M, T, Ts...>>
	TCNN_HOST_DEVICE tmat(Ts... coeffs) : d{coeffs...} {}

	TCNN_HOST_DEVICE tmat(const T* coeffs) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N; ++i) {
			TCNN_PRAGMA_UNROLL
			for (uint32_t j = 0; j < M; ++j) {
				m[i][j] = *(coeffs++);
			}
		}
	}

	template <size_t A>
	TCNN_HOST_DEVICE tmat(const tvec<T, M, A>& a) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N; ++i) {
			m[i] = a;
		}
	}

	template <size_t A>
	TCNN_HOST_DEVICE tmat(const tvec<T, M, A>& a, const tvec<T, M, A>& b) {
		static_assert(N == 2, "Matrix must have 2 columns.");
		m[0] = a; m[1] = b;
	}

	template <size_t A>
	TCNN_HOST_DEVICE tmat(const tvec<T, M, A>& a, const tvec<T, M, A>& b, const tvec<T, M, A>& c) {
		static_assert(N == 3, "Matrix must have 3 columns.");
		m[0] = a; m[1] = b; m[2] = c;
	}

	template <size_t A>
	TCNN_HOST_DEVICE tmat(const tvec<T, M, A>& a, const tvec<T, M, A>& b, const tvec<T, M, A>& c, const tvec<T, M, A>& d) {
		static_assert(N == 4, "Matrix must have 4 columns.");
		m[0] = a; m[1] = b; m[2] = c; m[3] = d;
	}

	template <typename U, uint32_t P, uint32_t O>
	TCNN_HOST_DEVICE tmat(const tmat<U, P, O>& other) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N; ++i) {
			TCNN_PRAGMA_UNROLL
			for (uint32_t j = 0; j < M; ++j) {
				m[i][j] = i < P && j < O ? other[i][j] : (i == j ? (T)1 : (T)0);
			}
		}
	}

	template <size_t A>
	TCNN_HOST_DEVICE tvec<T, M, A> operator*(const tvec<T, N, A>& v) const {
		tvec<T, M, A> result((T)0);
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N; ++i) {
			TCNN_PRAGMA_UNROLL
			for (uint32_t j = 0; j < M; ++j) {
				result[j] += m[i][j] * v[i];
			}
		}
		return result;
	}

	template <uint32_t K>
	TCNN_HOST_DEVICE tmat<T, K, M> operator*(const tmat<T, K, N>& other) const {
		tmat<T, K, M> result;
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < K; ++i) {
			result[i] = (*this) * other[i];
		}
		return result;
	}

	TCNN_HOST_DEVICE tvec<T, M>& at(uint32_t idx) { return m[idx]; }
	TCNN_HOST_DEVICE tvec<T, M> at(uint32_t idx) const { return m[idx]; }

	TCNN_HOST_DEVICE tvec<T, M>& operator[](uint32_t idx) { return m[idx]; }
	TCNN_HOST_DEVICE tvec<T, M> operator[](uint32_t idx) const { return m[idx]; }

	TCNN_HOST_DEVICE T* data() { return d; }
	TCNN_HOST_DEVICE const T* data() const { return d; }

	union {
		tvec<T, M> m[N];
		T d[M*N];
	};
};

template <typename T, uint32_t N>
TCNN_HOST_DEVICE tmat<T, N, N>& operator*=(tmat<T, N, N>& m, const tmat<T, N, N>& other) {
	m = m * other;
	return m;
}

template <typename T, uint32_t N, uint32_t M>
TCNN_HOST_DEVICE T frobenius_norm(const tmat<T, N, M>& m) {
	T result = (T)0;
	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N; ++i) {
		result += length2(m[i]);
	}
	return sqrt(result);
}

template <typename T, uint32_t N, uint32_t M>
TCNN_HOST_DEVICE tmat<T, M, N> transpose(const tmat<T, N, M>& m) {
	tmat<T, M, N> result;
	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N; ++i) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t j = 0; j < M; ++j) {
			result[j][i] = m[i][j];
		}
	}
	return result;
}

template <typename T, uint32_t N, uint32_t M>
TCNN_HOST_DEVICE tvec<T, N> row(const tmat<T, N, M>& m, int r) {
	tvec<T, N> result;
	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N; ++i) {
		result[i] = m[i][r];
	}
	return result;
}

template <typename T, uint32_t N, uint32_t M, size_t A>
TCNN_HOST_DEVICE tmat<T, N, M> row(const tmat<T, N, M>& m, int r, const tvec<T, N, A>& v) {
	tmat<T, N, M> result = m;
	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N; ++i) {
		result[i][r] = v[i];
	}
	return result;
}

#define TMAT tmat<T, N, M>

#define CWISE_OP(operation, type_a, type_b, expr) \
template <typename T, uint32_t N, uint32_t M> \
TCNN_HOST_DEVICE TMAT operation(type_a a, type_b b) { \
	TMAT result; \
	TCNN_PRAGMA_UNROLL \
	for (uint32_t i = 0; i < N; ++i) { \
		TCNN_PRAGMA_UNROLL \
		for (uint32_t j = 0; j < M; ++j) { \
			result[i][j] = expr; \
		} \
	} \
	return result; \
}

CWISE_OP(operator+, const TMAT&, const TMAT&, a[i][j] + b[i][j])
CWISE_OP(operator-, const TMAT&, const TMAT&, a[i][j] - b[i][j])

CWISE_OP(operator*, T, const TMAT&, a * b[i][j])
CWISE_OP(operator*, const TMAT&, T, a[i][j] * b)
CWISE_OP(operator/, const TMAT&, T, a[i][j] / b)

#undef CWISE_OP

#define INPLACE_OP(operation, type_b, expr) \
template <typename T, uint32_t N, uint32_t M> \
TCNN_HOST_DEVICE TMAT& operation(TMAT& a, type_b b) { \
	TCNN_PRAGMA_UNROLL \
	for (uint32_t i = 0; i < N; ++i) { \
		TCNN_PRAGMA_UNROLL \
		for (uint32_t j = 0; j < M; ++j) { \
			expr; \
		} \
	} \
	return a; \
}

INPLACE_OP(operator+=, const TMAT&, a[i][j] += b[i][j])
INPLACE_OP(operator-=, const TMAT&, a[i][j] -= b[i][j])

INPLACE_OP(operator*=, T, a[i][j] *= b)
INPLACE_OP(operator/=, T, a[i][j] /= b)

#undef INPLACE_OP

#define REDUCTION_OP(operation, type_result, init, expr, ...) \
template <typename T, uint32_t N, uint32_t M> \
TCNN_HOST_DEVICE type_result operation(__VA_ARGS__) { \
	type_result result = init; \
	TCNN_PRAGMA_UNROLL \
	for (uint32_t i = 0; i < N; ++i) { \
		TCNN_PRAGMA_UNROLL \
		for (uint32_t j = 0; j < M; ++j) { \
			expr; \
		} \
	} \
	return result; \
}

REDUCTION_OP(operator==, bool, true,  result &= a[i][j] == b[i][j], const TMAT& a, const TMAT& b)
REDUCTION_OP(operator!=, bool, false, result |= a[i][j] != b[i][j], const TMAT& a, const TMAT& b)
REDUCTION_OP(isfinite, bool, true, result &= isfinite(a[i][j]), const TMAT& a)

#undef REDUCTION_OP

// The following implementations of determinants, adjoints, inverses, and quaternions
// (and only those) were adapted from glm per the MIT license, which is included below in full.
// ================================================================================
// The MIT License
// --------------------------------------------------------------------------------
// Copyright (c) 2005 - G-Truc Creation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

template <typename T>
TCNN_HOST_DEVICE T determinant(const tmat<T, 2, 2>& m) {
	return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

template <typename T>
TCNN_HOST_DEVICE T determinant(const tmat<T, 3, 3>& m) {
	return
		 m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) +
		-m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2]) +
		 m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2])
		;
}

template <typename T>
TCNN_HOST_DEVICE T determinant(const tmat<T, 4, 4>& m) {
	T s0 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
	T s1 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
	T s2 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
	T s3 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
	T s4 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
	T s5 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

	tvec<T, 4> coeff{
		 (m[1][1] * s0 - m[1][2] * s1 + m[1][3] * s2),
		-(m[1][0] * s0 - m[1][2] * s3 + m[1][3] * s4),
		 (m[1][0] * s1 - m[1][1] * s3 + m[1][3] * s5),
		-(m[1][0] * s2 - m[1][1] * s4 + m[1][2] * s5),
	};

	return
		m[0][0] * coeff[0] + m[0][1] * coeff[1] +
		m[0][2] * coeff[2] + m[0][3] * coeff[3]
		;
}

template <typename T>
TCNN_HOST_DEVICE tmat<T, 2, 2> adjoint(const tmat<T, 2, 2>& m) {
	return {
		 m[1][1], -m[0][1],
		-m[1][0],  m[0][0],
	};
}

template <typename T>
TCNN_HOST_DEVICE tmat<T, 3, 3> adjoint(const tmat<T, 3, 3>& m) {
	const T m00 = determinant(tmat<T, 2, 2>{m[1][1], m[2][1], m[1][2], m[2][2]});
	const T m01 = determinant(tmat<T, 2, 2>{m[0][1], m[2][1], m[0][2], m[2][2]});
	const T m02 = determinant(tmat<T, 2, 2>{m[0][1], m[1][1], m[0][2], m[1][2]});

	const T m10 = determinant(tmat<T, 2, 2>{m[1][0], m[2][0], m[1][2], m[2][2]});
	const T m11 = determinant(tmat<T, 2, 2>{m[0][0], m[2][0], m[0][2], m[2][2]});
	const T m12 = determinant(tmat<T, 2, 2>{m[0][0], m[1][0], m[0][2], m[1][2]});

	const T m20 = determinant(tmat<T, 2, 2>{m[1][0], m[2][0], m[1][1], m[2][1]});
	const T m21 = determinant(tmat<T, 2, 2>{m[0][0], m[2][0], m[0][1], m[2][1]});
	const T m22 = determinant(tmat<T, 2, 2>{m[0][0], m[1][0], m[0][1], m[1][1]});

	return {
		 m00, -m01,  m02,
		-m10,  m11, -m12,
		 m20, -m21,  m22,
	};
}

template <typename T>
TCNN_HOST_DEVICE tmat<T, 4, 4> adjoint(const tmat<T, 4, 4>& m) {
	const T m00 = determinant(tmat<T, 3, 3>{m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], m[3][3]});
	const T m01 = determinant(tmat<T, 3, 3>{m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], m[3][3]});
	const T m02 = determinant(tmat<T, 3, 3>{m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], m[3][3]});
	const T m03 = determinant(tmat<T, 3, 3>{m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], m[3][2]});

	const T m10 = determinant(tmat<T, 3, 3>{m[0][1], m[0][2], m[0][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], m[3][3]});
	const T m11 = determinant(tmat<T, 3, 3>{m[0][0], m[0][2], m[0][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], m[3][3]});
	const T m12 = determinant(tmat<T, 3, 3>{m[0][0], m[0][1], m[0][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], m[3][3]});
	const T m13 = determinant(tmat<T, 3, 3>{m[0][0], m[0][1], m[0][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], m[3][2]});

	const T m20 = determinant(tmat<T, 3, 3>{m[0][1], m[0][2], m[0][3], m[1][1], m[1][2], m[1][3], m[3][1], m[3][2], m[3][3]});
	const T m21 = determinant(tmat<T, 3, 3>{m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3], m[3][0], m[3][2], m[3][3]});
	const T m22 = determinant(tmat<T, 3, 3>{m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[3][0], m[3][1], m[3][3]});
	const T m23 = determinant(tmat<T, 3, 3>{m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[3][0], m[3][1], m[3][2]});

	const T m30 = determinant(tmat<T, 3, 3>{m[0][1], m[0][2], m[0][3], m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3]});
	const T m31 = determinant(tmat<T, 3, 3>{m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3]});
	const T m32 = determinant(tmat<T, 3, 3>{m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3]});
	const T m33 = determinant(tmat<T, 3, 3>{m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]});

	return {
		 m00, -m10,  m20, -m30,
		-m01,  m11, -m21,  m31,
		 m02, -m12,  m22, -m32,
		-m03,  m13, -m23,  m33,
	};
}

template <typename T, uint32_t N>
TCNN_HOST_DEVICE tmat<T, N, N> inverse(const tmat<T, N, N>& m) {
	return adjoint(m) / determinant(m);
}

template <typename T, size_t A>
TCNN_HOST_DEVICE tmat<T, 3, 3> rotmat(T angle, const tvec<T, 3, A>& axis) {
	T s, c;
	sincos(angle, &s, &c);
	T oc = (T)1 - c;

	return {
		oc * axis.x * axis.x + c,          oc * axis.x * axis.y + axis.z * s, oc * axis.z * axis.x - axis.y * s,
		oc * axis.x * axis.y - axis.z * s, oc * axis.y * axis.y + c,          oc * axis.y * axis.z + axis.x * s,
		oc * axis.z * axis.x + axis.y * s, oc * axis.y * axis.z - axis.x * s, oc * axis.z * axis.z + c,
	};
}

template <typename T, size_t A>
TCNN_HOST_DEVICE tmat<T, 3, 3> rotmat(const tvec<T, 3, A>& v) {
	T angle = length(v);
	if (angle == (T)0) {
		return tmat<T, 3, 3>::identity();
	}

	return rotmat(angle, v / angle);
}

template <typename T, uint32_t N>
TCNN_HOST_DEVICE tmat<T, N, N> mat_sqrt(const tmat<T, N, N>& m, T eps = (T)1e-10f) {
	tmat<T, N, N> X = m, Y = tmat<T, N, N>::identity();
	for (uint32_t i = 0; i < 32; ++i) {
		if (frobenius_norm(X * X - m) < eps) {
			return X;
		}

		tmat<T, N, N> iX = inverse(X);
		X = (T)0.5f * (X + inverse(Y));
		Y = (T)0.5f * (Y + iX);
	}

	return X;
}

template <typename T, uint32_t N>
TCNN_HOST_DEVICE tmat<T, N, N> mat_log_hawkins(const tmat<T, N, N>& m, T eps = (T)1e-10f) {
	tmat<T, N, N> A = m - tmat<T, N, N>::identity(), Z = A, X = A;
	for (uint32_t i = 2; i < 32; ++i) {
		if (frobenius_norm(Z) < eps) {
			return X;
		}

		Z = Z * A;
		X += ((T)1 / (T)i) * Z;
	}

	return X;
}

template <typename T, uint32_t N>
TCNN_HOST_DEVICE tmat<T, N, N> mat_exp_pade(const tmat<T, N, N>& m) {
	// Pade approximation with scaling; same as Matlab.
	// Pseudocode translated from Hawkins and Grimm [2007]
	tmat<T, N, N> mX = tmat<T, N, N>::identity(), mD = tmat<T, N, N>::identity(), mN = tmat<T, N, N>::identity();
	T c = (T)1;
	constexpr uint32_t q = 6; // Matlab's default when using this algorithm

	T s = -(T)1;
	for (uint32_t k = 1; k <= q; ++k) {
		c = c * (q - k + 1) / (k * (2 * q - k + 1));
		mX = m * mX;
		auto cmX = c * mX;
		mN = mN + cmX;
		mD = mD + s * cmX;
		s = -s;
	}

	return inverse(mD) * mN;
}

template <typename T, uint32_t N>
TCNN_HOST_DEVICE tmat<T, N, N> mat_log(const tmat<T, N, N>& m) {
	tmat<T, N, N> result(m);

	uint32_t j = 0;
	for (; j < 32; ++j) {
		if (frobenius_norm(result - tmat<T, N, N>::identity()) < (T)1e-5f) {
			break;
		}

		result = mat_sqrt(result);
	}

	result = mat_log_hawkins(result);
	return (T)scalbnf(1.0f, j) * result;
}

template <typename T, uint32_t N>
TCNN_HOST_DEVICE tmat<T, N, N> mat_exp(const tmat<T, N, N>& m) {
	uint32_t N_SQUARING = max(0, 1 + (int)floor(log2(frobenius_norm(m))));

	tmat<T, N, N> result = (T)scalbnf(1.0f, -N_SQUARING) * m;
	result = mat_exp_pade(result);

	for (uint32_t i = 0; i < N_SQUARING; ++i) {
		result *= result;
	}

	return result;
}

template <typename T>
TCNN_HOST_DEVICE tmat<T, 3, 3> orthogonalize(const tmat<T, 3, 3>& m) {
	// Iteration to bring an almost orthogonal matrix nearer to its closest
	// orthogonal matrix. This can be run multiple times until convergence
	// is measured or, alternatively, once per frame on something like a
	// camera matrix to ensure it does not degenerate over time.
	return (T)1.5f * m - (T)0.5f * (m * transpose(m) * m);
}

template <typename T>
TCNN_HOST_DEVICE tmat<T, 3, 3> so3_log(const tmat<T, 3, 3>& m) {
	T tr = clamp(m[0][0] + m[1][1] + m[2][2], -(T)1 + std::numeric_limits<T>::epsilon(), (T)1);
	T radians = acosf((tr - (T)1) / (T)2);
	return radians / sqrt(((T)1 + tr) * ((T)3 - tr)) * (m - transpose(m));
}

template <typename T>
TCNN_HOST_DEVICE tmat<T, 3, 3> so3_exp(const tmat<T, 3, 3>& m) {
	tvec<T, 3> axis = {-m[2][1], m[2][0], -m[1][0]};
	T radians_sq = length2(axis);
	if (radians_sq == (T)0) {
		return tmat<T, 3, 3>::identity();
	}

	T radians = sqrt(radians_sq);
	return tmat<T, 3, 3>::identity() + (sin(radians) / radians) * m + (((T)1 - cos(radians)) / radians_sq) * (m * m);
}

template <typename T>
TCNN_HOST_DEVICE tmat<T, 4, 3> se3_log(const tmat<T, 4, 3>& m) {
	auto omega = so3_log(tmat<T, 3, 3>(m));
	tvec<T, 3> axis = {-omega[2][1], omega[2][0], -omega[1][0]};
	T radians_sq = length2(axis);
	auto inv_trans = tmat<T, 3, 3>::identity();
	if (radians_sq > (T)0) {
		T radians = sqrt(radians_sq);
		inv_trans += -(T)0.5 * omega + (((T)1 - (T)0.5 * radians * cos((T)0.5 * radians) / sin((T)0.5 * radians)) / radians_sq) * (omega * omega);
	}

	return {omega[0], omega[1], omega[2], inv_trans * m[3]};
}

template <typename T>
TCNN_HOST_DEVICE tmat<T, 4, 3> se3_exp(const tmat<T, 4, 3>& m) {
	tmat<T, 3, 3> omega = m;
	tvec<T, 3> axis = {-omega[2][1], omega[2][0], -omega[1][0]};
	T radians_sq = length2(axis);
	auto trans = tmat<T, 3, 3>::identity();
	if (radians_sq > (T)0) {
		T radians = sqrt(radians_sq);
		trans += (((T)1 - cos(radians)) / radians_sq) * omega + ((radians - sin(radians)) / (radians * radians_sq)) * (omega * omega);
	}

	auto rot = so3_exp(omega);
	return {rot[0], rot[1], rot[2], trans * m[3]};
}

template <typename T>
TCNN_HOST_DEVICE tmat<T, 4, 4> se3_log(const tmat<T, 4, 4>& m) {
	auto result = tmat<T, 4, 4>(se3_log(tmat<T, 4, 3>(m)));
	result[3][3] = (T)0;
	return result;
}

template <typename T>
TCNN_HOST_DEVICE tmat<T, 4, 4> se3_exp(const tmat<T, 4, 4>& m) {
	return tmat<T, 4, 4>(se3_exp(tmat<T, 4, 3>(m)));
}

#define DEF_NON_TEMPLATED_MATRIX_TYPES(name, T) \
template <uint32_t N, uint32_t M> \
using name = tmat<T, N, M>; \
using name##4x4 = name<4, 4>; \
using name##4x3 = name<4, 3>; \
using name##4x2 = name<4, 2>; \
using name##3x4 = name<3, 4>; \
using name##3x3 = name<3, 3>; \
using name##3x2 = name<3, 2>; \
using name##2x4 = name<2, 4>; \
using name##2x3 = name<2, 3>; \
using name##2x2 = name<2, 2>; \
using name##4 = name##4x4; \
using name##3 = name##3x3; \
using name##2 = name##2x2;

DEF_NON_TEMPLATED_MATRIX_TYPES(mat, float)
DEF_NON_TEMPLATED_MATRIX_TYPES(dmat, double)
#if defined(__CUDACC__)
DEF_NON_TEMPLATED_MATRIX_TYPES(hmat, __half)
#endif

template <typename T>
struct tquat {
	tquat() = default;
	TCNN_HOST_DEVICE tquat(T w, T x, T y, T z) : w{w}, x{x}, y{y}, z{z} {}
	TCNN_HOST_DEVICE tquat(const tmat<T, 3, 3>& m) {
		// Code adapted from https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
		T tr = m[0][0] + m[1][1] + m[2][2];

		if (tr > (T)0) {
			T S = sqrt(tr + (T)1) * (T)2; // S=4*qw
			w = (T)0.25 * S;
			x = (m[1][2] - m[2][1]) / S;
			y = (m[2][0] - m[0][2]) / S;
			z = (m[0][1] - m[1][0]) / S;
		} else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
			T S = sqrt((T)1 + m[0][0] - m[1][1] - m[2][2]) * (T)2; // S=4*x
			w = (m[1][2] - m[2][1]) / S;
			x = (T)0.25 * S;
			y = (m[1][0] + m[0][1]) / S;
			z = (m[2][0] + m[0][2]) / S;
		} else if (m[1][1] > m[2][2]) {
			T S = sqrt((T)1 + m[1][1] - m[0][0] - m[2][2]) * (T)2; // S=4*y
			w = (m[2][0] - m[0][2]) / S;
			x = (m[1][0] + m[0][1]) / S;
			y = (T)0.25 * S;
			z = (m[2][1] + m[1][2]) / S;
		} else {
			T S = sqrt((T)1 + m[2][2] - m[0][0] - m[1][1]) * (T)2; // S=4*z
			w = (m[0][1] - m[1][0]) / S;
			x = (m[2][0] + m[0][2]) / S;
			y = (m[2][1] + m[1][2]) / S;
			z = (T)0.25 * S;
		}
	}

	T w, x, y, z;
};

template <typename T> TCNN_HOST_DEVICE tquat<T> operator-(const tquat<T>& a) { return {-a.w, -a.x, -a.y, -a.z}; }
template <typename T> TCNN_HOST_DEVICE tquat<T> operator+(const tquat<T>& a, const tquat<T>& b) { return {a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z}; }
template <typename T> TCNN_HOST_DEVICE tquat<T> operator-(const tquat<T>& a, const tquat<T>& b) { return {a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z}; }
template <typename T> TCNN_HOST_DEVICE tquat<T> operator*(T a, const tquat<T>& b) { return {a * b.w, a * b.x, a * b.y, a * b.z}; }
template <typename T> TCNN_HOST_DEVICE tquat<T> operator*(const tquat<T>& a, T b) { return {a.w * b, a.x * b, a.y * b, a.z * b}; }
template <typename T> TCNN_HOST_DEVICE tquat<T> operator/(const tquat<T>& a, T b) { return {a.w / b, a.x / b, a.y / b, a.z / b}; }

template <typename T> TCNN_HOST_DEVICE T dot(const tquat<T>& a, const tquat<T>& b) { return (a.w * b.w + a.x * b.x) + (a.y * b.y + a.z * b.z); }
template <typename T> TCNN_HOST_DEVICE T length2(const tquat<T>& a) { return dot(a, a); }
template <typename T> TCNN_HOST_DEVICE T length(const tquat<T>& a) { return sqrt(length2(a)); }

template <typename T> TCNN_HOST_DEVICE tquat<T> mix(const tquat<T>& a, const tquat<T>& b, T t) { return a * ((T)1 - t) + b * t; }

template <typename T>
TCNN_HOST_DEVICE tquat<T> normalize(const tquat<T>& a) {
	T len = length(a);
	if (len <= (T)0) {
		return {(T)1, (T)0, (T)0, (T)0};
	}
	return a / len;
}

template <typename T>
TCNN_HOST_DEVICE tquat<T> cross(const tquat<T>& a, const tquat<T>& b) {
	return {
		a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
		a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
		a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
		a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x
	};
}

template <typename T>
TCNN_HOST_DEVICE tquat<T> slerp(const tquat<T>& x, const tquat<T>& y, T t) {
	tquat<T> z = y;

	T cos_theta = dot(x, y);

	// If cos_theta < 0, the interpolation will take the long way around the sphere.
	// To fix this, one quat must be negated.
	if (cos_theta < (T)0) {
		z = -y;
		cos_theta = -cos_theta;
	}

	// Perform a linear interpolation when cos_theta is close to 1 to avoid side effect of sin(angle) becoming a zero denominator
	if (cos_theta > (T)1 - std::numeric_limits<T>::epsilon()) {
		return mix(x, z, t);
	} else {
		// Essential Mathematics, page 467
		T angle = acos(cos_theta);
		return (sin(((T)1 - t) * angle) * x + sin(t * angle) * z) / sin(angle);
	}
}

template <typename T>
TCNN_HOST_DEVICE T angle(const tquat<T>& x) {
	return acos(clamp(x.w, (T)-1, (T)1)) * (T)2;
}

template <typename T>
TCNN_HOST_DEVICE tvec<T, 3> axis(const tquat<T>& x) {
	const T tmp1 = (T)1 - x.w * x.w;
	if (tmp1 <= (T)0) {
		return {(T)0, (T)0, (T)1};
	}

	const T tmp2 = (T)1 / sqrt(tmp1);
	return {x.x * tmp2, x.y * tmp2, x.z * tmp2};
}

template <typename T>
TCNN_HOST_DEVICE tmat<T, 3, 3> to_mat3(const tquat<T>& q) {
	T qxx = q.x * q.x, qyy = q.y * q.y, qzz = q.z * q.z;
	T qxz = q.x * q.z, qxy = q.x * q.y, qyz = q.y * q.z;
	T qwx = q.w * q.x, qwy = q.w * q.y, qwz = q.w * q.z;

	return {
		(T)1 - (T)2 * (qyy +  qzz), (T)2 * (qxy + qwz), (T)2 * (qxz - qwy),
		(T)2 * (qxy - qwz), (T)1 - (T)2 * (qxx +  qzz), (T)2 * (qyz + qwx),
		(T)2 * (qxz + qwy), (T)2 * (qyz - qwx), (T)1 - (T)2 * (qxx +  qyy),
	};
}

template <typename T>
TCNN_HOST_DEVICE tmat<T, 3, 3> slerp(const tmat<T, 3, 3>& a, const tmat<T, 3, 3>& b, float t) {
	return to_mat3(normalize(slerp(normalize(tquat<T>(a)), normalize(tquat<T>(b)), t)));
}

template <typename T>
TCNN_HOST_DEVICE tvec<T, 3> rotvec(const tmat<T, 3, 3>& mat) {
	tquat<T> tmp = mat;
	return axis(tmp) * angle(tmp);
}

using quat = tquat<float>;

}
