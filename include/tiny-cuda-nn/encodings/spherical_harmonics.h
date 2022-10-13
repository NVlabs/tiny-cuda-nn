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

/** @file   spherical_harmonics.h
 *  @author Alex Evans and Thomas Müller, NVIDIA
 *  @brief  Implementation of a spherical harmonics based frequency encoding.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

template <typename T, typename INPUT_T>
__global__ void kernel_sh(
	const uint32_t num_elements,
	const uint32_t degree,
	const uint32_t num_to_pad,
	MatrixView<const INPUT_T> data_in,
	MatrixView<T> data_out
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_out.advance_cols(i);

	TCNN_PRAGMA_UNROLL
	for (uint32_t j = 0; j < num_to_pad; ++j) {
		data_out(j) = (T)1.0f;
	}

	data_out.advance_rows(num_to_pad);

	INPUT_T x = data_in(0, i) * INPUT_T(2) - INPUT_T(1);
	INPUT_T y = data_in(1, i) * INPUT_T(2) - INPUT_T(1);
	INPUT_T z = data_in(2, i) * INPUT_T(2) - INPUT_T(1);

	// Let compiler figure out how to sequence/reorder these calculations w.r.t. branches
	INPUT_T xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z;
	INPUT_T x4=x2*x2, y4=y2*y2, z4=z2*z2;
	INPUT_T x6=x4*x2, y6=y4*y2, z6=z4*z2;

	// SH polynomials generated using scripts/gen_sh.py based on the recurrence relations in appendix A1 of https://www.ppsloan.org/publications/StupidSH36.pdf

	auto write_sh = [&]() {
		data_out(0) = INPUT_T(0.28209479177387814);                          // 1/(2*sqrt(pi))
		if (degree <= 1) { return; }
		data_out(1) = -INPUT_T(0.48860251190291987)*y;                               // -sqrt(3)*y/(2*sqrt(pi))
		data_out(2) = INPUT_T(0.48860251190291987)*z;                                // sqrt(3)*z/(2*sqrt(pi))
		data_out(3) = -INPUT_T(0.48860251190291987)*x;                               // -sqrt(3)*x/(2*sqrt(pi))
		if (degree <= 2) { return; }
		data_out(4) = INPUT_T(1.0925484305920792)*xy;                                // sqrt(15)*xy/(2*sqrt(pi))
		data_out(5) = -INPUT_T(1.0925484305920792)*yz;                               // -sqrt(15)*yz/(2*sqrt(pi))
		data_out(6) = INPUT_T(0.94617469575755997)*z2 - INPUT_T(0.31539156525251999);                         // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
		data_out(7) = -INPUT_T(1.0925484305920792)*xz;                               // -sqrt(15)*xz/(2*sqrt(pi))
		data_out(8) = INPUT_T(0.54627421529603959)*x2 - INPUT_T(0.54627421529603959)*y2;                              // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
		if (degree <= 3) { return; }
		data_out(9) = INPUT_T(0.59004358992664352)*y*(-INPUT_T(3.0)*x2 + y2);                         // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
		data_out(10) = INPUT_T(2.8906114426405538)*xy*z;                             // sqrt(105)*xy*z/(2*sqrt(pi))
		data_out(11) = INPUT_T(0.45704579946446572)*y*(INPUT_T(1.0) - INPUT_T(5.0)*z2);                                // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
		data_out(12) = INPUT_T(0.3731763325901154)*z*(INPUT_T(5.0)*z2 - INPUT_T(3.0));                         // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
		data_out(13) = INPUT_T(0.45704579946446572)*x*(INPUT_T(1.0) - INPUT_T(5.0)*z2);                                // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
		data_out(14) = INPUT_T(1.4453057213202769)*z*(x2 - y2);                              // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
		data_out(15) = INPUT_T(0.59004358992664352)*x*(-x2 + INPUT_T(3.0)*y2);                                // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
		if (degree <= 4) { return; }
		data_out(16) = INPUT_T(2.5033429417967046)*xy*(x2 - y2);                             // 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
		data_out(17) = INPUT_T(1.7701307697799304)*yz*(-INPUT_T(3.0)*x2 + y2);                                // 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
		data_out(18) = INPUT_T(0.94617469575756008)*xy*(INPUT_T(7.0)*z2 - INPUT_T(1.0));                               // 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
		data_out(19) = INPUT_T(0.66904654355728921)*yz*(INPUT_T(3.0) - INPUT_T(7.0)*z2);                               // 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
		data_out(20) = -INPUT_T(3.1735664074561294)*z2 + INPUT_T(3.7024941420321507)*z4 + INPUT_T(0.31735664074561293);                                // 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
		data_out(21) = INPUT_T(0.66904654355728921)*xz*(INPUT_T(3.0) - INPUT_T(7.0)*z2);                               // 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
		data_out(22) = INPUT_T(0.47308734787878004)*(x2 - y2)*(INPUT_T(7.0)*z2 - INPUT_T(1.0));                                // 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
		data_out(23) = INPUT_T(1.7701307697799304)*xz*(-x2 + INPUT_T(3.0)*y2);                                // 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
		data_out(24) = -INPUT_T(3.7550144126950569)*x2*y2 + INPUT_T(0.62583573544917614)*x4 + INPUT_T(0.62583573544917614)*y4;                         // 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
		if (degree <= 5) { return; }
		data_out(25) = INPUT_T(0.65638205684017015)*y*(INPUT_T(10.0)*x2*y2 - INPUT_T(5.0)*x4 - y4);                            // 3*sqrt(154)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		data_out(26) = INPUT_T(8.3026492595241645)*xy*z*(x2 - y2);                           // 3*sqrt(385)*xy*z*(x2 - y2)/(4*sqrt(pi))
		data_out(27) = -INPUT_T(0.48923829943525038)*y*(INPUT_T(3.0)*x2 - y2)*(INPUT_T(9.0)*z2 - INPUT_T(1.0));                         // -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
		data_out(28) = INPUT_T(4.7935367849733241)*xy*z*(INPUT_T(3.0)*z2 - INPUT_T(1.0));                              // sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
		data_out(29) = INPUT_T(0.45294665119569694)*y*(INPUT_T(14.0)*z2 - INPUT_T(21.0)*z4 - INPUT_T(1.0));                             // sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
		data_out(30) = INPUT_T(0.1169503224534236)*z*(-INPUT_T(70.0)*z2 + INPUT_T(63.0)*z4 + INPUT_T(15.0));                            // sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
		data_out(31) = INPUT_T(0.45294665119569694)*x*(INPUT_T(14.0)*z2 - INPUT_T(21.0)*z4 - INPUT_T(1.0));                             // sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
		data_out(32) = INPUT_T(2.3967683924866621)*z*(x2 - y2)*(INPUT_T(3.0)*z2 - INPUT_T(1.0));                               // sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
		data_out(33) = -INPUT_T(0.48923829943525038)*x*(x2 - INPUT_T(3.0)*y2)*(INPUT_T(9.0)*z2 - INPUT_T(1.0));                         // -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
		data_out(34) = INPUT_T(2.0756623148810411)*z*(-INPUT_T(6.0)*x2*y2 + x4 + y4);                         // 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
		data_out(35) = INPUT_T(0.65638205684017015)*x*(INPUT_T(10.0)*x2*y2 - x4 - INPUT_T(5.0)*y4);                            // 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
		if (degree <= 6) { return; }
		data_out(36) = INPUT_T(1.3663682103838286)*xy*(-INPUT_T(10.0)*x2*y2 + INPUT_T(3.0)*x4 + INPUT_T(3.0)*y4);                               // sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
		data_out(37) = INPUT_T(2.3666191622317521)*yz*(INPUT_T(10.0)*x2*y2 - INPUT_T(5.0)*x4 - y4);                            // 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		data_out(38) = INPUT_T(2.0182596029148963)*xy*(x2 - y2)*(INPUT_T(11.0)*z2 - INPUT_T(1.0));                             // 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
		data_out(39) = -INPUT_T(0.92120525951492349)*yz*(INPUT_T(3.0)*x2 - y2)*(INPUT_T(11.0)*z2 - INPUT_T(3.0));                               // -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
		data_out(40) = INPUT_T(0.92120525951492349)*xy*(-INPUT_T(18.0)*z2 + INPUT_T(33.0)*z4 + INPUT_T(1.0));                           // sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
		data_out(41) = INPUT_T(0.58262136251873131)*yz*(INPUT_T(30.0)*z2 - INPUT_T(33.0)*z4 - INPUT_T(5.0));                            // sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
		data_out(42) = INPUT_T(6.6747662381009842)*z2 - INPUT_T(20.024298714302954)*z4 + INPUT_T(14.684485723822165)*z6 - INPUT_T(0.31784601133814211);                         // sqrt(13)*(105*z2 - 315*z4 + 231*z6 - 5)/(32*sqrt(pi))
		data_out(43) = INPUT_T(0.58262136251873131)*xz*(INPUT_T(30.0)*z2 - INPUT_T(33.0)*z4 - INPUT_T(5.0));                            // sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
		data_out(44) = INPUT_T(0.46060262975746175)*(x2 - y2)*(INPUT_T(11.0)*z2*(INPUT_T(3.0)*z2 - INPUT_T(1.0)) - INPUT_T(7.0)*z2 + INPUT_T(1.0));                               // sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2 + 1)/(64*sqrt(pi))
		data_out(45) = -INPUT_T(0.92120525951492349)*xz*(x2 - INPUT_T(3.0)*y2)*(INPUT_T(11.0)*z2 - INPUT_T(3.0));                               // -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
		data_out(46) = INPUT_T(0.50456490072872406)*(INPUT_T(11.0)*z2 - INPUT_T(1.0))*(-INPUT_T(6.0)*x2*y2 + x4 + y4);                          // 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
		data_out(47) = INPUT_T(2.3666191622317521)*xz*(INPUT_T(10.0)*x2*y2 - x4 - INPUT_T(5.0)*y4);                            // 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
		data_out(48) = INPUT_T(10.247761577878714)*x2*y4 - INPUT_T(10.247761577878714)*x4*y2 + INPUT_T(0.6831841051919143)*x6 - INPUT_T(0.6831841051919143)*y6;                         // sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
		if (degree <= 7) { return; }
		data_out(49) = INPUT_T(0.70716273252459627)*y*(-INPUT_T(21.0)*x2*y4 + INPUT_T(35.0)*x4*y2 - INPUT_T(7.0)*x6 + y6);                              // 3*sqrt(715)*y*(-21*x2*y4 + 35*x4*y2 - 7*x6 + y6)/(64*sqrt(pi))
		data_out(50) = INPUT_T(5.2919213236038001)*xy*z*(-INPUT_T(10.0)*x2*y2 + INPUT_T(3.0)*x4 + INPUT_T(3.0)*y4);                             // 3*sqrt(10010)*xy*z*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
		data_out(51) = -INPUT_T(0.51891557872026028)*y*(INPUT_T(13.0)*z2 - INPUT_T(1.0))*(-INPUT_T(10.0)*x2*y2 + INPUT_T(5.0)*x4 + y4);                          // -3*sqrt(385)*y*(13*z2 - 1)*(-10*x2*y2 + 5*x4 + y4)/(64*sqrt(pi))
		data_out(52) = INPUT_T(4.1513246297620823)*xy*z*(x2 - y2)*(INPUT_T(13.0)*z2 - INPUT_T(3.0));                           // 3*sqrt(385)*xy*z*(x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
		data_out(53) = -INPUT_T(0.15645893386229404)*y*(INPUT_T(3.0)*x2 - y2)*(INPUT_T(13.0)*z2*(INPUT_T(11.0)*z2 - INPUT_T(3.0)) - INPUT_T(27.0)*z2 + INPUT_T(3.0));                              // -3*sqrt(35)*y*(3*x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
		data_out(54) = INPUT_T(0.44253269244498261)*xy*z*(-INPUT_T(110.0)*z2 + INPUT_T(143.0)*z4 + INPUT_T(15.0));                              // 3*sqrt(70)*xy*z*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
		data_out(55) = INPUT_T(0.090331607582517306)*y*(-INPUT_T(135.0)*z2 + INPUT_T(495.0)*z4 - INPUT_T(429.0)*z6 + INPUT_T(5.0));                              // sqrt(105)*y*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
		data_out(56) = INPUT_T(0.068284276912004949)*z*(INPUT_T(315.0)*z2 - INPUT_T(693.0)*z4 + INPUT_T(429.0)*z6 - INPUT_T(35.0));                              // sqrt(15)*z*(315*z2 - 693*z4 + 429*z6 - 35)/(32*sqrt(pi))
		data_out(57) = INPUT_T(0.090331607582517306)*x*(-INPUT_T(135.0)*z2 + INPUT_T(495.0)*z4 - INPUT_T(429.0)*z6 + INPUT_T(5.0));                              // sqrt(105)*x*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
		data_out(58) = INPUT_T(0.07375544874083044)*z*(x2 - y2)*(INPUT_T(143.0)*z2*(INPUT_T(3.0)*z2 - INPUT_T(1.0)) - INPUT_T(187.0)*z2 + INPUT_T(45.0));                         // sqrt(70)*z*(x2 - y2)*(143*z2*(3*z2 - 1) - 187*z2 + 45)/(64*sqrt(pi))
		data_out(59) = -INPUT_T(0.15645893386229404)*x*(x2 - INPUT_T(3.0)*y2)*(INPUT_T(13.0)*z2*(INPUT_T(11.0)*z2 - INPUT_T(3.0)) - INPUT_T(27.0)*z2 + INPUT_T(3.0));                              // -3*sqrt(35)*x*(x2 - 3*y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
		data_out(60) = INPUT_T(1.0378311574405206)*z*(INPUT_T(13.0)*z2 - INPUT_T(3.0))*(-INPUT_T(6.0)*x2*y2 + x4 + y4);                         // 3*sqrt(385)*z*(13*z2 - 3)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
		data_out(61) = -INPUT_T(0.51891557872026028)*x*(INPUT_T(13.0)*z2 - INPUT_T(1.0))*(-INPUT_T(10.0)*x2*y2 + x4 + INPUT_T(5.0)*y4);                          // -3*sqrt(385)*x*(13*z2 - 1)*(-10*x2*y2 + x4 + 5*y4)/(64*sqrt(pi))
		data_out(62) = INPUT_T(2.6459606618019)*z*(INPUT_T(15.0)*x2*y4 - INPUT_T(15.0)*x4*y2 + x6 - y6);                               // 3*sqrt(10010)*z*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
		data_out(63) = INPUT_T(0.70716273252459627)*x*(-INPUT_T(35.0)*x2*y4 + INPUT_T(21.0)*x4*y2 - x6 + INPUT_T(7.0)*y6);                              // 3*sqrt(715)*x*(-35*x2*y4 + 21*x4*y2 - x6 + 7*y6)/(64*sqrt(pi))
	};

	write_sh();
}

template <typename T, typename INPUT_T>
__global__ void kernel_sh_backward(
	const uint32_t num_elements,
	const uint32_t degree,
	const uint32_t num_to_pad,
	MatrixView<const T> dL_dy,
	MatrixView<const INPUT_T> data_in,
	MatrixView<INPUT_T> dL_dx
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	dL_dy.advance(num_to_pad, i);

	INPUT_T x = data_in(0, i) * INPUT_T(2) - INPUT_T(1);
	INPUT_T y = data_in(1, i) * INPUT_T(2) - INPUT_T(1);
	INPUT_T z = data_in(2, i) * INPUT_T(2) - INPUT_T(1);

	// Let compiler figure out how to sequence/reorder these calculations w.r.t. branches
	INPUT_T xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z;
	INPUT_T x4=x2*x2, y4=y2*y2, z4=z2*z2;
	INPUT_T x6=x4*x2, y6=y4*y2, z6=z4*z2;

	INPUT_T dx = 0.0f;
	INPUT_T dy = 0.0f;
	INPUT_T dz = 0.0f;

	auto compute_grad = [&]() {
		// dx += (INPUT_T)dL_dy(0) * (0);                                // 0
		// dy += (INPUT_T)dL_dy(0) * (0);                                // 0
		// dz += (INPUT_T)dL_dy(0) * (0);                                // 0
		if (degree <= 1) { return; }
		// dx += (INPUT_T)dL_dy(1) * (0);                                // 0
		dy += (INPUT_T)dL_dy(1) * (INPUT_T(-0.48860251190291992));                                // -sqrt(3)/(2*sqrt(pi))
		// dz += (INPUT_T)dL_dy(1) * (0);                                // 0
		// dx += (INPUT_T)dL_dy(2) * (0);                                // 0
		// dy += (INPUT_T)dL_dy(2) * (0);                                // 0
		dz += (INPUT_T)dL_dy(2) * (INPUT_T(0.48860251190291992));                         // sqrt(3)/(2*sqrt(pi))
		dx += (INPUT_T)dL_dy(3) * (INPUT_T(-0.48860251190291992));                                // -sqrt(3)/(2*sqrt(pi))
		// dy += (INPUT_T)dL_dy(3) * (0);                                // 0
		// dz += (INPUT_T)dL_dy(3) * (0);                                // 0
		if (degree <= 2) { return; }
		dx += (INPUT_T)dL_dy(4) * (INPUT_T(1.0925484305920792)*y);                                // sqrt(15)*y/(2*sqrt(pi))
		dy += (INPUT_T)dL_dy(4) * (INPUT_T(1.0925484305920792)*x);                                // sqrt(15)*x/(2*sqrt(pi))
		// dz += (INPUT_T)dL_dy(4) * (0);                                // 0
		// dx += (INPUT_T)dL_dy(5) * (0);                                // 0
		dy += (INPUT_T)dL_dy(5) * (INPUT_T(-1.0925484305920792)*z);                               // -sqrt(15)*z/(2*sqrt(pi))
		dz += (INPUT_T)dL_dy(5) * (INPUT_T(-1.0925484305920792)*y);                               // -sqrt(15)*y/(2*sqrt(pi))
		// dx += (INPUT_T)dL_dy(6) * (0);                                // 0
		// dy += (INPUT_T)dL_dy(6) * (0);                                // 0
		dz += (INPUT_T)dL_dy(6) * (INPUT_T(1.8923493915151202)*z);                                // 3*sqrt(5)*z/(2*sqrt(pi))
		dx += (INPUT_T)dL_dy(7) * (INPUT_T(-1.0925484305920792)*z);                               // -sqrt(15)*z/(2*sqrt(pi))
		// dy += (INPUT_T)dL_dy(7) * (0);                                // 0
		dz += (INPUT_T)dL_dy(7) * (INPUT_T(-1.0925484305920792)*x);                               // -sqrt(15)*x/(2*sqrt(pi))
		dx += (INPUT_T)dL_dy(8) * (INPUT_T(1.0925484305920792)*x);                                // sqrt(15)*x/(2*sqrt(pi))
		dy += (INPUT_T)dL_dy(8) * (INPUT_T(-1.0925484305920792)*y);                               // -sqrt(15)*y/(2*sqrt(pi))
		// dz += (INPUT_T)dL_dy(8) * (0);                                // 0
		if (degree <= 3) { return; }
		dx += (INPUT_T)dL_dy(9) * (INPUT_T(-3.5402615395598609)*xy);                              // -3*sqrt(70)*xy/(4*sqrt(pi))
		dy += (INPUT_T)dL_dy(9) * (INPUT_T(-1.7701307697799304)*x2 + INPUT_T(1.7701307697799304)*y2);                              // 3*sqrt(70)*(-x2 + y2)/(8*sqrt(pi))
		// dz += (INPUT_T)dL_dy(9) * (0);                                // 0
		dx += (INPUT_T)dL_dy(10) * (INPUT_T(2.8906114426405538)*yz);                              // sqrt(105)*yz/(2*sqrt(pi))
		dy += (INPUT_T)dL_dy(10) * (INPUT_T(2.8906114426405538)*xz);                              // sqrt(105)*xz/(2*sqrt(pi))
		dz += (INPUT_T)dL_dy(10) * (INPUT_T(2.8906114426405538)*xy);                              // sqrt(105)*xy/(2*sqrt(pi))
		// dx += (INPUT_T)dL_dy(11) * (0);                               // 0
		dy += (INPUT_T)dL_dy(11) * (INPUT_T(0.45704579946446572) - INPUT_T(2.2852289973223288)*z2);                                // sqrt(42)*(1 - 5*z2)/(8*sqrt(pi))
		dz += (INPUT_T)dL_dy(11) * (INPUT_T(-4.5704579946446566)*yz);                             // -5*sqrt(42)*yz/(4*sqrt(pi))
		// dx += (INPUT_T)dL_dy(12) * (0);                               // 0
		// dy += (INPUT_T)dL_dy(12) * (0);                               // 0
		dz += (INPUT_T)dL_dy(12) * (INPUT_T(5.597644988851731)*z2 - INPUT_T(1.1195289977703462));                          // 3*sqrt(7)*(5*z2 - 1)/(4*sqrt(pi))
		dx += (INPUT_T)dL_dy(13) * (INPUT_T(0.45704579946446572) - INPUT_T(2.2852289973223288)*z2);                                // sqrt(42)*(1 - 5*z2)/(8*sqrt(pi))
		// dy += (INPUT_T)dL_dy(13) * (0);                               // 0
		dz += (INPUT_T)dL_dy(13) * (INPUT_T(-4.5704579946446566)*xz);                             // -5*sqrt(42)*xz/(4*sqrt(pi))
		dx += (INPUT_T)dL_dy(14) * (INPUT_T(2.8906114426405538)*xz);                              // sqrt(105)*xz/(2*sqrt(pi))
		dy += (INPUT_T)dL_dy(14) * (INPUT_T(-2.8906114426405538)*yz);                             // -sqrt(105)*yz/(2*sqrt(pi))
		dz += (INPUT_T)dL_dy(14) * (INPUT_T(1.4453057213202769)*x2 - INPUT_T(1.4453057213202769)*y2);                              // sqrt(105)*(x2 - y2)/(4*sqrt(pi))
		dx += (INPUT_T)dL_dy(15) * (INPUT_T(-1.7701307697799304)*x2 + INPUT_T(1.7701307697799304)*y2);                             // 3*sqrt(70)*(-x2 + y2)/(8*sqrt(pi))
		dy += (INPUT_T)dL_dy(15) * (INPUT_T(3.5402615395598609)*xy);                              // 3*sqrt(70)*xy/(4*sqrt(pi))
		// dz += (INPUT_T)dL_dy(15) * (0);                               // 0
		if (degree <= 4) { return; }
		dx += (INPUT_T)dL_dy(16) * (INPUT_T(2.5033429417967046)*y*(INPUT_T(3.0)*x2 - y2));                         // 3*sqrt(35)*y*(3*x2 - y2)/(4*sqrt(pi))
		dy += (INPUT_T)dL_dy(16) * (INPUT_T(2.5033429417967046)*x*(x2 - INPUT_T(3.0)*y2));                         // 3*sqrt(35)*x*(x2 - 3*y2)/(4*sqrt(pi))
		// dz += (INPUT_T)dL_dy(16) * (0);                               // 0
		dx += (INPUT_T)dL_dy(17) * (INPUT_T(-10.620784618679583)*xy*z);                           // -9*sqrt(70)*xy*z/(4*sqrt(pi))
		dy += (INPUT_T)dL_dy(17) * (INPUT_T(5.3103923093397913)*z*(-x2 + y2));                            // 9*sqrt(70)*z*(-x2 + y2)/(8*sqrt(pi))
		dz += (INPUT_T)dL_dy(17) * (INPUT_T(1.7701307697799304)*y*(INPUT_T(-3.0)*x2 + y2));                                // 3*sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
		dx += (INPUT_T)dL_dy(18) * (INPUT_T(0.94617469575756008)*y*(INPUT_T(7.0)*z2 - INPUT_T(1.0)));                               // 3*sqrt(5)*y*(7*z2 - 1)/(4*sqrt(pi))
		dy += (INPUT_T)dL_dy(18) * (INPUT_T(0.94617469575756008)*x*(INPUT_T(7.0)*z2 - INPUT_T(1.0)));                               // 3*sqrt(5)*x*(7*z2 - 1)/(4*sqrt(pi))
		dz += (INPUT_T)dL_dy(18) * (INPUT_T(13.246445740605839)*xy*z);                            // 21*sqrt(5)*xy*z/(2*sqrt(pi))
		// dx += (INPUT_T)dL_dy(19) * (0);                               // 0
		dy += (INPUT_T)dL_dy(19) * (INPUT_T(0.66904654355728921)*z*(INPUT_T(3.0) - INPUT_T(7.0)*z2));                               // 3*sqrt(10)*z*(3 - 7*z2)/(8*sqrt(pi))
		dz += (INPUT_T)dL_dy(19) * (INPUT_T(2.0071396306718676)*y*(INPUT_T(1.0) - INPUT_T(7.0)*z2));                                // 9*sqrt(10)*y*(1 - 7*z2)/(8*sqrt(pi))
		// dx += (INPUT_T)dL_dy(20) * (0);                               // 0
		// dy += (INPUT_T)dL_dy(20) * (0);                               // 0
		dz += (INPUT_T)dL_dy(20) * (INPUT_T(14.809976568128603)*z*z*z - INPUT_T(6.3471328149122579)*z);                                // (105*z**3 - 45*z)/(4*sqrt(pi))
		dx += (INPUT_T)dL_dy(21) * (INPUT_T(0.66904654355728921)*z*(INPUT_T(3.0) - INPUT_T(7.0)*z2));                               // 3*sqrt(10)*z*(3 - 7*z2)/(8*sqrt(pi))
		// dy += (INPUT_T)dL_dy(21) * (0);                               // 0
		dz += (INPUT_T)dL_dy(21) * (INPUT_T(2.0071396306718676)*x*(INPUT_T(1.0) - INPUT_T(7.0)*z2));                                // 9*sqrt(10)*x*(1 - 7*z2)/(8*sqrt(pi))
		dx += (INPUT_T)dL_dy(22) * (INPUT_T(0.94617469575756008)*x*(INPUT_T(7.0)*z2 - INPUT_T(1.0)));                               // 3*sqrt(5)*x*(7*z2 - 1)/(4*sqrt(pi))
		dy += (INPUT_T)dL_dy(22) * (INPUT_T(0.94617469575756008)*y*(INPUT_T(1.0) - INPUT_T(7.0)*z2));                               // 3*sqrt(5)*y*(1 - 7*z2)/(4*sqrt(pi))
		dz += (INPUT_T)dL_dy(22) * (INPUT_T(6.6232228703029197)*z*(x2 - y2));                             // 21*sqrt(5)*z*(x2 - y2)/(4*sqrt(pi))
		dx += (INPUT_T)dL_dy(23) * (INPUT_T(5.3103923093397913)*z*(-x2 + y2));                            // 9*sqrt(70)*z*(-x2 + y2)/(8*sqrt(pi))
		dy += (INPUT_T)dL_dy(23) * (INPUT_T(10.620784618679583)*xy*z);                            // 9*sqrt(70)*xy*z/(4*sqrt(pi))
		dz += (INPUT_T)dL_dy(23) * (INPUT_T(1.7701307697799304)*x*(-x2 + INPUT_T(3.0)*y2));                                // 3*sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
		dx += (INPUT_T)dL_dy(24) * (INPUT_T(2.5033429417967046)*x*(x2 - INPUT_T(3.0)*y2));                         // 3*sqrt(35)*x*(x2 - 3*y2)/(4*sqrt(pi))
		dy += (INPUT_T)dL_dy(24) * (INPUT_T(2.5033429417967046)*y*(INPUT_T(-3.0)*x2 + y2));                                // 3*sqrt(35)*y*(-3*x2 + y2)/(4*sqrt(pi))
		// dz += (INPUT_T)dL_dy(24) * (0);                               // 0
		if (degree <= 5) { return; }
		dx += (INPUT_T)dL_dy(25) * (INPUT_T(13.127641136803401)*xy*(-x2 + y2));                           // 15*sqrt(154)*xy*(-x2 + y2)/(8*sqrt(pi))
		dy += (INPUT_T)dL_dy(25) * (INPUT_T(19.6914617052051)*x2*y2 - INPUT_T(3.2819102842008503)*x4 - INPUT_T(3.2819102842008503)*y4);                             // 15*sqrt(154)*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
		// dz += (INPUT_T)dL_dy(25) * (0);                               // 0
		dx += (INPUT_T)dL_dy(26) * (INPUT_T(8.3026492595241645)*yz*(INPUT_T(3.0)*x2 - y2));                                // 3*sqrt(385)*yz*(3*x2 - y2)/(4*sqrt(pi))
		dy += (INPUT_T)dL_dy(26) * (INPUT_T(8.3026492595241645)*xz*(x2 - INPUT_T(3.0)*y2));                                // 3*sqrt(385)*xz*(x2 - 3*y2)/(4*sqrt(pi))
		dz += (INPUT_T)dL_dy(26) * (INPUT_T(8.3026492595241645)*xy*(x2 - y2));                            // 3*sqrt(385)*xy*(x2 - y2)/(4*sqrt(pi))
		dx += (INPUT_T)dL_dy(27) * (INPUT_T(2.9354297966115022)*xy*(INPUT_T(1.0) - INPUT_T(9.0)*z2));                               // 3*sqrt(770)*xy*(1 - 9*z2)/(16*sqrt(pi))
		dy += (INPUT_T)dL_dy(27) * (INPUT_T(-1.4677148983057511)*(x2 - y2)*(INPUT_T(9.0)*z2 - INPUT_T(1.0)));                               // -3*sqrt(770)*(x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
		dz += (INPUT_T)dL_dy(27) * (INPUT_T(8.8062893898345074)*yz*(INPUT_T(-3.0)*x2 + y2));                               // 9*sqrt(770)*yz*(-3*x2 + y2)/(16*sqrt(pi))
		dx += (INPUT_T)dL_dy(28) * (INPUT_T(4.7935367849733241)*yz*(INPUT_T(3.0)*z2 - INPUT_T(1.0)));                               // sqrt(1155)*yz*(3*z2 - 1)/(4*sqrt(pi))
		dy += (INPUT_T)dL_dy(28) * (INPUT_T(4.7935367849733241)*xz*(INPUT_T(3.0)*z2 - INPUT_T(1.0)));                               // sqrt(1155)*xz*(3*z2 - 1)/(4*sqrt(pi))
		dz += (INPUT_T)dL_dy(28) * (INPUT_T(4.7935367849733241)*xy*(INPUT_T(9.0)*z2 - INPUT_T(1.0)));                               // sqrt(1155)*xy*(9*z2 - 1)/(4*sqrt(pi))
		// dx += (INPUT_T)dL_dy(29) * (0);                               // 0
		dy += (INPUT_T)dL_dy(29) * (INPUT_T(6.3412531167397574)*z2 - INPUT_T(9.5118796751096362)*z4 - INPUT_T(0.45294665119569694));                                // sqrt(165)*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
		dz += (INPUT_T)dL_dy(29) * (INPUT_T(12.682506233479513)*yz*(INPUT_T(1.0) - INPUT_T(3.0)*z2));                               // 7*sqrt(165)*yz*(1 - 3*z2)/(4*sqrt(pi))
		// dx += (INPUT_T)dL_dy(30) * (0);                               // 0
		// dy += (INPUT_T)dL_dy(30) * (0);                               // 0
		dz += (INPUT_T)dL_dy(30) * (INPUT_T(-24.559567715218954)*z2 + INPUT_T(36.839351572828434)*z4 + INPUT_T(1.754254836801354));                         // 15*sqrt(11)*(-14*z2 + 21*z4 + 1)/(16*sqrt(pi))
		dx += (INPUT_T)dL_dy(31) * (INPUT_T(6.3412531167397574)*z2 - INPUT_T(9.5118796751096362)*z4 - INPUT_T(0.45294665119569694));                                // sqrt(165)*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
		// dy += (INPUT_T)dL_dy(31) * (0);                               // 0
		dz += (INPUT_T)dL_dy(31) * (INPUT_T(12.682506233479513)*xz*(INPUT_T(1.0) - INPUT_T(3.0)*z2));                               // 7*sqrt(165)*xz*(1 - 3*z2)/(4*sqrt(pi))
		dx += (INPUT_T)dL_dy(32) * (INPUT_T(4.7935367849733241)*xz*(INPUT_T(3.0)*z2 - INPUT_T(1.0)));                               // sqrt(1155)*xz*(3*z2 - 1)/(4*sqrt(pi))
		dy += (INPUT_T)dL_dy(32) * (INPUT_T(4.7935367849733241)*yz*(INPUT_T(1.0) - INPUT_T(3.0)*z2));                               // sqrt(1155)*yz*(1 - 3*z2)/(4*sqrt(pi))
		dz += (INPUT_T)dL_dy(32) * (INPUT_T(2.3967683924866621)*(x2 - y2)*(INPUT_T(9.0)*z2 - INPUT_T(1.0)));                                // sqrt(1155)*(x2 - y2)*(9*z2 - 1)/(8*sqrt(pi))
		dx += (INPUT_T)dL_dy(33) * (INPUT_T(-13.209434084751759)*x2*z2 + INPUT_T(1.4677148983057511)*x2 + INPUT_T(13.209434084751759)*y2*z2 - INPUT_T(1.4677148983057511)*y2);                               // 3*sqrt(770)*(-9*x2*z2 + x2 + 9*y2*z2 - y2)/(32*sqrt(pi))
		dy += (INPUT_T)dL_dy(33) * (INPUT_T(2.9354297966115022)*xy*(INPUT_T(9.0)*z2 - INPUT_T(1.0)));                               // 3*sqrt(770)*xy*(9*z2 - 1)/(16*sqrt(pi))
		dz += (INPUT_T)dL_dy(33) * (INPUT_T(8.8062893898345074)*xz*(-x2 + INPUT_T(3.0)*y2));                               // 9*sqrt(770)*xz*(-x2 + 3*y2)/(16*sqrt(pi))
		dx += (INPUT_T)dL_dy(34) * (INPUT_T(8.3026492595241645)*xz*(x2 - INPUT_T(3.0)*y2));                                // 3*sqrt(385)*xz*(x2 - 3*y2)/(4*sqrt(pi))
		dy += (INPUT_T)dL_dy(34) * (INPUT_T(8.3026492595241645)*yz*(INPUT_T(-3.0)*x2 + y2));                               // 3*sqrt(385)*yz*(-3*x2 + y2)/(4*sqrt(pi))
		dz += (INPUT_T)dL_dy(34) * (INPUT_T(-12.453973889286246)*x2*y2 + INPUT_T(2.0756623148810411)*x4 + INPUT_T(2.0756623148810411)*y4);                          // 3*sqrt(385)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
		dx += (INPUT_T)dL_dy(35) * (INPUT_T(19.6914617052051)*x2*y2 - INPUT_T(3.2819102842008503)*x4 - INPUT_T(3.2819102842008503)*y4);                             // 15*sqrt(154)*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
		dy += (INPUT_T)dL_dy(35) * (INPUT_T(13.127641136803401)*xy*(x2 - y2));                            // 15*sqrt(154)*xy*(x2 - y2)/(8*sqrt(pi))
		// dz += (INPUT_T)dL_dy(35) * (0);                               // 0
		if (degree <= 6) { return; }
		dx += (INPUT_T)dL_dy(36) * (INPUT_T(4.0991046311514854)*y*(INPUT_T(-10.0)*x2*y2 + INPUT_T(5.0)*x4 + y4));                           // 3*sqrt(6006)*y*(-10*x2*y2 + 5*x4 + y4)/(32*sqrt(pi))
		dy += (INPUT_T)dL_dy(36) * (INPUT_T(4.0991046311514854)*x*(INPUT_T(-10.0)*x2*y2 + x4 + INPUT_T(5.0)*y4));                           // 3*sqrt(6006)*x*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
		// dz += (INPUT_T)dL_dy(36) * (0);                               // 0
		dx += (INPUT_T)dL_dy(37) * (INPUT_T(47.332383244635047)*xy*z*(-x2 + y2));                         // 15*sqrt(2002)*xy*z*(-x2 + y2)/(8*sqrt(pi))
		dy += (INPUT_T)dL_dy(37) * (INPUT_T(11.833095811158762)*z*(INPUT_T(6.0)*x2*y2 - x4 - y4));                         // 15*sqrt(2002)*z*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
		dz += (INPUT_T)dL_dy(37) * (INPUT_T(2.3666191622317521)*y*(INPUT_T(10.0)*x2*y2 - INPUT_T(5.0)*x4 - y4));                            // 3*sqrt(2002)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		dx += (INPUT_T)dL_dy(38) * (INPUT_T(2.0182596029148963)*y*(INPUT_T(3.0)*x2 - y2)*(INPUT_T(11.0)*z2 - INPUT_T(1.0)));                         // 3*sqrt(91)*y*(3*x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
		dy += (INPUT_T)dL_dy(38) * (INPUT_T(2.0182596029148963)*x*(x2 - INPUT_T(3.0)*y2)*(INPUT_T(11.0)*z2 - INPUT_T(1.0)));                         // 3*sqrt(91)*x*(x2 - 3*y2)*(11*z2 - 1)/(8*sqrt(pi))
		dz += (INPUT_T)dL_dy(38) * (INPUT_T(44.401711264127719)*xy*z*(x2 - y2));                          // 33*sqrt(91)*xy*z*(x2 - y2)/(4*sqrt(pi))
		dx += (INPUT_T)dL_dy(39) * (INPUT_T(5.5272315570895412)*xy*z*(INPUT_T(3.0) - INPUT_T(11.0)*z2));                            // 3*sqrt(2730)*xy*z*(3 - 11*z2)/(16*sqrt(pi))
		dy += (INPUT_T)dL_dy(39) * (INPUT_T(-2.7636157785447706)*z*(x2 - y2)*(INPUT_T(11.0)*z2 - INPUT_T(3.0)));                            // -3*sqrt(2730)*z*(x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
		dz += (INPUT_T)dL_dy(39) * (INPUT_T(-2.7636157785447706)*y*(INPUT_T(3.0)*x2 - y2)*(INPUT_T(11.0)*z2 - INPUT_T(1.0)));                                // -3*sqrt(2730)*y*(3*x2 - y2)*(11*z2 - 1)/(32*sqrt(pi))
		dx += (INPUT_T)dL_dy(40) * (INPUT_T(0.92120525951492349)*y*(INPUT_T(-18.0)*z2 + INPUT_T(33.0)*z4 + INPUT_T(1.0)));                           // sqrt(2730)*y*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
		dy += (INPUT_T)dL_dy(40) * (INPUT_T(0.92120525951492349)*x*(INPUT_T(-18.0)*z2 + INPUT_T(33.0)*z4 + INPUT_T(1.0)));                           // sqrt(2730)*x*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
		dz += (INPUT_T)dL_dy(40) * (INPUT_T(11.054463114179082)*xy*z*(INPUT_T(11.0)*z2 - INPUT_T(3.0)));                            // 3*sqrt(2730)*xy*z*(11*z2 - 3)/(8*sqrt(pi))
		// dx += (INPUT_T)dL_dy(41) * (0);                               // 0
		dy += (INPUT_T)dL_dy(41) * (INPUT_T(0.58262136251873131)*z*(INPUT_T(30.0)*z2 - INPUT_T(33.0)*z4 - INPUT_T(5.0)));                            // sqrt(273)*z*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
		dz += (INPUT_T)dL_dy(41) * (INPUT_T(2.9131068125936568)*y*(INPUT_T(18.0)*z2 - INPUT_T(33.0)*z4 - INPUT_T(1.0)));                             // 5*sqrt(273)*y*(18*z2 - 33*z4 - 1)/(16*sqrt(pi))
		// dx += (INPUT_T)dL_dy(42) * (0);                               // 0
		// dy += (INPUT_T)dL_dy(42) * (0);                               // 0
		dz += (INPUT_T)dL_dy(42) * (INPUT_T(2.6699064952403937)*z*(INPUT_T(-30.0)*z2 + INPUT_T(33.0)*z4 + INPUT_T(5.0)));                            // 21*sqrt(13)*z*(-30*z2 + 33*z4 + 5)/(16*sqrt(pi))
		dx += (INPUT_T)dL_dy(43) * (INPUT_T(0.58262136251873131)*z*(INPUT_T(30.0)*z2 - INPUT_T(33.0)*z4 - INPUT_T(5.0)));                            // sqrt(273)*z*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
		// dy += (INPUT_T)dL_dy(43) * (0);                               // 0
		dz += (INPUT_T)dL_dy(43) * (INPUT_T(2.9131068125936568)*x*(INPUT_T(18.0)*z2 - INPUT_T(33.0)*z4 - INPUT_T(1.0)));                             // 5*sqrt(273)*x*(18*z2 - 33*z4 - 1)/(16*sqrt(pi))
		dx += (INPUT_T)dL_dy(44) * (INPUT_T(0.92120525951492349)*x*(INPUT_T(-18.0)*z2 + INPUT_T(33.0)*z4 + INPUT_T(1.0)));                           // sqrt(2730)*x*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
		dy += (INPUT_T)dL_dy(44) * (INPUT_T(0.92120525951492349)*y*(INPUT_T(18.0)*z2 - INPUT_T(33.0)*z4 - INPUT_T(1.0)));                            // sqrt(2730)*y*(18*z2 - 33*z4 - 1)/(32*sqrt(pi))
		dz += (INPUT_T)dL_dy(44) * (INPUT_T(5.5272315570895412)*z*(x2 - y2)*(INPUT_T(11.0)*z2 - INPUT_T(3.0)));                             // 3*sqrt(2730)*z*(x2 - y2)*(11*z2 - 3)/(16*sqrt(pi))
		dx += (INPUT_T)dL_dy(45) * (INPUT_T(-2.7636157785447706)*z*(x2 - y2)*(INPUT_T(11.0)*z2 - INPUT_T(3.0)));                            // -3*sqrt(2730)*z*(x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
		dy += (INPUT_T)dL_dy(45) * (INPUT_T(5.5272315570895412)*xy*z*(INPUT_T(11.0)*z2 - INPUT_T(3.0)));                            // 3*sqrt(2730)*xy*z*(11*z2 - 3)/(16*sqrt(pi))
		dz += (INPUT_T)dL_dy(45) * (INPUT_T(-2.7636157785447706)*x*(x2 - INPUT_T(3.0)*y2)*(INPUT_T(11.0)*z2 - INPUT_T(1.0)));                                // -3*sqrt(2730)*x*(x2 - 3*y2)*(11*z2 - 1)/(32*sqrt(pi))
		dx += (INPUT_T)dL_dy(46) * (INPUT_T(2.0182596029148963)*x*(x2 - INPUT_T(3.0)*y2)*(INPUT_T(11.0)*z2 - INPUT_T(1.0)));                         // 3*sqrt(91)*x*(x2 - 3*y2)*(11*z2 - 1)/(8*sqrt(pi))
		dy += (INPUT_T)dL_dy(46) * (INPUT_T(-2.0182596029148963)*y*(INPUT_T(3.0)*x2 - y2)*(INPUT_T(11.0)*z2 - INPUT_T(1.0)));                                // -3*sqrt(91)*y*(3*x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
		dz += (INPUT_T)dL_dy(46) * (INPUT_T(11.10042781603193)*z*(INPUT_T(-6.0)*x2*y2 + x4 + y4));                         // 33*sqrt(91)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
		dx += (INPUT_T)dL_dy(47) * (INPUT_T(11.833095811158762)*z*(INPUT_T(6.0)*x2*y2 - x4 - y4));                         // 15*sqrt(2002)*z*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
		dy += (INPUT_T)dL_dy(47) * (INPUT_T(47.332383244635047)*xy*z*(x2 - y2));                          // 15*sqrt(2002)*xy*z*(x2 - y2)/(8*sqrt(pi))
		dz += (INPUT_T)dL_dy(47) * (INPUT_T(2.3666191622317521)*x*(INPUT_T(10.0)*x2*y2 - x4 - INPUT_T(5.0)*y4));                            // 3*sqrt(2002)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
		dx += (INPUT_T)dL_dy(48) * (INPUT_T(4.0991046311514854)*x*(INPUT_T(-10.0)*x2*y2 + x4 + INPUT_T(5.0)*y4));                           // 3*sqrt(6006)*x*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
		dy += (INPUT_T)dL_dy(48) * (INPUT_T(4.0991046311514854)*y*(INPUT_T(10.0)*x2*y2 - INPUT_T(5.0)*x4 - y4));                            // 3*sqrt(6006)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		// dz += (INPUT_T)dL_dy(48) * (0);                               // 0
		if (degree <= 7) { return; }
		dx += (INPUT_T)dL_dy(49) * (INPUT_T(9.9002782553443485)*xy*(INPUT_T(10.0)*x2*y2 - INPUT_T(3.0)*x4 - INPUT_T(3.0)*y4));                               // 21*sqrt(715)*xy*(10*x2*y2 - 3*x4 - 3*y4)/(32*sqrt(pi))
		dy += (INPUT_T)dL_dy(49) * (INPUT_T(-74.252086915082614)*x2*y4 + INPUT_T(74.252086915082614)*x4*y2 - INPUT_T(4.9501391276721742)*x6 + INPUT_T(4.9501391276721742)*y6);                               // 21*sqrt(715)*(-15*x2*y4 + 15*x4*y2 - x6 + y6)/(64*sqrt(pi))
		// dz += (INPUT_T)dL_dy(49) * (0);                               // 0
		dx += (INPUT_T)dL_dy(50) * (INPUT_T(15.875763970811402)*yz*(INPUT_T(-10.0)*x2*y2 + INPUT_T(5.0)*x4 + y4));                          // 9*sqrt(10010)*yz*(-10*x2*y2 + 5*x4 + y4)/(32*sqrt(pi))
		dy += (INPUT_T)dL_dy(50) * (INPUT_T(15.875763970811402)*xz*(INPUT_T(-10.0)*x2*y2 + x4 + INPUT_T(5.0)*y4));                          // 9*sqrt(10010)*xz*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
		dz += (INPUT_T)dL_dy(50) * (INPUT_T(5.2919213236038001)*xy*(INPUT_T(-10.0)*x2*y2 + INPUT_T(3.0)*x4 + INPUT_T(3.0)*y4));                              // 3*sqrt(10010)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
		dx += (INPUT_T)dL_dy(51) * (INPUT_T(-10.378311574405206)*xy*(x2 - y2)*(INPUT_T(13.0)*z2 - INPUT_T(1.0)));                           // -15*sqrt(385)*xy*(x2 - y2)*(13*z2 - 1)/(16*sqrt(pi))
		dy += (INPUT_T)dL_dy(51) * (INPUT_T(0.51891557872026028)*(INPUT_T(13.0)*z2 - INPUT_T(1.0))*(INPUT_T(10.0)*x2*y2 - INPUT_T(5.0)*x4 + INPUT_T(4.0)*y2*(INPUT_T(5.0)*x2 - y2) - y4));                              // 3*sqrt(385)*(13*z2 - 1)*(10*x2*y2 - 5*x4 + 4*y2*(5*x2 - y2) - y4)/(64*sqrt(pi))
		dz += (INPUT_T)dL_dy(51) * (INPUT_T(13.491805046726766)*yz*(INPUT_T(10.0)*x2*y2 - INPUT_T(5.0)*x4 - y4));                           // 39*sqrt(385)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		dx += (INPUT_T)dL_dy(52) * (INPUT_T(4.1513246297620823)*yz*(INPUT_T(3.0)*x2 - y2)*(INPUT_T(13.0)*z2 - INPUT_T(3.0)));                                // 3*sqrt(385)*yz*(3*x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
		dy += (INPUT_T)dL_dy(52) * (INPUT_T(4.1513246297620823)*xz*(x2 - INPUT_T(3.0)*y2)*(INPUT_T(13.0)*z2 - INPUT_T(3.0)));                                // 3*sqrt(385)*xz*(x2 - 3*y2)*(13*z2 - 3)/(8*sqrt(pi))
		dz += (INPUT_T)dL_dy(52) * (INPUT_T(12.453973889286248)*xy*(x2 - y2)*(INPUT_T(13.0)*z2 - INPUT_T(1.0)));                            // 9*sqrt(385)*xy*(x2 - y2)*(13*z2 - 1)/(8*sqrt(pi))
		dx += (INPUT_T)dL_dy(53) * (INPUT_T(0.93875360317376422)*xy*(INPUT_T(66.0)*z2 - INPUT_T(143.0)*z4 - INPUT_T(3.0)));                          // 9*sqrt(35)*xy*(66*z2 - 143*z4 - 3)/(32*sqrt(pi))
		dy += (INPUT_T)dL_dy(53) * (INPUT_T(-0.46937680158688211)*(x2 - y2)*(INPUT_T(13.0)*z2*(INPUT_T(11.0)*z2 - INPUT_T(3.0)) - INPUT_T(27.0)*z2 + INPUT_T(3.0)));                           // -9*sqrt(35)*(x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
		dz += (INPUT_T)dL_dy(53) * (INPUT_T(-6.8841930899409371)*yz*(INPUT_T(3.0)*x2 - y2)*(INPUT_T(13.0)*z2 - INPUT_T(3.0)));                               // -33*sqrt(35)*yz*(3*x2 - y2)*(13*z2 - 3)/(16*sqrt(pi))
		dx += (INPUT_T)dL_dy(54) * (INPUT_T(0.44253269244498261)*yz*(INPUT_T(-110.0)*z2 + INPUT_T(143.0)*z4 + INPUT_T(15.0)));                               // 3*sqrt(70)*yz*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
		dy += (INPUT_T)dL_dy(54) * (INPUT_T(0.44253269244498261)*xz*(INPUT_T(-110.0)*z2 + INPUT_T(143.0)*z4 + INPUT_T(15.0)));                               // 3*sqrt(70)*xz*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
		dz += (INPUT_T)dL_dy(54) * (INPUT_T(2.2126634622249131)*xy*(INPUT_T(-66.0)*z2 + INPUT_T(143.0)*z4 + INPUT_T(3.0)));                          // 15*sqrt(70)*xy*(-66*z2 + 143*z4 + 3)/(32*sqrt(pi))
		// dx += (INPUT_T)dL_dy(55) * (0);                               // 0
		dy += (INPUT_T)dL_dy(55) * (INPUT_T(-12.194767023639836)*z2 + INPUT_T(44.714145753346067)*z4 - INPUT_T(38.752259652899923)*z6 + INPUT_T(0.45165803791258652));                               // sqrt(105)*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
		dz += (INPUT_T)dL_dy(55) * (INPUT_T(1.6259689364853116)*yz*(INPUT_T(110.0)*z2 - INPUT_T(143.0)*z4 - INPUT_T(15.0)));                         // 9*sqrt(105)*yz*(110*z2 - 143*z4 - 15)/(32*sqrt(pi))
		// dx += (INPUT_T)dL_dy(56) * (0);                               // 0
		// dy += (INPUT_T)dL_dy(56) * (0);                               // 0
		dz += (INPUT_T)dL_dy(56) * (INPUT_T(64.528641681844675)*z2 - INPUT_T(236.60501950009714)*z4 + INPUT_T(205.05768356675085)*z6 - INPUT_T(2.3899496919201733));                         // 7*sqrt(15)*(135*z2 - 495*z4 + 429*z6 - 5)/(32*sqrt(pi))
		dx += (INPUT_T)dL_dy(57) * (INPUT_T(-12.194767023639836)*z2 + INPUT_T(44.714145753346067)*z4 - INPUT_T(38.752259652899923)*z6 + INPUT_T(0.45165803791258652));                               // sqrt(105)*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
		// dy += (INPUT_T)dL_dy(57) * (0);                               // 0
		dz += (INPUT_T)dL_dy(57) * (INPUT_T(1.6259689364853116)*xz*(INPUT_T(110.0)*z2 - INPUT_T(143.0)*z4 - INPUT_T(15.0)));                         // 9*sqrt(105)*xz*(110*z2 - 143*z4 - 15)/(32*sqrt(pi))
		dx += (INPUT_T)dL_dy(58) * (INPUT_T(0.44253269244498261)*xz*(INPUT_T(-110.0)*z2 + INPUT_T(143.0)*z4 + INPUT_T(15.0)));                               // 3*sqrt(70)*xz*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
		dy += (INPUT_T)dL_dy(58) * (INPUT_T(0.44253269244498261)*yz*(INPUT_T(110.0)*z2 - INPUT_T(143.0)*z4 - INPUT_T(15.0)));                                // 3*sqrt(70)*yz*(110*z2 - 143*z4 - 15)/(32*sqrt(pi))
		dz += (INPUT_T)dL_dy(58) * (INPUT_T(0.07375544874083044)*(x2 - y2)*(INPUT_T(143.0)*z2*(INPUT_T(3.0)*z2 - INPUT_T(1.0)) + INPUT_T(132.0)*z2*(INPUT_T(13.0)*z2 - INPUT_T(5.0)) - INPUT_T(187.0)*z2 + INPUT_T(45.0)));                               // sqrt(70)*(x2 - y2)*(143*z2*(3*z2 - 1) + 132*z2*(13*z2 - 5) - 187*z2 + 45)/(64*sqrt(pi))
		dx += (INPUT_T)dL_dy(59) * (INPUT_T(30.97886890473422)*x2*z2 - INPUT_T(67.120882626924143)*x2*z4 - INPUT_T(1.4081304047606462)*x2 - INPUT_T(30.97886890473422)*y2*z2 + INPUT_T(67.120882626924143)*y2*z4 + INPUT_T(1.4081304047606462)*y2);                            // 9*sqrt(35)*(66*x2*z2 - 143*x2*z4 - 3*x2 - 66*y2*z2 + 143*y2*z4 + 3*y2)/(64*sqrt(pi))
		dy += (INPUT_T)dL_dy(59) * (INPUT_T(0.93875360317376422)*xy*(INPUT_T(-66.0)*z2 + INPUT_T(143.0)*z4 + INPUT_T(3.0)));                         // 9*sqrt(35)*xy*(-66*z2 + 143*z4 + 3)/(32*sqrt(pi))
		dz += (INPUT_T)dL_dy(59) * (INPUT_T(-6.8841930899409371)*xz*(x2 - INPUT_T(3.0)*y2)*(INPUT_T(13.0)*z2 - INPUT_T(3.0)));                               // -33*sqrt(35)*xz*(x2 - 3*y2)*(13*z2 - 3)/(16*sqrt(pi))
		dx += (INPUT_T)dL_dy(60) * (INPUT_T(4.1513246297620823)*xz*(x2 - INPUT_T(3.0)*y2)*(INPUT_T(13.0)*z2 - INPUT_T(3.0)));                                // 3*sqrt(385)*xz*(x2 - 3*y2)*(13*z2 - 3)/(8*sqrt(pi))
		dy += (INPUT_T)dL_dy(60) * (INPUT_T(-4.1513246297620823)*yz*(INPUT_T(3.0)*x2 - y2)*(INPUT_T(13.0)*z2 - INPUT_T(3.0)));                               // -3*sqrt(385)*yz*(3*x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
		dz += (INPUT_T)dL_dy(60) * (INPUT_T(3.1134934723215619)*(INPUT_T(13.0)*z2 - INPUT_T(1.0))*(INPUT_T(-6.0)*x2*y2 + x4 + y4));                          // 9*sqrt(385)*(13*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
		dx += (INPUT_T)dL_dy(61) * (INPUT_T(-0.51891557872026028)*(INPUT_T(13.0)*z2 - INPUT_T(1.0))*(INPUT_T(-10.0)*x2*y2 + INPUT_T(4.0)*x2*(x2 - INPUT_T(5.0)*y2) + x4 + INPUT_T(5.0)*y4));                            // -3*sqrt(385)*(13*z2 - 1)*(-10*x2*y2 + 4*x2*(x2 - 5*y2) + x4 + 5*y4)/(64*sqrt(pi))
		dy += (INPUT_T)dL_dy(61) * (INPUT_T(10.378311574405206)*xy*(x2 - y2)*(INPUT_T(13.0)*z2 - INPUT_T(1.0)));                            // 15*sqrt(385)*xy*(x2 - y2)*(13*z2 - 1)/(16*sqrt(pi))
		dz += (INPUT_T)dL_dy(61) * (INPUT_T(13.491805046726766)*xz*(INPUT_T(10.0)*x2*y2 - x4 - INPUT_T(5.0)*y4));                           // 39*sqrt(385)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
		dx += (INPUT_T)dL_dy(62) * (INPUT_T(15.875763970811402)*xz*(INPUT_T(-10.0)*x2*y2 + x4 + INPUT_T(5.0)*y4));                          // 9*sqrt(10010)*xz*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
		dy += (INPUT_T)dL_dy(62) * (INPUT_T(15.875763970811402)*yz*(INPUT_T(10.0)*x2*y2 - INPUT_T(5.0)*x4 - y4));                           // 9*sqrt(10010)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		dz += (INPUT_T)dL_dy(62) * (INPUT_T(39.6894099270285)*x2*y4 - INPUT_T(39.6894099270285)*x4*y2 + INPUT_T(2.6459606618019)*x6 - INPUT_T(2.6459606618019)*y6);                          // 3*sqrt(10010)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
		dx += (INPUT_T)dL_dy(63) * (INPUT_T(-74.252086915082614)*x2*y4 + INPUT_T(74.252086915082614)*x4*y2 - INPUT_T(4.9501391276721742)*x6 + INPUT_T(4.9501391276721742)*y6);                               // 21*sqrt(715)*(-15*x2*y4 + 15*x4*y2 - x6 + y6)/(64*sqrt(pi))
		dy += (INPUT_T)dL_dy(63) * (INPUT_T(9.9002782553443485)*xy*(INPUT_T(-10.0)*x2*y2 + INPUT_T(3.0)*x4 + INPUT_T(3.0)*y4));                              // 21*sqrt(715)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
		// dz += (INPUT_T)dL_dy(63) * (0);                               // 0
	};

	compute_grad();

	// Multiplication by 2 due to the conversion
	// from [0,1]^3 to directions in [-1,1]^3.
	// See implementation in `kernel_sh`.
	dL_dx(0, i) = dx * INPUT_T(2.0);
	dL_dx(1, i) = dy * INPUT_T(2.0);
	dL_dx(2, i) = dz * INPUT_T(2.0);
}

template <typename T, typename INPUT_T = float>
class SphericalHarmonicsEncoding : public Encoding<T, INPUT_T> {
public:
	SphericalHarmonicsEncoding(uint32_t degree, uint32_t n_dims_to_encode)
	: m_degree{degree}, m_n_dims_to_encode{n_dims_to_encode} {
		m_n_output_dims = degree * degree;

		if (n_dims_to_encode != 3) {
			throw std::runtime_error{"Can only encode 3D directions in spherical harmonics."};
		}

		if (m_degree <= 0) {
			throw std::runtime_error{"Spherical harmonics must have positive degree."};
		}

		if (m_degree > 8) {
			throw std::runtime_error{"Spherical harmonics are only implemented up to degree 8."};
		}
	}

	std::unique_ptr<Context> forward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<INPUT_T>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		if (!output || padded_output_width() == 0) {
			return std::make_unique<Context>();
		}

		linear_kernel(kernel_sh<T, INPUT_T>, 0, stream,
			input.n(),
			m_degree,
			m_n_to_pad,
			input.view(),
			output->view()
		);

		return std::make_unique<Context>();
	}

	void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<INPUT_T>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<INPUT_T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		EGradientMode param_gradients_mode = EGradientMode::Overwrite
	) override {
		if (!dL_dinput || padded_output_width() == 0) {
			return;
		}

		linear_kernel(kernel_sh_backward<T, INPUT_T>, 0, stream,
			input.n(),
			m_degree,
			m_n_to_pad,
			dL_doutput.view(),
			input.view(),
			dL_dinput->view()
		);
	}

	uint32_t input_width() const override {
		return m_n_dims_to_encode;
	}

	uint32_t padded_output_width() const override {
		return m_n_output_dims + m_n_to_pad;
	}

	uint32_t output_width() const override {
		return padded_output_width();
	}

	uint32_t required_input_alignment() const override {
		return 1;
	}

	void set_padded_output_width(uint32_t padded_output_width) override {
		CHECK_THROW(padded_output_width >= m_n_output_dims);
		m_n_to_pad = padded_output_width - m_n_output_dims;
	}

	uint32_t required_output_alignment() const override {
		return 1;
	}

	MatrixLayout preferred_output_layout() const override {
		return SoA;
	}

	json hyperparams() const override {
		return {
			{"otype", "SphericalHarmonics"},
			{"degree", m_degree},
		};
	}

private:
	uint32_t m_degree;
	uint32_t m_n_dims_to_encode;
	uint32_t m_n_trailing_dims_to_ignore;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_to_pad = 0;
};

TCNN_NAMESPACE_END
