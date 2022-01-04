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


template <typename T>
__global__ void kernel_sh(
	const uint32_t num_elements,
	const uint32_t sh_degree,
	const uint32_t num_to_pad,
	PitchedPtr<const float> data_in,
	PitchedPtr<T> data_out,
	float* __restrict__ dy_dx = nullptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	#pragma unroll
	for (uint32_t j = 0; j < num_to_pad; ++j) {
		data_out(i)[j] = (T)1.0f;
	}

	T* out = data_out(i) + num_to_pad;

	float x = data_in(i)[0] * 2.f - 1.f;
	float y = data_in(i)[1] * 2.f - 1.f;
	float z = data_in(i)[2] * 2.f - 1.f;

	// Let compiler figure out how to sequence/reorder these calculations w.r.t. branches
	float xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z, xyz=xy*z;
	float x4=x2*x2, y4=y2*y2, z4=z2*z2;
	float x6=x4*x2, y6=y4*y2, z6=z4*z2;

	// SH polynomials generated using scripts/gen_sh.py based on the recurrence relations in appendix A1 of https://www.ppsloan.org/publications/StupidSH36.pdf

	auto write_sh = [&]() {
		out[0] = 0.28209479177387814f ;                          // 1/(2*sqrt(pi))
		if (sh_degree <= 1) { return; }
		out[1] = -0.48860251190291987f*y ;                               // -sqrt(3)*y/(2*sqrt(pi))
		out[2] = 0.48860251190291987f*z ;                                // sqrt(3)*z/(2*sqrt(pi))
		out[3] = -0.48860251190291987f*x ;                               // -sqrt(3)*x/(2*sqrt(pi))
		if (sh_degree <= 2) { return; }
		out[4] = 1.0925484305920792f*xy ;                                // sqrt(15)*xy/(2*sqrt(pi))
		out[5] = -1.0925484305920792f*yz ;                               // -sqrt(15)*yz/(2*sqrt(pi))
		out[6] = 0.94617469575755997f*z2 - 0.31539156525251999f ;                         // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
		out[7] = -1.0925484305920792f*xz ;                               // -sqrt(15)*xz/(2*sqrt(pi))
		out[8] = 0.54627421529603959f*x2 - 0.54627421529603959f*y2 ;                              // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
		if (sh_degree <= 3) { return; }
		out[9] = 0.59004358992664352f*y*(-3.0f*x2 + y2) ;                         // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
		out[10] = 2.8906114426405538f*xy*z ;                             // sqrt(105)*xy*z/(2*sqrt(pi))
		out[11] = 0.45704579946446572f*y*(1.0f - 5.0f*z2) ;                                // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
		out[12] = 0.3731763325901154f*z*(5.0f*z2 - 3.0f) ;                         // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
		out[13] = 0.45704579946446572f*x*(1.0f - 5.0f*z2) ;                                // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
		out[14] = 1.4453057213202769f*z*(x2 - y2) ;                              // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
		out[15] = 0.59004358992664352f*x*(-x2 + 3.0f*y2) ;                                // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
		if (sh_degree <= 4) { return; }
		out[16] = 2.5033429417967046f*xy*(x2 - y2) ;                             // 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
		out[17] = 1.7701307697799304f*yz*(-3.0f*x2 + y2) ;                                // 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
		out[18] = 0.94617469575756008f*xy*(7.0f*z2 - 1.0f) ;                               // 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
		out[19] = 0.66904654355728921f*yz*(3.0f - 7.0f*z2) ;                               // 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
		out[20] = -3.1735664074561294f*z2 + 3.7024941420321507f*z4 + 0.31735664074561293f ;                                // 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
		out[21] = 0.66904654355728921f*xz*(3.0f - 7.0f*z2) ;                               // 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
		out[22] = 0.47308734787878004f*(x2 - y2)*(7.0f*z2 - 1.0f) ;                                // 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
		out[23] = 1.7701307697799304f*xz*(-x2 + 3.0f*y2) ;                                // 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
		out[24] = -3.7550144126950569f*x2*y2 + 0.62583573544917614f*x4 + 0.62583573544917614f*y4 ;                         // 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
		if (sh_degree <= 5) { return; }
		out[25] = 0.65638205684017015f*y*(10.0f*x2*y2 - 5.0f*x4 - y4) ;                            // 3*sqrt(154)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		out[26] = 8.3026492595241645f*xy*z*(x2 - y2) ;                           // 3*sqrt(385)*xy*z*(x2 - y2)/(4*sqrt(pi))
		out[27] = -0.48923829943525038f*y*(3.0f*x2 - y2)*(9.0f*z2 - 1.0f) ;                         // -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
		out[28] = 4.7935367849733241f*xy*z*(3.0f*z2 - 1.0f) ;                              // sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
		out[29] = 0.45294665119569694f*y*(14.0f*z2 - 21.0f*z4 - 1.0f) ;                             // sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
		out[30] = 0.1169503224534236f*z*(-70.0f*z2 + 63.0f*z4 + 15.0f) ;                            // sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
		out[31] = 0.45294665119569694f*x*(14.0f*z2 - 21.0f*z4 - 1.0f) ;                             // sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
		out[32] = 2.3967683924866621f*z*(x2 - y2)*(3.0f*z2 - 1.0f) ;                               // sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
		out[33] = -0.48923829943525038f*x*(x2 - 3.0f*y2)*(9.0f*z2 - 1.0f) ;                         // -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
		out[34] = 2.0756623148810411f*z*(-6.0f*x2*y2 + x4 + y4) ;                         // 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
		out[35] = 0.65638205684017015f*x*(10.0f*x2*y2 - x4 - 5.0f*y4) ;                            // 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
		if (sh_degree <= 6) { return; }
		out[36] = 1.3663682103838286f*xy*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4) ;                               // sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
		out[37] = 2.3666191622317521f*yz*(10.0f*x2*y2 - 5.0f*x4 - y4) ;                            // 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		out[38] = 2.0182596029148963f*xy*(x2 - y2)*(11.0f*z2 - 1.0f) ;                             // 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
		out[39] = -0.92120525951492349f*yz*(3.0f*x2 - y2)*(11.0f*z2 - 3.0f) ;                               // -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
		out[40] = 0.92120525951492349f*xy*(-18.0f*z2 + 33.0f*z4 + 1.0f) ;                           // sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
		out[41] = 0.58262136251873131f*yz*(30.0f*z2 - 33.0f*z4 - 5.0f) ;                            // sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
		out[42] = 6.6747662381009842f*z2 - 20.024298714302954f*z4 + 14.684485723822165f*z6 - 0.31784601133814211f ;                         // sqrt(13)*(105*z2 - 315*z4 + 231*z6 - 5)/(32*sqrt(pi))
		out[43] = 0.58262136251873131f*xz*(30.0f*z2 - 33.0f*z4 - 5.0f) ;                            // sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
		out[44] = 0.46060262975746175f*(x2 - y2)*(11.0f*z2*(3.0f*z2 - 1.0f) - 7.0f*z2 + 1.0f) ;                               // sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2 + 1)/(64*sqrt(pi))
		out[45] = -0.92120525951492349f*xz*(x2 - 3.0f*y2)*(11.0f*z2 - 3.0f) ;                               // -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
		out[46] = 0.50456490072872406f*(11.0f*z2 - 1.0f)*(-6.0f*x2*y2 + x4 + y4) ;                          // 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
		out[47] = 2.3666191622317521f*xz*(10.0f*x2*y2 - x4 - 5.0f*y4) ;                            // 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
		out[48] = 10.247761577878714f*x2*y4 - 10.247761577878714f*x4*y2 + 0.6831841051919143f*x6 - 0.6831841051919143f*y6 ;                         // sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
		if (sh_degree <= 7) { return; }
		out[49] = 0.70716273252459627f*y*(-21.0f*x2*y4 + 35.0f*x4*y2 - 7.0f*x6 + y6) ;                              // 3*sqrt(715)*y*(-21*x2*y4 + 35*x4*y2 - 7*x6 + y6)/(64*sqrt(pi))
		out[50] = 5.2919213236038001f*xy*z*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4) ;                             // 3*sqrt(10010)*xy*z*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
		out[51] = -0.51891557872026028f*y*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + 5.0f*x4 + y4) ;                          // -3*sqrt(385)*y*(13*z2 - 1)*(-10*x2*y2 + 5*x4 + y4)/(64*sqrt(pi))
		out[52] = 4.1513246297620823f*xy*z*(x2 - y2)*(13.0f*z2 - 3.0f) ;                           // 3*sqrt(385)*xy*z*(x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
		out[53] = -0.15645893386229404f*y*(3.0f*x2 - y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f) ;                              // -3*sqrt(35)*y*(3*x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
		out[54] = 0.44253269244498261f*xy*z*(-110.0f*z2 + 143.0f*z4 + 15.0f) ;                              // 3*sqrt(70)*xy*z*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
		out[55] = 0.090331607582517306f*y*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f) ;                              // sqrt(105)*y*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
		out[56] = 0.068284276912004949f*z*(315.0f*z2 - 693.0f*z4 + 429.0f*z6 - 35.0f) ;                              // sqrt(15)*z*(315*z2 - 693*z4 + 429*z6 - 35)/(32*sqrt(pi))
		out[57] = 0.090331607582517306f*x*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f) ;                              // sqrt(105)*x*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
		out[58] = 0.07375544874083044f*z*(x2 - y2)*(143.0f*z2*(3.0f*z2 - 1.0f) - 187.0f*z2 + 45.0f) ;                         // sqrt(70)*z*(x2 - y2)*(143*z2*(3*z2 - 1) - 187*z2 + 45)/(64*sqrt(pi))
		out[59] = -0.15645893386229404f*x*(x2 - 3.0f*y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f) ;                              // -3*sqrt(35)*x*(x2 - 3*y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
		out[60] = 1.0378311574405206f*z*(13.0f*z2 - 3.0f)*(-6.0f*x2*y2 + x4 + y4) ;                         // 3*sqrt(385)*z*(13*z2 - 3)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
		out[61] = -0.51891557872026028f*x*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + x4 + 5.0f*y4) ;                          // -3*sqrt(385)*x*(13*z2 - 1)*(-10*x2*y2 + x4 + 5*y4)/(64*sqrt(pi))
		out[62] = 2.6459606618019f*z*(15.0f*x2*y4 - 15.0f*x4*y2 + x6 - y6) ;                               // 3*sqrt(10010)*z*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
		out[63] = 0.70716273252459627f*x*(-35.0f*x2*y4 + 21.0f*x4*y2 - x6 + 7.0f*y6) ;                              // 3*sqrt(715)*x*(-35*x2*y4 + 21*x4*y2 - x6 + 7*y6)/(64*sqrt(pi))
	};

	write_sh();

	if (dy_dx == nullptr) {
		return;
	}

	uint32_t stride = sh_degree;

	float* dx = &dy_dx[i * stride * 3];
	float* dy = dx + stride;
	float* dz = dy + stride;

	auto write_sh_dx = [&]() {
		dx[0] = 0.0f ;                             // 0
		if (sh_degree <= 1) { return; }
		dx[1] = 0.0f ;                             // 0
		dx[2] = 0.0f ;                             // 0
		dx[3] = -0.48860251190291992f ;                          // -sqrt(3)/(2*sqrt(pi))
		if (sh_degree <= 2) { return; }
		dx[4] = 1.0925484305920792f*y ;                          // sqrt(15)*y/(2*sqrt(pi))
		dx[5] = 0.0f ;                             // 0
		dx[6] = 0.0f ;                             // 0
		dx[7] = -1.0925484305920792f*z ;                         // -sqrt(15)*z/(2*sqrt(pi))
		dx[8] = 1.0925484305920792f*x ;                          // sqrt(15)*x/(2*sqrt(pi))
		if (sh_degree <= 3) { return; }
		dx[9] = -3.5402615395598609f*xy ;                                // -3*sqrt(70)*xy/(4*sqrt(pi))
		dx[10] = 2.8906114426405538f*yz ;                                // sqrt(105)*yz/(2*sqrt(pi))
		dx[11] = 0.0f ;                            // 0
		dx[12] = 0.0f ;                            // 0
		dx[13] = 0.45704579946446572f - 2.2852289973223288f*z2 ;                          // sqrt(42)*(1 - 5*z2)/(8*sqrt(pi))
		dx[14] = 2.8906114426405538f*xz ;                                // sqrt(105)*xz/(2*sqrt(pi))
		dx[15] = -1.7701307697799304f*x2 + 1.7701307697799304f*y2 ;                               // 3*sqrt(70)*(-x2 + y2)/(8*sqrt(pi))
		if (sh_degree <= 4) { return; }
		dx[16] = 2.5033429417967046f*y*(3.0f*x2 - y2) ;                           // 3*sqrt(35)*y*(3*x2 - y2)/(4*sqrt(pi))
		dx[17] = -10.620784618679583f*xy*z ;                             // -9*sqrt(70)*xy*z/(4*sqrt(pi))
		dx[18] = 0.94617469575756008f*y*(7.0f*z2 - 1.0f) ;                         // 3*sqrt(5)*y*(7*z2 - 1)/(4*sqrt(pi))
		dx[19] = 0.0f ;                            // 0
		dx[20] = 0.0f ;                            // 0
		dx[21] = 0.66904654355728921f*z*(3.0f - 7.0f*z2) ;                         // 3*sqrt(10)*z*(3 - 7*z2)/(8*sqrt(pi))
		dx[22] = 0.94617469575756008f*x*(7.0f*z2 - 1.0f) ;                         // 3*sqrt(5)*x*(7*z2 - 1)/(4*sqrt(pi))
		dx[23] = 5.3103923093397913f*z*(-x2 + y2) ;                              // 9*sqrt(70)*z*(-x2 + y2)/(8*sqrt(pi))
		dx[24] = 2.5033429417967046f*x*(x2 - 3.0f*y2) ;                           // 3*sqrt(35)*x*(x2 - 3*y2)/(4*sqrt(pi))
		if (sh_degree <= 5) { return; }
		dx[25] = 13.127641136803401f*xy*(-x2 + y2) ;                             // 15*sqrt(154)*xy*(-x2 + y2)/(8*sqrt(pi))
		dx[26] = 8.3026492595241645f*yz*(3.0f*x2 - y2) ;                          // 3*sqrt(385)*yz*(3*x2 - y2)/(4*sqrt(pi))
		dx[27] = 2.9354297966115022f*xy*(1.0f - 9.0f*z2) ;                         // 3*sqrt(770)*xy*(1 - 9*z2)/(16*sqrt(pi))
		dx[28] = 4.7935367849733241f*yz*(3.0f*z2 - 1.0f) ;                         // sqrt(1155)*yz*(3*z2 - 1)/(4*sqrt(pi))
		dx[29] = 0.0f ;                            // 0
		dx[30] = 0.0f ;                            // 0
		dx[31] = 6.3412531167397574f*z2 - 9.5118796751096362f*z4 - 0.45294665119569694f ;                          // sqrt(165)*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
		dx[32] = 4.7935367849733241f*xz*(3.0f*z2 - 1.0f) ;                         // sqrt(1155)*xz*(3*z2 - 1)/(4*sqrt(pi))
		dx[33] = -13.209434084751759f*x2*z2 + 1.4677148983057511f*x2 + 13.209434084751759f*y2*z2 - 1.4677148983057511f*y2 ;                         // 3*sqrt(770)*(-9*x2*z2 + x2 + 9*y2*z2 - y2)/(32*sqrt(pi))
		dx[34] = 8.3026492595241645f*xz*(x2 - 3.0f*y2) ;                          // 3*sqrt(385)*xz*(x2 - 3*y2)/(4*sqrt(pi))
		dx[35] = 19.6914617052051f*x2*y2 - 3.2819102842008503f*x4 - 3.2819102842008503f*y4 ;                               // 15*sqrt(154)*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
		if (sh_degree <= 6) { return; }
		dx[36] = 4.0991046311514854f*y*(-10.0f*x2*y2 + 5.0f*x4 + y4) ;                             // 3*sqrt(6006)*y*(-10*x2*y2 + 5*x4 + y4)/(32*sqrt(pi))
		dx[37] = 47.332383244635047f*xy*z*(-x2 + y2) ;                           // 15*sqrt(2002)*xy*z*(-x2 + y2)/(8*sqrt(pi))
		dx[38] = 2.0182596029148963f*y*(3.0f*x2 - y2)*(11.0f*z2 - 1.0f) ;                           // 3*sqrt(91)*y*(3*x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
		dx[39] = 5.5272315570895412f*xy*z*(3.0f - 11.0f*z2) ;                              // 3*sqrt(2730)*xy*z*(3 - 11*z2)/(16*sqrt(pi))
		dx[40] = 0.92120525951492349f*y*(-18.0f*z2 + 33.0f*z4 + 1.0f) ;                             // sqrt(2730)*y*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
		dx[41] = 0.0f ;                            // 0
		dx[42] = 0.0f ;                            // 0
		dx[43] = 0.58262136251873131f*z*(30.0f*z2 - 33.0f*z4 - 5.0f) ;                              // sqrt(273)*z*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
		dx[44] = 0.92120525951492349f*x*(-18.0f*z2 + 33.0f*z4 + 1.0f) ;                             // sqrt(2730)*x*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
		dx[45] = -2.7636157785447706f*z*(x2 - y2)*(11.0f*z2 - 3.0f) ;                              // -3*sqrt(2730)*z*(x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
		dx[46] = 2.0182596029148963f*x*(x2 - 3.0f*y2)*(11.0f*z2 - 1.0f) ;                           // 3*sqrt(91)*x*(x2 - 3*y2)*(11*z2 - 1)/(8*sqrt(pi))
		dx[47] = 11.833095811158762f*z*(6.0f*x2*y2 - x4 - y4) ;                           // 15*sqrt(2002)*z*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
		dx[48] = 4.0991046311514854f*x*(-10.0f*x2*y2 + x4 + 5.0f*y4) ;                             // 3*sqrt(6006)*x*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
		if (sh_degree <= 7) { return; }
		dx[49] = 9.9002782553443485f*xy*(10.0f*x2*y2 - 3.0f*x4 - 3.0f*y4) ;                         // 21*sqrt(715)*xy*(10*x2*y2 - 3*x4 - 3*y4)/(32*sqrt(pi))
		dx[50] = 15.875763970811402f*yz*(-10.0f*x2*y2 + 5.0f*x4 + y4) ;                            // 9*sqrt(10010)*yz*(-10*x2*y2 + 5*x4 + y4)/(32*sqrt(pi))
		dx[51] = -10.378311574405206f*xy*(x2 - y2)*(13.0f*z2 - 1.0f) ;                             // -15*sqrt(385)*xy*(x2 - y2)*(13*z2 - 1)/(16*sqrt(pi))
		dx[52] = 4.1513246297620823f*yz*(3.0f*x2 - y2)*(13.0f*z2 - 3.0f) ;                          // 3*sqrt(385)*yz*(3*x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
		dx[53] = 0.93875360317376422f*xy*(66.0f*z2 - 143.0f*z4 - 3.0f) ;                            // 9*sqrt(35)*xy*(66*z2 - 143*z4 - 3)/(32*sqrt(pi))
		dx[54] = 0.44253269244498261f*yz*(-110.0f*z2 + 143.0f*z4 + 15.0f) ;                         // 3*sqrt(70)*yz*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
		dx[55] = 0.0f ;                            // 0
		dx[56] = 0.0f ;                            // 0
		dx[57] = -12.194767023639836f*z2 + 44.714145753346067f*z4 - 38.752259652899923f*z6 + 0.45165803791258652f ;                         // sqrt(105)*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
		dx[58] = 0.44253269244498261f*xz*(-110.0f*z2 + 143.0f*z4 + 15.0f) ;                         // 3*sqrt(70)*xz*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
		dx[59] = 30.97886890473422f*x2*z2 - 67.120882626924143f*x2*z4 - 1.4081304047606462f*x2 - 30.97886890473422f*y2*z2 + 67.120882626924143f*y2*z4 + 1.4081304047606462f*y2 ;                              // 9*sqrt(35)*(66*x2*z2 - 143*x2*z4 - 3*x2 - 66*y2*z2 + 143*y2*z4 + 3*y2)/(64*sqrt(pi))
		dx[60] = 4.1513246297620823f*xz*(x2 - 3.0f*y2)*(13.0f*z2 - 3.0f) ;                          // 3*sqrt(385)*xz*(x2 - 3*y2)*(13*z2 - 3)/(8*sqrt(pi))
		dx[61] = -0.51891557872026028f*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + 4.0f*x2*(x2 - 5.0f*y2) + x4 + 5.0f*y4) ;                              // -3*sqrt(385)*(13*z2 - 1)*(-10*x2*y2 + 4*x2*(x2 - 5*y2) + x4 + 5*y4)/(64*sqrt(pi))
		dx[62] = 15.875763970811402f*xz*(-10.0f*x2*y2 + x4 + 5.0f*y4) ;                            // 9*sqrt(10010)*xz*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
		dx[63] = -74.252086915082614f*x2*y4 + 74.252086915082614f*x4*y2 - 4.9501391276721742f*x6 + 4.9501391276721742f*y6 ;                         // 21*sqrt(715)*(-15*x2*y4 + 15*x4*y2 - x6 + y6)/(64*sqrt(pi))
	};

	auto write_sh_dy = [&]() {
		dy[0] = 0.0f ;                             // 0
		if (sh_degree <= 1) { return; }
		dy[1] = -0.48860251190291992f ;                          // -sqrt(3)/(2*sqrt(pi))
		dy[2] = 0.0f ;                             // 0
		dy[3] = 0.0f ;                             // 0
		if (sh_degree <= 2) { return; }
		dy[4] = 1.0925484305920792f*x ;                          // sqrt(15)*x/(2*sqrt(pi))
		dy[5] = -1.0925484305920792f*z ;                         // -sqrt(15)*z/(2*sqrt(pi))
		dy[6] = 0.0f ;                             // 0
		dy[7] = 0.0f ;                             // 0
		dy[8] = -1.0925484305920792f*y ;                         // -sqrt(15)*y/(2*sqrt(pi))
		if (sh_degree <= 3) { return; }
		dy[9] = -1.7701307697799304f*x2 + 1.7701307697799304f*y2 ;                                // 3*sqrt(70)*(-x2 + y2)/(8*sqrt(pi))
		dy[10] = 2.8906114426405538f*xz ;                                // sqrt(105)*xz/(2*sqrt(pi))
		dy[11] = 0.45704579946446572f - 2.2852289973223288f*z2 ;                          // sqrt(42)*(1 - 5*z2)/(8*sqrt(pi))
		dy[12] = 0.0f ;                            // 0
		dy[13] = 0.0f ;                            // 0
		dy[14] = -2.8906114426405538f*yz ;                               // -sqrt(105)*yz/(2*sqrt(pi))
		dy[15] = 3.5402615395598609f*xy ;                                // 3*sqrt(70)*xy/(4*sqrt(pi))
		if (sh_degree <= 4) { return; }
		dy[16] = 2.5033429417967046f*x*(x2 - 3.0f*y2) ;                           // 3*sqrt(35)*x*(x2 - 3*y2)/(4*sqrt(pi))
		dy[17] = 5.3103923093397913f*z*(-x2 + y2) ;                              // 9*sqrt(70)*z*(-x2 + y2)/(8*sqrt(pi))
		dy[18] = 0.94617469575756008f*x*(7.0f*z2 - 1.0f) ;                         // 3*sqrt(5)*x*(7*z2 - 1)/(4*sqrt(pi))
		dy[19] = 0.66904654355728921f*z*(3.0f - 7.0f*z2) ;                         // 3*sqrt(10)*z*(3 - 7*z2)/(8*sqrt(pi))
		dy[20] = 0.0f ;                            // 0
		dy[21] = 0.0f ;                            // 0
		dy[22] = 0.94617469575756008f*y*(1.0f - 7.0f*z2) ;                         // 3*sqrt(5)*y*(1 - 7*z2)/(4*sqrt(pi))
		dy[23] = 10.620784618679583f*xy*z ;                              // 9*sqrt(70)*xy*z/(4*sqrt(pi))
		dy[24] = 2.5033429417967046f*y*(-3.0f*x2 + y2) ;                          // 3*sqrt(35)*y*(-3*x2 + y2)/(4*sqrt(pi))
		if (sh_degree <= 5) { return; }
		dy[25] = 19.6914617052051f*x2*y2 - 3.2819102842008503f*x4 - 3.2819102842008503f*y4 ;                               // 15*sqrt(154)*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
		dy[26] = 8.3026492595241645f*xz*(x2 - 3.0f*y2) ;                          // 3*sqrt(385)*xz*(x2 - 3*y2)/(4*sqrt(pi))
		dy[27] = -1.4677148983057511f*(x2 - y2)*(9.0f*z2 - 1.0f) ;                         // -3*sqrt(770)*(x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
		dy[28] = 4.7935367849733241f*xz*(3.0f*z2 - 1.0f) ;                         // sqrt(1155)*xz*(3*z2 - 1)/(4*sqrt(pi))
		dy[29] = 6.3412531167397574f*z2 - 9.5118796751096362f*z4 - 0.45294665119569694f ;                          // sqrt(165)*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
		dy[30] = 0.0f ;                            // 0
		dy[31] = 0.0f ;                            // 0
		dy[32] = 4.7935367849733241f*yz*(1.0f - 3.0f*z2) ;                         // sqrt(1155)*yz*(1 - 3*z2)/(4*sqrt(pi))
		dy[33] = 2.9354297966115022f*xy*(9.0f*z2 - 1.0f) ;                         // 3*sqrt(770)*xy*(9*z2 - 1)/(16*sqrt(pi))
		dy[34] = 8.3026492595241645f*yz*(-3.0f*x2 + y2) ;                         // 3*sqrt(385)*yz*(-3*x2 + y2)/(4*sqrt(pi))
		dy[35] = 13.127641136803401f*xy*(x2 - y2) ;                              // 15*sqrt(154)*xy*(x2 - y2)/(8*sqrt(pi))
		if (sh_degree <= 6) { return; }
		dy[36] = 4.0991046311514854f*x*(-10.0f*x2*y2 + x4 + 5.0f*y4) ;                             // 3*sqrt(6006)*x*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
		dy[37] = 11.833095811158762f*z*(6.0f*x2*y2 - x4 - y4) ;                           // 15*sqrt(2002)*z*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
		dy[38] = 2.0182596029148963f*x*(x2 - 3.0f*y2)*(11.0f*z2 - 1.0f) ;                           // 3*sqrt(91)*x*(x2 - 3*y2)*(11*z2 - 1)/(8*sqrt(pi))
		dy[39] = -2.7636157785447706f*z*(x2 - y2)*(11.0f*z2 - 3.0f) ;                              // -3*sqrt(2730)*z*(x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
		dy[40] = 0.92120525951492349f*x*(-18.0f*z2 + 33.0f*z4 + 1.0f) ;                             // sqrt(2730)*x*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
		dy[41] = 0.58262136251873131f*z*(30.0f*z2 - 33.0f*z4 - 5.0f) ;                              // sqrt(273)*z*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
		dy[42] = 0.0f ;                            // 0
		dy[43] = 0.0f ;                            // 0
		dy[44] = 0.92120525951492349f*y*(18.0f*z2 - 33.0f*z4 - 1.0f) ;                              // sqrt(2730)*y*(18*z2 - 33*z4 - 1)/(32*sqrt(pi))
		dy[45] = 5.5272315570895412f*xy*z*(11.0f*z2 - 3.0f) ;                              // 3*sqrt(2730)*xy*z*(11*z2 - 3)/(16*sqrt(pi))
		dy[46] = -2.0182596029148963f*y*(3.0f*x2 - y2)*(11.0f*z2 - 1.0f) ;                          // -3*sqrt(91)*y*(3*x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
		dy[47] = 47.332383244635047f*xy*z*(x2 - y2) ;                            // 15*sqrt(2002)*xy*z*(x2 - y2)/(8*sqrt(pi))
		dy[48] = 4.0991046311514854f*y*(10.0f*x2*y2 - 5.0f*x4 - y4) ;                              // 3*sqrt(6006)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		if (sh_degree <= 7) { return; }
		dy[49] = -74.252086915082614f*x2*y4 + 74.252086915082614f*x4*y2 - 4.9501391276721742f*x6 + 4.9501391276721742f*y6 ;                         // 21*sqrt(715)*(-15*x2*y4 + 15*x4*y2 - x6 + y6)/(64*sqrt(pi))
		dy[50] = 15.875763970811402f*xz*(-10.0f*x2*y2 + x4 + 5.0f*y4) ;                            // 9*sqrt(10010)*xz*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
		dy[51] = 0.51891557872026028f*(13.0f*z2 - 1.0f)*(10.0f*x2*y2 - 5.0f*x4 + 4.0f*y2*(5.0f*x2 - y2) - y4) ;                                // 3*sqrt(385)*(13*z2 - 1)*(10*x2*y2 - 5*x4 + 4*y2*(5*x2 - y2) - y4)/(64*sqrt(pi))
		dy[52] = 4.1513246297620823f*xz*(x2 - 3.0f*y2)*(13.0f*z2 - 3.0f) ;                          // 3*sqrt(385)*xz*(x2 - 3*y2)*(13*z2 - 3)/(8*sqrt(pi))
		dy[53] = -0.46937680158688211f*(x2 - y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f) ;                             // -9*sqrt(35)*(x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
		dy[54] = 0.44253269244498261f*xz*(-110.0f*z2 + 143.0f*z4 + 15.0f) ;                         // 3*sqrt(70)*xz*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
		dy[55] = -12.194767023639836f*z2 + 44.714145753346067f*z4 - 38.752259652899923f*z6 + 0.45165803791258652f ;                         // sqrt(105)*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
		dy[56] = 0.0f ;                            // 0
		dy[57] = 0.0f ;                            // 0
		dy[58] = 0.44253269244498261f*yz*(110.0f*z2 - 143.0f*z4 - 15.0f) ;                          // 3*sqrt(70)*yz*(110*z2 - 143*z4 - 15)/(32*sqrt(pi))
		dy[59] = 0.93875360317376422f*xy*(-66.0f*z2 + 143.0f*z4 + 3.0f) ;                           // 9*sqrt(35)*xy*(-66*z2 + 143*z4 + 3)/(32*sqrt(pi))
		dy[60] = -4.1513246297620823f*yz*(3.0f*x2 - y2)*(13.0f*z2 - 3.0f) ;                         // -3*sqrt(385)*yz*(3*x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
		dy[61] = 10.378311574405206f*xy*(x2 - y2)*(13.0f*z2 - 1.0f) ;                              // 15*sqrt(385)*xy*(x2 - y2)*(13*z2 - 1)/(16*sqrt(pi))
		dy[62] = 15.875763970811402f*yz*(10.0f*x2*y2 - 5.0f*x4 - y4) ;                             // 9*sqrt(10010)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		dy[63] = 9.9002782553443485f*xy*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4) ;                                // 21*sqrt(715)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
	};

	auto write_sh_dz = [&]() {
		dz[0] = 0.0f ;                             // 0
		if (sh_degree <= 1) { return; }
		dz[1] = 0.0f ;                             // 0
		dz[2] = 0.48860251190291992f ;                           // sqrt(3)/(2*sqrt(pi))
		dz[3] = 0.0f ;                             // 0
		if (sh_degree <= 2) { return; }
		dz[4] = 0.0f ;                             // 0
		dz[5] = -1.0925484305920792f*y ;                         // -sqrt(15)*y/(2*sqrt(pi))
		dz[6] = 1.8923493915151202f*z ;                          // 3*sqrt(5)*z/(2*sqrt(pi))
		dz[7] = -1.0925484305920792f*x ;                         // -sqrt(15)*x/(2*sqrt(pi))
		dz[8] = 0.0f ;                             // 0
		if (sh_degree <= 3) { return; }
		dz[9] = 0.0f ;                             // 0
		dz[10] = 2.8906114426405538f*xy ;                                // sqrt(105)*xy/(2*sqrt(pi))
		dz[11] = -4.5704579946446566f*yz ;                               // -5*sqrt(42)*yz/(4*sqrt(pi))
		dz[12] = 5.597644988851731f*z2 - 1.1195289977703462f ;                            // 3*sqrt(7)*(5*z2 - 1)/(4*sqrt(pi))
		dz[13] = -4.5704579946446566f*xz ;                               // -5*sqrt(42)*xz/(4*sqrt(pi))
		dz[14] = 1.4453057213202769f*x2 - 1.4453057213202769f*y2 ;                                // sqrt(105)*(x2 - y2)/(4*sqrt(pi))
		dz[15] = 0.0f ;                            // 0
		if (sh_degree <= 4) { return; }
		dz[16] = 0.0f ;                            // 0
		dz[17] = 1.7701307697799304f*y*(-3.0f*x2 + y2) ;                          // 3*sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
		dz[18] = 13.246445740605839f*xy*z ;                              // 21*sqrt(5)*xy*z/(2*sqrt(pi))
		dz[19] = 2.0071396306718676f*y*(1.0f - 7.0f*z2) ;                          // 9*sqrt(10)*y*(1 - 7*z2)/(8*sqrt(pi))
		dz[20] = 14.809976568128603f*pow(z, 3) - 6.3471328149122579f*z ;                          // (105*z**3 - 45*z)/(4*sqrt(pi))
		dz[21] = 2.0071396306718676f*x*(1.0f - 7.0f*z2) ;                          // 9*sqrt(10)*x*(1 - 7*z2)/(8*sqrt(pi))
		dz[22] = 6.6232228703029197f*z*(x2 - y2) ;                               // 21*sqrt(5)*z*(x2 - y2)/(4*sqrt(pi))
		dz[23] = 1.7701307697799304f*x*(-x2 + 3.0f*y2) ;                          // 3*sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
		dz[24] = 0.0f ;                            // 0
		if (sh_degree <= 5) { return; }
		dz[25] = 0.0f ;                            // 0
		dz[26] = 8.3026492595241645f*xy*(x2 - y2) ;                              // 3*sqrt(385)*xy*(x2 - y2)/(4*sqrt(pi))
		dz[27] = 8.8062893898345074f*yz*(-3.0f*x2 + y2) ;                         // 9*sqrt(770)*yz*(-3*x2 + y2)/(16*sqrt(pi))
		dz[28] = 4.7935367849733241f*xy*(9.0f*z2 - 1.0f) ;                         // sqrt(1155)*xy*(9*z2 - 1)/(4*sqrt(pi))
		dz[29] = 12.682506233479513f*yz*(1.0f - 3.0f*z2) ;                         // 7*sqrt(165)*yz*(1 - 3*z2)/(4*sqrt(pi))
		dz[30] = -24.559567715218954f*z2 + 36.839351572828434f*z4 + 1.754254836801354f ;                           // 15*sqrt(11)*(-14*z2 + 21*z4 + 1)/(16*sqrt(pi))
		dz[31] = 12.682506233479513f*xz*(1.0f - 3.0f*z2) ;                         // 7*sqrt(165)*xz*(1 - 3*z2)/(4*sqrt(pi))
		dz[32] = 2.3967683924866621f*(x2 - y2)*(9.0f*z2 - 1.0f) ;                          // sqrt(1155)*(x2 - y2)*(9*z2 - 1)/(8*sqrt(pi))
		dz[33] = 8.8062893898345074f*xz*(-x2 + 3.0f*y2) ;                         // 9*sqrt(770)*xz*(-x2 + 3*y2)/(16*sqrt(pi))
		dz[34] = -12.453973889286246f*x2*y2 + 2.0756623148810411f*x4 + 2.0756623148810411f*y4 ;                            // 3*sqrt(385)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
		dz[35] = 0.0f ;                            // 0
		if (sh_degree <= 6) { return; }
		dz[36] = 0.0f ;                            // 0
		dz[37] = 2.3666191622317521f*y*(10.0f*x2*y2 - 5.0f*x4 - y4) ;                              // 3*sqrt(2002)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		dz[38] = 44.401711264127719f*xy*z*(x2 - y2) ;                            // 33*sqrt(91)*xy*z*(x2 - y2)/(4*sqrt(pi))
		dz[39] = -2.7636157785447706f*y*(3.0f*x2 - y2)*(11.0f*z2 - 1.0f) ;                          // -3*sqrt(2730)*y*(3*x2 - y2)*(11*z2 - 1)/(32*sqrt(pi))
		dz[40] = 11.054463114179082f*xy*z*(11.0f*z2 - 3.0f) ;                              // 3*sqrt(2730)*xy*z*(11*z2 - 3)/(8*sqrt(pi))
		dz[41] = 2.9131068125936568f*y*(18.0f*z2 - 33.0f*z4 - 1.0f) ;                               // 5*sqrt(273)*y*(18*z2 - 33*z4 - 1)/(16*sqrt(pi))
		dz[42] = 2.6699064952403937f*z*(-30.0f*z2 + 33.0f*z4 + 5.0f) ;                              // 21*sqrt(13)*z*(-30*z2 + 33*z4 + 5)/(16*sqrt(pi))
		dz[43] = 2.9131068125936568f*x*(18.0f*z2 - 33.0f*z4 - 1.0f) ;                               // 5*sqrt(273)*x*(18*z2 - 33*z4 - 1)/(16*sqrt(pi))
		dz[44] = 5.5272315570895412f*z*(x2 - y2)*(11.0f*z2 - 3.0f) ;                               // 3*sqrt(2730)*z*(x2 - y2)*(11*z2 - 3)/(16*sqrt(pi))
		dz[45] = -2.7636157785447706f*x*(x2 - 3.0f*y2)*(11.0f*z2 - 1.0f) ;                          // -3*sqrt(2730)*x*(x2 - 3*y2)*(11*z2 - 1)/(32*sqrt(pi))
		dz[46] = 11.10042781603193f*z*(-6.0f*x2*y2 + x4 + y4) ;                           // 33*sqrt(91)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
		dz[47] = 2.3666191622317521f*x*(10.0f*x2*y2 - x4 - 5.0f*y4) ;                              // 3*sqrt(2002)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
		dz[48] = 0.0f ;                            // 0
		if (sh_degree <= 7) { return; }
		dz[49] = 0.0f ;                            // 0
		dz[50] = 5.2919213236038001f*xy*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4) ;                                // 3*sqrt(10010)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
		dz[51] = 13.491805046726766f*yz*(10.0f*x2*y2 - 5.0f*x4 - y4) ;                             // 39*sqrt(385)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		dz[52] = 12.453973889286248f*xy*(x2 - y2)*(13.0f*z2 - 1.0f) ;                              // 9*sqrt(385)*xy*(x2 - y2)*(13*z2 - 1)/(8*sqrt(pi))
		dz[53] = -6.8841930899409371f*yz*(3.0f*x2 - y2)*(13.0f*z2 - 3.0f) ;                         // -33*sqrt(35)*yz*(3*x2 - y2)*(13*z2 - 3)/(16*sqrt(pi))
		dz[54] = 2.2126634622249131f*xy*(-66.0f*z2 + 143.0f*z4 + 3.0f) ;                            // 15*sqrt(70)*xy*(-66*z2 + 143*z4 + 3)/(32*sqrt(pi))
		dz[55] = 1.6259689364853116f*yz*(110.0f*z2 - 143.0f*z4 - 15.0f) ;                           // 9*sqrt(105)*yz*(110*z2 - 143*z4 - 15)/(32*sqrt(pi))
		dz[56] = 64.528641681844675f*z2 - 236.60501950009714f*z4 + 205.05768356675085f*z6 - 2.3899496919201733f ;                           // 7*sqrt(15)*(135*z2 - 495*z4 + 429*z6 - 5)/(32*sqrt(pi))
		dz[57] = 1.6259689364853116f*xz*(110.0f*z2 - 143.0f*z4 - 15.0f) ;                           // 9*sqrt(105)*xz*(110*z2 - 143*z4 - 15)/(32*sqrt(pi))
		dz[58] = 0.07375544874083044f*(x2 - y2)*(143.0f*z2*(3.0f*z2 - 1.0f) + 132.0f*z2*(13.0f*z2 - 5.0f) - 187.0f*z2 + 45.0f) ;                         // sqrt(70)*(x2 - y2)*(143*z2*(3*z2 - 1) + 132*z2*(13*z2 - 5) - 187*z2 + 45)/(64*sqrt(pi))
		dz[59] = -6.8841930899409371f*xz*(x2 - 3.0f*y2)*(13.0f*z2 - 3.0f) ;                         // -33*sqrt(35)*xz*(x2 - 3*y2)*(13*z2 - 3)/(16*sqrt(pi))
		dz[60] = 3.1134934723215619f*(13.0f*z2 - 1.0f)*(-6.0f*x2*y2 + x4 + y4) ;                            // 9*sqrt(385)*(13*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
		dz[61] = 13.491805046726766f*xz*(10.0f*x2*y2 - x4 - 5.0f*y4) ;                             // 39*sqrt(385)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
		dz[62] = 39.6894099270285f*x2*y4 - 39.6894099270285f*x4*y2 + 2.6459606618019f*x6 - 2.6459606618019f*y6 ;                            // 3*sqrt(10010)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
		dz[63] = 0.0f ;                            // 0
	};

	write_sh_dx();
	write_sh_dy();
	write_sh_dz();
}

template <typename T>
__global__ void kernel_sh_backward(
	const uint32_t num_elements,
	const uint32_t sh_degree,
	PitchedPtr<const T> dL_dy,
	const float* dy_dx,
	PitchedPtr<float> dL_dx)
{
	const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (encoded_index >= num_elements) return;

	const uint32_t i = encoded_index / 3;
	const uint32_t j = encoded_index - i * 3;

	const uint32_t outputs_per_input = sh_degree * sh_degree;

	float result = 0;
	for (int k = 0; k < outputs_per_input; ++k) {
		result += (float)dL_dy(i)[j * outputs_per_input + k] * dy_dx[i * outputs_per_input * 3 + j * outputs_per_input + k];
	}

	// Multiplication by 2 due to the conversion
	// from [0,1]^3 to directions in [-1,1]^3.
	// See implementation in `kernel_sh`.
	dL_dx(i)[j] = result * 2.0f;
}

template <typename T>
class SphericalHarmonicsEncoding : public Encoding<T> {
public:
	SphericalHarmonicsEncoding(uint32_t sh_degree, uint32_t n_dims_to_encode)
	: m_sh_degree{sh_degree}, m_n_dims_to_encode{n_dims_to_encode} {
		m_n_padded_output_dims = m_n_output_dims = sh_degree * sh_degree;

		if (n_dims_to_encode != 3) {
			throw std::runtime_error{"Can only encode 3D directions in spherical harmonics."};
		}

		if (m_sh_degree <= 0) {
			throw std::runtime_error{"Spherical harmonics must have positive degree."};
		}

		if (m_sh_degree > 8) {
			throw std::runtime_error{"Spherical harmonics are only implemented up to degree 8."};
		}
	}

	void encode(
		cudaStream_t stream,
		const uint32_t num_elements,
		PitchedPtr<const float> inputs,
		PitchedPtr<T> outputs,
		float* dy_dx = nullptr,
		bool is_inference = false
	) const override {
		if (m_n_padded_output_dims == 0) {
			return;
		}

		linear_kernel(kernel_sh<T>, 0, stream,
			num_elements,
			m_sh_degree,
			m_n_to_pad,
			inputs,
			outputs,
			dy_dx
		);
	}

	void backward(
		cudaStream_t stream,
		const uint32_t num_elements,
		PitchedPtr<const T> dL_dy, // Same shape as outputs
		const float* dy_dx, // encoded output dims x num_elements
		PitchedPtr<float> dL_dx, // Same shape as inputs
		PitchedPtr<const float> inputs,
		bool accumulate_param_gradients
	) override {
		if (m_n_padded_output_dims == 0) {
			return;
		}

		// Can't compute input gradients if insufficient info is available
		if (!dy_dx || !dL_dx) {
			return;
		}

		linear_kernel(kernel_sh_backward<T>, 0, stream,
			num_elements * 3,
			m_sh_degree,
			dL_dy,
			dy_dx,
			dL_dx
		);
	}

	uint32_t num_dims_to_encode() const override {
		return m_n_dims_to_encode;
	}

	uint32_t num_encoded_dims() const override {
		return m_n_padded_output_dims;
	}

	uint32_t num_forward_gradient_dims() const override {
		return m_sh_degree * m_sh_degree * 3;
	}

	void set_alignment(uint32_t alignment) override {
		alignment = lcm(alignment, min_alignment());
		m_n_padded_output_dims = next_multiple(m_n_output_dims, alignment);
		m_n_to_pad = m_n_padded_output_dims - m_n_output_dims;
	}

	uint32_t min_alignment() const override {
		return 1;
	}

private:
	uint32_t m_sh_degree;
	uint32_t m_n_dims_to_encode;
	uint32_t m_n_trailing_dims_to_ignore;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_padded_output_dims;
	uint32_t m_n_to_pad = 0;
};

TCNN_NAMESPACE_END
