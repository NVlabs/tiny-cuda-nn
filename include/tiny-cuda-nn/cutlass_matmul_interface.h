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

/** @file   cutlass_matmul_interface.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Matrix multiplication template interface. Separated into its own header file to improve
 *          compilation speed.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/misc_kernels.h>

#include <cutlass/half.h>


TCNN_NAMESPACE_BEGIN

template <Activation activation, typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, MatrixLayout LayoutC, typename TypeD, MatrixLayout LayoutD>
void fc_multiply(cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrix<TypeC, LayoutC>& C, GPUMatrix<TypeD, LayoutD>& D, network_precision_t beta = (network_precision_t)0, network_precision_t alpha = (network_precision_t)1);

TCNN_NAMESPACE_END
