// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Copyright (c) 2018, Hisilicon Limited
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef GFPQ_HPP_
#define GFPQ_HPP_

#define GFPQ_BIT_WIDTH_8BIT (8)
#define GFPQ_BIT_WIDTH_16BIT (16)

#define GFPQ_MODE_INIT (0)
#define GFPQ_MODE_UPDATE (1)
#define GFPQ_MODE_APPLY_ONLY (2)

#define GFPQ_SUCCESS 0x00000000
#define GFPQ_ERR_GENERIC 0xFFFF0000
#define GFPQ_ERR_BAD_FORMAT 0xFFFF0005
#define GFPQ_ERR_BAD_PARAMETERS 0xFFFF0006
#define GFPQ_ERR_OUT_OF_MEMORY 0xFFFF000C
#define GFPQ_ERR_SHORT_BUFFER 0xFFFF0010
#define GFPQ_ERR_NOT_SUPPORT 0xFFFF0011

#define GFPQ_INFO_LEN_MAX (128)
#define GFPQ_COMPILER_REVISION (139977)
#define GFPQ_DATE_TIME_STR_LEN (64)
#define GFPQ_PARAM_LEN (16)

/**
 * libgfpq.so -> libgfpq.so.${GFPQ_SOVERSION}
 * libgfpq.so.${GFPQ_SOVERSION} ->
 * libgfpq.so.${GFPQ_VERSION_MAJOR}.${GFPQ_VERSION_MINOR}.${GFPQ_VERSION_PATCH}
 */
typedef struct _GFPQ_INFO_ST {
  unsigned int u32LibVerMajor;              /* GFPQ_VERSION_MAJOR */
  unsigned int u32LibVerMinor;              /* GFPQ_VERSION_MINOR */
  unsigned int u32LibVerPatch;              /* GFPQ_VERSION_PATCH */
  unsigned int u32LibSoVer;                 /* GFPQ_SOVERSION */
  unsigned int u32ComRevision;              /* compiler revision */
  char u8BuildTime[GFPQ_DATE_TIME_STR_LEN]; /* library build time */
} GFPQ_INFO_ST;

typedef struct _GFPQ_PARAM_ST {
  /** GFPQ mode for param.
   * GFPQ_MODE_INIT: There is no valid parameter in param[]. Generate the
   * parameter and filled in param[].
   * GFPQ_MODE_UPDATE: There is parameter in param[]. Generate new parameter,
   * update param[] when the new parameter is better.
   * GFPQ_MODE_APPLY_ONLY: There is parameter in param[]. Don't generate
   * parameter. Just use the param[].
   */
  unsigned int mode;
  unsigned char buf[GFPQ_PARAM_LEN];
} GFPQ_PARAM_ST;

/**
 * @brief Quantization and anti quantization float data.
 * @param [in/out] pf32FloatData: Data buffer for quantization and anti
 * quantization
 * @param [in] u32DataCnt: Data count
 * @param [in] u32BitWidth: Quantization bit width
 * @param [in] pstParam: Quantization param. Set NULL when not need.
 * @return 0 or error code
 */
int HI_GFPQ_QuantAndDeQuant(float *pf32FloatData, unsigned int u32DataCnt,
                            unsigned int u32BitWidth, GFPQ_PARAM_ST *pstParam);

/**
 * @brief Quantization and anti quantization float data with high precision data
 * type.
 * @param [in/out] p64DoubleData: Data buffer for quantization and anti
 * quantization
 * @param [in] u32DataCnt: Data count
 * @param [in] u32BitWidth: Quantization bit width
 * @param [in] pstParam: Quantization param. Set NULL when not need.
 * @return 0 or error code
 */
int HI_GFPQ_QuantAndDeQuant(double *p64DoubleData, unsigned int u32DataCnt,
                            unsigned int u32BitWidth, GFPQ_PARAM_ST *pstParam);

/**
 * @brief Quantization and anti quantization float data via GPU.
 * @param [in/out] pf32FloatData: Data buffer for quantization and anti
 * quantization. It MUST be GPU memery.
 * @param [in] u32DataCnt: Data count
 * @param [in] u32BitWidth: Quantization bit width
 * @param [in] pstParam: Quantization param. Set NULL when not need.
 * @param [in] cuda_stream: CUDA stream handle. Set NULL or not used when not
 * need.
 * @param [in] cublas_handle: Created by cublasCreate(). Set NULL or not used
 * when you not care performace. About 1 millisecond difference.
 * @return 0 or error code
 */
int HI_GFPQ_QuantAndDeQuant_GPU(float *pf32FloatData, unsigned int u32DataCnt,
                                unsigned int u32BitWidth,
                                GFPQ_PARAM_ST *pstParam, void *cuda_stream = 0,
                                void *cublas_handle = 0);

/**
 * @brief Quantization and anti quantization float data with high precision data
 * type via GPU.
 * @param [in/out] p64DoubleData: Data buffer for quantization and anti
 * quantization. It MUST be GPU memery.
 * @param [in] u32DataCnt: Data count
 * @param [in] u32BitWidth: Quantization bit width
 * @param [in] pstParam: Quantization param. Set NULL when not need.
 * @param [in] cuda_stream: CUDA stream handle. Set NULL or not used when not
 * need.
 * @param [in] cublas_handle: Created by cublasCreate(). Set NULL or not used
 * when you not care performace. About 1 millisecond difference.
 * @return 0 or error code
 */
int HI_GFPQ_QuantAndDeQuant_GPU(double *p64DoubleData, unsigned int u32DataCnt,
                                unsigned int u32BitWidth,
                                GFPQ_PARAM_ST *pstParam, void *cuda_stream = 0,
                                void *cublas_handle = 0);

#ifdef __cplusplus
extern "C" {
#endif
/* caffe need to overload function with <float> and <double>, but python not
 * support overload function.
 * So add another interface in extern "C" for python.
 */
int HI_GFPQ_QuantAndDeQuant_PY(float *pf32FloatData, unsigned int u32DataCnt,
                               unsigned int u32BitWidth,
                               GFPQ_PARAM_ST *pstParam);
int HI_GFPQ_QuantAndDeQuant_GPU_PY(float *pf32FloatData,
                                   unsigned int u32DataCnt,
                                   unsigned int u32BitWidth,
                                   GFPQ_PARAM_ST *pstParam, void *cuda_stream,
                                   void *cublas_handle);
#ifdef __cplusplus
}
#endif /* __cplusplus */

/**
 * @brief Get GFPQ library version, compiler revision and build date.
 * @param [in/out] pstInfo: Library information.
 * @return 0 or error code
 */
int HI_GFPQ_GetInfo(GFPQ_INFO_ST *pstInfo);

#endif /* GFPQ_HPP_ */
