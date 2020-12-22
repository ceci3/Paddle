//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <sstream>
#include "paddle/fluid/operators/his_quantdequant_op.h"

#include "gtest/gtest.h"
#include "paddle/fluid/framework/dim.h"
#define DATA_RANGE (1024 * 1024)

TEST(Dim, Equality) {
  float *data_buf = NULL;
  unsigned int mem_size = 1024 * 1024 * 50;
  unsigned int data_num = mem_size / sizeof(float);
  float *data_run = NULL;
  GFPQ_PARAM_ST *pstParam = NULL;
  int bit_width = GFPQ_BIT_WIDTH_8BIT;
  data_buf = (float *)malloc(mem_size);

  for (int i = 0; i < data_num; i++) {
    data_buf[i] =
        ((float)(rand() % DATA_RANGE) - DATA_RANGE / 2) / (DATA_RANGE / 4);
  }
  cudaMalloc((void **)&data_run, data_num * sizeof(float));
  cudaMemcpy(data_run, &data_buf[0], data_num * sizeof(float),
             cudaMemcpyHostToDevice);

  int ret =
      HI_GFPQ_QuantAndDeQuant_GPU(data_run, data_num, bit_width, pstParam);
  LOG(ERROR) << ret;
}
