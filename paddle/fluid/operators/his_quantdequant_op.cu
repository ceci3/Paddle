/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdlib.h>
#include "paddle/fluid/operators/his_quantdequant_op.h"
#define DATA_RANGE (1024 * 1024)

namespace paddle {
namespace operators {

template <typename T>
struct Hi_GFPQ_QuantAndDeQuantFunctor<paddle::platform::CUDADeviceContext, T> {
  void operator()(T *out, unsigned int data_size, unsigned int bit) {
    GFPQ_PARAM_ST *gfpq_param = NULL;
    int ret = HI_GFPQ_QuantAndDeQuant_GPU(out, data_size, bit, gfpq_param);
    if (ret) {
      LOG(ERROR) << "HI_GFPQ_QuantAndDeQuant_GPU failed";
    }
  }
};

template struct Hi_GFPQ_QuantAndDeQuantFunctor<
    paddle::platform::CUDADeviceContext, float>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(his_quantdequant,
                        ops::HisQuantDequantKernel<CUDA, float>);
