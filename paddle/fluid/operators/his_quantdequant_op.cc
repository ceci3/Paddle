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

#include "paddle/fluid/operators/his_quantdequant_op.h"

namespace paddle {
namespace operators {

template <typename T>
struct Hi_GFPQ_QuantAndDeQuantFunctor<paddle::platform::CPUDeviceContext, T> {
  void operator()(T *out, unsigned int data_size, unsigned int bit) {
    HI_GFPQ_QuantAndDeQuant(out, data_size, bit, NULL);
  }
};

template struct Hi_GFPQ_QuantAndDeQuantFunctor<
    paddle::platform::CPUDeviceContext, float>;

class HisQuantDequantOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "HisQuantDequant");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "HisQuantDequant");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class HisQuantDequantOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddOutput("Out", "(Tensor) Output is float data type.");
    AddAttr<int>("bit_length", "(int, default 8)").SetDefault(8);
    AddComment(R"DOC(
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;
REGISTER_OPERATOR(
    his_quantdequant, ops::HisQuantDequantOp, ops::HisQuantDequantOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(his_quantdequant,
                       ops::HisQuantDequantKernel<CPU, float>);
// int a = TouchOpRegistrar_his_quantdequant();
