//===- Passes.td - HIVM pass definition file ---------------*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertToHIVMOp : Pass<"convert-to-hivm-op"> {
  let summary = "Convert Ops from other dialects to HIVM Ops";
  let constructor = "mlir::hivm::createConvertToHIVMOpPass()";
  let dependentDialects = ["arith::ArithDialect", "hivm::HIVMDialect",
                           "tensor::TensorDialect",
  ];
}

def TritonGlobalKernelArgsToHIVMOp
    : Pass<"triton-global-kernel-args-to-hivm-op", "func::FuncOp"> {
  let summary = "";
  let constructor = "mlir::hivm::createTritonGlobalKernelArgsToHIVMOpPass()";
  let dependentDialects = ["hivm::HIVMDialect", "hacc::HACCDialect", "annotation::AnnotationDialect"];
}

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_PASSES_TD
