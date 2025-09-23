//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
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
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_PASSES_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Pass/Pass.h"
#include <memory>

/// Defines a scope for reinterpret map pass.
enum class MultiBufferStrategy {
  NO_LIMIT = 0,
  ONLY_CUBE,
  ONLY_VECTOR,
  CUBE_NO_L0C,
};

namespace mlir {

namespace hivm {

enum class SyncMode {
  NORMAL,
  BARRIERALL, // only for debug
};

} // namespace hivm
} // namespace mlir

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

namespace hivm {

/// Create a pass to convert ops from other dialects to HIVM Ops.
std::unique_ptr<Pass> createConvertToHIVMOpPass();

/// Create a pass to convert args of global kernel function to HIVM Ops.
std::unique_ptr<Pass> createTritonGlobalKernelArgsToHIVMOpPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_PASSES_H
