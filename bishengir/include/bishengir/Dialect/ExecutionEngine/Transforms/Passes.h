//===- Passes.h - Execution Engine pass entrypoints -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_EXECUTION_ENGINE_TRANSFORMS_PASSES_H
#define BISHENGIR_EXECUTION_ENGINE_TRANSFORMS_PASSES_H

#include "bishengir/Dialect/ExecutionEngine/IR/ExecutionEngine.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL
#include "bishengir/Dialect/ExecutionEngine/Transforms/Passes.h.inc"

namespace execution_engine {

/// Create a pass to create wrappers for the only host related functions.
std::unique_ptr<Pass> createCreateHostMainPass(
    const ExecutionEngineHostMainCreatorOptions &options = {});

/// Create a pass to convert HIVM operations to upstream dialect's equivalent.
std::unique_ptr<Pass> createConvertHIVMToUpstreamPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/ExecutionEngine/Transforms/Passes.h.inc"

} // namespace execution_engine
} // namespace mlir

#endif // BISHENGIR_EXECUTION_ENGINE_TRANSFORMS_PASSES_H
