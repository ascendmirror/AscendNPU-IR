//===- OutlineScope.cpp --------- Outline Scope Pass ----------------------===//
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
// This file implements a pass to propagate scopes
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "outline-scope"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace scope {
#define GEN_PASS_DEF_OUTLINESCOPE
#include "bishengir/Dialect/Scope/Transforms/Passes.h.inc"

namespace {
class OutlineScopePass : public impl::OutlineScopeBase<OutlineScopePass> {
public:
  explicit OutlineScopePass() : OutlineScopeBase() {}
  void runOnOperation() final;
};

// create init scopeic_int and bind_scopeic_shape for func arguments
void initScopeForFuncArgs(func::FuncOp func) {}
} // namespace

void OutlineScopePass::runOnOperation() {
  func::FuncOp func = getOperation();
  initScopeForFuncArgs(func);
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);

  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createOutlineScopePass() {
  return std::make_unique<OutlineScopePass>();
}

} // namespace scope
} // namespace mlir