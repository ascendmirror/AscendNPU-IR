//===----------------- CanonicalizeExt.cpp ------------------*- C++ -*-===//
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

#include "bishengir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgExtensions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "bishengir-canonicalize-ext"

namespace mlir {
#define GEN_PASS_DEF_CANONICALIZER
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

namespace {

using namespace mlir;

} // namespace

struct ExternalCanonicalizer
    : public mlir::impl::CanonicalizerBase<ExternalCanonicalizer> {

  using mlir::impl::CanonicalizerBase<ExternalCanonicalizer>::CanonicalizerBase;

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("canonicalize-ext");
  }
  ::llvm::StringRef getArgument() const final { return "canonicalize-ext"; }

  ::llvm::StringRef getDescription() const final {
    return "Canonicalize operations";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("ExternalCanonicalizer");
  }
  ::llvm::StringRef getName() const final { return "ExternalCanonicalizer"; }

  FrozenRewritePatternSet patterns;

  LogicalResult initialize(MLIRContext *context) final {
    RewritePatternSet patterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, context);

    this->patterns = FrozenRewritePatternSet(std::move(patterns),
                                             disabledPatterns, enabledPatterns);
    return success();
  }

  void runOnOperation() final {
    GreedyRewriteConfig config;
    config.useTopDownTraversal = topDownProcessingEnabled;
    config.enableRegionSimplification = enableRegionSimplification;
    config.maxIterations = maxIterations;
    config.maxNumRewrites = maxNumRewrites;

    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    if (auto converged =
            applyPatternsGreedily(getOperation(), patterns, config);
        testConvergence && failed(converged))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass>
bishengir::createCanonicalizeExtPass(const CanonicalizerOptions &options) {
  return std::make_unique<ExternalCanonicalizer>(options);
}
