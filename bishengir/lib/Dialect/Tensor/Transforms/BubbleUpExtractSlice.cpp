//===- BubbleUpExtractSlice.cpp ---------------------------------*- C++ -*-===//
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
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_BUBBLEUPEXTRACTSLICE
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;

namespace {
struct BubbleUpExtractSlicePass
    : public impl::BubbleUpExtractSliceBase<BubbleUpExtractSlicePass> {
  explicit BubbleUpExtractSlicePass(
      const BubbleUpExtractSliceOptions &options)
      : BubbleUpExtractSliceBase(options) {}
  void runOnOperation() override;
};
} // namespace

void BubbleUpExtractSlicePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  RewritePatternSet patterns(funcOp.getContext());
  linalg::BubbleUpExtractSliceOptions linalgOptions;
  linalgOptions.aggressive = this->aggressive;
  linalg::populateBubbleUpExtractSliceOpPatterns(patterns, linalgOptions);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> mlir::tensor::createBubbleUpExtractSlicePass(
    const BubbleUpExtractSliceOptions &options) {
  return std::make_unique<BubbleUpExtractSlicePass>(options);
}
