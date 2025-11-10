//===- HIVMBubbleUpExtractSlice.cpp - Bubble Up ExtractSliceOp on HIVM ops ===//
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
//============================================================================//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/CSEPattern.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/MoveUpAffine.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/Pattern.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Transforms/Passes.h"
#include "bishengir/Transforms/Transforms.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
#define GEN_PASS_DEF_HIVMBUBBLEUPEXTRACTSLICE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-bubble-up-extract-slice"

namespace {

using namespace mlir::hivm::detail;
class HIVMBubbleUpExtractSlicePass
    : public impl::HIVMBubbleUpExtractSliceBase<HIVMBubbleUpExtractSlicePass> {
public:
  using Base::Base;

  static bool traceAndCheckIsGM(Value value) {
    return !traceDefOp<memref::AllocOp>(value).has_value();
  }

 // If the bubble-up-ed extract slice op was "stuck" in the middle, return
 // failure.
  LogicalResult
  verifyMarkedExtractSlicesAreBubbledUp(func::FuncOp funcOp) const {
    auto walkResult = funcOp->walk([&](tensor::ExtractSliceOp extractSliceOp) {
      if (!extractSliceOp->hasAttrOfType<UnitAttr>("bubbled_slice"))
        return WalkResult::advance();

      auto extractSrc = extractSliceOp->getOperand(0);
      if (isa<BlockArgument>(extractSrc))
        return WalkResult::advance();

      if (!isa<bufferization::ToTensorOp, tensor::EmptyOp>(
              extractSrc.getDefiningOp())) {
        extractSliceOp.emitOpError() << "not bubbled up";
        return WalkResult::interrupt();
       }
       return WalkResult::advance();
     });

    return failure(walkResult.wasInterrupted());
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    GreedyRewriteConfig config;
    BubbleUpListener listener = BubbleUpListener();
    config.maxIterations = 50;
    config.listener = &listener;

    bool isChanged;
    int64_t iterationCount = 0;
    int64_t maxIterations = 5;
    do {
      isChanged = false;
      // Apply canonicalization
      PassManager pm(funcOp->getContext());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      if (failed(pm.run(funcOp)))
        return signalPassFailure();

      // Apply bubble up patterns
      RewritePatternSet patterns(funcOp.getContext());
      populateHoistAffinePattern(patterns);
      populateBubbleUpExtractSliceOpPatterns(patterns);
      populateCSEPattern(patterns);
      tensor::populateFoldTensorEmptyPatterns(patterns,
                                              /*foldSingleUseOnly=*/true);
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns),
                                              config, &isChanged)))
        return signalPassFailure();

      iterationCount++;
    } while (isChanged && iterationCount < maxIterations);

    if (failed(verifyMarkedExtractSlicesAreBubbledUp(funcOp))) {
      funcOp.emitOpError("some slice ops were not bubbled up");
       return signalPassFailure();
     }
   }

private:
  void
  populateBubbleUpExtractSliceOpPatterns(RewritePatternSet &patterns) const {
    auto *context = patterns.getContext();

    // Create strategies
    SmallVector<std::shared_ptr<BubbleUpStrategy>> strategies;
    strategies.push_back(std::make_shared<BroadcastBubbleUpStrategy>());
    strategies.push_back(std::make_shared<ReduceBubbleUpStrategy>());
    strategies.push_back(std::make_shared<ExpandBubbleUpStrategy>());
    strategies.push_back(std::make_shared<CollapseBubbleUpStrategy>());
    strategies.push_back(std::make_shared<ElementwiseBubbleUpStrategy>());
    strategies.push_back(std::make_shared<LoopBubbleUpStrategy>());
    strategies.push_back(std::make_shared<ExtractSliceBubbleUpStrategy>());
    strategies.push_back(std::make_shared<InsertSliceBubbleUpStrategy>());

    // Add pattern with strategies
    patterns.add<BubbleUpPattern>(context, std::move(strategies));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createHIVMBubbleUpExtractSlicePass() {
  return std::make_unique<HIVMBubbleUpExtractSlicePass>();
}
