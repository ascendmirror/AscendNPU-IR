//===- InlineLoadCopy.cpp ----- inline copied load ------------------===//
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
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/IR/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_INLINELOADCOPY
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-inline-load-copy"

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct LoadCopyInlinePattern : public OpRewritePattern<hivm::CopyOp> {
  using OpRewritePattern<hivm::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (!copyOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(copyOp,
                                         " op should have buffer semantics.");
    }
    auto src = copyOp.getSrc();
    hivm::LoadOp matchedLoad = nullptr;
    int numberUses = 0;
    for (Operation *user : src.getUsers()) {
      numberUses += 1;
      auto load = dyn_cast<hivm::LoadOp>(user);
      if (!load)
        continue;
      // Ensure the load writes into exactly this buffer.
      if (load.getDst() != src)
        continue;

      matchedLoad = load;
    }
    if (!matchedLoad || numberUses != 2) {
      return rewriter.notifyMatchFailure(
          copyOp, "no LoadOp found that writes into copy src");
    }

    rewriter.replaceOpWithNewOp<hivm::LoadOp>(
        copyOp, TypeRange{}, matchedLoad.getSrc(), copyOp.getDst(),
        matchedLoad.getPadModeAttr(), matchedLoad.getPadValue(),
        matchedLoad.getLeftPaddingNum(), matchedLoad.getRightPaddingNum());
    rewriter.eraseOp(matchedLoad);
    return success();
  }
};

struct InlineLoadCopyPass
    : public impl::InlineLoadCopyBase<InlineLoadCopyPass> {
public:
  void runOnOperation() override;
};
} // namespace

void InlineLoadCopyPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<LoadCopyInlinePattern>(patterns.getContext());
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<Pass> mlir::hivm::createInlineLoadCopyPass() {
  return std::make_unique<InlineLoadCopyPass>();
}
