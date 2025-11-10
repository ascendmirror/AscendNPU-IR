//===- MoveUpAffineMap.cpp ------------------------------------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/MoveUpAffine.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "move-up-affine"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::hivm::detail {

// We need this move up affine patterns because
// when we bubble up extract slice, extract slice might get bubbled
// higher than it's defining values.
// For example:
// for {
//   xxx
//   xxx
//   %0 = affine.apply
//   extract_slice [%0]
// }
// ->
// for {
//   extract_slice [%0]
//   xxx
//   xxx
//   %0 = affine.apply
// }
// so we need to move up the affine apply too
// ->
// for {
//   %0 = affine.apply
//   extract_slice [%0]
//   xxx
//   xxx
// }
template <typename AffineOpTy>
struct MoveUpAffine : OpRewritePattern<AffineOpTy> {
  using OpRewritePattern<AffineOpTy>::OpRewritePattern;

  constexpr static llvm::StringLiteral kHoistedAttrName = "hoisted";

  explicit MoveUpAffine(MLIRContext *ctx, PatternBenefit benefit = 100)
      : OpRewritePattern<AffineOpTy>(ctx, benefit) {};

  LogicalResult matchAndRewrite(AffineOpTy op,
                                PatternRewriter &rewriter) const final {
    auto blocks = llvm::map_to_vector(
        op->getOperands(), [](Value opr) { return opr.getParentBlock(); });

    auto anchorBlock = blocks.front();
    if (llvm::any_of(blocks,
                     [&anchorBlock](Block *b) { return anchorBlock != b; }))
      return rewriter.notifyMatchFailure(
          op, "not all operands are in the same block");

    // Get the lowest dominating operation
    Operation *lastDefOp = nullptr;
    for (Value operand : op->getOperands()) {
      if (auto defOp = operand.getDefiningOp()) {
        if (!lastDefOp || lastDefOp->isBeforeInBlock(defOp))
          lastDefOp = defOp;
      }
    }

    // If the `lastDefOp` is null, it means that the lowest dominating value is
    // a block argument. So set the insertion point to the front of the block.
    Operation *insertPoint = &anchorBlock->front();
    if (lastDefOp)
      insertPoint = lastDefOp->getNextNode();

    func::FuncOp enclosingFunc = op->template getParentOfType<func::FuncOp>();
    if (!enclosingFunc)
      return failure();

    DominanceInfo domInfo(enclosingFunc);
    if (anchorBlock == op->getBlock() &&
        domInfo.dominates((Operation *)op, insertPoint))
      return rewriter.notifyMatchFailure(
          op, "can not be moved up to a higher position");

    rewriter.moveOpBefore(op, insertPoint);
    return success();
  }
};

void populateMoveUpAffinePattern(RewritePatternSet &patterns) {
  patterns.add<MoveUpAffine<affine::AffineApplyOp>,
               MoveUpAffine<affine::AffineMinOp>,
               MoveUpAffine<affine::AffineMaxOp>>(patterns.getContext());
}

} // namespace mlir::hivm::detail