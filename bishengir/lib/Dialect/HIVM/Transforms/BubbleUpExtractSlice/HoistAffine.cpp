//===- HoistAffine.cpp ----------------------------------------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/HoistAffine.h"

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
struct HoistAffine : OpRewritePattern<AffineOpTy> {
  using OpRewritePattern<AffineOpTy>::OpRewritePattern;

  explicit HoistAffine(MLIRContext *ctx, PatternBenefit benefit = 100)
      : OpRewritePattern<AffineOpTy>(ctx, benefit) {};

  LogicalResult matchAndRewrite(AffineOpTy op,
                                PatternRewriter &rewriter) const final {
    auto operandBlocks = llvm::map_to_vector(
        op->getOperands(), [](Value opr) { return opr.getParentBlock(); });

    auto anchorBlock = operandBlocks.front();
    if (llvm::any_of(operandBlocks,
                     [&anchorBlock](Block *b) { return anchorBlock != b; }))
      return rewriter.notifyMatchFailure(
          op, "not all operands are in the same block");

    // Get the lowest dominating operation in the block.
    Operation *lastDepOp = nullptr;
    Value lastDepVal = Value();
    for (Value operand : op->getOperands()) {
      if (auto ba = dyn_cast<BlockArgument>(operand)) {
        lastDepVal = ba;
        continue;
      }
      auto definingOp = operand.getDefiningOp();
      assert(definingOp);
      if (!lastDepOp || lastDepOp->isBeforeInBlock(definingOp)) {
        lastDepOp = definingOp;
        lastDepVal = operand;
      }
    }

    // If the `lastDepOp` is null, it means that the lowest dominating value is
    // a block argument. So set the insertion point to the front of the block.
    Operation *insertPoint =
        lastDepOp ? lastDepOp->getNextNode() : &anchorBlock->front();

    if (insertPoint->getBlock() != op->getBlock()) {
      rewriter.moveOpBefore(op, insertPoint);
      return success();
    }

    // Adjust insertion point down to avoid oscillation.
    // For example:
    //
    // ```mlir
    // %def = some_op
    // first_use(%def)
    // second_use(%def)
    // ...
    // third_use(%def)
    // ```
    // If we matched the "second_use", the `insertPoint` will be "first_use",
    // and vice versa. Because there is no domination relationship between the
    // two.
    // We can break the tie by moving the insertion point to end of the
    // consecutive chain of users.
    auto lastDepValUser = SetVector<Operation *>{lastDepVal.getUsers().begin(),
                                                 lastDepVal.getUsers().end()};
    while (insertPoint && lastDepValUser.contains(insertPoint))
      insertPoint = insertPoint->getNextNode();

    if (!insertPoint || insertPoint->getBlock() != op->getBlock())
      return rewriter.notifyMatchFailure(op, "invalid insertion point");

    if (insertPoint == op || op->isBeforeInBlock(insertPoint))
      return rewriter.notifyMatchFailure(
          op, "op cannot be moved to a higher place in block");

    rewriter.moveOpBefore(op, insertPoint);
    return success();
  }
};

void populateHoistAffinePattern(RewritePatternSet &patterns) {
  patterns
      .add<HoistAffine<affine::AffineApplyOp>, HoistAffine<affine::AffineMinOp>,
           HoistAffine<affine::AffineMaxOp>>(patterns.getContext());
}

} // namespace mlir::hivm::detail