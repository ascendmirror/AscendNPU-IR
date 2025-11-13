//===- ReplicateOutEmptyTensor.cpp ---- Clone Tensor Empty Pass -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_REPLICATEOUTEMPTYTENSOR
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::tensor;
namespace mlir::tensor::detail {

struct ReplicateEmptyOutPattern : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp op,
                                PatternRewriter &rewriter) const override {

    if (op->hasOneUse() || op->use_empty())
      return failure();
    SmallVector<OpOperand *> uses;
    for (OpOperand &user : op.getResult().getUses()) {
      Operation *userOp = user.getOwner();
      if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(userOp)) {
        // Go check dps inits
        if (dpsOp.isDpsInit(&user)) {
          uses.push_back(&user);
        }
      }
    }
    // Matching here, check only inits of users
    if (uses.size() <= 1)
      return failure();
    for (OpOperand *user : uses) {
      Operation *clonedEmptyOp = rewriter.clone(*op);
      rewriter.modifyOpInPlace(op, [&clonedEmptyOp, &user]() {
        user->set(cast<tensor::EmptyOp>(clonedEmptyOp).getResult());
      });
    }
    return success();
  }
};
} // namespace mlir::tensor::detail

struct ReplicateOutEmptyTensorPass
    : public impl::ReplicateOutEmptyTensorBase<ReplicateOutEmptyTensorPass> {
public:
  void runOnOperation() override;
};

void ReplicateOutEmptyTensorPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.add<mlir::tensor::detail::ReplicateEmptyOutPattern>(context);
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<Pass> mlir::tensor::createReplicateOutEmptyTensorPass() {
  return std::make_unique<ReplicateOutEmptyTensorPass>();
}