//===--- CanonicalizeIterArg.cpp - Eliminate unused iter args -------------===//
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

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "bishengir/Dialect/SCF/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_CANONICALIZEITERARG
#include "bishengir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "scf-canonicalize-iter-arg"

using namespace mlir;

/// "Unchanged" in this context means it is not modified before being passed
/// back to the yield. Here we check specifically for nested scf structures.
static bool isIterArgUnchanged(Value yielded, BlockArgument iterArg,
                               SetVector<Value> &possibleInitAlias) {
  possibleInitAlias.insert(yielded);

  if (yielded == iterArg)
    return true;

  auto loop =
      dyn_cast<LoopLikeOpInterface>(iterArg.getParentBlock()->getParentOp());
  assert(loop && "expecting iterarg to be block argument of loop-like op");
  Value tiedInit = loop.getTiedLoopInit(iterArg)->get();
  if (tiedInit == yielded)
    return true;

  auto res = dyn_cast<OpResult>(yielded);
  // Don't think block argument will be valid in this case
  if (!res)
    return false;
  unsigned resNo = res.getResultNumber();
  Operation *defining = res.getOwner();
  // The yielded value is different than init value at first glance, value is
  // defined outside the loop, but is a different than the init value.
  if (!loop->isAncestor(defining))
    return false;

  // For IfOps, it is "unchanged" if both its yielded value are the same value
  if (auto ifOp = dyn_cast<scf::IfOp>(defining)) {
    Value thenYieldVal = ifOp.thenYield().getOperand(resNo);
    // Since ifOp has a result, it must also have an else block
    Value elseYieldVal = ifOp.elseYield().getOperand(resNo);
    return isIterArgUnchanged(thenYieldVal, iterArg, possibleInitAlias) &&
           isIterArgUnchanged(elseYieldVal, iterArg, possibleInitAlias);
  }

  // ForOps doesn't change the iterarg on each iteration if itself doesn't
  // change its corresponding iterArg, also if its init value is the same as
  // the iter arg
  if (auto innerLoop = dyn_cast<LoopLikeOpInterface>(defining)) {
    return isIterArgUnchanged(innerLoop.getInits()[resNo], iterArg,
                              possibleInitAlias) &&
           isIterArgUnchanged(innerLoop.getYieldedValues()[resNo],
                              innerLoop.getRegionIterArgs()[resNo],
                              possibleInitAlias);
  }
  // We don't check other cases... for now (tm)
  return false;
}

namespace {

template <typename LoopT>
struct CanonicalizeIterArgPattern : public OpRewritePattern<LoopT> {
public:
  using OpRewritePattern<LoopT>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(LoopT op, mlir::PatternRewriter &rewriter) const override {
    bool changed = false;
    SetVector<Value> possibleInitAlias;
    for (BlockArgument arg : op.getRegionIterArgs()) {
      Value yieldVal = op.getTiedLoopYieldedValue(arg)->get();
      Value initVal = op.getTiedLoopInit(arg)->get();
      Value resultVal = op.getTiedLoopResult(arg);
      // Additional check to make sure we didn't clean this already
      if (yieldVal != initVal &&
          isIterArgUnchanged(yieldVal, arg, possibleInitAlias)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Matched " << yieldVal << "\n\tas unchanged\n\n");
        while (!possibleInitAlias.empty()) {
          Value alias = possibleInitAlias.pop_back_val();
          if (alias != initVal)
            alias.replaceAllUsesWith(initVal);
        }
        resultVal.replaceAllUsesWith(initVal);
        changed = true;
      }
      possibleInitAlias.clear();
    }
    return success(changed);
  }
};

struct CanonicalizeIterArgPass
    : public impl::CanonicalizeIterArgBase<CanonicalizeIterArgPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<CanonicalizeIterArgPattern<scf::ForOp>,
                    CanonicalizeIterArgPattern<scf::WhileOp>>(
        patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> scf::createCanonicalizeIterArgPass() {
  return std::make_unique<CanonicalizeIterArgPass>();
}
