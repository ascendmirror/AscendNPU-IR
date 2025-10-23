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

#include "mlir/IR/Value.h"
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

static bool isInLoopBody(Value x, Block *body) {
  if (!body)
    return false;
  if (auto barg = dyn_cast<BlockArgument>(x))
    return barg.getOwner() == body;
  if (auto *defOp = x.getDefiningOp())
    return defOp->getBlock() == body;
  return false;
}

static LogicalResult
traceYieldOperand(Value v, Block *body, BlockArgument allowedIterArg,
                  SmallPtrSetImpl<Operation *> &deletableOps) {
  SmallVector<Value, 8> worklist;
  SmallPtrSet<Value, 16> visitedVals;
  worklist.push_back(v);
  visitedVals.insert(v);

  SmallPtrSet<Operation *, 16> visitedOps;

  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (cur == allowedIterArg)
      continue; // reach only accepted leaf

    Operation *def = cur.getDefiningOp();
    if (!def)
      return failure();

    if (def->getBlock() != body)
      return failure();

    // What if OP has memory effects ?

    visitedOps.insert(def);
    for (Value operand : def->getOperands()) {
      if (!isInLoopBody(operand, body))
        return failure();
      if (visitedVals.insert(operand).second) // avoid double visit
        worklist.push_back(operand);
    }
  }

  deletableOps.insert(visitedOps.begin(), visitedOps.end());
  return success();
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

template <typename LoopT>
struct RemoveDeadIterArgPattern : public OpRewritePattern<LoopT> {
public:
  using OpRewritePattern<LoopT>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(LoopT forOp, mlir::PatternRewriter &rewriter) const override {
    unsigned numResults = forOp.getNumResults();
    if (numResults == 0)
      return failure();
    Block *body = forOp.getBody();
    auto yield = cast<scf::YieldOp>(body->getTerminator());

    SmallVector<unsigned, 4> removableIdxs;
    SmallVector<SmallPtrSet<Operation *, 8>, 4> opsToErasePerIdx;

    for (unsigned i = 0, e = numResults; i < e; ++i) {
      Value res = forOp.getResult(i);
      if (!res.use_empty())
        continue;
      BlockArgument iterArg = body->getArgument(i + 1); // iterarg[i]
      Value yielded = yield.getOperand(i);

      SmallPtrSet<Operation *, 8> deletable;
      if (failed(traceYieldOperand(yielded, body, iterArg, deletable)))
        continue;

      removableIdxs.push_back(i);
      opsToErasePerIdx.emplace_back(std::move(deletable));
    }

    if (removableIdxs.empty())
      return failure();

    // have to rewrite for op with new types
    llvm::SmallBitVector keep(numResults, true);
    for (unsigned idx : removableIdxs)
      keep.reset(idx);
    SmallVector<Type, 4> newResultTypes;
    for (unsigned i = 0; i < numResults; ++i)
      if (keep.test(i))
        newResultTypes.push_back(forOp.getResult(i).getType());
    SmallVector<Value, 4> newInitArgs;
    newInitArgs.reserve(forOp.getInitArgs().size());
    for (auto [i, init] : llvm::enumerate(forOp.getInitArgs()))
      if (keep.test(i))
        newInitArgs.push_back(init);
    rewriter.setInsertionPoint(forOp);
    auto newFor = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange newIterArgs) {
          // Map and Clone block
          IRMapping mapping;
          mapping.map(forOp.getInductionVar(), iv);

          unsigned newIdx = 0;
          for (unsigned i = 0; i < numResults; ++i) {
            if (keep.test(i)) {
              mapping.map(forOp.getRegionIterArgs()[i], newIterArgs[newIdx++]);
            }
          }

          SmallPtrSet<Operation *, 16> deletableAll;
          for (auto &set : opsToErasePerIdx)
            deletableAll.insert(set.begin(), set.end());

          for (Operation &op : forOp.getBody()->without_terminator()) {
            if (deletableAll.contains(&op))
              continue;
            b.clone(op, mapping);
          }

          auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
          SmallVector<Value, 4> newYieldOperands;
          newYieldOperands.reserve(newResultTypes.size());
          for (unsigned i = 0; i < numResults; ++i) {
            if (!keep.test(i))
              continue;
            Value mapped = mapping.lookupOrNull(oldYield.getOperand(i));
            if (!mapped)
              mapped = oldYield.getOperand(i); // safe fallback
            newYieldOperands.push_back(mapped);
          }
          b.create<scf::YieldOp>(loc, newYieldOperands);
        });
    SmallVector<Value, 4> newResults(newFor.getResults().begin(),
                                     newFor.getResults().end());
    unsigned newIdx = 0;
    for (unsigned i = 0; i < numResults; ++i) {
      if (keep.test(i)) {
        forOp.getResult(i).replaceAllUsesWith(newResults[newIdx++]);
      }
    }

    rewriter.eraseOp(forOp);
    return success();
  }
};

struct CanonicalizeIterArgPass
    : public impl::CanonicalizeIterArgBase<CanonicalizeIterArgPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<CanonicalizeIterArgPattern<scf::ForOp>,
                    CanonicalizeIterArgPattern<scf::WhileOp>,
                    RemoveDeadIterArgPattern<scf::ForOp>>(
        patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> scf::createCanonicalizeIterArgPass() {
  return std::make_unique<CanonicalizeIterArgPass>();
}
