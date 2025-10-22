//===--------- ForwardUnitMemory.cpp - Forward unit memref opt ------------===//
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

#include "bishengir/Dialect/MemRef/Transforms/Passes.h"

#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/Utils/ValueDependencyAnalyzer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/IR/CFG.h"

#define DEBUG_TYPE "memref-forward-unit-memory"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_FORWARDUNITMEMORY
#include "bishengir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct MemrefForwardUnitMemory
    : public impl::ForwardUnitMemoryBase<MemrefForwardUnitMemory> {
  using Base::Base;
  void runOnOperation() override;
};

class ForwardUnitStore : public mlir::OpRewritePattern<memref::StoreOp> {
public:
  explicit ForwardUnitStore(MLIRContext *context)
      : OpRewritePattern<memref::StoreOp>(context) {}

  LogicalResult matchAndRewrite(memref::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto memoryAllocated = storeOp.getMemRef();
    if (!utils::isScalarLike(memoryAllocated))
      return rewriter.notifyMatchFailure(storeOp,
                                         "Memory from store is not scalar");

    auto *parentRegion = storeOp.getOperation()->getParentRegion();
    if (!parentRegion)
      return rewriter.notifyMatchFailure(storeOp, "Parent of storeOp is null");
    DominanceInfo info(storeOp->getParentOp());

    // This checks if all the usages of the memref of the store is read only
    // (Except the store in the beginning)
    // Store is a write to mem, check how many read and write after this
    auto allocatedMemoryUses = llvm::map_to_vector(memoryAllocated.getUses(),
                                    [](OpOperand &usage) { return &usage; });
    for (auto &usage : allocatedMemoryUses) {
      if (usage->getOwner() == storeOp)
        continue;
      auto memoryModificationOp =
          dyn_cast_if_present<MemoryEffectOpInterface>(usage->getOwner());
      LDBG("Find allocated memory uses: " << *usage->getOwner());
      if (!memoryModificationOp)
        return rewriter.notifyMatchFailure(
            memoryModificationOp,
            "Memory modification op doesn't exist");
      auto writeEffect = memoryModificationOp.getEffectOnValue<
        MemoryEffects::Write>(memoryAllocated);
      if (writeEffect.has_value())
        return rewriter.notifyMatchFailure(
            memoryModificationOp,
            "store's memref is not all read");
      auto *readOp = usage->getOwner();
      if (readOp->getParentRegion() != parentRegion)
        return rewriter.notifyMatchFailure(readOp,
                                           "read and store parent is different");

      // Check whether the write dominate the read
      if (!info.dominates(storeOp, readOp)) {
        return rewriter.notifyMatchFailure(
            readOp, "store doesn't dominate reads");
      }
    }
    // Only will be catching one read besides storeOp
    // Check if the store op was loaded from somewhere
    // Check if there's a writeOp to that too until here
    auto valueStored = storeOp.getValueToStore();

    // storedValueOp is a loadOp that loads from a memory
    // --- loadSource anything is ok here
    // %storedValueOp = memref.load %loadSource
    // --- Anything in this area write is not ok for the loadSource
    // memref.store %storedValueOp, %memoryAllocated[0] // The only write
    // --- Anything in this area write is not ok on the loadSource
    // --- also the memoryAllocated
    // %memoryAllocated[0] -> read
    // %memoryAllocated[0] -> read
    // %memoryAllocated[0] -> read
    if (auto storedValueOp =
            dyn_cast_if_present<memref::LoadOp>(valueStored.getDefiningOp())) {
      auto loadSource = storedValueOp.getMemRef();
      if (!utils::isScalarLike(loadSource))
        return rewriter.notifyMatchFailure(storedValueOp,
                                           "previous load is not scalar");
      // TODO: How to make sure this memory doesnt have any view like aliases??
      // Check if there's any write in this load

      auto loadSourceUsage = llvm::map_to_vector(
          loadSource.getUses(), [](OpOperand &usage) { return &usage; });

      // If it's not, make sure all write is done before this usage
      for (auto &usage : loadSourceUsage) {
        auto *potentialWriteOp = usage->getOwner();
        auto memoryModificationOp =
            dyn_cast_if_present<MemoryEffectOpInterface>(potentialWriteOp);
        LDBG("Find load source usage: " << *usage->getOwner());
        if (!memoryModificationOp) continue;
        auto writeEffect =
            memoryModificationOp.getEffectOnValue<MemoryEffects::Write>(
                memoryAllocated);
        if (!writeEffect)
          continue;
        if (potentialWriteOp->getParentRegion() != parentRegion) {
          return rewriter.notifyMatchFailure(
              potentialWriteOp,
              "potential write doesn't have the same parent region");
        }
        if (info.dominates(storedValueOp.getOperation(), potentialWriteOp)) {
          return rewriter.notifyMatchFailure(
              potentialWriteOp,
              "there is a write on after the value acquirement");
        }
      }
      // Safe!
      bool replaced = false;
      for (auto &usage : allocatedMemoryUses) {
        if (usage->getOwner() == storeOp)
          continue;
        replaced = true;
        usage->assign(storedValueOp.getMemRef());
      }
      return success(replaced);
    }
    return rewriter.notifyMatchFailure(
        storeOp, "Value to store definition not a loadOp");
  }
};

} // namespace

void MemrefForwardUnitMemory::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<ForwardUnitStore>(ctx);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
    return signalPassFailure();
  IRRewriter rewriter(ctx);
  LDBG("func propagated\n" << *funcOp << "\n");
  memref::eraseDeadAllocAndStores(rewriter, funcOp);
}

std::unique_ptr<Pass> memref::createForwardUnitMemoryPass() {
  return std::make_unique<MemrefForwardUnitMemory>();
}