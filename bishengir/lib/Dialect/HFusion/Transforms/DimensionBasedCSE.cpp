//===- DimensionBasedCSE.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_DIMENSIONBASEDCSE
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hfusion-dim-cse"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir::hfusion::detail {
namespace {

static bool isEqual(Operation *lhs, Operation *rhs) {
  return OperationEquivalence::isEquivalentTo(
      lhs, rhs, OperationEquivalence::IgnoreLocations);
}
enum class JoinResult { Match, NoMatch, Fail };

class CSEDimensionAnalyzer {
public:
  explicit CSEDimensionAnalyzer(func::FuncOp func) : analyzer(func) {
    (void)analyzer.initialize();
    for (auto &val : func.getArguments()) {
      processValue(val);
    }
    func.walk([&](Operation *op) {
      for (auto res : op->getResults())
        processValue(res);
    });
  }

  void processValue(Value val) {
    if (!analyzer.existArgumentRef(val))
      return;
    auto tensorType = dyn_cast_if_present<RankedTensorType>(val.getType());
    if (!tensorType)
      return;
    auto argRef = analyzer.getArgumentRef(val);
    for (auto &el : argRef) {
      el = analyzer.getShapeElementParent(el);
    }
    for (auto &u : argRef) {
      auto &curElementMap = exclusiveElements[u];
      for (auto &v : argRef) {
        if (u == v)
          continue;
        LDBG("Adding pair " << u << " " << v);
        curElementMap[v]++;
      }
    }
  }

  int getExclusiveParent(int el) {
    if (exclusiveParent.count(el))
      return exclusiveParent[el];
    return exclusiveParent[el] = el;
  }

  JoinResult checkValidity(Value valA, Value valB) {
    if (!analyzer.existArgumentRef(valA))
      return JoinResult::NoMatch;
    if (!analyzer.existArgumentRef(valB))
      return JoinResult::NoMatch;
    auto refA = analyzer.getArgumentRef(valA);
    auto refB = analyzer.getArgumentRef(valB);
    if (refA.size() != refB.size())
      return JoinResult::NoMatch;
    for (const auto &[idx, elA, elB] : llvm::enumerate(refA, refB)) {
      auto parentA = getExclusiveParent(analyzer.getShapeElementParent(elA));
      auto parentB = getExclusiveParent(analyzer.getShapeElementParent(elB));
      LDBG("relationship check: " << parentA << " " << parentB);
      if (exclusiveElements[parentA].count(parentB)) {
        // This doesn't guarantee that there are exclusive that is not inferred
        // while merging
        return JoinResult::NoMatch;
      }
      LDBG("Safe");
    }
    return JoinResult::Match;
  }
  JoinResult join(Value valA, Value valB) {
    auto isValid = checkValidity(valA, valB);
    if (isValid != JoinResult::Match)
      return isValid;
    auto refA = analyzer.getArgumentRef(valA);
    auto refB = analyzer.getArgumentRef(valB);
    LDBG("Merging " << valA << " " << valB);
    for (const auto &[idx, elA, elB] : llvm::enumerate(refA, refB)) {
      int shapeElA = analyzer.getShapeElementParent(elA);
      int shapeElB = analyzer.getShapeElementParent(elB);
      if (shapeElA == shapeElB)
        continue;
      int exA = getExclusiveParent(shapeElA);
      int exB = getExclusiveParent(shapeElB);
      if (exclusiveElements[exA].count(exB)) {
        LDBG("Rollback detected");
        // This case needs rollback
        return JoinResult::Fail;
      }
      LDBG("Safe to join, joining " << shapeElA << " " << shapeElB);
      analyzer.joinShape(shapeElA, shapeElB);
      // which one is bigger
      if (exclusiveElements[exA].size() > exclusiveElements[exB].size()) {
        std::swap(exA, exB);
      }
      // exA must be smaller, exB is the new parent
      for (auto [v, amount] : exclusiveElements[exA]) {
        auto parV = getExclusiveParent(v);
        assert(parV == v);
        if (exclusiveElements[parV].count(exB)) {
          assert(exclusiveElements[parV].count(exA));
          exclusiveElements[parV][exB] += exclusiveElements[parV][exA];
          exclusiveElements[parV].erase(exA);
        }
        exclusiveElements[exB][parV] += amount;
      }
    }
    return JoinResult::Match;
  }

  void markDimension() {}

private:
  DimensionAnalyzer analyzer;
  DenseMap<int64_t, DenseMap<int64_t, int64_t>> exclusiveElements;
  DenseMap<int64_t, int64_t> exclusiveParent;
};

struct DimensionBasedCSEPass
    : public impl::DimensionBasedCSEBase<DimensionBasedCSEPass> {
  /// Main entry point for the pass
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    CSEDimensionAnalyzer analyzer(funcOp);
    SmallVector<SmallVector<Operation *>> groups;
    auto result = funcOp.walk<WalkOrder::PreOrder>(
        [&](tensor::EmptyOp emptyOp) -> WalkResult {
          bool assignedGroup = false;
          for (auto &group : groups) {
            if (isEqual(emptyOp, group.front())) {
              LDBG("Checking equality: " << emptyOp << " " << *group.front());
              JoinResult res =
                  analyzer.join(emptyOp, cast<tensor::EmptyOp>(group.front()));
              if (res == JoinResult::Fail)
                return WalkResult::interrupt();
              if (res == JoinResult::NoMatch)
                continue;
              group.push_back(emptyOp);
              assignedGroup = true;
              break;
            }
          }
          if (!assignedGroup) {
            groups.push_back({emptyOp});
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      return;
    // CSE all groups
    LDBG("Has " << groups.size() << " worklist");
    for (auto &group : groups) {
      LDBG("CSE-ing group with size " << group.size());
      LDBG(group.size());

      for (size_t i = 1; i < group.size(); ++i) {
        group[i]->replaceAllUsesWith(group.front());
        group[i]->erase();
      }
    }
  }
};
} // namespace
} // namespace mlir::hfusion::detail

std::unique_ptr<Pass> mlir::hfusion::createDimensionBasedCSEPass() {
  return std::make_unique<detail::DimensionBasedCSEPass>();
}
