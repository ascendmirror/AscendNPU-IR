//===------------- GraphSyncSolver.cpp ---- Graph Sync Solver -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolver.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverTester.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"
#include <cstring>

#define DEBUG_TYPE "hivm-graph-sync-solver"

namespace mlir {
#define GEN_PASS_DEF_GRAPHSYNCSOLVER
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm::syncsolver;

namespace mlir {
struct GraphSyncSolverPass
    : public impl::GraphSyncSolverBase<GraphSyncSolverPass> {
  explicit GraphSyncSolverPass(const GraphSyncSolverOptions &options)
      : GraphSyncSolverBase(options) {}

public:
  void runOnOperation() override;
};
} // namespace mlir

void GraphSyncSolverPass::runOnOperation() {
  if (SyncTester::runTestMode()) {
    return;
  }

  auto func = getOperation();
  if (hacc::utils::isHost(func))
    return;

  Solver solver(func);
  solver.enableUnitFlagFeature = this->enableUnitFlag;
  LLVM_DEBUG(llvm::dbgs() << "before:\n"
                          << solver.funcIr->str(0, true) << '\n';);

  LLVM_DEBUG({
    for (auto &occ : solver.syncIr) {
      llvm::dbgs() << std::string(occ->depth, ' ') << occ->op->id << ' '
                   << occ->syncIrIndex << ' ' << occ->startIndex << ' '
                   << occ->endIndex << '\n';
      llvm::dbgs() << occ->op->str(occ->depth, false) << '\n';
    }
  });

  solver.solve();

  LLVM_DEBUG({
    solver.generateFuncIrResultOps();
    llvm::dbgs() << "after:\n" << solver.funcIr->str(0, true) << '\n';
  });

  solver.generateResultOps();
}

std::unique_ptr<Pass>
mlir::hivm::createGraphSyncSolverPass(const GraphSyncSolverOptions &options) {
  return std::make_unique<GraphSyncSolverPass>(options);
}