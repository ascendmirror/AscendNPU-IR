//===------------- GraphSyncSolver.cpp ---- Graph Sync Solver -----===//
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