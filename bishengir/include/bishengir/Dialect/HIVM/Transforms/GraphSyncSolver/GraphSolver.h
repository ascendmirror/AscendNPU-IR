//===------------- GraphSolver.h ---- Graph Sync Solver -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENG_DIALECT_HIVM_GRAPHSOLVER_GRAPHSOLVER_H
#define BISHENG_DIALECT_HIVM_GRAPHSOLVER_GRAPHSOLVER_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"
#include <set>

namespace mlir::hivm::syncsolver {

class GraphSolver {
public:
  struct Edge {
    mlir::hivm::PIPE pipeFromId{mlir::hivm::PIPE::PIPE_UNASSIGNED};
    mlir::hivm::PIPE pipeToId{mlir::hivm::PIPE::PIPE_UNASSIGNED};
    int startIndex{-1};
    int endIndex{-1};
    Edge() = delete;
    Edge(mlir::hivm::PIPE pipeFromId, mlir::hivm::PIPE pipeToId, int startIndex,
         int endIndex)
        : pipeFromId(pipeFromId), pipeToId(pipeToId), startIndex(startIndex),
          endIndex(endIndex) {}
    bool operator<(const Edge &other) const;
  };

  llvm::DenseMap<mlir::hivm::PIPE,
                 llvm::DenseMap<mlir::hivm::PIPE, std::set<Edge>>>
      adjacencyList;

  void addPair(mlir::hivm::PIPE startPipeId, mlir::hivm::PIPE endPipeId,
               int startIndex, int endIndex);

  void addConflictPair(syncsolver::ConflictPair *conflictPair);

  void optimizeAdjacencyList();

  std::optional<int> runDijkstra(mlir::hivm::PIPE startPipe,
                                 mlir::hivm::PIPE endPipe, int startIndex,
                                 int maxDistance);
};
} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_GRAPHSOLVER_GRAPHSOLVER_H
