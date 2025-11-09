//===------------- Utility.h ---- Graph Sync Solver -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENG_DIALECT_HIVM_GRAPHSYNCSOLVER_UTILITY_H
#define BISHENG_DIALECT_HIVM_GRAPHSYNCSOLVER_UTILITY_H

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include <memory>

// #define NORM_EVENT_ID_NUM (uint8_t)4
#define NORM_EVENT_ID_NUM (uint8_t)8

namespace mlir::hivm::syncsolver {

struct Occurrence {
  OperationBase *op{nullptr};
  Occurrence *parentOcc{nullptr};
  int depth{-1};
  int startIndex{-1};
  int endIndex{-1};
  int syncIrIndex{-1};
  int syncIrEndIndex{-1};
  int loopSplitIndex{-1};

  Occurrence(OperationBase *op, Occurrence *parentOcc, int depth,
             int startIndex, int endIdx)
      : op(op), parentOcc(parentOcc), depth(depth), startIndex(startIndex),
        endIndex(endIdx) {}

  static bool sameScope(Occurrence *occ1, Occurrence *occ2);

  static int getDepth(Occurrence *occ);

  Occurrence *getParentWithOp(OperationBase *op);

  Occurrence *getNthParent(int dist);

  static llvm::DenseMap<std::pair<Occurrence *, Occurrence *>,
                        std::pair<Occurrence *, Occurrence *>>
      getLCAOccMem;

  static void resetLCAMem() { getLCAOccMem.clear(); }

  static std::pair<Occurrence *, Occurrence *> getLCAPair(Occurrence *occ1,
                                                          Occurrence *occ2);

  static Occurrence *getParentloop(Occurrence *occ);

  static Occurrence *getParentCondition(Occurrence *occ);

  bool isProperAncestor(Occurrence *occ);

  std::vector<Occurrence *> getAllParents();
};

struct ConflictPair {

  static int globalDebugIdCounter;

  int debugId{-1};
  OperationBase *op1{nullptr};
  OperationBase *op2{nullptr};
  OperationBase *opSet{nullptr};
  OperationBase *opWait{nullptr};
  hivm::PIPE setPipe{hivm::PIPE::PIPE_UNASSIGNED};
  hivm::PIPE waitPipe{hivm::PIPE::PIPE_UNASSIGNED};
  int startIndex{-1};
  int endIndex{-1};
  llvm::SmallVector<hivm::EVENT> eventIds;
  bool isInnerBackward{false};
  bool isUseless{false};
  bool couldNotRun{false};
  LoopLikeOpInterface multibufferLoopPar{nullptr};
  bool setOnLastIterOnly{false};
  bool waitOnFirstIterOnly{false};
  bool replacedWithUnitFlag{false};

  ConflictPair(OperationBase *op1, OperationBase *op2, OperationBase *opSet,
               OperationBase *opWait, hivm::PIPE setPipe, hivm::PIPE waitPipe,
               int startIndex, int endIndex)
      : op1(op1), op2(op2), opSet(opSet), opWait(opWait), setPipe(setPipe),
        waitPipe(waitPipe), startIndex(startIndex), endIndex(endIndex) {
    debugId = globalDebugIdCounter++;
    if (setPipe == waitPipe) {
      this->opSet = opWait;
      this->startIndex = endIndex;
    }
  };

  bool isBarrier() const { return setPipe == waitPipe; }

  std::string str() const;

  void updateSetWaitOps(Occurrence *setOcc, Occurrence *waitOcc) {
    if (setOcc != nullptr) {
      opSet = setOcc->op;
      startIndex = setOcc->endIndex;
    }
    if (waitOcc != nullptr) {
      opWait = waitOcc->op;
      endIndex = waitOcc->startIndex;
    }
  }

  std::unique_ptr<ConflictPair> clone() {
    auto clonedConflictPair = std::make_unique<ConflictPair>(
        op1, op2, opSet, opWait, setPipe, waitPipe, startIndex, endIndex);
    clonedConflictPair->eventIds = eventIds;
    clonedConflictPair->isInnerBackward = isInnerBackward;
    clonedConflictPair->isUseless = isUseless;
    clonedConflictPair->multibufferLoopPar = multibufferLoopPar;
    clonedConflictPair->setOnLastIterOnly = setOnLastIterOnly;
    clonedConflictPair->waitOnFirstIterOnly = waitOnFirstIterOnly;
    return clonedConflictPair;
  }

  std::unique_ptr<ConflictPair> clone(uint32_t startIndex, uint32_t endIndex) {
    auto clonedConflictPair = this->clone();
    clonedConflictPair->startIndex = startIndex;
    clonedConflictPair->endIndex = endIndex;
    return clonedConflictPair;
  }
};

struct MmadL1SyncArgs {
  MmadL1SyncArgs() = default;
  MmadL1SyncArgs(Value L0WaitL1AEvent, Value L0WaitL1BEvent,
                 Value L1AWaitL0Event, Value L1BWaitL0Event, Value KLoopDBCond,
                 Value BackPipeMPipeMTE1Event0, Value BackPipeMPipeMTE1Event1)
      : L0WaitL1AEvent(L0WaitL1AEvent), L0WaitL1BEvent(L0WaitL1BEvent),
        L1AWaitL0Event(L1AWaitL0Event), L1BWaitL0Event(L1BWaitL0Event),
        KLoopDBCond(KLoopDBCond),
        BackPipeMPipeMTE1Event0(BackPipeMPipeMTE1Event0),
        BackPipeMPipeMTE1Event1(BackPipeMPipeMTE1Event1) {}

  Value L0WaitL1AEvent;
  Value L0WaitL1BEvent;
  Value L1AWaitL0Event;
  Value L1BWaitL0Event;
  Value KLoopDBCond;
  Value BackPipeMPipeMTE1Event0;
  Value BackPipeMPipeMTE1Event1;
};

bool checkIntersect(int l1, int r1, int l2, int r2);

bool checkIntersect(ConflictPair *conflictPair1, ConflictPair *conflictPair2);

std::vector<std::pair<int, int>> getRanges(ConflictPair *conflictPair);

llvm::SmallVector<hivm::EVENT> getHWAvailableEventIds(hivm::PIPE setPipe,
                                                      hivm::PIPE waitPipe);

Value getIsFirstIterationValue(scf::ForOp forOp, Location loc,
                               IRRewriter &rewriter);

Value getIsLastIterationValue(scf::ForOp forOp, Location loc,
                              IRRewriter &rewriter);

std::string op2str(Value val);

std::string op2str(Operation *op);

bool checkAllLoopParentsAreForLoops(Operation *op);

} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_GRAPHSYNCSOLVER_UTILITY_H
