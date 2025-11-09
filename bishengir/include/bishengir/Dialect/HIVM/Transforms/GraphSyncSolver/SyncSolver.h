//===------------- SyncSolver.h ---- Graph Sync Solver --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENG_DIALECT_HIVM_GRAPHSYNCSOLVER_SYNCSOLVER_H
#define BISHENG_DIALECT_HIVM_GRAPHSYNCSOLVER_SYNCSOLVER_H

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <deque>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

using SyncMap =
    std::map<mlir::hivm::syncsolver::OperationBase *,
             std::deque<std::unique_ptr<mlir::hivm::syncsolver::SyncOp>>>;
using SyncBeforeAfterMap = std::pair<SyncMap, SyncMap>;

namespace mlir::hivm::syncsolver {

struct ProcessingOrder {
  Occurrence *occ{nullptr};
  int start{-1};
  int end{-1};
  bool reverseOrder{false};
  bool isUseless{false};
  bool skip{false};
  ProcessingOrder(Occurrence *occ, int start, int end, bool reverseOrder,
                  bool isUseless, bool skip = false)
      : occ(occ), start(start), end(end), reverseOrder(reverseOrder),
        isUseless(isUseless), skip(skip) {}
};

class Solver {
public:
  uint64_t globalIndex{0};
  uint64_t globalCodeGenIndex{0};
  bool resultFuncIrWasGenerated{false};
  bool considerMergedBackwardSyncEventIds{true};
  bool disableMultiEventIdForBarrierAllPairs{true};
  bool reuseSyncPairToSaveEventIds{true};
  bool enableUnitFlagFeature{false};

  func::FuncOp func;
  std::unique_ptr<OperationBase> funcIr;
  std::vector<std::unique_ptr<Occurrence>> syncIr;
  llvm::DenseMap<OperationBase *, std::vector<Occurrence *>> opAllOccurrences;
  std::vector<std::unique_ptr<ConflictPair>> chosenConflictedPairs,
      persistentChosenConflictedPairs;
  llvm::DenseMap<std::tuple<OperationBase *, OperationBase *, OperationBase *,
                            hivm::PIPE, hivm::PIPE>,
                 ConflictPair *>
      syncedPairs, replacedWithReusableSyncedPairs;
  llvm::DenseMap<OperationBase *, llvm::DenseSet<ConflictPair *>>
      scopeOpChosenConflicts;
  llvm::DenseMap<Occurrence *, llvm::DenseSet<ConflictPair *>>
      scopeOccChosenConflicts, persistentScopeOccChosenConflicts;
  llvm::DenseMap<std::pair<Occurrence *, Occurrence *>,
                 llvm::DenseSet<ConflictPair *>>
      scopeOccPairChosenConflicts, persistentScopeOccPairChosenConflicts;
  llvm::DenseMap<std::pair<OperationBase *, OperationBase *>,
                 llvm::DenseSet<ConflictPair *>>
      scopeOpPairChosenConflicts;
  std::vector<ProcessingOrder> processingOrders;
  llvm::DenseSet<std::pair<Occurrence *, Occurrence *>> processedOccPairs;
  llvm::DenseSet<Occurrence *> skipOcc;
  llvm::DenseMap<LoopLikeOpInterface, Value> nestedIndexModularMem;
  llvm::DenseMap<LoopLikeOpInterface,
                 llvm::DenseMap<std::pair<hivm::EVENT, hivm::EVENT>, Value>>
      bufferSelectedMem;
  llvm::DenseMap<Occurrence *, llvm::SmallVector<Occurrence *>> occChildrenMem;
  std::map<OperationBase *, llvm::DenseMap<std::pair<hivm::PIPE, hivm::PIPE>,
                                           llvm::DenseSet<hivm::EVENT>>>
      backwardSyncEvents, backwardSyncEventsAfterMerge;

  llvm::DenseMap<OperationBase *, uint64_t> codeGenStartIndex, codeGenEndIndex;
  llvm::DenseMap<OperationBase *, uint64_t> codeGenInclusiveStartIndex,
      codeGenInclusiveEndIndex;
  llvm::DenseMap<std::tuple<hivm::PIPE, hivm::PIPE, hivm::EVENT>,
                 std::set<std::pair<uint64_t, SetWaitOp *>>>
      setWaitFlagOpsIndex;
  llvm::DenseMap<
      std::pair<syncsolver::RWOperation *, syncsolver::RWOperation *>,
      std::vector<std::pair<hivm::PIPE, hivm::PIPE>>>
      checkMemoryConflictsMem, checkTestMemoryConflictsMem;
  llvm::DenseSet<std::pair<hivm::PIPE, hivm::PIPE>> barrierAllPairs;
  llvm::DenseMap<std::pair<hivm::PIPE, hivm::PIPE>, int> reusePairs;
  llvm::DenseMap<OperationBase *,
                 llvm::DenseSet<std::pair<Occurrence *, int32_t>>>
      insertedBarrierAllBefore;
  llvm::DenseMap<hivm::MmadL1Op, MmadL1SyncArgs> mmadl1SyncArgsMap;
  llvm::DenseSet<RWOperation *> unitFlagFeaturedOps;

public:
  Solver(func::FuncOp func) : func(func) {
    syncsolver::OperationBase::resetLCAMem();
    syncsolver::Occurrence::resetLCAMem();
    auto funcOp = std::make_unique<syncsolver::Function>(func.getOperation());
    auto scopeOp = funcIrBuilder(func.getRegion(), funcOp.get());
    funcOp->body.push_back(std::move(scopeOp));
    funcIr = std::move(funcOp);
    syncIrBuilder(funcIr.get());
  }
  Solver(std::unique_ptr<OperationBase> funcIr) : funcIr(std::move(funcIr)) {
    syncsolver::OperationBase::resetLCAMem();
    syncsolver::Occurrence::resetLCAMem();
    syncIrBuilder(this->funcIr.get());
  }

  void solve(int runNum = 0);

  void solveTest(int runNum = 0);

  void generateFuncIrResultOps();

  void generateResultOps();

  SyncBeforeAfterMap getBeforeAfterSyncMaps();

private:
  void reset();

  void processOrders();

  void processOrdersTest();

  std::unique_ptr<Scope> funcIrBuilder(Region &region, OperationBase *parentOp);

  std::unique_ptr<OperationBase> getDecomposedMmadl1(hivm::MmadL1Op mmadl1Op,
                                                     OperationBase *parentOp);

  void generateProcessingOrders(Occurrence *scopeOcc, int l, int r,
                                bool isUseless);

  void generateProcessingOrders(int l, int r, bool isUseless);

  void generateProcessingOrders(int l1, int r1, int l2, int r2, bool isUseless);

  void syncIrBuilder(OperationBase *op, Occurrence *parentOcc = nullptr,
                     int depth = 0, bool isUseless = false);

  llvm::SmallVector<Value> collectPointerOps(Value val);

  llvm::SmallVector<Value> getMemOps(const SmallVector<Value> &vals);

  std::pair<llvm::SmallVector<Value>, llvm::SmallVector<Value>>
  getReadWriteMemOps(Operation *op);

  template <typename OP>
  typename std::enable_if<std::is_same_v<OP, memref::LoadOp> ||
                              std::is_same_v<OP, affine::AffineLoadOp> ||
                              std::is_same_v<OP, affine::AffineStoreOp> ||
                              std::is_same_v<OP, memref::StoreOp>,
                          std::unique_ptr<OperationBase>>::type
  getLoadStoreOp(OP *op, OperationBase *parentOp);

  bool
  checkEventIdNum(const llvm::SmallVector<llvm::SmallVector<int>> &memValsList1,
                  const llvm::SmallVector<llvm::SmallVector<int>> &memValsList2,
                  int lcmLen, int eventIdNum);

  std::optional<LoopLikeOpInterface>
  checkDoubleMultiBufferEventId(const llvm::SmallVector<Value> &memValsList1,
                                const llvm::SmallVector<Value> &memValsList2);

  std::optional<LoopLikeOpInterface>
  checkDoubleMultiBufferEventId(hivm::PointerCastOp pointerCastOp1,
                                hivm::PointerCastOp pointerCastOp2);

  std::optional<LoopLikeOpInterface>
  checkDoubleMultiBufferEventId(RWOperation *rwOp1, RWOperation *rwOp2);

  std::pair<uint32_t, LoopLikeOpInterface> getEventIdNum(Occurrence *occ1,
                                                         Occurrence *occ2,
                                                         hivm::PIPE setPipe,
                                                         hivm::PIPE waitPipe);

  uint32_t getTestEventIdNum(RWOperation *rwOp1, RWOperation *rwOp2);

  uint32_t getTestEventIdNum(Occurrence *occ1, Occurrence *occ2,
                             hivm::PIPE setPipe, hivm::PIPE waitPipe);

  bool checkGraphConflict(Occurrence *occ1, Occurrence *occ2,
                          hivm::PIPE startPipe, hivm::PIPE endPipe,
                          uint32_t eventIdNum);

  std::vector<std::pair<hivm::PIPE, hivm::PIPE>>
  checkMemoryConflicts(RWOperation *rwOp1, RWOperation *rwOp2);

  std::vector<std::pair<hivm::PIPE, hivm::PIPE>>
  checkTestMemoryConflicts(RWOperation *rwOp1, RWOperation *rwOp2);

  bool checkRWMemoryConflicts(const llvm::SmallVector<Value> &memValsList1,
                              const llvm::SmallVector<Value> &memValsList2);

  bool checkTestRWMemoryConflicts(
      const llvm::SmallVector<llvm::SmallVector<int>> &memValsList1,
      const llvm::SmallVector<llvm::SmallVector<int>> &memValsList2);

  bool checkPointerCastMemConflict(hivm::PointerCastOp pointerCastOp1,
                                   hivm::PointerCastOp pointerCastOp2);

  bool checkImpossibleOpPair(OperationBase *op1, OperationBase *op2);

  bool checkImpossibleOccPair(Occurrence *occ1, Occurrence *occ2);

  bool checkAlreadySynced(Occurrence *occ1, Occurrence *occ2);

  bool skipMMad1DecomposedLoopOpt(Occurrence *occ1, Occurrence *occ2);

  llvm::SmallVector<hivm::EVENT>
  getAvailableEventIds(ConflictPair *conflictPair);

  llvm::SmallVector<hivm::EVENT>
  getAnyAvailableEventId(ConflictPair *conflictPair, uint32_t count,
                         bool reversedPriority);

  llvm::SmallVector<hivm::EVENT>
  getAnyAvailableMultiBufferEventIds(ConflictPair *conflictPair, uint32_t count,
                                     bool reversedPriority);

  bool checkVisited(Occurrence *occ1, Occurrence *occ2);

  bool checkSkippable(Occurrence *occ);

  std::optional<llvm::SmallVector<hivm::EVENT>>
  getOldEventIdIfExists(OperationBase *scopeOp, Occurrence *occ1,
                        Occurrence *occ2, ConflictPair *conflictPair);

  void memorizeSyncedPair(OperationBase *scopeOp, ConflictPair *conflictPair);

  void memorizeReusedSyncedPair(OperationBase *scopeOp,
                                ConflictPair *conflictPair,
                                ConflictPair *reusedConflictPair);

  void forgetSyncedPair(OperationBase *scopeOp, ConflictPair *conflictPair);

  std::pair<Occurrence *, Occurrence *> getSetWaitOcc(Occurrence *occ1,
                                                      Occurrence *occ2);

  void insertBarrierAllBefore(Occurrence *occ, bool isUseless,
                              bool isPersistent = false);

  bool isBackwardSync(Occurrence *occ1, Occurrence *occ2);

  ConflictPair *getReusableConflictPair(
      ConflictPair *conflictPair,
      const llvm::DenseSet<ConflictPair *> &conflictPairsSet);

  bool reuseConflictPair(ConflictPair *conflictPair, Occurrence *scopeOcc1,
                         Occurrence *scopeOcc2);

  void handleConflict(Occurrence *occ1, Occurrence *occ2, hivm::PIPE setPipe,
                      hivm::PIPE waitPipe, bool isUseless, uint32_t eventIdNum,
                      LoopLikeOpInterface multibufferLoopPar);

  Location getProperLoc(OperationBase *opBase);

  void setProperInsertionPoint(IRRewriter &rewriter, OperationBase *opBase,
                               bool insertAfterOp);

  void insertBarrierOp(IRRewriter &rewriter, OperationBase *opBase,
                       BarrierOp *barrierOp, bool insertAfterOp);

  void insertSetFlagOp(IRRewriter &rewriter, OperationBase *opBase,
                       SetFlagOp *setFlagOp, bool insertAfterOp);

  void insertWaitFlagOp(IRRewriter &rewriter, OperationBase *opBase,
                        WaitFlagOp *waitFlagOp, bool insertAfterOp);

  Value getEventIdValue(IRRewriter &rewriter, SetWaitOp *setWaitOp,
                        Location loc);

  llvm::LogicalResult handleMmadL1SyncOps(IRRewriter &rewriter,
                                          OperationBase *opBase,
                                          SyncOp *syncOp);

  Value getMultiBufferSelectOp(IRRewriter &rewriter, SetWaitOp *syncOp);

  void insertMultiBufferSetFlagOp(IRRewriter &rewriter, OperationBase *opBase,
                                  SetFlagOp *setFlagOp, bool insertAfterOp);

  void insertMultiBufferWaitFlagOp(IRRewriter &rewriter, OperationBase *opBase,
                                   WaitFlagOp *waitFlagOp, bool insertAfterOp);

  void insertMmadL1SyncArgs(IRRewriter &rewriter);

  hivm::UNIT_FLAG getUnitFlagMode(RWOperation *rwOp);

  Value getIsNotDeadLoopValue(scf::ForOp forOp, Location loc,
                              IRRewriter &rewriter);

  std::optional<mlir::Value> getUnitFlagCond(IRRewriter &rewriter,
                                             RWOperation *rwOp);

  void handleUnitFlagEnabledOps(IRRewriter &rewriter);

  void insertBarrierAllBeforeReturn(IRRewriter &rewriter);

  Occurrence *getFirstIterOcc(Occurrence *occ, Occurrence *parOcc);

  Occurrence *getLastIterOcc(Occurrence *occ, Occurrence *parOcc);

  std::pair<Occurrence *, Occurrence *>
  checkAndApplyMmadl0LoopOpt(ConflictPair *conflictPair, Occurrence *occ1,
                             Occurrence *occ2, Occurrence *parOcc1,
                             Occurrence *parOcc2);

  std::optional<std::pair<UNIT_FLAG, UNIT_FLAG>>
  checkUnitFlagPatterns(ConflictPair *conflictPair, Occurrence *occ1,
                        Occurrence *occ2, Occurrence *parentLCALoopOcc);

  std::optional<std::pair<UNIT_FLAG, UNIT_FLAG>>
  checkMmadl1FixpipeUnitFlagPattern(RWOperation *rwOp1, RWOperation *rwOp2,
                                    hivm::PIPE pipe1, hivm::PIPE pipe2);

  std::optional<std::pair<hivm::UNIT_FLAG, hivm::UNIT_FLAG>>
  checkMmadl1FixpipeSingleForLoopUnitFlagPattern(RWOperation *rwOp1,
                                                 RWOperation *rwOp2,
                                                 hivm::PIPE pipe1,
                                                 hivm::PIPE pipe2,
                                                 bool rw1IsFrontOcc);

  std::pair<bool, Occurrence *> checkMmadl0BackwardSyncOpt(Occurrence *loopOcc);

  void resetAndBuildSetWaitOpIndex(SyncMap &syncMapBefore,
                                   SyncMap &syncMapAfter);

  void collectSetWaitOpsIndexes(OperationBase *op, SyncMap &syncMapBefore,
                                SyncMap &syncMapAfter);

  bool checkMergeable(Scope *scopeOp, hivm::PIPE pipeSrc, hivm::PIPE pipeDst,
                      hivm::EVENT eventId, bool shouldBeUsedAtleastOnce);

  void mergeBackwardSyncEventIds(OperationBase *op);

  void mergeRootBackwardSyncEventIds(OperationBase *op);

  void pickAndInsertABarrierAll();
};

} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_GRAPHSYNCSOLVER_SYNCSOLVER_H
