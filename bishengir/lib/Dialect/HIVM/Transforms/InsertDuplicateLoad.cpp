//===- InsertDuplicateLoad.cpp ----------------------------------*- C++ -*-===//
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
//
// This pass inserts duplicate load on GM for min cv function.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_INSERTDUPLICATELOAD
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "insert-duplicate-load"
namespace {

class OpInfo {
public:
   bool isFor{false};
   int iterArgsIdx{-1};
   unsigned opArgIdx{0};
};

struct InsertDuplicateLoadPass
    : public impl::InsertDuplicateLoadBase<InsertDuplicateLoadPass> {
  using Base::Base;
  void runOnOperation() override;
};

void saveOpInfo(Operation *op,
                               SmallVectorImpl<std::pair<Operation *, OpInfo>> &defChain,
                               int iterArgsIdx = -1) {
   OpInfo opInfo;
   opInfo.isFor = isa<scf::ForOp>(op);
   opInfo.iterArgsIdx = iterArgsIdx;
   defChain.push_back(std::make_pair(op, opInfo));
}

template <typename OpType>
std::optional<Operation *>
traceDefOpAndSave(Value v,
                                   SmallVectorImpl<std::pair<Operation *, OpInfo>> &defChain,
                                   bool isSingleChain = false) {
  if (isSingleChain && getUsersNum(v) != 1)
    return std::nullopt;
  if (Operation *definingOp = v.getDefiningOp<OpType>()) {
       saveOpInfo(definingOp, defChain);
    return definingOp;
  } else if (auto reshapeOp = v.getDefiningOp<tensor::ReshapeOp>()) {
       saveOpInfo(reshapeOp, defChain);
    return traceDefOpAndSave<OpType>(reshapeOp.getSource(), defChain, isSingleChain);
  } else if (auto memrefCollapseShape =
                 v.getDefiningOp<memref::CollapseShapeOp>()) {
       saveOpInfo(memrefCollapseShape, defChain);
    return traceDefOpAndSave<OpType>(memrefCollapseShape.getViewSource(),
                              defChain, isSingleChain);
  } else if (auto tensorCollapseShape =
                 v.getDefiningOp<tensor::CollapseShapeOp>()) {
       saveOpInfo(tensorCollapseShape, defChain);
    return traceDefOpAndSave<OpType>(tensorCollapseShape.getSrc(), defChain, isSingleChain);
  } else if (auto subViewOp = v.getDefiningOp<memref::SubViewOp>()) {
    saveOpInfo(subViewOp, defChain);
       return traceDefOpAndSave<OpType>(subViewOp.getViewSource(), defChain, isSingleChain);
  } else if (auto toMemrefOp = v.getDefiningOp<bufferization::ToMemrefOp>()) {
    saveOpInfo(toMemrefOp, defChain);
       return traceDefOpAndSave<OpType>(toMemrefOp.getOperand(), defChain, isSingleChain);
  } else if (auto toTensorOp = v.getDefiningOp<bufferization::ToTensorOp>()) {
    saveOpInfo(toTensorOp, defChain);
       return traceDefOpAndSave<OpType>(toTensorOp.getOperand(), defChain, isSingleChain);
  } else if (auto viewOp = v.getDefiningOp<memref::ViewOp>()) {
    saveOpInfo(viewOp, defChain);
       return traceDefOpAndSave<OpType>(viewOp.getViewSource(), defChain, isSingleChain);
  } else if (auto reshapeOp = v.getDefiningOp<memref::ReshapeOp>()) {
    saveOpInfo(reshapeOp, defChain);
       return traceDefOpAndSave<OpType>(reshapeOp.getViewSource(), defChain, isSingleChain);
  } else if (auto expandShapeOp = v.getDefiningOp<memref::ExpandShapeOp>()) {
    saveOpInfo(expandShapeOp, defChain);
       return traceDefOpAndSave<OpType>(expandShapeOp.getViewSource(), defChain, isSingleChain);
  } else if (auto tensorExpandShapeOp =
                 v.getDefiningOp<tensor::ExpandShapeOp>()) {
    saveOpInfo(tensorExpandShapeOp, defChain);
       return traceDefOpAndSave<OpType>(tensorExpandShapeOp->getOperand(0),
                              defChain, isSingleChain);
  } else if (auto extractStridedMetadataOp =
                 v.getDefiningOp<memref::ExtractStridedMetadataOp>()) {
    saveOpInfo(extractStridedMetadataOp, defChain);
       return traceDefOpAndSave<OpType>(extractStridedMetadataOp.getViewSource(),
                              defChain, isSingleChain);
  } else if (auto castOp = v.getDefiningOp<memref::CastOp>()) {
    saveOpInfo(castOp, defChain);
       return traceDefOpAndSave<OpType>(castOp.getViewSource(), defChain, isSingleChain);
  } else if (auto reinterpretCastOp =
                 v.getDefiningOp<memref::ReinterpretCastOp>()) {
    saveOpInfo(reinterpretCastOp, defChain);
       return traceDefOpAndSave<OpType>(reinterpretCastOp.getViewSource(), defChain, isSingleChain);
  } else if (auto blockArg = dyn_cast_if_present<BlockArgument>(v)) {
    if (auto scfForOp = dyn_cast_if_present<scf::ForOp>(
            blockArg.getOwner()->getParentOp())) {
      if (OpOperand *iterArgOperand = scfForOp.getTiedLoopInit(blockArg)){
               saveOpInfo(scfForOp, defChain, blockArg.getArgNumber() - 1);
        return traceDefOpAndSave<OpType>(iterArgOperand->get(), defChain, isSingleChain);
           }
    }
  } else if (auto forOp = v.getDefiningOp<scf::ForOp>()) {
    const unsigned int index = cast<OpResult>(v).getResultNumber();
    Value yieldedValue = forOp.getYieldedValues()[index];
    saveOpInfo(forOp, defChain);
       return traceDefOpAndSave<OpType>(yieldedValue, defChain, isSingleChain);
  } else if (auto ifOp = v.getDefiningOp<scf::IfOp>()) {
    const unsigned int index = cast<OpResult>(v).getResultNumber();
    Block &thenBlock = ifOp.getThenRegion().front();
    Value yieldedValue = thenBlock.getTerminator()->getOperand(index);
    saveOpInfo(ifOp, defChain);
       return traceDefOpAndSave<OpType>(yieldedValue, defChain, isSingleChain);
  } else if (auto extractSliceOp = v.getDefiningOp<tensor::ExtractSliceOp>()) {
    saveOpInfo(extractSliceOp, defChain);
       return traceDefOpAndSave<OpType>(extractSliceOp.getSource(), defChain, isSingleChain);
  } else if (auto insertSliceOp = v.getDefiningOp<tensor::InsertSliceOp>()) {
    saveOpInfo(insertSliceOp, defChain);
       return traceDefOpAndSave<OpType>(insertSliceOp.getSource(), defChain, isSingleChain);
  }
  return std::nullopt;
}

bool isMemrefRoot(Operation *op) { return isa<memref::AllocOp>(op); }

bool isMemrefViewLikeOp(Operation *op) {
    return isa<memref::SubViewOp, memref::CollapseShapeOp,
               memref::ReinterpretCastOp, memref::ExpandShapeOp, memref::CastOp>(
        op);
}

Operation *getNewestFromMapping(Operation *oldOp, IRMapping &mapping) {
   Operation *op = mapping.lookupOrNull(oldOp);
   while(op) {
       oldOp = op;
       op = mapping.lookupOrNull(oldOp);
   }
   return oldOp;
}

// BFS to find the first load op for root alloc between root op and guard op.
template <typename OpType>
FailureOr<Operation *> getActualLoadChain(Operation *rootOp,
                                          Operation *guardOp) {
    SmallVector<Operation *> userList = {rootOp};
   SmallPtrSet<Operation *, 8> visited;
   Block *limitBlock = rootOp->getBlock();
   while (!userList.empty()) {
       Operation *iterOp = userList.pop_back_val();
       if (!visited.insert(iterOp).second)
           continue;
       for (Operation *user : iterOp->getResult(0).getUsers()) {
           if (isMemrefViewLikeOp(user)) {
               userList.push_back(user);
           } else if (isa<OpType>(user)){
               if (user->getBlock() == limitBlock)
                   return user;
               return failure();
           } else if (isa<scf::ForOp, scf::WhileOp, scf::IfOp,
                       scf::ParallelOp>(user)){
               return failure();
           }
       }
   }
   return failure();
}

// Collect all ops into result to clone recursively.
void collectDependentOps(Operation *op, DenseSet<Operation *> &visited,
                         SmallVectorImpl<Operation *> &result) {
   if (!visited.insert(op).second) {
       return;
   }
   for (Value operand : op->getOperands()) {
       if (auto *defOp = operand.getDefiningOp()) {
           if (defOp->getBlock() != op->getBlock()) {
               continue;
           }
           if (isMemrefRoot(defOp)) {
               result.push_back(defOp);
           } else {
               collectDependentOps(defOp, visited, result);
           }
       } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
           continue;
       }
   }
   result.push_back(op);
}

/// Pattern to clone load chain.
///
/// For example:
/// ```
/// %t = memref.alloc()
/// memref.subview ins(%t) outs(%t_subview)
/// load ins(%GM) outs(%t_subview)
/// bufferization.to_tensor ins(%t) outs(%t1)
///
/// vector ins(%t1)
/// mmadL1 ins(%t1)
/// ```
///
/// Is convert into:
/// ```
/// %t = memref.alloc()
/// memref.subview ins(%t) outs(%t_subview)
/// load ins(%GM) outs(%t_subview)
/// bufferization.to_tensor ins(%t) outs(%t1)
///
/// %t_c = memref.alloc()
/// memref.subview ins(%t_c) outs(%t_subview_c)
/// load ins(%GM) outs(%t_subview_c)
/// bufferization.to_tensor ins(%t_c) outs(%t2)
/// 
/// vector ins(%t1)
/// mmadL1 ins(%t2)
/// ```

FailureOr<scf::ForOp> createNewSCFFor(PatternRewriter &rewriter,
                                      IRMapping &mapping, scf::ForOp oldFor,
                                                                           OpInfo opInfo, Value newVal){
   SmallVector<Value> newIterArgs(oldFor.getInitArgs().begin(),
                                                                oldFor.getInitArgs().end());

   newIterArgs.push_back(newVal);
   rewriter.setInsertionPointAfter(oldFor);
   auto newFor = rewriter.create<scf::ForOp>(
       oldFor->getLoc(), oldFor.getLowerBound(), oldFor.getUpperBound(),
       oldFor.getStep(), newIterArgs);
   mapping.map(oldFor.getInductionVar(), newFor.getInductionVar());
   for (auto [oldArg, newArg] :
          llvm::zip(oldFor.getRegionIterArgs(), newFor.getRegionIterArgs())) {
       mapping.map(oldArg, newArg);
   }
   Block &oldBlock = oldFor.getRegion().front();
   Block &newBlock = newFor.getRegion().front();
   if (!newBlock.mightHaveTerminator()) {
       rewriter.setInsertionPointToEnd(&newBlock);
       rewriter.create<scf::YieldOp>(newFor->getLoc());
   }
   auto *oldTerminator = oldBlock.getTerminator();
   auto *newTerminator = newBlock.getTerminator();
   if (!oldTerminator || !newTerminator) {
       llvm_unreachable("terminator not found");
       return failure();
   }
   rewriter.setInsertionPoint(newTerminator);
   SmallVector<Operation *> oldOps;
   for (Operation &op : oldBlock.without_terminator()) {
       oldOps.push_back(&op);
   }
   for (Operation *op : oldOps) {
       rewriter.clone(*op, mapping);
   }

   SmallVector<Value> newYieldOperands;
   for (OpOperand &operand : oldTerminator->getOpOperands()) {
       newYieldOperands.push_back(mapping.lookupOrDefault(operand.get()));
   }
   newYieldOperands.push_back(newYieldOperands[opInfo.iterArgsIdx]);
   rewriter.modifyOpInPlace(
       newTerminator, [&](){ newTerminator->setOperands(newYieldOperands); });
   return newFor;
}

LogicalResult cloneDefChainForCubeOp(
       PatternRewriter &rewriter,
       SmallVectorImpl<std::pair<Operation *, OpInfo>> &defChain,
       IRMapping &mapping, Value lastResVal) {
  SmallVector<Operation *> toBeErased;
   while (defChain.size() > 0) {
       std::pair<Operation *, OpInfo> opInfoPair = defChain.pop_back_val();
       Operation *opToBeCloned = getNewestFromMapping(opInfoPair.first, mapping);
       toBeErased.push_back(opToBeCloned);
       OpInfo opInfo = opInfoPair.second;

       if (opInfo.iterArgsIdx >= 0){ // Need to add extra iter_args for scf.for.
           if (!isa<scf::ForOp>(opToBeCloned))
               return failure();
           FailureOr<scf::ForOp> maybeNewFor =
                   createNewSCFFor(rewriter, mapping, cast<scf::ForOp>(opToBeCloned),
                                                   opInfo, lastResVal);
           if (failed(maybeNewFor)){
               return failure();
           }
           scf::ForOp newFor = maybeNewFor.value();
           assert(newFor && "Fail to create new scf.for");
           for (size_t i = 0; i < opToBeCloned->getNumResults(); ++i) {
               rewriter.replaceAllUsesWith(opToBeCloned->getResult(i),
                                                                       newFor->getResult(i));
           }
           lastResVal =
                   newFor.getRegionIterArg(newFor.getRegionIterArgs().size() - 1);
       } else {
           rewriter.setInsertionPoint(opToBeCloned);
           Operation *clonedOp = rewriter.clone(*opToBeCloned, mapping);
           if (!clonedOp)
               return failure();
           rewriter.modifyOpInPlace(clonedOp, [&](){
               clonedOp->setOperand(opInfo.opArgIdx, lastResVal);
           });
           for (size_t i = 0; i < opToBeCloned->getNumResults(); ++i) {
               rewriter.replaceAllUsesWith(opToBeCloned->getResult(i),
                                                                       clonedOp->getResult(i));
           }
           lastResVal = clonedOp->getResult(0);
       }
       // TODO: Support scf.while
   }
   for (Operation *op : toBeErased) {
       rewriter.eraseOp(op);
   }
   return success();
}

LogicalResult
cloneLoadForCubeOp(PatternRewriter &rewriter,
                                    SmallVectorImpl<std::pair<Operation *, OpInfo>> &defChain) {
  if (defChain.empty())
       return failure();
   Operation *allocOp = defChain.pop_back_val().first;
   if (!isa<memref::AllocOp>(allocOp)){
       return failure();
   }
   Operation *toTensorOp = defChain.pop_back_val().first;
   if (!isa<bufferization::ToTensorOp>(toTensorOp)){
       return failure();
   }

   FailureOr<Operation *> maybeLoadOp = 
           getActualLoadChain<hivm::LoadOp>(allocOp, toTensorOp);
   if(!succeeded(maybeLoadOp)) {
       return failure();
   }
   DenseSet<Operation *> visited;
   SmallVector<Operation *> deps;
   collectDependentOps(maybeLoadOp.value(), visited, deps);
   Operation *lastInsertOp = nullptr;
   IRMapping mapping;
   rewriter.setInsertionPointAfter(toTensorOp);
   for (Operation *op : deps) {
       lastInsertOp = rewriter.clone(*op, mapping);
   }
   lastInsertOp = rewriter.clone(*toTensorOp, mapping);
   if (!lastInsertOp)
       return failure();
   return cloneDefChainForCubeOp(rewriter, defChain, mapping,
                                                               lastInsertOp->getResult(0));
}

FailureOr<bool> hasVecUsers(Value val) {
   bool hasVectorUser = false;
   for (Operation *user : val.getUsers()) {
       if (isa<hivm::HIVMStructuredOp>(user)) {
           FailureOr<TCoreType> opType = getCoreType(user);
           if (!succeeded(opType))
               return failure();
           if (opType.value() == TCoreType::VECTOR) {
               hasVectorUser = true;
               break;
           }
       }
   }
   return hasVectorUser;
}

struct InsertDuplicateLoadPattern : public OpRewritePattern<hivm::MmadL1Op> {
   using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;
   LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                 PatternRewriter &rewriter) const override {
       for (unsigned int i = 0; i < op.getNumOperands(); ++i) {
           Value val = op->getOperand(i);
           FailureOr<bool> hasVectorUser = hasVecUsers(val);
           if (!succeeded(hasVectorUser))
               return failure();
           if (!hasVectorUser.value())
               continue;
           SmallVector<std::pair<Operation *, OpInfo>, 8> defChain;
           OpInfo mmadInfo;
           mmadInfo.opArgIdx = i;
           defChain.push_back(std::make_pair(op, mmadInfo));
           std::optional<Operation *> maybeAllocOp =
                   traceDefOpAndSave<memref::AllocOp>(val, defChain);
           if (!maybeAllocOp.has_value()) {
               continue;
           }
           if (failed(cloneLoadForCubeOp(rewriter, defChain))) {
               return failure();
           }
           // Directly return for next matchAndRewrite.
           // because current op may be changed.
           return success();
       }
       return failure();
   }
};
} // namespace

void InsertDuplicateLoadPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto *context = &getContext();
  auto funcOp = getOperation();
  RewritePatternSet patterns(context);
  patterns.insert<InsertDuplicateLoadPattern>(patterns.getContext());
  GreedyRewriteConfig config = GreedyRewriteConfig();
  bool changed = false;
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config,
                                   &changed))) {
    signalPassFailure();
  }
  if (!changed)
       return;
  PassManager pm(context);
  pm.addPass(createCSEPass());
  if (failed(pm.run(funcOp))) {
       funcOp->emitError() << "Failed to run CSE optimization passes.\n";
       signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createInsertDuplicateLoadPass() {
  return std::make_unique<InsertDuplicateLoadPass>();
}