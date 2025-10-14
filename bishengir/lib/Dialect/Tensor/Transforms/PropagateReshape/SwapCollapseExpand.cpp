//===- SwapCollapseExpand.cpp ---------------------------------------------===//
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
// Swap collapse and expand order so collapse can be put down and expand can be
// put up
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/SwapCollapseExpand.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "llvm/ADT/SmallPtrSet.h"

#define DEBUG_TYPE "propagate-reshape"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

namespace mlir {
namespace tensor {
using namespace mlir::hfusion;
using namespace mlir::tensor::reshape_utils;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;

// %b = collapse %a
// <AxBxCxDxExFxG> -> <AxBCDxExFG>
// [[0], [1, 2, 3], [4], [5, 6]]
// %c = expand %b
// <AxBCDxExFG> -> <AxBCDxE1xE2xFG>
// [[0], [1], [2, 3], [4]]
//
// |
// v
//
// %tmp = expand %a
// <AxBxCxDxExFxG> -> <AxBxCxDxE1xE2xFxG>
// [[0], [1], [2], [3], [4, 5], [6], [7]]
// %c = collapse %tmp
// <AxBxCxDxE1xE2xFxG> -> <AxBCDxE1xE2xFG>
// [[0], [1, 2, 3], [4], [5], [6, 7]]
namespace {

bool areReassociationsCompatible(
    ArrayRef<ReassociationIndices> collapseReassoc,
    ArrayRef<ReassociationIndices> expandReassoc,
    SmallVector<ReassociationIndices> &supposedExpand,
    SmallVector<ReassociationIndices> &supposedCollapse,
    ArrayRef<int64_t> collapseSourceShape, ArrayRef<int64_t> expandShapeResult,
    SmallVector<int64_t> &newExpandShape) {
  // Check if collapse and expand reassociations are inverses of each other
  if (collapseReassoc.size() != expandReassoc.size())
    return false;
  for (size_t i = 0; i < collapseReassoc.size(); ++i) {
    bool isCollapsing = collapseReassoc[i].size() > 1;
    bool isExpanding = expandReassoc[i].size() > 1;
    if (isCollapsing && isExpanding) {
      return false;
    }
    if (isExpanding) {
      for (auto el : expandReassoc[i]) {
        assert(el >= 0 && static_cast<size_t>(el) < expandShapeResult.size());
        newExpandShape.push_back(expandShapeResult[el]);
        supposedCollapse.push_back({-1});
      }
      supposedExpand.push_back(expandReassoc[i]);
    } else {
      for (auto el : collapseReassoc[i]) {
        newExpandShape.push_back(collapseSourceShape[el]);
        supposedExpand.push_back({-1});
      }
      supposedCollapse.push_back(collapseReassoc[i]);
    }
  }
  return true;
}
} // namespace

static bool isSwapCollapseExpandApplicable(tensor::ExpandShapeOp expandOp,
                                           tensor::CollapseShapeOp collapseOp) {
  auto *definedCollapse = collapseOp.getSrc().getDefiningOp();
  if (!definedCollapse || isStopPropagatable(definedCollapse))
    return false;

  if (llvm::all_of(expandOp->getUsers(),
                   [&](Operation *op) { return isOutOp(op); })) {
    return false;
  }

  return true;
}

static Divisibility
buildDivisibilityInformation(tensor::CollapseShapeOp collapseOp,
                             tensor::ExpandShapeOp expandOp,
                             PatternRewriter &rewriter) {
  auto collapseReassoc = collapseOp.getReassociationIndices();
  auto expandReassoc = expandOp.getReassociationIndices();
  Divisibility divisibility(rewriter.getContext());
  size_t intermediateRank = collapseReassoc.size();
  // Build divisibility information
  for (size_t i = 0; i < intermediateRank; ++i) {
    divisibility.addIntermediateInfo(collapseReassoc[i], expandReassoc[i]);
  }

  for (size_t resultIdx = 0; resultIdx < intermediateRank; ++resultIdx) {
    const auto &sourceGroup = collapseReassoc[resultIdx];
    if (expandReassoc[resultIdx].size() != 1)
      continue;
    const auto realResultIndex = expandReassoc[resultIdx][0];
    if (sourceGroup.size() > 1) {
      for (int64_t sourceDim : sourceGroup) {
        divisibility.addSourceDividesResult(sourceDim, realResultIndex);
      }
    }
  }

  for (size_t sourceIdx = 0; sourceIdx < intermediateRank; ++sourceIdx) {
    const auto &resultGroup = expandReassoc[sourceIdx];
    if (collapseReassoc[sourceIdx].size() != 1)
      continue;
    const auto realSourceIndex = collapseReassoc[sourceIdx][0];
    if (resultGroup.size() > 1) {
      for (int64_t resultDim : resultGroup) {
        divisibility.addResultDividesSource(resultDim, realSourceIndex);
      }
    }
  }

  return divisibility;
}

static bool computeReassociations(
    tensor::CollapseShapeOp collapseOp, tensor::ExpandShapeOp expandOp,
    PatternRewriter &rewriter,
    SmallVector<ReassociationIndices> &newReassociationExpand,
    SmallVector<ReassociationIndices> &newReassociationCollapse,
    SmallVector<OpFoldResult> &newExpandShape) {
  auto collapseReassoc = collapseOp.getReassociationIndices();
  auto expandReassoc = expandOp.getReassociationIndices();
  SmallVector<int64_t> newExpandShapeInt;

  // Try fixed reassociations first
  if (areReassociationsCompatible(
          collapseReassoc, expandReassoc, newReassociationExpand,
          newReassociationCollapse,
          utils::getShape(collapseOp.getSrc().getType()),
          utils::getShape(expandOp.getResult().getType()), newExpandShapeInt)) {
    newExpandShape =
        getAsIndexOpFoldResult(rewriter.getContext(), newExpandShapeInt);
    return true;
  }

  // Fixed reassociations failed, try loose reassociations
  newExpandShape.clear();
  newReassociationExpand.clear();
  newReassociationCollapse.clear();
  LLVM_DEBUG(llvm::dbgs() << "Fixed reassociations fail\n";);

  Divisibility divisibility =
      buildDivisibilityInformation(collapseOp, expandOp, rewriter);

  auto collapseSourceShape =
      tensor::getMixedSizesOrOutputShape(rewriter, collapseOp.getSrc());
  auto expandShapeResult =
      tensor::getMixedSizesOrOutputShape(rewriter, expandOp.getResult());

  if (!areLooseReassociationsCompatible(
          newReassociationExpand, newReassociationCollapse, collapseSourceShape,
          expandShapeResult, newExpandShape, divisibility)) {
    LLVM_DEBUG(llvm::dbgs() << "Loose reassociations fail\n";);
    return false;
  }

  return true;
}

LogicalResult
SwapCollapseExpand::matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                    PatternRewriter &rewriter) const {
  tensor::CollapseShapeOp collapseOp =
      expandOp.getSrc().getDefiningOp<tensor::CollapseShapeOp>();
  if (!collapseOp || !isSwapCollapseExpandApplicable(expandOp, collapseOp))
    return rewriter.notifyMatchFailure(expandOp,
                                       "Swap collapse expand not applicable");
  LLVM_DEBUG(llvm::dbgs() << "Trying to swap collapse expand here\n";);
  SmallVector<ReassociationIndices> newReassociationExpand;
  SmallVector<ReassociationIndices> newReassociationCollapse;
  SmallVector<OpFoldResult> newExpandShape;

  if (!computeReassociations(collapseOp, expandOp, rewriter,
                             newReassociationExpand, newReassociationCollapse,
                             newExpandShape)) {
    return rewriter.notifyMatchFailure(collapseOp,
                                       "Reassociations not computable");
  }

  // Step 3: Create replacement operations
  renumberReassociation(newReassociationExpand);
  renumberReassociation(newReassociationCollapse);
  rewriter.setInsertionPointAfter(expandOp);

  auto newExpandType = RankedTensorType::get(getStaticOpFoldRes(newExpandShape),
                                             getElementTypeOrSelf(expandOp));
  tensor::ExpandShapeOp newExpandOp;

  newExpandOp = rewriter.create<tensor::ExpandShapeOp>(
      collapseOp.getLoc(), newExpandType, collapseOp.getSrc(),
      newReassociationExpand);

  LDBG(newExpandOp);

  auto newCollapseOp = rewriter.create<tensor::CollapseShapeOp>(
      expandOp.getLoc(), expandOp.getResult().getType(),
      newExpandOp.getResult(), newReassociationCollapse);

  rewriter.replaceAllUsesWith(expandOp, newCollapseOp.getResult());
  return success();
}

} // namespace tensor
} // namespace mlir
