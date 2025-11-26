//===------------------------- AdjustAlignUtil.cpp ------------------------===//
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

#include "bishengir/Dialect/HIVM/Interfaces/FlattenInterface.h"
#include "bishengir/Dialect/HIVM/Transforms/AlignBuffer/Util.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#define DEBUG_TYPE "hivm-align-buffer-util"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define DBGSNL() (llvm::dbgs() << "\n")

// using namespace mlir::hivm::util;

namespace mlir {
namespace hivm {
std::optional<int> getPrevUncontiguousDim(
    const int flattenAlignDim,
    const SmallVector<ReassociationIndices> &flattenReassociations,
    const llvm::SmallVectorImpl<MemRefType> &origMemRefTypes) {
  if (flattenAlignDim == 0) {
    return std::nullopt;
  }

  return getLastNotUnitDim(origMemRefTypes,
                           flattenReassociations,
                           flattenAlignDim - 1);
}

bool isAlreadyAligned(MemRefType memrefType, int32_t alignDim) {
  auto ElemBits = getElementTypeOrSelf(memrefType).getIntOrFloatBitWidth();
  auto hwAlignBits = getHWAlignBytes(memrefType.getMemorySpace()) * 8;
  if (auto strided =
          llvm::dyn_cast<StridedLayoutAttr>(memrefType.getLayout())) {
    auto alignDimStride = strided.getStrides()[alignDim];
    LDBG("alignDimStride: " << alignDimStride << "\n");
    if ((alignDimStride != ShapedType::kDynamic) &&
        (alignDimStride * static_cast<int64_t>(ElemBits)) %
                static_cast<int64_t>(hwAlignBits) ==
            0) {
      return true;
    }
  }

  assert(memrefType.hasRank());
  auto rank = memrefType.getRank();
  ReassociationIndices reassociationIndices =
      llvm::to_vector(llvm::seq<int64_t>(alignDim, rank));
  if (!util::isGuaranteedCollapsibleStrictly(memrefType,
                                             reassociationIndices)) {
    return false;
  }

  int64_t collapsedSize = 1;
  auto shapes = memrefType.getShape();
  for (int64_t i = rank - 1; i > alignDim; i--) {
    if (!ShapedType::isDynamic(shapes[i])) {
      collapsedSize = collapsedSize * shapes[i];
    }
  }
  // use bit considering bool type
  return (collapsedSize * static_cast<int64_t>(ElemBits)) %
             static_cast<int64_t>(hwAlignBits) ==
         0;
}

bool isNoNeedAlign(Value operand, std::optional<int32_t> alignDim) {
  if (!alignDim.has_value()) {
    // no align, no need to adjust
    return true;
  }

  auto memRefType = dyn_cast<MemRefType>(operand.getType());
  if (!memRefType || !memRefType.hasRank()) {
    return true;
  }
  auto memSpace = memRefType.getMemorySpace();
  auto hivmAddressSpace = dyn_cast<AddressSpaceAttr>(memSpace);
  if (!hivmAddressSpace ||
      hivmAddressSpace.getAddressSpace() == hivm::AddressSpace::GM) {
    LDBG("no need to storage align for gm\n");
    return true;
  }

  auto checkDims = llvm::seq<int32_t>(0, alignDim.value() + 1);
  if (llvm::all_of(checkDims, [&](int32_t idx) {
        return memRefType.getShape()[idx] == 1;
      })) {
    // if the dims before alignDim is 1, no need to do
    // storage align
    LDBG("no need to storage align when to the dims before alignDim are 1 "
         "is 1\n");
    return true;
  }

  if (isAlreadyAligned(memRefType, alignDim.value())) {
    // already aligned, no need to do storage align
    LDBG("dim " << alignDim.value() << " of " << memRefType
                << " already aligned, no need to storage align\n");
    return true;
  }

  return false;
}

std::optional<int32_t>
getMapedAlignDim(std::optional<int32_t> alignDim,
                 SmallVector<ReassociationIndices> reassociationIndices) {
  if (!alignDim.has_value()) {
    return std::nullopt;
  }
  for (auto [idx, reassoc] : llvm::enumerate(reassociationIndices)) {
    for (auto index : reassoc) {
      if (index == alignDim.value()) {
        return idx;
      }
    }
  }
  return std::nullopt;
}

std::optional<int>
getLastNotUnitDim(const SmallVectorImpl<MemRefType> &memRefTypes,
                  const llvm::ArrayRef<ReassociationIndices> &continuousReassociations,
                  int64_t startIdx) {

  for (; startIdx >= 0; startIdx--) {
    const auto& reassociations = continuousReassociations[startIdx];
    
    for (auto index : llvm::reverse(reassociations)) {
      if (llvm::all_of(memRefTypes, [&](MemRefType memRefType) {
            return memRefType.getShape()[index] == 1;
          })) {
        continue;
      }
      return index;
    }
  }
  return std::nullopt;
}

bool isLastBrc(hivm::VBrcOp brcOp, FlattenResult &flattenResult) {
  auto flattenedTypes = flattenResult.getOperandTypes(DpsKind::kDpsAll);
  auto flattenedMemrefTypes = util::getMemRefTypes(flattenedTypes);
  if (llvm::any_of(flattenedMemrefTypes, [&](MemRefType memRefType) {
        return !isLastMemrefDimUnitStride(memRefType);
      })) {
    LDBG("not last brc: last stride is not 1\n");
    return false;
  }

  auto brcDims = brcOp.getBroadcastDims();
  auto dstType = brcOp.getDst().getType();
  auto dstRank = cast<ShapedType>(dstType).getRank();
  brcDims =
      brcDims.empty() ? llvm::to_vector(llvm::seq<int64_t>(dstRank)) : brcDims;
  auto lastBrcDim = brcDims.back();

  auto flattenRank = flattenResult.getRankAfterFlatten();
  auto flattenedAssociations = flattenResult.reassociation[0];
  auto flattenLastBrcDim = getMapedAlignDim(lastBrcDim, flattenedAssociations);
  return flattenLastBrcDim.has_value() &&
         (flattenLastBrcDim.value() == flattenRank - 1);
}

/// judge if last two dimensions are contiguous
bool isLastTwoDimContiguous(FlattenResult &flattenResult) {
  auto flattenRank = flattenResult.getRankAfterFlatten();
  if (flattenRank == 1) {
    return true;
  }

  auto flattenOperandTypes = flattenResult.getOperandTypes(DpsKind::kDpsAll);
  bool isLastDimContiguous = true;
  for (auto flattenOperType : flattenOperandTypes) {
    isLastDimContiguous =
        isLastDimContiguous &&
        (isa<MemRefType>(flattenOperType)
             ? isLastMemrefDimUnitStride(cast<MemRefType>(flattenOperType))
             : true);
  }
  if (!isLastDimContiguous) {
    return false;
  }

  auto contiguousMask = detail::getContiguousAxesImpl(flattenOperandTypes);
  return contiguousMask.test(flattenRank - 1);
}

std::optional<int32_t> adjustBrcAlignDim(hivm::VBrcOp brcOp, Value operand,
                                         std::optional<int32_t> alignDim) {
  if (!alignDim.has_value()) {
    return alignDim;
  }
  auto dstType = brcOp.getDst().getType();
  if (getElementTypeOrSelf(dstType).isInteger(8) ||
      getElementTypeOrSelf(dstType).isInteger(1)) {
    return alignDim;
  }

  auto hivmFlattenInterfaceOp =
      cast<hivm::FlattenInterface>(brcOp.getOperation());
  FlattenOptions flattenOptions;
  flattenOptions.checkMarkStride = true;
  auto flattenResult = hivmFlattenInterfaceOp.getFlattened(flattenOptions);
  assert(succeeded(flattenResult));
  if (!isLastBrc(brcOp, flattenResult.value())) {
    return alignDim;
  }
  auto flattenRank = flattenResult->getRankAfterFlatten();
  auto flattenAlignDim =
      getMapedAlignDim(alignDim, flattenResult->reassociation[0]);
  if (!flattenAlignDim.has_value() ||
      flattenAlignDim.value() != flattenRank - 2 ||
      !isLastTwoDimContiguous(flattenResult.value())) {
    return alignDim;
  }
  // adjust to find previous uncontiguous dimension to do alignment
  auto operTypes = brcOp.getHIVMOperandTypes(/*includeExtraBuffer=*/false);
  auto memrefTypes = util::getMemRefTypes(operTypes);
  auto flattenedAssociations = flattenResult->reassociation[0];
  auto adjustAlignDim = getPrevUncontiguousDim(
      flattenAlignDim.value(), flattenedAssociations, memrefTypes);
  // TODO: check if it is needed
  if (isNoNeedAlign(operand, adjustAlignDim)) {
    return std::nullopt;
  }
  LDBG("adjust the alignDim of operand " << operand << " in brcOp : " << brcOp
                                         << " as belows\n");
  LDBG("adjustAlignDim : " << adjustAlignDim << "\n");
  return adjustAlignDim;
}

bool isLastReduce(hivm::VReduceOp reduceOp, FlattenResult &flattenResult) {
  auto flattenedTypes = flattenResult.getOperandTypes(DpsKind::kDpsAll);
  auto flattenedMemrefTypes = util::getMemRefTypes(flattenedTypes);
  if (llvm::any_of(flattenedMemrefTypes, [&](MemRefType memRefType) {
        LDBG("type : " << memRefType << "\n");
        return !isLastMemrefDimUnitStride(memRefType);
      })) {
    LDBG("not last reduce: last stride is not 1\n");
    return false;
  }

  auto reduceDims = reduceOp.getReduceDims();
  auto lastReduceDim = reduceDims.back();
  auto flattenRank = flattenResult.getRankAfterFlatten();
  auto flattenedAssociations = flattenResult.reassociation[0];
  auto flattenLastReduceDim =
      getMapedAlignDim(lastReduceDim, flattenedAssociations);
  return flattenLastReduceDim.has_value() &&
         (flattenLastReduceDim.value() == flattenRank - 1);
}

std::optional<int32_t> adjustReduceAlignDim(hivm::VReduceOp reduceOp,
                                            Value operand,
                                            std::optional<int32_t> alignDim) {
  if (!alignDim.has_value()) {
    return alignDim;
  }

  auto hivmFlattenInterfaceOp =
      cast<hivm::FlattenInterface>(reduceOp.getOperation());
  FlattenOptions flattenOptions;
  flattenOptions.checkMarkStride = true;
  auto flattenResult = hivmFlattenInterfaceOp.getFlattened(flattenOptions);
  assert(succeeded(flattenResult));
  if (!isLastReduce(reduceOp, flattenResult.value())) {
    return alignDim;
  }

  auto flattenRank = flattenResult->getRankAfterFlatten();
  auto flattenAlignDim =
      getMapedAlignDim(alignDim, flattenResult->reassociation[0]);
  if (!flattenAlignDim.has_value() ||
      flattenAlignDim.value() != flattenRank - 2) {
    return alignDim;
  }

  auto flattenSrcType =
      flattenResult->getOperandTypeAfterFlattened(reduceOp.getSrc());
  assert(flattenSrcType != std::nullopt);
  auto lastReduceDimSize =
      cast<ShapedType>(flattenSrcType.value()).getShape()[flattenRank - 1];
  auto elemType = getElementTypeOrSelf(flattenSrcType.value());
  if (static_cast<unsigned int>(lastReduceDimSize) *
              elemType.getIntOrFloatBitWidth() / 8 <=
          util::BL &&
      isa<FloatType>(elemType) && llvm::isPowerOf2_64(lastReduceDimSize) &&
      reduceOp.getArith().getReduceOp() == hivm::ReduceOperation::sum) {
    // no need to storage align for rank - 2 axis when dim size of last
    // reduce axis is less than block size and power of 2 for float reduce sum
    return std::nullopt;
  }

  const auto &inits = reduceOp.getDpsInits();
  bool isInitOper =
      std::find(inits.begin(), inits.end(), operand) != inits.end();
  // TODO: check if init is last two dimension contiguous and decide if do
  // stride alignment after library support init uncontiguous
  if (!isInitOper) {
    return alignDim;
  }

  // adjust to find previous uncontiguous dimension to do alignment
  auto operTypes = reduceOp.getHIVMOperandTypes(/*includeExtraBuffer=*/false);
  auto memrefTypes = util::getMemRefTypes(operTypes);
  auto flattenedAssociations = flattenResult->reassociation[0];
  auto adjustAlignDim = getPrevUncontiguousDim(
      flattenAlignDim.value(), flattenedAssociations, memrefTypes);
  if (isNoNeedAlign(operand, adjustAlignDim)) {
    return std::nullopt;
  }

  LDBG("adjust the alignDim of operand "
       << operand << " in reduceOp : " << reduceOp << " as belows\n");
  LDBG("adjustAlignDim : " << adjustAlignDim << "\n");
  return adjustAlignDim;
}

std::optional<int32_t>
adjustDeInterleaveAlignDim(hivm::VDeinterleaveOp deinterleaveOp, Value operand,
                           std::optional<int32_t> alignDim) {
  if (!alignDim.has_value()) {
    return alignDim;
  }

  auto hivmFlattenInterfaceOp =
      cast<hivm::FlattenInterface>(deinterleaveOp.getOperation());
  FlattenOptions flattenOptions;
  flattenOptions.checkMarkStride = true;
  auto flattenResult = hivmFlattenInterfaceOp.getFlattened(flattenOptions);
  assert(succeeded(flattenResult));
  auto flattenRank = flattenResult->getRankAfterFlatten();
  auto flattenAlignDim =
      getMapedAlignDim(alignDim, flattenResult->reassociation[0]);
  if (!flattenAlignDim.has_value() ||
      flattenAlignDim.value() != flattenRank - 1) {
    return alignDim;
  }

  auto flattenSrcType = flattenResult->getOperandTypes(DpsKind::kDpsInput)[0];
  auto flattenSrcMemrefType = dyn_cast<MemRefType>(flattenSrcType);
  if (flattenSrcMemrefType == nullptr) {
    return alignDim;
  }

  int64_t srcOffset;
  SmallVector<int64_t> srcStrides;
  [[maybe_unused]] auto successStrides =
      getStridesAndOffset(flattenSrcMemrefType, srcStrides, srcOffset);
  assert(succeeded(successStrides));
  if (static_cast<uint64_t>(srcStrides.back()) !=
      deinterleaveOp.getChannelNum()) {
    return alignDim;
  }

  auto flattenDstType = flattenResult->getOperandTypes(DpsKind::kDpsInit)[0];
  auto flattenDstMemrefType = dyn_cast<MemRefType>(flattenDstType);
  if (flattenDstMemrefType == nullptr) {
    return alignDim;
  }
  if (!isLastMemrefDimUnitStride(flattenDstMemrefType)) {
    return alignDim;
  }

  // adjust to find previous uncontiguous dimension to do alignment
  auto operTypes =
      deinterleaveOp.getHIVMOperandTypes(/*includeExtraBuffer=*/false);
  auto memrefTypes = util::getMemRefTypes(operTypes);
  auto flattenedAssociations = flattenResult->reassociation[0];
  auto adjustAlignDim = getPrevUncontiguousDim(
      flattenAlignDim.value(), flattenedAssociations, memrefTypes);
  if (isNoNeedAlign(operand, adjustAlignDim)) {
    return std::nullopt;
  }

  LDBG("adjust the alignDim of operand " << operand << " in deinterleaveOp : "
                                         << deinterleaveOp << " as belows\n");
  LDBG("adjustAlignDim : " << adjustAlignDim << "\n");
  return adjustAlignDim;
}

std::optional<int32_t> adjustCopyAlignDim(hivm::CopyOp copyOp, Value operand,
                                          std::optional<int32_t> alignDim) {
  auto associations = copyOp.getReassociationIndices(/*isCollapse=*/true);
  if (!alignDim.has_value() || associations.empty() ||
      (copyOp.getDst() != operand)) {
    return alignDim;
  }

  for (auto r : associations) {
    r.pop_back();
    if (llvm::find(r, alignDim.value()) != r.end()) {
      // no need to align the collapsible dimension for copy dst
      return std::nullopt;
    }
  }

  return alignDim;
}

std::optional<int32_t> adjustInlineBrcOp(HIVMStructuredOp hivmOp, Value operand,
                                         std::optional<int32_t> alignDim) {
  if (!hivmOp.isInlineBroadcastable()) {
    return alignDim;
  }
  auto rank = hivmOp.getRank(hivmOp.getDpsInitOperand(0));
  auto origOTFBrcDims = hivmOp.getInlinedBroadcastableAxes(nullptr);
  bool isLastOTFBrc =
      llvm::find(origOTFBrcDims, rank - 1) != origOTFBrcDims.end();
  if (!isLastOTFBrc || rank <= 1) {
    return alignDim;
  }

  auto operMemType = dyn_cast<MemRefType>(operand.getType());
  auto operShape = operMemType.getShape();
  if (operShape.back() != 1) {
    return alignDim;
  }

  int64_t offset;
  SmallVector<int64_t> strides;
  if (getStridesAndOffset(operMemType, strides, offset).failed()) {
    return alignDim;
  }

  if (strides[rank - 2] != 1) {
    return alignDim;
  }

  auto hivmFlattenInterfaceOp =
      cast<hivm::FlattenInterface>(hivmOp.getOperation());
  FlattenOptions flattenOptions;
  flattenOptions.checkMarkStride = true;
  auto flattenResult = hivmFlattenInterfaceOp.getFlattened(flattenOptions);
  auto flattenAlignDim =
      getMapedAlignDim(alignDim, flattenResult->reassociation[0]);
  assert(flattenAlignDim.has_value() && "flattenAlignDim should have value");
  // adjust to find previous uncontiguous dimension to do alignment
  auto operTypes = hivmOp.getHIVMOperandTypes(/*includeExtraBuffer=*/false);
  auto memrefTypes = util::getMemRefTypes(operTypes);
  auto flattenedAssociations = flattenResult->reassociation[0];
  auto adjustAlignDim = getPrevUncontiguousDim(
      flattenAlignDim.value(), flattenedAssociations, memrefTypes);
  // TODO: check if it is needed
  if (isNoNeedAlign(operand, adjustAlignDim)) {
    return std::nullopt;
  }
  LDBG("adjust the alignDim of operand " << operand << " in hivmOp : " << hivmOp
                                         << " as belows\n");
  LDBG("adjustAlignDim : " << adjustAlignDim << "\n");
  return adjustAlignDim;
}

std::optional<int32_t> adjustAlignDim(Operation *op, Value operand,
                                      std::optional<int32_t> alignDim) {
  if (isNoNeedAlign(operand, alignDim)) {
    return std::nullopt;
  }

  if (auto implByScalarOp = dyn_cast<hivm::ImplByScalarOpInterface>(op)) {
    if (implByScalarOp.shouldLowerToScalarLoops()) {
      LDBG("no need to stride align because it will be decomposed to scalar "
           "operation later\n");
      return std::nullopt;
    }
  }

  if (auto brcOp = dyn_cast<hivm::VBrcOp>(op)) {
    return adjustBrcAlignDim(brcOp, operand, alignDim);
  } else if (auto reduceOp = dyn_cast<hivm::VReduceOp>(op)) {
    return adjustReduceAlignDim(reduceOp, operand, alignDim);
  } else if (auto deinterleaveOp = dyn_cast<hivm::VDeinterleaveOp>(op)) {
    return adjustDeInterleaveAlignDim(deinterleaveOp, operand, alignDim);
  } else if (auto loadOp = dyn_cast<hivm::LoadOp>(op)) {
    return loadOp.getMayImplicitTransposeWithLastAxis() ? std::nullopt
                                                        : alignDim;
  } else if (auto storeOp = dyn_cast<hivm::StoreOp>(op)) {
    return storeOp.getMayImplicitTransposeWithLastAxis() ? std::nullopt
                                                         : alignDim;
  } else if (auto copyOp = dyn_cast<hivm::CopyOp>(op)) {
    return adjustCopyAlignDim(copyOp, operand, alignDim);
  } else if (auto hivmOp = dyn_cast<HIVMStructuredOp>(op)) {
    if (hivmOp.isInlineBroadcastable()) {
      return adjustInlineBrcOp(hivmOp, operand, alignDim);
    }
  }
  return alignDim;
}

std::vector<std::pair<int32_t, int32_t>>
sortAlignInfo(ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes) {
  std::vector<std::pair<int32_t, int32_t>> alignInfos;
  for (auto [dim, byte] : llvm::zip(alignDims, alignBytes)) {
    alignInfos.push_back(std::make_pair(dim, byte));
  }
  std::sort(alignInfos.begin(), alignInfos.end(),
            [](auto l, auto r) { return l.first < r.first; });
  return alignInfos;
}

std::pair<llvm::SmallVector<int32_t>, llvm::SmallVector<int32_t>>
adjustAlignInfo(Operation *op, Value operand,
                const ArrayRef<int32_t> &alignDims,
                const ArrayRef<int32_t> &alignBytes) {
  llvm::SmallVector<int32_t> adjustedAlignDims;
  llvm::SmallVector<int32_t> adjustedAlignBytes;
  for (auto [alignDim, alignByte] : llvm::zip(alignDims, alignBytes)) {
    auto adjustedAlignDim = adjustAlignDim(op, operand, alignDim);
    if (!adjustedAlignDim.has_value()) {
      continue;
    }

    auto it = llvm::find(adjustedAlignDims, adjustedAlignDim.value());
    if (it == adjustedAlignDims.end()) {
      adjustedAlignDims.push_back(adjustedAlignDim.value());
      adjustedAlignBytes.push_back(alignByte);
    } else {
      auto pos = it - adjustedAlignDims.begin();
      adjustedAlignBytes[pos] = std::lcm(adjustedAlignBytes[pos], alignByte);
    }
  }

  if (adjustedAlignDims.empty()) {
    return std::make_pair(adjustedAlignDims, adjustedAlignBytes);
  }

  auto sortedInfo = sortAlignInfo(adjustedAlignDims, adjustedAlignBytes);
  SmallVector<int32_t> sortedAlignDims;
  SmallVector<int32_t> sortedAlignBytes;
  for (auto [sortedAlignDim, sortedAlignByte] : sortedInfo) {
    sortedAlignDims.push_back(sortedAlignDim);
    sortedAlignBytes.push_back(sortedAlignByte);
  }

  return std::make_pair(sortedAlignDims, sortedAlignBytes);
}

void dump(const ArrayRef<int32_t> &alignDims,
          const ArrayRef<int32_t> &alignBytes, StringRef debugType) {
  for (auto [alignDim, alignByte] : llvm::zip(alignDims, alignBytes)) {
    llvm::dbgs() << '[' << debugType << "] alignDim : " << alignDim
                 << ", alignByte : " << alignByte << "\n";
  }
}

static void collectOpAlignInfo(
    Operation *op, SmallVector<int64_t> checkDims,
    llvm::SmallDenseMap<Value, uint32_t> *alignBytes,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  assert(alignBytes != nullptr);
  for (auto oper : op->getOperands()) {
    auto elemTypeBytes = getElementTypeOrSelf(oper).getIntOrFloatBitWidth() / 8;
    auto shape = cast<ShapedType>(oper.getType()).getShape();
    for (auto checkDim : checkDims) {
      assert(checkDim >= 0 && checkDim < static_cast<int64_t>(shape.size()));
      assert((*alignBytes)[oper] != 0);
      if (ShapedType::isDynamic(shape[checkDim]) ||
          (shape[checkDim] * elemTypeBytes) % (*alignBytes)[oper] != 0) {
        auto operAlignInfo = std::make_unique<OperAlignInfo>(
            oper, checkDim, (*alignBytes)[oper]);
        operAlignInfoList->push_back(std::move(operAlignInfo));
      }
    }
  }
}

static LogicalResult getCastDstUnAlignSizeInfo(
    Value dst, SmallVector<int64_t> castAlignDims,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  // get alignment bytes
  ShapedType dstType = cast<ShapedType>(dst.getType());
  auto maybeHwAlignBytes = getHWAlignBytes(dstType);
  if (!maybeHwAlignBytes.has_value()) {
    return failure();
  }

  // collect unalign info of dst if cast dst dims are not aligned
  auto hwAlignBytes = maybeHwAlignBytes.value();
  auto elemTypeBytes = getElementTypeOrSelf(dstType).getIntOrFloatBitWidth() /
                       mlir::utils::INTR_BITS_PER_BYTE;
  assert(elemTypeBytes == 1 && "Dst only supports b8 in cast overflow.");
  auto shape = cast<ShapedType>(dst.getType()).getShape();
  uint64_t numElemPerBlock = mlir::utils::INTR_BYTES_PER_BLOCK / elemTypeBytes;
  int64_t rank = dstType.getRank();
  // For example (a, b)strides<n1, 1>*i32 cast to (a, b)strides<n2, 1>*i8:
  // 1. (a, b)strides<n1, 1>*i32 view as (a, b*4)strides<n1*4, 1>*i8
  // 2. i8 transpose: Used to separate the high and low bits of int32, make sure
  //    the shape of tranpose is 32*32 aligned.
  // 3. i8 copyubtoub: Take out the lower 8 bits.
  // 4. i8 transpose: Transpose back to get the final cast result, make sure the
  //    shape of tranpose is 32*32 aligned.
  // 5. When n2 is aligned with multiple blocks, need to add another copyubtoub
  //    to adjust it to the target stride.
  if (rank == 1) {
    // The 1D scene is quite special and needs to be converted into a
    // corresponding 2D scene to implement.
    hwAlignBytes = numElemPerBlock * numElemPerBlock;
  }
  for (auto checkDim : castAlignDims) {
    if (ShapedType::isDynamic(shape[checkDim]) ||
        (static_cast<uint64_t>(shape[checkDim]) * elemTypeBytes) %
                hwAlignBytes !=
            0) {
      auto dstAlignInfo =
          std::make_unique<OperAlignInfo>(dst, checkDim, hwAlignBytes);
      operAlignInfoList->push_back(std::move(dstAlignInfo));
    }
  }
  return success();
}

static LogicalResult getCastSrcUnAlignSizeInfo(
    Value src, SmallVector<int64_t> castAlignDims, int64_t bytesFactor,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  // get alignment bytes
  ShapedType srcType = cast<ShapedType>(src.getType());
  auto maybeHwAlignBytes = getHWAlignBytes(srcType);
  if (!maybeHwAlignBytes.has_value()) {
    return failure();
  }

  // collect unalign info of src if cast src dims are not aligned.
  auto hwAlignBytes = maybeHwAlignBytes.value();
  auto elemTypeBytes = getElementTypeOrSelf(srcType).getIntOrFloatBitWidth() /
                       mlir::utils::INTR_BITS_PER_BYTE;
#ifndef NDEBUG
  const int b16InByte = 2;
  const int b32InByte = 4;
  assert((elemTypeBytes == b16InByte || elemTypeBytes == b32InByte) &&
         "Src only supports b32/b16 in cast overflow.");
#endif
  auto shape = cast<ShapedType>(src.getType()).getShape();
  int64_t numElemPerBlock = mlir::utils::INTR_BYTES_PER_BLOCK / elemTypeBytes;
  int64_t numElemPerBlockForDst = numElemPerBlock * bytesFactor;
  int64_t rank = srcType.getRank();
  // For example (a, b)strides<n1, 1>*i32 cast to (a, b)strides<n2, 1>*i8:
  // 1. (a, b)strides<n1, 1>*i32 view as (a, b*4)strides<n1*4, 1>*i8
  // 2. i8 transpose: Used to separate the high and low bits of int32, make sure
  //    the shape of tranpose is 32*32 aligned.
  // 3. i8 copyubtoub: Take out the lower 8 bits.
  // 4. i8 transpose: Transpose back to get the final cast result, make sure the
  //    shape of tranpose is 32*32 aligned.
  // 5. When n2 is aligned with multiple blocks, need to add another copyubtoub
  //    to adjust it to the target stride.
  if (rank == 1) {
    // The 1D scene is quite special and needs to be converted into a
    // corresponding 2D scene to implement.
    if (!ShapedType::isDynamic(shape[0]) && shape[0] <= numElemPerBlockForDst) {
      hwAlignBytes = static_cast<unsigned>(
          CEIL_FACTOR(shape[0] * bytesFactor, numElemPerBlockForDst) *
          numElemPerBlockForDst);
    } else {
      hwAlignBytes = static_cast<unsigned>(numElemPerBlockForDst *
                                           numElemPerBlockForDst * bytesFactor);
    }
    if (ShapedType::isDynamic(shape[0]) ||
        (shape[0] * elemTypeBytes) % hwAlignBytes != 0) {
      auto srcAlignInfo = std::make_unique<OperAlignInfo>(src, 0, hwAlignBytes);
      operAlignInfoList->push_back(std::move(srcAlignInfo));
    }
  } else {
#ifndef NDEBUG
    const int supportedCastAlignDimSize = 2;
    assert(castAlignDims.size() == supportedCastAlignDimSize &&
           "When cast rank >= 2, castAlignDims size must be equal to 2");
#endif
    // Align the second axis in castAlignDims.
    if (ShapedType::isDynamic(shape[castAlignDims[1]]) ||
        (shape[castAlignDims[1]] * elemTypeBytes) % hwAlignBytes != 0) {
      auto srcAlignInfo =
          std::make_unique<OperAlignInfo>(src, castAlignDims[1], hwAlignBytes);
      operAlignInfoList->push_back(std::move(srcAlignInfo));
    }
    // Align the first axis in castAlignDims.
    hwAlignBytes = static_cast<unsigned>(numElemPerBlockForDst * bytesFactor);
    if (ShapedType::isDynamic(shape[castAlignDims[0]]) ||
        (static_cast<uint64_t>(shape[castAlignDims[0]]) * elemTypeBytes) %
                hwAlignBytes !=
            0) {
      auto srcAlignInfo =
          std::make_unique<OperAlignInfo>(src, castAlignDims[0], hwAlignBytes);
      operAlignInfoList->push_back(std::move(srcAlignInfo));
    }
  }
  return success();
}

LogicalResult getUnAlignSizeInfo(
    VTransposeOp op,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  // get alignment bytes
  auto srcType = op.getSrc().getType();
  auto maybeHwAlignBytes = getHWAlignBytes(srcType);
  if (!maybeHwAlignBytes.has_value()) {
    return failure();
  }

  // get transpose loop dims
  SmallVector<int64_t> transposeLoopDims;
  op.getTransposeLoopDims(transposeLoopDims);

  // collect unalign info of all operands if transpose dims are not aligned
  auto hwAlignBytes = maybeHwAlignBytes.value();
  llvm::SmallDenseMap<Value, uint32_t> operHwAlignBytes;
  for (auto oper : op->getOperands()) {
    operHwAlignBytes[oper] = hwAlignBytes;
  }
  collectOpAlignInfo(op.getOperation(), transposeLoopDims, &operHwAlignBytes,
                     operAlignInfoList);

  auto elemTypeBytes = getElementTypeOrSelf(srcType).getIntOrFloatBitWidth() /
                       mlir::utils::INTR_BITS_PER_BYTE;
  const int b32InByte = 4;
  if (elemTypeBytes != b32InByte) {
    return success();
  }

  // when it is B32 type, judge if there is one dim that is already double
  // aligned. if not, should choose one dim to do double alignment, e.g.
  // 8x16xf32 or 16x8xf32.
  auto srcShape = cast<ShapedType>(op.getSrc().getType()).getShape();
  bool isAlreadyDoubleAlign = false;
  for (auto transDim : transposeLoopDims) {
    auto alignedSrcDimBytes =
        CEIL_FACTOR(static_cast<uint64_t>(srcShape[transDim]) * elemTypeBytes,
                    hwAlignBytes);
    if (alignedSrcDimBytes % (hwAlignBytes * 2) == 0) {
      isAlreadyDoubleAlign = true;
    }
  }
  if (isAlreadyDoubleAlign) {
    return success();
  }

  // must choose double align dim from two transpose dims
  if (transposeLoopDims.size() != 2) {
    // For B32, do transpose decompose first
    return failure();
  }

  operAlignInfoList->clear();
  // choose transdim 0 as double align dim
  auto srcTrans0AlignInfo = std::make_unique<OperAlignInfo>(
      op.getSrc(), transposeLoopDims[0], hwAlignBytes);
  operAlignInfoList->push_back(std::move(srcTrans0AlignInfo));
  auto srcTrans1AlignInfo = std::make_unique<OperAlignInfo>(
      op.getSrc(), transposeLoopDims[1], hwAlignBytes * 2);
  operAlignInfoList->push_back(std::move(srcTrans1AlignInfo));

  auto dstTrans0AlignInfo = std::make_unique<OperAlignInfo>(
      op.getDst(), transposeLoopDims[0], hwAlignBytes * 2);
  operAlignInfoList->push_back(std::move(dstTrans0AlignInfo));
  auto dstTrans1AlignInfo = std::make_unique<OperAlignInfo>(
      op.getDst(), transposeLoopDims[1], hwAlignBytes);
  operAlignInfoList->push_back(std::move(dstTrans1AlignInfo));
  return success();
}

LogicalResult getUnAlignSizeInfo(
    VCastOp op,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  auto srcType = cast<ShapedType>(op.getSrc()[0].getType());
  auto dstType = cast<ShapedType>(op.getDst()[0].getType());
  auto srcElemTypeBytes =
      getElementTypeOrSelf(srcType).getIntOrFloatBitWidth() /
      mlir::utils::INTR_BITS_PER_BYTE;
  auto dstElemTypeBytes =
      getElementTypeOrSelf(dstType).getIntOrFloatBitWidth() /
      mlir::utils::INTR_BITS_PER_BYTE;
  auto bytesFactor = srcElemTypeBytes / dstElemTypeBytes;

  // Get the cast axis that needs to be aligned.
  SmallVector<int64_t> castAlignDims;
  int64_t rank = srcType.getRank();
  if (rank == 1) {
    castAlignDims.push_back(0);
  } else if (rank >= 2) {
    castAlignDims.push_back(rank - 2);
    castAlignDims.push_back(rank - 1);
  } else {
    llvm_unreachable("cast op rank need lager than 0.");
  }

  // Get the unalign information of the axis corresponding to cast src.
  if (failed(getCastSrcUnAlignSizeInfo(op.getSrc()[0], castAlignDims,
                                       bytesFactor, operAlignInfoList))) {
    return failure();
  }
  // Get the unalign information of the axis corresponding to cast dst.
  if (failed(getCastDstUnAlignSizeInfo(op.getDst()[0], castAlignDims,
                                       operAlignInfoList))) {
    return failure();
  }
  return success();
}

LogicalResult getUnAlignSizeInfo(
    VSortOp op,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  // get alignment bytes
  ShapedType srcType = cast<ShapedType>(op.getSrc().getType());
  auto maybeHwAlignBytes = getHWAlignBytes(srcType);
  if (!maybeHwAlignBytes.has_value()) {
    return failure();
  }

  // Get the sort axis that needs to be aligned.
  SmallVector<int64_t> sortAlignDims;
  int64_t rank = srcType.getRank();
  sortAlignDims.push_back(rank - 1);

  llvm::SmallDenseMap<Value, uint32_t> operHwAlignBytes;
  for (auto oper : op->getOperands()) {
    ShapedType operType = cast<ShapedType>(oper.getType());
    auto elemTypeBytes =
        getElementTypeOrSelf(operType).getIntOrFloatBitWidth() / 8;
    unsigned int numElemPerBlock =
        mlir::utils::INTR_BYTES_PER_BLOCK / elemTypeBytes;
    operHwAlignBytes[oper] =
        maybeHwAlignBytes.value() * (VBITSORT_NUM_PER_REPEAT / numElemPerBlock);
  }
  collectOpAlignInfo(op.getOperation(), sortAlignDims, &operHwAlignBytes,
                     operAlignInfoList);
  return success();
}

} // namespace hivm
} // namespace mlir
