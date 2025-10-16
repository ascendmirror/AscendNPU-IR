//===--- NormalizeTensorOps.cpp -  optimize tensor ops --------------------===//
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
// This file implements a pass to canonicalize pad operation sourced from
// Insert_slice operation
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

namespace mlir {
#define GEN_PASS_DEF_NORMALIZETENSOROPS
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "normalize-tensor-ops"

using namespace mlir;
using namespace mlir::tensor;

namespace {
struct NormalizeTensorOps
    : public impl::NormalizeTensorOpsBase<NormalizeTensorOps> {
  explicit NormalizeTensorOps(bool skipAlignedSlice)
      : skipAlignedSlice(skipAlignedSlice) {}
  void runOnOperation() override;

private:
  // option for `FoldInsertSliceToConcat` to decide whether we can skip folding
  // aligned insert_slice to concat
  bool skipAlignedSlice{false};
};
} // namespace

bool isStatic(llvm::ArrayRef<int64_t> arr, unsigned int padrank) {
  return arr.size() == padrank && llvm::none_of(arr, [](int64_t val) {
           return ShapedType::isDynamic(val);
         });
}

LogicalResult canFoldInsertPad(tensor::PadOp padOp, PatternRewriter &rewriter) {
  auto insertOp = padOp.getSource().getDefiningOp<InsertSliceOp>();
  if (!insertOp)
    return rewriter.notifyMatchFailure(padOp, "not defined by insert");
  // only apply if stride is 1
  auto insertStride = insertOp.getStaticStrides();
  if (!llvm::all_of(insertStride,
                    [&](int64_t sValue) { return sValue == 1; })) {
    return rewriter.notifyMatchFailure(insertOp, "stride not ones");
  }
  // Check if insert destination is uniform tensor (All elements same)
  auto insertDest = insertOp.getDest();
  Operation *defOp = insertDest.getDefiningOp();
  if (!defOp) {
    return rewriter.notifyMatchFailure(insertOp,
                                       "insert destination not defined");
  }
  auto constantOp = dyn_cast_if_present<arith::ConstantOp>(defOp);
  auto fillOp = dyn_cast_if_present<linalg::FillOp>(defOp);
  Value padValue = padOp.getConstantPaddingValue();
  if (!padValue)
    return rewriter.notifyMatchFailure(padOp, "pad value not constant");
  auto padConstantOp = padValue.getDefiningOp<arith::ConstantOp>();
  if (!padConstantOp)
    return rewriter.notifyMatchFailure(padOp, "pad value not defined constant");
  Attribute padAttr = padConstantOp.getValue();
  if (constantOp) {
    auto constValue = constantOp.getValue();
    auto denseAttr = dyn_cast_if_present<DenseIntOrFPElementsAttr>(constValue);
    if (!denseAttr || !denseAttr.isSplat()) {
      return rewriter.notifyMatchFailure(insertOp,
                                         "insert destination not splat");
    }
    auto splatAttr = denseAttr.getSplatValue<TypedAttr>();
    if (padAttr != splatAttr) {
      return rewriter.notifyMatchFailure(padOp, "pad value doesnt match splat");
    }
  } else if (fillOp) {
    auto fillValue = fillOp.value();
    if (padValue != fillValue) {
      return rewriter.notifyMatchFailure(padOp, "pad value doesnt match fill");
    }
  } else {
    return rewriter.notifyMatchFailure(
        insertOp, "insert destination not constant or fill operation");
  }
  return success();
}

/// This pattern identifies pad operations where the source is a insert_slice
/// operation where the destination of the insert is a uniform tensor
/// in this case insert can be treated as general pad and fold with the old
/// padding into a single padding, if the pad value is also the same
/// %fill = linalg.fill ins(%cst : f32) outs(%0 :tensor<6140xf32>)->tensor<f32>
/// %inserted_slice = tensor.insert_slice %source into %fill[2046] [2047] [1] :
/// tensor<2047xf32> into tensor<6140xf32>
/// %padded = tensor.pad %inserted_slice low[0] high[-2047]{pad value = %cst} :
/// tensor<6140xf32> to tensor<4093xf32>
/// into:
/// %padded = tensor.pad %source low[2046] high[0] {pad value = %cst} :
/// tensor<2047xf32> to tensor<4093xf32>
struct FoldInsertPadPattern : public OpRewritePattern<tensor::PadOp> {
public:
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    if (failed(canFoldInsertPad(padOp, rewriter)))
      return failure();

    auto insertOp = padOp.getSource().getDefiningOp<InsertSliceOp>();
    auto newSource = insertOp.getSource();
    auto insertDest = insertOp.getDest();
    auto destType = insertDest.getType();
    int64_t rank = destType.getRank();
    SmallVector<OpFoldResult> newHigh;
    SmallVector<OpFoldResult> newLow;
    SmallVector<int64_t> newHighInt;
    SmallVector<int64_t> newLowInt;
    llvm::ArrayRef<int64_t> oldHighStatic = padOp.getStaticHigh();
    llvm::ArrayRef<int64_t> oldLowStatic = padOp.getStaticLow();
    llvm::ArrayRef<int64_t> insertOffsets = insertOp.getStaticOffsets();
    bool isStaticOffset = isStatic(insertOffsets, rank);
    bool isStaticLow = isStatic(oldHighStatic, rank);
    bool isStaticHigh = isStatic(oldLowStatic, rank);
    if (!isStaticLow || !isStaticHigh || !isStaticOffset)
      return rewriter.notifyMatchFailure(insertOp,
                                         "offsets or pad nums not static");
    for (unsigned i = 0; i < rank; ++i) {
      // Insert + pad becomes pad with :
      int64_t offsetCurrent = insertOp.getStaticOffset(i);
      // low = offset + oldLowPad
      auto newLowElement = offsetCurrent + oldLowStatic[i];
      // high = InsertDimSize - offset - insertSize + oldPadHigh
      auto newHighElement = destType.getDimSize(i) - offsetCurrent -
                            insertOp.getStaticSize(i) + oldHighStatic[i];
      newHighInt.push_back(newHighElement);
      newLowInt.push_back(newLowElement);
    }
    newHigh = getAsIndexOpFoldResult(rewriter.getContext(), newHighInt);
    newLow = getAsIndexOpFoldResult(rewriter.getContext(), newLowInt);
    auto padLoc = insertOp.getLoc();
    auto resultType = padOp.getResult().getType();
    Value padValue = padOp.getConstantPaddingValue();
    auto newPadOp = rewriter.create<PadOp>(padLoc, resultType, newSource,
                                           newLow, newHigh, padValue);
    rewriter.replaceOp(padOp, newPadOp.getResult());
    return success();
  }
};

/// This pattern identifies a pad operation whose source is another pad
/// we can fold it into single pad with summed pad values
/// %padded = tensor.pad %source low[2046] high[2047] {pad value = %cst} :
/// tensor<2047xf32> to tensor<6140xf32>
/// %padded2 = tensor.pad %padded low[0] high[-2047] {pad value = %cst} :
/// tensor<6140xf32> to tensor<4093xf32>
/// becomes:
/// %padded = tensor.pad %source low[2046] high[0] {pad value = %cst} :
/// tensor<2047xf32> to tensor<4093xf32>
struct FoldDoublePadPattern : public OpRewritePattern<tensor::PadOp> {
public:
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    auto oldpadOp = padOp.getSource().getDefiningOp<tensor::PadOp>();
    if (!oldpadOp)
      return rewriter.notifyMatchFailure(padOp, "not a defined by other pad");

    // We can only fold if the padding value is the same as the dst splat value
    Value newPadValue = padOp.getConstantPaddingValue();
    if (!newPadValue)
      return rewriter.notifyMatchFailure(padOp, "pad value not constant");
    Value oldPadValue = padOp.getConstantPaddingValue();
    if (!oldPadValue)
      return rewriter.notifyMatchFailure(padOp, "pad value not constant");
    if (oldPadValue != newPadValue) {
      return rewriter.notifyMatchFailure(padOp, "pad values are not equal");
    }

    unsigned padrank = static_cast<unsigned>(padOp.getResultType().getRank());
    unsigned oldPadrank =
        static_cast<unsigned>(oldpadOp.getResultType().getRank());
    if (padrank != oldPadrank)
      return failure();
    SmallVector<long> newTotalHigh;
    SmallVector<long> newTotalLow;
    SmallVector<OpFoldResult> newHighFold;
    SmallVector<OpFoldResult> newLowFold;
    llvm::ArrayRef<int64_t> oldHighStatic = oldpadOp.getStaticHigh();
    llvm::ArrayRef<int64_t> oldLowStatic = oldpadOp.getStaticLow();
    llvm::ArrayRef<int64_t> newHighStatic = padOp.getStaticHigh();
    llvm::ArrayRef<int64_t> newLowStatic = padOp.getStaticLow();
    bool isStaticLow = isStatic(oldLowStatic, oldPadrank);
    bool isStaticHigh = isStatic(oldHighStatic, oldPadrank);
    bool isStaticLowToo = isStatic(newLowStatic, padrank);
    bool isStaticHighToo = isStatic(newHighStatic, padrank);
    if (!isStaticLow || !isStaticHigh || !isStaticLowToo || !isStaticHighToo)
      return rewriter.notifyMatchFailure(padOp, "Pad not static");

    for (unsigned i = 0; i < padrank; ++i) {
      auto newLowElement = newLowStatic[i] + oldLowStatic[i];
      auto newHighElement = newHighStatic[i] + oldHighStatic[i];
      newTotalHigh.push_back(newHighElement);
      newTotalLow.push_back(newLowElement);
    }
    newHighFold = getAsIndexOpFoldResult(rewriter.getContext(), newTotalHigh);
    newLowFold = getAsIndexOpFoldResult(rewriter.getContext(), newTotalLow);
    auto padLoc = padOp.getLoc();
    auto resultType = padOp.getResult().getType();
    auto newSource = oldpadOp.getSource();
    auto newPadOp =
        rewriter.create<PadOp>(padLoc, resultType, newSource, newLowFold,
                               newHighFold, newPadValue, padOp.getNofold());
    rewriter.replaceOp(padOp, newPadOp.getResult());

    return success();
  }
};

struct FoldStaticNegativeHighPad : public OpRewritePattern<tensor::PadOp> {
public:
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    ArrayRef<int64_t> highs = padOp.getStaticHigh();
    if (llvm::any_of(highs, [](int64_t s) {
          return ShapedType::isDynamic(s) || s > 0;
        })) {
      // TODO: support cases with mixed positive/negative high paddings.
      // e.g.
      // tensor.pad %arg low[0, 0, 0] high[0, 1, -1] :
      //                 tensor<8x8x8xf32> to tensor<8x9x7xf32>
      // ==>
      // %slice = tensor.extract_slice %arg [0, 0, 0] [0, 0, 7]
      //                 tensor<8x8x8xf32> to tensor<8x8x7xf32>
      // tensor.pad %slice low[0, 0, 0] high[0, 1, 0] :
      //                 tensor<8x8x7xf32> to tensor<8x9x7xf32>
      return rewriter.notifyMatchFailure(
          padOp, "cannot fold pad with dynamic or positive high padding num");
    }

    ArrayRef<int64_t> sizes = padOp.getSourceType().getShape();
    if (ShapedType::isDynamicShape(sizes)) {
      return rewriter.notifyMatchFailure(
          padOp, "cannot fold pad with dynamic source size");
    }

    ArrayRef<int64_t> lows = padOp.getStaticLow();
    if (llvm::any_of(lows, [](int64_t s) { return s != 0; })) {
      return rewriter.notifyMatchFailure(
          padOp, "only fold pad with zero low padding num");
    }

    if (llvm::none_of(highs, [](int64_t s) { return s < 0; })) {
      return rewriter.notifyMatchFailure(
          padOp, "only fold pad with static negative high padding num");
    }

    int64_t rank = padOp.getSourceType().getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> newSizes;
    for (int64_t i = 0; i < rank; ++i) {
      int64_t curSize = sizes[i] + highs[i];
      newSizes.push_back(rewriter.getIndexAttr(curSize));
    }

    auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        padOp->getLoc(), padOp.getResultType(), padOp.getSource(), offsets,
        newSizes, strides);
    rewriter.replaceOp(padOp, newSliceOp.getResult());
    return success();
  }
};

struct FoldInsertSliceToConcat
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  explicit FoldInsertSliceToConcat(mlir::MLIRContext *ctx,
                                   bool skipAlignedSlice = false)
      : OpRewritePattern<tensor::InsertSliceOp>(ctx),
        skipAlignedSlice(skipAlignedSlice) {}

  LogicalResult matchAndRewrite(tensor::InsertSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    // sliceOp must have strides size one
    ArrayRef<int64_t> strides = sliceOp.getStaticStrides();
    if (llvm::any_of(strides, [](int64_t s) { return s != 1; })) {
      return rewriter.notifyMatchFailure(
          sliceOp, "only can fold insert_slice with strides equal to one");
    }

    // slice op must have static shaped src and dst tensor with same rank
    RankedTensorType srcType = sliceOp.getSourceType();
    RankedTensorType dstType = sliceOp.getResultType();
    if (srcType.getRank() != dstType.getRank()) {
      // insert_slice supports insertion into a tensor of higher rank than the
      // source tensor, we should avoid this situation
      return rewriter.notifyMatchFailure(
          sliceOp, "cannot fold insert_slice with different src and dst rank");
    }

    // find out on which dimension to concat the slices
    auto sliceDimMaybe = getUniqueConcatDim(sliceOp);
    if (!sliceDimMaybe.has_value()) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "should slice exactly one dimension");
    }

    int64_t sliceDim = sliceDimMaybe.value();
    if (skipAlignedSlice && isOffsetsAligned(sliceOp, sliceDim)) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "not fold slice with aligned offsets");
    }

    Value newConcat = convertInsertSliceToConcatOp(sliceOp, sliceDim, rewriter);
    rewriter.replaceOp(sliceOp, {newConcat});
    return success();
  }

private:
  bool isOffsetsAligned(InsertSliceOp sliceOp, int64_t sliceDim) const {
    ArrayRef<int64_t> offsets = sliceOp.getStaticOffsets();
    ArrayRef<int64_t> sizes = sliceOp.getStaticSizes();
    int64_t rank = sliceOp.getResultType().getRank();
    int64_t bytesPerElem =
        sliceOp.getResultType().getElementTypeBitWidth() / utils::kBitsToByte;
    int64_t accumSize = 1;
    for (int64_t dim = rank - 1; dim > sliceDim; dim--) {
      accumSize *= sizes[dim];
    }
    int64_t offset =
        (ShapedType::isDynamic(offsets[sliceDim]) ? 1 : offsets[sliceDim]);
    int64_t accumOffsetInBytes = offset * accumSize * bytesPerElem;
    return accumOffsetInBytes % utils::INTR_BYTES_PER_BLOCK == 0;
  }

  void appendIfValueNotEmpty(SmallVector<Value> &vec, Value v) const {
    // helper func to make sure not concat zero sized tensor.
    // v will be empty Value() if it is zero sized tensor.
    if (v) {
      vec.push_back(v);
    }
  }

  std::optional<int64_t>
  getUniqueConcatDim(tensor::InsertSliceOp sliceOp) const {
    RankedTensorType dstType = sliceOp.getResultType();
    SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();

    int64_t rank = dstType.getRank();
    int64_t sliceDim = -1;
    for (int64_t dim = 0; dim < rank; ++dim) {
      bool insertFullDim = true;
      if (dstType.isDynamicDim(dim)) {
        // conservatively treat dynamic dim insert_slice not full slice,
        // since we do not analyze size equivalence
        insertFullDim = false;
      } else {
        insertFullDim = isConstantIntValue(sizes[dim], dstType.getDimSize(dim));
      }

      if (isConstantIntValue(offsets[dim], 0) && insertFullDim) {
        // if slice the whole dimension, not interested
        continue;
      }
      // according to the definition of torch scatter_slice, there should only
      // be one dimension to slice
      if (sliceDim == -1) {
        sliceDim = dim;
      } else {
        return std::nullopt;
      }
    }
    if (sliceDim == -1) {
      // no slice dim found
      return std::nullopt;
    }
    return sliceDim;
  }

  SmallVector<OpFoldResult>
  cloneWithSubstitution(ArrayRef<OpFoldResult> data, int64_t substituteDim,
                        OpFoldResult substituteData,
                        PatternRewriter &rewriter) const {
    SmallVector<OpFoldResult> result(data.begin(), data.end());
    result[substituteDim] = substituteData;
    return result;
  }

  // extract slices from sliceOp src/dst and concat
  Value convertInsertSliceToConcatOp(tensor::InsertSliceOp sliceOp,
                                     int64_t sliceDim,
                                     PatternRewriter &rewriter) const {
    Location loc = sliceOp->getLoc();
    SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = sliceOp.getMixedStrides();
    auto sliceFrom = [&](Value from, OpFoldResult curOffset,
                         OpFoldResult curSize) -> Value {
      if (isConstantIntValue(curSize, 0)) {
        // should not extract and concat zero sized tensor.
        // if zero sized, return empty Value()
        return Value();
      }
      SmallVector<OpFoldResult> newOffsets =
          cloneWithSubstitution(offsets, sliceDim, curOffset, rewriter);
      SmallVector<OpFoldResult> newSizes =
          cloneWithSubstitution(sizes, sliceDim, curSize, rewriter);
      SmallVector<OpFoldResult> newStrides = cloneWithSubstitution(
          strides, sliceDim, rewriter.getIndexAttr(1), rewriter);
      auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
          loc, from, newOffsets, newSizes, newStrides);
      return newSliceOp;
    };
    OpFoldResult sliceOffset = offsets[sliceDim];
    OpFoldResult sliceSize = sizes[sliceDim];
    OpFoldResult sliceDstSize =
        tensor::getMixedSize(rewriter, loc, sliceOp.getResult(), sliceDim);
    Value src = sliceOp.getSource();
    Value dst = sliceOp->getOperand(1);
    SmallVector<Value> concatInputs;
    // extract `0 ~ offset` from dst tensor
    appendIfValueNotEmpty(
        concatInputs, sliceFrom(dst, rewriter.getIndexAttr(0), sliceOffset));
    // extract `offset ~ offset+size` from src tensor, exactly src tensor
    appendIfValueNotEmpty(concatInputs, src);
    // extract `offset + size ~ sliceDstSize` from dst tensor
    AffineExpr dim0;
    AffineExpr dim1;
    bindDims(rewriter.getContext(), dim0, dim1);
    auto subMap = AffineMap::get(2, 0, {dim0 - dim1});
    auto sub = [&](OpFoldResult v1, OpFoldResult v2) {
      return affine::makeComposedFoldedAffineApply(rewriter, loc, subMap,
                                                   {v1, v2});
    };
    auto addMap = AffineMap::get(2, 0, {dim0 + dim1});
    auto add = [&](OpFoldResult v1, OpFoldResult v2) {
      return affine::makeComposedFoldedAffineApply(rewriter, loc, addMap,
                                                   {v1, v2});
    };
    appendIfValueNotEmpty(
        concatInputs,
        sliceFrom(dst, add(sliceOffset, sliceSize),
                  sub(sliceDstSize, add(sliceOffset, sliceSize))));
    return rewriter.create<tensor::ConcatOp>(loc, sliceOp.getResultType(),
                                             sliceDim, concatInputs);
  }

  bool skipAlignedSlice{false};
};

struct NormalizeLastDimConcatToInterleave
    : public OpRewritePattern<tensor::ConcatOp> {
public:
  using OpRewritePattern<tensor::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    // TODO: remove this check after support interleave with channel_num >= 2
    if (concatOp->getNumOperands() != 2) {
      return rewriter.notifyMatchFailure(
          concatOp, "hfusion interleave op currently only supports channel num "
                    "equals to 2");
    }

    int64_t concatDim = static_cast<int64_t>(concatOp.getDim());
    int64_t rank = concatOp.getRank();
    if (concatDim != rank - 1) {
      return rewriter.notifyMatchFailure(
          concatOp, "only can normalize last dim concat op to interleave op");
    }

    SmallVector<Value> inputs = concatOp.getInputs();
    if (!llvm::all_of(inputs, [&](Value value) {
          return isTensorWithOneSizeDim(value, concatDim);
        })) {
      return rewriter.notifyMatchFailure(
          concatOp, "only can normalize last dim concat op to interleave op if "
                    "last dim is one size");
    }

    Location loc = concatOp->getLoc();
    auto newOp = rewriter.create<hfusion::InterleaveOp>(
        loc, ValueRange{concatOp->getResults()}, ValueRange{inputs});
    rewriter.replaceOp(concatOp, newOp);
    return success();
  }

private:
  bool isTensorWithOneSizeDim(Value value, int64_t dim) const {
    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType) {
      return false;
    }
    return tensorType.getDimSize(dim) == 1;
  }
};

struct FoldGenerateToFill : public OpRewritePattern<tensor::GenerateOp> {
public:
  using OpRewritePattern<tensor::GenerateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::GenerateOp genOp,
                                PatternRewriter &rewriter) const override {
    Block &body = genOp.getBody().front();
    if (!llvm::hasSingleElement(body))
      return rewriter.notifyMatchFailure(genOp, "Body not single element");

    auto yieldOp = dyn_cast<tensor::YieldOp>(body.getTerminator());
    if (!yieldOp)
      return rewriter.notifyMatchFailure(genOp, "Terminator not a YieldOp");

    Value fillValue =
        yieldOp.getValue(); // TODO: Might need some check if constant scalar
    auto dynSizes = genOp.getDynamicExtents();
    auto resType = genOp.getType();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        genOp.getLoc(), cast<RankedTensorType>(resType).getShape(),
        cast<RankedTensorType>(resType).getElementType(), ValueRange(dynSizes));
    auto fillOp = rewriter.create<linalg::FillOp>(
        genOp.getLoc(), ValueRange(fillValue), ValueRange(emptyOp));
    rewriter.replaceOp(genOp, fillOp);
    return success();
  }
};

void NormalizeTensorOps::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<FoldInsertPadPattern>(patterns.getContext());
  patterns.insert<FoldDoublePadPattern>(patterns.getContext());
  // TODO: we can disable it if insert_slice implements tiling interface
  patterns.insert<FoldInsertSliceToConcat>(patterns.getContext(),
                                           this->skipAlignedSlice);
  patterns.insert<FoldGenerateToFill>(patterns.getContext());
  patterns.insert<NormalizeLastDimConcatToInterleave>(patterns.getContext());
  patterns.insert<FoldStaticNegativeHighPad>(patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass>
mlir::tensor::createNormalizeTensorOpsPass(bool skipAlignedSlice) {
  return std::make_unique<NormalizeTensorOps>(skipAlignedSlice);
}
