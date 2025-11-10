//===- TileAndBindSubBlock.cpp---------------------------------------------===//
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
// This pass tiles and binds sub block for mix cv function.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/Pattern.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/Helper.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Passes.h"
#include "bishengir/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <utility>

namespace mlir {
#define GEN_PASS_DEF_TILEANDBINDSUBBLOCK
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-bind-sub-block"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
static constexpr llvm::StringLiteral kLimitedSubBlockOpAttrName =
    "limit_sub_block_id0";
static constexpr llvm::StringLiteral kTiledOp = "tiled_op";
} // namespace

namespace {

struct TileAndBindSubBlockPass
    : public impl::TileAndBindSubBlockBase<TileAndBindSubBlockPass> {
  using Base::Base;
  FailureOr<func::FuncOp> attemptBindSubBlock(func::FuncOp func);
  void runOnOperation() override;
};
} // namespace

// This function is from TileUsingInterface.cpp
// Check if `stride` evenly divides the trip count `size - offset`.
static bool tileDividesIterationDomain(Range loopRange) {
  std::optional<int64_t> offsetAsInt = getConstantIntValue(loopRange.offset);
  if (!offsetAsInt)
    return false;
  std::optional<int64_t> sizeAsInt = getConstantIntValue(loopRange.size);
  if (!sizeAsInt)
    return false;
  std::optional<int64_t> strideAsInt = getConstantIntValue(loopRange.stride);
  if (!strideAsInt)
    return false;
  return ((sizeAsInt.value() - offsetAsInt.value()) % strideAsInt.value() == 0);
}

/// This function is from TileUsingInterface.cpp.cpp
/// Returns the bounded tile size given the current `iv`, `loopRange` and
/// `tileSize`, i.e., `min(tileSize, range.end() - iv)`.
static OpFoldResult getBoundedTileSize(RewriterBase &rewriter, Location loc,
                                       Range loopRange, Value iv,
                                       OpFoldResult tileSize) {
  ImplicitLocOpBuilder::InsertionGuard g(rewriter);
  std::optional<int64_t> ts = getConstantIntValue(tileSize);
  if (ts && ts.value() == 1)
    return tileSize;

  if (tileDividesIterationDomain(
          Range{loopRange.offset, loopRange.size, tileSize}))
    return tileSize;

  // The tile size to use (to avoid out of bounds access) is  minimum of
  // `tileSize` and `ub - iv`, where `iv` is the induction variable of the tiled
  // loop.
  // Since the sub block loop is normalized (with lb=0, ub=2, step=1), we have
  // to multiple the `iv` with the `tileSize`.
  AffineExpr s0, s1, d0;
  bindDims(rewriter.getContext(), d0);
  bindSymbols(rewriter.getContext(), s0, s1);
  AffineMap minMap =
      AffineMap::get(1, 2, {s0, s1 - (d0 * s0)}, rewriter.getContext());
  Value size = getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.size);
  return affine::makeComposedFoldedAffineMin(
      rewriter, loc, minMap, SmallVector<OpFoldResult>{iv, tileSize, size});
}

/// This function calculates the tile size by dividing the dimension size
/// by kSubBlockDim (using ceiling division).
///
/// For static dimensions: tile_size = ceil(dim_size / kSubBlockDim)
/// For dynamic dimensions: creates affine operations to compute at runtime
///
/// @param rewriter The rewriter.
/// @param loc The location.
/// @param dimSize The original dimension size.
/// @return The computed tile size as an OpFoldResult, or failure if the
///         static dimension size is less than kSubBlockDim
static FailureOr<OpFoldResult>
getSingleTileSize(RewriterBase &rewriter, Location loc, OpFoldResult dimSize) {
  ImplicitLocOpBuilder::InsertionGuard g(rewriter);
  auto maybeStaticDimSize = getConstantIntValue(dimSize);
  // Case 1: Static dimension - compute tile size at compile time
  if (maybeStaticDimSize.has_value()) {
    assert(!ShapedType::isDynamic(maybeStaticDimSize.value()));
    if (maybeStaticDimSize.value() < kSubBlockDim) {
      return emitError(loc)
             << "dimension size (" << maybeStaticDimSize.value()
             << ") is less than minimum tile size (" << kSubBlockDim << ")";
    }
    int64_t tileSize =
        llvm::divideCeil(maybeStaticDimSize.value(), kSubBlockDim);
    return getAsIndexOpFoldResult(rewriter.getContext(), tileSize);
  }
  auto dynDimSize = dimSize.get<Value>();
  rewriter.setInsertionPointAfterValue(dynDimSize);
  // Case 2: Dynamic dimension - generate runtime computation
  // Create affine expression: ceil(dim0 / kSubBlockDim)
  AffineExpr dim0;
  bindDims(rewriter.getContext(), dim0);
  auto ceilDivMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                    dim0.ceilDiv(kSubBlockDim));
  auto tileSizeOp = rewriter.create<affine::AffineApplyOp>(
      loc, ceilDivMap, ValueRange{dynDimSize});
  return getAsOpFoldResult(tileSizeOp);
}

namespace {

static OpFoldResult calculateOffsetAtTilingDim(RewriterBase &rewriter,
                                               Location loc, Value inductionVar,
                                               OpFoldResult singleTileSize) {
  ImplicitLocOpBuilder::InsertionGuard g(rewriter);
  AffineExpr mulExpr =
      rewriter.getAffineSymbolExpr(0) * rewriter.getAffineSymbolExpr(1);
  OpFoldResult offsetAtTileDim = affine::makeComposedFoldedAffineApply(
      rewriter, loc, mulExpr, {inductionVar, singleTileSize});
  return offsetAtTileDim;
}

static FailureOr<linalg::SliceParameters>
calculateTiledParams(RewriterBase &rewriter, hivm::StoreOp storeOp,
                     int64_t tilingDim, scf::ForOp containingLoop) {
  Location loc = storeOp.getLoc();
  linalg::SliceParameters result;

  // Get the original shape info
  auto tilingOp = cast<TilingInterface>(storeOp.getOperation());
  auto iterationDomain = tilingOp.getIterationDomain(rewriter);
  auto [originalOffsets, originalSizes, originalStrides] =
      getOffsetsSizesAndStrides(iterationDomain);

  for (unsigned int i = 0; i < storeOp.getNumLoops(); i++) {
    // We only support unit stride for now
    result.strides.push_back(rewriter.getIndexAttr(1));
    if (i != tilingDim) {
      result.offsets.push_back(rewriter.getIndexAttr(0));
      result.sizes.push_back(originalSizes[i]);
      continue;
    }

    LDBG("original loop dim size: " << originalSizes[i]);
    auto unboundedTileSize = getSingleTileSize(rewriter, loc, originalSizes[i]);
    if (failed(unboundedTileSize))
      return rewriter.notifyMatchFailure(storeOp, "tile size is invalid");
    // Offset is dependent on tile size (the unbounded tile should be fine)
    auto offsetAtTileDim = calculateOffsetAtTilingDim(
        rewriter, loc, containingLoop.getInductionVar(),
        unboundedTileSize.value());

    rewriter.setInsertionPoint(storeOp);
    auto boundedTileSize = getBoundedTileSize(
        rewriter, loc, iterationDomain[tilingDim],
        containingLoop.getInductionVar(), unboundedTileSize.value());
    LDBG("bounded tile size is :" << boundedTileSize);

    result.offsets.push_back(offsetAtTileDim);
    result.sizes.push_back(boundedTileSize);
  }
  return result;
}

/// try to tile store ops and bind sub block mapping
class TileAndSliceStore : public OpRewritePattern<hivm::StoreOp> {
public:
  hivm::detail::DimensionAnalyzer &analyzer;

  explicit TileAndSliceStore(MLIRContext *context,
                             hivm::detail::DimensionAnalyzer &analyzer)
      : OpRewritePattern<hivm::StoreOp>(context, /*benefit=*/1),
        analyzer(analyzer) {}
  LogicalResult matchAndRewrite(hivm::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    if (storeOp->hasAttrOfType<UnitAttr>(kTiledOp))
      return rewriter.notifyMatchFailure(storeOp, "op is already tiled");

    auto maybeContainingLoop = findContainingSubblockLoop(storeOp);
    if (failed(maybeContainingLoop))
      return rewriter.notifyMatchFailure(storeOp, "failed to find parent loop");

    int64_t tilingDim = tilingDimMap_.at(storeOp.getSrc());
    if (tilingDim == -1)
      return rewriter.notifyMatchFailure(storeOp, "no parallel dim to tile");
    LDBG("subblock tiling dim is: " << tilingDim);

    auto maybeSliceParams = calculateTiledParams(rewriter, storeOp, tilingDim,
                                                 maybeContainingLoop.value());
    if (failed(maybeSliceParams))
      return rewriter.notifyMatchFailure(storeOp,
                                         "failed to get offset/size/stride");

    auto tilingOp = cast<TilingInterface>(storeOp.getOperation());
    FailureOr<TilingResult> tileResult = tilingOp.getTiledImplementation(
        rewriter, maybeSliceParams.value().offsets,
        maybeSliceParams.value().sizes);
    if (failed(tileResult))
      return rewriter.notifyMatchFailure(storeOp, "failed to tile");

    auto *tiledStoreOp = tileResult->tiledOps.front();
    tiledStoreOp->setAttr(kTiledOp, UnitAttr::get(storeOp->getContext()));
    rewriter.replaceOp(storeOp, tiledStoreOp);
    return success();
  }
};

/// add if (sublock_id == 0) guard for each store op.
/// e.g.
/// case 1: store op without results
///   store op
/// is changed to
///   if (subblock_id == 0)
///     store op
/// case 2: store op with results
///   %res = store op
/// is changed to
///   if (subblock_id == 0)
///     %res = store op
///     yield %res
///   else
///     yield store's outs
struct LimitUniqueSubBlockIdToStore : public OpRewritePattern<hivm::StoreOp> {
public:
  using OpRewritePattern<hivm::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::StoreOp op,
                                PatternRewriter &rewriter) const override {
    if (auto ifOpOld = dyn_cast_if_present<scf::IfOp>(op->getParentOp())) {
      if (ifOpOld->hasAttrOfType<UnitAttr>(kLimitedSubBlockOpAttrName))
        return failure();
    }
    auto loc = op.getLoc();
    auto subBlockIdxOp =
        rewriter.create<hivm::GetSubBlockIdxOp>(loc, rewriter.getI64Type());
    auto subBlockIndex =
        rewriter
            .create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                        subBlockIdxOp.getResult())
            .getResult();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto cond = rewriter.create<arith::CmpIOp>(loc, rewriter.getI1Type(),
                                               arith::CmpIPredicate::eq,
                                               subBlockIndex, zero);

    if (op.getResults().empty()) {
      // case 1: store op without results
      auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange(), cond, false);
      auto thenBodyBuilder = ifOp.getThenBodyBuilder(rewriter.getListener());
      thenBodyBuilder.clone(*op.getOperation());
      rewriter.replaceOp(op, ifOp);
      rewriter.modifyOpInPlace(ifOp, [&]() {
        ifOp->setAttr(kLimitedSubBlockOpAttrName,
                      UnitAttr::get(ifOp->getContext()));
      });
      return success();
    }

    // case 2: store op with results
    Type dstType = op.getDst().getType();
    auto ifOp = rewriter.create<scf::IfOp>(loc, dstType, cond, true);
    // then block
    {
      PatternRewriter::InsertionGuard insertionGuard(rewriter);
      auto thenBodyBuilder = ifOp.getThenBodyBuilder(rewriter.getListener());
      auto cloneStoreOp = thenBodyBuilder.clone(*op.getOperation());
      Value thenYield = cloneStoreOp->getResults()[0];
      ifOp.getThenBodyBuilder().create<scf::YieldOp>(loc, thenYield);
    }

    // else block
    {
      rewriter.setInsertionPointToEnd(&ifOp.getElseRegion().front());
      rewriter.create<scf::YieldOp>(loc, op.getDst());
    }
    rewriter.modifyOpInPlace(ifOp, [&]() {
      ifOp->setAttr(kLimitedSubBlockOpAttrName,
                    UnitAttr::get(ifOp->getContext()));
    });
    rewriter.replaceOp(op, ifOp);
    return success();
  }
};

} // namespace

static LogicalResult LimitUniqueSubBlockToStore(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<LimitUniqueSubBlockIdToStore>(funcOp.getContext());
  GreedyRewriteConfig config;
  config.maxIterations = kMaxIterations;
  return applyPatternsGreedily(funcOp, std::move(patterns));
}

static scf::ForOp createSubBlockLoop(Location loc, OpBuilder &builder,
                                     int64_t lowerBound, int64_t step,
                                     int64_t upperBound) {
  auto loopLowerBound =
      builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(lowerBound));
  auto loopStep =
      builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(step));
  auto loopUpperBound =
      builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(upperBound));
  auto subBlockLoop =
      builder.create<scf::ForOp>(loc, loopLowerBound, loopUpperBound, loopStep);
  subBlockLoop->setAttr(kMapForToForallAttrName,
                        UnitAttr::get(subBlockLoop->getContext()));

  SmallVector<Attribute> mappingNames;
  mappingNames.push_back(HIVMSubBlockMappingAttr::get(
      subBlockLoop->getContext(), hivm::MappingId::DimX));
  subBlockLoop->setAttr(
      kMappingAttrName,
      ArrayAttr::get(subBlockLoop->getContext(), mappingNames));
  return subBlockLoop;
}

static void failAndRevert(func::FuncOp func, StringRef errorMsg){
  LDBGS("tile and bind subblock fail for "
                    << func.getSymNameAttr() << "\n\n");
  func->emitError(errorMsg);
  LLVM_DEBUG(func->dump());
  func->erase();
}

static void populateBindSubBlockBubbleUpPassManager(PassManager &pm) {
  pm.addPass(createHIVMBubbleUpExtractSlicePass());
  CanonicalizerOptions options;
  SmallVector<std::string> disabledPatterns(
      {"ReinterpretCastConstantArgumentFolder"});
  options.disabledPatterns = disabledPatterns;
  pm.addPass(bishengir::createExtendedCanonicalizerPass(options));
  pm.addPass(createCSEPass());
}

static LogicalResult tileAndSliceStore(func::FuncOp func) {
  RewritePatternSet patterns(func->getContext());
  patterns.add<TileAndSliceStore>(func->getContext());
  GreedyRewriteConfig config;

  auto listener = hivm::detail::BubbleUpListener();
  config.listener = &listener;
  bool tiled = false;
  if (failed(applyPatternsAndFoldGreedily(newFunc, std::move(patterns), config,
                                          &tiled))) {
    failAndRevert(newFunc, "Failed to apply tile and slice store pattern");
    return failure();
  }

  if (!tiled) {
    failAndRevert(newFunc, "No effect after tile and slice store");
     return failure();
   }return success();
}

/// Attempts to tile and bind sub-blocks within a function
///
/// This function performs a series of transformations on vector functions:
/// 1. Creates a BindSubBlock Loop that includes whole function body
/// i.e.
/// func {
///   for {
///     func_body
///   } {sub_block_loop}
/// }
/// 2. Insert a extract slice before all storeOps
/// And then we rely on run bubbleUpExtractSlice to tile all ops
///
/// @param func The function to transform (should be a clone if rollback is
/// needed)
/// @return Success if transformation completed, failure otherwise
FailureOr<func::FuncOp>
TileAndBindSubBlockPass::attemptBindSubBlock(func::FuncOp func) {
  // This only apply for aiv func. Should be check before calling.
  OpBuilder builder(func->getContext());
  builder.setInsertionPoint(func);
  // We cloned newFunc for processing.
  func::FuncOp newFunc = cast<func::FuncOp>(builder.cloneWithoutRegions(func));
  newFunc.setName(func.name().str() + "_Processing");
  newFunc.addEntryBlock();
  builder.setInsertionPointToStart(&newFunc.getBody().getBlocks().front());

  auto subBlockLoop =
      createSubBlockLoop(func->getLoc(), builder, 0, 1, kSubBlockDim);

  IRMapping map;
  for (size_t i = 0; i < func.getNumArguments(); i++) {
    map.map(func.getArgument(i), newFunc.getArgument(i));
  }

  builder.setInsertionPointToStart(subBlockLoop.getBody(0));
  // We are trying to wrap subblock loop to the whole function body.
  // so we clone the whole function body inside the loop.
  func.getBody().cloneInto(&subBlockLoop.getBodyRegion(0), map);

  // bb0 is the loop body when the loop is created (empty with a terminator)
  // bb1 is the cloned function body
  auto &bb0 = subBlockLoop.getBodyRegion(0).getBlocks().front();
  auto *bb1 = bb0.getNextNode();
  if (!bb1)
    llvm::report_fatal_error("Failed to find function body");

  Operation *terminator = bb0.getTerminator();
  // We need to merge bb0 and bb1 because a loop body can only have 1 blocks
  if (bb1->mightHaveTerminator()) {
    builder.setInsertionPointToEnd(&newFunc.getBody().getBlocks().front());
    builder.clone(*bb1->getTerminator(), map);
    bb1->getOperations().pop_back();
  }
  bb0.getOperations().splice(terminator->getIterator(), bb1->getOperations());
  // We need to handle the terminators. clone function body's (bb1) terminator
  // outside of subblock loop body and use as cloned newFunc's terminator.
  bb1->erase();

  LDBG("Moudle:" << *getOperation());
  if (failed(tileAndSliceStore(newFunc))) {
    failAndRevert(newFunc, "Failed to analyze dimensions");
    return failure();
  }

  PassManager pm(newFunc->getContext());
  populateBindSubBlockBubbleUpPassManager(pm);

  LogicalResult bubbleUpResult = pm.run(newFunc);
  if (bubbleUpResult.failed() || newFunc.verify().failed() ||
      newFunc.verifyBody().failed() || newFunc.verifyRegions().failed()) {
    failAndRevert(newFunc, "Failed to bubble up");
    return failure();
  }

  RewritePatternSet patternsPost(&getContext());
  patternsPost.add<mlir::hivm::detail::BubbleUpSubviewFromTiling>(
      &getContext());
  if (failed(applyPatternsGreedily(newFunc, std::move(patternsPost)))) {
    failAndRevert(newFunc, "Failed to applyPatternGreedily");
    return failure();
  }

  return newFunc;
}

/// Walks through all functions in the module and attempts to tile and bind
/// sub-blocks for vector functions.
///
/// Functions are cloned before transformation to allow rollback on failure.
/// If attempt to bind some block fail it will rollback to 1:1 and limit to
/// unique block to store.
void TileAndBindSubBlockPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
#ifndef NDEBUG
  uint64_t tiledFunctionCount = 0;
#endif

  // Collect functions to process (can't modify while iterating)
  SmallVector<func::FuncOp> functionsToProcess;
  moduleOp->walk(
      [&](func::FuncOp funcOp) { functionsToProcess.push_back(funcOp); });

  // Process each function
  for (func::FuncOp originalFunc : functionsToProcess) {
    // Only process vector functions
    auto symNameStr = originalFunc.getSymNameAttr().str();
    auto funcCoreType = queryFuncCoreType(originalFunc);
    if (!funcCoreType.has_value() ||
        funcCoreType.value() != TFuncCoreType::AIV ||
        !originalFunc->hasAttrOfType<UnitAttr>(hivm::TPartOfMixAttr::name))
      continue;
    // Clone the function for safe transformation
    OpBuilder builder(originalFunc);
    // Attempt transformation on the clone
    FailureOr<func::FuncOp> res = attemptBindSubBlock(originalFunc);
    if (failed(res)) {
      if (failed(LimitUniqueSubBlockToStore(originalFunc))) {
        LDBG("Failed to limit unique subblock: " << symNameStr
                          << "\n");
        signalPassFailure();
      }
      LDBG("Failed to transform function: " << symNameStr
                  << ", keeping original\n");
      return;
    }
    auto processedFunc = res.value();
    processedFunc.setName(originalFunc.getName().str() + "_processing");
    if (succeeded(res)) {
      // Success: Remove original and rename clone
      originalFunc.erase();
      processedFunc.setName(symNameStr);
#ifndef NDEBUG
      tiledFunctionCount++;
      LDBG("Successfully transformed function #"
                        << tiledFunctionCount << ": " << symNameStr << "\n");
#endif
    }
  }

#ifndef NDEBUG
  LDBG("TileAndBindSubBlock pass completed. "
                    << "Successfully transformed " << tiledFunctionCount
                    << " functions.\n");
#endif
}

std::unique_ptr<Pass> mlir::hivm::createTileAndBindSubBlockPass() {
  return std::make_unique<TileAndBindSubBlockPass>();
}
