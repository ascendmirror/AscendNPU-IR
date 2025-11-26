//===- RewriteUnalignedDMA.cpp -- Rewrite unaligned HIVM DMA invocations --===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
#define GEN_PASS_DEF_REWRITEUNALIGNEDDMA
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

static constexpr int dmaRequiredAlign = 32;

static bool isAlignedMemref(Value memref) {
  assert(dyn_cast<MemRefType>(memref.getType()));

  for (auto user : memref.getUsers()) {
    if (auto assume = dyn_cast<memref::AssumeAlignmentOp>(user)) {
      return assume.getAlignment() >= dmaRequiredAlign;
    }
  }

  return false;
}

static void emitElementwiseCopy(Value src, Value dst, OpBuilder& builder, Location loc) {
  auto srcType = cast<MemRefType>(src.getType());
  auto dstType = cast<MemRefType>(dst.getType());
  assert(srcType.getElementType() == dstType.getElementType());
  assert(srcType.getRank() == dstType.getRank());
  auto rank = srcType.getRank();

  if (rank == 0) {
    auto v = builder.create<memref::LoadOp>(loc, src);
    builder.create<memref::StoreOp>(loc, v, dst);
    return;
  }

  SmallVector<Value, 4> lbs(rank), ubs(rank), steps(rank);
  for (auto i = 0; i < rank; i++) {
    lbs[i] = builder.create<arith::ConstantIndexOp>(loc, 0);
    ubs[i] = builder.create<memref::DimOp>(loc, src, i);
    steps[i] = builder.create<arith::ConstantIndexOp>(loc, 1);
  }

  std::function<void(int, std::vector<Value>&, OpBuilder&)> buildLoop =
    [&](int dim, std::vector<Value>& ivs, OpBuilder& b) -> void {
      if (dim == rank) {
        auto val = b.create<memref::LoadOp>(loc, src, ivs);
        b.create<memref::StoreOp>(loc, val, dst, ivs);
        return;
      }

      auto forOp = b.create<scf::ForOp>(loc, lbs[dim], ubs[dim], steps[dim]);
      OpBuilder innerBuilder(forOp.getBodyRegion());
      auto newIvs = ivs;
      newIvs.push_back(forOp.getInductionVar());
      buildLoop(dim + 1, newIvs, innerBuilder);
    };

  // todo use smallvector
  std::vector<Value> ivs;
  buildLoop(0, ivs, builder);
}

static void rewriteCopy(CopyOp op) {
  auto src = op.getSrc();
  auto dst = op.getDst();

  auto srcTy = cast<MemRefType>(src.getType());
  auto dstTy = cast<MemRefType>(dst.getType());

  assert(srcTy.getElementType() == dstTy.getElementType());

  auto srcAligned = isAlignedMemref(src);
  auto dstAligned = isAlignedMemref(dst);

  if (srcAligned && dstAligned) {
    return;
  }

  OpBuilder builder(op);
  auto loc = op.getLoc();

  emitElementwiseCopy(src, dst, builder, loc);

  op->erase();
}

static Value createAlignedTempBuffer(Value srcMemref, OpBuilder& builder, Location loc) {
  auto srcType = cast<MemRefType>(srcMemref.getType());
  auto tmpType = MemRefType::get(
      srcType.getShape(),
      srcType.getElementType(),
      srcType.getLayout(),
      srcType.getMemorySpace()
  );

  SmallVector<Value, 4> dynSizes;
  dynSizes.reserve(tmpType.getNumDynamicDims());

  for (int dim = 0, e = tmpType.getRank(); dim < e; dim++) {
    if (tmpType.isDynamicDim(dim)) {
      dynSizes.push_back(builder.create<memref::DimOp>(loc, srcMemref, dim));
    }
  }

  auto alignAttr = builder.getI64IntegerAttr(dmaRequiredAlign);
  auto alloc = builder.create<memref::AllocOp>(loc, tmpType, dynSizes, alignAttr);

  return alloc;
}

static void rewriteStore(StoreOp op) {
  auto src = op.getSrc();
  auto dst = op.getDst();

  auto srcTy = cast<MemRefType>(src.getType());
  auto dstTy = cast<MemRefType>(dst.getType());

  assert(srcTy.getElementType() == dstTy.getElementType());

  if (isAlignedMemref(src)) {
    return;
  }

  OpBuilder builder(op);
  auto loc = op.getLoc();

  auto tmpSrc = createAlignedTempBuffer(src, builder, loc);
  emitElementwiseCopy(src, tmpSrc, builder, loc);

  builder.create<StoreOp>(loc, op.getResultTypes(), tmpSrc, dst);
  op->erase();
}

static void rewriteLoad(LoadOp op) {
  auto src = op.getSrc();
  auto dst = op.getDst();

  auto srcTy = cast<MemRefType>(src.getType());
  auto dstTy = cast<MemRefType>(dst.getType());

  assert(srcTy.getElementType() == dstTy.getElementType());

  if (isAlignedMemref(dst)) {
    return;
  }

  OpBuilder builder(op);
  auto loc = op.getLoc();

  auto tmpDst = createAlignedTempBuffer(dst, builder, loc);
  auto newOp = builder.create<LoadOp>(loc, op.getResultTypes(), src, tmpDst);

  OpBuilder afterBuilder(newOp);
  afterBuilder.setInsertionPointAfter(newOp);
  emitElementwiseCopy(tmpDst, dst, afterBuilder, loc);

  op->erase();
}

struct RewriteUnalignedDMAPass
    : public impl::RewriteUnalignedDMABase<RewriteUnalignedDMAPass> {
  void runOnOperation() override {
    auto func = getOperation();

    func.walk([&](hivm::CopyOp op) {
      rewriteCopy(op);
    });

    func.walk([&](hivm::StoreOp op) {
      rewriteStore(op);
    });

    func.walk([&](hivm::LoadOp op) {
      rewriteLoad(op);
    });
  }
};

std::unique_ptr<mlir::Pass>
mlir::hivm::createRewriteUnalignedDMAPass() {
  return std::make_unique<RewriteUnalignedDMAPass>();
}