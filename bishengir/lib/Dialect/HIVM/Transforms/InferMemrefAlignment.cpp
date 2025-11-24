//===- InferMemrefAlignment.cpp -- Infer and Erase Memref Alignment Passes ===//
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
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include <numeric>
#include <optional>
#include <type_traits>

namespace mlir {
#define GEN_PASS_DEF_INFERMEMREFALIGNMENT
#define GEN_PASS_DEF_ERASEMEMREFALIGNMENTMARKS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using U = unsigned;

struct AlignmentValue {
  std::optional<U> align = std::nullopt;

  AlignmentValue() = default;
  AlignmentValue(U a) : align(a) {}

  static constexpr U maxAlign = 32;

  static bool isPowerOfTwo(U n) { return ((n & (n - 1)) == 0); }

  static AlignmentValue getUninitialized() { return AlignmentValue(); }

  static AlignmentValue getUnknown() { return AlignmentValue(1); }

  static AlignmentValue getFromExact(U a) {
    if (a == 0) {
      return getUnknown();
    }
    static_assert(std::is_unsigned<U>());

    auto isPow2 = [](U n) { return (n & (n - 1)) == 0; };
    assert(isPow2(a));

    return AlignmentValue(std::min(a, maxAlign));
  }

  bool isUnitialized() const { return !align.has_value(); }

  U getAlignment() const {
    assert(!isUnitialized());
    return *align;
  }

  static AlignmentValue allocAlignment() { return 32; }

  static AlignmentValue join(const AlignmentValue &l, const AlignmentValue &r) {
    if (l.isUnitialized()) {
      return r;
    }
    if (r.isUnitialized()) {
      return l;
    }

    auto a = l.getAlignment();
    auto b = r.getAlignment();
    auto g = std::gcd(a, b);
    return getFromExact(g);
  }

  static AlignmentValue scaled(const AlignmentValue &l,
                               const AlignmentValue &r) {
    if (l.isUnitialized() || r.isUnitialized()) {
      return getUnknown();
    }
    auto a = l.getAlignment();
    auto b = r.getAlignment();
    auto c = a * b;
    return getFromExact(c);
  }

  bool operator==(const AlignmentValue &av) const { return align == av.align; }

  void print(llvm::raw_ostream &os) const {
    if (isUnitialized()) {
      os << "⊥";
    } else {
      os << "align=" << *align;
    }
  }
};

using AV = AlignmentValue;
using Latt = dataflow::Lattice<AV>;
using AABase = dataflow::SparseForwardDataFlowAnalysis<Latt>;

namespace std {
static AV max(const AV& l, const AV& r) {
  return AV::getFromExact(std::max(l.getAlignment(), r.getAlignment()));
}
} // namespace std

struct AlignmentAnalysis : public AABase {

  AlignmentAnalysis(DataFlowSolver &solver) : AABase(solver) {}

  void setToEntryState(Latt *lattice) override {
    this->propagateIfChanged(lattice, lattice->join(AV::getUnknown()));
  }

  LogicalResult visitOperation(Operation *op, ArrayRef<const Latt *> operands,
                               ArrayRef<Latt *> results) override {
    if (results.empty()) {
      return success();
    }

    auto getOperand = [&](U idx) -> std::optional<AV> {
      if (idx >= operands.size() || !operands[idx]) {
        return std::nullopt;
      }
      auto &av = operands[idx]->getValue();
      if (av.isUnitialized()) {
        return std::nullopt;
      }
      return av;
    };

    auto getValueAV = [&](Value v) -> std::optional<AV> {
      auto opOperands = op->getOperands();
      for (auto i = 0; i < opOperands.size(); i++) {
        if (opOperands[i] != v) {
          continue;
        }
        if (!operands[i]) {
          return std::nullopt;
        }
        auto &av = operands[i]->getValue();
        if (av.isUnitialized()) {
          return std::nullopt;
        }
        return av;
      }
      return std::nullopt;
    };

    auto updateAllRes = [&](const AV &av) {
      for (auto *res : results) {
        this->propagateIfChanged(res, res->join(av));
      }
    };

    auto updateSingleRes = [&](const AV &av) {
      this->propagateIfChanged(results.front(), results.front()->join(av));
    };

    auto setAllUnknown = [&]() { updateAllRes(AV::getUnknown()); };

    auto pow2Divisor = [&](U v) -> U {
      if (v == 0) {
        return AV::maxAlign;
      }

      U p = 1;
      while ((v % (p << 1)) == 0 && (p << 1) <= AV::maxAlign) {
        p <<= 1;
      }
      assert(AV::isPowerOfTwo(p));
      assert(v % p == 0);

      return p;
    };

    auto constTemplate = [&](const std::function<int64_t()> &vGetter) {
      auto v = vGetter();
      v = v < 0 ? -v : v;
      auto align = pow2Divisor(v);
      updateSingleRes(AV::getFromExact(align));

      return success();
    };

    if (auto cInt = dyn_cast<arith::ConstantIntOp>(op)) {
      return constTemplate([&]() { return cInt.value(); });
    }

    if (auto cIdx = dyn_cast<arith::ConstantIndexOp>(op)) {
      return constTemplate([&]() { return cIdx.value(); });
    }

    if (auto assume = dyn_cast<memref::AssumeAlignmentOp>(op)) {
      auto base = getOperand(0).value_or(AV::getUnknown());
      auto assumed = AV::getFromExact(assume.getAlignment());
      updateSingleRes(std::max(base, assumed));
      return success();
    }

    if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
      updateSingleRes(AV::allocAlignment());
      return success();
    }

    if (isa<memref::CastOp, memref::ReshapeOp, memref::CollapseShapeOp,
            memref::ExpandShapeOp, memref::TransposeOp,
            memref::MemorySpaceCastOp>(op)) {
      if (auto base = getOperand(0)) {
        for (auto *res : results) {
          this->propagateIfChanged(res, res->join(*base));
        }
      } else {
        setAllUnknown();
      }
      return success();
    }

    if (auto view = dyn_cast<memref::ViewOp>(op)) {
      auto src = view.getSource();
      auto shift = view.getByteShift();

      auto baseAV = getValueAV(src);
      auto shiftAV = getValueAV(shift);

      auto newAlign = AV::join(baseAV.value_or(AV()), shiftAV.value_or(AV()));
      updateSingleRes(newAlign);
      return success();
    }

    if (auto subView = dyn_cast<memref::SubViewOp>(op)) {
      auto src = subView.getSource();
      auto srcType = dyn_cast<mlir::MemRefType>(src.getType());
      if (!srcType) {
        setAllUnknown();
        return success();
      }

      SmallVector<int64_t, 4> strides;
      int64_t offset;
      if (failed(getStridesAndOffset(srcType, strides, offset))) {
        setAllUnknown();
        return success();
      }

      auto elemBytes = srcType.getElementTypeBitWidth() / 8;
      assert(srcType.getElementTypeBitWidth() % 8 == 0);

      auto mixedOffsets = subView.getMixedOffsets();
      if (mixedOffsets.size() != strides.size()) {
        setAllUnknown();
        return success();
      }

      U staticBytes = 0;
      U dynAlign = 0;
      auto hasDyn = false;

      auto uabs = [](auto x) { return static_cast<U>(std::abs(x)); };

      for (auto it : llvm::enumerate(mixedOffsets)) {
        auto strideElems = strides[it.index()];
        if (ShapedType::isDynamic(strideElems)) {
          setAllUnknown();
          return success();
        }

        auto strideBytes = uabs(strideElems) * elemBytes;

        auto ofr = it.value();
        if (auto attr = dyn_cast<Attribute>(ofr)) {
          auto off = cast<IntegerAttr>(attr).getInt();
          staticBytes += uabs(off) * strideBytes;
        } else {
          hasDyn = true;
          auto offAV = getValueAV(ofr.get<Value>()).value_or(AV());
          auto stridePow2 = pow2Divisor(strideBytes);
          auto termAlign = AV::getFromExact(offAV.getAlignment() * stridePow2)
                               .getAlignment();

          dynAlign =
              (dynAlign == 0) ? termAlign : std::gcd(dynAlign, termAlign);
        }
      }

      U deltaAlign = 0;
      if (staticBytes != 0) {
        deltaAlign = pow2Divisor(staticBytes);
      }
      if (hasDyn) {
        deltaAlign =
            (deltaAlign == 0) ? dynAlign : std::gcd(deltaAlign, dynAlign);
      }

      auto baseAlign = getValueAV(src).value_or(AV()).getAlignment();

      if (!hasDyn && staticBytes == 0) {
        updateSingleRes(AV::getFromExact(baseAlign));
        return success();
      }

      if (deltaAlign == 0) {
        setAllUnknown();
        return success();
      }

      auto newAlign = std::gcd(baseAlign, deltaAlign);
      updateSingleRes(AV::getFromExact(newAlign ? newAlign : 1));
      return success();
    }

    if (auto rc = dyn_cast<memref::ReinterpretCastOp>(op)) {
      auto src = rc.getSource();
      auto resType = dyn_cast<mlir::MemRefType>(rc.getResult().getType());
      if (!resType) {
        setAllUnknown();
        return success();
      }

      auto elemBytes = resType.getElementTypeBitWidth() / 8;
      assert(resType.getElementTypeBitWidth() % 8 == 0);

      auto mixedOffsets = rc.getMixedOffsets();
      if (mixedOffsets.size() != 1) {
        setAllUnknown();
        return success();
      }

      auto ofr = mixedOffsets.front();
      U staticBytes = 0;
      U dynAlign = 0;
      auto hasDyn = false;

      auto uabs = [](auto x) { return static_cast<U>(std::abs(x)); };

      if (auto attr = dyn_cast<Attribute>(ofr)) {
        auto off = cast<IntegerAttr>(attr).getInt();
        staticBytes = uabs(off) * elemBytes;
      } else {
        hasDyn = true;
        auto offAV = getValueAV(ofr.get<Value>()).value_or(AV());
        auto elemPow2 = pow2Divisor(elemBytes);
        dynAlign =
            AV::getFromExact(offAV.getAlignment() * elemPow2).getAlignment();
      }

      U deltaAlign = 0;
      if (staticBytes != 0) {
        deltaAlign = pow2Divisor(staticBytes);
      }
      if (hasDyn) {
        deltaAlign =
            (deltaAlign == 0) ? dynAlign : std::gcd(deltaAlign, dynAlign);
      }

      auto baseAlign = getValueAV(src).value_or(AV()).getAlignment();

      if (!hasDyn && staticBytes == 0) {
        updateSingleRes(AV::getFromExact(baseAlign));
        return success();
      }

      if (deltaAlign == 0) {
        setAllUnknown();
        return success();
      }

      auto newAlign = std::gcd(baseAlign, deltaAlign);
      updateSingleRes(AV::getFromExact(newAlign ? newAlign : 1));
      return success();
    }

    auto simple2OpTemplate =
        [&](const std::function<AV(const AV &, const AV &)> &f) {
          auto l = getOperand(0);
          auto r = getOperand(1);
          if (!l || !r) {
            return success();
          }
          updateSingleRes(f(*l, *r));
          return success();
        };

    if (isa<arith::AddIOp, arith::SubIOp>(op)) {
      return simple2OpTemplate(AV::join);
    }

    if (isa<arith::MulIOp>(op)) {
      return simple2OpTemplate(AV::scaled);
    }

    if (isa<arith::SelectOp>(op)) {
      return simple2OpTemplate(AV::join);
    }

    if (auto shl = dyn_cast<arith::ShLIOp>(op)) {
      auto base = getOperand(0);
      U factor = 1;
      if (auto cInt = shl.getRhs().getDefiningOp<arith::ConstantIntOp>()) {
        auto sh = cInt.value();
        if (sh > 0 && sh < 32) {
          factor = 1u << sh;
        }
      } else if (auto cIdx =
                     shl.getRhs().getDefiningOp<arith::ConstantIndexOp>()) {
        auto sh = cIdx.value();
        if (sh > 0 && sh < 32) {
          factor = 1u << sh;
        }
      }
      updateSingleRes(AV::getFromExact(base->getAlignment() * factor));
      return success();
    }

    this->setAllToEntryStates(results);
    return success();
  }
};

struct InferMemrefAlignmentPass
    : public impl::InferMemrefAlignmentBase<InferMemrefAlignmentPass> {

  void runOnOperation() override {
    auto func = getOperation();

    OpBuilder builder(func.getContext());
    auto &entryBlock = func.front();
    builder.setInsertionPointToStart(&entryBlock);

    for (auto arg : entryBlock.getArguments()) {
      if (!dyn_cast<MemRefType>(arg.getType())) {
        continue;
      }

      auto alignAttr = builder.getI32IntegerAttr(AV::maxAlign);
      auto assume = builder.create<memref::AssumeAlignmentOp>(arg.getLoc(), arg,
                                                              alignAttr);
      for (auto result : assume->getResults()) {
        arg.replaceUsesWithIf(result, [&](OpOperand &use) {
          return use.getOwner() != assume.getOperation();
        });
      }
    }

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<AlignmentAnalysis>();

    if (failed(solver.initializeAndRun(func))) {
      func.emitError() << "AlignmentAnalysis dataflow failed";
      signalPassFailure();
      return;
    }

    SmallVector<std::pair<Value, U>, 16> toAnnotate;

    auto collectValue = [&](Value v) {
      if (!dyn_cast<MemRefType>(v.getType())) {
        return;
      }

      if (auto *def = v.getDefiningOp()) {
        if (isa<memref::AssumeAlignmentOp>(def)) {
          return;
        }
      }

      auto *state = solver.lookupState<Latt>(v);
      if (!state) {
        return;
      }

      auto &val = state->getValue();
      if (val.isUnitialized()) {
        return;
      }
      auto align = val.getAlignment();
      if (align <= 1) {
        return;
      }
      toAnnotate.emplace_back(v, align);
    };

    for (auto arg : func.getArguments()) {
      collectValue(arg);
    }

    func.walk([&](Operation *op) {
      for (auto res : op->getResults()) {
        collectValue(res);
      }
    });

    for (auto &[v, align] : toAnnotate) {
      if (!dyn_cast<MemRefType>(v.getType())) {
        return;
      }
      auto loc = v.getLoc();
      auto *defOp = v.getDefiningOp();

      if (defOp) {
        builder.setInsertionPointAfter(defOp);
      } else {
        auto *block = cast<BlockArgument>(v).getOwner();
        builder.setInsertionPointToStart(block);
        loc = func.getLoc();
      }

      auto alignAttr = builder.getI32IntegerAttr(align);
      builder.create<memref::AssumeAlignmentOp>(loc, v, alignAttr);
    }
  }
};

struct EraseMemrefAlignmentMarksPass
    : public impl::EraseMemrefAlignmentMarksBase<
          EraseMemrefAlignmentMarksPass> {

  void runOnOperation() override {
    auto func = getOperation();

    SmallVector<memref::AssumeAlignmentOp, 16> toErase;
    func.walk([&](memref::AssumeAlignmentOp op) {
      auto src = op.getMemref();
      for (auto res : op->getResults()) {
        res.replaceAllUsesWith(src);
      }
      toErase.push_back(op);
    });

    for (auto op : toErase) {
      op.erase();
    }
  }
};

std::unique_ptr<mlir::Pass> mlir::hivm::createEraseMemrefAlignmentMarksPass() {
  return std::make_unique<EraseMemrefAlignmentMarksPass>();
}

std::unique_ptr<mlir::Pass>
mlir::hivm::createInferMemrefAlignmentPass() {
  return std::make_unique<InferMemrefAlignmentPass>();
}