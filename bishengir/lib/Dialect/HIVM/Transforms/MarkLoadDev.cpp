//===------------------ MarkLoadDev.cpp -------------------===//
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
// This pass marks the memref.loads that need to be converted to load_devs.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include <optional>
#include <vector>

namespace mlir {
#define GEN_PASS_DEF_MARKLOADDEV
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-mark-load-dev"

namespace {
struct MarkLoadDev
    : public impl::MarkLoadDevBase<MarkLoadDev> {
  using Base::Base;
  void runOnOperation() override;
};

std::optional<Value> traceToFuncArg(Value v, func::FuncOp f) {
  for (auto val : f.getBody().getArguments()) {
    if (val == v) {
      return val;
    }
  }
  if (auto viewLikeOp = v.getDefiningOp<ViewLikeOpInterface>()) {
    return traceToFuncArg(viewLikeOp.getViewSource(), f);
  }
  return std::nullopt;
}

std::vector<Operation*> traceToEndUsers(Value v) {
  std::vector<Operation*> endUsers;
  for (Operation *userOp : v.getUsers()) {
    if (isa<ViewLikeOpInterface>(userOp)) {
      auto nextLayerEndUsers = traceToEndUsers(userOp->getOpResult(0));
      for (Operation *nextLayerEndUser :nextLayerEndUsers) {
        endUsers.push_back(nextLayerEndUser);
      }
    } else {
      endUsers.push_back(userOp);
    }
  }
  return endUsers;
}

struct SearchAndMark : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;
  constexpr static llvm::StringRef finishedMarkingLoadDev = "finishedMarkingLoadDev";
  constexpr static llvm::StringRef needLoadDev = "needLoadDev";
  LogicalResult matchAndRewrite(memref::LoadOp memrefLoadOp,
                                PatternRewriter &rewriter) const override {
    // first check if the op has already been marked
    if (memrefLoadOp.getOperation()->hasAttr(finishedMarkingLoadDev)) {
      return failure();
    }
    auto f = memrefLoadOp->getParentOfType<func::FuncOp>();
    if (!f) {
      return failure();
    }
    auto arg = traceToFuncArg(memrefLoadOp.getMemref(), f);
    if (!arg.has_value()) {
      return failure();
    }
    auto endUsers = traceToEndUsers(arg.value());
    if (endUsers.size() > 1) {
      // if the load source is also used by others
      // then to avoid cache consistency problems it needs to
      // be implemented using ld_dev
      // Note: the value 0 is only used as a placeholder for the attribute
      memrefLoadOp.getOperation()->setAttr(needLoadDev, rewriter.getI32IntegerAttr(0));
    }
    memrefLoadOp.getOperation()->setAttr(finishedMarkingLoadDev,
                                    rewriter.getI32IntegerAttr(0));
    return success();
  }
};

void MarkLoadDev::runOnOperation() {
  func::FuncOp funcOp = cast<func::FuncOp>(getOperation());
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.add<SearchAndMark>(patterns.getContext());
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createMarkLoadDevPass() {
  return std::make_unique<MarkLoadDev>();
}
