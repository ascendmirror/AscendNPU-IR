//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
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

#include "bishengir/Dialect/Annotation/Transforms/BufferizableOpInterfaceImpl.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/UnstructuredControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::annotation;

namespace mlir {
namespace annotation {
namespace {
/// Bufferization of annotation.mark.
struct MarkOpInterface
    : public BufferizableOpInterface::ExternalModel<MarkOpInterface,
                                                    annotation::MarkOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    // Mark operands always bufferize inplace. Otherwise, an alloc + copy
    // may be generated inside the block.
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto markOp = cast<annotation::MarkOp>(op);

    Value newSrc;
    const auto &src = markOp.getSrc();
    Value value = src;
    if (isa<TensorType>(value.getType())) {
      FailureOr<Value> maybeBuffer = getBuffer(rewriter, value, options);
      if (failed(maybeBuffer))
        return failure();
      Value buffer = *maybeBuffer;
      newSrc = buffer;
    } else {
      newSrc = value;
    }

    DictionaryAttr newAttrs = op->getAttrDictionary();
    auto newOp =
        replaceOpWithNewBufferizedOp<annotation::MarkOp>(rewriter, op, newSrc);
    // Forward the old attributes to the new operation
    newOp->setAttrs(newAttrs);
    return success();
  }
};
} // namespace
} // namespace annotation
} // namespace mlir

void mlir::annotation::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, annotation::AnnotationDialect *dialect) {
        MarkOp::attachInterface<MarkOpInterface>(*ctx);
      });
}
