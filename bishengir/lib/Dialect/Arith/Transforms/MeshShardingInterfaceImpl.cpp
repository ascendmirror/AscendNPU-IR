//===- MeshShardingInterfaceImpl.cpp - Impl. of Sharding Interface for arith==//
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

#include "bishengir/Dialect/Arith/Transforms/MeshShardingInterfaceImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/MeshShardingInterfaceImpl.h"
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
namespace {
/// Only implement the constant op for now
struct ArithConstantOpShardingInterface
    : public mesh::ShardingInterface::ExternalModel<
          ArithConstantOpShardingInterface, arith::ConstantOp> {
  /// Helper for determining the rank of result tensor
  LogicalResult getResultRank(Operation *op, int64_t &result) const {
    if (op->getNumResults() != 1)
      return op->emitOpError(
          "Expecting single result for elemwise op for sharding");

    auto tensorTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!tensorTy)
      return failure();

    result = tensorTy.getRank();
    return success();
  }

  /// Required method for ShardingInterface, elemwise operators are always
  /// parallel
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    int64_t rank = 0;
    if (getResultRank(op, rank).failed())
      return {};
    return SmallVector<utils::IteratorType>(rank,
                                            utils::IteratorType::parallel);
  };

  /// Required method for ShardingInterface, one AffineMap for every operand and
  /// result. For elemwise ops, the map will always be identity.
  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    int64_t rank = 0;
    if (getResultRank(op, rank).failed())
      return {};

    // Making an assumption that there is only one result with no operands
    SmallVector<AffineMap> retVal;
    auto resultMap = AffineMap::getMultiDimIdentityMap(rank, op->getContext());
    retVal.push_back(resultMap);
    return retVal;
  }

  /// Required method for mesh-spmdization. Update output type as well as the
  /// input attribute type.
  LogicalResult spmdize(Operation *op, ArrayRef<Value> spmdizedOperands,
                        ArrayRef<mesh::MeshShardingAttr> operandShardings,
                        ArrayRef<mesh::MeshShardingAttr> resultShardings,
                        IRMapping &spmdizationMap,
                        SymbolTableCollection &symbolTable,
                        OpBuilder &builder) const {
    if (resultShardings.empty())
      return success();

    auto newOp = cast<arith::ConstantOp>(builder.clone(*op, spmdizationMap));

    // Get result type from sharding
    Value result = newOp->getResult(0);
    // If output is not a tensor, simply cloning is enough.
    if (!isa<TensorType>(result.getType()))
      return success();

    mesh::MeshShardingAttr sharding = resultShardings[0];
    auto newTy = cast<ShapedType>(shardType(
        result.getType(), mesh::getMesh(newOp, sharding.getMesh(), symbolTable),
        sharding));

    // ConstantOp shouldn't have any operands, no operand shardings. Shard
    // dense attribute instead
    auto denseAttr = dyn_cast<DenseElementsAttr>(newOp.getValueAttr());
    if (!denseAttr || !denseAttr.isSplat())
      return op->emitError("Can only spmd-ize splat constant tensors");

    DenseElementsAttr newAttr;
    Type elTy = denseAttr.getElementType();
    if (elTy.isInteger())
      newAttr = DenseElementsAttr::get(newTy, denseAttr.getSplatValue<APInt>());
    else
      newAttr =
          DenseElementsAttr::get(newTy, denseAttr.getSplatValue<APFloat>());

    result.setType(newTy);
    newOp.setValueAttr(newAttr);

    return success();
  }
};
} // namespace

void arith::registerShardingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    arith::ConstantOp::attachInterface<ArithConstantOpShardingInterface>(*ctx);
  });
}
