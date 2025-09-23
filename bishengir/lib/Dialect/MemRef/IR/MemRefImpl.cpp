//===- MemRefImpl.cpp -----------------------------------------------------===//
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

#include "bishengir/Dialect/MemRef/IR/MemRefImpl.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace memref {

Value createMemRefAllocOpWithTargetElemType(
    OpBuilder &builder, Location loc, Value source, Type targetElemType,
    std::optional<MemRefLayoutAttrInterface> layout) {
  auto shapedType = cast<ShapedType>(source.getType());
  ArrayRef<int64_t> staticShapes = shapedType.getShape();
  llvm::SmallVector<Value, 2> dynamicSizes;
  for (size_t i = 0; i < staticShapes.size(); i++) {
    if (staticShapes[i] == ShapedType::kDynamic) {
      Operation *dynDimOp = builder.create<memref::DimOp>(loc, source, i);
      dynamicSizes.push_back(dynDimOp->getResults()[0]);
    }
  }
  MemRefType origType = cast<MemRefType>(shapedType);
  MemRefType newMemTy = MemRefType::get(
      staticShapes, targetElemType,
      layout.has_value() ? layout.value() : origType.getLayout(),
      origType.getMemorySpace());
  return builder.create<memref::AllocOp>(loc, newMemTy, dynamicSizes);
}

Value createMemRefAllocOp(OpBuilder &builder, Location loc, Value source) {
  auto elementType = mlir::getElementTypeOrSelf(source);
  auto emptyOp =
      createMemRefAllocOpWithTargetElemType(builder, loc, source, elementType);
  return emptyOp;
}

} // namespace memref
} // namespace mlir
