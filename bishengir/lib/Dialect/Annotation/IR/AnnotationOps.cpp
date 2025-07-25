/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file AnnotationOps.cpp
 * \brief Implementation of Annotation Dialect Ops
 */

#include "bishengir/Dialect/Annotation/IR/Annotation.h"

using namespace mlir;
using namespace mlir::annotation;

//===----------------------------------------------------------------------===//
// MarkOp
//===----------------------------------------------------------------------===//

void MarkOp::build(OpBuilder &odsBuilder, OperationState &odsState, Value src) {
  build(odsBuilder, odsState, src, /*values=*/ValueRange{}, /*keys=*/nullptr);
}

bool MarkOp::isAnnotatedBy(StringRef key) {
  return isAnnotatedByStaticAttr(key) || isAnnotatedByDynamicAttr(key);
}

bool MarkOp::isAnnotatedByStaticAttr(StringRef key) {
  return (*this)->hasAttr(key);
}

bool MarkOp::isAnnotatedByDynamicAttr(StringRef key) {
  if (!getKeys())
    return false;

  return llvm::any_of(getKeysAttr().getValue(), [&](Attribute attr) {
    return cast<StringAttr>(attr).getValue() == key;
  });
}

OpFoldResult MarkOp::getMixedAttrValue(StringRef key) {
  if (isAnnotatedByStaticAttr(key))
    return OpFoldResult{getStaticAttrValue(key)};

  return OpFoldResult{getDynamicAttrValue(key)};
}

Attribute MarkOp::getStaticAttrValue(StringRef key) {
  return (*this)->getAttr(key);
}

Value MarkOp::getDynamicAttrValue(StringRef key) {
  for (auto [storedKey, value] :
       llvm::zip_equal(getKeysAttr().getValue(), getValues())) {
    if (cast<StringAttr>(storedKey).getValue() == key)
      return value;
  }
  return Value();
}