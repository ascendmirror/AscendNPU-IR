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
 * \file HIVMDialect.cpp
 * \brief BiShengIR HIVM Dialect implementation
 */

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::hivm;

#include "bishengir/Dialect/HIVM/IR/HIVMEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMAttrs.cpp.inc"

#include "bishengir/Dialect/HIVM/IR/HIVMDialect.cpp.inc"

void hivm::HIVMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMAttrs.cpp.inc"
      >();
}
