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
 * \file HIVM.h
 * \brief Hybrid Intelligence Virtual Machine Dialect
 */

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVM_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVM_H

#include "bishengir/Config/bishengir-config.h"

#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
#include "bishengir/Interfaces/AggregatedOpInterface.h"
#endif

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

//===----------------------------------------------------------------------===//
// HIVM Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVMDialect.h.inc"

//===----------------------------------------------------------------------===//
// HIVM Enums
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVMEnums.h.inc"

//===----------------------------------------------------------------------===//
// HIVM Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMAttrs.h.inc"

//===----------------------------------------------------------------------===//
// HIVM Trait and Interface
//===----------------------------------------------------------------------===//

#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
#include "bishengir/Dialect/HIVM/IR/HIVMTraits.h"

#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#endif

//===----------------------------------------------------------------------===//
// HIVM Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMOps.h.inc"

#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.h.inc"
#endif

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.h.inc"

#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.h.inc"
#endif

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVM_H
