//===- HIVM.h - Hybrid Intelligence Virtual Machine Dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVM_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVM_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
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

#include "bishengir/Dialect/HIVM/IR/HIVMTraits.h"

#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"

//===----------------------------------------------------------------------===//
// HIVM Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMOps.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.h.inc"

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVM_H
