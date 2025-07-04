//===- HACC.h - Heterogeneous Async Computing Call dialect ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HACC_IR_HACC_H
#define BISHENGIR_DIALECT_HACC_IR_HACC_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

//===----------------------------------------------------------------------===//
// HACC Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACCEnums.h.inc"

#include "bishengir/Dialect/HACC/IR/HACCBaseDialect.h.inc"

// generated type declarations
#define GET_TYPEDEF_CLASSES
#include "bishengir/Dialect/HACC/IR/HACCTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HACC/IR/HACCAttrs.h.inc"


#endif // BISHENGIR_DIALECT_HACC_IR_HACC_H
