//===- ExecutionEngine.h - Execution Engine dialect -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_EXECUTION_ENGINE_IR_EXECUTIONENGINE_H
#define BISHENGIR_DIALECT_EXECUTION_ENGINE_IR_EXECUTIONENGINE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// Execution Engine Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "bishengir/Dialect/ExecutionEngine/IR/ExecutionEngineBaseDialect.h.inc"

//===----------------------------------------------------------------------===//
// Execution Engine Enums
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/ExecutionEngine/IR/ExecutionEngineEnums.h.inc"

//===----------------------------------------------------------------------===//
// Execution Engine Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/ExecutionEngine/IR/ExecutionEngineAttrs.h.inc"

#endif // BISHENGIR_DIALECT_EXECUTION_ENGINE_IR_EXECUTIONENGINE_H
