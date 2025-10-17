//===- ExecutionEngineDialect.cpp - Implementation of Execution Engine dialect
//                                  and types -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/ExecutionEngine/IR/ExecutionEngine.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

void mlir::execution_engine::ExecutionEngineDialect::initialize() {}

#include "bishengir/Dialect/ExecutionEngine/IR/ExecutionEngineBaseDialect.cpp.inc"
