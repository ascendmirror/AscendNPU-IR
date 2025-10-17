//===- ExecutionEngineBase.td - Base definitions for ExecutionEngine dialect --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for base Execution Engine support.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_EXECUTION_ENGINE_IR_EXECUTIONENGINEBASE_TD
#define BISHENGIR_DIALECT_EXECUTION_ENGINE_IR_EXECUTIONENGINEBASE_TD

include "mlir/IR/OpBase.td"

def ExecutionEngine_Dialect : Dialect {
  let name = "execution_engine";
  let description = [{
    Execution Engine dialect.
  }];
  let cppNamespace = "::mlir::execution_engine";
}

#endif // BISHENGIR_DIALECT_EXECUTION_ENGINE_IR_EXECUTIONENGINEBASE_TD
