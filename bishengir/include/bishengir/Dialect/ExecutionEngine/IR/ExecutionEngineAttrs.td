//===- ExecutionEngineAttrs.td - ExecutionEngine attributes definitions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR Execution Engine attributes.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_EXECUTION_ENGINE_EXECUTIONENGINEATTRS_TD
#define BISHENGIR_DIALECT_EXECUTION_ENGINE_EXECUTIONENGINEATTRS_TD

include "mlir/IR/EnumAttr.td"
include "bishengir/Dialect/ExecutionEngine/IR/ExecutionEngineBase.td"

//===----------------------------------------------------------------------===//
// Execution Engine Attributes
//===----------------------------------------------------------------------===//

class ExecutionEngine_Attr<string attrName, string attrMnemonic>
    : AttrDef<ExecutionEngine_Dialect, attrName, []> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Execution Engine Enums.
//===----------------------------------------------------------------------===//

class ExecutionEngine_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::execution_engine";
}

//===----------------------------------------------------------------------===//
// Argument Type
//===----------------------------------------------------------------------===//

def ExecutionEngine_ArgType_InputFilePath : I32EnumAttrCase<"InputFilePath", 1, "input">;
def ExecutionEngine_ArgType_OutputFilePath : I32EnumAttrCase<"OutputFilePath", 2, "output">;
def ExecutionEngine_ArgType_ArgDynSize : I32EnumAttrCase<"ArgDynSize", 3, "dyn_size">;
def ExecutionEngine_ArgType_MemPoolSpace : I32EnumAttrCase<"MemPoolSpace", 4, "mem_pool">;

def ExecutionEngine_ArgTypeEnum : ExecutionEngine_I32Enum<
  "ArgType", "Execution Engine user argument type", [
    ExecutionEngine_ArgType_InputFilePath,
    ExecutionEngine_ArgType_OutputFilePath,
    ExecutionEngine_ArgType_ArgDynSize,
    ExecutionEngine_ArgType_MemPoolSpace
  ]>;

//===----------------------------------------------------------------------===//
// Execution Engine Argument Type Attribute
//===----------------------------------------------------------------------===//

def ExecutionEngine_ArgTypeAttr : ExecutionEngine_Attr<"ArgType", "arg_type"> {
  let parameters = (ins
    "::mlir::execution_engine::ArgType":$argType,
    "::std::optional<::mlir::hivm::AddressSpace>":$memPoolSpace
  );
  let description = [{
    Execution Engine user argument type attribute.
  }];
  let builders = [
    AttrBuilder<(ins "::mlir::execution_engine::ArgType":$nonMemPoolArgType)>,
    AttrBuilder<(ins "::mlir::hivm::AddressSpace":$memPoolSpace)>
  ];
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Execution Engine Memory Pool Argument Attribute
//===----------------------------------------------------------------------===//

def ExecutionEngine_MemoryPoolAttr : ExecutionEngine_Attr<"MemoryPool", "memory_pool"> {
  let description = [{
    Execution Engine memory pool argument attribute.
  }];
}

#endif // BISHENGIR_DIALECT_EXECUTION_ENGINE_EXECUTIONENGINEATTRS_TD
