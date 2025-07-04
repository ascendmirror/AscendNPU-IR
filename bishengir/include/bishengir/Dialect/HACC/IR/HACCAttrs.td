//===- HACCAttrs.td - HACC attributes definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for enumeration and attributes used in
// heterogeneous async computing call operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HACC_IR_HACCATTRS_TD
#define BISHENGIR_DIALECT_HACC_IR_HACCATTRS_TD

include "bishengir/Dialect/HACC/IR/HACCBase.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"

//===----------------------------------------------------------------------===//
// HACC Enums and Attributes
//===----------------------------------------------------------------------===//

class HACC_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hacc";
}

class HACC_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<HACC_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Function Type
//===----------------------------------------------------------------------===//

def HACC_HOST : I32EnumAttrCase<"HOST", 1>;
def HACC_DEVICE : I32EnumAttrCase<"DEVICE", 2>;
def HACC_FuncTypeEnum : HACC_I32Enum<"HACCFuncType", "HACC Function Category", [
  HACC_HOST,
  HACC_DEVICE
]> {}

def HACC_FuncTypeAttr : HACC_Attr<"HACCFuncType", "function_kind"> {
  let parameters = (ins EnumParameter<HACC_FuncTypeEnum>:$function_kind);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HACC function type.
  }];
}

#endif // BISHENGIR_DIALECT_HACC_IR_HACCATTRS_TD
