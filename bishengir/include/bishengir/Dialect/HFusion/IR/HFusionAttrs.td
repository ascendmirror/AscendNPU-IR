//===- HFusionAttrs.td - HFusion dialect base support ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for enumeration and attributes used in hybrid
// fusion operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONATTRS_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONATTRS_TD

include "bishengir/Dialect/HFusion/IR/HFusionBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// HFusion Attributes.
//===----------------------------------------------------------------------===//

class HFusion_Attr<string attrName, string attrMnemonic, list<Trait> traits = [
]> : AttrDef<HFusion_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def HFusion_ReduceWithIndexOpAttr : HFusion_Attr<"ReduceWithIndexKind", "reduce_with_index_kind"> {
  let parameters = (ins
    EnumParameter<ReduceWithIndexKind>:$reduce_with_index_kind
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    The kind of reduce with index.
  }];
}

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONATTRS_TD
