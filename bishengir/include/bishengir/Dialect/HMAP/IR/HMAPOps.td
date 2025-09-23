//===-------- HMAPOps.td - ----- HMAP op definitions -------*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef HMAP_IR_HMAPOPS
#define HMAP_IR_HMAPOPS

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "bishengir/Dialect/HMAP/IR/HMAPBase.td"

//===----------------------------------------------------------------------===//
// HMAP Dialect operations.
//===----------------------------------------------------------------------===//

class HMAP_Op<string mnemonic, list<Trait> traits = []> :
    Op<HMAP_Dialect, mnemonic, traits> {
}

class HMAP_CollectiveCommunicationOpBase<
    string mnemonic, list<Trait> traits = []> :
    HMAP_Op<mnemonic,
      !listconcat(traits,
        [
          DeclareOpInterfaceMethods<SymbolUserOpInterface>,
          DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
        ])> {
  dag commonArgs = (ins
    FlatSymbolRefAttr:$mesh,
    DefaultValuedAttr<DenseI16ArrayAttr, "{}">:$mesh_axes
  );
}

def HMAP_AllToAllVOp : HMAP_CollectiveCommunicationOpBase<"all_to_all_v", [
  Pure]> {
  let summary = "All-to-all vector over devices.";
  let description = [{
    Performs an all-to-all-v on tensor pieces
  }];
  // TODO: How to interact with mesh?
  let arguments = !con(commonArgs, (ins
                                    AnyNon0RankedTensor:$input,
                                    AnyTensor:$input_splits,
                                    AnyTensor:$output_splits
                                   ) );
  let results = (outs
                 AnyNon0RankedTensor:$result
                );

  let assemblyFormat = [{
    $input `on` $mesh (`mesh_axes` `=` $mesh_axes^)?
    `input_splits` `=` $input_splits `:` type($input_splits)
    `output_splits` `=` $output_splits `:` type($output_splits)
    attr-dict `:` type($input) `->` type($result)
  }];
}

#endif // HMAP_IR_HMAPOPS
