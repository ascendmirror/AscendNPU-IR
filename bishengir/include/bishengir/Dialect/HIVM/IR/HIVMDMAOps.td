//===- HIVMDMAOps.td - HIVM dialect DMA operations ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for HIVM dialect data movement action
// (DMA) operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMDMAOPS_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMDMAOPS_TD

include "bishengir/Dialect/HIVM/IR/HIVMBase.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.td"
include "bishengir/Dialect/HIVM/IR/HIVMTraits.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// HIVM DMA Operations.
//===----------------------------------------------------------------------===//

class HIVM_DmaOp<string mnemonic, list<Trait> traits = []> :
  HIVM_StructuredOp<mnemonic, !listconcat(
    [AlwaysSpeculatable,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>
    ],
    traits)> {
  code DmaOpBaseDecl = [{
    ShapedType getSrcOperandType() {
      return getSrc().getType().cast<ShapedType>();
    }
    ShapedType getDstOperandType() {
      return getDst().getType().cast<ShapedType>();
    }
    // Implement functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }] # structuredOpBaseDecls # opNameDeclaration # parallelIndexingMapsDecl;
}

def LoadOp : HIVM_DmaOp<"load",
  [AttrSizedOperandSegments,
   SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE2">,
   DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>,
   OperElemTypeConstraints<[0], [I8, UI8, I16, UI16, F16, BF16,
                                 I32, UI32, F32, UI64, I64]>
  ]> {
  let summary = "HIVM data load operation";
  let description = [{
    Loads the data from the global memory to the local buffer.
    Currently only support loading to the unified buffer.

    Examples:
    ```mlir
    hivm.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
    ```

    Constraints:
    - `src` and `dst` are expected to have the same element type.
    - If `pad_mode` is not set, `src` and `dst` shape should be the same.
    - supports both left and right padding.
    - `pad_value` should have the same element type as `src` and `dst`.
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       OptionalAttr<HIVM_PadModeAttr>:$pad_mode,
                       Optional<AnyType>:$pad_value,
                       Optional<Index>:$left_padding_num,
                       Optional<AnyType>:$right_padding_num
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let builders = [
      OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst)>,
      OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                                      "Value":$left_padding_num)>,
      OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                                      "hivm::PadModeAttr":$pad_mode, "Value":$pad_value)>,
      OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                                      "hivm::PadModeAttr":$pad_mode, "Value":$pad_value,
                                        "Value":$left_padding_num)>,
  ];
  let assemblyFormat = [{
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`pad_mode` `=` $pad_mode^)?
    (`pad_value` `=` $pad_value^ `:` type($pad_value))?
    (`left_padding_num` `=` $left_padding_num^ `:` type($left_padding_num))?
    (`right_padding_num` `=` $right_padding_num^ `:` type($right_padding_num))?
    (`->` type($result_tensor)^)?
  }];
  let hasFolder = 1;
  let hasVerifier = 1; // verify parameter according to hardware version
  let extraClassDeclaration = DmaOpBaseDecl;
}

def StoreOp : HIVM_DmaOp<"store",
  [SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE3">,
   DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>,
   OperElemTypeConstraints<[0], [I8, UI8, I16, UI16, F16, BF16,
                                 I32, UI32, F32, UI64, I64]>
  ]> {
  let summary = "HIVM data store operation";
  let description = [{
    Stores the data on local buffer to global memory.
    Currently only support storing data on the unified buffer.

    Examples:
    ```mlir
    hivm.store ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>)
    ```

    Constraints:
    - `src` and `dst` are expected to have the same element type.
    - If `atomic_kind` is set, the kind is one of `add`, `max`, `min`.
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       OptionalAttr<HIVM_AtomicKindAttr>:$atomic_kind
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let builders = [
      OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst)>
  ];
  let assemblyFormat = [{
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`atomic` `=` $atomic_kind^)?
    (`->` type($result_tensor)^)?
  }];
  let hasFolder = 1;
  let hasVerifier = 1; // verify parameter according to hardware version
  let extraClassDeclaration = DmaOpBaseDecl # [{
    // Return whether atomic store is enabled.
    bool isAtomic();
  }];
}

def CopyOp : HIVM_DmaOp<"copy",
  [SinglePipeOpTrait,
   DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>,
   OperElemTypeConstraints<[0], [I8, UI8, I16, UI16, F16, BF16,
                                 I32, UI32, F32, UI64, I64]>
  ]> {
let summary = "HIVM data copy operation";
let description = [{
    Copy the data between local memory hierarchies. Currently only
    support copying from unified buffer to unified buffer.

    Examples:
    ```mlir
    hivm.copy ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
    ```

    Constraints:
    - `src` and `dst` are expected to have the same element type.
    - If `pad_mode` is not set, `src` and `dst` shape should be the same.
    - Only support left padding.
    - `pad_value` should have the same element type as `src` and `dst`.
}];
let arguments = ( ins TensorOrMemref:$src,
                      TensorOrMemref:$dst,
                      OptionalAttr<HIVM_PadModeAttr>:$pad_mode,
                      Optional<AnyType>:$pad_value
);
let results = (outs Optional<AnyRankedTensor>:$result_tensor);
let builders = [
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst)>
];
let assemblyFormat = [{
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`pad_mode` `=` $pad_mode^)?
    (`pad_value` `=` $pad_value^ `:` type($pad_value))?
    (`->` type($result_tensor)^)?
}];
let hasFolder = 1;
let hasVerifier = 1; // verify parameter according to hardware version
let extraClassDeclaration = DmaOpBaseDecl # [{
  // Declare functions necessary for SinglePipeOpTrait.
  PIPE getPipe();
}];
}

def FixpipeOp : HIVM_DmaOp<"fixpipe",
  [SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_FIX">,
   HIVMCoreTypeInterface, CubeCoreTypeTrait,
   DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "HIVM data copy operation from L0C to L1 or Global Memory";
  let description = [{
    Fixpipe is pipeline that performing data movement from L0C to OUT or L1,
    with on-the-fly fixed function of pre-stage quantization,
    pre-stage ReLU, element-wise add, post-stage ReLU, post-stage quantization.

    Additionally, Fixpipe is also capable of layout transform.
  }];
  let arguments = (ins AnyShaped:$src,
                       AnyShaped:$dst,
                       OptionalAttr<UnitAttr>:$enable_nz2nd,
                       DefaultValuedOptionalAttr<HIVM_FixpipePreQuantModeAttr,
                         "FixpipePreQuantMode::NO_QUANT">:$pre_quant,
                       DefaultValuedOptionalAttr<HIVM_FixpipePreReluModeAttr,
                         "FixpipePreReluMode::NO_RELU">:$pre_relu,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$channel_split
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let builders = [
    OpBuilder<(ins "TypeRange":$result, "Value":$inputs, "Value":$outputs,
                    "UnitAttr":$enable_nz2nd, "int64_t":$pre_quant,
                    "int64_t":$pre_relu, "BoolAttr":$channel_split)
    >
  ];
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`->` type($result_tensor)^)?
  }];
  let extraClassDeclaration = DmaOpBaseDecl # [{
    int getFixpipeState();
    int needFixpipePreFuse();
    bool hasStore();
  }];
}

def ND2NZOp : HIVM_DmaOp<"nd2nz",
  [SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE2">,
   HIVMCoreTypeInterface, CubeCoreTypeTrait,
   DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "HIVM data copy operation with on-the-fly ND to NZ layout transformation";
  let description = [{
    - `dst_continuous`: if present, signify that the source data is stored continuously
      in the destination buffer. This must be set in order for this op to be converted to
      library function call.
  }];
  let arguments = (ins AnyShaped:$src,
                       AnyShaped:$dst,
                       OptionalAttr<UnitAttr>:$dst_continuous
  );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensor);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`->` type($result_tensor)^)?
  }];
  let extraClassDeclaration = DmaOpBaseDecl;
}

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMDMAOPS_TD
