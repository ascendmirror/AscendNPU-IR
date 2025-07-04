//===-------- HFusionOps.td - HFusion op definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the operation definition file for hybrid fusion operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONOPS_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "bishengir/Dialect/HFusion/IR/HFusionBase.td"
include "bishengir/Dialect/HFusion/IR/HFusionAttrs.td"

// Base class for HFusion dialect ops that do not correspond to library calls.
class HFusion_Op<string mnemonic, list<Trait> traits = []> :
    Op<HFusion_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// HFusion Types
//===----------------------------------------------------------------------===//

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "Tensor or Memref">;

//===----------------------------------------------------------------------===//
// IsInfOp
//===----------------------------------------------------------------------===//

def IsInfOp : HFusion_Op<"isinf", [Pure, SameOperandsAndResultRank,
                                   AllTypesMatch<["output"]>]> {
  let summary = [{
      Calculates whether elements of a float type tensor equal positive
      infinity or negative infinity.
  }];
  let description = [{
      Calculates whether elements of a float type tensor is infinity.
      Both positive infinity and negative infinity work.
  }];
  let arguments = (ins RankedTensorOf<[BF16, F16, F32]>:$input);
  let results = (outs RankedTensorOf<[I1]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// IsNanOp
//===----------------------------------------------------------------------===//

def IsNanOp : HFusion_Op<"isnan", [Pure, SameOperandsAndResultRank,
                                   AllTypesMatch<["output"]>]> {
  let summary = [{
      Calculates whether elements of a float type tensor is NAN.
  }];
  let description = [{
      Calculates whether elements of a float type tensor is NAN.
  }];
  let arguments = (ins RankedTensorOf<[BF16, F16, F32]>:$input);
  let results = (outs RankedTensorOf<[I1]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// IsFiniteOp
//===----------------------------------------------------------------------===//

def IsFiniteOp : HFusion_Op<"isfinite", [Pure, SameOperandsAndResultRank,
                                         AllTypesMatch<["output"]>]> {
  let summary = [{
      Calculates whether elements of a float type tensor is finite.
  }];
  let description = [{
      Calculates whether elements of a float type tensor is finite
      (i.e., not NaN or infinity).
  }];
  let arguments = (ins RankedTensorOf<[BF16, F16, F32]>:$input);
  let results = (outs RankedTensorOf<[I1]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONOPS_TD
