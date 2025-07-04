//===- HFusionEnums.td - HFusion dialect base support ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for enums used in hfusion operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONENUMS_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONENUMS_TD

include "bishengir/Dialect/HFusion/IR/HFusionBase.td"
include "mlir/IR/EnumAttr.td"

// Define the function attribute enums matching the OpDSL functions.
def UnaryFn : I32EnumAttr<"UnaryFn", "", [
  I32EnumAttrCase<"relu", 0>,
  I32EnumAttrCase<"sqrt", 1>,
  I32EnumAttrCase<"rsqrt", 2>,
  I32EnumAttrCase<"rec", 3>,
  I32EnumAttrCase<"vnot", 4>,
  I32EnumAttrCase<"tanh", 5>,
  I32EnumAttrCase<"sin", 6>,
  I32EnumAttrCase<"cos", 7>,
  I32EnumAttrCase<"atan", 8>,
  I32EnumAttrCase<"tan", 9>,
  I32EnumAttrCase<"absi", 10>,
  I32EnumAttrCase<"erf", 11>,
  I32EnumAttrCase<"log2", 12>,
  I32EnumAttrCase<"log10", 13>,
  I32EnumAttrCase<"log1p", 14>,
  I32EnumAttrCase<"exp2", 15>,
  I32EnumAttrCase<"expm1", 16>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

def BinaryFn : I32EnumAttr<"BinaryFn", "", [
  I32EnumAttrCase<"vor", 0>,
  I32EnumAttrCase<"vand", 1>,
  I32EnumAttrCase<"vxor", 2>,
  I32EnumAttrCase<"minf", 3>,
  I32EnumAttrCase<"maxf", 4>,
  I32EnumAttrCase<"powf", 5>,
  I32EnumAttrCase<"mod", 6>,
  I32EnumAttrCase<"shli", 7>,
  I32EnumAttrCase<"shrsi", 8>,
  I32EnumAttrCase<"shrui", 9>,
  I32EnumAttrCase<"ceildivsi", 10>,
  I32EnumAttrCase<"ceildivui", 11>,
  I32EnumAttrCase<"floordivsi", 12>,
  I32EnumAttrCase<"powi", 13>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

def CompareFn : I32EnumAttr<"CompareFn", "", [
  I32EnumAttrCase<"veq", 0>,
  I32EnumAttrCase<"vne", 1>,
  I32EnumAttrCase<"vle", 2>,
  I32EnumAttrCase<"vlt", 3>,
  I32EnumAttrCase<"vge", 4>,
  I32EnumAttrCase<"vgt", 5>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

def TernaryFn : I32EnumAttr<"TernaryFn", "", [
  I32EnumAttrCase<"select", 0>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

def TypeFn : I32EnumAttr<"TypeFn", "", [
  I32EnumAttrCase<"cast_signed", 0>,
  I32EnumAttrCase<"cast_unsigned", 1>,
  I32EnumAttrCase<"bitcast", 2>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

def RoundMode : I32EnumAttr<"RoundMode", "", [
  I32EnumAttrCase<"RINT", 0, "rint">,
  I32EnumAttrCase<"ROUND", 1, "round">,
  I32EnumAttrCase<"FLOOR", 2, "floor">,
  I32EnumAttrCase<"CEIL", 3, "ceil">,
  I32EnumAttrCase<"TRUNC", 4, "trunc">,
  I32EnumAttrCase<"ODD", 5, "odd">,
  I32EnumAttrCase<"TRUNCWITHOVERFLOW", 6, "truncwithoverflow">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
  let description = [{
    - RINT: round to nearest, tie to even (c language rint)
    - ROUND: round to nearest, tie away from zero (c language round)
    - FLOOR: round to minus infinity (c language floor)
    - CEIL: round to positive infinity (c language ceil)
    - TRUNC: round to zero (c language trunc)
    - ODD: round to odd (Von Neumann rounding)
  }];
}

def AtomicKind : I32EnumAttr<"AtomicKind", "", [
  I32EnumAttrCase<"NONE", 0, "none">,
  I32EnumAttrCase<"ADD", 1, "add">,
  I32EnumAttrCase<"MAX", 2, "max">,
  I32EnumAttrCase<"MIN", 3, "min">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

def ReduceWithIndexKind : I32EnumAttr<"ReduceWithIndexKind", "", [
  I32EnumAttrCase<"MIN", 0, "min">,
  I32EnumAttrCase<"MAX", 1, "max">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hfusion";
}

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONENUMS_TD
