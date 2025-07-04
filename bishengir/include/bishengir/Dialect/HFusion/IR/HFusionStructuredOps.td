//===- HFusionStructuredOps.td - HFusion library ops -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the operation definition file for structured operations on buffers
// that correspond to underlying library calls.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONSTRUCTUREDOPS_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONSTRUCTUREDOPS_TD

include "bishengir/Dialect/HFusion/IR/HFusionAttrs.td"
include "bishengir/Dialect/HFusion/IR/HFusionBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

// Base Tablegen class for HFusion ops that follows LinalgOp Interface.
class HFusionStructuredBase_Op<string mnemonic, list<Trait> props = []>
  : Op<HFusion_Dialect, mnemonic, !listconcat([
       SingleBlockImplicitTerminator<"mlir::linalg::YieldOp">,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DestinationStyleOpInterface,
       LinalgStructuredInterface,
       ReifyRankedShapedTypeOpInterface], props)> {
  code structuredOpsBaseDecls = [{
    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      return !this->getBody()->getOps<mlir::linalg::IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return llvm::cast<mlir::linalg::LinalgOp>(
               getOperation()).reifyResultShapes(b, reifiedReturnShapes);
    }
  }];
}

//===----------------------------------------------------------------------===//
// Named HFusion ops, implemented as a declarative configurations of generic ops.
//===----------------------------------------------------------------------===//

include "bishengir/Dialect/HFusion/IR/HFusionNamedStructuredOps.yamlgen.td"

//===----------------------------------------------------------------------===//
// ReduceWithIndexOp
//===----------------------------------------------------------------------===//

def ReduceWithIndexOp : HFusionStructuredBase_Op<"reduce_with_index",
    [AttrSizedOperandSegments]> {
  let summary = "max/min reduce with index.";
  let description = [{
    Using max/min to perform a reduce operation on an AnyShaped.
    It supports two modes: (1) take input (AnyShaped) and
    index (AnyShaped), produce resulting input and index;
    (2) take input, produce resulting input and index.
    Currently only one reduction dimension is supported.
  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$inits,
                       HFusion_ReduceWithIndexOpAttr:$reduce_kind,
                       ConfinedAttr<DenseI64ArrayAttr,
                         [DenseArrayStrictlySorted<DenseI64ArrayAttr>]>:$dimensions
  );
  let results = (outs Variadic<AnyTensor>:$result);
  let regions = (region AnyRegion:$region);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
      (ins
        "TypeRange":$types,
        "ValueRange":$inputs,
        "ValueRange":$inits,
        "::mlir::hfusion::ReduceWithIndexKindAttr":$reduce_kind,
        "DenseI64ArrayAttr":$dimensions),
      [{
        $_state.addAttribute("reduce_kind", reduce_kind);
        $_state.addAttribute("dimensions", dimensions);
        buildStructuredOp($_builder, $_state, types, inputs, inits,
          {}, ReduceWithIndexOp::getRegionBuilder());
      }]
    >,
    OpBuilder<
      (ins
        "TypeRange":$types,
        "ValueRange":$inputs,
        "ValueRange":$inits,
        "::mlir::hfusion::ReduceWithIndexKindAttr":$reduce_kind,
        "ArrayRef<int64_t>":$dimensions),
      [{
        $_state.addAttribute("reduce_kind", reduce_kind);
        $_state.addAttribute("dimensions", $_builder.getDenseI64ArrayAttr(dimensions));
        buildStructuredOp($_builder, $_state, types, inputs, inits,
          {}, ReduceWithIndexOp::getRegionBuilder());
      }]
    >
  ];

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Declare functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable();

    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
    std::string getLibraryCallName();
    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>)>
    getRegionBuilder();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ArangeOp
//===----------------------------------------------------------------------===//

def ArangeOp : HFusionStructuredBase_Op<"arange", [AttrSizedOperandSegments]> {
  let arguments = (ins Optional<Index>:$offset,
                       Variadic<Index>:$strides,
                       AnyShaped:$init
  );
  let results = (outs Optional<AnyShaped>:$result_tensor);
  let regions = (region AnyRegion:$region);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder< (ins "Value":$init),
      [{
        SmallVector<Value, 8> strides;
        hfusion::ArangeOp::getStridesFromValue($_builder, $_state.location, init, strides);
        hfusion::ArangeOp::build($_builder, $_state, strides, init);
      }]
    >,
    OpBuilder< (ins "ValueRange":$strides,
                    "Value":$init),
      [{
        $_state.addOperands(strides);
        $_state.addOperands(init);
        if (isa<TensorType>(init.getType()))
          $_state.addTypes(init.getType());
        $_state.addAttribute("operandSegmentSizes",
                             $_builder.getDenseI32ArrayAttr({0, static_cast<int32_t>(strides.size()), 1}));
        Region &region = *$_state.addRegion();
        fillStructuredOpRegion($_builder, region, strides, init.getType(),
                               $_state.attributes.getAttrs(), ArangeOp::getRegionBuilder());
      }]
    >,
    OpBuilder< (ins "Value":$offset,
                    "ValueRange":$strides,
                    "Value":$init),
      [{
        SmallVector<Value, 8> inputs{offset};
        inputs.append(strides.begin(), strides.end());
        $_state.addOperands(inputs);
        $_state.addOperands(init);
        if (isa<TensorType>(init.getType()))
          $_state.addTypes(init.getType());
        $_state.addAttribute("operandSegmentSizes",
                             $_builder.getDenseI32ArrayAttr({1, static_cast<int32_t>(strides.size()), 1}));
        Region &region = *$_state.addRegion();
        fillStructuredOpRegion($_builder, region, ValueRange(inputs), init.getType(),
                               $_state.attributes.getAttrs(), ArangeOp::getRegionBuilder());
      }]
    >
  ];

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Declare functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable();

    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>)>
    getRegionBuilder();
    /// Precondition: `val` must be of type ShapedType
    static void getStridesFromValue(OpBuilder &builder, Location loc, Value val, SmallVectorImpl<Value> &strides);
    std::string getLibraryCallName();
  }];

  let hasCustomAssemblyFormat = 1;
}

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONSTRUCTUREDOPS_TD
