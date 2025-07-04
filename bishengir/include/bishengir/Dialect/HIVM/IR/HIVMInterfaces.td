//===- HIVMInterfaces.td - HIVM dialect interfaces defs ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the interface for HIVM dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

include "bishengir/Dialect/HIVM/Interfaces/ExtraBufferOpInterface.td"
include "bishengir/Dialect/HIVM/Interfaces/OpPipeInterface.td"

def HIVMCoreTypeInterface : OpInterface<"CoreTypeInterface"> {
  let description = [{
    Interface to query the core type of HIVM Ops.

    There are 2 use cases:
    1. CoreType is staticly attatched to an op by trait `CoreTypeTrait`.
    2. CoreType is dynamicly inferred through `inferCoreTypeInterface`.
  }];
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
    /*desc=*/"Returns the core type of an op.",
    /*retTy=*/"std::optional<::mlir::hivm::TCoreType>",
    /*methodName=*/"getCoreType",
    /*args=*/(ins),
    /*methodBody=*/[{
      return ::mlir::hivm::detail::queryCoreTypeHelper(tablegen_opaque_val);
    }]>,
  ];
}

def HIVMInferCoreTypeInterface : OpInterface<"InferCoreTypeInterface"> {
  let description = [{
    Interface to infer the core type of HIVM Ops.
  }];
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
    /*desc=*/"Returns the core type of an op.",
    /*retTy=*/"std::optional<::mlir::hivm::TCoreType>",
    /*methodName=*/"inferCoreType",
    /*args=*/(ins),
    /*methodBody=*/"",
    /*defaultImplementation=*/[{
      return std::nullopt;
    }]
    >,
  ];
}

def HIVMStructuredOpInterface : OpInterface<"HIVMStructuredOp",
  [DestinationStyleOpInterface, OpPipeInterface, HIVMCoreTypeInterface]> {
  let cppNamespace = "::mlir::hivm";
  let description = [{
      Hybrid Intelligence Virtual Machine (HIVM) Structured Operation interface.
  }];
  let methods = [
    //===------------------------------------------------------------------===//
    // Operation property handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return whether the op is an elemwise n-ary op.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isElemwiseNaryOp",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::isElemwiseNaryOpImpl($_op);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether the op support inline broadcast.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isInlineBroadcastable",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ConcreteOp::template hasTrait<mlir::OpTrait::BroadcastableOTF>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether the op support inline transpose.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isInlineTransposable",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ConcreteOp::template hasTrait<mlir::OpTrait::TransposableOTF>();
      }]
    >,
    //===------------------------------------------------------------------===//
    // Loop types handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the total number of loops within the current operation.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumLoops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getIteratorTypesArray().size();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of parallel loops.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumParallelLoops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::count($_op.getIteratorTypesArray(),
          mlir::hivm::IteratorType::kParallel);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are parallel loops.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getParallelLoopDims",
      /*args=*/(ins "SmallVectorImpl<int64_t> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::findPositionsOfType($_op.getIteratorTypesArray(),
          mlir::hivm::IteratorType::kParallel, res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are reduction loops.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getReductionLoopDims",
      /*args=*/(ins "SmallVectorImpl<int64_t> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::findPositionsOfType($_op.getIteratorTypesArray(),
          mlir::hivm::IteratorType::kReduction, res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are broadcast loops (including inline broadcast).
      }],
      /*retTy=*/"void",
      /*methodName=*/"getBroadcastLoopDims",
      /*args=*/(ins "SmallVectorImpl<int64_t> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::findPositionsOfType($_op.getIteratorTypesArray(),
          mlir::hivm::IteratorType::kBroadcast, res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are transpose loops (including inline transpose).
      }],
      /*retTy=*/"void",
      /*methodName=*/"getTransposeLoopDims",
      /*args=*/(ins "SmallVectorImpl<int64_t> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::findPositionsOfType($_op.getIteratorTypesArray(),
          mlir::hivm::IteratorType::kTranspose, res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the permutation array.
      }],
      /*retTy=*/"ArrayRef<int64_t>",
      /*methodName=*/"getPermutationArray",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getPermutationArray($_op);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are pad loops.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getPadLoopDims",
      /*args=*/(ins "SmallVectorImpl<int64_t> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::findPositionsOfType($_op.getIteratorTypesArray(),
          mlir::hivm::IteratorType::kPad, res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are concat loops.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getConcatLoopDims",
      /*args=*/(ins "SmallVectorImpl<int64_t> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::findPositionsOfType($_op.getIteratorTypesArray(),
          mlir::hivm::IteratorType::kConcat, res);
      }]
    >,
    //===------------------------------------------------------------------===//
    // Other interface methods.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the iterator types of the current operation.
      }],
      /*retTy=*/"SmallVector<::mlir::hivm::IteratorType>",
      /*methodName=*/"getIteratorTypesArray",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if ($_op.isElemwiseNaryOp())
          return hivm::detail::getIteratorTypesArrayForElemwiseOp($_op);

        llvm_unreachable("get iterator not implemented");
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the iterator types of the current operation.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"setIteratorTypesArray",
      /*args=*/(ins "const IteratorType":$iteratorType, "const DenseI64ArrayAttr &":$arrayAttr),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if ($_op.isElemwiseNaryOp())
          return hivm::detail::setIteratorTypesArrayForElemwiseOp($_op, iteratorType, arrayAttr);

        llvm_unreachable("set iterator not implemented");
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get memory effects for HIVM Structured Op.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getEffects",
      /*args=*/(ins "::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<MemoryEffects::Effect>>&":$effects),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if ($_op.hasPureTensorSemantics()) {
          return;
        }
        ::mlir::hivm::detail::getEffectsImpl(effects,
          $_op.getResults(), $_op.getDpsInputs(), $_op.getDpsInits());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get operands with target memref space.
      }],
      /*retTy=*/"SmallVector<Value>",
      /*methodName=*/"getTargetSpaceOperands",
      /*args=*/(ins "hivm::AddressSpace":$AddressSpace,"bool":$includeTmpBuffer),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getTargetSpaceOperandsImpl(
          $_op, AddressSpace, includeTmpBuffer);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the op with or without temp buffer.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpOperand *>",
      /*methodName=*/"getHIVMOperands",
      /*args=*/(ins "bool":$includeExtraBuffer),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getHIVMOperandsImpl($_op, includeExtraBuffer);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the op type with or without temp buffer.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::Type>",
      /*methodName=*/"getHIVMOperandTypes",
      /*args=*/(ins "bool":$includeExtraBuffer),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getHIVMOperandTypesImpl($_op, includeExtraBuffer);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Check if the operand is vector only at a specific index.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isVectorOnlyOperand",
      /*args=*/(ins "size_t":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::isVectorOnlyOperandImpl($_op, idx);
      }]
    >,
    /// Need to implement this for every Concrete op
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing maps attribute within the current operation.
      }],
      /*retTy=*/"ArrayAttr",
      /*methodName=*/"getIndexingMaps",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("getIndexingMaps not implemented");
        return ::mlir::ArrayAttr();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing maps within the current operation.
      }],
      /*retTy=*/"SmallVector<AffineMap>",
      /*methodName=*/"getIndexingMapsArray",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto range = $_op.getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        return {range.begin(), range.end()};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing maps within the current operation.
      }],
      /*retTy=*/"SmallVector<OpFoldResult>",
      /*methodName=*/"createFlatListOfMixedShapeOperand",
      /*args=*/(ins "OpBuilder &":$b, "Location &":$loc),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return cast<HIVMStructuredOp>(this->getOperation()).createFlatListOfOperandDims(b, loc);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing maps within the current operation.
      }],
      /*retTy=*/"SmallVector<AffineMap>",
      /*methodName=*/"getIndexingMapsArrayForSCF",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getIndexingMapsArray();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Hook to provide a custom AffineMap used to compute all the operand
        subshapes given loop bounds. This is used to answer the question: "given
        an iteration space over the codomain, what are the subshapes of the
        operands involved in the computation".
        The default behavior is to just concatenate all the indexing maps.
        A custom AffineMap allows providing a map that can be used to
        compute subshapes even in cases where the concatenation of indexing maps
        (i.e. the data traversal order) is not a simple permutation of the loop
        traversal order. It is then possible to define ops with skewed data
        traversal order for which we can still easily compute hyperrectangular
        loop bounds and subviews.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getLoopsToShapesMap",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto maps = $_op.getIndexingMapsArray();
        return concatAffineMaps(maps);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `opOperand` rank or zero for scalars or
        vectors not wrapped within a tensor or a memref.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getRank",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        Type t = opOperand->get().getType();
        // A VectorType is an elemental type, do not consider its rank for the operand.
        if (isa<VectorType>(t))
          return 0;
        // Tensor and Memref container types have a rank.
        if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
          // Failsafe.
          assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
                          "expected a ranked tensor or memref in LinalgInterface::getRank");
          return shapedType.getRank();
        }
        return 0;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `opOperand` shape or an empty vector for scalars or vectors
        not wrapped within a tensor or a memref.
      }],
      /*retTy=*/"ArrayRef<int64_t>",
      /*methodName=*/"getShape",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        Type t = opOperand->get().getType();
        if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
          // Failsafe.
          assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
                          "expected a ranked tensor or memref in LinalgInterface::getRank");
          return shapedType.getShape();
        }
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Like `getShape`, but only returns statically-known information, without
        generating any new IR. For each shape dimension, returns >=0 if that
        dimension is statically known, or ShapedType::kDynamic otherwise.
      }],
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getStaticShape",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<int64_t> res;
        for (OpOperand &opOperand : this->getOperation()->getOpOperands())
          llvm::append_range(res, getShape(&opOperand));
        return res;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if any of the operands has a dynamic shape.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasDynamicShape",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::any_of(getStaticShape(), ShapedType::isDynamic);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the input or output indexing map for `opOperand`.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getMatchingIndexingMap",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        auto indexingMaps =
            $_op.getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        return *(indexingMaps.begin() + opOperand->getOperandNumber());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing map for a `result`.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getIndexingMapMatchingResult",
      /*args=*/(ins "OpResult":$result),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(result.getOwner() == this->getOperation());
        auto indexingMaps =
            $_op.getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        return *(indexingMaps.begin() + $_op.getNumDpsInputs() +
                          result.getResultNumber());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the block argument for an `opOperand`.
      }],
      /*retTy=*/"BlockArgument",
      /*methodName=*/"getMatchingBlockArgument",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("not implemented");
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the single block constituting the body of the operation by
        calling the getBody method on the concrete operation.
      }],
      /*retTy=*/"Block*",
      /*methodName=*/"getBlock",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("HIVM Op is defined to not have block");
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return yes if an operation has an index semantics
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasIndexSemantics",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
  ];

  let extraClassDeclaration = [{
    /// Return the flat list of all operand dimension sizes in the order they
    /// appear in the operands.
    SmallVector<OpFoldResult> createFlatListOfOperandDims(OpBuilder &, Location);

    /// Return the flat list of all operands' static dimension sizes in the
    /// order they appear in the operands. All operand dimension sizes have to
    /// be statically known.
    SmallVector<int64_t, 4> createFlatListOfOperandStaticDims();

    /// Create the loop ranges to materialize the computation over the current
    /// operands. This is done by applying `getShapesToLoopsMap` to
    /// `createFlatListOfOperandDims`.
    SmallVector<Range, 4> createLoopRanges(OpBuilder &b, Location loc);

    /// Compute the static loop sizes necessary to vectorize the computation.
    /// This is done by applying `getShapesToLoopsMap` to
    /// `createFlatListOfOperandStaticDims`.
    SmallVector<int64_t, 4> computeStaticLoopSizes();

    /// Returns the value that expresses the shape of the output in terms of
    /// shape of the input operands where possible
    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes);

    /// Return the index in the indexingMaps vector that corresponds to this `opOperand`
    int64_t getIndexingMapIndex(OpOperand *opOperand);
  }];
}

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_TD
