//===- HIVMBase.td - HIVM dialect base support -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for base Hybrid Intelligence Virtual
// Machine (HIVM) support.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMBASE_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMBASE_TD

include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// HIVM Dialect.
//===----------------------------------------------------------------------===//

def HIVM_Dialect : Dialect {
  let name = "hivm";
  let description = [{
    HIVM (Hybrid Intelligence Virtual Machine) dialect.
  }];
  let cppNamespace = "::mlir::hivm";
  let dependentDialects = [
    "arith::ArithDialect", "math::MathDialect",
    "memref::MemRefDialect", "tensor::TensorDialect"
  ];
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// HIVM Types.
//===----------------------------------------------------------------------===//

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "Tensor or Memref">;

//===----------------------------------------------------------------------===//
// HIVM Dialect Operations.
//===----------------------------------------------------------------------===//

class HIVM_Op<string mnemonic, list<Trait> traits = []>
    : Op<HIVM_Dialect, !strconcat("hir.", mnemonic), traits> {
  // C++ function that returns the op name.
  code opNameDeclaration = [{
    static StringRef getOpName() { return "}] # mnemonic # [{"; }
  }];
}

class HIVM_StructuredOp<string mnemonic, list<Trait> traits = []>
    : HIVM_Op<mnemonic, 
      !listconcat([
        HIVMStructuredOpInterface,
        MemoryEffectsOpInterface,
        DeclareOpInterfaceMethods<HIVMStructuredOpInterface>
      ], traits)> {
  // C++ functions that are required for all structured ops.
  code structuredOpBaseDecls = [{
    bool payloadUsesValueFromOperand(OpOperand * opOperand) {
      if (isDpsInit(opOperand)) return false;
      return !getMatchingBlockArgument(opOperand).use_empty();
    }
  }];
  // C++ functions that are required for parellel structured ops.
  code parallelIndexingMapsDecl = [{
    ArrayAttr getIndexingMaps() {
      MLIRContext *context = getContext();
      AffineMap scalarMap = AffineMap::get(getNumParallelLoops(), 0, context);
      AffineMap tensorMap = AffineMap::getMultiDimIdentityMap(
        getNumParallelLoops(), context);
      SmallVector<AffineMap> indexingMaps;
      for (OpOperand &opOperand : getOperation()->getOpOperands())
        indexingMaps.push_back(getRank(&opOperand) == 0 ? scalarMap : tensorMap);
      return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
    }
  }];
}

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMBASE_TD
