/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file HIVMBase.td
 * \brief HIVM dialect base support
 * \details This is the definition file for base Hybrid Intelligence Virtual
 *          Machine (HIVM) support.
 */

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
    "arith::ArithDialect",
#ifndef BISHENGIR_BUILD_STANDALONE_IR_ONLY
    "bishengir::memref_ext::MemRefExtDialect",
#endif
    "math::MathDialect",
    "memref::MemRefDialect",
#ifndef BISHENGIR_BUILD_STANDALONE_IR_ONLY
    "hacc::HACCDialect",
#endif
    "tensor::TensorDialect"
  ];
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// HIVM Types.
//===----------------------------------------------------------------------===//

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "Tensor or Memref">;

//===----------------------------------------------------------------------===//
// HIVM Dialect operations.
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
        DeclareOpInterfaceMethods<HIVMStructuredOpInterface>,
      ], traits)> {
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
