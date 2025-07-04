//===- ExtraBufferOpInterface.td - Ops with Temp. Buffer ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the ExtraBufferOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_EXTRABUFFEROPINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_EXTRABUFFEROPINTERFACE_TD

include "mlir/IR/OpBase.td"

def ExtraBufferOpInterface : OpInterface<"ExtraBufferOpInterface"> {
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Allocate extra temp buffers for the current operation.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"allocExtraBuffersIfPossible",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the extra temp buffers.
      }],
      /*retTy=*/"OperandRange",
      /*methodName=*/"getExtraBuffers",
      /*args=*/(ins)
    >,
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_EXTRABUFFEROPINTERFACE_TD
