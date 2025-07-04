//===- HIVMInterfaces.h - HIVM dialect interface definitions ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_H

#include "bishengir/Dialect/HIVM/Interfaces/ExtraBufferOpInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/OpLayoutInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/OpPipeInterface.h"

#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace hivm {
/// Forward declarations.
enum class TCoreType : uint32_t;
enum class IteratorType : uint32_t;
enum class AddressSpace : uint32_t;

namespace detail {
std::optional<TCoreType> queryCoreTypeHelper(Operation *op);

/// Return positions in `iteratorTypes` that match `iteratorTypeName`.
inline void findPositionsOfType(ArrayRef<IteratorType> iteratorTypes,
                                IteratorType iteratorTypeName,
                                SmallVectorImpl<int64_t> &res) {
  for (const auto &en : llvm::enumerate(iteratorTypes)) {
    if (en.value() == iteratorTypeName)
      res.push_back(en.index());
  }
}

/// Implementation to get HIVM Structured Op's memory effects.
void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, const ValueRange inputOperands,
    ValueRange outputOperands);

/// Implementation to get operands with or without extra buffer.
SmallVector<OpOperand *> getHIVMOperandsImpl(Operation *op,
                                             bool includeExtraBuffer = false);

/// Implementation to get operand types with or without extra buffer.
SmallVector<Type> getHIVMOperandTypesImpl(Operation *op,
                                          bool includeExtraBuffer = false);

/// Get Operands with target memref space
SmallVector<Value> getTargetSpaceOperandsImpl(Operation *op,
                                              AddressSpace hivmSpace,
                                              bool includeExtraBuffer);

/// check if the operand is vector only at a specific index
bool isVectorOnlyOperandImpl(Operation *op, size_t idx);

} // namespace detail
} // namespace hivm
} // namespace mlir

#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h.inc"

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_H
