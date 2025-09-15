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
 * \file HIVMImpl.h
 * \brief HIVM implementation
 */

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMIMPL_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVMIMPL_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace hivm {

/// get operation core type
FailureOr<TCoreType> getCoreType(Operation *op);

//===----------------------------------------------------------------------===//
// Printing/parsing for EventID
//===----------------------------------------------------------------------===//

ParseResult
parseEventID(OpAsmParser &parser, EventAttr &eventIDAttr,
             std::optional<OpAsmParser::UnresolvedOperand> &eventIDValue);

void printEventID(OpAsmPrinter &printer, Operation *op, EventAttr eventIDAttr,
                  Value eventIDValue);

//===----------------------------------------------------------------------===//
// Printing/parsing for FlagID
//===----------------------------------------------------------------------===//

ParseResult
parseFlagID(OpAsmParser &parser, IntegerAttr &flagIDAttr,
            std::optional<OpAsmParser::UnresolvedOperand> &flagIDValue);

void printFlagID(OpAsmPrinter &printer, Operation *op, IntegerAttr flagIDAttr,
                 Value flagIDValue);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMIMPL_H