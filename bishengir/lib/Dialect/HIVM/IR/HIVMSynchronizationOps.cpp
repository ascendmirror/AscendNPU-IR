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
 * \file HIVMSynchronizationOps.cpp
 * \brief HIVM dialect synchronization ops implementation.
 */

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"

#include <set>

using namespace mlir;
using namespace mlir::hivm;

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Printing/parsing for EventID
//===----------------------------------------------------------------------===//

ParseResult hivm::parseEventID(
    OpAsmParser &parser, EventAttr &eventIDAttr,
    std::optional<OpAsmParser::UnresolvedOperand> &eventIDValue) {
  OpAsmParser::UnresolvedOperand operand;
  auto res = parser.parseOptionalOperand(operand);
  if (res.has_value() && succeeded(res.value())) {
    eventIDValue = operand;
    return success();
  }
  eventIDValue = std::nullopt;
  if (parser.parseCustomAttributeWithFallback(eventIDAttr, Type{}))
    return failure();

  return success();
}

void hivm::printEventID(OpAsmPrinter &printer, Operation *op,
                        EventAttr eventIDAttr, Value eventIDValue) {
  if (eventIDAttr) {
    eventIDAttr.print(printer);
    return;
  }
  printer << eventIDValue;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for FlagID
//===----------------------------------------------------------------------===//

ParseResult
hivm::parseFlagID(OpAsmParser &parser, IntegerAttr &flagIDAttr,
                  std::optional<OpAsmParser::UnresolvedOperand> &flagIDValue) {
  OpAsmParser::UnresolvedOperand operand;
  auto res = parser.parseOptionalOperand(operand);
  if (res.has_value() && succeeded(res.value())) {
    flagIDValue = operand;
    return success();
  }
  flagIDValue = std::nullopt;
  int64_t integer;
  if (failed(parser.parseInteger(integer)))
    return failure();

  flagIDAttr = IntegerAttr::get(parser.getBuilder().getI64Type(), integer);
  return success();
}

void hivm::printFlagID(OpAsmPrinter &printer, Operation *op,
                       IntegerAttr flagIDAttr, Value flagIDValue) {
  if (flagIDAttr) {
    printer << flagIDAttr.getValue();
    return;
  }
  printer << flagIDValue;
}

//===----------------------------------------------------------------------===//
// SetFlagOp
//===----------------------------------------------------------------------===//

LogicalResult SetFlagOp::verify() {
  auto eventIDAttr = getStaticEventId();
  auto eventID = getDynamicEventId();
  if (eventIDAttr.has_value() && eventID != TypedValue<IntegerType>{}) {
    return emitOpError("Only one Event ID is supported!");
  }
  if (!eventIDAttr.has_value() && eventID == TypedValue<IntegerType>{}) {
    return emitOpError("Event ID is needed!");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// WaitFlagOp
//===----------------------------------------------------------------------===//

LogicalResult WaitFlagOp::verify() {
  auto eventIDAttr = getStaticEventId();
  auto eventID = getDynamicEventId();
  if (eventIDAttr.has_value() && eventID != TypedValue<IntegerType>{}) {
    return emitOpError("Only one Event ID is supported!");
  }
  if (!eventIDAttr.has_value() && eventID == TypedValue<IntegerType>{}) {
    return emitOpError("Event ID is needed!");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SyncBlockSetOp
//===----------------------------------------------------------------------===//

OpFoldResult SyncBlockSetOp::getFlagId() {
  if (auto attr = getStaticFlagId()) {
    return attr.value();
  }
  return getDynamicFlagId();
}

LogicalResult SyncBlockSetOp::verify() {
  auto flagIdAttr = getStaticFlagId();
  auto flagIdValue = getDynamicFlagId();
  if (flagIdAttr.has_value() && flagIdValue != TypedValue<IntegerType>{}) {
    return emitOpError("Only one flag ID is supported!");
  }

  if (!flagIdAttr.has_value() && flagIdValue == TypedValue<IntegerType>{}) {
    return emitOpError("Flag ID is needed!");
  }
  return success();
}

void SyncBlockSetOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                           TCoreTypeAttr tcore_type, PipeAttr tpipe,
                           PipeAttr pipe, OpFoldResult flag_id) {
  if (auto attr = dyn_cast_if_present<Attribute>(flag_id)) {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe,
          /*static_flag_id=*/cast<IntegerAttr>(attr),
          /*dynamic_flag_id=*/nullptr, /*ffts_base_addr=*/nullptr,
          /*tsync_instr_mode=*/{});
  } else {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe,
          /*static_flag_id=*/nullptr,
          /*dynamic_flag_id=*/flag_id.get<Value>(),
          /*ffts_base_addr=*/nullptr,
          /*tsync_instr_mode=*/{});
  }
}

void SyncBlockSetOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                           TCoreTypeAttr tcore_type, PipeAttr tpipe,
                           PipeAttr pipe, OpFoldResult flag_id,
                           Value ffts_base_addr,
                           SyncBlockInstrModeAttr tsync_instr_mode) {
  if (auto attr = dyn_cast_if_present<Attribute>(flag_id)) {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe,
          /*static_flag_id=*/cast<IntegerAttr>(attr),
          /*dynamic_flag_id=*/nullptr, ffts_base_addr, tsync_instr_mode);
  } else {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe,
          /*static_flag_id=*/nullptr,
          /*dynamic_flag_id=*/flag_id.get<Value>(), ffts_base_addr,
          tsync_instr_mode);
  }
}

//===----------------------------------------------------------------------===//
// SyncBlockWaitOp
//===----------------------------------------------------------------------===//

OpFoldResult SyncBlockWaitOp::getFlagId() {
  if (auto attr = getStaticFlagId()) {
    return attr.value();
  }
  return getDynamicFlagId();
}

LogicalResult SyncBlockWaitOp::verify() {
  auto flagIdAttr = getStaticFlagId();
  auto flagIdValue = getDynamicFlagId();
  if (flagIdAttr.has_value() && flagIdValue != TypedValue<IntegerType>{}) {
    return emitOpError("Only one flag ID is supported!");
  }

  if (!flagIdAttr.has_value() && flagIdValue == TypedValue<IntegerType>{}) {
    return emitOpError("Flag ID is needed!");
  }
  return success();
}

void SyncBlockWaitOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                            TCoreTypeAttr tcore_type, PipeAttr tpipe,
                            PipeAttr pipe, OpFoldResult flag_id) {
  if (auto attr = dyn_cast_if_present<Attribute>(flag_id)) {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe,
          /*static_flag_id=*/cast<IntegerAttr>(attr),
          /*dynamic_flag_id=*/nullptr);
  } else {
    build(odsBuilder, odsState, tcore_type, tpipe, pipe,
          /*static_flag_id=*/nullptr, /*dynamic_flag_id=*/flag_id.get<Value>());
  }
}

//===----------------------------------------------------------------------===//
// SyncBlockOp
//===----------------------------------------------------------------------===//

LogicalResult SyncBlockOp::verify() {
  auto syncBlockMode = getSyncBlockModeAttr().getSyncMode();
  if (syncBlockMode == SyncBlockMode::BARRIER_CUBE ||
      syncBlockMode == SyncBlockMode::BARRIER_VECTOR) {
    if (getTvectorPipeAttr() != nullptr) {
      return emitOpError("tvector_pipe should not be defined!");
    }
    if (getTcubePipeAttr() != nullptr) {
      return emitOpError("tcube_pip should not be defined!");
    }
  }
  if (syncBlockMode == SyncBlockMode::ALL_CUBE) {
    if (getTcubePipeAttr() == nullptr) {
      return emitOpError("tcube_pipe should defined!");
    }
    if (getTcubePipeAttr().getPipe() != PIPE::PIPE_FIX) {
      return emitOpError("TPipe is illegal. TPipe of ALL_CUBE is PIPE_FIX!");
    }
  }
  if (syncBlockMode == SyncBlockMode::ALL_VECTOR) {
    if (getTvectorPipeAttr() == nullptr) {
      return emitOpError("tvector_pipe should be defined!");
    }
    if (getTvectorPipeAttr().getPipe() != PIPE::PIPE_MTE3) {
      return emitOpError("TPipe is illegal. TPipe of ALL_CUBE is PIPE_MTE3!");
    }
  }
  if (syncBlockMode == SyncBlockMode::ALL) {
    if (getTcubePipeAttr() == nullptr || getTvectorPipeAttr() == nullptr) {
      return emitOpError("tvector_pipe and tcube_pipe should be defined!");
    }
    if (getTcubePipeAttr().getPipe() != PIPE::PIPE_FIX) {
      return emitOpError("Cube pipe is illegal. Cube pipe is PIPE_FIX!");
    }
    if (getTvectorPipeAttr().getPipe() != PIPE::PIPE_MTE3) {
      return emitOpError("Vector pipe is illegal. Vector pipe is PIPE_MTE3!");
    }
  }
  return success();
}
