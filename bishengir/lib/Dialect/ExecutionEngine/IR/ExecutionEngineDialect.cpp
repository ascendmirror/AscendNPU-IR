//===- ExecutionEngineDialect.cpp - Implementation of Execution Engine dialect
//                                  and types -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/ExecutionEngine/IR/ExecutionEngine.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "bishengir/Dialect/ExecutionEngine/IR/ExecutionEngineEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/ExecutionEngine/IR/ExecutionEngineAttrs.cpp.inc"

using namespace mlir;

void mlir::execution_engine::ExecutionEngineDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "bishengir/Dialect/ExecutionEngine/IR/ExecutionEngineAttrs.cpp.inc"
      >();
}

#include "bishengir/Dialect/ExecutionEngine/IR/ExecutionEngineBaseDialect.cpp.inc"

LogicalResult mlir::execution_engine::ArgTypeAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    execution_engine::ArgType argType,
    std::optional<hivm::AddressSpace> memPoolSpace) {
  if (argType != execution_engine::ArgType::MemPoolSpace && memPoolSpace)
    return emitError() << "cannot have a memory pool space for `"
                       << execution_engine::stringifyArgType(argType)
                       << "` argument type";
  if (argType == execution_engine::ArgType::MemPoolSpace && !memPoolSpace)
    return emitError() << "must have a memory pool space";
  return success();
}

execution_engine::ArgTypeAttr mlir::execution_engine::ArgTypeAttr::get(
    MLIRContext *context, execution_engine::ArgType nonMemPoolArgType) {
  return get(context, nonMemPoolArgType, std::nullopt);
}

execution_engine::ArgTypeAttr
mlir::execution_engine::ArgTypeAttr::get(MLIRContext *context,
                                         hivm::AddressSpace memPoolSpace) {
  return get(context, execution_engine::ArgType::MemPoolSpace, memPoolSpace);
}

Attribute mlir::execution_engine::ArgTypeAttr::parse(AsmParser &odsParser,
                                                     Type odsType) {
  if (failed(odsParser.parseLess()))
    return {};
  const auto argType = FieldParser<execution_engine::ArgType>::parse(odsParser);
  if (failed(argType))
    return {};
  if (*argType != execution_engine::ArgType::MemPoolSpace) {
    if (failed(odsParser.parseGreater()))
      return {};
    return get(odsParser.getContext(), *argType);
  }
  if (failed(odsParser.parseComma()))
    return {};
  const auto memSpace = FieldParser<hivm::AddressSpace>::parse(odsParser);
  if (failed(memSpace) || failed(odsParser.parseGreater()))
    return {};
  return get(odsParser.getContext(), *memSpace);
}

void mlir::execution_engine::ArgTypeAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << '<' << getArgType();
  if (getMemPoolSpace())
    odsPrinter << ", " << *getMemPoolSpace();
  odsPrinter << '>';
}
