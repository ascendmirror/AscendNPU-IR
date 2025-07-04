//===- InitAllDialects.h - BiShengIR Dialects Registration ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all bishengir
// related dialects to the system.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_INITALLDIALECTS_H
#define BISHENGIR_INITALLDIALECTS_H

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

namespace bishengir {

/// Add all the bishengir-specific dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<mlir::annotation::AnnotationDialect,
                  mlir::hacc::HACCDialect,
                  mlir::hfusion::HFusionDialect,
                  mlir::hivm::HIVMDialect>();
  // clang-format on
}

/// Append all the bishengir-specific dialects to the registry contained in the
/// given context.
inline void registerAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  bishengir::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace bishengir

#endif // BISHENGIR_INITALLDIALECTS_H
