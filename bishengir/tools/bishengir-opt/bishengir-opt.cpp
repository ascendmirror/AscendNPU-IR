//===- bishengir-opt.cpp - BiShengIR Optimizer Driver -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for bishengir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "bishengir/InitAllDialects.h"

#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  // Register dialects.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  bishengir::registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "BiShengIR optimizer driver\n", registry));
}
