//===- bishengir-opt.cpp - BiShengIR Optimizer Driver -----------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Main entry function for bishengir-opt built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "bishengir/InitAllDialects.h"
#include "bishengir/InitAllExtensions.h"
#include "bishengir/InitAllPasses.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/InitLLVM.h"

#ifdef MLIR_INCLUDE_TESTS
#include "Test/InitTestDialect.h"
#include "Test/TestPasses.h"
#endif

namespace mlir {
namespace test {
void registerTestTransformDialectEraseSchedulePass();
} // namespace test
} // namespace mlir

namespace test {
void registerTestDialect(::mlir::DialectRegistry &registry);
void registerTestTransformDialectExtension(::mlir::DialectRegistry &registry);
} // namespace test

int main(int argc, char **argv) {
  // Register dialects.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  bishengir::registerAllDialects(registry);

  // Register passes.
  mlir::registerAllPasses();
  bishengir::registerAllPasses();

  // Register dialect extensions.
  mlir::registerAllExtensions(registry);
  bishengir::registerAllExtensions(registry);

#ifdef MLIR_INCLUDE_TESTS
  ::bishengir_test::registerTestDialect(registry);
  ::bishengir_test::registerAllTestPasses();
  ::mlir::test::registerTestTransformDialectEraseSchedulePass();
  ::test::registerTestDialect(registry);
  ::test::registerTestTransformDialectExtension(registry);
#endif

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "BiShengIR optimizer driver\n", registry));
}
