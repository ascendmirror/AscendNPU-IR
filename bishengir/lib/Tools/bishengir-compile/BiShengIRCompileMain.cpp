//===- BiShengIRCompile.cpp - BiShengIR Compile Tool Support -----*- C++-*-===//
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

#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/PassPipeline.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"

#define DEBUG_TYPE "bishengir-compile"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace bishengir;
using namespace llvm;
using namespace mlir;

namespace {

/// Get the BiShengIR HIVM Compiler binary name.
StringRef getBiShengIRHIVMCompilerName() {
  const char *kBiShengIRHIVMBinaryName = "bishengir-hivm-compile";
  return kBiShengIRHIVMBinaryName;
}

/// Calls `bishengir-hivm-compile` to run HIVM optimization passes.
FailureOr<OwningModuleRef> runExternalHIVMOptimizationPipeline(
    ModuleOp module, const bishengir::BiShengIRCompileMainConfig &config) {
  TempDirectoriesStore tempDirsStore;
  std::string inputFile = "module.hivm.mlir";
  std::string outputFile = "module.hivm.opt.mlir";
  auto inputFileHandler = getTempFile(inputFile, tempDirsStore);
  auto outputFileHandler = getTempFile(outputFile, tempDirsStore);
  if (!inputFileHandler || !outputFileHandler) {
    llvm::dbgs()
        << "[ERROR] Failed to create temporary input/output files needed "
           "to run hivm pipeline.\n";
    return failure();
  }

  inputFile = inputFileHandler->outputFilename();
  outputFile = outputFileHandler->outputFilename();

  module.print(inputFileHandler->os(),
               mlir::OpPrintingFlags().enableDebugInfo(
                   config.getEnableSanitizer() || config.getEnableDebugInfo()));
  inputFileHandler->os().flush();

  std::vector<std::string> arguments;
  arguments.emplace_back("");
  arguments.push_back(inputFile);

  auto hivmCompileArgs = config.getHIVMCompileArgsDashDash();
  arguments.insert(arguments.end(), hivmCompileArgs.begin(),
                   hivmCompileArgs.end());

  arguments.emplace_back("-o");
  arguments.push_back(outputFile);
  std::string enableHIVMOpt = "--enable-hivm-compile=";
  enableHIVMOpt += config.getEnableHIVMCompile() ? "true" : "false";
  arguments.emplace_back(enableHIVMOpt);
  arguments.emplace_back("--convert-hir-to-lir=false");

  SmallVector<StringRef> argumentsRef(arguments.begin(), arguments.end());
  if (failed(execute(getBiShengIRHIVMCompilerName(), getBiShengInstallPath(),
                     argumentsRef))) {
    return failure();
  }

  std::string errorMessage;
  auto file = mlir::openInputFile(outputFile, &errorMessage);
  if (!file) {
    llvm::errs() << "[ERROR] Failed to open: " << outputFile
                 << " error message: " << errorMessage << '\n';
    return failure();
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), mlir::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> moduleRef =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, module->getContext());
  if (!moduleRef) {
    llvm::errs() << "[ERROR] Failed to open: " << outputFile << '\n';
    return failure();
  }
  module.erase();
  auto parsedModule = moduleRef.release();
  return OwningModuleRef(parsedModule);
}

/// Calls `bishengir-hivm-compile` to lower HIVM IR with minimum op set.
LogicalResult runExternalHIVMLoweringPipeline(
    ModuleOp module, const bishengir::BiShengIRCompileMainConfig &config) {
  TempDirectoriesStore tempDirsStore;
  std::string inputFile = "module.hivm.opt.mlir";
  std::string outputFile = config.getOutputFile();
  auto inputFileHandler = getTempFile(inputFile, tempDirsStore);
  if (!inputFileHandler) {
    llvm::dbgs()
        << "[ERROR] Failed to create temporary input file needed to run "
           "hivm compile.\n";
    return failure();
  }
  inputFile = inputFileHandler->outputFilename();

  module.print(inputFileHandler->os(),
               mlir::OpPrintingFlags().enableDebugInfo(
                   config.getEnableSanitizer() || config.getEnableDebugInfo()));
  inputFileHandler->os().flush();

  std::vector<std::string> arguments;
  arguments.emplace_back("");
  arguments.push_back(inputFile);

  auto hivmCompileArgs = config.getHIVMCompileArgsDashDash();
  arguments.insert(arguments.end(), hivmCompileArgs.begin(),
                   hivmCompileArgs.end());

  arguments.emplace_back("-o");
  arguments.push_back(outputFile);
  arguments.emplace_back("--enable-hivm-compile=false");
  arguments.emplace_back("--convert-hir-to-lir=true");

  SmallVector<StringRef> argumentsRef(arguments.begin(), arguments.end());
  if (failed(execute(getBiShengIRHIVMCompilerName(), getBiShengInstallPath(),
                     argumentsRef))) {
    return failure();
  }

  return success();
}

} // namespace

FailureOr<OwningModuleRef>
bishengir::runBiShengIRPipeline(ModuleOp mod,
                                BiShengIRCompileMainConfig config) {
  MLIRContext *ctx = mod->getContext();
  mlir::DiagnosticEngine &diagEngine = ctx->getDiagEngine();
  std::vector<Diagnostic> collectedDiagnostics;
  // Collect diagnostics and emit them afterwards because we have tuning
  // mechanism.
  auto handlerID = diagEngine.registerHandler([&](Diagnostic &diag) {
    collectedDiagnostics.emplace_back(std::move(diag));
  });

  bool hirCompileSuccess = false;
  int tryTimes = config.getEnableTuningMode() ? 1 : 5;
  for (int i = 0; i < tryTimes; i++) {
    LDBG("Attempt number: " << i << " with max buffer count tuning delta: "
                            << config.getHfusionMaxBufferCountTuning());

    ModuleOp hirCompileModule = cast<ModuleOp>(mod->clone());
    auto buildPipeline =
        std::bind(buildBiShengHIRPipeline, std::placeholders::_1, config);
    bool step1 = succeeded(
        runPipeline(hirCompileModule, buildPipeline, config, "BiShengHIR"));
    bool step2 = false;
    if (step1) {
      auto runExternalResult =
          runExternalHIVMOptimizationPipeline(hirCompileModule, config);
      if (!failed(runExternalResult)) {
        hirCompileModule = runExternalResult->release();
        step2 = true;
      }
    }
    if (step1 && step2) {
      hirCompileSuccess = true;
      mod.erase();
      mod = hirCompileModule;
      break;
    }
    hirCompileModule.erase();

    // increase max buffers by 2 in HFusion auto schedule
    config.increaseMaxBufferCountTuning(2);
  }

  // Restore to the default handler.
  diagEngine.eraseHandler(handlerID);

  if (!hirCompileSuccess) {
    for (auto &diag : llvm::reverse(collectedDiagnostics)) {
      diagEngine.emit(std::move(diag));
    }
    return failure();
  }

  if (config.shouldEnableCPURunner()) {
    std::string errorMessage;
    auto fileHandle =
        mlir::openOutputFile(config.getOutputFile(), &errorMessage);
    if (!fileHandle) {
      llvm::errs() << "[ERROR] " << errorMessage << "\n";
      return failure();
    }
    fileHandle->os() << mod << '\n';
    fileHandle->keep();
    return OwningModuleRef(mod);
  }

  // After the above procedure, we should have lower util the minimum HIVM op
  // set. So set -enable-hivm-compile to false.
  config.setEnableHIVMCompile(false);
  auto res = runExternalHIVMLoweringPipeline(mod, config);
  if (res.failed())
    mod.emitWarning(
        "External bishengir-hivm-compile run fails, returning module before "
        "running external compiler");

  return OwningModuleRef(mod);
}
