//===- bishengir-cpu-runner.cpp - MLIR CPU Execution Driver----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry point to a command line utility that executes an MLIR file on the
// CPU by  translating MLIR to LLVM IR before JIT-compiling and executing the
// latter.
//
//===----------------------------------------------------------------------===//

#include "bishengir/InitAllDialects.h"
#include "bishengir/InitAllExtensions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#define DEBUG_TYPE "bishengir-cpu-runner"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static llvm::cl::opt<std::string> executionInputOpt{
    "execution-input",
    llvm::cl::desc("File path to which the generated input is dumped.")};

static llvm::cl::opt<std::string> executionOutputOpt{
    "execution-output",
    llvm::cl::desc("File path to which the generated output is dumped.")};

static llvm::cl::list<std::size_t> executionArgumentsOpt{
    "execution-arguments",
    llvm::cl::desc("Arguments to be passed the entry point function."),
    llvm::cl::CommaSeparated};

static void replaceArgsWithConstants(mlir::RewriterBase &rewriter,
                                     mlir::LLVM::LLVMFuncOp entryPointOp) {
  using namespace mlir;
  const auto loc = entryPointOp.getLoc();
  auto &entryPointBody = entryPointOp.getFunctionBody();

  auto createConstStr = [&](const std::string &name,
                            llvm::SmallString<50> str) -> Value {
    str.push_back('\0');
    auto globalStr = rewriter.create<LLVM::GlobalOp>(
        loc, LLVM::LLVMArrayType::get(rewriter.getI8Type(), str.size()), true,
        LLVM::Linkage::Internal, entryPointOp.getSymName().str() + "_" + name,
        rewriter.getStringAttr(str));
    auto address = rewriter.create<LLVM::AddressOfOp>(loc, globalStr);
    rewriter.moveOpBefore(globalStr, entryPointOp);
    return address;
  };

  rewriter.replaceAllUsesWith(
      entryPointBody.getArgument(0),
      createConstStr("execution_input", StringRef(executionInputOpt)));

  rewriter.replaceAllUsesWith(
      entryPointBody.getArgument(1),
      createConstStr("execution_output", StringRef(executionOutputOpt)));

  for (auto argument : entryPointBody.getArguments().drop_front(2)) {
    auto constantOp = rewriter.create<LLVM::ConstantOp>(
        loc, argument.getType(),
        executionArgumentsOpt[argument.getArgNumber() - 2]);
    rewriter.replaceAllUsesWith(argument, constantOp);
  }
}

static mlir::LogicalResult
replaceEntryPointArgs(mlir::Operation *op,
                      const mlir::JitRunnerOptions &options) {
  using namespace mlir;
  auto moduleOp = cast<ModuleOp>(op);
  auto *entryPointOp =
      SymbolTable::lookupSymbolIn(moduleOp, options.mainFuncName);
  if (!entryPointOp)
    return moduleOp.emitError("Couldn't find the entry point function \"")
           << options.mainFuncName << '"';

  auto entryPointFuncOp = cast<LLVM::LLVMFuncOp>(entryPointOp);
  auto &entryPointBody = entryPointFuncOp.getFunctionBody();

  if (entryPointBody.getNumArguments() < 2)
    return entryPointFuncOp.emitError("The entry point function \"")
           << entryPointFuncOp.getSymName()
           << "\" should have at least 2 arguments, but found "
           << entryPointBody.getNumArguments() << '!';

  if (const auto dynamicDims = entryPointBody.getNumArguments() - 2;
      dynamicDims != executionArgumentsOpt.size())
    return entryPointFuncOp.emitError("Expected ")
           << dynamicDims << " dynamic dimensions, but received "
           << executionArgumentsOpt.size() << '!';

  IRRewriter rewriter(op->getContext());
  rewriter.setInsertionPointToStart(&entryPointBody.front());

  replaceArgsWithConstants(rewriter, entryPointFuncOp);

  // FIX: there seems to be a bug in LLVM; because using this should be enough
  // but it fails:
  /**
    entryPointFuncOp.eraseArguments(
        llvm::BitVector(entryPointBody.getNumArguments(), true));
   */
  entryPointFuncOp.setFunctionType(LLVM::LLVMFunctionType::get(
      entryPointFuncOp.getFunctionType().getReturnType(), {}, false));
  entryPointFuncOp.removeArgAttrsAttr();
  while (entryPointBody.getNumArguments() > 0)
    entryPointBody.eraseArgument(entryPointBody.getNumArguments() - 1);

  LDBG("Module after transformation:\n" << moduleOp);

  return verify(moduleOp);
}

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::registerAllDialects(registry);
  bishengir::registerAllDialects(registry);
  bishengir::registerAllExtensions(registry);

  mlir::JitRunnerConfig config;
  config.mlirTransformer = replaceEntryPointArgs;

  return mlir::JitRunnerMain(argc, argv, registry, config);
}
