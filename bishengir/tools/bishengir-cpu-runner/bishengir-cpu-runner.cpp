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

using MemoryPoolData = mlir::DenseMap<mlir::hivm::AddressSpace, std::size_t>;

struct MemoryPoolParser : public llvm::cl::parser<MemoryPoolData> {
private:
  static std::optional<std::size_t>
  parseSuffixedSizeImpl(llvm::StringRef sizeStr,
                        llvm::function_ref<void(const std::string &)> errorFn) {
    std::ptrdiff_t size;
    if (sizeStr.consumeInteger(10, size) || size < 0) {
      errorFn("expected a valid space size, but found \"" + sizeStr.str() +
              "\"");
      return std::nullopt;
    }
    constexpr const unsigned invalidShift = -1;
    const auto shift = llvm::StringSwitch<unsigned>(sizeStr)
                           .Cases("", "B", 0)
                           .Cases("k", "K", "kB", "KB", 10)
                           .Cases("M", "MB", 20)
                           .Cases("G", "GB", 30)
                           .Cases("T", "TB", 40)
                           .Default(invalidShift);
    if (shift == invalidShift) {
      errorFn("unknown suffix: " + sizeStr.str());
      return std::nullopt;
    }
    const auto spaceSize = std::size_t(size) << shift;
    if (spaceSize < (std::size_t)size) {
      errorFn("memory pool size is too big");
      return std::nullopt;
    }
    return spaceSize;
  }

public:
  explicit MemoryPoolParser(llvm::cl::Option &o)
      : llvm::cl::parser<parser_data_type>(o) {}

  void printOptionInfo(const llvm::cl::Option &opt,
                       size_t globalWidth) const final {
    const auto helpMsg =
        "  --" + opt.ArgStr.str() + "=<size>@<address-space>[,...]";
    llvm::outs() << helpMsg;
    opt.printHelpStr(opt.HelpStr, globalWidth, helpMsg.size() + 3);
  }

  static std::size_t parseSuffixedSize(llvm::StringRef sizeStr) {
    return *parseSuffixedSizeImpl(sizeStr, [&sizeStr](
                                               const std::string &message) {
      llvm::report_fatal_error("Cannot parse \"" + sizeStr + "\": " + message);
    });
  }

  // Return true on error.
  static bool parse(llvm::cl::Option &opt, llvm::StringRef argName,
                    llvm::StringRef arg, parser_data_type &value) {
    mlir::SmallVector<mlir::StringRef> args;
    arg.split(args, ',', -1, false);

    bool isFailure = false;
    for (auto arg : args) {
      mlir::SmallVector<mlir::StringRef> info;
      arg.split(info, '@', -1, false);
      if (info.size() != 2) {
        isFailure = opt.error(
            "in " + arg + ": need to specify the option in the correct format");
        continue;
      }

      const auto addressSpaceStr = info.back();
      const auto addressSpace =
          mlir::hivm::symbolizeAddressSpace(addressSpaceStr);
      if (!addressSpace) {
        isFailure = true;
        (void)opt.error("in " + arg +
                        ": expected a valid HIVM address space, but found \"" +
                        addressSpaceStr + "\"");
        continue;
      }

      const auto spaceSizeStr = info.front();
      const auto spaceSize = parseSuffixedSizeImpl(
          spaceSizeStr, [&opt, &arg](const std::string &message) {
            (void)opt.error("in " + arg +
                            ": expected a valid memory size: " + message);
          });
      if (!spaceSize) {
        isFailure = true;
        continue;
      }

      value[*addressSpace] = *spaceSize;
    }

    return isFailure;
  }
};

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

static llvm::cl::opt<MemoryPoolData, false, MemoryPoolParser>
    executionMemoryPoolOpt{
        "execution-mem-pool",
        llvm::cl::desc("Memory pool sizes in bytes to be used for direct "
                       "memory access in different address spaces."),
        llvm::cl::init(
            MemoryPoolData({{mlir::hivm::AddressSpace::GM,
                             MemoryPoolParser::parseSuffixedSize("100GB")},
                            {mlir::hivm::AddressSpace::UB,
                             MemoryPoolParser::parseSuffixedSize("100KB")}}))};

static mlir::LogicalResult
replaceArgsWithConstants(mlir::RewriterBase &rewriter,
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

  auto execArgOpt = executionArgumentsOpt.begin();
  for (auto [arg, attrs] :
       llvm::zip_equal(entryPointBody.getArguments(),
                       entryPointOp.getAllArgAttrs().getValue())) {
    const auto dictAttrs = cast<DictionaryAttr>(attrs);
    const auto argType = dictAttrs.getAs<execution_engine::ArgTypeAttr>(
        execution_engine::ArgTypeAttr::name);
    Value val;
    switch (argType.getArgType()) {
    case execution_engine::ArgType::ArgDynSize:
      if (execArgOpt == executionArgumentsOpt.end())
        return emitError(arg.getLoc(),
                         "need a dynamic size argument to be passed");
      val =
          rewriter.create<LLVM::ConstantOp>(loc, arg.getType(), *execArgOpt++);
      break;
    case execution_engine::ArgType::InputFilePath:
      val = createConstStr("execution_input", StringRef(executionInputOpt));
      break;
    case execution_engine::ArgType::OutputFilePath:
      val = createConstStr("execution_output", StringRef(executionOutputOpt));
      break;
    case execution_engine::ArgType::MemPoolSpace:
      val = rewriter.create<LLVM::ConstantOp>(
          loc, arg.getType(),
          executionMemoryPoolOpt[*argType.getMemPoolSpace()]);
      break;
    }
    rewriter.replaceAllUsesWith(arg, val);
  }

  if (execArgOpt != executionArgumentsOpt.end())
    return entryPointOp.emitError()
           << "needs less dynamic size arguments to be passed, expected "
           << execArgOpt - executionArgumentsOpt.begin() << " but found "
           << executionArgumentsOpt.size();

  return success();
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

  IRRewriter rewriter(op->getContext());
  rewriter.setInsertionPointToStart(&entryPointBody.front());

  if (failed(replaceArgsWithConstants(rewriter, entryPointFuncOp)))
    return failure();

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
