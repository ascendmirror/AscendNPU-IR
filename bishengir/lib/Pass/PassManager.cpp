//===- PassManager.cpp - Pass Management Interface --------------*- C++ -*-===//
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

#include "bishengir/Pass/PassManager.h"
#include "bishengir/Tools/BiShengIRConfigBase/Config.h"

#if MLIR_ENABLE_EXECUTION_ENGINE
#include "bishengir/ExecutionEngine/Passes.h"
#include "bishengir/Pass/CPURunnerMetadata.h"
#include "bishengir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/ScopedPrinter.h"

#define DEBUG_TYPE "bishengir-pass-manager"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBGSNL() LLVM_DEBUG(llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace bishengir;

namespace bishengir {

template <bool includePassInfo>
void CPURunnerMetadataParser<includePassInfo>::printOptionInfo(
    const llvm::cl::Option &opt, size_t globalWidth) const {
  auto helpMsg = "  --" + llvm::to_string(opt.ArgStr) + "=";

  if constexpr (includePassInfo)
    helpMsg += "<pass>[,<index>][,<options>]";
  else
    helpMsg += "[<options>]";

  llvm::outs() << helpMsg;
  opt.printHelpStr(opt.HelpStr, globalWidth, helpMsg.size() + 3);
  execution_engine::CPURunnerPipelineOptions().printHelp(2, globalWidth);
}

template <bool includePassInfo>
bool CPURunnerMetadataParser<includePassInfo>::parse(llvm::cl::Option &opt,
                                                     StringRef argName,
                                                     StringRef arg,
                                                     parser_data_type &value) {
  if (opt.getNumOccurrences() > 1)
    return opt.error("Option shouldn't be used multiple times!");
  value.numOccurrences++;

  SmallVector<StringRef> args;
  arg.split(args, ',', 2, false);
  args = llvm::to_vector(llvm::reverse(args));

  if constexpr (includePassInfo) {
    if (args.empty())
      return opt.error("At least the pass name should be provided!");

    if (args.back().empty() || !PassInfo::lookup(args.back()))
      return opt.error("\"" + args.back() + "\" is not a pass!");
    value.passName = args.pop_back_val();

    if (args.empty())
      return false;

    if (std::ptrdiff_t passIndex; !args.back().getAsInteger(10, passIndex)) {
      args.pop_back();
      if (passIndex <= 0)
        return opt.error(
            "Pass index should be a positive non-zero integer, but found " +
            llvm::to_string(passIndex) + "!");
      value.passIndex = static_cast<decltype(value.passIndex)>(passIndex);
    }
  }

  if (args.empty())
    return false;

  return failed(value.options.parseFromString(args.back()));
}

template struct bishengir::CPURunnerMetadataParser<true>;
template struct bishengir::CPURunnerMetadataParser<false>;
} // namespace bishengir

namespace {

static void verifyOptionUsage(const BiShengIRCompileConfigBase &config,
                              MLIRContext &ctx) {
  if (config.CPURunnerOpt().numOccurrences +
          config.CPURunnerBeforeOpt().numOccurrences +
          config.CPURunnerAfterOpt().numOccurrences >
      1)
    llvm::report_fatal_error(
        "Cannot combine any of multiple cpu-runner options.");

  if (ctx.isMultithreadingEnabled())
    llvm::report_fatal_error(
        "Cannot run the cpu-runner with multithreading enabled.");
}

static void executeCPURunnerPasses(Operation *op,
                                   const BiShengIRCompileConfigBase &config) {
  PassManager pm(op->getContext());
  execution_engine::buildCPURunnerPipeline(
      pm, (config.CPURunnerOpt().numOccurrences != 0)
              ? config.CPURunnerOpt().options
              : ((config.CPURunnerBeforeOpt().numOccurrences != 0)
                     ? config.CPURunnerBeforeOpt()
                     : config.CPURunnerAfterOpt())
                    .options);
  LDBG("Op before CPU runner:\n" << *op);
  if (failed(mlir::applyPassManagerCLOptions(pm)) || failed(pm.run(op))) {
    LDBG("Op after CPU runner failed:\n" << *op);
    llvm::report_fatal_error(
        "[CPU Runner] Failed to run the CPU runner pipeline!");
  }
}

class CPURunnerPassExecutionHandler {
public:
  CPURunnerPassExecutionHandler(
      const bishengir::BiShengIRCompileConfigBase &config) {
    using namespace std::placeholders;

    shouldStopBefore = shouldStopAfter = [](std::size_t) { return false; };

    if (const auto runnerBefore = config.CPURunnerBeforeOpt();
        runnerBefore.numOccurrences) {
      stoppingPassName = runnerBefore.passName;

      shouldStopBefore = [config = runnerBefore](std::size_t passIndex) {
        assert(passIndex <= config.passIndex);
        return config.passIndex == passIndex;
      };
    } else if (const auto runnerAfter = config.CPURunnerAfterOpt();
               runnerAfter.numOccurrences) {
      stoppingPassName = runnerAfter.passName;

      shouldStopAfter = [config = runnerAfter](std::size_t passIndex) {
        assert(passIndex <= config.passIndex);
        return config.passIndex == passIndex;
      };
    }
  }

  ~CPURunnerPassExecutionHandler() {
    if (!stoppingPassName.empty())
      llvm::report_fatal_error(("Couldn't find `" + stoppingPassName +
                                "` pass with the required index!")
                                   .c_str());
  }

  void operator()(llvm::function_ref<void()> actionFn,
                  const tracing::Action &action) {
    // If a normal pass is running, ignore nested passes
    if (isAPassRunning)
      return actionFn();
    auto *passExecution = dyn_cast<PassExecutionAction>(&action);
    // The handler only cares about pass execution
    if (!passExecution)
      return actionFn();

    auto &pass = passExecution->getPass();
    auto *op = passExecution->getOp();
    // Initialize the count of any operation to its parent's count
    if (currentLevelCounters.find(op) == currentLevelCounters.end()) {
      currentLevelCounters[op] = currentLevelCounters[nullptr];
      LDBG("Pass count of `" << stoppingPassName << "` on " << op->getName()
                             << ' ' << op << " is initialized with "
                             << currentLevelCounters[op]);
    }

    // If the execution was stopped before, don't execute anything else
    if (shouldStopBefore(currentLevelCounters[op]) ||
        shouldStopAfter(currentLevelCounters[op])) {
      LDBG("Stopping the execution of `" << pass.getArgument() << "` on "
                                         << op->getName() << ' ' << op);
      return;
    }

    // Handle `mlir::detail::OpToOpPassAdaptor`-like passes for nesting behavior
    if (pass.getArgument().empty()) {
      handleNesting(actionFn, pass, op);
      return;
    }

    // Increment the count for the pass
    if (pass.getArgument() == stoppingPassName) {
      ++currentLevelCounters[op];
      LDBG("Pass count of `" << stoppingPassName << "` on " << op->getName()
                             << ' ' << op << " is updated to "
                             << currentLevelCounters[op]);

      // Stop if the limit was reached
      if (shouldStopBefore(currentLevelCounters[op])) {
        stoppingPassName.clear();
        return;
      }
    }

    // Execute the pass normally and don't consider nested passes
    LDBG("Executing pass `" << pass.getArgument() << "` on " << op->getName()
                            << ' ' << op);
    isAPassRunning = true;
    actionFn();
    isAPassRunning = false;

    if (pass.getArgument() == stoppingPassName &&
        shouldStopAfter(currentLevelCounters[op]))
      stoppingPassName.clear();
  }

private:
  void handleNesting(llvm::function_ref<void()> actionFn, const Pass &pass,
                     Operation *op) {
    LDBG("Entering nest on " << op->getName() << ' ' << op
                             << " with a pass count of "
                             << currentLevelCounters[op]);
    PassCountPerOp nestedCounter;
    // Push the parent's count to the child's map
    nestedCounter[nullptr] = currentLevelCounters[op];
    pastLevelsCounters.push_back(std::move(currentLevelCounters));
    currentLevelCounters = std::move(nestedCounter);

    actionFn();

    // Update the count of the parent with the child's max one
    pastLevelsCounters.back()[op] =
        llvm::max_element(currentLevelCounters, [](const auto &first,
                                                   const auto &second) {
          return first.getSecond() < second.getSecond();
        })->getSecond();
    currentLevelCounters = pastLevelsCounters.pop_back_val();
    LDBG("Exiting nest on " << op->getName() << ' ' << op
                            << " with a pass count of "
                            << currentLevelCounters[op]);
  }

private:
  using PassCountPerOp = DenseMap<Operation *, std::size_t>;
  SmallVector<PassCountPerOp, 1> pastLevelsCounters;
  PassCountPerOp currentLevelCounters;

  std::function<bool(std::size_t)> shouldStopBefore, shouldStopAfter;

  std::string stoppingPassName;
  bool isAPassRunning = false;
};
} // namespace

LogicalResult bishengir::BiShengIRPassManager::run(Operation *op) {
  if (!config.shouldEnableCPURunner())
    return PassManager::run(op);

  auto &ctx = *op->getContext();
  verifyOptionUsage(config, ctx);

  ctx.registerActionHandler(CPURunnerPassExecutionHandler(config));
  if (failed(PassManager::run(op)))
    return failure();
  ctx.registerActionHandler(nullptr);

  executeCPURunnerPasses(op, config);
  return success();
}

#endif // MLIR_ENABLE_EXECUTION_ENGINE
