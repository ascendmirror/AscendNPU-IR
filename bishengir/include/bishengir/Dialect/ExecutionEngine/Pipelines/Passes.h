//===- Passes.h - ExecutionEngine pipeline entry points ---------*- C++ -*-===//
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
// This header file defines prototypes of all Execution Engine pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_EXECUTION_ENGINE_PIPELINES_PASSES_H
#define BISHENGIR_DIALECT_EXECUTION_ENGINE_PIPELINES_PASSES_H

#include "mlir/Pass/PassOptions.h"

namespace mlir::execution_engine {

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

struct CPURunnerPipelineOptions
    : public PassPipelineOptions<CPURunnerPipelineOptions> {
  CPURunnerPipelineOptions() = default;

  CPURunnerPipelineOptions(const CPURunnerPipelineOptions &other)
      : CPURunnerPipelineOptions() {
    *this = other;
  }

  CPURunnerPipelineOptions &operator=(const CPURunnerPipelineOptions &other) {
    this->copyOptionValuesFrom(other);
    return *this;
  }

  PassOptions::Option<std::string> wrapperName{
      *this, "wrapper-name",
      ::llvm::cl::desc("Name of the wrapper function to be generated for the "
                       "single host entry function provided."),
      ::llvm::cl::init("main")};
};

/// Create a pipeline to lower everything to be compatible with the CPU runner.
void buildCPURunnerPipeline(OpPassManager &pm,
                            const CPURunnerPipelineOptions &options = {});

/// Register all Execution Engine related pipelines.
void registerAllPipelines();

} // namespace mlir::execution_engine

#endif // BISHENGIR_DIALECT_EXECUTION_ENGINE_PIPELINES_PASSES_H
