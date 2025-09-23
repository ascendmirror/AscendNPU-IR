//===- GetUnitFlagModeCondition.cpp - get unit flag mode condition impls --===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

using namespace mlir;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// BatchMmadL1Op
//===----------------------------------------------------------------------===//

std::optional<Value> BatchMmadL1Op::getUnitFlagModeCondition() {
  return getUnitFlagCond();
}

std::optional<UnitFlagAttr> BatchMmadL1Op::getUnitFlagModeValue() {
  return getUnitFlagMode();
}

//===----------------------------------------------------------------------===//
// FixpipeOp
//===----------------------------------------------------------------------===//

std::optional<Value> FixpipeOp::getUnitFlagModeCondition() {
  return getUnitFlagCond();
}

std::optional<UnitFlagAttr> FixpipeOp::getUnitFlagModeValue() {
  return getUnitFlagMode();
}

//===----------------------------------------------------------------------===//
// MmadL1Op
//===----------------------------------------------------------------------===//

std::optional<Value> MmadL1Op::getUnitFlagModeCondition() {
  return getUnitFlagCond();
}

std::optional<UnitFlagAttr> MmadL1Op::getUnitFlagModeValue() {
  return getUnitFlagMode();
}
