//===-- Mesh.cpp - C Interface for Mesh dialect -------------*- C -*-===//
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

#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Mesh/IR/MeshDialect.h"
#include "mlir/Pass/Pass.h"

#include "bishengir-c/Dialect/Mesh.h"
#include "bishengir/Dialect/Mesh/Transforms/Passes.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Mesh, mesh, mlir::mesh::MeshDialect)

#include "bishengir/Dialect/Mesh/Transforms/Passes.capi.h.inc"

using namespace mlir;
using namespace mlir::mesh;

#ifdef __cplusplus
extern "C" {
#endif

#include "bishengir/Dialect/Mesh/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
