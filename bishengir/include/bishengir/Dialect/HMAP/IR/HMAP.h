//===- HMAP.h - Hybrid Mesh Aware Parallelism dialect ------------*- C++-*-===//
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

#ifndef BISHENGIR_DIALECT_HMAP_IR_HMAP_H
#define BISHENGIR_DIALECT_HMAP_IR_HMAP_H

#include "mlir/Dialect/Mesh/IR/MeshDialect.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/IR/Dialect.h"

//===----------------------------------------------------------------------===//
// HMAP Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HMAP/IR/HMAPOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// HMAP Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/HMAP/IR/HMAPOps.h.inc"

#endif // BISHENGIR_DIALECT_HMAP_IR_HMAP_H
