//===- HMAPBase.td - Base of hybrid mesh aware parallelism dialect ---------==//
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
#ifndef HMAP_IR_HMAPBASE
#define HMAP_IR_HMAPBASE

include "mlir/IR/OpBase.td"

def HMAP_Dialect : Dialect {
  let name = "hmap";

  let cppNamespace = "::mlir::hmap";

  let description = [{
    HMAP (Hybrid Mesh Aware Parallelism) dialect.
  }];

  let dependentDialects = [
    "mesh::MeshDialect",
  ];
}

#endif // HMAP_IR_HMAPBASE
