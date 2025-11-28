//===- Passes.td - Scope pass definition file --------------*- tablegen -*-===//
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

#ifndef BISHENGIR_DIALECT_SCOPE_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_SCOPE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def OutlineScope : Pass<"outline-scope", "mlir::ModuleOp"> {
  let summary = "Outline scope region within ScopeOp.";
  let description = [{
    Propagate scopes bound for argument by replacing the usage of 
    `tensor.dim`, to make the relationships between tensor dynamic 
    dims more explicit.

    Example:

    ```mlir
    %dim0 = tensor.dim %arg0, %c0 : tensor<?x640xf16>
    %out0 = tensor.empty(%dim0) : tensor<?x640xf16>
    %add0 = linalg.elemwise_binary ins(%arg0, %arg0) outs(%out0)
    %concat = tensor.concat dim(0) %add0, %add0 : tensor<?x640xf16>
    %dim1 = tensor.dim %concat, %c0 : tensor<?x640xf16>
    %out1 = tensor.empty(%dim1) : tensor<?x640xf16>
    %add1 = linalg.elemwise_binary ins(%concat, %concat) outs(%out1)
    ```

    will be transformed into:

    ```mlir
    %S0 = scope.scopeic_int @S0 : index
    %S1 = scope.scopeic_int @S1 [%S0, %S0], affine_map<()[s0, s1] -> (s0 + s1)>
    scope.bind_scopeic_shape %arg0, [%S0]
    
    %out0 = tensor.empty(%S0) : tensor<?x640xf16>
    scope.bind_scopeic_shape %0, [%S0]
    %add0 = linalg.elemwise_binary ins(%arg0, %arg0) outs(%out0)
    scope.bind_scopeic_shape %add0, [%S0]

    // result dim for concat will be reified to an `affine.apply` op.
    // this pass keeps the affine map in `%S1`
    %concat = tensor.concat dim(0) %add0, %add0 : tensor<?x640xf16>
    scope.bind_scopeic_shape %concat, [%S1]

    %out1 = tensor.empty(%S1) : tensor<?x640xf16>
    scope.bind_scopeic_shape %out1, [%S1]
    %add1 = linalg.elemwise_binary ins(%concat, %concat) outs(%out1)
    scope.bind_scopeic_shape %add1, [%S1]
    ```
  }];
  let constructor = "mlir::scope::createOutlineScopePass()";
  let dependentDialects = [];
}

#endif // BISHENGIR_DIALECT_SCOPE_TRANSFORMS_PASSES_TD
