// RUN: bishengir-opt --canonicalize-tensor-reshape="injective-dynamic=true" -split-input-file %s | FileCheck %s --check-prefix="INJECTIVE"
// RUN: bishengir-opt --canonicalize-tensor-reshape="injective-dynamic=false" -split-input-file %s | FileCheck %s --check-prefix="NON-INJECTIVE"

// NON-INJECTIVE-LABEL:   func.func @reshape_normalize(
// NON-INJECTIVE:           %[[VAL_5:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2]] : tensor<1x?x4096xf32> into tensor<?x4096xf32>
// NON-INJECTIVE:           return %[[VAL_5]] : tensor<?x4096xf32>
// NON-INJECTIVE:         }
func.func @reshape_normalize(%arg0: tensor<1x?x4096xbf16>, %arg1: i64, %arg2: tensor<1x24576xbf16>, %arg3: i64, %arg4: tensor<1x?x4096xf32>) -> tensor<?x4096xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %from_elements = tensor.from_elements %arg1, %arg3 : tensor<2xi64>
    %reshape = tensor.reshape %arg4(%from_elements) : (tensor<1x?x4096xf32>, tensor<2xi64>) -> tensor<?x4096xf32>
    return %reshape : tensor<?x4096xf32>
}

// -----
// NON-INJECTIVE-LABEL: all_static
// NON-INJECTIVE: collapse_shape
// NON-INJECTIVE-SAME: {{\[\[}}0, 1], [2]]
func.func @all_static(%arg0: tensor<16x4x64xf16>, %arg1: tensor<2xi64>) -> tensor<64x64xf16> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
  %reshape = tensor.reshape %arg0(%arg1) : (tensor<16x4x64xf16>, tensor<2xi64>) -> tensor<64x64xf16>
  return %reshape : tensor<64x64xf16>
}

// -----
// NON-INJECTIVE-LABEL: non_inferrable_dynamic
// NON-INJECTIVE-LABEL: reshape
func.func @non_inferrable_dynamic(%arg0: tensor<16x4x64xf16>, %arg1: tensor<2xi64>) -> tensor<?x64xf16> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
  %reshape = tensor.reshape %arg0(%arg1) : (tensor<16x4x64xf16>, tensor<2xi64>) -> tensor<?x64xf16>
  return %reshape : tensor<?x64xf16>
}

// -----
// INJECTIVE-LABEL:   func.func @reshape_normalize_2(
// INJECTIVE:           %[[VAL_5:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2], [3]] : tensor<1x?x4096x?xf32> into tensor<?x4096x?xf32>
// INJECTIVE:           return %[[VAL_5]] : tensor<?x4096x?xf32>
// NON-INJECTIVE-LABEL:   func.func @reshape_normalize_2(
// NON-INJECTIVE:           reshape
// NON-INJECTIVE:           return
func.func @reshape_normalize_2(%arg0: tensor<1x?x4096x?xbf16>, %arg1: i64, %arg2: tensor<1x24576xbf16>, %arg3: i64, %arg4: tensor<1x?x4096x?xf32>, %arg5: i64) -> tensor<?x4096x?xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %from_elements = tensor.from_elements %arg1, %arg3, %arg5 : tensor<3xi64>
    %reshape = tensor.reshape %arg4(%from_elements) : (tensor<1x?x4096x?xf32>, tensor<3xi64>) -> tensor<?x4096x?xf32>
    return %reshape : tensor<?x4096x?xf32>
}