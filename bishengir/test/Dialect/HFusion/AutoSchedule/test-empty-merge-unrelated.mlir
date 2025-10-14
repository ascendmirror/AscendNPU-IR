// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20" -split-input-file | FileCheck %s

// CHECK: func.func @mlir_fused_native_layer_norm_0_tiling_function(
// CHECK: scf.for
// CHECK: scf.for
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>>>} {
  func.func @mlir_fused_native_layer_norm_0(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>) -> (tensor<1024x1xf32>, tensor<1024x1xf32>, tensor<1024x1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %cst = arith.constant 9.99999974E-6 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.024000e+03 : f32
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %2 = tensor.empty() : tensor<1024x1024xf32>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<1024xf32>) outs(%2 : tensor<1024x1024xf32>) dimensions = [0]
    %broadcasted_3 = linalg.broadcast ins(%arg2 : tensor<1024xf32>) outs(%2 : tensor<1024x1024xf32>) dimensions = [0]
    %reduced = linalg.reduce ins(%arg0 : tensor<1024x1024xf32>) outs(%1 : tensor<1024xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %13 = arith.addf %in, %init : f32
        linalg.yield %13 : f32
      }
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced, %cst_2 : tensor<1024xf32>, f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %broadcasted_4 = linalg.broadcast ins(%3 : tensor<1024xf32>) outs(%2 : tensor<1024x1024xf32>) dimensions = [1]
    %expanded = tensor.expand_shape %3 [[0, 1]] output_shape [1024, 1] : tensor<1024xf32> into tensor<1024x1xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %broadcasted_4 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %4 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %reduced_5 = linalg.reduce ins(%5 : tensor<1024x1024xf32>) outs(%1 : tensor<1024xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %13 = arith.addf %in, %init : f32
        linalg.yield %13 : f32
      }
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced_5, %cst_2 : tensor<1024xf32>, f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%6, %cst : tensor<1024xf32>, f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %8 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%7 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %9 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%8 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %broadcasted_6 = linalg.broadcast ins(%9 : tensor<1024xf32>) outs(%2 : tensor<1024x1024xf32>) dimensions = [1]
    %expanded_7 = tensor.expand_shape %9 [[0, 1]] output_shape [1024, 1] : tensor<1024xf32> into tensor<1024x1xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %broadcasted_6 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %broadcasted : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%11, %broadcasted_3 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %expanded, %expanded_7, %12 : tensor<1024x1xf32>, tensor<1024x1xf32>, tensor<1024x1024xf32>
  }
}
