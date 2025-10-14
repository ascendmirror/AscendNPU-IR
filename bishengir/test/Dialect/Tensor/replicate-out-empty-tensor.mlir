// RUN: bishengir-opt -replicate-out-empty-tensor %s | FileCheck %s

func.func @test(%arg0: tensor<8xf32>, %arg1 : tensor<8xf32>, %arg2 : tensor<8xf32>) -> tensor<8xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>}
{
  // CHECK: tensor.empty
  %1 = tensor.empty() : tensor<8xf32>
  %2 = linalg.elemwise_binary { mul, fun = #linalg.binary_fn<mul> } ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
  // CHECK: tensor.empty
  %4 = linalg.elemwise_binary { add, fun = #linalg.binary_fn<add> } ins(%2, %arg2 : tensor<8xf32>, tensor<8xf32>) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
  return %4 : tensor<8xf32>
}