// RUN: bishengir-opt -pass-pipeline="builtin.module( \
// RUN:       hfusion-fuse-ops, \
// RUN:       canonicalize, \
// RUN:       func.func(sharding-propagation,mesh-spmdization))" %s | FileCheck %s
mesh.mesh @net(shape = 8)
func.func @mlp(%x : tensor<65536x65536xf16>,
               %w_ff0 : tensor<65536x8192xf16>,
               %b_ff0 : tensor<65536x8192xf16>,
               %w_ff1 : tensor<8192x256xf16>,
               %b_ff1: tensor<65536x256xf16>) -> tensor<65536x256xf16> attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {

  %shard = mesh.shard %x to <@net, [[0]]> : tensor<65536x65536xf16>
  %0 = tensor.empty() : tensor<65536x8192xf16>
  %mm0 = linalg.matmul ins(%shard, %w_ff0 : tensor<65536x65536xf16>, tensor<65536x8192xf16>) outs(%0 : tensor<65536x8192xf16>) -> tensor<65536x8192xf16>
  %1 = tensor.empty() : tensor<65536x8192xf16>
  %ff0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%mm0, %b_ff0 : tensor<65536x8192xf16>, tensor<65536x8192xf16>) outs(%1 : tensor<65536x8192xf16>) -> tensor<65536x8192xf16>
  %2 = tensor.empty() : tensor<65536x8192xf16>
  //CHECK:  %[[RELU:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%[[FF0:.*]] : tensor<8192x8192xf16>) outs(%[[two:.*]] : tensor<8192x8192xf16>) -> tensor<8192x8192xf16>
  %relu = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%ff0: tensor<65536x8192xf16>) outs(%2 : tensor<65536x8192xf16>) -> tensor<65536x8192xf16>
  %4 = tensor.empty() : tensor<65536x256xf16>
  %mm1 = linalg.matmul ins(%relu, %w_ff1 : tensor<65536x8192xf16>, tensor<8192x256xf16>) outs(%4 : tensor<65536x256xf16>) -> tensor<65536x256xf16>
  %5 = tensor.empty() : tensor<65536x256xf16>
  %ff1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%mm1, %b_ff1: tensor<65536x256xf16>, tensor<65536x256xf16>) outs(%5:tensor<65536x256xf16>) -> tensor<65536x256xf16>

  return %ff1 : tensor<65536x256xf16>
}
