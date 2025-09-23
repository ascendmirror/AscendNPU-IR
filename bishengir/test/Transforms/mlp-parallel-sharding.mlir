// RUN: bishengir-opt -pass-pipeline='builtin.module(canonicalize, \
// RUN: func.func(sharding-propagation,mesh-spmdization),canonicalize)' %s | FileCheck %s

// First dim: 64K / 8
#map = affine_map<(d0, d1) -> (d0, d1)>
mesh.mesh @net(shape = 8)

// CHECK: @mlp([[X:%[a-zA-Z0-9]+]]: tensor<8192x65536xf16>
func.func @mlp(%x : tensor<65536x65536xf16>,
               %w_ff0 : tensor<65536x8192xf16>,
               %b_ff0 : tensor<65536x8192xf16>,
               %w_ff1 : tensor<8192x256xf16>,
               %b_ff1: tensor<65536x256xf16>) -> tensor<65536x256xf16> {

  %shard = mesh.shard %x to <@net, [[0]]> : tensor<65536x65536xf16>
  %0 = tensor.empty() : tensor<65536x8192xf16>
// CHECK: ins([[X]]
  %mm0 = linalg.matmul ins(%shard, %w_ff0 : tensor<65536x65536xf16>, tensor<65536x8192xf16>) outs(%0 : tensor<65536x8192xf16>) -> tensor<65536x8192xf16>
  %1 = tensor.empty() : tensor<65536x8192xf16>
  %ff0 = linalg.add ins(%mm0, %b_ff0 : tensor<65536x8192xf16>, tensor<65536x8192xf16>) outs(%1 : tensor<65536x8192xf16>) -> tensor<65536x8192xf16>
  %2 = tensor.empty() : tensor<65536x8192xf16>
  %relu = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%ff0 : tensor<65536x8192xf16>) outs(%2:tensor<65536x8192xf16>) {
  ^bb0(%in: f16, %out: f16):
    %c0 = arith.constant 0.0 : f16
    %max = arith.maximumf %in, %c0 : f16
    linalg.yield %max : f16
  } -> tensor<65536x8192xf16> 
  %3 = tensor.empty() : tensor<65536x256xf16>
  %mm1 = linalg.matmul ins(%relu, %w_ff1 : tensor<65536x8192xf16>, tensor<8192x256xf16>) outs(%3:tensor<65536x256xf16>) -> tensor<65536x256xf16>
  %4 = tensor.empty() : tensor<65536x256xf16>
  %ff1 = linalg.add ins(%mm1, %b_ff1: tensor<65536x256xf16>, tensor<65536x256xf16>) outs(%4:tensor<65536x256xf16>) -> tensor<65536x256xf16>

  return %ff1 : tensor<65536x256xf16>
}
