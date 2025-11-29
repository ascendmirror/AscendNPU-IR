// RUN: bishengir-opt -transform-interpreter -cse -split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @tile_insert_slice_0(
func.func @tile_insert_slice_0(%arg0: tensor<32x192xf32>,
                               %arg1: tensor<256x192xf32>) -> tensor<256x192xf32> {
  %inserted = tensor.insert_slice %arg0 into %arg1[0, 0] [32, 192] [1, 1] : tensor<32x192xf32> into tensor<256x192xf32>
  return %inserted : tensor<256x192xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %slice = transform.structured.match ops{["tensor.insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b, %c = transform.structured.tile_using_for %slice tile_sizes [2, 48]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @tile_insert_slice_1(
func.func @tile_insert_slice_1(%arg0: tensor<32x192xf32>,
                               %arg1: tensor<256x192xf32>) -> tensor<256x192xf32> {
  %inserted = tensor.insert_slice %arg0 into %arg1[0, 0] [32, 192] [1, 1] : tensor<32x192xf32> into tensor<256x192xf32>
  %empty = tensor.empty() : tensor<256x192xf32>
  %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
                                ins(%inserted, %arg1 : tensor<256x192xf32>, tensor<256x192xf32>)
                                outs(%empty : tensor<256x192xf32>) -> tensor<256x192xf32>
  return %mul : tensor<256x192xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["linalg.elemwise_binary"]} in %0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [16, 48] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %2 = transform.structured.match ops{["tensor.insert_slice"]} in %0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %2 into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @tile_insert_slice_partial_overlap(
func.func @tile_insert_slice_partial_overlap(%arg0: tensor<64x96xf32>,
                                              %arg1: tensor<128x192xf32>) -> tensor<128x192xf32> {
  // Insert at offset, not aligned with tile boundaries
  %inserted = tensor.insert_slice %arg0 into %arg1[32, 48] [64, 96] [1, 1] : tensor<64x96xf32> into tensor<128x192xf32>
  %empty = tensor.empty() : tensor<128x192xf32>
  %add = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
                                ins(%inserted, %arg1 : tensor<128x192xf32>, tensor<128x192xf32>)
                                outs(%empty : tensor<128x192xf32>) -> tensor<128x192xf32>
  return %add : tensor<128x192xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["linalg.elemwise_binary"]} in %0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [16, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %2 = transform.structured.match ops{["tensor.insert_slice"]} in %0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %2 into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @tile_insert_slice_chain(
func.func @tile_insert_slice_chain(%arg0: tensor<32x48xf32>,
                                   %arg1: tensor<64x96xf32>,
                                   %arg2: tensor<128x192xf32>) -> tensor<128x192xf32> {
  // Chain of inserts: small into medium, medium into large
  %inserted0 = tensor.insert_slice %arg0 into %arg1[0, 0] [32, 48] [1, 1] : tensor<32x48xf32> into tensor<64x96xf32>
  %inserted1 = tensor.insert_slice %inserted0 into %arg2[0, 0] [64, 96] [1, 1] : tensor<64x96xf32> into tensor<128x192xf32>
  %empty = tensor.empty() : tensor<128x192xf32>
  %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
                                ins(%inserted1, %arg2 : tensor<128x192xf32>, tensor<128x192xf32>)
                                outs(%empty : tensor<128x192xf32>) -> tensor<128x192xf32>
  return %mul : tensor<128x192xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["linalg.elemwise_binary"]} in %0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [16, 24] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    // Match all insert_slice ops and fuse them
    %2 = transform.structured.match ops{["tensor.insert_slice"]} in %0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %2 into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @tile_insert_slice_comes_after_scp_for(
// CHECK: scf.for
// CHECK: tensor.insert_slice
func.func @tile_insert_slice_comes_after_scp_for(%arg0: tensor<32x192xf32>,
                                                 %arg1: tensor<256x192xf32>) -> tensor<256x192xf32> {
  %inserted = tensor.insert_slice %arg0 into %arg1[0, 0] [32, 192] [1, 1] : tensor<32x192xf32> into tensor<256x192xf32>
  %empty = tensor.empty() : tensor<256x192xf32>
  %sub = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
                                ins(%inserted, %arg1 : tensor<256x192xf32>, tensor<256x192xf32>)
                                outs(%empty : tensor<256x192xf32>) -> tensor<256x192xf32>
  return %sub : tensor<256x192xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["linalg.elemwise_binary"]} in %0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [16, 48] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %2 = transform.structured.match ops{["tensor.insert_slice"]} in %0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %2 into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @tile_insert_slice_with_stride_error(
func.func @tile_insert_slice_with_stride_error(%arg0: tensor<16x96xf32>,
                                         %arg1: tensor<128x192xf32>) -> tensor<128x192xf32> {
  // expected-error @+1 {{insert slice must be continuous to tile}}
  %inserted = tensor.insert_slice %arg0 into %arg1[0, 0] [16, 96] [2, 1] : tensor<16x96xf32> into tensor<128x192xf32>
  %empty = tensor.empty() : tensor<128x192xf32>
  %max = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
                                ins(%inserted, %arg1 : tensor<128x192xf32>, tensor<128x192xf32>)
                                outs(%empty : tensor<128x192xf32>) -> tensor<128x192xf32>
  return %max : tensor<128x192xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["linalg.elemwise_binary"]} in %0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [32, 48] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %2 = transform.structured.match ops{["tensor.insert_slice"]} in %0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %2 into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @tile_insert_slice_no_op(
// CHECK: scf.for
// CHECK: tensor.insert_slice
func.func @tile_insert_slice_no_op(%arg0: tensor<128x192xf32>,
                                   %arg1: tensor<128x192xf32>) -> tensor<128x192xf32> {
  // Insert slice with size equal to the destination tensor size. This should not be tiled.
  %inserted = tensor.insert_slice %arg0 into %arg1[0, 0] [128, 192] [1, 1] : tensor<128x192xf32> into tensor<128x192xf32>
  %empty = tensor.empty() : tensor<128x192xf32>
  %add = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
                                ins(%inserted, %arg1 : tensor<128x192xf32>, tensor<128x192xf32>)
                                outs(%empty : tensor<128x192xf32>) -> tensor<128x192xf32>
  return %add : tensor<128x192xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // Try to tile the linalg op.
    %1 = transform.structured.match ops{["linalg.elemwise_binary"]} in %0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [16, 48] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    // Try to fuse the insert_slice into the loops.
    %2 = transform.structured.match ops{["tensor.insert_slice"]} in %0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %2 into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
