// REQUIRES: execution-engine
// RUN: bishengir-opt --execution-engine-convert-hivm-to-upstream %s --split-input-file | FileCheck %s
// RUN: bishengir-opt --lower-for-cpu-runner-pipeline %s --split-input-file

func.func @tensor_direct_linalg_lowering(%a: tensor<1x?x10xf32>, %b: tensor<?x5x10xf32>, %c: tensor<5x?x10xf32>) -> tensor<5x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: linalg.abs
    %0 = hivm.hir.vabs ins(%a: tensor<1x?x10xf32>) outs(%c: tensor<5x?x10xf32>) broadcast = [0] -> tensor<5x?x10xf32>

    // CHECK: linalg.add
    %1 = hivm.hir.vadd ins(%b, %b: tensor<?x5x10xf32>, tensor<?x5x10xf32>) outs(%0: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

    // CHECK: linalg.sub
    %2 = hivm.hir.vsub ins(%0, %1: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.mul
    %3 = hivm.hir.vmul ins(%1, %2: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.div
    %4 = hivm.hir.vdiv ins(%2, %3: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.max
    %5 = hivm.hir.vmax ins(%3, %4: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.min
    %6 = hivm.hir.vmin ins(%4, %5: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.exp
    %7 = hivm.hir.vexp ins(%6: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.log
    %8 = hivm.hir.vln ins(%7: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.rsqrt
    %9 = hivm.hir.vrsqrt ins(%8: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.sqrt
    %10 = hivm.hir.vsqrt ins(%9: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.tanh
    %11 = hivm.hir.vtanh ins(%10: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.reciprocal
    %12 = hivm.hir.vrec ins(%11: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.select
    %13 = arith.constant true
    %14 = hivm.hir.vsel ins(%13, %c, %c: i1, tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%c: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.erf
    %15 = hivm.hir.verf ins(%14: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.copy
    %16 = hivm.hir.store ins(%15: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.transpose
    %17 = hivm.hir.vtranspose ins(%b: tensor<?x5x10xf32>) outs(%16: tensor<5x?x10xf32>) permutation = [1, 0, 2] -> tensor<5x?x10xf32>

    func.return %17: tensor<5x?x10xf32>
}

// -----

func.func @memref_direct_linalg_lowering(%a: memref<1x?x10xf32>, %b: memref<?x5x10xf32>, %c: memref<5x?x10xf32>, %d: memref<5x?x10xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: linalg.abs
    hivm.hir.vabs ins(%a: memref<1x?x10xf32>) outs(%c: memref<5x?x10xf32>) broadcast = [0]

    // CHECK: linalg.add
    hivm.hir.vadd ins(%b, %b: memref<?x5x10xf32>, memref<?x5x10xf32>) outs(%c: memref<5x?x10xf32>) transpose = [1, 0, 2]

    // CHECK: linalg.sub
    hivm.hir.vsub ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.mul
    hivm.hir.vmul ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.div
    hivm.hir.vdiv ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.max
    hivm.hir.vmax ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.min
    hivm.hir.vmin ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.exp
    hivm.hir.vexp ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.log
    hivm.hir.vln ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.rsqrt
    hivm.hir.vrsqrt ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.sqrt
    hivm.hir.vsqrt ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.tanh
    hivm.hir.vtanh ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.reciprocal
    hivm.hir.vrec ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.select
    %13 = arith.constant true
    hivm.hir.vsel ins(%13, %c, %c: i1, memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.erf
    hivm.hir.verf ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.copy
    hivm.hir.store ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.transpose
    hivm.hir.vtranspose ins(%b: memref<?x5x10xf32>) outs(%c: memref<5x?x10xf32>) permutation = [1, 0, 2]

    func.return
}

// -----

func.func @map_like_lowering(%a: tensor<?x5x10xf32>, %aT: tensor<5x?x10xf32>, %b: memref<5x1x10xi32>, %bB: memref<5x?x10xi32>) -> tensor<5x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: linalg.map
    // CHECK:   arith.maxnumf
    %0 = hivm.hir.vrelu ins(%a: tensor<?x5x10xf32>) outs(%aT: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

    // CHECK: linalg.map
    // CHECK:   arith.maxsi
    hivm.hir.vrelu ins(%b: memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]

    // CHECK: linalg.map {{.*}}
    // CHECK-NEXT: (%[[input_f:.*]]:
    // CHECK:   %[[bitcast:.*]] = arith.bitcast %[[input_f]]
    // CHECK:   %[[vnot:.*]] = arith.xori
    // CHECK-SAME:                  %[[bitcast]]
    // CHECK:   arith.bitcast %[[vnot]]
    %1 = hivm.hir.vnot ins(%a: tensor<?x5x10xf32>) outs(%0: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

    // CHECK: linalg.map {{.*}}
    // CHECK-NEXT: (%[[input_i:.*]]:
    // CHECK:   arith.xori
    // CHECK-SAME:   %[[input_i]]
    hivm.hir.vnot ins(%b: memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]

    func.return %1: tensor<5x?x10xf32>
}

// -----

func.func @bitwise_like_lowering(%a: tensor<?x5x10xf32>, %aT: tensor<5x?x10xf32>, %b: memref<5x1x10xi32>, %bB: memref<5x?x10xi32>) -> tensor<5x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: linalg.map {{.*}}
    // CHECK-2: arith.bitcast
    // CHECK:   arith.andi
    // CHECK:   arith.bitcast
    %0 = hivm.hir.vand ins(%a, %a: tensor<?x5x10xf32>, tensor<?x5x10xf32>) outs(%aT: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

    // CHECK: linalg.map
    // CHECK-SAME:  arith.andi
    hivm.hir.vand ins(%b, %b: memref<5x1x10xi32>, memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]

    // CHECK: linalg.map {{.*}}
    // CHECK-2: arith.bitcast
    // CHECK:   arith.ori
    // CHECK:   arith.bitcast
    %1 = hivm.hir.vor ins(%a, %a: tensor<?x5x10xf32>, tensor<?x5x10xf32>) outs(%0: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

    // CHECK: linalg.map
    // CHECK-SAME:  arith.ori
    hivm.hir.vor ins(%b, %b: memref<5x1x10xi32>, memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]

    // CHECK: linalg.map
    // CHECK-SAME:  arith.xori
    hivm.hir.vxor ins(%bB, %bB: memref<5x?x10xi32>, memref<5x?x10xi32>) outs(%bB: memref<5x?x10xi32>)

    func.return %1: tensor<5x?x10xf32>
}

// -----

func.func @cumulative_like_lowering(%a: tensor<5x?x10xf32>, %b: memref<5x?x10xi32>) -> tensor<5x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: linalg.generic
    // CHECK-SAME:  outs({{.*}}: tensor<5x?x10xf32>, tensor<5x1x1xf32>)
    // CHECK-NEXT:  ^bb0(%[[in:.*]]: f32, %{{.*}}: f32, %[[out:.*]]: f32)
    // CHECK-NEXT:      %[[res:.*]] = arith.mulf
    // CHECK-DAG-SAME:      %[[in]]
    // CHECK-DAG-SAME:      %[[out]]
    // CHECK-NEXT:      linalg.yield %[[res]], %[[res]]
    %0 = hivm.hir.vcumprod ins(%a: tensor<5x?x10xf32>) outs(%a: tensor<5x?x10xf32>) cum_dims = [0] -> tensor<5x?x10xf32>

    // CHECK: linalg.generic
    // CHECK-SAME:  outs({{.*}}: memref<5x?x10xi32>, memref<5x?x1xi32>)
    // CHECK-NEXT:  ^bb0(%[[in:.*]]: i32, %{{.*}}: i32, %[[out:.*]]: i32)
    // CHECK-NEXT:      %[[res:.*]] = arith.muli
    // CHECK-DAG-SAME:      %[[in]]
    // CHECK-DAG-SAME:      %[[out]]
    // CHECK-NEXT:      linalg.yield %[[res]], %[[res]]
    hivm.hir.vcumprod ins(%b: memref<5x?x10xi32>) outs(%b: memref<5x?x10xi32>) cum_dims = [1]

    // CHECK: linalg.generic
    // CHECK-SAME:  outs({{.*}}: tensor<5x?x10xf32>, tensor<5x1x1xf32>)
    // CHECK-NEXT:  ^bb0(%[[in:.*]]: f32, %{{.*}}: f32, %[[out:.*]]: f32)
    // CHECK-NEXT:      %[[res:.*]] = arith.addf
    // CHECK-DAG-SAME:      %[[in]]
    // CHECK-DAG-SAME:      %[[out]]
    // CHECK-NEXT:      linalg.yield %[[res]], %[[res]]
    %1 = hivm.hir.vcumsum ins(%a: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) cum_dims = [0] -> tensor<5x?x10xf32>

    // CHECK: linalg.generic
    // CHECK-SAME:  outs({{.*}}: memref<5x?x10xi32>, memref<5x?x1xi32>)
    // CHECK-NEXT:  ^bb0(%[[in:.*]]: i32, %{{.*}}: i32, %[[out:.*]]: i32)
    // CHECK-NEXT:      %[[res:.*]] = arith.addi
    // CHECK-DAG-SAME:      %[[in]]
    // CHECK-DAG-SAME:      %[[out]]
    // CHECK-NEXT:      linalg.yield %[[res]], %[[res]]
    hivm.hir.vcumsum ins(%b: memref<5x?x10xi32>) outs(%b: memref<5x?x10xi32>) cum_dims = [1]

    func.return %0: tensor<5x?x10xf32>
}

// -----

// CHECK-DAG: #[[IdentityMap:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[NonIndexingMap:.*]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-DAG: #[[ZeroDim1Map:.*]] = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
// CHECK-DAG: #[[ZeroDim01Map:.*]] = affine_map<(d0, d1, d2) -> (0, 0, d2)>

// CHECK: @brc_lowering
func.func @brc_lowering(%a: tensor<1x1x10xf32>, %b: tensor<5x?x10xf32>, %c: memref<5x1x10xi32>, %d: memref<5x?x10xi32>) -> tensor<5x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    %c0_f = arith.constant 0.: f32

    // CHECK: linalg.generic
    // CHECK-SAME:  indexing_maps = [#[[ZeroDim01Map]], #[[IdentityMap]]]
    // CHECK-NEXT:  ^bb0(%[[in:[^:]*]]: f32
    // CHECK-NEXT:      linalg.yield %[[in]]
    %0 = hivm.hir.vbrc ins(%a: tensor<1x1x10xf32>) outs(%b: tensor<5x?x10xf32>) broadcast_dims = [0, 1] -> tensor<5x?x10xf32>

    // CHECK: linalg.generic
    // CHECK-SAME:  indexing_maps = [#[[NonIndexingMap]], #[[IdentityMap]]]
    // CHECK-NEXT:  ^bb0(%[[in:[^:]*]]: f32
    // CHECK-NEXT:      linalg.yield %[[in]]
    %1 = hivm.hir.vbrc ins(%c0_f: f32) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    %c0_i = arith.constant 0: i32

    // CHECK: linalg.generic
    // CHECK-SAME:  indexing_maps = [#[[ZeroDim1Map]], #[[IdentityMap]]]
    // CHECK-NEXT:  ^bb0(%[[in:[^:]*]]: i32
    // CHECK-NEXT:      linalg.yield %[[in]]
    hivm.hir.vbrc ins(%c: memref<5x1x10xi32>) outs(%d: memref<5x?x10xi32>) broadcast_dims = [1]

    // CHECK: linalg.generic
    // CHECK-SAME:  indexing_maps = [#[[NonIndexingMap]], #[[IdentityMap]]]
    // CHECK-NEXT:  ^bb0(%[[in:[^:]*]]: i32
    // CHECK-NEXT:      linalg.yield %[[in]]
    hivm.hir.vbrc ins(%c0_i: i32) outs(%d: memref<5x?x10xi32>)

    func.return %1: tensor<5x?x10xf32>
}

// -----

func.func @arange_lowering(%a: tensor<5x?x10xf32>, %b: memref<5x?x10xi32>) -> tensor<5x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: %[[C0:.*]] = arith.constant 0
    %c0 = arith.constant 0: index
    // CHECK: %[[C1:.*]] = arith.constant 1
    %c1 = arith.constant 1: index
    // CHECK: %[[C2:.*]] = arith.constant 2
    %c2 = arith.constant 2: index
    // CHECK: %[[C3:.*]] = arith.constant 3
    %c3 = arith.constant 3: index

    // CHECK: linalg.generic
    // CHECK:   %[[i:.*]] = linalg.index 0
    // CHECK:   %[[mul1:.*]] = arith.muli %[[C0]], %[[i]]
    // CHECK:   %[[j:.*]] = linalg.index 1
    // CHECK:   %[[mul2:.*]] = arith.muli %[[C3]], %[[j]]
    // CHECK:   %[[add1:.*]] = arith.addi %[[mul1]], %[[mul2]]
    // CHECK:   %[[k:.*]] = linalg.index 2
    // CHECK:   %[[mul3:.*]] = arith.muli %[[C2]], %[[k]]
    // CHECK:   %[[add2:.*]] = arith.addi %[[add1]], %[[mul3]]
    // CHECK:   %[[int1:.*]] = arith.index_castui %[[add2]]
    // CHECK:   %[[float1:.*]] = arith.uitofp %[[int1]]
    // CHECK:   linalg.yield %[[float1]]
    %0 = hivm.hir.varange offset[] strides[%c0, %c3, %c2] outs(%a: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.generic
    // CHECK:   %[[i:.*]] = linalg.index 0
    // CHECK:   %[[mul4:.*]] = arith.muli %[[C1]], %[[i]]
    // CHECK:   %[[add3:.*]] = arith.addi %[[C3]], %[[mul4]]
    // CHECK:   %[[j:.*]] = linalg.index 1
    // CHECK:   %[[mul5:.*]] = arith.muli %[[C1]], %[[j]]
    // CHECK:   %[[add4:.*]] = arith.addi %[[add3]], %[[mul5]]
    // CHECK:   %[[k:.*]] = linalg.index 2
    // CHECK:   %[[mul6:.*]] = arith.muli %[[C1]], %[[k]]
    // CHECK:   %[[add5:.*]] = arith.addi %[[add4]], %[[mul6]]
    // CHECK:   %[[int2:.*]] = arith.index_castui %[[add5]]
    // CHECK:   linalg.yield %[[int2]]
    hivm.hir.varange offset[%c3] strides[%c1, %c1, %c1] outs(%b: memref<5x?x10xi32>)

    func.return %0: tensor<5x?x10xf32>
}

// -----

// CHECK-LABEL: @concat_lowering
// CHECK-SAME:      %[[a:[^:]*]]: {{[^,]*}}, 
// CHECK-SAME:      %[[b:[^:]*]]: {{[^,]*}}, 
// CHECK-SAME:      %[[c:[^:]*]]: {{[^,]*}}, 
// CHECK-SAME:      %[[d:[^:]*]]: {{[^,]*}}, 
// CHECK-SAME:      %[[e:[^:]*]]: {{[^,]*}}, 
// CHECK-SAME:      %[[f:[^:]*]]: {{[^,]*}}
func.func @concat_lowering(%a: tensor<5x?x10xf32>, %b: tensor<?x?x10xf32>, %c: tensor<?x?x10xf32>, %d: memref<5x?x10xi32>, %e: memref<?x?x10xi32>, %f: memref<?x?x10xi32>) -> tensor<?x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: tensor.concat
    %0 = hivm.hir.vconcat dim(0) ins(%a, %b: tensor<5x?x10xf32>, tensor<?x?x10xf32>) outs(%c: tensor<?x?x10xf32>) -> tensor<?x?x10xf32>

    // CHECK-DAG: %[[tensorD:.*]] = bufferization.to_tensor %[[d]]
    // CHECK-DAG: %[[tensorE:.*]] = bufferization.to_tensor %[[e]]
    // CHECK:   %[[concat:.*]] = tensor.concat dim(0)
    // CHECK-DAG-SAME:                      %[[tensorD]]
    // CHECK-DAG-SAME:                      %[[tensorE]]
    // CHECK:   %[[bufDst:.*]] = bufferization.to_memref %[[concat]]
    // CHECK:   memref.copy %[[bufDst]], %[[f]]
    hivm.hir.vconcat dim(0) ins(%d, %e: memref<5x?x10xi32>, memref<?x?x10xi32>) outs(%f: memref<?x?x10xi32>)

    func.return %0: tensor<?x?x10xf32>
}

// -----

func.func @reduce_lowering(%a: tensor<5x?x10xf32>, %ai: tensor<5x?x10xi32>, %b: tensor<5x1x10xf32>, %bi: tensor<5x1x10xi32>, %c: tensor<5x1x10xi32>, %d: memref<5x?x10xi32>, %e: memref<1x?x10xi32>, %f: memref<1x?x10xi32>) -> (tensor<5x1x10xf32>, tensor<5x1x10xi32>, tensor<5x1x10xi32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: linalg.generic
    // CHECK-SAME:  iterator_types = ["parallel", "reduction", "parallel"]
    // CHECK:   arith.addf
    %0 = hivm.hir.vreduce <sum> ins(%a: tensor<5x?x10xf32>) outs(%b: tensor<5x1x10xf32>) reduce_dims = [1] -> tensor<5x1x10xf32>

    // CHECK: linalg.generic
    // CHECK-SAME:  iterator_types = ["reduction", "parallel", "parallel"]
    // CHECK:   arith.addi
    hivm.hir.vreduce <sum> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x?x10xi32>) reduce_dims = [0]

    // CHECK: linalg.generic
    // CHECK:   arith.mulf
    %1 = hivm.hir.vreduce <prod> ins(%a: tensor<5x?x10xf32>) outs(%0: tensor<5x1x10xf32>) reduce_dims = [1] -> tensor<5x1x10xf32>

    // CHECK: linalg.generic
    // CHECK:   arith.muli
    hivm.hir.vreduce <prod> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x?x10xi32>) reduce_dims = [0]

    // CHECK: linalg.generic
    // CHECK:   ^bb0(%[[in1:[^:]*]]: f32, %[[out1:[^:]*]]: f32):
    // CHECK:       %[[cmp1:.*]] = arith.cmpf ugt, %[[in1]], %[[out1]]
    // CHECK:       %[[select1:.*]] = arith.select %[[cmp1]], %[[in1]], %[[out1]]
    // CHECK:       linalg.yield %[[select1]]
    %2 = hivm.hir.vreduce <any> ins(%a: tensor<5x?x10xf32>) outs(%1: tensor<5x1x10xf32>) reduce_dims = [1] -> tensor<5x1x10xf32>

    // CHECK: linalg.generic
    // CHECK:   ^bb0(%[[in2:[^:]*]]: i32, %[[out2:[^:]*]]: i32):
    // CHECK:       %[[cmp2:.*]] = arith.cmpi sgt, %[[in2]], %[[out2]]
    // CHECK:       %[[select2:.*]] = arith.select %[[cmp2]], %[[in2]], %[[out2]]
    // CHECK:       linalg.yield %[[select2]]
    hivm.hir.vreduce <any> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x?x10xi32>) reduce_dims = [0]

    // CHECK: linalg.generic
    // CHECK:   ^bb0(%[[in3:[^:]*]]: f32, %[[out3:[^:]*]]: f32):
    // CHECK:       %[[cmp3:.*]] = arith.cmpf ugt, %[[in3]], %[[out3]]
    // CHECK:       %[[select3:.*]] = arith.select %[[cmp3]], %[[in3]], %[[out3]]
    // CHECK:       linalg.yield %[[select3]]
    %3 = hivm.hir.vreduce <max> ins(%a: tensor<5x?x10xf32>) outs(%2: tensor<5x1x10xf32>) reduce_dims = [1] -> tensor<5x1x10xf32>

    // CHECK: linalg.generic
    // CHECK:   ^bb0(%[[in4:[^:]*]]: i32, %[[out4:[^:]*]]: i32):
    // CHECK:       %[[cmp4:.*]] = arith.cmpi sgt, %[[in4]], %[[out4]]
    // CHECK:       %[[select4:.*]] = arith.select %[[cmp4]], %[[in4]], %[[out4]]
    // CHECK:       linalg.yield %[[select4]]
    hivm.hir.vreduce <max> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x?x10xi32>) reduce_dims = [0]

    // CHECK: linalg.generic
    // CHECK:   ^bb0(%[[in5:[^:]*]]: f32, %[[out5:[^:]*]]: f32, %[[out6:[^:]*]]: i32):
    // CHECK:       %[[cmp5:.*]] = arith.cmpf ugt, %[[in5]], %[[out5]]
    // CHECK:       %[[select5:.*]] = arith.select %[[cmp5]], %[[in5]], %[[out5]]
    // CHECK:       %[[cmp6:.*]] = arith.cmpf oeq, %[[select5]], %[[out5]]
    // CHECK:       %[[index1:.*]] = linalg.index 1
    // CHECK:       %[[index_i1:.*]] = arith.index_castui %[[index1]]
    // CHECK:       %[[cmp7:.*]] = arith.cmpi ult, %[[index_i1]], %[[out6]]
    // CHECK:       %[[and1:.*]] = arith.andi %[[cmp6]], %[[cmp7]]
    // CHECK:       %[[or1:.*]] = arith.ori %[[cmp5]], %[[and1]]
    // CHECK:       %[[select6:.*]] = arith.select %[[or1]], %[[index_i1]], %[[out6]]
    // CHECK:       linalg.yield %[[select5]], %[[select6]]
    %4, %id1 = hivm.hir.vreduce <max_with_index> ins(%a: tensor<5x?x10xf32>) outs(%3, %c: tensor<5x1x10xf32>, tensor<5x1x10xi32>) reduce_dims = [1] -> tensor<5x1x10xf32>, tensor<5x1x10xi32>

    // CHECK: linalg.generic
    // CHECK:   ^bb0(%[[in6:[^:]*]]: i32, %[[out7:[^:]*]]: i32, %[[out8:[^:]*]]: i32):
    // CHECK:       %[[cmp8:.*]] = arith.cmpi sgt, %[[in6]], %[[out7]]
    // CHECK:       %[[select7:.*]] = arith.select %[[cmp8]], %[[in6]], %[[out7]]
    // CHECK:       %[[cmp9:.*]] = arith.cmpi eq, %[[select7]], %[[out7]]
    // CHECK:       %[[index2:.*]] = linalg.index 0
    // CHECK:       %[[index_i2:.*]] = arith.index_castui %[[index2]]
    // CHECK:       %[[cmp10:.*]] = arith.cmpi ult, %[[index_i2]], %[[out8]]
    // CHECK:       %[[and2:.*]] = arith.andi %[[cmp9]], %[[cmp10]]
    // CHECK:       %[[or2:.*]] = arith.ori %[[cmp8]], %[[and2]]
    // CHECK:       %[[select8:.*]] = arith.select %[[or2]], %[[index_i2]], %[[out8]]
    // CHECK:       linalg.yield %[[select7]], %[[select8]]
    hivm.hir.vreduce <max_with_index> ins(%d: memref<5x?x10xi32>) outs(%e, %f: memref<1x?x10xi32>, memref<1x?x10xi32>) reduce_dims = [0]

    // CHECK: linalg.generic
    // CHECK:   ^bb0(%[[in7:[^:]*]]: f32, %[[out9:[^:]*]]: f32):
    // CHECK:       %[[cmp11:.*]] = arith.cmpf ult, %[[in7]], %[[out9]]
    // CHECK:       %[[select9:.*]] = arith.select %[[cmp11]], %[[in7]], %[[out9]]
    // CHECK:       linalg.yield %[[select9]]
    %5 = hivm.hir.vreduce <all> ins(%a: tensor<5x?x10xf32>) outs(%4: tensor<5x1x10xf32>) reduce_dims = [1] -> tensor<5x1x10xf32>

    // CHECK: linalg.generic
    // CHECK:   ^bb0(%[[in8:[^:]*]]: i32, %[[out10:[^:]*]]: i32):
    // CHECK:       %[[cmp12:.*]] = arith.cmpi slt, %[[in8]], %[[out10]]
    // CHECK:       %[[select10:.*]] = arith.select %[[cmp12]], %[[in8]], %[[out10]]
    // CHECK:       linalg.yield %[[select10]]
    hivm.hir.vreduce <all> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x?x10xi32>) reduce_dims = [0]

    // CHECK: linalg.generic
    // CHECK:   ^bb0(%[[in9:[^:]*]]: f32, %[[out11:[^:]*]]: f32):
    // CHECK:       %[[cmp13:.*]] = arith.cmpf ult, %[[in9]], %[[out11]]
    // CHECK:       %[[select11:.*]] = arith.select %[[cmp13]], %[[in9]], %[[out11]]
    // CHECK:       linalg.yield %[[select11]]
    %6 = hivm.hir.vreduce <min> ins(%a: tensor<5x?x10xf32>) outs(%5: tensor<5x1x10xf32>) reduce_dims = [1] -> tensor<5x1x10xf32>

    // CHECK: linalg.generic
    // CHECK:   ^bb0(%[[in10:[^:]*]]: i32, %[[out12:[^:]*]]: i32):
    // CHECK:       %[[cmp14:.*]] = arith.cmpi slt, %[[in10]], %[[out12]]
    // CHECK:       %[[select12:.*]] = arith.select %[[cmp14]], %[[in10]], %[[out12]]
    // CHECK:       linalg.yield %[[select12]]
    hivm.hir.vreduce <min> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x?x10xi32>) reduce_dims = [0]

    // CHECK: linalg.generic
    // CHECK:   ^bb0(%[[in11:[^:]*]]: f32, %[[out13:[^:]*]]: f32, %[[out14:[^:]*]]: i32):
    // CHECK:       %[[cmp13:.*]] = arith.cmpf ult, %[[in11]], %[[out13]]
    // CHECK:       %[[select13:.*]] = arith.select %[[cmp13]], %[[in11]], %[[out13]]
    // CHECK:       %[[cmp14:.*]] = arith.cmpf oeq, %[[select13]], %[[out13]]
    // CHECK:       %[[index3:.*]] = linalg.index 1
    // CHECK:       %[[index_i3:.*]] = arith.index_castui %[[index3]]
    // CHECK:       %[[cmp15:.*]] = arith.cmpi ult, %[[index_i3]], %[[out14]]
    // CHECK:       %[[and3:.*]] = arith.andi %[[cmp14]], %[[cmp15]]
    // CHECK:       %[[or3:.*]] = arith.ori %[[cmp13]], %[[and3]]
    // CHECK:       %[[select14:.*]] = arith.select %[[or3]], %[[index_i3]], %[[out14]]
    // CHECK:       linalg.yield %[[select13]], %[[select14]]
    %7, %id2 = hivm.hir.vreduce <min_with_index> ins(%a: tensor<5x?x10xf32>) outs(%6, %id1: tensor<5x1x10xf32>, tensor<5x1x10xi32>) reduce_dims = [1] -> tensor<5x1x10xf32>,  tensor<5x1x10xi32>

    // CHECK: linalg.generic
    // CHECK:   ^bb0(%[[in12:[^:]*]]: i32, %[[out15:[^:]*]]: i32, %[[out16:[^:]*]]: i32):
    // CHECK:       %[[cmp16:.*]] = arith.cmpi slt, %[[in12]], %[[out15]]
    // CHECK:       %[[select15:.*]] = arith.select %[[cmp16]], %[[in12]], %[[out15]]
    // CHECK:       %[[cmp17:.*]] = arith.cmpi eq, %[[select15]], %[[out15]]
    // CHECK:       %[[index4:.*]] = linalg.index 0
    // CHECK:       %[[index_i4:.*]] = arith.index_castui %[[index4]]
    // CHECK:       %[[cmp18:.*]] = arith.cmpi ult, %[[index_i4]], %[[out16]]
    // CHECK:       %[[and4:.*]] = arith.andi %[[cmp17]], %[[cmp18]]
    // CHECK:       %[[or4:.*]] = arith.ori %[[cmp16]], %[[and4]]
    // CHECK:       %[[select16:.*]] = arith.select %[[or4]], %[[index_i4]], %[[out16]]
    // CHECK:       linalg.yield %[[select15]], %[[select16]]
    hivm.hir.vreduce <min_with_index> ins(%d: memref<5x?x10xi32>) outs(%e, %f: memref<1x?x10xi32>, memref<1x?x10xi32>) reduce_dims = [0]

    // CHECK: linalg.generic
    // CHECK:   arith.xori
    %8 = hivm.hir.vreduce <xori> ins(%ai: tensor<5x?x10xi32>) outs(%bi: tensor<5x1x10xi32>) reduce_dims = [1] -> tensor<5x1x10xi32>

    // CHECK: linalg.generic
    // CHECK:   arith.xori
    hivm.hir.vreduce <xori> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x?x10xi32>) reduce_dims = [0]

    // CHECK: linalg.generic
    // CHECK:   arith.ori
    %9 = hivm.hir.vreduce <ori> ins(%ai: tensor<5x?x10xi32>) outs(%8: tensor<5x1x10xi32>) reduce_dims = [1] -> tensor<5x1x10xi32>

    // CHECK: linalg.generic
    // CHECK:   arith.ori
    hivm.hir.vreduce <ori> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x?x10xi32>) reduce_dims = [0]

    // CHECK: linalg.generic
    // CHECK:   arith.andi
    %10 = hivm.hir.vreduce <andi> ins(%ai: tensor<5x?x10xi32>) outs(%9: tensor<5x1x10xi32>) reduce_dims = [1] -> tensor<5x1x10xi32>

    // CHECK: linalg.generic
    // CHECK:   arith.andi
    hivm.hir.vreduce <andi> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x?x10xi32>) reduce_dims = [0]

    func.return %7, %id2, %10: tensor<5x1x10xf32>, tensor<5x1x10xi32>, tensor<5x1x10xi32>
}

// -----

// CHECK-LABEL: memref_load_lowering
// CHECK-SAME:              %[[src:[^:]*]]: {{[^,]*}}, %[[dst:[^:]*]]
func.func @memref_load_lowering(%src: memref<5x?x?xi32>, %dst: memref<5x?x?xi32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
    // CHECK: %[[c0:.*]] = arith.constant 0
    %c0 = arith.constant 0: i32
    %c2 = arith.constant 2: index
    %c5 = arith.constant 5: index

    %d2 = memref.dim %dst, %c2: memref<5x?x?xi32>
    // CHECK: %[[fifth:.*]] = arith.divui
    %fifth_d2 = arith.divui %d2, %c5: index
    // CHECK: %[[two_fifth:.*]] = arith.addi %[[fifth]], %[[fifth]]
    %two_fifth_d2 = arith.addi %fifth_d2, %fifth_d2: index

    // CHECK: linalg.copy
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>)

    // CHECK: linalg.copy
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadNull>

    // CHECK: %[[src_dim1:.*]] = memref.dim %[[src]]
    // CHECK: %[[dst_dim1:.*]] = memref.dim %[[dst]]
    // CHECK: %[[right_pad1:.*]] = arith.subi %[[dst_dim1]], %[[src_dim1]]
    // CHECK: %[[main_subview1:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[src_dim1]]]
    // CHECK: memref.copy %[[src]], %[[main_subview1]]
    // CHECK: %[[right_subview1:.*]] = memref.subview %[[dst]][0, 0, %[[src_dim1]]] [5, {{[^,]*}}, %[[right_pad1]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview1]]
    // CHECK-NEXT: ^bb0(%[[in1:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in1]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadValue> pad_value = %c0: i32

    // CHECK: %[[src_dim2:.*]] = memref.dim %[[src]]
    // CHECK: %[[right_offset1:.*]] = arith.addi %[[src_dim2]], %[[fifth]]
    // CHECK: %[[dst_dim2:.*]] = memref.dim %[[dst]]
    // CHECK: %[[right_pad2:.*]] = arith.subi %[[dst_dim2]], %[[right_offset1]]
    // CHECK: %[[left_subview1:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[left_subview1]]
    // CHECK-NEXT:  ^bb0(%[[in2:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in2]]
    // CHECK: %[[main_subview2:.*]] = memref.subview %[[dst]][0, 0, %[[fifth]]] [5, {{[^,]*}}, %[[src_dim2]]]
    // CHECK: memref.copy %[[src]], %[[main_subview2]]
    // CHECK: %[[right_subview2:.*]] = memref.subview %[[dst]][0, 0, %[[right_offset1]]] [5, {{[^,]*}}, %[[right_pad2]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview2]]
    // CHECK-NEXT: ^bb0(%[[in3:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in3]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadValue> pad_value = %c0: i32 left_padding_num = %fifth_d2: index

    // CHECK: %[[src_dim3:.*]] = memref.dim %[[src]]
    // CHECK: %[[total_src2:.*]] = arith.addi %[[src_dim3]], %[[fifth]]
    // CHECK: %[[dst_dim3:.*]] = memref.dim %[[dst]]
    // CHECK: %[[left_pad1:.*]] = arith.subi %[[dst_dim3]], %[[total_src2]]
    // CHECK: %[[right_offset2:.*]] = arith.addi %[[left_pad1]], %[[src_dim3]]
    // CHECK: %[[left_subview1:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[left_pad1]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[left_subview1]]
    // CHECK-NEXT:  ^bb0(%[[in4:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in4]]
    // CHECK: %[[main_subview3:.*]] = memref.subview %[[dst]][0, 0, %[[left_pad1]]] [5, {{[^,]*}}, %[[src_dim3]]]
    // CHECK: memref.copy %[[src]], %[[main_subview3]]
    // CHECK: %[[right_subview3:.*]] = memref.subview %[[dst]][0, 0, %[[right_offset2]]] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview3]]
    // CHECK-NEXT: ^bb0(%[[in5:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in5]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadValue> pad_value = %c0: i32 right_padding_num = %fifth_d2: index

    // CHECK: %[[src_dim4:.*]] = memref.dim %[[src]]
    // CHECK: %[[right_offset3:.*]] = arith.addi %[[src_dim4]], %[[fifth]]
    // CHECK: %[[left_subview2:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[left_subview2]]
    // CHECK-NEXT:  ^bb0(%[[in6:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in6]]
    // CHECK: %[[main_subview4:.*]] = memref.subview %[[dst]][0, 0, %[[fifth]]] [5, {{[^,]*}}, %[[src_dim4]]]
    // CHECK: memref.copy %[[src]], %[[main_subview4]]
    // CHECK: %[[right_subview4:.*]] = memref.subview %[[dst]][0, 0, %[[right_offset3]]] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview4]]
    // CHECK-NEXT: ^bb0(%[[in7:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in7]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadValue> pad_value = %c0: i32 left_padding_num = %fifth_d2: index right_padding_num = %two_fifth_d2: index

    // CHECK: %[[src_dim5:.*]] = memref.dim %[[src]]
    // CHECK: %[[dst_dim5:.*]] = memref.dim %[[dst]]
    // CHECK: %[[right_pad3:.*]] = arith.subi %[[dst_dim5]], %[[src_dim5]]
    // CHECK: %[[pad_subview1:.*]] = memref.subview %[[src]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[main_subview5:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[src_dim5]]]
    // CHECK: memref.copy %[[src]], %[[main_subview5]]
    // CHECK: %[[right_subview5:.*]] = memref.subview %[[dst]][0, 0, %[[src_dim5]]] [5, {{[^,]*}}, %[[right_pad3]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview1]]
    // CHECK-SAME: outs(%[[right_subview5]]
    // CHECK-NEXT: ^bb0(%[[in8:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in8]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadFirstElem>

    // CHECK: %[[src_dim6:.*]] = memref.dim %[[src]]
    // CHECK: %[[right_offset4:.*]] = arith.addi %[[src_dim6]], %[[two_fifth]]
    // CHECK: %[[dst_dim6:.*]] = memref.dim %[[dst]]
    // CHECK: %[[right_pad4:.*]] = arith.subi %[[dst_dim6]], %[[right_offset4]]
    // CHECK: %[[pad_subview2:.*]] = memref.subview %[[src]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[left_subview3:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview2]]
    // CHECK-SAME: outs(%[[left_subview3]]
    // CHECK-NEXT:  ^bb0(%[[in9:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in9]]
    // CHECK: %[[main_subview6:.*]] = memref.subview %[[dst]][0, 0, %[[two_fifth]]] [5, {{[^,]*}}, %[[src_dim6]]]
    // CHECK: memref.copy %[[src]], %[[main_subview6]]
    // CHECK: %[[right_subview6:.*]] = memref.subview %[[dst]][0, 0, %[[right_offset4]]] [5, {{[^,]*}}, %[[right_pad4]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview2]]
    // CHECK-SAME: outs(%[[right_subview6]]
    // CHECK-NEXT: ^bb0(%[[in10:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in10]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadFirstElem> left_padding_num = %two_fifth_d2: index

    // CHECK: %[[src_dim7:.*]] = memref.dim %[[src]]
    // CHECK: %[[total_src3:.*]] = arith.addi %[[src_dim7]], %[[two_fifth]]
    // CHECK: %[[dst_dim7:.*]] = memref.dim %[[dst]]
    // CHECK: %[[left_pad2:.*]] = arith.subi %[[dst_dim7]], %[[total_src3]]
    // CHECK: %[[right_offset5:.*]] = arith.addi %[[left_pad2]], %[[src_dim7]]
    // CHECK: %[[pad_subview3:.*]] = memref.subview %[[src]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[left_subview4:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[left_pad2]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview3]]
    // CHECK-SAME: outs(%[[left_subview4]]
    // CHECK-NEXT:  ^bb0(%[[in11:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in11]]
    // CHECK: %[[main_subview7:.*]] = memref.subview %[[dst]][0, 0, %[[left_pad2]]] [5, {{[^,]*}}, %[[src_dim7]]]
    // CHECK: memref.copy %[[src]], %[[main_subview7]]
    // CHECK: %[[right_subview7:.*]] = memref.subview %[[dst]][0, 0, %[[right_offset5]]] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview3]]
    // CHECK-SAME: outs(%[[right_subview7]]
    // CHECK-NEXT: ^bb0(%[[in12:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in12]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadFirstElem> right_padding_num = %two_fifth_d2: index

    // CHECK: %[[src_dim8:.*]] = memref.dim %[[src]]
    // CHECK: %[[right_offset6:.*]] = arith.addi %[[src_dim8]], %[[two_fifth]]
    // CHECK: %[[pad_subview4:.*]] = memref.subview %[[src]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[left_subview5:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview4]]
    // CHECK-SAME: outs(%[[left_subview5]]
    // CHECK-NEXT:  ^bb0(%[[in13:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in13]]
    // CHECK: %[[main_subview8:.*]] = memref.subview %[[dst]][0, 0, %[[two_fifth]]] [5, {{[^,]*}}, %[[src_dim8]]]
    // CHECK: memref.copy %[[src]], %[[main_subview8]]
    // CHECK: %[[right_subview8:.*]] = memref.subview %[[dst]][0, 0, %[[right_offset6]]] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview4]]
    // CHECK-SAME: outs(%[[right_subview8]]
    // CHECK-NEXT: ^bb0(%[[in14:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in14]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadFirstElem> left_padding_num = %two_fifth_d2: index right_padding_num = %fifth_d2: index

    func.return
}

// -----

// CHECK-LABEL: tensor_load_lowering
// CHECK-SAME:              %[[tensor_src:[^:]*]]: {{[^,]*}}, %[[tensor_dst:[^:]*]]
func.func @tensor_load_lowering(%src: tensor<5x?x?xf32>, %dst: tensor<5x?x?xf32>) -> tensor<5x?x?xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
    // CHECK: %[[c0:.*]] = arith.constant 0
    %c0 = arith.constant 0.: f32
    %c2 = arith.constant 2: index
    %c5 = arith.constant 5: index

    %d2 = tensor.dim %dst, %c2: tensor<5x?x?xf32>
    // CHECK: %[[fifth:.*]] = arith.divui
    %fifth_d2 = arith.divui %d2, %c5: index
    // CHECK: %[[two_fifth:.*]] = arith.addi %[[fifth]], %[[fifth]]
    %two_fifth_d2 = arith.addi %fifth_d2, %fifth_d2: index

    // CHECK: linalg.copy
    %0 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%dst: tensor<5x?x?xf32>) -> tensor<5x?x?xf32>

    // CHECK: %[[tensor_dst1:.*]] = linalg.copy
    %1 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%0: tensor<5x?x?xf32>) pad_mode = <PadNull> -> tensor<5x?x?xf32>

    // CHECK: %[[src1:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst1:.*]] = bufferization.to_memref %[[tensor_dst1]]
    // CHECK: %[[dst1:.*]] = bufferization.clone %[[original_dst1]]
    // CHECK: %[[src_dim1:.*]] = memref.dim %[[src1]]
    // CHECK: %[[dst_dim1:.*]] = memref.dim %[[dst1]]
    // CHECK: %[[right_pad1:.*]] = arith.subi %[[dst_dim1]], %[[src_dim1]]
    // CHECK: %[[main_subview1:.*]] = memref.subview %[[dst1]][0, 0, 0] [5, {{[^,]*}}, %[[src_dim1]]]
    // CHECK: memref.copy %[[src1]], %[[main_subview1]]
    // CHECK: %[[right_subview1:.*]] = memref.subview %[[dst1]][0, 0, %[[src_dim1]]] [5, {{[^,]*}}, %[[right_pad1]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview1]]
    // CHECK-NEXT: ^bb0(%[[in1:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in1]]
    // CHECK: %[[tensor_dst2:.*]] = bufferization.to_tensor %[[dst1]] restrict
    %2 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%1: tensor<5x?x?xf32>) pad_mode = <PadValue> pad_value = %c0: f32 -> tensor<5x?x?xf32>

    // CHECK: %[[src2:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst2:.*]] = bufferization.to_memref %[[tensor_dst2]]
    // CHECK: %[[dst2:.*]] = bufferization.clone %[[original_dst2]]
    // CHECK: %[[src_dim2:.*]] = memref.dim %[[src2]]
    // CHECK: %[[right_offset1:.*]] = arith.addi %[[src_dim2]], %[[fifth]]
    // CHECK: %[[dst_dim2:.*]] = memref.dim %[[dst2]]
    // CHECK: %[[right_pad2:.*]] = arith.subi %[[dst_dim2]], %[[right_offset1]]
    // CHECK: %[[left_subview1:.*]] = memref.subview %[[dst2]][0, 0, 0] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[left_subview1]]
    // CHECK-NEXT:  ^bb0(%[[in2:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in2]]
    // CHECK: %[[main_subview2:.*]] = memref.subview %[[dst2]][0, 0, %[[fifth]]] [5, {{[^,]*}}, %[[src_dim2]]]
    // CHECK: memref.copy %[[src2]], %[[main_subview2]]
    // CHECK: %[[right_subview2:.*]] = memref.subview %[[dst2]][0, 0, %[[right_offset1]]] [5, {{[^,]*}}, %[[right_pad2]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview2]]
    // CHECK-NEXT: ^bb0(%[[in3:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in3]]
    // CHECK: %[[tensor_dst3:.*]] = bufferization.to_tensor %[[dst2]] restrict
    %3 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%2: tensor<5x?x?xf32>) pad_mode = <PadValue> pad_value = %c0: f32 left_padding_num = %fifth_d2: index -> tensor<5x?x?xf32>

    // CHECK: %[[src3:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst3:.*]] = bufferization.to_memref %[[tensor_dst3]]
    // CHECK: %[[dst3:.*]] = bufferization.clone %[[original_dst3]]
    // CHECK: %[[src_dim3:.*]] = memref.dim %[[src3]]
    // CHECK: %[[total_src2:.*]] = arith.addi %[[src_dim3]], %[[fifth]]
    // CHECK: %[[dst_dim3:.*]] = memref.dim %[[dst3]]
    // CHECK: %[[left_pad1:.*]] = arith.subi %[[dst_dim3]], %[[total_src2]]
    // CHECK: %[[right_offset2:.*]] = arith.addi %[[left_pad1]], %[[src_dim3]]
    // CHECK: %[[left_subview1:.*]] = memref.subview %[[dst3]][0, 0, 0] [5, {{[^,]*}}, %[[left_pad1]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[left_subview1]]
    // CHECK-NEXT:  ^bb0(%[[in4:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in4]]
    // CHECK: %[[main_subview3:.*]] = memref.subview %[[dst3]][0, 0, %[[left_pad1]]] [5, {{[^,]*}}, %[[src_dim3]]]
    // CHECK: memref.copy %[[src3]], %[[main_subview3]]
    // CHECK: %[[right_subview3:.*]] = memref.subview %[[dst3]][0, 0, %[[right_offset2]]] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview3]]
    // CHECK-NEXT: ^bb0(%[[in5:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in5]]
    // CHECK: %[[tensor_dst4:.*]] = bufferization.to_tensor %[[dst3]] restrict
    %4 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%3: tensor<5x?x?xf32>) pad_mode = <PadValue> pad_value = %c0: f32 right_padding_num = %fifth_d2: index -> tensor<5x?x?xf32>

    // CHECK: %[[src4:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst4:.*]] = bufferization.to_memref %[[tensor_dst4]]
    // CHECK: %[[dst4:.*]] = bufferization.clone %[[original_dst4]]
    // CHECK: %[[src_dim4:.*]] = memref.dim %[[src4]]
    // CHECK: %[[right_offset3:.*]] = arith.addi %[[src_dim4]], %[[fifth]]
    // CHECK: %[[left_subview2:.*]] = memref.subview %[[dst4]][0, 0, 0] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[left_subview2]]
    // CHECK-NEXT:  ^bb0(%[[in6:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in6]]
    // CHECK: %[[main_subview4:.*]] = memref.subview %[[dst4]][0, 0, %[[fifth]]] [5, {{[^,]*}}, %[[src_dim4]]]
    // CHECK: memref.copy %[[src4]], %[[main_subview4]]
    // CHECK: %[[right_subview4:.*]] = memref.subview %[[dst4]][0, 0, %[[right_offset3]]] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview4]]
    // CHECK-NEXT: ^bb0(%[[in7:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in7]]
    // CHECK: %[[tensor_dst5:.*]] = bufferization.to_tensor %[[dst4]] restrict
    %5 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%4: tensor<5x?x?xf32>) pad_mode = <PadValue> pad_value = %c0: f32 left_padding_num = %fifth_d2: index right_padding_num = %two_fifth_d2: index -> tensor<5x?x?xf32>

    // CHECK: %[[src5:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst5:.*]] = bufferization.to_memref %[[tensor_dst5]]
    // CHECK: %[[dst5:.*]] = bufferization.clone %[[original_dst5]]
    // CHECK: %[[src_dim5:.*]] = memref.dim %[[src5]]
    // CHECK: %[[dst_dim5:.*]] = memref.dim %[[dst5]]
    // CHECK: %[[right_pad3:.*]] = arith.subi %[[dst_dim5]], %[[src_dim5]]
    // CHECK: %[[pad_subview1:.*]] = memref.subview %[[src5]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[main_subview5:.*]] = memref.subview %[[dst5]][0, 0, 0] [5, {{[^,]*}}, %[[src_dim5]]]
    // CHECK: memref.copy %[[src5]], %[[main_subview5]]
    // CHECK: %[[right_subview5:.*]] = memref.subview %[[dst5]][0, 0, %[[src_dim5]]] [5, {{[^,]*}}, %[[right_pad3]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview1]]
    // CHECK-SAME: outs(%[[right_subview5]]
    // CHECK-NEXT: ^bb0(%[[in8:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in8]]
    // CHECK: %[[tensor_dst6:.*]] = bufferization.to_tensor %[[dst5]] restrict
    %6 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%5: tensor<5x?x?xf32>) pad_mode = <PadFirstElem> -> tensor<5x?x?xf32>

    // CHECK: %[[src6:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst6:.*]] = bufferization.to_memref %[[tensor_dst6]]
    // CHECK: %[[dst6:.*]] = bufferization.clone %[[original_dst6]]
    // CHECK: %[[src_dim6:.*]] = memref.dim %[[src6]]
    // CHECK: %[[right_offset4:.*]] = arith.addi %[[src_dim6]], %[[two_fifth]]
    // CHECK: %[[dst_dim6:.*]] = memref.dim %[[dst6]]
    // CHECK: %[[right_pad4:.*]] = arith.subi %[[dst_dim6]], %[[right_offset4]]
    // CHECK: %[[pad_subview2:.*]] = memref.subview %[[src6]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[left_subview3:.*]] = memref.subview %[[dst6]][0, 0, 0] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview2]]
    // CHECK-SAME: outs(%[[left_subview3]]
    // CHECK-NEXT:  ^bb0(%[[in9:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in9]]
    // CHECK: %[[main_subview6:.*]] = memref.subview %[[dst6]][0, 0, %[[two_fifth]]] [5, {{[^,]*}}, %[[src_dim6]]]
    // CHECK: memref.copy %[[src6]], %[[main_subview6]]
    // CHECK: %[[right_subview6:.*]] = memref.subview %[[dst6]][0, 0, %[[right_offset4]]] [5, {{[^,]*}}, %[[right_pad4]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview2]]
    // CHECK-SAME: outs(%[[right_subview6]]
    // CHECK-NEXT: ^bb0(%[[in10:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in10]]
    // CHECK: %[[tensor_dst7:.*]] = bufferization.to_tensor %[[dst6]] restrict
    %7 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%6: tensor<5x?x?xf32>) pad_mode = <PadFirstElem> left_padding_num = %two_fifth_d2: index -> tensor<5x?x?xf32>

    // CHECK: %[[src7:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst7:.*]] = bufferization.to_memref %[[tensor_dst7]]
    // CHECK: %[[dst7:.*]] = bufferization.clone %[[original_dst7]]
    // CHECK: %[[src_dim7:.*]] = memref.dim %[[src7]]
    // CHECK: %[[total_src3:.*]] = arith.addi %[[src_dim7]], %[[two_fifth]]
    // CHECK: %[[dst_dim7:.*]] = memref.dim %[[dst7]]
    // CHECK: %[[left_pad2:.*]] = arith.subi %[[dst_dim7]], %[[total_src3]]
    // CHECK: %[[right_offset5:.*]] = arith.addi %[[left_pad2]], %[[src_dim7]]
    // CHECK: %[[pad_subview3:.*]] = memref.subview %[[src7]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[left_subview4:.*]] = memref.subview %[[dst7]][0, 0, 0] [5, {{[^,]*}}, %[[left_pad2]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview3]]
    // CHECK-SAME: outs(%[[left_subview4]]
    // CHECK-NEXT:  ^bb0(%[[in11:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in11]]
    // CHECK: %[[main_subview7:.*]] = memref.subview %[[dst7]][0, 0, %[[left_pad2]]] [5, {{[^,]*}}, %[[src_dim7]]]
    // CHECK: memref.copy %[[src7]], %[[main_subview7]]
    // CHECK: %[[right_subview7:.*]] = memref.subview %[[dst7]][0, 0, %[[right_offset5]]] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview3]]
    // CHECK-SAME: outs(%[[right_subview7]]
    // CHECK-NEXT: ^bb0(%[[in12:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in12]]
    // CHECK: %[[tensor_dst8:.*]] = bufferization.to_tensor %[[dst7]] restrict
    %8 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%7: tensor<5x?x?xf32>) pad_mode = <PadFirstElem> right_padding_num = %two_fifth_d2: index -> tensor<5x?x?xf32>

    // CHECK: %[[src8:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst8:.*]] = bufferization.to_memref %[[tensor_dst8]]
    // CHECK: %[[dst8:.*]] = bufferization.clone %[[original_dst8]]
    // CHECK: %[[src_dim8:.*]] = memref.dim %[[src8]]
    // CHECK: %[[right_offset6:.*]] = arith.addi %[[src_dim8]], %[[two_fifth]]
    // CHECK: %[[pad_subview4:.*]] = memref.subview %[[src8]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[left_subview5:.*]] = memref.subview %[[dst8]][0, 0, 0] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview4]]
    // CHECK-SAME: outs(%[[left_subview5]]
    // CHECK-NEXT:  ^bb0(%[[in13:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in13]]
    // CHECK: %[[main_subview8:.*]] = memref.subview %[[dst8]][0, 0, %[[two_fifth]]] [5, {{[^,]*}}, %[[src_dim8]]]
    // CHECK: memref.copy %[[src8]], %[[main_subview8]]
    // CHECK: %[[right_subview8:.*]] = memref.subview %[[dst8]][0, 0, %[[right_offset6]]] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview4]]
    // CHECK-SAME: outs(%[[right_subview8]]
    // CHECK-NEXT: ^bb0(%[[in14:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in14]]
    // CHECK: %[[tensor_dst9:.*]] = bufferization.to_tensor %[[dst8]] restrict
    %9 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%8: tensor<5x?x?xf32>) pad_mode = <PadFirstElem> left_padding_num = %two_fifth_d2: index right_padding_num = %fifth_d2: index -> tensor<5x?x?xf32>

    func.return %9: tensor<5x?x?xf32>
}
