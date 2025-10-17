// REQUIRES: execution-engine
// RUN: bishengir-opt --execution-engine-create-host-main --execution-engine-convert-hivm-to-upstream %s | FileCheck %s
// RUN: bishengir-opt --lower-for-cpu-runner-pipeline %s

// CHECK-LABEL: pointer_cast_lowering
// CHECK-SAME:              %[[zero_pool:[^:]*]]:
// CHECK-SAME:              %[[gm_pool:[^:]*]]:
// CHECK-SAME:              %[[ub_pool:[^:]*]]:
func.func @pointer_cast_lowering() attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
    // CHECK: %[[c32:.*]] = arith.constant 32
    %c32 = arith.constant 32: i64
    // CHECK: %[[size:.*]] = arith.constant 100
    %c100 = arith.constant 100: index

    // CHECK: %[[index0:.*]] = arith.index_castui %[[c32]] : i64 to index
    // CHECK: memref.view %[[zero_pool]][%[[index0]]][%[[size]]] : memref<?xi8> to memref<?xf32>
    %0 = hivm.hir.pointer_cast(%c32)[%c100]: memref<?xf32>

    // CHECK: %[[index1:.*]] = arith.index_castui %[[c32]] : i64 to index
    // CHECK: memref.view %[[gm_pool]][%[[index1]]][] : memref<?xi8> to memref<100x3xbf16>
    %1 = hivm.hir.pointer_cast(%c32)[]: memref<100x3xbf16, #hivm.address_space<gm>>

    // CHECK: %[[index2:.*]] = arith.index_castui %[[c32]] : i64 to index
    // CHECK: memref.view %[[ub_pool]][%[[index2]]][%[[size]]] : memref<?xi8> to memref<100x?xi64>
    %2 = hivm.hir.pointer_cast(%c32)[%c100]: memref<100x?xi64, #hivm.address_space<ub>>
    return
}
