// RUN: bishengir-opt -convert-to-hivm-pipeline %s | FileCheck %s

module {
  func.func @test_elemwise_binary_ops(%src1 : memref<6x6xf32>, 
                                      %src2 : memref<6x6xf32>, 
                                      %dst : memref<6x6xf32>) {
    // CHECK: hivm.hir.vadd
    linalg.elemwise_binary {fun = #linalg.binary_fn<add>} 
        ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
        outs(%dst : memref<6x6xf32>)
    // CHECK: hivm.hir.vmul
    linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} 
        ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
        outs(%dst : memref<6x6xf32>)
    // CHECK: hivm.hir.vsub
    linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} 
        ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
        outs(%dst : memref<6x6xf32>)
    // CHECK: hivm.hir.vdiv
    linalg.elemwise_binary {fun = #linalg.binary_fn<div>} 
        ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
        outs(%dst : memref<6x6xf32>)
    // CHECK: hivm.hir.vmax
    linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} 
        ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
        outs(%dst : memref<6x6xf32>)
    // CHECK: hivm.hir.vmin
    linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} 
        ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
        outs(%dst : memref<6x6xf32>)
    // CHECK: hivm.hir.vmax
    linalg.elemwise_binary {fun = #linalg.binary_fn<max_unsigned>} 
        ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
        outs(%dst : memref<6x6xf32>)
    // CHECK: hivm.hir.vmin
    linalg.elemwise_binary {fun = #linalg.binary_fn<min_unsigned>} 
        ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
        outs(%dst : memref<6x6xf32>)
    return
  }
}
