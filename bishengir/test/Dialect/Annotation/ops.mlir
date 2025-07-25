// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s

module {
  func.func @annotation_mark(%arg0: memref<16xi64>)
  {
    %0 = "some_op"() : () -> tensor<128x4096xf16>
    // CHECK: annotation.mark
    annotation.mark %0 keys = ["key"] values = [%arg0: memref<16xi64>] : tensor<128x4096xf16>
    return
  }
}

// -----

module {
  func.func @annotation_mark_static_attr()
  {
    %0 = "some_op"() : () -> tensor<128x4096xf16>
    // CHECK: annotation.mark
    annotation.mark %0 {key = 10 : i64} : tensor<128x4096xf16>
    return
  }
}
