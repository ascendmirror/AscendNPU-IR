// REQUIRES: execution-engine
// RUN: mkdir %t.dir && bishengir-opt --lower-for-cpu-runner-pipeline="wrapper-name=main" %s | bishengir-cpu-runner -e main --execution-input=%t.dir/input.txt --execution-output=%t.dir/output.txt --execution-arguments=5,5 --entry-point-result=void --shared-libs=%lib/libbishengir_runner_utils.so,%lib/libmlir_runner_utils.so,%lib/libmlir_c_runner_utils.so

module {
    func.func @kernel(%arg0: memref<?x5xbf16>, %arg1: memref<?x5xbf16>) -> (memref<5xbf16>, memref<5xbf16>, memref<5xbf16>) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
        %c0_bf16 = arith.constant 0.: bf16
        %c0 = arith.constant 0: i64
        %c10 = arith.constant 10: i64
        %c20 = arith.constant 20: i64
        %1 = hivm.hir.pointer_cast(%c0) : memref<5xbf16>
        linalg.fill ins(%c0_bf16: bf16) outs(%1: memref<5xbf16>)
        linalg.reduce {arith.addf} ins(%arg0: memref<?x5xbf16>) outs(%1: memref<5xbf16>) dimensions = [0]
        %2 = hivm.hir.pointer_cast(%c10) : memref<5xbf16>
        linalg.fill ins(%c0_bf16: bf16) outs(%2: memref<5xbf16>)
        linalg.reduce {arith.addf} ins(%arg1: memref<?x5xbf16>) outs(%2: memref<5xbf16>) dimensions = [0]
        %3 = hivm.hir.pointer_cast(%c20) : memref<5xbf16>
        linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %2: memref<5xbf16>, memref<5xbf16>) outs(%3: memref<5xbf16>)
        func.return %1, %2, %3 :memref<5xbf16>, memref<5xbf16>, memref<5xbf16>
    }
}
