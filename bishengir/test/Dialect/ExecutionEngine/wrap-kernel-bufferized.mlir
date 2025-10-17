// REQUIRES: execution-engine
// RUN: bishengir-opt --execution-engine-create-host-main %s | FileCheck %s

module {
    func.func @kernel(%arg0: memref<?x5xbf16>, %arg1: memref<?x5xbf16>) -> (memref<5xbf16>, memref<5xbf16>, memref<5xbf16>) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
        %c0 = arith.constant 0.: bf16
        %1 = memref.alloc() : memref<5xbf16>
        linalg.fill ins(%c0: bf16) outs(%1: memref<5xbf16>)
        linalg.reduce {arith.addf} ins(%arg0: memref<?x5xbf16>) outs(%1: memref<5xbf16>) dimensions = [0]
        %2 = memref.alloc() : memref<5xbf16>
        linalg.fill ins(%c0: bf16) outs(%2: memref<5xbf16>)
        linalg.reduce {arith.addf} ins(%arg1: memref<?x5xbf16>) outs(%2: memref<5xbf16>) dimensions = [0]
        %3 = memref.alloc() : memref<5xbf16>
        linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %2: memref<5xbf16>, memref<5xbf16>) outs(%3: memref<5xbf16>)
        func.return %1, %2, %3 :memref<5xbf16>, memref<5xbf16>, memref<5xbf16>
    }
}

// CHECK-LABEL:   func.func private @closeFileHandle(!llvm.ptr)
// CHECK-LABEL:   func.func private @printDataBF16(!llvm.ptr, memref<*xbf16>) attributes {llvm.emit_c_interface}
// CHECK-LABEL:   func.func private @getFileHandle(!llvm.ptr) -> !llvm.ptr
// CHECK-LABEL:   func.func private @getDataBF16(memref<*xbf16>) attributes {llvm.emit_c_interface}

// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:[^:]*]]: memref<?x5xbf16>,
// CHECK-SAME:                      %[[VAL_1:[^:]*]]: memref<?x5xbf16>,
// CHECK-SAME:                      %[[VAL_2:[^:]*]]: memref<?xi8, #hivm.address_space<zero>> {execution_engine.memory_pool}) -> (memref<5xbf16>, memref<5xbf16>, memref<5xbf16>)

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:[^:]*]]: index {execution_engine.arg_type = #execution_engine.arg_type<dyn_size>},
// CHECK-SAME:                    %[[VAL_1:[^:]*]]: index {execution_engine.arg_type = #execution_engine.arg_type<dyn_size>},
// CHECK-SAME:                    %[[VAL_2:[^:]*]]: index {execution_engine.arg_type = #execution_engine.arg_type<mem_pool, zero>},
// CHECK-SAME:                    %[[VAL_3:[^:]*]]: !llvm.ptr {execution_engine.arg_type = #execution_engine.arg_type<input>},
// CHECK-SAME:                    %[[VAL_4:[^:]*]]: !llvm.ptr {execution_engine.arg_type = #execution_engine.arg_type<output>}) {
// CHECK:           %[[VAL_5:.*]] = memref.alloc(%[[VAL_2]]) : memref<?xi8, #hivm.address_space<zero>>
// CHECK:           %[[VAL_6:.*]] = memref.alloc(%[[VAL_0]]) : memref<?x5xbf16>
// CHECK:           %[[VAL_7:.*]] = memref.alloc(%[[VAL_1]]) : memref<?x5xbf16>
// CHECK:           %[[VAL_8:.*]] = memref.cast %[[VAL_6]] : memref<?x5xbf16> to memref<*xbf16>
// CHECK:           %[[VAL_9:.*]] = memref.cast %[[VAL_7]] : memref<?x5xbf16> to memref<*xbf16>
// CHECK:           call @getDataBF16(%[[VAL_8]]) : (memref<*xbf16>) -> ()
// CHECK:           call @getDataBF16(%[[VAL_9]]) : (memref<*xbf16>) -> ()
// CHECK:           %[[VAL_10:.*]] = call @getFileHandle(%[[VAL_3]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           call @printDataBF16(%[[VAL_10]], %[[VAL_8]]) : (!llvm.ptr, memref<*xbf16>) -> ()
// CHECK:           call @printDataBF16(%[[VAL_10]], %[[VAL_9]]) : (!llvm.ptr, memref<*xbf16>) -> ()
// CHECK:           call @closeFileHandle(%[[VAL_10]]) : (!llvm.ptr) -> ()
// CHECK:           %[[VAL_11:.*]] = memref.cast %[[VAL_8]] : memref<*xbf16> to memref<?x5xbf16>
// CHECK:           %[[VAL_12:.*]] = memref.cast %[[VAL_9]] : memref<*xbf16> to memref<?x5xbf16>
// CHECK:           %[[VAL_13:.*]]:3 = call @kernel(%[[VAL_11]], %[[VAL_12]], %[[VAL_5]]) : (memref<?x5xbf16>, memref<?x5xbf16>, memref<?xi8, #hivm.address_space<zero>>) -> (memref<5xbf16>, memref<5xbf16>, memref<5xbf16>)
// CHECK:           %[[VAL_14:.*]] = memref.cast %[[VAL_13]]#0 : memref<5xbf16> to memref<*xbf16>
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_13]]#1 : memref<5xbf16> to memref<*xbf16>
// CHECK:           %[[VAL_16:.*]] = memref.cast %[[VAL_13]]#2 : memref<5xbf16> to memref<*xbf16>
// CHECK:           %[[VAL_17:.*]] = call @getFileHandle(%[[VAL_4]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           call @printDataBF16(%[[VAL_17]], %[[VAL_14]]) : (!llvm.ptr, memref<*xbf16>) -> ()
// CHECK:           call @printDataBF16(%[[VAL_17]], %[[VAL_15]]) : (!llvm.ptr, memref<*xbf16>) -> ()
// CHECK:           call @printDataBF16(%[[VAL_17]], %[[VAL_16]]) : (!llvm.ptr, memref<*xbf16>) -> ()
// CHECK:           call @closeFileHandle(%[[VAL_17]]) : (!llvm.ptr) -> ()
// CHECK:           return
// CHECK:         }
