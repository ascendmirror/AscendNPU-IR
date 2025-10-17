// REQUIRES: execution-engine
// RUN: bishengir-opt --execution-engine-create-host-main %s | FileCheck %s

module {
    func.func @kernel(%arg0: memref<?x5xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<?x5xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<5xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}, %arg3: memref<5xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>}, %arg4: memref<5xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<2>}) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
        %c0 = arith.constant 0.: f32
        linalg.fill ins(%c0: f32) outs(%arg2: memref<5xf32, #hivm.address_space<gm>>)
        linalg.reduce {arith.addf} ins(%arg0: memref<?x5xf32, #hivm.address_space<gm>>) outs(%arg2: memref<5xf32, #hivm.address_space<gm>>) dimensions = [0]
        linalg.fill ins(%c0: f32) outs(%arg3: memref<5xf32, #hivm.address_space<gm>>)
        linalg.reduce {arith.addf} ins(%arg1: memref<?x5xf32, #hivm.address_space<gm>>) outs(%arg3: memref<5xf32, #hivm.address_space<gm>>) dimensions = [0]
        linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg2, %arg3: memref<5xf32, #hivm.address_space<gm>>, memref<5xf32, #hivm.address_space<gm>>) outs(%arg4: memref<5xf32, #hivm.address_space<gm>>)
        func.return
    }
}

// CHECK-LABEL:   func.func private @closeFileHandle(!llvm.ptr)
// CHECK-LABEL:   func.func private @printDataF32(!llvm.ptr, memref<*xf32>) attributes {llvm.emit_c_interface}
// CHECK-LABEL:   func.func private @getFileHandle(!llvm.ptr) -> !llvm.ptr
// CHECK-LABEL:   func.func private @getDataF32(memref<*xf32>) attributes {llvm.emit_c_interface}

// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:[^:]*]]: memref<?x5xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>},
// CHECK-SAME:                      %[[VAL_1:[^:]*]]: memref<?x5xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>},
// CHECK-SAME:                      %[[VAL_2:[^:]*]]: memref<5xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>},
// CHECK-SAME:                      %[[VAL_3:[^:]*]]: memref<5xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>},
// CHECK-SAME:                      %[[VAL_4:[^:]*]]: memref<5xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<2>})

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:[^:]*]]: index {execution_engine.arg_type = #execution_engine.arg_type<dyn_size>},
// CHECK-SAME:                    %[[VAL_1:[^:]*]]: index {execution_engine.arg_type = #execution_engine.arg_type<dyn_size>},
// CHECK-SAME:                    %[[VAL_2:[^:]*]]: !llvm.ptr {execution_engine.arg_type = #execution_engine.arg_type<input>},
// CHECK-SAME:                    %[[VAL_3:[^:]*]]: !llvm.ptr {execution_engine.arg_type = #execution_engine.arg_type<output>}) {
// CHECK:           %[[VAL_4:.*]] = memref.alloc(%[[VAL_0]]) : memref<?x5xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_5:.*]] = memref.alloc(%[[VAL_1]]) : memref<?x5xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<5xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<5xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_8:.*]] = memref.alloc() : memref<5xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_9:.*]] = memref.cast %[[VAL_4]] : memref<?x5xf32, #hivm.address_space<gm>> to memref<*xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_10:.*]] = memref.memory_space_cast %[[VAL_9]] : memref<*xf32, #hivm.address_space<gm>> to memref<*xf32>
// CHECK:           %[[VAL_11:.*]] = memref.cast %[[VAL_5]] : memref<?x5xf32, #hivm.address_space<gm>> to memref<*xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_12:.*]] = memref.memory_space_cast %[[VAL_11]] : memref<*xf32, #hivm.address_space<gm>> to memref<*xf32>
// CHECK:           %[[VAL_13:.*]] = memref.cast %[[VAL_6]] : memref<5xf32, #hivm.address_space<gm>> to memref<*xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_14:.*]] = memref.memory_space_cast %[[VAL_13]] : memref<*xf32, #hivm.address_space<gm>> to memref<*xf32>
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_7]] : memref<5xf32, #hivm.address_space<gm>> to memref<*xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_16:.*]] = memref.memory_space_cast %[[VAL_15]] : memref<*xf32, #hivm.address_space<gm>> to memref<*xf32>
// CHECK:           %[[VAL_17:.*]] = memref.cast %[[VAL_8]] : memref<5xf32, #hivm.address_space<gm>> to memref<*xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_18:.*]] = memref.memory_space_cast %[[VAL_17]] : memref<*xf32, #hivm.address_space<gm>> to memref<*xf32>
// CHECK:           call @getDataF32(%[[VAL_10]]) : (memref<*xf32>) -> ()
// CHECK:           call @getDataF32(%[[VAL_12]]) : (memref<*xf32>) -> ()
// CHECK:           call @getDataF32(%[[VAL_14]]) : (memref<*xf32>) -> ()
// CHECK:           call @getDataF32(%[[VAL_16]]) : (memref<*xf32>) -> ()
// CHECK:           call @getDataF32(%[[VAL_18]]) : (memref<*xf32>) -> ()
// CHECK:           %[[VAL_19:.*]] = call @getFileHandle(%[[VAL_2]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           call @printDataF32(%[[VAL_19]], %[[VAL_10]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @printDataF32(%[[VAL_19]], %[[VAL_12]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @printDataF32(%[[VAL_19]], %[[VAL_14]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @printDataF32(%[[VAL_19]], %[[VAL_16]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @printDataF32(%[[VAL_19]], %[[VAL_18]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @closeFileHandle(%[[VAL_19]]) : (!llvm.ptr) -> ()
// CHECK:           %[[VAL_20:.*]] = memref.cast %[[VAL_10]] : memref<*xf32> to memref<?x5xf32>
// CHECK:           %[[VAL_21:.*]] = memref.memory_space_cast %[[VAL_20]] : memref<?x5xf32> to memref<?x5xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_22:.*]] = memref.cast %[[VAL_12]] : memref<*xf32> to memref<?x5xf32>
// CHECK:           %[[VAL_23:.*]] = memref.memory_space_cast %[[VAL_22]] : memref<?x5xf32> to memref<?x5xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_24:.*]] = memref.cast %[[VAL_14]] : memref<*xf32> to memref<5xf32>
// CHECK:           %[[VAL_25:.*]] = memref.memory_space_cast %[[VAL_24]] : memref<5xf32> to memref<5xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_26:.*]] = memref.cast %[[VAL_16]] : memref<*xf32> to memref<5xf32>
// CHECK:           %[[VAL_27:.*]] = memref.memory_space_cast %[[VAL_26]] : memref<5xf32> to memref<5xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_28:.*]] = memref.cast %[[VAL_18]] : memref<*xf32> to memref<5xf32>
// CHECK:           %[[VAL_29:.*]] = memref.memory_space_cast %[[VAL_28]] : memref<5xf32> to memref<5xf32, #hivm.address_space<gm>>
// CHECK:           call @kernel(%[[VAL_21]], %[[VAL_23]], %[[VAL_25]], %[[VAL_27]], %[[VAL_29]]) : (memref<?x5xf32, #hivm.address_space<gm>>, memref<?x5xf32, #hivm.address_space<gm>>, memref<5xf32, #hivm.address_space<gm>>, memref<5xf32, #hivm.address_space<gm>>, memref<5xf32, #hivm.address_space<gm>>) -> ()
// CHECK:           %[[VAL_30:.*]] = memref.cast %[[VAL_25]] : memref<5xf32, #hivm.address_space<gm>> to memref<*xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_31:.*]] = memref.memory_space_cast %[[VAL_30]] : memref<*xf32, #hivm.address_space<gm>> to memref<*xf32>
// CHECK:           %[[VAL_32:.*]] = memref.cast %[[VAL_27]] : memref<5xf32, #hivm.address_space<gm>> to memref<*xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_33:.*]] = memref.memory_space_cast %[[VAL_32]] : memref<*xf32, #hivm.address_space<gm>> to memref<*xf32>
// CHECK:           %[[VAL_34:.*]] = memref.cast %[[VAL_29]] : memref<5xf32, #hivm.address_space<gm>> to memref<*xf32, #hivm.address_space<gm>>
// CHECK:           %[[VAL_35:.*]] = memref.memory_space_cast %[[VAL_34]] : memref<*xf32, #hivm.address_space<gm>> to memref<*xf32>
// CHECK:           %[[VAL_36:.*]] = call @getFileHandle(%[[VAL_3]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           call @printDataF32(%[[VAL_36]], %[[VAL_31]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @printDataF32(%[[VAL_36]], %[[VAL_33]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @printDataF32(%[[VAL_36]], %[[VAL_35]]) : (!llvm.ptr, memref<*xf32>) -> ()
// CHECK:           call @closeFileHandle(%[[VAL_36]]) : (!llvm.ptr) -> ()
// CHECK:           return
// CHECK:         }
