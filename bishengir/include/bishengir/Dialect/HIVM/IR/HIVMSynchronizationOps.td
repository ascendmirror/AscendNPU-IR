/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file HIVMSynchronizationOps.td
 * \brief HIVM dialect Sync. Ops.
 * \details This is the definition file for HIVM dialect
 *          synchronization operations.
 */

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMSYNCHRONIZATIONOPS_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMSYNCHRONIZATIONOPS_TD

include "bishengir/Dialect/HIVM/IR/HIVMBase.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.td"
include "bishengir/Dialect/HIVM/IR/HIVMTraits.td"

include "mlir/IR/BuiltinAttributes.td"

//===----------------------------------------------------------------------===//
// HIVM pipe synchronization operations.
//===----------------------------------------------------------------------===//

class HIVM_SynchronizationOp<string mnemonic, list<Trait> traits = []> :
  HIVM_Op<mnemonic, !listconcat(
    [
#ifndef BISHENGIR_BUILD_STANDALONE_IR_ONLY
      DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface,
      ["inferCoreType"]>
#endif
    ], traits)> {
}

def SetFlagOp : HIVM_SynchronizationOp<"set_flag"> {
  let summary = "hivm set flag.";
  let arguments = (ins HIVM_PipeAttr:$set_pipe,
                       HIVM_PipeAttr:$wait_pipe,
                       OptionalAttr<HIVM_EventAttr>:$static_event_id,
                       Optional<I64>:$dynamic_event_id
  );
  let assemblyFormat = [{
    `[`
    $set_pipe
    `,` $wait_pipe
    `,` custom<EventID>($static_event_id, $dynamic_event_id)
    `]` attr-dict
  }];
  let hasVerifier = 1;
}

def WaitFlagOp : HIVM_SynchronizationOp<"wait_flag"> {
  let summary = "hivm wait flag.";
  let arguments = (ins HIVM_PipeAttr:$set_pipe,
                       HIVM_PipeAttr:$wait_pipe,
                       OptionalAttr<HIVM_EventAttr>:$static_event_id,
                       Optional<I64>:$dynamic_event_id
  );
  let assemblyFormat = [{
    `[`
    $set_pipe
    `,` $wait_pipe
    `,` custom<EventID>($static_event_id, $dynamic_event_id)
    `]` attr-dict
  }];
  let hasVerifier = 1;
}

def PipeBarrierOp : HIVM_SynchronizationOp<"pipe_barrier"> {
  let summary = "hivm pipe barrier.";
  let arguments = (ins HIVM_PipeAttr:$pipe);
  let assemblyFormat = "`[` $pipe `]` attr-dict";
}

//===----------------------------------------------------------------------===//
// HIVM block synchronization operations.
//===----------------------------------------------------------------------===//

def SyncBlockOp : HIVM_SynchronizationOp<"sync_block"> {
  let summary = "hivm sync block between different kernels.";
  let description = [{
    There are three sync block modes:
      - ALL_CUBE : All vectors are synchronized to a same point.
                   `tcube_pipe` needs to be set to the pipe that
                   the cube core is waiting for.
      - ALL_VECTOR : All cube are synchronized to a same point.
                   `tvector_pipe` needs to be set to the pipe that the
                   vector core is waiting for.
      - ALL : All aic/aiv are synchronized to same point.
              `tvector_pipe` needs to be set to the pipe that the
              vector core is waiting for.

    Note:
      - SyncBlockOp can only use after data is moved to gm.
      - `$ffts_base_addr` must be set in Ascend910B. Every time FFTS collect
        one specific `$flag_id` from all subblocks, FFTS would set the flag ID
        back to the block in the group to do synchronization.
  }];
  let arguments = (ins HIVM_SyncBlockModeAttr:$sync_block_mode,
                       Builtin_IntegerAttr:$flag_id,
                       Optional<I64>:$ffts_base_addr,
                       OptionalAttr<HIVM_PipeAttr>:$tcube_pipe,
                       OptionalAttr<HIVM_PipeAttr>:$tvector_pipe
  );
  let assemblyFormat = [{
    attr-dict `[` $sync_block_mode `,` $flag_id`]`
    (`ffts_base_addr` `=` $ffts_base_addr^)?
    (`tcube_pipe` `=` $tcube_pipe^)?
    (`tvector_pipe` `=` $tvector_pipe^)?
  }];
  let hasVerifier = 1;
}

def SyncBlockSetOp : HIVM_SynchronizationOp<"sync_block_set", [AttrSizedOperandSegments]> {
  let summary = "hivm set block sync.";
  let arguments = (ins HIVM_TCoreTypeAttr:$tcore_type,
                       HIVM_PipeAttr:$tpipe,
                       HIVM_PipeAttr:$pipe,
                       OptionalAttr<Builtin_IntegerAttr>:$static_flag_id,
                       Optional<I64>:$dynamic_flag_id,
                       Optional<I64>:$ffts_base_addr,
                       DefaultValuedOptionalAttr<HIVM_SyncBlockInstrModeAttr,
                         "INTRA_BLOCK_SYNCHRONIZATION">:$tsync_instr_mode
  );
  let assemblyFormat = [{
    attr-dict `[` $tcore_type `,` $tpipe `,` $pipe`]`
    `flag` `=` custom<FlagID>($static_flag_id, $dynamic_flag_id)
    (`ffts_base_addr` `=` $ffts_base_addr^)?
    (`syn_instr_mode` `=` $tsync_instr_mode^)?
  }];
  let builders = [
    OpBuilder<(ins "TCoreTypeAttr":$tcore_type, "PipeAttr":$tpipe, "PipeAttr":$pipe,
                   "OpFoldResult":$flag_id)>,
    OpBuilder<(ins "TCoreTypeAttr":$tcore_type, "PipeAttr":$tpipe, "PipeAttr":$pipe,
                   "OpFoldResult":$flag_id, "Value":$ffts_base_addr,
                   "SyncBlockInstrModeAttr":$tsync_instr_mode)>
  ];
  let extraClassDeclaration = [{
    ::mlir::OpFoldResult getFlagId();
  }];
  let hasVerifier = 1;
}

def SyncBlockWaitOp : HIVM_SynchronizationOp<"sync_block_wait"> {
  let summary = "hivm wait block sync.";
  let arguments = (ins HIVM_TCoreTypeAttr:$tcore_type,
                       HIVM_PipeAttr:$tpipe,
                       HIVM_PipeAttr:$pipe,
                       OptionalAttr<Builtin_IntegerAttr>:$static_flag_id,
                       Optional<I64>:$dynamic_flag_id
  );
  let assemblyFormat = [{
    attr-dict `[` $tcore_type `,` $tpipe `,` $pipe`]`
    `flag` `=` custom<FlagID>($static_flag_id, $dynamic_flag_id)
  }];
  let builders = [
    OpBuilder<(ins "TCoreTypeAttr":$tcore_type, "PipeAttr":$tpipe, "PipeAttr":$pipe,
                   "OpFoldResult":$flag_id)>
  ];
  let extraClassDeclaration = [{
    ::mlir::OpFoldResult getFlagId();
  }];
  let hasVerifier = 1;
}

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMSYNCHRONIZATIONOPS_TD
