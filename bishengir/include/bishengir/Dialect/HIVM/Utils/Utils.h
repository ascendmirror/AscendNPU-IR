//===- Utils.h - Utilities to support the HIVM dialect -----------*- C++-*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_HIVM_UTILS_UTILS_H
#define MLIR_DIALECT_HIVM_UTILS_UTILS_H

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <cassert>
#include <queue>
#include <set>
#include <type_traits>

namespace mlir {
namespace utils {
// Value comparator for std::map
inline bool isLessValue(const Value &a, const Value &b) {
  return a.getImpl() < b.getImpl();
}

struct ValueComparator {
  bool operator()(const Value &a, const Value &b) const {
    return isLessValue(a, b);
  }
};
} // namespace utils

namespace hivm {

// TODO : put it into platform info
#define MASTK_MODE_CTROL_BIT 56

static constexpr llvm::StringLiteral kMappingAttrName = "mapping";
static constexpr llvm::StringLiteral kMapForToForallAttrName =
    "map_for_to_forall";

/// TODO: add into hivm attrs
static constexpr llvm::StringLiteral kBufferSizeInByteAttr =
    "buffer_size_in_byte";

static constexpr llvm::StringLiteral kLogicalBlockNumAttr = "logical_block_num";

const std::string Ascend910BCubeTriple = "ascend_910b_cube-unknown-cce-unknown";
const std::string Ascend910BDataLayout =
    "e-i1:8:32-i8:8:32-i16:16:32-i64:64-f16:16:32-v16:16-v32:32-n64-S64";

const std::set<hivm::AddressSpace> LocalBufferSpace{
    hivm::AddressSpace::UB, hivm::AddressSpace::L1, hivm::AddressSpace::L0C};

/// Set the input type's memory scope to the input HIVM Address Space.
void setBaseMemRefTypeScope(Value val, AddressSpaceAttr targetMemScope);

/// Get the root MemRef AllocOp for the input operand, return failure if there
/// is unsupported Ops on the search path or if the defining op is not a MemRef
/// AllocOp.
FailureOr<memref::AllocOp> getMemRefAlloc(Value operand);

SmallVector<Value>
getValueListFromMixedTypeLists(SmallVector<Value> dynamicValues,
                               ArrayRef<int64_t> staticValues, Location loc,
                               OpBuilder &builder);

// Get value's shape as operands.
FailureOr<SmallVector<Value>> getValueFromShape(Value currentValue,
                                                OpBuilder &builder);

bool IsAscend910B(Attribute triple);

/// Returns the result of MLIR's alignUp operation on constants. The RHS is
/// expected to be non-zero.
uint64_t AlignUp(uint64_t lhs, uint64_t rhs);

/// Obtain the current number of supported pipe flow types.
constexpr unsigned int getPipeNum() {
  return static_cast<unsigned int>(hivm::PIPE::PIPE_NUM);
}

/// Determine value as buffer type.
std::optional<AddressSpaceAttr> GetBufferSpaceAttr(Value operand);

/// Get operation all touch buffer.
SmallVector<Value> getOpTouchBuffer(Operation *op);

/// Determine whether there is a Local Buffer in the current operation.
bool isOpTouchLocalBuffer(Operation *op);

/// Determine whether there is in ub buffer.
bool isLocalBuffer(std::optional<AddressSpaceAttr> memorySpaceAttr);

/// Determine whether there is a global Buffer in the current operation.
bool isOpTouchGlobalBuffer(Operation *op);

/// Utilities for Map Forall To HIVMBlocks pass and transform op
struct ForallRewriteResult {
  Value mappingId;
};

/// Eliminates scf.forall ops, move their bodies to their current location, and
/// replace uses of the index variable with delinearized hivm blk idx, via
/// affine.delinearize_index.
///
/// Requires forallOp to be the top level forall of a nest, and all forall's be
/// normalized. Dynamic upper bounds are ok.
DiagnosedSilenceableFailure mapForallToBlocksImpl(
    RewriterBase &rewriter, scf::ForallOp forallOp, ForallRewriteResult &result,
    std::optional<transform::TransformOpInterface> transformOp = std::nullopt);

/// Remove attr from markOp, and remove markOp if no attr left.
void removeMarkOpAttr(annotation::MarkOp markOp, ::llvm::StringLiteral attrName,
                      bool removeOp = true);

// Remove attr from markOp, but use rewriter
void removeMarkOpAttr(annotation::MarkOp markOp, StringRef attrName,
                      RewriterBase &rewriter, bool removeOp = true);

// Check whether current for loop is subblock binded.
bool isSubBlockBindedFor(scf::ForOp op);

// Find containing subblock loop of current op.
FailureOr<scf::ForOp> findContainingSubblockLoop(Operation *op);

/// Get parent loop of val.
/// If val is yielded by the parent loop, need to get parent of parent loop.
LoopLikeOpInterface getParentLoop(Value val);

/// Flatten ptrCastOp's parent and ancestor loops into one dimension and then
/// modulo modular.
/// In the position of ptrCastOp, affineApply and indexCastOp would be
/// created.
///
/// \return IndexCastOp of affineApply
Value createNestedIndexModular(OpBuilder &builder, Operation *op,
                               int modular = 2);

Value createNestedIndexModular(OpBuilder &builder, LoopLikeOpInterface loopOp,
                               int modular = 2);

Value createNestedIndexForOp(OpBuilder &builder, Operation *operation);

// Util func `traceForPotentialMatrixC` aims to judge whether current operand
// value of store-related operation could come from matmul result MatrixC.
//
// And it should be used with fixpipe optimization.
FailureOr<SmallVector<Operation *>> traceForPotentialMatrixC(Value v,
                                                             Block *storeBlock);

bool isMarkedAsHIVMElementwiseOp(Operation *op);

bool isMixModule(ModuleOp mod);

bool isAICModule(ModuleOp mod);

bool isAIVModule(ModuleOp mod);

/// Getter setter of the hivm.module_core_type attribute.
TModuleCoreTypeAttr getModuleCoreTypeAttr(ModuleOp mod);
void setModuleCoreTypeAttr(ModuleOp mod, TModuleCoreType coreType);
void removeModuleCoreTypeAttr(ModuleOp mod);

/// Get user op of the 'op'
/// Constraints: Skip tensor::CollapseShapeOp/ExpandShapeOp
/// Constraints: Skip memref::CollapseShapeOp/ExpandShapeOp
/// Constraints: Skip memref::SubViewOp/ViewOp/ReinterpretCastOp
/// Constraints: Skip bufferization::ToMemrefOp
void getOpUsers(Operation *op, SmallVector<Operation *, 8> &userOps);

bool isLastDimTranspose(hivm::VTransposeOp op);

bool isLastTwoAxesTranspose(hivm::VTransposeOp op);

// Create local workspace of current block
Value createAllocLocalWorkSpace(OpBuilder &builder, Location loc,
                                SmallVector<int64_t> shape, Type elementType);

Value getLocalWorkSpaceTensor(PatternRewriter &rewriter, Location loc,
                              ArrayRef<int64_t> targetShapes, Type elementType);

// Create local lock var
hivm::CreateSyncBlockLockOp createSyncBlockLockVar(OpBuilder &builder,
                                                   Location loc);

/// get Operation alias pair.
std::vector<std::pair<Value, Value>> getOperationAliasInfo(Operation *op);

/// Get buffer static size.
std::optional<uint32_t> GetBufferSize(Value buffer);

// get is operation aligned according to the broadcast/reduce dim and rank
AlignKind isBrcOpAligned(VBrcOp vbrcOp, int dim, int rank);

// set bind sub block attr
void setSubBlockMapping(RewriterBase &rewriter, Operation *loop);

/// find vector ops between store and targetOp
template <typename OpType>
LogicalResult traceHIVMOpUntil(RewriterBase &rewriter, Operation *op,
                               SmallVector<Operation *> &tracedOps) {
  std::queue<Operation *> q;
  q.push(op);
  auto parentOp = op->getParentOp();

  while (!q.empty()) {
    Operation *curOp = q.front();
    q.pop();

    if (parentOp != curOp->getParentOp())
      return failure();

    if (isa<OpType>(curOp)) {
      assert(tracedOps.size() >= 1 && "there should be vector ops");
      tracedOps.push_back(curOp);
      return success();
    }

    for (const Value &src : curOp->getOperands()) {
      Operation *defOp = src.getDefiningOp();
      if (defOp != nullptr)
        q.push(defOp);
    }

    if (curOp->getDialect()->getNamespace() ==
        HIVMDialect::getDialectNamespace()) {
      tracedOps.push_back(curOp);
    }
  }

  return failure();
}

namespace util {
// Returns if the given source MemRef type is collapsible with the specified
// reassociation indices. This function works as a strict extension based
// on `memref::CollapseShapeOp::isGuaranteedCollapsible`, which has weak
// constraints on the strides of trailing one-size dimensions.
bool isGuaranteedCollapsibleStrictly(
    MemRefType srcType, ArrayRef<ReassociationIndices> reassociation);

/// Return the MemRefTypes
SmallVector<MemRefType> getMemRefTypes(TypeRange types);

/// Judge if all MemRefTypes has same rank value
bool isAllSameRank(const SmallVectorImpl<MemRefType> &memrefTypes);

bool isLastDimContiguous(Value operand);

/// Check if the operation is hivm::PointerCastOp with GM space
/// Used to check if it is lowered from triton::IntToPtrOp
bool isGMPointerCastOp(Operation *op);
} // namespace util
} // namespace hivm
} // namespace mlir

#endif // MLIR_DIALECT_HIVM_UTILS_UTILS_H
