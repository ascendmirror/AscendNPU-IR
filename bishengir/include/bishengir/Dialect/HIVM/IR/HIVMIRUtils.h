//===- HIVMIRUtils.h - HIVM IR Utilities ------------------------*- C++ -*-===//
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


#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMIRUTILS_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVMIRUTILS_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <optional>
#include <set>
#include <type_traits>
#include <vector>

namespace mlir {
namespace hivm {


// The amount of data processed by the VBITSORT instruction in one repeat.
constexpr const int VBITSORT_NUM_PER_REPEAT = 32;
  const std::map<TFuncCoreType, TCoreType> kTFuncCoreType2TCoreType = {
    {TFuncCoreType::AIC, TCoreType::CUBE},
    {TFuncCoreType::AIV, TCoreType::VECTOR},
    {TFuncCoreType::MIX, TCoreType::CUBE_OR_VECTOR},
};

struct AlignInfo {
    llvm::SmallVector<int32_t> alignDims;
    llvm::SmallVector<int32_t> alignBytes;
  
    AlignInfo(ArrayRef<int32_t> alignDims_, ArrayRef<int32_t> alignBytes_) {
      alignDims = SmallVector<int32_t>(alignDims_);
      alignBytes = SmallVector<int32_t>(alignBytes_);
    }
  
    AlignInfo(llvm::SmallVector<int32_t> alignDims_,
              llvm::SmallVector<int32_t> alignBytes_) {
      alignDims = alignDims_;
      alignBytes = alignBytes_;
    }
  
    AlignInfo(AlignInfo &&other) {
      alignDims = other.alignDims;
      alignBytes = other.alignBytes;
    }
  
    bool operator==(const AlignInfo &other);
    bool operator!=(const AlignInfo &other);
  
    void dump();
  };
  
  struct OperAlignInfo : public AlignInfo {
    Value operand;
  
    OperAlignInfo(Value operand_, ArrayRef<int32_t> alignDims_,
                  ArrayRef<int32_t> alignBytes_)
        : AlignInfo(alignDims_, alignBytes_) {
      operand = operand_;
    }
  
    OperAlignInfo(Value operand_, llvm::SmallVector<int32_t> alignDims_,
                  llvm::SmallVector<int32_t> alignBytes_)
        : AlignInfo(alignDims_, alignBytes_) {
      operand = operand_;
    }
  };

// TODO : move to platform file
const std::set<std::string> HWSupportedCast{
    "bfloat16_t_to_float_rintmode",   "bfloat16_t_to_int32_t_roundmode",
    "bfloat16_t_to_int32_t_ceilmode", "bfloat16_t_to_int32_t_floormode",
    "bfloat16_t_to_int32_t_rintmode", "bfloat16_t_to_int32_t_truncmode",
    "half_to_float_roundmode",        "half_to_float_floormode",
    "half_to_float_rintmode",         "half_to_int16_t_roundmode",
    "half_to_int16_t_ceilmode",       "half_to_int16_t_floormode",
    "half_to_int16_t_rintmode",       "half_to_int16_t_truncmode",
    "half_to_int32_t_roundmode",      "half_to_int32_t_ceilmode",
    "half_to_int32_t_floormode",      "half_to_int32_t_rintmode",
    "half_to_int32_t_truncmode",      "half_to_int4_t_roundmode",
    "half_to_int4_t_ceilmode",        "half_to_int4_t_floormode",
    "half_to_int4_t_rintmode",        "half_to_int4_t_truncmode",
    "half_to_int8_t_roundmode",       "half_to_int8_t_ceilmode",
    "half_to_int8_t_floormode",       "half_to_int8_t_rintmode",
    "half_to_int8_t_truncmode",       "half_to_uint8_t_roundmode",
    "half_to_uint8_t_ceilmode",       "half_to_uint8_t_floormode",
    "half_to_uint8_t_rintmode",       "half_to_uint8_t_truncmode",
    "float_to_bfloat16_t_roundmode",  "float_to_bfloat16_t_ceilmode",
    "float_to_bfloat16_t_floormode",  "float_to_bfloat16_t_rintmode",
    "float_to_bfloat16_t_truncmode",  "float_to_half_roundmode",
    "float_to_half_ceilmode",         "float_to_half_floormode",
    "float_to_half_oddmode",          "float_to_half_rintmode",
    "float_to_half_truncmode",        "float_to_float_roundmode",
    "float_to_float_ceilmode",        "float_to_float_floormode",
    "float_to_float_rintmode",        "float_to_float_truncmode",
    "float_to_int16_t_roundmode",     "float_to_int16_t_ceilmode",
    "float_to_int16_t_floormode",     "float_to_int16_t_rintmode",
    "float_to_int16_t_truncmode",     "float_to_int32_t_roundmode",
    "float_to_int32_t_ceilmode",      "float_to_int32_t_floormode",
    "float_to_int32_t_rintmode",      "float_to_int32_t_truncmode",
    "float_to_int64_t_roundmode",     "float_to_int64_t_ceilmode",
    "float_to_int64_t_floormode",     "float_to_int64_t_rintmode",
    "float_to_int64_t_truncmode",     "int16_t_to_half_roundmode",
    "int16_t_to_half_ceilmode",       "int16_t_to_half_floormode",
    "int16_t_to_half_rintmode",       "int16_t_to_half_truncmode",
    "int16_t_to_float_rintmode",      "int16_t_to_float_truncmode",
    "int32_t_to_float_roundmode",     "int32_t_to_float_ceilmode",
    "int32_t_to_float_floormode",     "int32_t_to_float_rintmode",
    "int32_t_to_float_truncmode",     "int32_t_to_int16_t_rintmode",
    "int32_t_to_int64_t_rintmode",    "int4_t_to_half_rintmode",
    "int64_t_to_float_roundmode",     "int64_t_to_float_ceilmode",
    "int64_t_to_float_floormode",     "int64_t_to_float_rintmode",
    "int64_t_to_float_truncmode",     "int64_t_to_int32_t_rintmode",
    "int8_t_to_half_rintmode",        "int8_t_to_half_truncmode",
    "uint8_t_to_half_rintmode",       "half_to_int32_t_rintmode",
    "half_to_float_truncmode",        "bfloat16_t_to_float_roundmode"};

/// Create nested loops by choosing `loopDims` of `target`.
/// For example:
///  `target` = memref<Ax16xBxf32>
///  `loopDims` = {0, 1}
/// The generated loops are:
///   scf.for 0 ... A
///     scf.for 0 ... 16
/// The optional `lowBound` can be used to specify the lower bound.
/// The optional `forInitArgs` can be used to specify the iter arg's initial
/// value.
/// For example:
///   scf.for lowBound ... A iter_arg(%iter1 = forInitArgs[0])
///     scf.for lowBound ... 16 iter_arg(%iter2 = forInitArgs[1])
template <typename Func>
std::vector<scf::ForOp> createNestedLoops(
    OpBuilder &rewriter, Location loc, Value target, std::set<int> loopDims,
    Func buildLoopBody, int lowBound = 0,
    std::optional<SmallVector<Value>> forInitArgs = std::nullopt) {
  std::vector<scf::ForOp> nestedFor;
  llvm::SmallVector<Value> indexes;
  if (forInitArgs.has_value())
    assert(loopDims.size() == 1 &&
           "Only support non-nested loop to use iterator arg");

  auto index = [&rewriter, &loc](int i) {
    return rewriter.create<arith::ConstantIndexOp>(loc, i);
  };
  ShapedType dstType = dyn_cast<ShapedType>(target.getType());
  assert(dstType != nullptr);
  for (int dim = 0; dim < dstType.getRank(); dim++) {
    if (!loopDims.count(dim))
      continue;
    Value upperBound;
    if (dstType.isDynamicDim(dim)) {
      upperBound = rewriter.create<memref::DimOp>(loc, target, dim);
    } else {
      upperBound = index(dstType.getDimSize(dim));
    }
    scf::ForOp forOp =
        forInitArgs.has_value()
            ? rewriter.create<scf::ForOp>(loc, index(lowBound), upperBound,
                                          index(1), forInitArgs.value())
            : rewriter.create<scf::ForOp>(loc, index(lowBound), upperBound,
                                          index(1));
    nestedFor.push_back(forOp);
    indexes.push_back(forOp.getInductionVar());
    rewriter.setInsertionPointToStart(forOp.getBody());
  }
  if constexpr (std::is_invocable_v<Func, SmallVector<Value>>)
    buildLoopBody(indexes);
  else
    buildLoopBody(indexes, nestedFor[0].getRegionIterArgs());

  return nestedFor;
}

// Get updated BaseMemRefType with new address space
BaseMemRefType getBaseMemRefTypeWithNewScope(BaseMemRefType type,
  AddressSpaceAttr targetMemScope);

namespace util {


  constexpr static unsigned int VL = 256;
  constexpr static unsigned int BL = VL / 8;
  const static int vectorBlockSizeBit = 256;
  const static int srcNumPerRepeatOfVBRCBIntrin = 8;
  constexpr static unsigned int INTR_BYTES_PER_BLOCK = 32;
  constexpr static unsigned int INTR_BYTES_PER_REPEAT = 256;
  constexpr static unsigned int VNCHWCONV_INTR_BYTES_PER_REPEAT = 512;
  
inline int64_t ceilFactor(int64_t x, int64_t y) { return (x + y - 1) / y * y; }

// Check if reduce operation is argmin or argmax
bool isArgminOrArgmax(ReduceOperation op);

// Check if transpose is on last two axes
bool isLastTwoAxesTranspose(hivm::VTransposeOp op);

                                             uint32_t getHWAlignBytes(Attribute spaceAttr);
                                             std::optional<uint32_t> getHWAlignBytes(Type t);
                                             
                                             LogicalResult getUnAlignSizeInfo(
                                                VTransposeOp op,
                                                std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList);
                                            
                                            LogicalResult getUnAlignSizeInfo(
                                                VCastOp op,
                                                std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList);
                                            
                                            LogicalResult getUnAlignSizeInfo(
                                                VSortOp op,
                                                std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList);
} // namespace util
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMIRUTILS_H

