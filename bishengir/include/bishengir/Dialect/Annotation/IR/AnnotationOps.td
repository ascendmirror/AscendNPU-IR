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
 * \file AnnotationOps.h
 * \brief Annotation op definitions
 */

#ifndef BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONOPS_TD
#define BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONOPS_TD

include "bishengir/Dialect/Annotation/IR/AnnotationBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MarkOp : Op<Annotation_Dialect, "mark", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Annotate an IR value with key-value attribute pairs";
  let description = [{
    The `annotation.mark` operation can be used to annotated an IR value
    with a set of key-value attributes pairs.

    The value can be both static (i.e. an attribute) or dynamic
    (a.k.a. an IR value).

    Example:

    ```mlir
    annotation.mark %target keys = ["key"] values = [%val]
    annotation.mark %target {key : val}
    ```
  }];
  let arguments = (ins AnyType:$src,
                       Variadic<AnyType>:$values,
                       OptionalAttr<StrArrayAttr>:$keys);
  let assemblyFormat = [{
    $src attr-dict
    (`keys` `=` $keys^)?
    (`values` `=` `[`$values^`:`type($values) `]`)?
    `:`type($src)
  }];
  let builders = [
    OpBuilder<(ins "Value":$src)>
  ];
  let extraClassDeclaration = [{
    /// Return whether the `annotation.mark` op is annotated
    /// by a key-value pair with the target `key`.
    bool isAnnotatedBy(StringRef key);

    bool isAnnotatedByStaticAttr(StringRef key);
    bool isAnnotatedByDynamicAttr(StringRef key);

    /// Return key-value pair's value with the target `key`.
    OpFoldResult getMixedAttrValue(StringRef key);

    Attribute getStaticAttrValue(StringRef key);
    Value getDynamicAttrValue(StringRef key);
  }];
}

#endif // BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONOPS_TD
