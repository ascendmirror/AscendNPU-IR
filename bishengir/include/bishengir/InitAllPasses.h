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
 * \file InitAllPasses.h
 * \brief BiShengIR Passes Registration
 * \details This file defines a helper to trigger the registration of all
 *          dialects and passes to the system.
 */

#ifndef BISHENGIR_INITALLPASSES_H
#define BISHENGIR_INITALLPASSES_H

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HIVM/Pipelines/Passes.h"

namespace bishengir {

// This function may be called to register the bishengir-specific MLIR passes
// with the global registry.
inline void registerAllPasses() {
  // Conversion passes
  bishengir::registerConversionPasses();

  // Dialect pipelines
  mlir::hivm::registerConvertToHIVMPipelines();
}

} // namespace bishengir

#endif // BISHENGIR_INITALLPASSES_H
