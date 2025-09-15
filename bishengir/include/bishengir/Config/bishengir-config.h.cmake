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
 * \file bishengir-config.h.cmake
 * \brief BiShengIR Configuration
 */

/* This file enumerates variables from the BiShengIR configuration so that they
   can be in exported headers and won't override package specific directives.
   Defining the variables here is preferable over specifying them in CMake files
   via `target_compile_definitions` because it is easier to ensure that they are
   defined consistently across all targets: They are guaranteed to be 0/1
   variables thanks to #cmakedefine01, so we can test with `#if` and find
   missing definitions or includes with `-Wundef`. With `#ifdef`, these mistakes
   can go unnoticed.

   This is a C header that can be included in the bishengir-c headers. */

#ifndef BISHENGIR_CONFIG_H
#define BISHENGIR_CONFIG_H

/* Experimental feature. If set, only build the IR in a standalone manner. */
#cmakedefine01 BISHENGIR_BUILD_STANDALONE_IR_ONLY

#endif
