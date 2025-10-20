//===-- Version.h - BiShengIR Version Number --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines version macros and version-related utility functions
/// for BiShengIR.
///
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_VERSION_VERSION_H
#define BISHENGIR_VERSION_VERSION_H

#include "llvm/ADT/StringRef.h"

namespace bishengir {

// /// Retrieves a string representing the complete BiShengIR version, which
// /// includes the bishengir version number, as well as embedded compiler
// versions
// /// and the vendor tag.
// const char *GetVersion();

/// Retrieves the repository path (e.g., Subversion path) that
/// identifies the particular BiShengIR branch, tag, or trunk from which this
/// BiShengIR was built.
std::string getBiShengIRRepositoryPath();

/// Retrieves the repository path from which LLVM was built.
///
/// This supports LLVM residing in a separate repository from BiShengIR.
std::string getLLVMRepositoryPath();

/// Retrieves the repository revision number (or identifier) from which
/// this BiShengIR was built.
std::string getBiShengIRRevision();

/// Retrieves the repository revision number (or identifier) from which
/// LLVM was built.
///
/// If BiShengIR and LLVM are in the same repository, this returns the same
/// string as getBiShengIRRevision.
std::string getLLVMRevision();

/// Retrieves the BiShengIR vendor tag.
std::string getBiShengIRVendor();

/// Retrieves a string representing the complete bishengir version,
/// which includes the bishengir version number, the repository version,
/// and the vendor tag.
std::string getBiShengIRFullVersion();

/// Like getBiShengIRFullVersion(), but with a custom tool name.
std::string getBiShengIRToolFullVersion(llvm::StringRef toolName);

} // namespace bishengir

#endif // BISHENGIR_VERSION_VERSION_H
