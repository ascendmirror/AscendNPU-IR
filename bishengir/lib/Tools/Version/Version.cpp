//===-- Version.cpp - BiShengIR Version Number ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines several version-related utility functions for Clang.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Version/Version.h"
#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Version/Version.inc"
#include "llvm/Support/raw_ostream.h"

#include "VCSVersion.inc"

namespace bishengir {

std::string getBiShengIRRepositoryPath() {
#ifdef BISHENGIR_REPOSITORY
  return BISHENGIR_REPOSITORY;
#else
  return "";
#endif
}

std::string getLLVMRepositoryPath() {
#ifdef LLVM_REPOSITORY
  return LLVM_REPOSITORY;
#else
  return "";
#endif
}

std::string getBiShengIRRevision() {
#ifdef BISHENGIR_REVISION
  return BISHENGIR_REVISION;
#else
  return "";
#endif
}

std::string getLLVMRevision() {
#ifdef LLVM_REVISION
  return LLVM_REVISION;
#else
  return "";
#endif
}

std::string getBiShengIRVendor() {
#ifdef BISHENGIR_VENDOR
  return BISHENGIR_VENDOR;
#else
  return "";
#endif
}

std::string getBiShengIRFullRepositoryVersion() {
  std::string buf;
  llvm::raw_string_ostream OS(buf);
  std::string Path = getBiShengIRRepositoryPath();
  std::string Revision = getBiShengIRRevision();
  if (!Path.empty() || !Revision.empty()) {
    OS << '(';
    if (!Path.empty())
      OS << Path;
    if (!Revision.empty()) {
      if (!Path.empty())
        OS << ' ';
      OS << Revision;
    }
    OS << ')';
  }
  // Support LLVM in a separate repository.
  std::string LLVMRev = getLLVMRevision();
  if (!LLVMRev.empty() && LLVMRev != Revision) {
    OS << " (";
    std::string LLVMRepo = getLLVMRepositoryPath();
    if (!LLVMRepo.empty())
      OS << LLVMRepo << ' ';
    OS << LLVMRev << ')';
  }
  return buf;
}

std::string getBiShengIRFullVersion() {
  return getBiShengIRToolFullVersion("bishengir");
}

/// Like getBiShengIRFullVersion(), but with a custom tool name.
std::string getBiShengIRToolFullVersion(llvm::StringRef ToolName) {
  std::string buf;
  llvm::raw_string_ostream OS(buf);
  OS << ToolName << " " << getBiShengIRVendor()
     << " version " BISHENGIR_VERSION_STRING;

  std::string repo = getBiShengIRFullRepositoryVersion();
  if (!repo.empty()) {
    OS << " " << repo;
  }

#if BISHENGIR_IS_DEBUG_BUILD
  OS << "\nDEBUG build";
#else
  OS << "\nOptimized build";
#endif
#ifndef NDEBUG
  OS << " with assertions.";
#endif
  OS << "\n";

  return buf;
}

} // namespace bishengir
