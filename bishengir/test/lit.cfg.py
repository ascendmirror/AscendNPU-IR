# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "BiShengIR"

# We prefer the lit internal shell which provides a better user experience on failures
# unless the user explicitly disables it with LIT_USE_INTERNAL_SHELL=0 env var.
use_lit_shell = True
lit_shell_env = os.environ.get("LIT_USE_INTERNAL_SHELL")
if lit_shell_env:
  use_lit_shell = not lit.util.pythonize_bool(lit_shell_env)

config.test_format = lit.formats.ShTest(execute_external=not use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [
    ".td",
    ".mlir",
    ".toy",
    ".ll",
    ".tc",
    ".py",
    ".yaml",
    ".test",
    ".pdll",
    ".c",
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.bishengir_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%host_cxx", config.host_cxx))
config.substitutions.append(("%host_cc", config.host_cc))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
]

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.mlir_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.bishengir_tools_dir, append_path=True)

tool_dirs = [config.mlir_tools_dir, config.llvm_tools_dir, config.bishengir_tools_dir]
tools = [
    "bishengir-opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# FileCheck -enable-var-scope is enabled by default in MLIR test
# This option avoids to accidentally reuse variable across -LABEL match,
# it can be explicitly opted-in by prefixing the variable name with $
config.environment["FILECHECK_OPTS"] = "-enable-var-scope --allow-unused-prefixes=false"
