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
config.name = "hugr-mlir"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.hugr_mlir_obj_root, "mlir/test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

# This allows us to override PYTHONPATH from the commandline
pythonpath_extensions = [
    os.path.join(config.mlir_obj_root, "python_packages", "mlir_core"),
]
pythonpath_extensions_override = os.environ.get("HUGR_MLIR_PYTHONPATH_OVERRIDE", None)
if pythonpath_extensions_override is not None:
    pythonpath_extensions.extend(pythonpath_extensions_override.split(":"))
else:
    pythonpath_extensions.extend([config.hugr_mlir_python_packages_dir])


llvm_config.with_environment(
    "PYTHONPATH",
    pythonpath_extensions,
    append_path=True,
)

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.hugr_mlir_obj_root, "test")
config.hugr_mlir_libs_dir = os.path.join(config.hugr_mlir_obj_root, "lib")

# This allows us to override where to find hugr-mlir-opt from the command line
# I expect lit does have this feature, but I can't find it.
# In particular this is useful to excercise the hugr-mlir-opt in the install tree
config.hugr_mlir_tools_dir = os.environ.get("HUGR_MLIR_TOOLS_DIR", config.hugr_mlir_tools_dir)

tool_dirs = [config.hugr_mlir_tools_dir, config.llvm_tools_dir]
tools = [
    "hugr-mlir-opt",
    "test-hugr-mlir-capi",
    "test-hugr-rs-bridge",
    ToolSubst("%PYTHON", config.python_executable),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
