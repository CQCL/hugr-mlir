# RUN: %PYTHON %s | FileCheck %s

import gc
import hugr_mlir

from hugr_mlir.dialects import hugr
from hugr_mlir.ir import *

with Context() as ctx, Location.unknown():
    hugr.register_dialect(ctx, load=True)
    mod = hugr.ModuleOp(hugr.ModuleOp.build_generic(results=[], attributes={},operands=[],regions=1))
    mod.sym_name = StringAttr.get("python_module")
    Block.create_at_start(mod.body,[])
    mod.verify()
    print(mod)

# CHECK: hugr.module @python_module
