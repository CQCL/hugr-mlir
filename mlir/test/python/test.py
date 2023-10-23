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
    set1 = hugr.ExtensionSetAttr.get()
    print(set1)
    ext1 = hugr.ExtensionAttr.get("ext1")
    print(ext1)
    set2 = hugr.ExtensionSetAttr.get([ext1])
    print(set2)

    fun_ty = hugr.FunctionType.get(set2, FunctionType.get([],[]))
    print(fun_ty)

# CHECK-LABEL: hugr.module @python_module
# CHECK: #hugr<exts[]>
# CHECK: #hugr<ext"ext1">
# CHECK: #hugr<exts["ext1"]>
# CHECK: !hugr<function ["ext1"]() -> ()>
