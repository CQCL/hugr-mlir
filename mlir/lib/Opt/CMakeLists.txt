add_mlir_library(HugrMLIROpt
    HugrOpt.cpp

    LINK_LIBS PUBLIC
    HugrMLIRDialect
    HugrMLIRAnalysis
    HugrMLIRTransforms
    HugrMLIRConversion
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRLLVMDialect
    MLIRFuncDialect
    MLIROptLib
    MLIRSCFToControlFlow
    MLIRControlFlowToLLVM
    MLIRMathToLLVM
    MLIRIndexToLLVM
    MLIRArithToLLVM
    MLIRFuncToLLVM
    MLIRUBToLLVM
    MLIRMemRefToLLVM
)

mlir_check_all_link_libraries(HugrMLIROpt)
