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
)

mlir_check_all_link_libraries(HugrMLIROpt)
