set(_doc_exclude_all "EXCLUDE_FROM_ALL")
if(HUGR_MLIR_INCLUDE_DOCS)
  set(_doc_exclude_all)
endif()

install(DIRECTORY .
    TYPE DOC
    ${_doc_exclude_all}
    FILES_MATCHING PATTERN "*.md"
)
