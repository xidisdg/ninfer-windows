ninfer_add_test(ninfer_artifact_reader_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_reader.cpp"
  LIBRARIES ninfer_artifact)

if(MSVC)
  # MSVC link.exe has no --wrap, so the CUDA fault-injection shims are not reachable there.
  ninfer_add_test(ninfer_artifact_materialization_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_materialization.cpp"
    LIBRARIES ninfer_artifact)
else()
  ninfer_add_test(ninfer_artifact_materialization_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_materialization.cpp"
              "${CMAKE_CURRENT_LIST_DIR}/materialization_cuda_errors.cpp"
    LIBRARIES ninfer_artifact)

  target_link_options(ninfer_artifact_materialization_test PRIVATE
    "LINKER:--wrap=cudaMalloc"
    "LINKER:--wrap=cudaMallocHost"
    "LINKER:--wrap=cudaFree"
    "LINKER:--wrap=cudaFreeHost"
    "LINKER:--wrap=cudaEventCreateWithFlags"
    "LINKER:--wrap=cudaEventRecord"
    "LINKER:--wrap=cudaMemcpyAsync"
    "LINKER:--wrap=cudaStreamSynchronize")
endif()

add_test(NAME ninfer_artifact_writer_interop_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/writer_interop.py"
    $<TARGET_FILE:ninfer_artifact_materialization_test>)

set_tests_properties(
  ninfer_artifact_writer_interop_test
  PROPERTIES SKIP_RETURN_CODE 77)

set_tests_properties(
  ninfer_artifact_materialization_test
  PROPERTIES SKIP_RETURN_CODE 77)
