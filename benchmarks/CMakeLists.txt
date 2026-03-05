# =============================================================================
# CppNet Benchmarks
# =============================================================================

function(cppnet_add_benchmark BENCH_NAME BENCH_SOURCE)
    add_executable(${BENCH_NAME} ${BENCH_SOURCE})
    target_link_libraries(${BENCH_NAME} PRIVATE CppNet)
endfunction()

cppnet_add_benchmark(mlp_benchmark         mlp_benchmark.cpp)
cppnet_add_benchmark(cnn_benchmark         cnn_benckmark.cpp)
cppnet_add_benchmark(sequence_benchmark    sequence_benckmark.cpp)
cppnet_add_benchmark(transformer_benchmark transformer_benckmark.cpp)
cppnet_add_benchmark(residual_benchmark    residual_benckmark.cpp)

# mlp_benchmark uses matplotlib-cpp, which requires Python3 + NumPy
target_include_directories(mlp_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mlp_benchmark PRIVATE Python3::Python Python3::NumPy)
