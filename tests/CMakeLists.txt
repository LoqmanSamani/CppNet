cmake_minimum_required(VERSION 3.18)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)

    message(STATUS "Building from tests directory - standalone mode")
    
    project(CppNetTests LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    
   
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(USE_CUDA TRUE)
        message(STATUS "CUDA found - GPU tests enabled")
    else()
        set(USE_CUDA FALSE)
        message(STATUS "CUDA not found - GPU tests disabled")
    endif()
    
    
    add_subdirectory(.. ${CMAKE_CURRENT_BINARY_DIR}/CppNet)
else()
    
    message(STATUS "Building tests as part of main project")
endif()


function(add_cppnet_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} PRIVATE CppNet)
    
    if(USE_CUDA)
        target_compile_definitions(${test_name} PRIVATE USE_CUDA)
    endif()
    
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
   
    add_test(NAME ${test_name} COMMAND ${test_name})
    
    message(STATUS "Added test: ${test_name}")
endfunction()


add_cppnet_test(linear_test linear_test.cpp)
add_cppnet_test(init_test init_test.cpp)


#if(USE_CUDA)
#    add_cppnet_test(test_matmul_gpu kernels/gpu/test_matmul_gpu.cu)
#endif()

# add_cppnet_test(activation_test activation_test.cpp)
# add_cppnet_test(loss_test loss_test.cpp)
# add_cppnet_test(optimizer_test optimizer_test.cpp)


message(STATUS "")
message(STATUS "=== Test Configuration ===")
message(STATUS "Tests will be linked against CppNet library")
message(STATUS "Run tests with: ctest or make test")
message(STATUS "========================")