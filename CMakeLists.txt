cmake_minimum_required(VERSION 3.18)
project(CppNet VERSION 0.1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)

# CUDA detection and configuration
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(USE_CUDA TRUE)
    message(STATUS "CUDA found, GPU support enabled")
    
    # Set CUDA architectures policy and default values
    cmake_policy(SET CMP0104 NEW)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")
    endif()
else()
    set(USE_CUDA FALSE)
    message(STATUS "CUDA not found, building CPU-only version")
endif()

# Library Sources
set(CPP_NET_SOURCES
    src/activations.cpp
    src/data.cpp
    src/layers.cpp
    src/losses.cpp
    src/metrics.cpp
    src/models.cpp
    src/optimizers.cpp
    src/regularization.cpp
    src/utils.cpp
    src/visualization.cpp
    src/kernels_cpu.cpp
)

# Only add CUDA sources if CUDA is available
if(USE_CUDA)
    list(APPEND CPP_NET_SOURCES src/kernels.cu)
endif()

# Create Library
add_library(CppNet STATIC ${CPP_NET_SOURCES})
target_include_directories(CppNet PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(CppNet PUBLIC Eigen3::Eigen)

# OpenMP support
if(OpenMP_CXX_FOUND)
    target_link_libraries(CppNet PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(CppNet PUBLIC USE_OPENMP)
endif()

# CUDA support
if(USE_CUDA)
    target_link_libraries(CppNet PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_definitions(CppNet PUBLIC USE_CUDA)
    set_target_properties(CppNet PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_STANDARD 17
    )
endif()

# Examples
#add_executable(matmul_test examples/matmul_test.cpp)
#target_link_libraries(matmul_test PRIVATE CppNet)

#add_executable(main_test examples/main_test.cpp)
#target_link_libraries(main_test PRIVATE CppNet)
add_executable(linear_test examples/linear_test.cpp)
target_link_libraries(linear_test PRIVATE CppNet)

# Set CUDA properties for executables too
if(USE_CUDA)
    target_compile_definitions(linear_test PRIVATE USE_CUDA)
    set_target_properties(linear_test PROPERTIES 
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        LINKER_LANGUAGE CUDA
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    # Explicitly link CUDA libraries to the executable
    target_link_libraries(linear_test PRIVATE CUDA::cudart)
endif()

# Installation
install(TARGETS CppNet
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

# Build Summary
message(STATUS "CppNet Build Summary:")
message(STATUS " - Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS " - C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS " - Eigen3: ${EIGEN3_INCLUDE_DIR}")
message(STATUS " - OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS " - CUDA: ${USE_CUDA}")
if(USE_CUDA)
    message(STATUS " - CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()