cmake_minimum_required(VERSION 3.10)

project(CppNet VERSION 0.1.0 LANGUAGES CXX CUDA)

# -------------------------------
# C++ Standard
# -------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -------------------------------
# Find Eigen
# -------------------------------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")

# -------------------------------
# Find OpenMP
# -------------------------------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
endif()

# -------------------------------
# Find CUDA Toolkit
# -------------------------------
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found, GPU support enabled")
    set(USE_CUDA TRUE)
else()
    message(STATUS "CUDA not found, building CPU-only version")
    set(USE_CUDA FALSE)
endif()

# -------------------------------
# Library Sources
# -------------------------------
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

if(USE_CUDA)
    list(APPEND CPP_NET_SOURCES src/kernels.cu)
endif()

# -------------------------------
# Create Library
# -------------------------------
add_library(CppNet STATIC ${CPP_NET_SOURCES})
target_include_directories(CppNet PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(CppNet PUBLIC Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CppNet PUBLIC OpenMP::OpenMP_CXX)
endif()
if(USE_CUDA)
    target_link_libraries(CppNet PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver)
    target_compile_definitions(CppNet PUBLIC USE_CUDA)
    set_target_properties(CppNet PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# -------------------------------
# Examples
# -------------------------------
add_executable(openmp_test examples/openmp_test.cpp)
add_executable(matmul_test examples/matmul_test.cpp)

# Link against your library
target_link_libraries(matmul_test PRIVATE CppNet)

if(CUDA_FOUND)
    target_compile_definitions(matmul_test PRIVATE USE_CUDA)
    target_link_libraries(matmul_test PRIVATE CUDA::cudart)
endif()


# -------------------------------
# Installation
# -------------------------------
install(TARGETS CppNet
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

# -------------------------------
# Build Summary
# -------------------------------
message(STATUS "CppNet Build Summary:")
message(STATUS "  - Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  - C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  - Eigen3: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  - OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "  - CUDA: ${USE_CUDA}")
