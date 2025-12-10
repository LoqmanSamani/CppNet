cmake_minimum_required(VERSION 3.18)

# Check for CUDA before declaring project languages
find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    project(CppNet VERSION 0.1.0 LANGUAGES CXX CUDA)
    set(USE_CUDA TRUE)
    message(STATUS "CUDA found, GPU support enabled")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")
    endif()
else()
    project(CppNet VERSION 0.1.0 LANGUAGES CXX)
    set(USE_CUDA FALSE)
    message(STATUS "CUDA not found, building CPU-only version")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Option to build tests (OFF by default for library-only builds)
option(BUILD_TESTS "Build test executables" OFF)

# Find packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)

# Library sources
set(CPU_SOURCES
    src/CppNet/activations/sigmoid.cpp
    src/CppNet/activations/relu.cpp
    src/CppNet/layers/linear.cpp
    src/CppNet/losses/binary_cross_entropy.cpp
    src/CppNet/losses/categorical_cross_entropy.cpp
    src/CppNet/metrics/metrics.cpp
    src/CppNet/models/models.cpp
    src/CppNet/optimizers/sgd.cpp
    src/CppNet/regularizations/regularizations.cpp
    src/CppNet/utils/utils.cpp
    src/CppNet/visualizations/visualizations.cpp
)

set(GPU_SOURCES
    src/CppNet/kernels/gpu/add_bias.cu
    src/CppNet/kernels/gpu/bias_grad.cu
    src/CppNet/kernels/gpu/elementwise.cu
    src/CppNet/kernels/gpu/matmul.cu
    src/CppNet/kernels/gpu/matmul_grad_input.cu
    src/CppNet/kernels/gpu/matmul_grad_weight.cu
)

set(CPP_NET_SOURCES ${CPU_SOURCES})
if(USE_CUDA)
    list(APPEND CPP_NET_SOURCES ${GPU_SOURCES})
endif()

# Create CppNet library
add_library(CppNet STATIC ${CPP_NET_SOURCES})

# Set library properties
set_target_properties(CppNet PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    PUBLIC_HEADER include/CppNet/CppNet.hpp
)

# Include directories
target_include_directories(CppNet PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(CppNet PUBLIC Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(CppNet PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(CppNet PUBLIC USE_OPENMP)
endif()

if(USE_CUDA)
    target_link_libraries(CppNet PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_definitions(CppNet PUBLIC USE_CUDA)
    set_target_properties(CppNet PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endif()

# Add tests subdirectory if BUILD_TESTS is ON
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install library
install(TARGETS CppNet
    EXPORT CppNetTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/CppNet
)

# Install headers
install(DIRECTORY include/CppNet
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Export targets for find_package
install(EXPORT CppNetTargets
    FILE CppNetTargets.cmake
    NAMESPACE CppNet::
    DESTINATION lib/cmake/CppNet
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CppNetConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CppNetConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/CppNetConfig.cmake"
    INSTALL_DESTINATION lib/cmake/CppNet
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CppNetConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CppNetConfigVersion.cmake"
    DESTINATION lib/cmake/CppNet
)