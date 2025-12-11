cmake_minimum_required(VERSION 3.18)

project(CppNet VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(USE_CUDA TRUE)
    message(STATUS "CUDA found, GPU support enabled")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")
    endif()
else()
    set(USE_CUDA FALSE)
    message(STATUS "CUDA not found, building CPU-only version")
endif()

option(BUILD_TESTS "Build test executables" ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)

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

add_library(CppNet STATIC ${CPP_NET_SOURCES})

set_target_properties(CppNet PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    PUBLIC_HEADER include/CppNet/CppNet.hpp
)

target_include_directories(CppNet PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

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


install(DIRECTORY include/CppNet
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT CppNetTargets
    FILE CppNetTargets.cmake
    NAMESPACE CppNet::
    DESTINATION lib/cmake/CppNet
)

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