cmake_minimum_required(VERSION 3.10)

# project name and version
project(CppNet VERSION 0.1.0 LANGUAGES CXX)

# set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


#find_package(Eigen3 REQUIRED)
#target_link_libraries(CppNet PRIVATE Eigen3::Eigen)

# find Eigen3 (required for linear algebra)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Eigen3 not found. Please install Eigen3 or set EIGEN3_INCLUDE_DIR.")
endif()

# find OpenMP for parallelization (optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
endif()

# define the source files for the CppNet library
set(CPP_NET_SOURCES
    src/activations.cpp
    src/data.cpp
    src/layers.cpp
    src/losses.cpp
    src/metrics.cpp
    src/models.cpp
    src/optimizers.cpp
    src/regularizations.cpp
    src/utils.cpp
    src/visualizations.cpp
)

# define the include directories
set(CPP_NET_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
)

# create the CppNet library (static library)
add_library(CppNet STATIC ${CPP_NET_SOURCES})

# link dependencies to the CppNet library
target_link_libraries(CppNet PUBLIC Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CppNet PUBLIC OpenMP::OpenMP_CXX)
endif()

# set include directories for the library
target_include_directories(CppNet PUBLIC ${CPP_NET_INCLUDE_DIRS})

# enable optimization flags for release build
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(CppNet PRIVATE -O3)
endif()

# define the example programs
set(EXAMPLES
    #transformer_example
    #att_example
    #dropout_example
    #openmp_test
    linear_test
)

# build each example program
foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} examples/${EXAMPLE}.cpp)
    target_link_libraries(${EXAMPLE} PRIVATE CppNet)
    target_include_directories(${EXAMPLE} PRIVATE ${CPP_NET_INCLUDE_DIRS})
endforeach()

# enable warnings and treat them as errors 
target_compile_options(CppNet PRIVATE -Wall -Wextra -Wpedantic)
foreach(EXAMPLE ${EXAMPLES})
    target_compile_options(${EXAMPLE} PRIVATE -Wall -Wextra -Wpedantic)
endforeach()


install(TARGETS CppNet
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

# print a summary of the build configuration
message(STATUS "CppNet Build Summary:")
message(STATUS "  - Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  - C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  - Eigen3: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  - OpenMP: ${OpenMP_CXX_FOUND}")


target_sources(CppNet PRIVATE src/layers.cpp src/optimizers.cpp)