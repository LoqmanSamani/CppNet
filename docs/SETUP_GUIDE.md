# CppNet Setup Guide

## Directory Structure

First, create the cmake directory and config file:

```bash
# From project root
mkdir -p cmake
```

Then create the file `cmake/CppNetConfig.cmake.in` with the content provided in the artifacts.

## Building the Library

### 1. Library Only (for distribution/installation)

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

This installs CppNet system-wide at:
- Headers: `/usr/local/include/CppNet/`
- Library: `/usr/local/lib/libCppNet.a`
- CMake config: `/usr/local/lib/cmake/CppNet/`

### 2. Library with Tests (for development)

```bash
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
ctest  # Run all tests
```

### 3. Fast Test Development Workflow

When working on a specific test, build directly from tests directory:

```bash
cd tests
mkdir -p build && cd build
cmake ..
make linear_test  # Only builds this test
./linear_test
```

Changes to library source files will trigger recompilation, but only for the test you're building.

## Using CppNet in Your Projects

### Method 1: With CMake (Recommended)

```cmake
# In your CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(MyProject)

find_package(CppNet REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app CppNet::CppNet)
```

### Method 2: Direct Compilation

```bash
g++ -std=c++17 my_app.cpp -lCppNet -lgomp -o my_app
```

### Method 3: Manual Include (without installation)

```bash
g++ -std=c++17 -I/path/to/CppNet/include my_app.cpp \
    /path/to/CppNet/build/libCppNet.a -lgomp -o my_app
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | OFF | Build test executables |
| `CMAKE_BUILD_TYPE` | Release | Debug or Release |
| `USE_CUDA` | Auto | Detected automatically |
| `CMAKE_CUDA_ARCHITECTURES` | 60;70;75;80;86 | GPU compute capabilities |

### Examples:

```bash
# Debug build with tests
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON

# Specify GPU architecture
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;86"

# Force CPU-only (even if CUDA available)
# Currently auto-detected; modify CMakeLists.txt to add option
```

## Testing Workflow Comparison

### Scenario 1: Testing a single component during development
**Fastest**: Build from `tests/` directory
```bash
cd tests/build
make linear_test  # ~5 seconds
```

### Scenario 2: Testing after major changes
**Recommended**: Build from root with tests
```bash
cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
ctest
```

### Scenario 3: Preparing for release
**Clean**: Build library only
```bash
cd build
cmake ..
make -j$(nproc)
sudo make install
```

## Troubleshooting

### "Could not find CppNet"
Make sure you installed it: `sudo make install` from build directory.

### CUDA linking errors
The library was built without CUDA. Rebuild with CUDA toolkit installed.

### Test won't rebuild
From tests/build: `make clean && cmake .. && make`

### Want to uninstall
```bash
cd build
sudo make uninstall  # If target exists
# Or manually:
sudo rm -rf /usr/local/include/CppNet
sudo rm /usr/local/lib/libCppNet.a
sudo rm -rf /usr/local/lib/cmake/CppNet
```

## Development Tips

1. **Use the tests/ workflow** for rapid iteration on individual components
2. **Run full builds occasionally** to catch integration issues
3. **Keep tests small and focused** on single components
4. **Use CppNet.hpp** in your applications for cleaner includes
5. **Check library info** with `CppNet::version()` and `CppNet::has_cuda_support()`