# CppNet Tests

This directory contains all tests for the CppNet library.

## Quick Testing Workflow

### Method 1: Fast Incremental Testing (Recommended during development)

Build and test a single component without rebuilding the entire library:

```bash
# From the tests directory
cd tests
mkdir -p build && cd build
cmake ..
make linear_test      # Only builds what's needed for this test
./linear_test         # Run the test
```

### Method 2: Build All Tests

```bash
# From project root
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make
ctest                 # Run all tests
```

### Method 3: Build Library Only (for releases)

```bash
# From project root
mkdir -p build && cd build
cmake ..              # BUILD_TESTS=OFF by default
make
sudo make install     # Install library system-wide
```

## Adding New Tests

To add a new test, simply add it to `tests/CMakeLists.txt`:

```cmake
add_cppnet_test(my_new_test my_new_test.cpp)
```

That's it! The helper function handles all the linking and configuration.

## Test Structure

```
tests/
├── CMakeLists.txt              # Test configuration
├── linear_test.cpp             # Linear layer tests
├── kernels/
│   └── gpu/
│       └── test_matmul_gpu.cu  # GPU kernel tests
└── [add more test files here]
```

## Running Specific Tests

```bash
# Run a specific test
./build/linear_test

# Run all tests with CTest
ctest

# Run tests with verbose output
ctest --verbose

# Run tests matching a pattern
ctest -R linear
```

## Tips for Fast Development

1. **During active development**: Build from `tests/` directory
   - Only rebuilds changed files
   - Much faster iteration
   - Automatically includes the library

2. **Before committing**: Build from root with all tests
   - Ensures everything works together
   - Catches integration issues

3. **For releases**: Build library only
   - No test overhead
   - Clean install