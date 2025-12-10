# Fast Testing Workflow Guide

## Setup (One-time)

1. **Create the tests/CMakeLists.txt file**
   
   Copy the provided `tests/CMakeLists.txt` content into `tests/CMakeLists.txt`

2. **Verify your test files exist**
   ```bash
   ls tests/linear_test.cpp
   ls tests/kernels/gpu/test_matmul_gpu.cu
   ```

## Daily Development Workflow

### Method 1: Build and test ONE component (FASTEST - ~5 seconds)

```bash
cd tests
mkdir -p build
cd build
cmake ..
make linear_test        # Only builds linear_test
./linear_test           # Run it
```

**When to use**: 
- Working on linear layer implementation
- Testing small changes
- Rapid iteration (change code → recompile → test)

**What happens**:
- CMake detects you're in tests/ directory
- Automatically includes parent (library)
- Only compiles what changed + the specific test
- ⚡ SUPER FAST - usually 5-15 seconds

### Method 2: Build ALL tests from tests directory

```bash
cd tests/build
cmake ..
make -j$(nproc)        # Builds all tests
ctest                  # Run all tests
```

**When to use**:
- Before committing changes
- Want to run full test suite
- Still faster than building from root

### Method 3: Add a new test (takes 10 seconds)

1. Create your test file: `tests/my_new_test.cpp`

2. Add ONE line to `tests/CMakeLists.txt`:
   ```cmake
   add_cppnet_test(my_new_test my_new_test.cpp)
   ```

3. Build it:
   ```bash
   cd tests/build
   cmake ..              # Picks up new test
   make my_new_test      # Build only this test
   ./my_new_test         # Run it
   ```

## Real-World Example

Let's say you're working on the sigmoid activation:

```bash
# Terminal 1: Your editor
vim src/CppNet/activations/sigmoid.cpp

# Terminal 2: Fast test cycle
cd tests/build

# First build
cmake ..
make sigmoid_test     # (after you added this test)
./sigmoid_test

# Make a change in sigmoid.cpp...
make sigmoid_test     # Rebuilds in ~3 seconds
./sigmoid_test

# Make another change...
make sigmoid_test     # Rebuilds in ~3 seconds
./sigmoid_test

# Happy with changes? Run all tests
make -j$(nproc)
ctest
```

## Comparison: Speed Test

**Testing linear layer changes:**

| Method | Time | Command |
|--------|------|---------|
| ❌ Root full rebuild | ~45s | `cd build && make` |
| ⚠️ Root with tests | ~60s | `cd build && make -DBUILD_TESTS=ON` |
| ✅ Tests directory | ~8s | `cd tests/build && make linear_test` |
| 🚀 Just recompile | ~3s | After first build, changes only |

**5-10x faster!**

## How It Works

The magic is in this part of `tests/CMakeLists.txt`:

```cmake
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    # We're in tests/ - include parent to build library
    add_subdirectory(.. ${CMAKE_CURRENT_BINARY_DIR}/CppNet)
```

**When you run from tests/**:
1. Detects you're building from tests directory
2. Includes parent CMakeLists.txt to build library
3. Only compiles changed files
4. Links your test against library
5. ⚡ Lightning fast

**When you run from root with `-DBUILD_TESTS=ON`**:
1. Builds entire library first
2. Then adds tests/ as subdirectory
3. Works normally, just slower

## Pro Tips

### 1. Keep test terminal open
```bash
cd tests/build
# Keep this terminal open all day
# Just run: make <test_name> whenever you change code
```

### 2. Use watch for auto-rebuild (optional)
```bash
# Install: sudo apt-get install inotify-tools
while inotifywait -e modify ../../src/CppNet/layers/linear.cpp; do
    make linear_test && ./linear_test
done
```

### 3. Test naming convention
```
linear_test.cpp         → tests Linear layer
activation_test.cpp     → tests all activations
sigmoid_test.cpp        → tests only sigmoid
loss_test.cpp           → tests loss functions
optimizer_sgd_test.cpp  → tests SGD optimizer
```

### 4. Multi-test for one component
```cmake
# In tests/CMakeLists.txt
add_cppnet_test(linear_basic_test linear_basic_test.cpp)
add_cppnet_test(linear_grad_test linear_grad_test.cpp)
add_cppnet_test(linear_perf_test linear_perf_test.cpp)
```

Then test individually:
```bash
make linear_basic_test && ./linear_basic_test
make linear_grad_test && ./linear_grad_test
```

## Troubleshooting

### "Target not found"
```bash
cd tests/build
rm -rf *
cmake ..
make linear_test
```

### Changes not reflected
```bash
# Force clean rebuild
cd tests/build
make clean
cmake ..
make linear_test
```

### Want to see what's being built
```bash
make linear_test VERBOSE=1
```

### Test linking errors
Make sure library compiled successfully:
```bash
cd tests/build
make CppNet  # Build library explicitly
make linear_test
```

## Adding GPU Tests

```cmake
# In tests/CMakeLists.txt
if(USE_CUDA)
    add_cppnet_test(my_gpu_test kernels/gpu/my_gpu_test.cu)
endif()
```

Then:
```bash
cd tests/build
cmake ..
make my_gpu_test
./my_gpu_test
```

## Summary

**For daily development**: 
```bash
cd tests/build && make <your_test>
```

**Before git commit**: 
```bash
cd tests/build && make -j$(nproc) && ctest
```

**For releases**: 
```bash
cd build && cmake .. && make -j$(nproc)
```

This workflow will save you HOURS of waiting for recompilation! 🚀