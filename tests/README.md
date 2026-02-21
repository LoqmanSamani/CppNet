# CppNet Tests

This directory contains comprehensive unit and integration tests for the CppNet library. It includes tests for layers, activations, losses, metrics, and GPU kernels.

## 🚀 Quick Start

Choose the workflow that matches your task:

| Task | Method | Time |
|------|--------|------|
| **Developing/Debugging** | Method 1 | ~10s |
| **Pre-commit validation** | Method 2 | ~30s |
| **Release build** | Method 3 | ~1m |

---

## Testing Methods

### Method 1: Fast Incremental Testing ⚡ (Recommended for Development)

**Use this when:** You're actively developing and want quick feedback on a specific test.

Build and run a single test component without rebuilding the entire library:

```bash
cd tests
mkdir -p build && cd build
cmake ..
make linear_test      # Only builds what's needed for this test
./linear_test         # Run the test
```

**Advantages:**
- Fastest turnaround (only rebuilds changed files)
- Perfect for iterative development
- Automatically includes the library

---

### Method 2: Build & Test Everything ✅ (Recommended Before Committing)

**Use this when:** You need to verify all tests pass together and catch integration issues.

```bash
cd /path/to/CppNet/root
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make
ctest                 # Run all tests at once
```

**Advantages:**
- Ensures all components work together
- Catches cross-module issues
- Validates the complete build

---

### Method 3: Build Library Only (For Releases)

**Use this when:** You're preparing a release or production deployment.

```bash
cd /path/to/CppNet/root
mkdir -p build && cd build
cmake ..              # BUILD_TESTS=OFF by default
make
sudo make install     # Install library system-wide
```

**Advantages:**
- No test overhead
- Cleaner production build
- Smaller binary footprint

---

## Adding New Tests

### Simple 3-Step Process

1. **Create your test file** in the `tests/` directory (e.g., `my_new_test.cpp`)

2. **Add to CMakeLists.txt**:
   ```cmake
   add_cppnet_test(my_new_test my_new_test.cpp)
   ```
   
3. **Done!** The helper function handles all linking and configuration automatically.

---

## Test Structure

```
tests/
├── CMakeLists.txt                    # Test configuration & helpers
├── linear_test.cpp                   # Linear layer tests
├── init_test.cpp                     # Initialization tests
├── usage_example.cpp                 # Usage demonstrations
├── kernels/
│   └── gpu/
│       └── test_matmul_gpu.cu        # GPU kernel tests (CUDA)
└── [add more test files here]
```

---

## Running Tests

### Run a Specific Test
```bash
./build/linear_test
```

### Run All Tests (from build directory)
```bash
ctest                 # Standard output
ctest --verbose       # Detailed output with each test
```

### Run Tests Matching a Pattern
```bash
ctest -R linear       # Only tests with "linear" in the name
ctest -R gpu          # Only GPU-related tests
```

### Run with Failure Details
```bash
ctest --output-on-failure    # Shows output only for failed tests
ctest -VV                    # Very verbose mode
```

---

## 💡 Best Practices for Development

### During Active Development
- ✅ Build from the `tests/` directory (Method 1)
- ✅ Only rebuild what changed
- ✅ Test frequently with quick feedback loops
- ❌ Don't rebuild from root repeatedly

### Before Committing or Opening a PR
- ✅ Run full test suite from root (Method 2)
- ✅ Ensure all tests pass
- ✅ Check integration between modules
- ✅ Verify GPU tests (if applicable)

### When Preparing a Release
- ✅ Use Method 3 for production build
- ✅ Verify library installs correctly
- ✅ Run one final full test pass
- ✅ Document any version changes

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **CMake not found** | Install cmake: `sudo apt install cmake` |
| **GPU tests fail** | Ensure CUDA is installed and compatible |
| **Build from wrong directory** | Always `cd tests/` for incremental builds |
| **Tests not found** | Run `cmake ..` after modifying CMakeLists.txt |