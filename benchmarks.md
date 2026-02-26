# CppNet Benchmarks

Detailed performance benchmarks for the CppNet deep learning library.
All benchmarks are reproducible via the scripts in the `examples/` directory.

---

## Table of Contents

- [MLP Device Benchmark — Spiral Classification](#mlp-device-benchmark--spiral-classification)
  - [Experiment Setup](#experiment-setup)
  - [Network Configurations](#network-configurations)
  - [Results Summary](#results-summary)
  - [Detailed Per-Config Results](#detailed-per-config-results)
  - [Speedup Analysis](#speedup-analysis)
  - [Accuracy Convergence](#accuracy-convergence)
  - [Key Takeaways](#key-takeaways)
  - [How to Reproduce](#how-to-reproduce)
- [CNN Device Benchmark — Image Classification](#cnn-device-benchmark--image-classification)
  - [Experiment Setup](#experiment-setup-1)
  - [Network Configurations](#network-configurations-1)
  - [Results Summary](#results-summary-1)
  - [Detailed Per-Config Results](#detailed-per-config-results-1)
  - [Speedup Analysis](#speedup-analysis-1)
  - [Accuracy Convergence](#accuracy-convergence-1)
  - [Key Takeaways](#key-takeaways-1)
  - [How to Reproduce](#how-to-reproduce-1)

---

## MLP Device Benchmark — Spiral Classification

**Source:** [`examples/spiral_classification.cpp`](examples/spiral_classification.cpp)  
**Date:** February 25, 2026

### Experiment Setup

| Parameter | Value |
|:----------|:------|
| **Dataset** | 2D spiral, synthetically generated |
| **Samples** | 15,000 (3,000 per class) |
| **Classes** | 5 |
| **Features** | 2 (x, y coordinates) |
| **Noise** | 0.05 |
| **Loss function** | Softmax Cross-Entropy (mean reduction) |
| **Optimizer** | Adam (β₁=0.9, β₂=0.999, ε=1e-10) |
| **Activations** | ReLU (between all hidden layers) |
| **Weight init** | Xavier |
| **Random seed** | 42 (data), 123 (training) |
| **OpenMP threads** | 4 (for `cpu` and `cpu-eigen` backends) |

#### Backends Tested

| Backend | Description |
|:--------|:------------|
| `cpu-eigen` | Eigen tensor contractions — highly optimized, SIMD-vectorized |
| `cpu` | Manual OpenMP-parallelized loops (4 threads) |
| `gpu` | CUDA kernels (matmul, bias, ReLU, SGD update) |

#### Hardware

> *(Replace with your actual hardware when sharing results.)*

The benchmark was run on a single machine with an NVIDIA GPU and a multi-core CPU.

---

### Network Configurations

Four MLP architectures of increasing width are tested. All use ReLU activations between hidden layers and Softmax Cross-Entropy as the output loss.

| Config | Architecture | Hidden Layers | Total Parameters (approx.) | Epochs | Batch Size | Learning Rate |
|:-------|:-------------|:--------------|:---------------------------|:-------|:-----------|:-------------|
| **Small** | 2→64→64→5 | 2 | ~4,500 | 50 | 128 | 0.001 |
| **Medium** | 2→128→256→128→5 | 3 | ~66,000 | 40 | 256 | 0.0005 |
| **Large** | 2→256→512→512→256→5 | 4 | ~660,000 | 10 | 256 | 0.0005 |
| **XLarge** | 2→512→1024→1024→512→5 | 4 | ~2,600,000 | 10 | 512 | 0.0003 |

---

### Results Summary

| Config | Architecture | cpu-eigen | cpu (OpenMP) | gpu (CUDA) | GPU Speedup vs cpu-eigen |
|:-------|:-------------|:----------|:-------------|:-----------|:-------------------------|
| Small | 2→64→64→5 | 7.4 s | 17.9 s | 3.6 s | **2.03x** |
| Medium | 2→128→256→128→5 | 61.6 s | 241.9 s | 8.9 s | **6.91x** |
| Large | 2→256→512→512→256→5 | 114.5 s | 669.9 s | 8.0 s | **14.31x** |
| XLarge | 2→512→1024→1024→512→5 | 473.1 s | 3,027.0 s | 18.7 s | **25.28x** |

---

### Detailed Per-Config Results

#### Small (2→64→64→5) — 50 epochs, batch 128, lr=0.001

| Device | Epoch 1 Loss | Epoch 1 Acc | Epoch 20 Loss | Epoch 20 Acc | Epoch 40 Loss | Epoch 40 Acc | Final Loss | Final Acc | Total Time |
|:-------|:-------------|:------------|:--------------|:-------------|:--------------|:-------------|:-----------|:----------|:-----------|
| cpu-eigen | 1.3704 | 33.73% | 0.2123 | 92.53% | 0.1985 | 92.66% | 0.1952 | 92.69% | 7.37 s |
| cpu | 1.3420 | 33.71% | 0.2108 | 92.59% | 0.1976 | 92.57% | 0.1945 | 92.68% | 17.88 s |
| gpu | 1.3604 | 34.20% | 0.2125 | 92.60% | 0.1971 | 92.63% | 0.1947 | 92.63% | 3.64 s |

#### Medium (2→128→256→128→5) — 40 epochs, batch 256, lr=0.0005

| Device | Epoch 1 Loss | Epoch 1 Acc | Epoch 20 Loss | Epoch 20 Acc | Final Loss | Final Acc | Total Time |
|:-------|:-------------|:------------|:--------------|:-------------|:-----------|:----------|:-----------|
| cpu-eigen | 1.3909 | 32.75% | 0.2003 | 92.68% | 0.1939 | 92.58% | 61.58 s |
| cpu | 1.3945 | 32.89% | 0.2011 | 92.70% | 0.1941 | 92.55% | 241.88 s |
| gpu | 1.4066 | 32.77% | 0.1999 | 92.64% | 0.1933 | 92.52% | 8.92 s |

#### Large (2→256→512→512→256→5) — 10 epochs, batch 256, lr=0.0005

| Device | Epoch 1 Loss | Epoch 1 Acc | Final Loss | Final Acc | Total Time |
|:-------|:-------------|:------------|:-----------|:----------|:-----------|
| cpu-eigen | 1.1732 | 41.62% | 0.2042 | 91.92% | 114.54 s |
| cpu | 1.1680 | 42.21% | 0.2065 | 91.77% | 669.87 s |
| gpu | 1.2074 | 39.62% | 0.2058 | 91.82% | 8.01 s |

#### XLarge (2→512→1024→1024→512→5) — 10 epochs, batch 512, lr=0.0003

| Device | Epoch 1 Loss | Epoch 1 Acc | Final Loss | Final Acc | Total Time |
|:-------|:-------------|:------------|:-----------|:----------|:-----------|
| cpu-eigen | 1.3246 | 32.71% | 0.2025 | 92.34% | 473.12 s |
| cpu | 1.3282 | 32.90% | 0.2035 | 92.34% | 3,027.00 s |
| gpu | 1.3505 | 32.38% | 0.2025 | 92.28% | 18.72 s |

---

### Speedup Analysis

Speedups are measured relative to the `cpu-eigen` baseline (the fastest CPU backend).

| Config | cpu-eigen | cpu (OpenMP) vs cpu-eigen | gpu (CUDA) vs cpu-eigen |
|:-------|:----------|:--------------------------|:------------------------|
| Small | 1.00x (baseline) | 0.41x (slower) | **2.03x** |
| Medium | 1.00x (baseline) | 0.25x (slower) | **6.91x** |
| Large | 1.00x (baseline) | 0.17x (slower) | **14.31x** |
| XLarge | 1.00x (baseline) | 0.16x (slower) | **25.28x** |

#### Observations

- **Eigen (`cpu-eigen`) significantly outperforms manual OpenMP (`cpu`) on all sizes.**  
  Eigen's SIMD-vectorized tensor contractions are far more efficient than the naive OpenMP loop implementation. The OpenMP backend is 2.4x–6.4x *slower* than Eigen.

- **GPU speedup scales with network width.**  
  For small networks (64-wide), GPU overhead limits the speedup to ~2x. As hidden layer widths grow to 512–1024, GPU throughput dominates and the speedup reaches **25x** over Eigen.

- **GPU time grows sub-linearly with parameters.**  
  From Small (~4.5k params) to XLarge (~2.6M params) — a ~578x increase in parameters — GPU time only increases from 3.6 s to 18.7 s (5.2x), demonstrating excellent GPU utilization at scale.

---

### Accuracy Convergence

All three backends converge to similar final accuracy (~92–93%) for each configuration, confirming numerical consistency across devices.

| Config | cpu-eigen | cpu | gpu |
|:-------|:----------|:----|:----|
| Small | 92.69% | 92.68% | 92.63% |
| Medium | 92.58% | 92.55% | 92.52% |
| Large | 91.92% | 91.77% | 91.82% |
| XLarge | 92.34% | 92.34% | 92.28% |

Minor variations are expected due to floating-point non-associativity across different computation orders.

---

### Key Takeaways

1. **Use `cpu-eigen` for small models.** Eigen's optimized contractions beat manual OpenMP loops and have lower overhead than GPU kernels for small matrices.

2. **Use `gpu` for medium-to-large models.** Once hidden widths reach 128+, CUDA kernels deliver substantial speedups (7x–25x) that grow with network size.

3. **The manual OpenMP backend (`cpu`) is currently slower than Eigen.** This is expected — Eigen leverages SIMD intrinsics and cache-optimal memory layouts that a simple parallel-for loop cannot match. The `cpu` backend exists as a reference implementation and a foundation for future optimization.

4. **GPU advantage is most dramatic for large matrix multiplications.** The dominant operation is `B×H * H×H` (batch × hidden × hidden). Larger batches and wider layers produce bigger matrices that better saturate GPU cores.

---

### How to Reproduce

```bash
git clone https://github.com/LoqmanSamani/CppNet.git
cd CppNet
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
make -j$(nproc)
./examples/spiral_classification
```

Ensure CUDA is installed and detected by CMake for GPU results. To run CPU-only:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON -DCUDAToolkit_ROOT=/nonexistent
```

---

## CNN Device Benchmark — Image Classification

**Source:** [`examples/cnn_benckmark.cpp`](examples/cnn_benckmark.cpp)  
**Date:** February 27, 2026

### Experiment Setup

| Parameter | Value |
|:----------|:------|
| **Dataset** | Synthetic CIFAR-10-shaped images (sinusoidal patterns + noise) |
| **Samples** | 1,000 (100 per class) |
| **Classes** | 10 |
| **Image size** | 3×32×32 (channels × height × width) |
| **Noise** | 0.1 |
| **Loss function** | Softmax Cross-Entropy (mean reduction) |
| **Optimizer** | Adam (β₁=0.9, β₂=0.999, ε=1e-10) |
| **Activations** | ReLU (after each conv and FC hidden layer) |
| **Pooling** | MaxPool2D (2×2) |
| **Weight init** | Xavier |
| **Random seed** | 42 (data), 123 (training) |
| **OpenMP threads** | 4 (for `cpu` and `cpu-eigen` backends) |

#### Backends Tested

| Backend | Description |
|:--------|:------------|
| `cpu-eigen` | Eigen tensor operations — optimized, SIMD-vectorized |
| `cpu` | Manual OpenMP-parallelized loops (4 threads) |
| `gpu` | CUDA kernels (conv2d forward/backward, maxpool2d forward/backward, matmul, bias, ReLU) |

#### Hardware

Same machine as the MLP benchmark: NVIDIA GeForce GTX 1650 (4 GB), multi-core CPU.

---

### Network Configurations

Two CNN architectures are tested. Both use ReLU activations after each conv layer and Softmax Cross-Entropy as the output loss.

| Config | Architecture | Conv Layers | FC Layers | Epochs | Batch Size | Learning Rate |
|:-------|:-------------|:------------|:----------|:-------|:-----------|:-------------|
| **Small** | Conv(3→16,k3,p1)→Pool2→Conv(16→32,k3,p1)→Pool2→Flat→10 | 2 | 1 (output only) | 3 | 32 | 0.001 |
| **Medium** | Conv(3→32,k3,p1)→Pool2→Conv(32→64,k3,p1)→Pool2→Flat→128→10 | 2 | 2 (128 hidden + output) | 2 | 32 | 0.0005 |

---

### Results Summary

| Config | Architecture | cpu-eigen | cpu (OpenMP) | gpu (CUDA) | GPU Speedup vs cpu-eigen |
|:-------|:-------------|:----------|:-------------|:-----------|:-------------------------|
| Small | Conv16→Conv32→FC10 | 84.8 s | 91.0 s | 2.9 s | **28.82x** |
| Medium | Conv32→Conv64→FC128→FC10 | 252.3 s | 277.4 s | 6.0 s | **41.97x** |

---

### Detailed Per-Config Results

#### Small (Conv(3→16)→Pool→Conv(16→32)→Pool→Flat→10) — 3 epochs, batch 32, lr=0.001

| Device | Epoch 1 Loss | Epoch 1 Acc | Final Loss | Final Acc | Total Time |
|:-------|:-------------|:------------|:-----------|:----------|:-----------|
| cpu-eigen | 0.9422 | 83.47% | 0.0049 | 100.00% | 84.78 s |
| cpu | 0.9299 | 84.38% | 0.0036 | 100.00% | 91.04 s |
| gpu | 1.0123 | 77.62% | 0.0053 | 100.00% | 2.94 s |

#### Medium (Conv(3→32)→Pool→Conv(32→64)→Pool→Flat→128→10) — 2 epochs, batch 32, lr=0.0005

| Device | Epoch 1 Loss | Epoch 1 Acc | Final Loss | Final Acc | Total Time |
|:-------|:-------------|:------------|:-----------|:----------|:-----------|
| cpu-eigen | 0.7907 | 83.77% | 0.0057 | 100.00% | 252.32 s |
| cpu | 0.8067 | 84.38% | 0.0070 | 100.00% | 277.37 s |
| gpu | 0.8434 | 80.44% | 0.0081 | 100.00% | 6.01 s |

---

### Speedup Analysis

Speedups are measured relative to the `cpu-eigen` baseline.

| Config | cpu-eigen | cpu (OpenMP) vs cpu-eigen | gpu (CUDA) vs cpu-eigen |
|:-------|:----------|:--------------------------|:------------------------|
| Small | 1.00x (baseline) | 0.93x (slightly slower) | **28.82x** |
| Medium | 1.00x (baseline) | 0.91x (slightly slower) | **41.97x** |

#### Observations

- **GPU speedup for CNNs is dramatically higher than for MLPs.** The Small CNN achieves ~29x speedup vs. ~2x for the Small MLP. Medium CNN reaches ~42x vs. ~7x for Medium MLP. This reflects the massive parallelism available in convolution operations.

- **Eigen and OpenMP are roughly equivalent for CNN workloads.** Unlike in the MLP benchmark where Eigen was 2–6x faster than OpenMP, for CNN operations the gap narrows to only ~7–9%. This suggests the bottleneck is in the conv2d loops rather than matrix multiplications where Eigen excels.

- **GPU time scales sub-linearly with compute.** Medium has ~4x more conv filters than Small, but GPU time only doubles (2.9s → 6.0s), showing excellent GPU utilization for larger convolution workloads.

- **All backends converge to 100% accuracy** on this synthetic dataset within very few epochs, confirming numerical correctness of the new CUDA Conv2D and MaxPool2D kernels.

---

### Accuracy Convergence

All three backends converge to identical final accuracy, confirming numerical consistency of the new GPU kernels.

| Config | cpu-eigen | cpu | gpu |
|:-------|:----------|:----|:----|
| Small | 100.00% | 100.00% | 100.00% |
| Medium | 100.00% | 100.00% | 100.00% |

---

### Key Takeaways

1. **CUDA Conv2D and MaxPool2D kernels provide massive speedups.** The new GPU kernels for convolution and pooling operations deliver 29x–42x speedups over CPU backends, significantly more than the MLP GPU speedups.

2. **Convolution operations are ideal for GPU acceleration.** The inherent parallelism in sliding-window convolution maps perfectly to GPU thread blocks, yielding higher speedup ratios than dense matrix multiplications.

3. **CPU backends perform similarly for CNN workloads.** Unlike MLPs where Eigen dominated, both CPU backends show comparable CNN performance since the convolution loops are implemented similarly.

4. **GPU kernels are numerically correct.** All backends converge to the same accuracy, validating the new `conv2d_forward`, `conv2d_backward`, `maxpool2d_forward`, and `maxpool2d_backward` CUDA kernels.

---

### How to Reproduce

```bash
git clone https://github.com/LoqmanSamani/CppNet.git
cd CppNet
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
make -j$(nproc)
./examples/cnn_benchmark
```

Ensure CUDA is installed and detected by CMake for GPU results.
