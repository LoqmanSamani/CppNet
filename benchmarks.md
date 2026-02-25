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
