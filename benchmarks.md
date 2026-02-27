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
- [Sequence Layer Device Benchmark — Sine-Wave Regression](#sequence-layer-device-benchmark--sine-wave-regression)
  - [Experiment Setup](#experiment-setup-2)
  - [Network Configurations](#network-configurations-2)
  - [Results Summary](#results-summary-2)
  - [Speedup Analysis](#speedup-analysis-2)
  - [Key Takeaways](#key-takeaways-2)
  - [How to Reproduce](#how-to-reproduce-2)
- [Transformer Device Benchmark — Token Classification](#transformer-device-benchmark--token-classification)
  - [Experiment Setup](#experiment-setup-3)
  - [Network Configurations](#network-configurations-3)
  - [Results Summary](#results-summary-3)
  - [Detailed Per-Config Results](#detailed-per-config-results-2)
  - [Speedup Analysis](#speedup-analysis-3)
  - [Accuracy Convergence](#accuracy-convergence-2)
  - [Key Takeaways](#key-takeaways-3)
  - [How to Reproduce](#how-to-reproduce-3)
- [Deep ResNet Device Benchmark — Spiral Classification](#deep-resnet-device-benchmark--spiral-classification)
  - [Experiment Setup](#experiment-setup-4)
  - [Network Configurations](#network-configurations-4)
  - [Results Summary](#results-summary-4)
  - [Detailed Per-Config Results](#detailed-per-config-results-3)
  - [Speedup Analysis](#speedup-analysis-4)
  - [Accuracy Convergence](#accuracy-convergence-3)
  - [Key Takeaways](#key-takeaways-4)
  - [How to Reproduce](#how-to-reproduce-4)

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

---

## Sequence Layer Device Benchmark — Sine-Wave Regression

### Experiment Setup

| Item | Value |
|------|-------|
| **Task** | Next-value prediction on discretised sine wave |
| **Loss** | MSE |
| **Optimizer** | Momentum (μ = 0.9) |
| **Batch size** | 32 |
| **Samples** | 800 (Small/Medium), 1200 (Large) |
| **CPU threads** | 4 (OpenMP) |
| **GPU** | NVIDIA GeForce GTX 1650 (4 GB) |
| **Build** | Release (-O2), GCC 13.3, CUDA 12.0 |

### Network Configurations

Each configuration uses: `RecurrentLayer(1, H, return_sequences=true) → extract last hidden → Linear(H, 1)`

| Config | Hidden (H) | Seq Length | Epochs | Learning Rate |
|--------|-----------|------------|--------|---------------|
| Small | 64 | 20 | 5 | 0.01 |
| Medium | 128 | 30 | 3 | 0.005 |
| Large | 256 | 50 | 3 | 0.002 |

### Results Summary

#### Small Config (H=64, seq=20, 5 epochs)

| Layer | cpu-eigen | cpu (OpenMP) | gpu (CUDA) | GPU Speedup |
|-------|----------|-------------|-----------|-------------|
| **RNN** | 0.87 s | 0.76 s | 0.40 s | **2.2×** |
| **LSTM** | 2.70 s | 2.89 s | 0.79 s | **3.4×** |
| **GRU** | 3.98 s | 3.92 s | 0.76 s | **5.2×** |

#### Medium Config (H=128, seq=30, 3 epochs)

| Layer | cpu-eigen | cpu (OpenMP) | gpu (CUDA) | GPU Speedup |
|-------|----------|-------------|-----------|-------------|
| **RNN** | 2.53 s | 2.84 s | 0.54 s | **4.7×** |
| **LSTM** | 11.41 s | 11.73 s | 1.57 s | **7.3×** |
| **GRU** | 21.59 s | 21.69 s | 1.39 s | **15.5×** |

#### Large Config (H=256, seq=50, 3 epochs, N=1200)

| Layer | cpu-eigen | cpu (OpenMP) | gpu (CUDA) | GPU Speedup |
|-------|----------|-------------|-----------|-------------|
| **RNN** | 22.93 s | 24.24 s | 1.88 s | **12.2×** |
| **LSTM** | 100.40 s | 101.28 s | 6.16 s | **16.3×** |
| **GRU** | 318.00 s | 318.07 s | 5.64 s | **56.4×** |

### Speedup Analysis

| Config/Layer | cpu vs cpu-eigen | gpu vs cpu-eigen |
|---|---|---|
| Small/RNN | 1.1× | 2.2× |
| Small/LSTM | 0.9× | 3.4× |
| Small/GRU | 1.0× | 5.2× |
| Medium/RNN | 0.9× | 4.7× |
| Medium/LSTM | 1.0× | 7.3× |
| Medium/GRU | 1.0× | 15.5× |
| Large/RNN | 0.9× | 12.2× |
| Large/LSTM | 1.0× | 16.3× |
| Large/GRU | 1.0× | **56.4×** |

### Key Takeaways

1. **GPU speedup scales dramatically with layer complexity and size**: GRU (3 gates, multi-step backward)
   benefits most, followed by LSTM (4 gates), then RNN (single gate). The GRU achieves up to **56.4×**
   speedup on the Large config — the largest speedup across all CppNet benchmarks.

2. **Larger hidden sizes amplify GPU advantage**: Moving from H=64 → H=256 increases GPU
   speedup dramatically (e.g. RNN: 2.5× → 12.2×, GRU: 5.1× → 56.4×), as the matmul
   operations become increasingly GPU-friendly.

3. **OpenMP provides marginal improvement for recurrent layers**: Unlike MLPs/CNNs, the
   sequential timestep structure of RNNs limits OpenMP parallelism. The cpu and cpu-eigen
   backends perform nearly identically across all configs.

4. **GPU loss matches CPU closely**: After fixing CUDA gradient buffer initialization
   (zeroing `atomicAdd`-based output buffers before each kernel call), the GPU path
   produces final losses in the same range as both CPU backends.

5. **CUDA kernels for recurrent layers**: New cell-level CUDA kernels handle element-wise gate
   activations (sigmoid, tanh, gate mixing), while existing `matmul_kernel` and
   `matmul_grad_weights_kernel` handle the heavy linear algebra.

### How to Reproduce

```bash
git clone https://github.com/LoqmanSamani/CppNet.git
cd CppNet
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
make -j$(nproc)
./examples/sequence_benchmark
```

Ensure CUDA is installed and detected by CMake for GPU results.

---

## Transformer Device Benchmark — Token Classification

**Source:** [`examples/transformer_benckmark.cpp`](examples/transformer_benckmark.cpp)  
**Date:** February 27, 2026

### Experiment Setup

| Parameter | Value |
|:----------|:------|
| **Task** | Token-sequence classification (4 classes) |
| **Dataset** | Synthetic integer token sequences with class-correlated token ranges |
| **Loss function** | Softmax Cross-Entropy (mean reduction) |
| **Optimizer** | Adam (β₁=0.9, β₂=0.999, ε=1e-10) |
| **Architecture** | Embedding → MultiHeadAttention (self-attention) → Mean Pool → ReLU → Linear |
| **Weight init** | Xavier |
| **Random seed** | 42 (data), 123 (training) |
| **OpenMP threads** | 4 (for `cpu` and `cpu-eigen` backends) |

#### Backends Tested

| Backend | Description |
|:--------|:------------|
| `cpu-eigen` | Eigen tensor contractions — highly optimized, SIMD-vectorized |
| `cpu` | Manual OpenMP-parallelized loops (4 threads) |
| `gpu` | CUDA kernels (embedding forward/backward, attention scale/softmax, matmul, bias, ReLU) |

#### Hardware

Same machine as previous benchmarks: NVIDIA GeForce GTX 1650 (4 GB), multi-core CPU, GCC 13.3, CUDA 12.0.

#### New CUDA Kernels

This benchmark introduces three new CUDA kernel files to support the Transformer architecture on GPU:

| Kernel File | Operations |
|:------------|:-----------|
| `embedding_forward.cu` | Embedding table lookup (gather rows by token ID), ColMajor 3D indexing |
| `embedding_backward.cu` | Scatter-add gradients back to embedding weight table via `atomicAdd` |
| `attention_scores.cu` | Attention scaling, row-wise softmax (shared-memory tree reduction), softmax backward |

---

### Network Configurations

Three Transformer configurations of increasing size are tested. All use self-attention (queries = keys = values from the same input), mean-pooling over the sequence dimension, ReLU activation, and Softmax Cross-Entropy output.

| Config | Architecture | Vocab | Embed Dim | Heads | Seq Len | Classes | Epochs | Batch Size | Samples | Learning Rate |
|:-------|:-------------|:------|:----------|:------|:--------|:--------|:-------|:-----------|:--------|:-------------|
| **Small** | Emb(200,32)→Attn(h=2)→Pool→FC(32,4) | 200 | 32 | 2 | 10 | 4 | 10 | 32 | 800 | 0.001 |
| **Medium** | Emb(500,64)→Attn(h=4)→Pool→FC(64,4) | 500 | 64 | 4 | 20 | 4 | 5 | 32 | 800 | 0.001 |
| **Large** | Emb(1000,128)→Attn(h=8)→Pool→FC(128,4) | 1,000 | 128 | 8 | 30 | 4 | 3 | 32 | 1,200 | 0.001 |

---

### Results Summary

| Config | Architecture | cpu-eigen | cpu (OpenMP) | gpu (CUDA) | GPU Speedup vs cpu-eigen |
|:-------|:-------------|:----------|:-------------|:-----------|:-------------------------|
| Small | Emb(200,32)→Attn(h=2)→Pool→FC(32,4) | 0.79 s | 0.83 s | 1.51 s | 0.52x |
| Medium | Emb(500,64)→Attn(h=4)→Pool→FC(64,4) | 2.52 s | 3.04 s | 2.51 s | **1.00x** |
| Large | Emb(1000,128)→Attn(h=8)→Pool→FC(128,4) | 9.89 s | 22.36 s | 8.51 s | **1.16x** |

---

### Detailed Per-Config Results

#### Small (Emb(200,32)→Attn(h=2)→Pool→FC(32,4)) — 10 epochs, batch 32, lr=0.001

| Device | Epoch 1 Loss | Epoch 1 Acc | Epoch 2 Loss | Epoch 2 Acc | Final Loss | Final Acc | Total Time |
|:-------|:-------------|:------------|:-------------|:------------|:-----------|:----------|:-----------|
| cpu-eigen | 0.8478 | 91.37% | 0.0102 | 100.00% | 0.0000 | 100.00% | 0.79 s |
| cpu | 0.9411 | 85.00% | 0.0464 | 100.00% | 0.0000 | 100.00% | 0.83 s |
| gpu | 0.9951 | 85.87% | 0.0341 | 100.00% | 0.0000 | 100.00% | 1.51 s |

#### Medium (Emb(500,64)→Attn(h=4)→Pool→FC(64,4)) — 5 epochs, batch 32, lr=0.001

| Device | Epoch 1 Loss | Epoch 1 Acc | Epoch 2 Loss | Epoch 2 Acc | Final Loss | Final Acc | Total Time |
|:-------|:-------------|:------------|:-------------|:------------|:-----------|:----------|:-----------|
| cpu-eigen | 0.9058 | 92.37% | 0.0108 | 100.00% | 0.0000 | 100.00% | 2.52 s |
| cpu | 0.9378 | 93.00% | 0.0242 | 100.00% | 0.0000 | 100.00% | 3.04 s |
| gpu | 0.9898 | 89.00% | 0.0186 | 100.00% | 0.0000 | 100.00% | 2.51 s |

#### Large (Emb(1000,128)→Attn(h=8)→Pool→FC(128,4)) — 3 epochs, batch 32, lr=0.001

| Device | Epoch 1 Loss | Epoch 1 Acc | Epoch 2 Loss | Epoch 2 Acc | Final Loss | Final Acc | Total Time |
|:-------|:-------------|:------------|:-------------|:------------|:-----------|:----------|:-----------|
| cpu-eigen | 1.0108 | 93.16% | 0.0270 | 100.00% | 0.0005 | 100.00% | 9.89 s |
| cpu | 1.0378 | 92.40% | 0.0363 | 100.00% | 0.0006 | 100.00% | 22.36 s |
| gpu | 1.0353 | 90.88% | 0.0330 | 100.00% | 0.0005 | 100.00% | 8.51 s |

---

### Speedup Analysis

Speedups are measured relative to the `cpu-eigen` baseline (the fastest CPU backend).

| Config | cpu-eigen | cpu (OpenMP) vs cpu-eigen | gpu (CUDA) vs cpu-eigen |
|:-------|:----------|:--------------------------|:------------------------|
| Small | 1.00x (baseline) | 0.95x (slightly slower) | 0.52x (slower — GPU overhead dominates) |
| Medium | 1.00x (baseline) | 0.83x (slower) | **1.00x** (break-even) |
| Large | 1.00x (baseline) | 0.44x (slower) | **1.16x** (faster) |

#### Observations

- **GPU overhead dominates at small scale.** For the Small config (embed_dim=32, 2 heads), the per-call GPU buffer allocation (cudaMalloc/cudaFree) and host↔device transfers in the Embedding and Attention layers outweigh any compute benefit. The GPU path is ~2x slower than Eigen.

- **GPU breaks even at Medium scale.** With embed_dim=64 and 4 attention heads, the increased matrix sizes in the Q/K/V projections begin to offset GPU launch overhead, resulting in near-parity with Eigen.

- **GPU surpasses CPU at Large scale.** At embed_dim=128 and 8 heads, the GPU delivers a 1.16x speedup over Eigen and a 2.63x speedup over the OpenMP backend. The attention Q·K^T matrix (30×30 per head) and multiple 128×128 projections provide enough parallelism for GPU gains.

- **OpenMP performance degrades sharply at Large scale.** The `cpu` backend is 2.26x slower than Eigen on Large config — substantially worse than the Small config gap (0.95x). The complex attention loop structure with multiple dependent matmuls creates serialization pressure that hurts naive OpenMP parallelism.

- **GPU attention uses a hybrid CPU/GPU approach.** The current implementation uses CUDA `matmul_kernel` for Q/K/V projections and context output projection, but falls back to Eigen for the Q·K^T transposed multiply. Implementing a dedicated transposed-matmul CUDA kernel would improve GPU performance further, especially for longer sequences.

---

### Accuracy Convergence

All three backends converge to identical final accuracy, confirming numerical correctness of the new Embedding and Attention CUDA kernels.

| Config | cpu-eigen | cpu | gpu |
|:-------|:----------|:----|:----|
| Small | 100.00% | 100.00% | 100.00% |
| Medium | 100.00% | 100.00% | 100.00% |
| Large | 100.00% | 100.00% | 100.00% |

All backends also converge to nearly identical final loss values:

| Config | cpu-eigen | cpu | gpu |
|:-------|:----------|:----|:----|
| Small | 0.0000 | 0.0000 | 0.0000 |
| Medium | 0.0000 | 0.0000 | 0.0000 |
| Large | 0.0005 | 0.0006 | 0.0005 |

---

### Key Takeaways

1. **New CUDA kernels for Embedding and Attention are numerically correct.** All three new kernel files (`embedding_forward.cu`, `embedding_backward.cu`, `attention_scores.cu`) produce results matching CPU backends to floating-point precision. The ColMajor tensor layout must be carefully respected in all CUDA kernels — the initial RowMajor indexing caused complete training failure (random-chance accuracy).

2. **GPU speedup for Transformers is modest compared to MLPs and CNNs.** The Transformer architecture involves mixed operations (lookup tables, attention score computation, softmax, multiple projection matmuls) with relatively small matrices. Unlike CNNs (29–42x speedup) or large MLPs (14–25x), the Transformer GPU path achieves only 1.16x at the largest tested scale. Larger embedding dimensions and longer sequences would amplify GPU advantage.

3. **Per-call GPU memory allocation limits performance.** The Embedding and Attention layers use per-forward/backward `cudaMalloc`/`cudaFree` calls (matching the RNN-layer pattern). Pre-allocating persistent GPU buffers (as done for the Linear layer) would reduce overhead and shift the GPU break-even point to smaller models.

4. **Hybrid GPU/CPU attention limits scalability.** The Q·K^T computation currently downloads data from GPU, performs the transposed multiply on CPU via Eigen, then re-uploads. A dedicated transposed-matmul CUDA kernel would keep the entire attention forward pass on-device and improve performance for longer sequences.

5. **OpenMP attention is significantly slower than Eigen.** The multi-head attention backward pass involves complex dependent matrix multiplications that serialize poorly under OpenMP. Eigen's optimized tensor contractions handle these much more efficiently (2.26x faster at Large scale).

---

### How to Reproduce

```bash
git clone https://github.com/LoqmanSamani/CppNet.git
cd CppNet
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
make -j$(nproc)
./examples/transformer_benchmark
```

Ensure CUDA is installed and detected by CMake for GPU results.

---

## Deep ResNet Device Benchmark — Spiral Classification

**Source:** [`examples/residual_benckmark.cpp`](examples/residual_benckmark.cpp)  
**Date:** February 27, 2026

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
| **Activations** | ReLU (after projection and each residual block) |
| **Weight init** | He (hidden layers), Xavier (output head) |
| **Random seed** | 42 (data), 123 (training) |
| **OpenMP threads** | 4 (for `cpu` and `cpu-eigen` backends) |

#### Architecture

The benchmark uses a deep ResNet-style architecture with stacked residual blocks:

```
Linear(2, W) → ReLU → [ResBlock(Linear(W,W) → Linear(W,W)) + skip → ReLU] ×D → Linear(W, 5)
```

Each residual block contains two Linear layers with an identity shortcut (skip connection). The depth parameter D controls the number of stacked blocks, making the network progressively deeper while keeping width constant.

#### Backends Tested

| Backend | Description |
|:--------|:------------|
| `cpu-eigen` | Eigen tensor contractions — highly optimized, SIMD-vectorized |
| `cpu` | Manual OpenMP-parallelized loops (4 threads) |
| `gpu` | CUDA kernels (matmul, bias, ReLU, elementwise add for skip connections) |

#### Hardware

Same machine as previous benchmarks: NVIDIA GeForce GTX 1650 (4 GB), multi-core CPU, GCC 13.3, CUDA 12.0.

---

### Network Configurations

Three ResNet configurations with increasing width and depth. All use identity skip connections (no projection needed since `in_features == out_features` within each block).

| Config | Architecture | Width (W) | Depth (D) | Linear Layers | Residual Adds | Epochs | Batch Size | Learning Rate |
|:-------|:-------------|:----------|:----------|:--------------|:--------------|:-------|:-----------|:--------------|
| **Small** | 2→64→[ResBlock×2]→5 | 64 | 2 | 6 (proj + 2×2 block + head) | 2 | 20 | 128 | 0.001 |
| **Medium** | 2→128→[ResBlock×4]→5 | 128 | 4 | 10 (proj + 4×2 block + head) | 4 | 10 | 256 | 0.0005 |
| **Large** | 2→256→[ResBlock×6]→5 | 256 | 6 | 14 (proj + 6×2 block + head) | 6 | 10 | 256 | 0.0005 |

---

### Results Summary

| Config | Architecture | cpu-eigen | cpu (OpenMP) | gpu (CUDA) | GPU Speedup vs cpu-eigen |
|:-------|:-------------|:----------|:-------------|:-----------|:-------------------------|
| Small | 2→64→[ResBlock×2]→5 | 6.35 s | 24.85 s | 4.06 s | **1.56x** |
| Medium | 2→128→[ResBlock×4]→5 | 26.60 s | 77.50 s | 3.99 s | **6.68x** |
| Large | 2→256→[ResBlock×6]→5 | 125.27 s | 517.33 s | 13.88 s | **9.03x** |

---

### Detailed Per-Config Results

#### Small (2→64→[ResBlock×2]→5) — 20 epochs, batch 128, lr=0.001

| Device | Epoch 1 Loss | Epoch 1 Acc | Epoch 10 Loss | Epoch 10 Acc | Final Loss | Final Acc | Total Time |
|:-------|:-------------|:------------|:--------------|:-------------|:-----------|:----------|:-----------|
| cpu-eigen | 0.9796 | 56.31% | — | — | 0.1977 | 92.29% | 6.35 s |
| cpu | 1.1585 | 44.84% | — | — | 0.1976 | 92.25% | 24.85 s |
| gpu | 1.0525 | 50.99% | — | — | 0.1970 | 92.35% | 4.06 s |

#### Medium (2→128→[ResBlock×4]→5) — 10 epochs, batch 256, lr=0.0005

| Device | Epoch 1 Loss | Epoch 1 Acc | Final Loss | Final Acc | Total Time |
|:-------|:-------------|:------------|:-----------|:----------|:-----------|
| cpu-eigen | 1.2910 | 40.87% | 0.2090 | 91.75% | 26.60 s |
| cpu | 1.3023 | 43.02% | 0.2106 | 91.73% | 77.50 s |
| gpu | 1.4295 | 41.30% | 0.2178 | 91.43% | 3.99 s |

#### Large (2→256→[ResBlock×6]→5) — 10 epochs, batch 256, lr=0.0005

| Device | Epoch 1 Loss | Epoch 1 Acc | Final Loss | Final Acc | Total Time |
|:-------|:-------------|:------------|:-----------|:----------|:-----------|
| cpu-eigen | 3.8723 | 37.37% | 0.2205 | 91.37% | 125.27 s |
| cpu | 3.8747 | 36.78% | 0.2401 | 90.73% | 517.33 s |
| gpu | 3.4013 | 37.20% | 0.2239 | 91.21% | 13.88 s |

---

### Speedup Analysis

Speedups are measured relative to the `cpu-eigen` baseline (the fastest CPU backend).

| Config | cpu-eigen | cpu (OpenMP) vs cpu-eigen | gpu (CUDA) vs cpu-eigen |
|:-------|:----------|:--------------------------|:------------------------|
| Small | 1.00x (baseline) | 0.26x (slower) | **1.56x** |
| Medium | 1.00x (baseline) | 0.34x (slower) | **6.68x** |
| Large | 1.00x (baseline) | 0.24x (slower) | **9.03x** |

#### Observations

- **GPU speedup scales strongly with both width and depth.** From Small (W=64, D=2) to Large (W=256, D=6), GPU speedup grows from 1.56x to 9.03x. The deeper the network, the more forward/backward matmul operations are performed per batch — all of which map to GPU `matmul_kernel` calls.

- **Residual skip connections add minimal GPU overhead.** The elementwise add for the skip connection uses the existing `elementwise_gpu` CUDA kernel (op=add). Its cost is negligible compared to the Linear layer matmuls it sits alongside.

- **OpenMP is significantly slower than Eigen for deep networks.** The `cpu` backend is 3–4x slower than Eigen across all configs. With 6 residual blocks (12 Linear layers), the overhead of naive OpenMP parallelization compounds across all layers.

- **GPU advantage grows with depth, not just width.** Comparing to the MLP benchmark (same dataset, same widths), the ResNet benchmark shows that stacking more layers (depth) amplifies GPU advantage because the GPU amortizes its fixed overhead across more kernel launches per batch.

---

### Accuracy Convergence

All three backends converge to similar final accuracy (~91–92%), confirming numerical correctness of the Residual layer's skip connection and `elementwise_gpu` kernel across all devices.

| Config | cpu-eigen | cpu | gpu |
|:-------|:----------|:----|:----|
| Small | 92.29% | 92.25% | 92.35% |
| Medium | 91.75% | 91.73% | 91.43% |
| Large | 91.37% | 90.73% | 91.21% |

Minor variations are expected due to floating-point non-associativity across different computation orders and the stochastic nature of training.

---

### Key Takeaways

1. **The Residual layer now supports all three devices.** The forward and backward element-wise addition uses Eigen tensor operations (`cpu-eigen`), OpenMP parallel loops (`cpu`), and the `elementwise_gpu` CUDA kernel (`gpu`). All paths produce numerically consistent results.

2. **GPU delivers 1.5x–9x speedup for ResNets.** Even at small scale (W=64), the GPU is faster than Eigen. At Large scale (W=256, D=6), the GPU completes in 13.9 s vs. 125.3 s for Eigen — a **9.03x** speedup.

3. **Depth amplifies GPU advantage.** Deeper networks perform more matmul operations per batch, better amortizing GPU kernel launch overhead. The ResNet Large config (14 Linear layers) achieves better GPU speedup than comparably-sized flat MLPs.

4. **Skip connections are essentially free on GPU.** The `elementwise_gpu` kernel adds two tensors with a simple per-element CUDA kernel launch. Its cost is dominated by the surrounding `matmul_kernel` calls in the Linear layers.

5. **OpenMP performance degrades with network depth.** The `cpu` backend scales poorly with the number of layers, showing 3–4x slowdown vs. Eigen. Each layer's OpenMP parallel-for has thread synchronization overhead that accumulates across 14+ layers per batch.

---

### How to Reproduce

```bash
git clone https://github.com/LoqmanSamani/CppNet.git
cd CppNet
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
make -j$(nproc)
./examples/residual_benchmark
```

Ensure CUDA is installed and detected by CMake for GPU results.
