# CppNet

<div align="center">
  <img src="imgs/log2.png" alt="CppNet Logo" width="1000"/>
</div>

<p align="center">
  <b>CppNet</b> is a high-performance C++17 deep learning library for building and training neural networks from scratch.<br/>
  Built on <a href="https://eigen.tuxfamily.org">Eigen</a> for fast tensor operations,
  <a href="https://www.openmp.org/">OpenMP</a> for CPU parallelism,
  and <a href="https://developer.nvidia.com/cuda-zone">CUDA</a> for GPU acceleration.
</p>

<p align="center">
  <a href="#installation"><img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++17"/></a>
  <a href="#installation"><img src="https://img.shields.io/badge/CMake-%E2%89%A53.18-blue.svg" alt="CMake"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"/></a>
  <a href="#gpu-acceleration"><img src="https://img.shields.io/badge/CUDA-optional-yellowgreen.svg" alt="CUDA"/></a>
</p>

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Overview](#api-overview)
  - [Layers](#layers)
  - [Activations](#activations)
  - [Losses](#losses)
  - [Optimizers](#optimizers)
  - [Metrics](#metrics)
  - [Regularizations](#regularizations)
  - [Utilities](#utilities)
  - [Visualization](#visualization)
- [Examples](#examples)
- [GPU Acceleration](#gpu-acceleration)
- [Benchmarks](#benchmarks)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **High Performance** — Vectorized tensor operations via Eigen, multi-threaded with OpenMP, optional CUDA GPU kernels.
- **Rich Layer Library** — Linear, Conv2D, MaxPool2D, RNN, LSTM, GRU, Multi-Head Attention, Dropout, BatchNorm, Embedding, Residual, GlobalPool, Flatten.
- **Multiple Backends** — Per-layer compute backend selection: `"cpu-eigen"` (Eigen contractions), `"cpu"` (OpenMP loops), `"gpu"` (CUDA kernels).
- **Modular Architecture** — Clean separation of layers, activations, losses, optimizers, metrics, regularizations, and utilities.
- **Training Utilities** — DataLoader with batching & shuffling, learning rate schedulers, early stopping callbacks, gradient clipping, model serialization.
- **Visualization** — Built-in `TrainingLogger` for tracking metrics and exporting training history to CSV.
- **Extensible** — Abstract base classes for layers, losses, and optimizers make it straightforward to add custom components.
- **Single-Header Access** — `#include <CppNet/CppNet.hpp>` brings in the entire library.

---

## Installation

### Prerequisites

| Dependency | Version | Required |
|:-----------|:--------|:---------|
| C++ compiler (GCC, Clang, MSVC) | C++17 support | Yes |
| [CMake](https://cmake.org) | &ge; 3.18 | Yes |
| [Eigen3](https://eigen.tuxfamily.org) | &ge; 3.3 | Yes |
| [OpenMP](https://www.openmp.org/) | any | Optional (CPU parallelism) |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-zone) | any | Optional (GPU acceleration) |

### Build from Source

```bash
git clone https://github.com/LoqmanSamani/CppNet.git
cd CppNet
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Install System-Wide

```bash
sudo make install
```

This installs headers to `/usr/local/include/CppNet/` and the static library to `/usr/local/lib/`.

### Use in Your CMake Project

```cmake
find_package(CppNet REQUIRED)
target_link_libraries(your_target PRIVATE CppNet::CppNet)
```

---

## Quick Start

A minimal binary classification example:

```cpp
#include <CppNet/CppNet.hpp>
#include <iostream>

int main() {
    // Define layers
    CppNet::Layers::Linear layer1(30, 64, "fc1", true, true, "cpu-eigen", "xavier");
    CppNet::Layers::Linear layer2(64, 1,  "fc2", true, true, "cpu-eigen", "xavier");
    CppNet::Activations::ReLU relu("cpu-eigen");
    CppNet::Activations::Sigmoid sigmoid;

    // Loss & optimizer
    CppNet::Losses::BinaryCrossEntropy loss_fn("mean");
    CppNet::Optimizers::Adam optimizer;
    float lr = 0.001;

    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        auto h = relu.forward(layer1.forward(X_train));
        auto pred = sigmoid.forward(layer2.forward(h));

        float loss = loss_fn.forward(pred, Y_train);
        auto grad = loss_fn.backward(pred, Y_train);

        grad = layer2.backward(sigmoid.backward(grad));
        layer1.backward(relu.backward(grad));

        layer2.step(optimizer, lr);
        layer1.step(optimizer, lr);

        std::cout << "Epoch " << epoch << " — Loss: " << loss << std::endl;
    }
    return 0;
}
```

---

## API Overview

### Layers

All layers inherit from `CppNet::Layers::Layer` and implement `forward()`, `backward()`, `step()`, `freeze()`, `unfreeze()`, and `print_layer_info()`.

| Layer | Description | Key Parameters |
|:------|:------------|:---------------|
| `Linear` | Fully connected layer | `in_size`, `out_size`, `bias`, `device`, `weight_init` |
| `Conv2D` | 2D convolution | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding` |
| `MaxPool2D` | 2D max pooling | `kernel_size`, `stride` |
| `Flatten` | Reshape to 2D | — |
| `RNN` | Vanilla recurrent layer | `input_size`, `hidden_size` |
| `LSTM` | Long Short-Term Memory | `input_size`, `hidden_size` |
| `GRU` | Gated Recurrent Unit | `input_size`, `hidden_size` |
| `MultiHeadAttention` | Scaled dot-product multi-head attention | `embed_dim`, `num_heads` |
| `Dropout` | Dropout regularization | `drop_rate` |
| `BatchNorm` | Batch normalization | `num_features` |
| `Embedding` | Embedding lookup table | `vocab_size`, `embed_dim` |
| `Residual` | Residual (skip) connection wrapper | — |
| `GlobalPool` | Global average/max pooling | — |

### Activations

| Activation | Function |
|:-----------|:---------|
| `ReLU` | $\max(0, x)$ |
| `LeakyReLU` | $\max(\alpha x, x)$ |
| `Sigmoid` | $\sigma(x) = \frac{1}{1 + e^{-x}}$ |
| `Tanh` | $\tanh(x)$ |
| `Softmax` | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ |

All activations support both 2D (`MatrixXd`) and 4D (`Tensor<double,4>`) inputs.

### Losses

| Loss | Typical Use |
|:-----|:------------|
| `MSE` | Regression |
| `MAE` | Regression |
| `Huber` | Robust regression |
| `BinaryCrossEntropy` | Binary classification |
| `CategoricalCrossEntropy` | Multi-class classification |
| `SoftmaxCrossEntropy` | Multi-class (fused softmax + CE) |

All support configurable reduction modes (`"mean"`, `"sum"`).

### Optimizers

| Optimizer | Description |
|:----------|:------------|
| `SGD` | Stochastic Gradient Descent |
| `Adam` | Adaptive Moment Estimation (default: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$) |
| `Adagrad` | Adaptive gradient accumulation |
| `Momentum` | SGD with momentum |
| `RMSProp` | Root Mean Square Propagation |

### Metrics

```cpp
CppNet::Metrics::accuracy(predictions, targets);
CppNet::Metrics::binary_accuracy(predictions, targets, 0.5);
CppNet::Metrics::precision(predictions, targets, 0.5);
CppNet::Metrics::recall(predictions, targets, 0.5);
CppNet::Metrics::f1_score(predictions, targets, 0.5);
```

### Regularizations

```cpp
CppNet::Regularizations::l1_penalty(weights, lambda);
CppNet::Regularizations::l2_penalty(weights, lambda);
CppNet::Regularizations::elastic_net_penalty(weights, lambda, l1_ratio);
// Corresponding gradient functions: l1_gradient, l2_gradient, elastic_net_gradient
```

### Utilities

| Utility | Description |
|:--------|:------------|
| **DataLoader** | Batched iteration with shuffling. Supports range-based `for` loops. |
| **Weight Init** | Xavier (uniform/normal), He (uniform/normal), constant, custom. |
| **Gradient Clipping** | `clip_by_value()` and `clip_by_norm()`. |
| **Serialization** | `save_model()` / `load_model()` for full model persistence; tensor-level binary I/O. |
| **LR Schedulers** | `StepLR`, `ExponentialLR`, `CosineAnnealingLR`. |
| **Callbacks** | `EarlyStopping` with configurable patience, delta, and mode. |
| **Elapsed Time** | Training duration measurement. |

**DataLoader example:**

```cpp
CppNet::Utils::DataLoader loader(X, Y, /*batch_size=*/32, /*shuffle=*/true);
for (auto& [x_batch, y_batch] : loader) {
    // forward / backward / step
}
loader.reset(); // re-shuffle for next epoch
```

**Learning rate scheduler example:**

```cpp
CppNet::Schedulers::CosineAnnealingLR scheduler(/*initial_lr=*/0.01, /*T_max=*/100);
for (int epoch = 0; epoch < 100; ++epoch) {
    float lr = scheduler.step();
    // ... train with lr
}
```

### Visualization

```cpp
CppNet::Visualizations::TrainingLogger logger;
// Inside training loop:
logger.log("train_loss", loss);
logger.log("val_accuracy", val_acc);
logger.next_epoch();
// After training:
logger.print_epoch_summary();
logger.export_csv("training_history.csv");
```

---

## Examples

The `examples/` directory contains complete, self-contained deep learning programs that train on **synthetic data** — no downloads required. Each example generates its own dataset, trains a model, and reports final metrics.

| Example | Architecture | Dataset | Result |
|:--------|:------------|:--------|:-------|
| [`mlp_classification.cpp`](examples/mlp_classification.cpp) | Linear→ReLU→Linear→ReLU→Linear | 3-class spiral (600 samples, 2D) | **~71% accuracy** |
| [`cnn_image_classification.cpp`](examples/cnn_image_classification.cpp) | Conv2D→ReLU→MaxPool2D→Flatten→Linear | 8×8 stripe images (400 samples) | **100% accuracy** |
| [`rnn_sequence_prediction.cpp`](examples/rnn_sequence_prediction.cpp) | LSTM(1,16)→Linear(16,1) | Sine-wave sequences (400 samples) | **MSE ≈ 0.00001** |
| [`transformer_classifier.cpp`](examples/transformer_classifier.cpp) | Embedding→Self-Attention+skip→ReLU→Linear | Token sequences (400 samples) | **100% accuracy** |
| [`resnet_classifier.cpp`](examples/resnet_classifier.cpp) | Linear→ReLU→ResBlock(32)→Linear→Sigmoid | Concentric circles (600 samples) | **~99% accuracy** |
| [`spiral_classification.cpp`](examples/spiral_classification.cpp) | MLP (variable width) — device benchmark | 5-class spiral (15,000 samples, 2D) | **~92% accuracy, up to 25x GPU speedup** |

Build and run:

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
make -j$(nproc)
./examples/mlp_classification
./examples/cnn_image_classification
./examples/rnn_sequence_prediction
./examples/transformer_classifier
./examples/resnet_classifier
./examples/spiral_classification
```

Each example demonstrates key patterns:
- **MLP**: Multi-class classification with softmax, manual forward/backward loop
- **CNN**: Image feature extraction, Conv2D + pooling pipeline
- **RNN/LSTM**: Time-series regression, sequence processing with hidden states
- **Transformer**: Token embedding + self-attention, skip connections, mean-pooling
- **ResNet**: Residual (skip) connections, gradient clipping, He initialization
- **Spiral Benchmark**: Multi-backend (cpu-eigen / cpu / gpu) performance comparison across network sizes

---

## GPU Acceleration

CppNet automatically detects CUDA at build time. When available, layers can target the GPU backend:

```cpp
CppNet::Layers::Linear layer(784, 256, "fc1", true, true, "gpu", "xavier");
```

Available CUDA kernels:
- Matrix multiplication (`matmul`, `matmul_grad_input`, `matmul_grad_weight`)
- Bias operations (`add_bias`, `bias_grad`)
- Elementwise operations
- ReLU forward & backward
- SGD update step

To force a CPU-only build even when CUDA is present:

```bash
cmake .. -DCUDAToolkit_ROOT=/nonexistent
```

---

## Benchmarks

### MLP Device Benchmark — Spiral Classification

A 5-class 2D spiral with 15,000 samples (3,000 per class) is classified by MLPs of increasing width.
Three backends are compared: **cpu-eigen** (Eigen contractions), **cpu** (OpenMP loops, 4 threads), and **gpu** (CUDA kernels).
All configurations use the Adam optimizer and ReLU activations.

| Config | Architecture | Epochs | Batch | cpu-eigen | cpu (OpenMP) | gpu (CUDA) | GPU Speedup vs cpu-eigen |
|:-------|:-------------|:-------|:------|:----------|:-------------|:-----------|:-------------------------|
| Small | 2→64→64→5 | 50 | 128 | 7.4 s | 17.9 s | 3.6 s | **2.03x** |
| Medium | 2→128→256→128→5 | 40 | 256 | 61.6 s | 241.9 s | 8.9 s | **6.91x** |
| Large | 2→256→512→512→256→5 | 10 | 256 | 114.5 s | 669.9 s | 8.0 s | **14.31x** |
| XLarge | 2→512→1024→1024→512→5 | 10 | 512 | 473.1 s | 3,027.0 s | 18.7 s | **25.28x** |

All configs converge to ~92–93% accuracy across all devices.
GPU advantage grows dramatically with network width — up to **25x** faster on the largest model.

> See [benchmarks.md](benchmarks.md) for full details, per-epoch logs, and methodology.

---

## Testing

CppNet uses [CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html) with **40 unit tests** covering every module:

```bash
cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

| Category | Tests |
|:---------|:------|
| **Layers** (13) | Linear, Conv2D, Flatten, MaxPool2D, RNN, Attention, BatchNorm, Dropout, Embedding, GlobalPool, GRU, LSTM, Residual |
| **Activations** (5) | ReLU, Sigmoid, Softmax, Tanh, LeakyReLU |
| **Losses** (6) | BinaryCrossEntropy, CategoricalCrossEntropy, MSE, MAE, Huber, SoftmaxCrossEntropy |
| **Optimizers** (5) | SGD, Adam, Momentum, Adagrad, RMSProp |
| **Utilities** (7) | Metrics, Regularizations, Callbacks, DataLoader, ElapsedTime, GradientClip, Init |
| **Other** (4) | Schedulers, Utils, Models, Visualizations |

Each test validates forward pass, backward pass (gradient shapes & values), and parameter updates where applicable.

---

## Project Structure

```
CppNet/
├── CMakeLists.txt              # Top-level build configuration
├── cmake/                      # CMake package config templates
├── include/CppNet/             # Public headers
│   ├── CppNet.hpp              # Single-include entry point
│   ├── activations/            # ReLU, Sigmoid, Softmax, Tanh, LeakyReLU
│   ├── layers/                 # Linear, Conv2D, RNN, LSTM, GRU, Attention, ...
│   ├── losses/                 # MSE, MAE, Huber, BCE, CCE, SoftmaxCE
│   ├── optimizers/             # SGD, Adam, Adagrad, Momentum, RMSProp
│   ├── models/                 # SequentialModel
│   ├── metrics/                # Accuracy, Precision, Recall, F1
│   ├── regularizations/        # L1, L2, Elastic Net
│   ├── kernels/gpu/            # CUDA kernels
│   ├── utils/                  # DataLoader, Init, Schedulers, Serialization, ...
│   └── visualizations/         # TrainingLogger
├── src/CppNet/                 # Implementation files (.cpp / .cu)
├── tests/                      # 40 CTest unit tests (layers, activations, losses, ...)
├── examples/                   # 6 deep learning examples (MLP, CNN, RNN, Transformer, ResNet, Benchmark)
└── docs/                       # Additional documentation
```

---

## Roadmap

- [x] Core layer library (Linear, Conv2D, Pooling, RNN, LSTM, GRU, Attention)
- [x] Activation functions (ReLU, Sigmoid, Tanh, Softmax, LeakyReLU)
- [x] Loss functions (MSE, MAE, Huber, BCE, CCE, SoftmaxCE)
- [x] Optimizers (SGD, Adam, Adagrad, Momentum, RMSProp)
- [x] DataLoader, LR schedulers, early stopping, gradient clipping
- [x] Model serialization (save/load)
- [x] CUDA GPU kernels for core operations
- [x] OpenMP CPU parallelism
- [x] Comprehensive test suite (40 unit tests)
- [x] Deep learning examples (MLP, CNN, RNN/LSTM, Transformer, ResNet)
- [ ] Expand GPU backend to cover all layers and operations
- [ ] Add Trainer abstraction with built-in training loop
- [ ] Additional examples (GANs, Reinforcement Learning, NLP pipelines)
- [ ] Python bindings (pybind11)
- [ ] Comprehensive API reference documentation

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository and create a feature branch.
2. Follow the existing coding style — headers in `include/CppNet/`, implementations in `src/CppNet/`.
3. Add tests for new functionality in `tests/`.
4. Make sure all tests pass: `cd build && ctest --output-on-failure`.
5. Open a pull request with a clear description of your changes.

---

## License

CppNet is released under the [MIT License](LICENSE).

Copyright &copy; 2025 Loghman Samani  
