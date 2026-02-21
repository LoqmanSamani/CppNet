### Instruction File for Developing CppNet Library

---

#### Overview

**CppNet** is a minimal, modular C++ deep learning library. It provides a complete pipeline for
building, training, and evaluating neural networks. The library leverages **Eigen** for default
tensor operations, **OpenMP** for CPU parallelism, and **CUDA** for GPU acceleration.

---

#### Core Modules

##### 1) Layers (`layers/`)
- **Linear** (Dense / Fully Connected) layer
- **Conv2D** (2D Convolutional) layer
- **Flatten** layer
- **MaxPool2D** (Max Pooling 2D) layer
- **RNN** (Recurrent Neural Network) layer
- **MultiHeadAttention** (Multi-Head Attention) layer
- Base class: `Layer` — all layers inherit from this and implement `is_trainable()` and `step()`.

##### 2) Activations (`activations/`)
- **ReLU**
- **LeakyReLU**
- **Sigmoid**
- **Softmax**
- **Tanh**
- Base class: `Activation` — all activations inherit from this and implement 2D/4D `forward()` and `backward()`.

##### 3) Losses (`losses/`)
- **BinaryCrossEntropy**
- **CategoricalCrossEntropy**
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **Huber** loss
- Base class: `Loss` (defined in `loss.hpp`) — all losses inherit from this and implement `forward()` and `backward()`.

##### 4) Optimizers (`optimizers/`)
- **SGD** (Stochastic Gradient Descent)
- **Adam**
- **Adagrad**
- **RMSProp** (named `mrs_prop` in the codebase)
- **Momentum**
- Base class: `Optimizer` — all optimizers inherit from this and implement `step()` for each supported layer type.

##### 5) CUDA Kernels (`kernels/gpu/`)
All GPU kernels needed to accelerate layers, activations, losses, and optimizers:
- `matmul.cu` — tiled shared-memory matrix multiplication
- `add_bias.cu` — bias addition kernel
- `matmul_grad_input.cu` — gradient w.r.t. input (dX = dY × W^T)
- `matmul_grad_weight.cu` — gradient w.r.t. weights (dW = X^T × dY)
- `bias_grad.cu` — bias gradient via parallel reduction
- `sgd_step.cu` — SGD weight update kernel
- `relu.cu` — element-wise ReLU forward kernel
- `relu_grad.cu` — element-wise ReLU backward kernel
- `elementwise.cu` — general element-wise operations kernel
- Additional kernels should be added as new layers/losses require them (e.g., sigmoid GPU, conv2d GPU, softmax GPU, etc.)

##### 6) Models (`models/`)
- **SequentialModel** — a container that chains layers sequentially; supports `add_layer()`, `forward()`, `backward()`, and `train_step()`.

##### 7) Metrics (`metrics/`)
- **Accuracy**, **Precision**, **Recall**, **F1Score** — evaluation metrics computed from predictions and targets.

##### 8) Regularizations (`regularizations/`)
- **L1** regularization
- **L2** regularization
- **ElasticNet** (combined L1 + L2)

##### 9) Utils (`utils/`)
- **Weight Initialization** (`init.hpp` / `init.cpp`) — Xavier, He, uniform, normal initialization strategies.
- **Data Utilities** (`utils.hpp` / `utils.cpp`) — data loading helpers (e.g., CSV loading), tensor manipulation, normalization, one-hot encoding.
- **Elapsed Time** (`elapsed_time.hpp` / `elapsed_time.cpp`) — timing utilities for benchmarking.

##### 10) Visualizations (`visualizations/`)
- **TrainingLogger** — logs training/validation loss and metrics per epoch; can export to CSV for external plotting.

---

#### Compute Backends

Every computational module (layers, activations, losses, optimizers) should support **three backends**:

| Backend     | Description                                    | Controlled by       |
|-------------|------------------------------------------------|----------------------|
| `cpu-eigen` | Default — uses Eigen tensor contractions       | `device = "cpu-eigen"` |
| `cpu`       | OpenMP-parallelized manual loops               | `device = "cpu"`       |
| `gpu`       | CUDA kernel launches (requires CUDA toolkit)   | `device = "gpu"`       |

The `device` parameter is passed at construction time. GPU kernels live in `kernels/gpu/`.

---

#### Step-by-Step Implementation Guide

1. **Implement utility/infrastructure modules first**:
   - Weight initialization (`utils/init.hpp` + `utils/init.cpp`)
   - Data utilities (`utils/utils.hpp` + `utils/utils.cpp`)
   - Elapsed time measurement (`utils/elapsed_time.hpp` + `utils/elapsed_time.cpp`)
   - Loss base class (`losses/loss.hpp`)
   - Layer base class (`layers/layer.hpp`) — already done
   - Activation base class (`activations/activation.hpp`) — already done
   - Optimizer base class (`optimizers/optimizer.hpp`) — already done

2. **Implement each layer** (`layers/`):
   - Each layer must support `cpu-eigen`, `cpu` (OpenMP), and `gpu` (CUDA) backends.
   - For GPU: implement required CUDA kernels in `kernels/gpu/` and declare them in `kernels/gpu/gpu.hpp`.
   - Implement: Linear ✅, Conv2D, Flatten, MaxPool2D, RNN, MultiHeadAttention.

3. **Implement activation functions** (`activations/`):
   - Each activation must support `cpu` and `gpu` backends with 2D and 4D tensor overloads.
   - Implement: ReLU ✅, Sigmoid ✅, LeakyReLU, Softmax, Tanh.

4. **Implement loss functions** (`losses/`):
   - Each loss must support `cpu` and `gpu` backends.
   - Implement: BinaryCrossEntropy ✅, CategoricalCrossEntropy ✅, MSE, MAE, Huber.

5. **Implement optimizers** (`optimizers/`):
   - Each optimizer must support `cpu` and `gpu` backends and provide a `step()` overload for each layer type.
   - Implement: SGD ✅, Adam, Adagrad, RMSProp, Momentum.

6. **Implement the Sequential Model** (`models/`):
   - Chains layers together, manages forward/backward pass, and integrates with optimizers.

7. **Implement metrics** (`metrics/`):
   - Accuracy, Precision, Recall, F1Score.

8. **Implement regularizations** (`regularizations/`):
   - L1, L2, ElasticNet — applied to layer weights during training.

9. **Implement visualizations** (`visualizations/`):
   - TrainingLogger for recording and exporting training history.

10. **Write tests** (`tests/`):
    - For every class/function, there is a corresponding test file in `tests/`.
    - Tests should verify forward/backward correctness, numerical gradients, edge cases, and device consistency.
    - Test files:
      - `tests/activations/test_relu.cpp`, `test_sigmoid.cpp`, `test_softmax.cpp`, `test_tanh.cpp`, `test_leacky_relu.cpp`
      - `tests/layers/test_linear.cpp`, `test_conv2d.cpp`, `test_flatten.cpp`, `test_max_pool2d.cpp`, `test_rnn.cpp`, `test_attention.cpp`
      - `tests/losses/test_binary_cross_entropy.cpp`, `test_categorical_cross_entropy.cpp`, `test_mse.cpp`, `test_mae.cpp`, `test_huber.cpp`
      - `tests/optimizers/test_sgd.cpp`, `test_adam.cpp`, `test_adagrad.cpp`, `test_mrs_prop.cpp`, `test_momentum.cpp`
      - `tests/kernels/test_matmul_gpu.cu`

11. **Update README.md** to accurately reflect the library structure, features, and build instructions.

---

#### File Naming Conventions

| Type        | Header location                    | Source location                    |
|-------------|------------------------------------|------------------------------------|
| Layer       | `include/CppNet/layers/<name>.hpp` | `src/CppNet/layers/<name>.cpp`     |
| Activation  | `include/CppNet/activations/<name>.hpp` | `src/CppNet/activations/<name>.cpp` |
| Loss        | `include/CppNet/losses/<name>.hpp` | `src/CppNet/losses/<name>.cpp`     |
| Optimizer   | `include/CppNet/optimizers/<name>.hpp` | `src/CppNet/optimizers/<name>.cpp` |
| CUDA Kernel | `include/CppNet/kernels/gpu/gpu.hpp` | `src/CppNet/kernels/gpu/<name>.cu` |
| Model       | `include/CppNet/models/<name>.hpp` | `src/CppNet/models/<name>.cpp`     |
| Metric      | `include/CppNet/metrics/<name>.hpp` | `src/CppNet/metrics/<name>.cpp`    |
| Regularizer | `include/CppNet/regularizations/<name>.hpp` | `src/CppNet/regularizations/<name>.cpp` |
| Utility     | `include/CppNet/utils/<name>.hpp`  | `src/CppNet/utils/<name>.cpp`      |

---

#### Build System Notes

- The root `CMakeLists.txt` defines the `CppNet` static library target.
- All CPU `.cpp` sources go in `CPU_SOURCES`; all `.cu` files go in `GPU_SOURCES`.
- When adding a new source file, it **must** be added to `CMakeLists.txt`.
- Tests are built via `tests/CMakeLists.txt` using CTest.
- If a `.cpp` file contains GPU dispatch code (e.g., calling CUDA APIs), it must be listed in `set_source_files_properties(... LANGUAGE CUDA)`.

---

#### Known Issues & TODOs

- [ ] `sgd_step.cu` has a bug: `blockIdx.x * blockIdx.x` should be `blockIdx.x * blockDim.x` and `LR` should be `float` not `int`.
- [ ] `elementwise.cu` kernel body implements matmul, not element-wise ops — needs rewrite.
- [ ] `gpu.hpp` `sgd_step_kernel` signature uses `int LR` — should be `float LR`.
- [ ] Loss base class is duplicated in `binary_cross_entropy.hpp` and `categorical_cross_entropy.hpp` — should be in `loss.hpp` only.
- [ ] `CategoricalCrossEntropy` class-index and sequence overloads return 0 (stub) — need implementation.
- [ ] All test files are currently empty — tests need to be written.
- [ ] `tests/CMakeLists.txt` is missing — needs to be created.
- [ ] Several source/header files referenced in the build system don't exist yet and must be created.
