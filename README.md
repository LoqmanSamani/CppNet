# CppNet

<div align="center">
  <img src="imgs/cpp-net-logo-.png" alt="CppNet Logo" width="400"/>
</div>

<p align="center">
  <b>CppNet</b> is a high-performance C++ deep learning library for building and training neural networks.  
  It uses <a href="https://eigen.tuxfamily.org">Eigen</a> for fast tensor operations,  
  <a href="https://www.openmp.org/">OpenMP</a> for CPU parallelism,  
  and <a href="https://developer.nvidia.com/cuda-zone">CUDA</a> for GPU acceleration.  
</p>

---

## ✨ Features

- 🚀 **High Performance**: Vectorized operations via Eigen and multi-threading with OpenMP.  
- 🔧 **GPU Acceleration**: CUDA support for heavy computations.  
- 🧩 **Modular API**: Clear separation of layers, losses, optimizers, metrics, and regularizations.  
- 📦 **Extensible**: Easy to add custom layers, losses, or optimizers.  
- 📊 **Visualization**: Tools for plotting training curves and inspecting models.  
- 🧪 **Examples Included**: CNN, Linear, Transformer, and more.  

---

## 📂 Full Project Directory Tree

```bash
CppNet/
│
├── CMakeLists.txt
├── LICENSE
├── README.md
├── imgs
├── build
├── templates
├── include/
│   └── CppNet/
│       ├── activations/
│       │   ├── relu.hpp
│       │   ├── sigmoid.hpp
│       │   └── tanh.hpp
│       │
│       ├── layers/
│       │   ├── dense.hpp
│       │   ├── conv2d.hpp
│       │   └── dropout.hpp
│       │
│       ├── losses/
│       │   ├── mse.hpp
│       │   └── cross_entropy.hpp
│       │
│       ├── kernels/
│       │   ├── cpu/
│       │   │   ├── matmul_cpu.hpp
│       │   │   ├── conv_cpu.hpp
│       │   │   └── common_cpu.hpp 
│       │   │
│       │   └── gpu/
│       │       ├── matmul_gpu.hpp
│       │       ├── conv_gpu.hpp
│       │       └── common_gpu.hpp
│       │
│       ├── models/
│       │   ├── sequential.hpp
│       │   └── model_loader.hpp
│       │
│       ├── optimizers/
│       │   ├── sgd.hpp
│       │   └── adam.hpp
│       │
│       ├── regularizations/
│       │   ├── l1.hpp
│       │   └── l2.hpp
│       │
│       ├── utils/
│       │   ├── tensor.hpp
│       │   ├── initializer.hpp
│       │   └── random.hpp
│       │
│       └── CppNet.hpp   
│
│
├── src/
│   ├── activations/
│   │   ├── relu.cpp
│   │   ├── sigmoid.cpp
│   │   └── tanh.cpp
│   │
│   ├── layers/
│   │   ├── dense.cpp
│   │   ├── conv2d.cpp
│   │   └── dropout.cpp
│   │
│   ├── losses/
│   │   ├── mse.cpp
│   │   └── cross_entropy.cpp
│   │
│   ├── kernels/
│   │   ├── cpu/
│   │   │   ├── matmul_cpu.cpp
│   │   │   ├── conv_cpu.cpp
│   │   │   └── common_cpu.cpp
│   │   │
│   │   └── gpu/
│   │       ├── matmul_gpu.cu
│   │       ├── conv_gpu.cu
│   │       └── common_gpu.cu
│   │
│   ├── models/
│   │   ├── sequential.cpp
│   │   └── model_loader.cpp
│   │
│   ├── optimizers/
│   │   ├── sgd.cpp
│   │   └── adam.cpp
│   │
│   ├── regularizations/
│   │   ├── l1.cpp
│   │   └── l2.cpp
│   │
│   └── utils/
│       ├── tensor.cpp
│       ├── initializer.cpp
│       └── random.cpp
│
│
├── tests/
│   ├── CMakeLists.txt
│   ├── kernels/
│   │   ├── cpu/
│   │   │   ├── test_matmul_cpu.cpp
│   │   │   └── test_conv_cpu.cpp
│   │   └── gpu/
│   │       ├── test_matmul_gpu.cu
│   │       └── test_conv_gpu.cu
│   │
│   ├── layers/
│   │   ├── test_dense.cpp
│   │   └── test_conv2d.cpp
│   │
│   ├── activations/
│   │   ├── test_relu.cpp
│   │   └── test_sigmoid.cpp
│   │
│   ├── utils/
│   │   ├── test_tensor.cpp
│   │   └── test_random.cpp
│   │
│   └── models/
│       └── test_sequential.cpp
│
│
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── design_notes.md
│
└── examples/
    ├── xor/
    │   ├── xor_train.cpp
    │   └── xor_predict.cpp
    ├── mnist/
    │   ├── mnist_train.cpp
    │   └── mnist_infer.cpp
    └── simple_dense.cpp
```

---

## ⚡ Installation

### Prerequisites
- **C++17 or newer**
- [Eigen](https://eigen.tuxfamily.org) (header-only, auto-detected if installed)
- [CMake](https://cmake.org) ≥ 3.14
- (Optional) CUDA Toolkit for GPU acceleration
- (Optional) OpenMP for CPU parallelism

### Build Instructions
```bash
git clone https://github.com/LoqmanSamani/CppNet.git
cd CppNet
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This will build the library and all example programs inside `examples/`.

---


## 📚 Roadmap

- [ ] Implement complete set of layers (CNN, RNN, Transformer, etc.)  
- [ ] Add GPU backend (CUDA kernels for layers and ops)  
- [ ] Add training utilities (Trainer, Callbacks, Checkpoints)  
- [ ] Add more examples (GANs, Reinforcement Learning, NLP models)  

---

## 🛠 Contributing

Contributions are welcome!  

Please follow consistent coding style (headers in `include/`, implementations in `src/`).

---

## 📜 License

CppNet is released under the [MIT License](LICENSE).  
