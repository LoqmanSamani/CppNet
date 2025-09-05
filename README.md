# CppNet

<div align="center">
  <img src="imgs/cpp-logo.png" alt="CppNet Logo" width="600"/>
</div>

<p align="center">
  <b>CppNet</b> is a high-performance C++ deep learning library for building and training neural networks.  
  It leverages <a href="https://eigen.tuxfamily.org">Eigen</a> for fast tensor operations,  
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

## 📂 Project Structure

```bash
CppNet/
├── CMakeLists.txt
├── examples/                   # Example programs
│   ├── cnn_example.cpp         # CNN example
│   ├── linear_example.cpp      # Linear layer example
│   └── transformer_example.cpp # Transformer example
├── include/                    # Public headers (library API)
│   ├── activations.hpp
│   ├── data.hpp
│   ├── layers.hpp
│   ├── losses.hpp
│   ├── metrics.hpp
│   ├── models.hpp
│   ├── optimizers.hpp
│   ├── regularization.hpp
│   ├── utils.hpp
│   └── visualization.hpp
├── src/                        # Implementations
│   ├── activations.cpp
│   ├── data.cpp
│   ├── layers.cpp
│   ├── losses.cpp
│   ├── metrics.cpp
│   ├── models.cpp
│   ├── optimizers.cpp
│   ├── regularization.cpp
│   ├── utils.cpp
│   └── visualization.cpp
├── LICENSE
└── README.md
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
