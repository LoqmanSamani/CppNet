<div align="center">
  <img src="imgs/cppnet_logo.png" alt="CppNet Logo" width="300"/>
</div>




# CppNet

A high-performance C++ deep learning library for building and training neural networks.


## Project Structure

```bash

CppNet/
├── CMakeLists.txt
├── examples/                   # Example scripts for testing and demonstrating usage
│   ├── cnn_example.cpp         # Example using a CNN
│   ├── linear_example.cpp      # Example using a Linear layer
│   └── transformer_example.cpp # Example using a Transformer
│       ...                     # More examples to be added
├── include/                    # Header files defining the library's interface
│   ├── activations.hpp         # Activation functions (ReLU, Sigmoid, etc.)
│   ├── data.hpp                # Data handling (DataLoader, TensorDataset, etc.)
│   ├── layers.hpp              # Layer classes (Linear, RNN, CNN, Transformer, etc.)
│   ├── losses.hpp              # Loss functions (MSE, CrossEntropy, etc.)
│   ├── metrics.hpp             # Metrics (Accuracy, Precision, etc.)
│   ├── models.hpp              # Model classes (Sequential, etc.)
│   ├── optimizers.hpp          # Optimizers (SGD, Adam, etc.)
│   ├── regularization.hpp      # Regularization (Dropout, L2, etc.)
│   ├── utils.hpp               # Utilities (tensor ops, initialization, serialization)
│   └── visualization.hpp       # Visualization tools
├── LICENSE
├── README.md
└── src/                        # Source files with implementations
    ├── activations.cpp
    ├── data.cpp
    ├── layers.cpp
    ├── losses.cpp
    ├── metrics.cpp
    ├── models.cpp
    ├── optimizers.cpp
    ├── regularization.cpp
    ├── utils.cpp
    └── visualization.cpp
```

## Getting Started

This library is under active development. More details on installation, usage, and examples will be added as the project progresses.

## License

See the LICENSE file for details.

