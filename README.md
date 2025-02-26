# RacoGrad: A Deep Learning Framework in Racket

RacoGrad is a deep learning framework implemented in Racket, a dialect of the Lisp/Scheme family of programming languages. It provides a comprehensive set of tools for creating, training, and evaluating neural networks, with a focus on both educational value and practical functionality.

## Features

- **Tensor Operations**: Comprehensive set of tensor operations with shape checking and broadcasting
- **Device-Aware Computation**: Support for CPU, GPU (via OpenCL), and MLX (Apple Silicon)
- **Automatic Differentiation**: Backpropagation for gradient computation
- **CNN Support**: Implementation of convolutional neural networks with LeNet-5 architecture
- **MNIST Dataset Handling**: Tools for loading, preprocessing, and training on MNIST
- **Early Stopping and Validation**: Techniques to prevent overfitting during training

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/dev-null321/RacoGrad.git
   cd RacoGrad
   ```

2. Download the MNIST dataset:
   ```bash
   mkdir -p mnist-data
   cd mnist-data
   ```
   
   Download the following files and place them in the mnist-data directory:

   Note I used Kaggle do download the dataset. Download it and place it in your directory. 
   
   Note: Remember to gunzip these files after downloading.

4. Compile the C extensions:
   ```bash
   ./compile_extensions.sh
   ```

## Usage

### Training a Logistic Regression Model on MNIST

```racket
#lang racket
(require "mnist.rkt")

;; The mnist.rkt module will automatically load and train a logistic regression model
;; on the MNIST dataset when required
```

### Training a CNN on MNIST

```racket
#lang racket
(require "CNN.rkt")

;; To train a CNN on the default device (MLX if available):
(train-cnn)

;; To specify the device:
(train-cnn 'cpu)  ; Use CPU
(train-cnn 'mlx)  ; Use MLX (Apple Silicon)
(train-cnn 'gpu)  ; Use GPU (via OpenCL)

;; To specify training parameters:
(train-cnn 'cpu 10 64)  ; 10 epochs, batch size 64
```

### Tensor Operations

```racket
(require "tensor.rkt")

;; Create a tensor
(define t (t:create '(2 3) #(1 2 3 4 5 6)))

;; Basic operations
(t:add t1 t2)      ; Add two tensors
(t:mul t1 t2)      ; Matrix multiplication
(t:scale t 2.0)    ; Scalar multiplication
(t:transpose t)    ; Transpose tensor

;; Device-aware tensors
(require "tensor_device.rkt")
(require "device.rkt")

;; Create a device tensor on CPU
(define dt (dt:create '(2 3) #(1 2 3 4 5 6) (cpu)))

;; Move to GPU if available
(dt:to dt (gpu))

;; Operations automatically use the appropriate device
(dt:add dt1 dt2)
```

## Documentation

For detailed information about the implementation and usage, refer to the following documents:

- [User Guide](./RACOGRAD_USER_GUIDE.md): Basic usage guide for users
- [Implementation Details](./RACOGRAD_IMPLEMENTATION_DETAILS.md): Technical details of the implementation
- [Optimization Strategy](./OPTIMIZATION_STRATEGY.md): Performance optimization strategies
- [GPU Acceleration](./gpu_acceleration.md): Details on GPU acceleration

## New Additions

This updated version of RacoGrad includes several major enhancements:

- **Convolutional Neural Networks**: Full implementation of CNN with backpropagation
- **Device-Aware Computing**: Abstraction layer for running on different hardware
- **Hardware Acceleration**: Support for GPU via OpenCL and Apple Silicon via MLX
- **Improved MNIST Training**: Added validation splits and early stopping
- **Better Documentation**: Comprehensive documentation of the implementation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
