# RacoGrad: A Deep Learning Framework in Racket

## Overview

RacoGrad is a deep learning framework implemented in Racket that provides functionality for creating, training, and evaluating neural networks. It includes support for:

- Tensor operations
- Automatic differentiation (autograd)
- Convolutional Neural Networks (CNNs)
- Device-aware computation (CPU, GPU via OpenCL, and Apple Silicon via MLX)
- MNIST dataset handling

## Getting Started

### Installation

1. Make sure you have Racket installed on your system
2. Clone this repository
3. Compile the C/C++ extensions by running:
   ```bash
   ./compile_extensions.sh
   ```

### Basic Usage

#### Training a Logistic Regression Model on MNIST

```racket
#lang racket
(require "mnist.rkt")

;; The mnist.rkt module will automatically load and train a logistic regression model
;; on the MNIST dataset when required
```

#### Training a CNN on MNIST

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

## Core Components

### Tensor Operations

The tensor operations are provided by "tensor.rkt" and "tensor_device.rkt":

```racket
(require "tensor.rkt")        ; Basic tensor operations
(require "tensor_device.rkt") ; Device-aware tensor operations

;; Create a tensor
(define t (t:create '(2 3) #(1 2 3 4 5 6)))

;; Create a device-aware tensor
(require "device.rkt")
(define dt (dt:create '(2 3) #(1 2 3 4 5 6) (cpu)))

;; Perform operations
(t:add t1 t2)            ; Add two tensors
(dt:add dt1 dt2)         ; Add two device tensors

;; Move tensor between devices
(dt:to dt (gpu))         ; Move tensor to GPU
```

### Automatic Differentiation

The autograd functionality is provided by "autograd.rkt":

```racket
(require "autograd.rkt")

;; Activation functions
(relu x)                ; ReLU activation
(sigmoid x)             ; Sigmoid activation

;; Forward pass
(dense-forward input weights biases activation-fn)

;; Backward pass
(dense-backward input weights biases output grad-output 
                activation-derivative learning-rate)
```

### CNN Operations

CNN operations are provided by "cnn_ops.rkt":

```racket
(require "cnn_ops.rkt")

;; These are lower-level C function bindings
;; Higher-level API is provided in CNN.rkt

;; Load optimal implementation for current device
(load-optimal-ops (current-device))
```

## Hardware Support

RacoGrad can utilize different hardware backends:

- **CPU**: Default, works on all systems
- **GPU**: Via OpenCL for systems with compatible GPUs
- **MLX**: Accelerated operations on Apple Silicon (M1/M2/M3)

To select a device:

```racket
(require "device.rkt")
(require "hardware_detection.rkt")

;; Set current device
(set-current-device! (cpu))  ; Use CPU
(set-current-device! (gpu))  ; Use GPU if available
(set-current-device! (mlx))  ; Use MLX if on Apple Silicon

;; Check hardware availability
(gpu-available?)            ; Returns #t if GPU is available
(device-available? 'mlx)    ; Returns #t if MLX is available
```

## Example: Custom Model

Here's how to create a custom neural network:

```racket
(require "tensor.rkt")
(require "autograd.rkt")

;; Define model parameters
(define input-size 784)
(define hidden-size 128)
(define output-size 10)

;; Initialize weights and biases
(define W1 (t:random (list input-size hidden-size) 0.01))
(define b1 (t:random (list 1 hidden-size) 0.01))
(define W2 (t:random (list hidden-size output-size) 0.01))
(define b2 (t:random (list 1 output-size) 0.01))

;; Forward pass function
(define (my-model input)
  (let* ([hidden (relu (dense-forward input W1 b1 identity))]
         [output (dense-forward hidden W2 b2 identity)])
    output))

;; Predict function
(define (predict output-tensor)
  (argmax vector-ref (t:data output-tensor)))
```

## Advanced Features

### Model Saving/Loading

To save and load models, you can use Racket's built-in serialization:

```racket
(require racket/serialize)

;; Save model parameters
(define out (open-output-file "model.dat" #:exists 'replace))
(serialize 
  (list (t:data weights) (t:shape weights)
        (t:data bias) (t:shape bias))
  out)
(close-output-port out)

;; Load model parameters
(define in (open-input-file "model.dat"))
(define params (deserialize in))
(define loaded-weights 
  (t:create (second params) (first params)))
(define loaded-bias
  (t:create (fourth params) (third params)))
(close-input-port in)
```

## Troubleshooting

- If you encounter errors with the C extensions, make sure to run the `compile_extensions.sh` script.
- Some operations require specific hardware. If not available, the framework will fall back to CPU.
- MNIST data needs to be in the `mnist-data` directory. The scripts expect the standard MNIST files:
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`
  - `t10k-images.idx3-ubyte`
  - `t10k-labels.idx1-ubyte`

## Performance Tips

- Use device-aware tensors (dt:*) rather than basic tensors (t:*) for better performance
- When running on Apple Silicon, use the MLX device for best performance
- Batch processing improves throughput significantly
- For large datasets, use mini-batch training rather than full batch