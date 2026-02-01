# RacoGrad: A Deep Learning Framework in Racket

RacoGrad is a deep learning framework implemented in Racket, a dialect of the Lisp/Scheme family of programming languages. It provides tensor operations, automatic differentiation, and neural network layers — designed for both education and experimentation.

## Features

- **Tensor Operations** — creation, arithmetic, broadcasting, slicing, concatenation, reductions
- **Activation Functions** — ReLU, Leaky ReLU, ELU, Sigmoid, Tanh, Softplus, Swish (all with derivatives)
- **Neural Network Layers** — Dense/fully-connected, CNN (Conv2D, MaxPool, Flatten, Softmax)
- **Loss Functions** — Mean Squared Error, Cross-Entropy
- **Training Utilities** — Backpropagation, mini-batch SGD, early stopping, validation splits
- **Device-Aware Computation** — CPU, GPU (OpenCL), Apple Silicon (MLX), CUDA
- **MNIST Support** — Data loading, preprocessing, training pipelines

## Quick Start

```bash
git clone https://github.com/dev-null321/RacoGrad.git
cd RacoGrad
```

### Run the test suite

```bash
racket test-suite.rkt
```

### Tensor Operations

```racket
(require "tensor.rkt")

;; Create tensors
(define a (t:create '(2 3) '(1 2 3 4 5 6)))
(define b (t:zeros '(2 3))       ; all zeros
(define c (t:ones '(2 3)))       ; all ones
(define d (t:eye 3))             ; 3×3 identity matrix
(define e (t:random '(4 4) 1.0)) ; random values in [0, 1)

;; Arithmetic (with broadcasting)
(t:add a c)                      ; elementwise add
(t:sub a c)                      ; elementwise subtract
(t:mul a (t:transpose c))        ; matrix multiply (2×3) × (3×2) → (2×2)
(t:scale a 2.0)                  ; scalar multiply
(t:dot (t:create '(3) '(1 2 3))
       (t:create '(3) '(4 5 6))) ; dot product → 32

;; Broadcasting: 2D + 1D automatically broadcasts along rows
(t:add (t:create '(2 3) '(1 2 3 4 5 6))
       (t:create '(3) '(10 20 30)))
;; → tensor '(2 3) '(11 22 33 14 25 36)

;; Elementwise functions
(t:exp a)   (t:log a)   (t:sqrt a)
(t:abs a)   (t:square a) (t:negate a)
(t:clip a 0.0 5.0)                    ; clamp values to [0, 5]
(t:map (lambda (x) (* x x)) a)        ; custom elementwise function

;; Reductions
(t:sum a)     ; → 21
(t:mean a)    ; → 3.5
(t:max-val a) ; → 6
(t:min-val a) ; → 1

;; Slicing and concatenation
(t:slice a 0 1)                        ; first row
(t:concat (list a (t:ones '(1 3))))    ; append a row of ones

;; Utilities
(t:shape a) ; → '(2 3)
(t:size a)  ; → 6
(t:rank a)  ; → 2
(t:print a) ; pretty-print with shape info
```

### Activation Functions

```racket
(require "autograd.rkt")

;; Standard activations (all operate on tensors, return tensors)
(relu x)           (relu-derivative x)
(sigmoid x)        (sigmoid-derivative x)
(leaky-relu x)     (leaky-relu-derivative x)     ; α=0.01 default
(elu x)            (elu-derivative x)             ; α=1.0 default
(softplus x)       (softplus-derivative x)
(swish x)          (swish-derivative x)
```

### Training a Neural Network

```racket
(require "FNN.rkt")

;; Define architecture: 784 → 128 → 64 → 10
(define-values (weights biases) 
  (initialize-neural-network '(784 128 64 10)))

;; Forward pass
(define-values (output hidden-activations z-values)
  (forward-pass input weights biases relu))

;; If you have MNIST data loaded:
(define-values (trained-w trained-b)
  (train-fnn X-train y-train X-val y-val
             '(128 64)  ; hidden layer sizes
             0.01       ; learning rate
             10         ; epochs
             32))       ; batch size
```

### CNN (LeNet-5 Architecture)

```racket
(require "CNN.rkt")

;; Train on MNIST (auto-detects best device)
(train-cnn 'cpu 5 32)  ; device, epochs, batch-size
(train-cnn 'mlx 10 64) ; Apple Silicon acceleration
```

## Project Structure

| File | Description |
|------|-------------|
| `tensor.rkt` | Core tensor data structure and operations |
| `autograd.rkt` | Activation functions and dense layer forward/backward |
| `FNN.rkt` | Feedforward neural network with training loop |
| `CNN.rkt` | Convolutional neural network (LeNet-5) |
| `device.rkt` | Device abstraction (CPU/GPU/MLX/CUDA) |
| `tensor_device.rkt` | Device-aware tensor operations |
| `hardware_detection.rkt` | Auto-detect available hardware |
| `ffi_ops.rkt` | C FFI bindings for fast math |
| `cnn_ops.rkt` | C FFI bindings for CNN operations |
| `test-suite.rkt` | Comprehensive test suite |
| `simple-test.rkt` | Quick smoke test |

## Optional: Compile C Extensions

For better performance, compile the C extensions:

```bash
./compile_extensions.sh
```

This enables accelerated matrix multiplication, convolution, and activation functions. Without these, RacoGrad falls back to pure Racket implementations.

## Hardware Acceleration

RacoGrad auto-detects and uses the best available hardware:

- **Apple Silicon (MLX)** — preferred on M1/M2/M3 Macs
- **NVIDIA GPU (CUDA)** — if available
- **OpenCL** — cross-platform GPU fallback
- **CPU with SIMD** — AVX/SSE optimizations
- **Pure Racket** — always works, no dependencies

## Contributing

Contributions welcome! Please run `racket test-suite.rkt` before submitting.

## License

MIT License
