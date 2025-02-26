# RacoGrad Implementation Details

This document provides a comprehensive overview of the implementation details of RacoGrad, a deep learning framework written in Racket. It covers the architecture, design decisions, and technical implementation details of each component.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Tensor Implementation](#tensor-implementation)
3. [Automatic Differentiation](#automatic-differentiation)
4. [Convolutional Neural Networks](#convolutional-neural-networks)
5. [Device Management](#device-management)
6. [Hardware Acceleration](#hardware-acceleration)
7. [MNIST Implementation](#mnist-implementation)
8. [Optimization Strategies](#optimization-strategies)
9. [C/C++ Extensions](#cc-extensions)
10. [Future Improvements](#future-improvements)

## System Architecture

RacoGrad is organized in a modular fashion with different components handling specific aspects of deep learning:

```
- tensor.rkt           # Core tensor operations
- tensor_device.rkt    # Device-aware tensor operations
- autograd.rkt         # Automatic differentiation
- device.rkt           # Device management
- hardware_detection.rkt # Hardware capability detection
- cnn_ops.rkt          # CNN operations
- CNN.rkt              # CNN implementation (LeNet)
- mnist.rkt            # MNIST dataset handling and training
- ffi_ops.rkt          # Foreign function interface to C extensions
- *.c/*.so/*.dylib     # Compiled C extensions
```

The system follows these key principles:
- Modularity: Functions are grouped by purpose
- Extensibility: Each component can be extended independently
- Performance: Performance-critical operations are implemented in C
- Device awareness: Operations adapt to the hardware they run on

## Tensor Implementation

### Basic Tensor (`tensor.rkt`)

The basic tensor implementation uses Racket's built-in vectors to store data:

```racket
(struct tensor (shape data) #:transparent)

;; Creating a tensor
(define (t:create shape data)
  (tensor shape (if (vector? data) data (list->vector data))))

;; Tensor operations (example)
(define (t:add t1 t2)
  (let* ([shape1 (t:shape t1)]
         [shape2 (t:shape t2)]
         [_ (check-shapes-match shape1 shape2 "t:add")]
         [size (apply * shape1)]
         [result (make-vector size 0)])
    (for ([i (in-range size)])
      (vector-set! result i
                   (+ (vector-ref (t:data t1) i)
                      (vector-ref (t:data t2) i))))
    (t:create shape1 result)))
```

Key design aspects:
- Tensors are immutable - operations return new tensors
- Shape checking is performed for operations
- Operations are element-wise unless specified otherwise (like matrix multiplication)
- Broadcasting is supported for certain operations (add, subtract, multiply)

### Device-Aware Tensor (`tensor_device.rkt`)

The device-aware tensor extends the basic tensor with device information:

```racket
(struct dt:tensor (shape data device) #:transparent)

(define (dt:create shape data [dev (current-device)])
  (let ([t (t:create shape data)])
    (dt:tensor (t:shape t) (t:data t) dev)))
```

Operations are dispatched based on the device:

```racket
(define (dt:add t1 t2)
  (cond
    [(and (dt:tensor? t1) (dt:tensor? t2))
     (let ([dev1 (dt:tensor-device t1)]
           [dev2 (dt:tensor-device t2)])
       (cond
         ;; Both on CPU
         [(and (cpu-device? dev1) (cpu-device? dev2))
          (let ([regular-t1 (t:create (dt:tensor-shape t1) (dt:tensor-data t1))]
                [regular-t2 (t:create (dt:tensor-shape t2) (dt:tensor-data t2))])
            (let ([sum (t:add regular-t1 regular-t2)])
              (dt:tensor (t:shape sum) (t:data sum) dev1)))]
         
         ;; Different devices - move to same device
         [(not (equal? dev1 dev2))
          (let ([target-dev (if (gpu-device? dev1) dev1 dev2)])
            (dt:add (dt:to t1 target-dev) (dt:to t2 target-dev)))]
         
         ;; Other cases...
         ))]
    ;; Handle regular tensors...
    ))
```

## Automatic Differentiation

The autograd system (`autograd.rkt`) implements manual backpropagation for a feedforward neural network:

```racket
;; Forward pass for dense layer
(define (dense-forward input weights biases activation-fn)
  (let* ([mul-result (t:mul input weights)]
         [output-dim (cadr (t:shape mul-result))]
         [reshaped-biases (t:reshape biases (list output-dim))]
         [z (t:add mul-result reshaped-biases)]
         [activation-output (activation-fn z)])
    activation-output))

;; Backward pass for dense layer
(define (dense-backward input weights biases output grad-output 
                        activation-derivative learning-rate)
  (let* ([grad-activation (activation-derivative output)]
         [grad-z (t:mul grad-output grad-activation)]
         [grad-weights (t:mul (t:transpose input) grad-z)]
         [bias-len (vector-length (t:data biases))]
         [grad-biases (t:create (list bias-len)
                                (for/vector ([j bias-len])
                                  (apply +
                                         (for/list ([i (car (t:shape grad-z))])
                                           (vector-ref (t:data grad-z)
                                                       (+ (* i bias-len) j))))))]
         [grad-input (t:mul grad-z (t:transpose weights))])
    (values grad-weights grad-biases grad-input)))
```

Key features:
- Manual gradient computation for dense (fully connected) layers
- Support for various activation functions (ReLU, sigmoid, tanh)
- Mean squared error loss function
- No dynamic computation graph - gradients are computed manually

### Activation Functions

```racket
(define (relu x)
  (t:create (t:shape x) 
            (for/vector ([v (t:data x)]) (max 0 v))))

(define (relu-derivative x)
  (t:create (t:shape x) 
            (for/vector ([v (t:data x)]) (if (> v 0) 1 0))))

(define (sigmoid x)
  (t:create (t:shape x)
            (for/vector ([v (t:data x)]) (/ 1 (+ 1 (exp (- v)))))))
```

## Convolutional Neural Networks

### CNN Operations (`cnn_ops.rkt`)

CNN operations are implemented in C for performance and exposed via FFI:

```racket
;; Load CPU implementations
(define cnn-lib (ffi-lib "cnn_ops"))

;; Define FFI bindings
(define c:conv2d-forward
  (get-ffi-obj "conv2d_forward" cnn-lib
               (_fun _int _int _int _int    ; batch_size, in_channels, in_height, in_width
                     _int _int _int         ; out_channels, filter_height, filter_width
                     _int _int              ; stride, padding
                     _f64vector _f64vector _f64vector -> _void)))
```

The system can also load optimized implementations based on the current device:

```racket
;; Try to load MLX implementations if on Apple Silicon
(define mlx-cnn-lib
  (with-handlers ([exn:fail? (lambda (e) #f)])
    (ffi-lib "mlx_cnn_ops")))

;; Function to reload operations based on device
(define (load-optimal-ops [dev (current-device)])
  (cond
    [(mlx-device? dev)
     (printf "Loading MLX optimized CNN operations for Apple Silicon~n")
     (when mlx:conv2d-forward
       (set! c:conv2d-forward mlx:conv2d-forward))
     ...]))
```

### CNN Implementation (`CNN.rkt`)

The CNN.rkt module implements a LeNet-5 style CNN:

```racket
(define (make-lenet [dev (current-device)]
                    [c1f #f] [c1b #f] [c2f #f] [c2b #f]
                    [f1w #f] [f1b #f] [f2w #f] [f2b #f]
                    [f3w #f] [f3b #f])
  ;; Create or reuse model parameters
  (let ([conv1-filters (if c1f c1f (dt:random (list 6 3 5 5) 0.1 dev))]
        [conv1-bias (if c1b c1b (dt:random (list 1 6) 0.1 dev))]
        ...))
```

The forward pass function is created and returned:

```racket
(define (forward-pass input)
  (let* ([input-with-channels (if (= (length (dt:shape input)) 3)
                                   ;; Add batch dimension if missing
                                   (dt:reshape input (list 1 (car (dt:shape input))
                                                          (cadr (dt:shape input))
                                                          (caddr (dt:shape input))))
                                   input)]
         ;; Compute convolution output
         [conv_out (conv2d input-with-channels conv1-filters 1 2)]
         ;; Apply bias, activation, and pooling
         ...
         ;; Final output
         [output (softmax fc3)])
    output))
```

Backpropagation is implemented by computing gradients and updating parameters:

```racket
;; In train-cnn function:
;; Backpropagation
(define output-grad 
  (let* ([pred-data (dt:data predictions)]
         [label-data (dt:data batch-labels)]
         [batch-size (car (dt:shape predictions))]
         [num-classes (cadr (dt:shape predictions))]
         [device (dt:device predictions)]
         [grad-data (make-vector (* batch-size num-classes) 0.0)])
    ;; Calculate gradient
    (for ([i (in-range (* batch-size num-classes))])
      (vector-set! grad-data i
                   (/ (- (vector-ref pred-data i)
                        (vector-ref label-data i))
                     batch-size)))
    (dt:create (dt:shape predictions) grad-data device)))

;; Backprop through fully connected layers
(define-values (grad-fc3-w grad-fc3-b grad-fc3)
  (compute-fc-gradients fc2-a output-grad fc3-weights fc3-bias))
...

;; Update parameters
(set! fc3-weights (update-params fc3-weights grad-fc3-w learning-rate))
(set! fc3-bias (update-params fc3-bias grad-fc3-b learning-rate))
...
```

## Device Management

### Device Structures (`device.rkt`)

Devices are represented as structures:

```racket
(struct device (type id) #:transparent)
(struct cpu-device device () #:transparent)
(struct gpu-device device ([platform #:mutable] [device-obj #:mutable]) #:transparent)
(struct mlx-device device () #:transparent)

;; Functions to create devices
(define (cpu) (cpu-device 'cpu 0))
(define (gpu [platform-idx 0] [device-idx 0])
  (gpu-device 'gpu 0 platform-idx device-idx))
(define (mlx) (mlx-device 'mlx 0))
```

A global parameter maintains the current device:

```racket
;; Current device parameter
(define current-device (make-parameter (cpu)))

;; Set current device
(define (set-current-device! dev)
  (current-device dev))
```

### Hardware Detection (`hardware_detection.rkt`)

Hardware capabilities are detected at runtime:

```racket
;; Check if GPU is available
(define (gpu-available?)
  (with-handlers ([exn:fail? (lambda (e) #f)])
    (let ([platform-count (cl_get_platform_count)])
      (> platform-count 0))))

;; Check if MLX is available (Apple Silicon)
(define (has-mlx?)
  (and (string=? (system-type 'machine) "aarch64")
       (string=? (system-type 'os) "macosx")
       (file-exists? "/usr/local/lib/libmlx.dylib")))
```

## MNIST Implementation

The MNIST module (`mnist.rkt`) implements logistic regression for digit classification:

```racket
;; Load MNIST data
(define (load-mnist-data type)
  (let* ([base-path "/Users/marq/Documents/racograd/mnist-data/"]
         [images-file (string-append base-path 
                                     (if (equal? type "train")
                                         "train-images.idx3-ubyte"
                                         "t10k-images.idx3-ubyte"))]
         [labels-file (string-append base-path 
                                     (if (equal? type "train")
                                         "train-labels.idx1-ubyte"
                                         "t10k-labels.idx1-ubyte"))])
    (values (read-idx3-ubyte images-file)
            (read-idx1-ubyte labels-file))))
```

The training loop:

```racket
;; Training loop using autograd
(define (train-batch X-batch y-batch)
  (let* ([y-pred (forward X-batch weights bias)]
         [loss (cross-entropy y-pred y-batch)]
         [batch-size (car (t:shape y-batch))]
         [dloss (t:sub y-pred y-batch)]  ; gradient is (pred - true)
         
         ;; Compute gradients
         [gradient-w (t:scale (t:mul (t:transpose X-batch) dloss) 
                            (/ 1.0 batch-size))]
         [gradient-b (t:create (t:shape bias)
                             (for/list ([j (in-range num-classes)])
                               (/ (for/sum ([i (in-range batch-size)])
                                    (vector-ref (t:data dloss) (+ (* i num-classes) j)))
                                  batch-size)))])
    
    ;; Update parameters
    (set! weights (t:sub weights (t:scale gradient-w learning-rate)))
    (set! bias (t:sub bias (t:scale gradient-b learning-rate)))
    
    loss))
```

Validation and early stopping:

```racket
;; Define early stopping parameters
(define patience 3)
(define min-delta 0.001)
(define wait 0)
(define best-val-accuracy 0.0)

;; Check early stopping condition
(if (>= wait patience)
    (begin
      (printf "Early stopping triggered after ~a epochs without improvement.~n" patience)
      #t)  ; Stop training
    #f))  ; Continue training
```

## C/C++ Extensions

The C extensions are compiled with:

```bash
# compile_extensions.sh
gcc -fPIC -shared -o matrix_multiplication.so matrix_multiplication.c
gcc -fPIC -shared -o cnn_ops.dylib cnn_ops.c
```

Example C implementation (simplified):

```c
// cnn_ops.c
void conv2d_forward(int batch_size, int in_channels, int in_height, int in_width,
                   int out_channels, int filter_height, int filter_width,
                   int stride, int padding,
                   double* input, double* filters, double* output) {
    // Calculate output dimensions
    int out_height = ((in_height + 2 * padding - filter_height) / stride) + 1;
    int out_width = ((in_width + 2 * padding - filter_width) / stride) + 1;
    
    // For each batch, output channel, output position
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < out_channels; o++) {
            for (int h = 0; h < out_height; h++) {
                for (int w = 0; w < out_width; w++) {
                    double sum = 0.0;
                    
                    // For each input channel, filter position
                    for (int c = 0; c < in_channels; c++) {
                        for (int fh = 0; fh < filter_height; fh++) {
                            for (int fw = 0; fw < filter_width; fw++) {
                                int h_pos = h * stride - padding + fh;
                                int w_pos = w * stride - padding + fw;
                                
                                // Check if position is valid
                                if (h_pos >= 0 && h_pos < in_height && 
                                    w_pos >= 0 && w_pos < in_width) {
                                    double input_val = input[
                                        b * (in_channels * in_height * in_width) +
                                        c * (in_height * in_width) +
                                        h_pos * in_width +
                                        w_pos
                                    ];
                                    
                                    double filter_val = filters[
                                        o * (in_channels * filter_height * filter_width) +
                                        c * (filter_height * filter_width) +
                                        fh * filter_width +
                                        fw
                                    ];
                                    
                                    sum += input_val * filter_val;
                                }
                            }
                        }
                    }
                    
                    // Set output value
                    output[
                        b * (out_channels * out_height * out_width) +
                        o * (out_height * out_width) +
                        h * out_width +
                        w
                    ] = sum;
                }
            }
        }
    }
}
```

## Optimization Strategies

Several optimization strategies are employed:

1. **Tensor Operations**: Element-wise operations are optimized in C
2. **Batching**: Mini-batch processing to improve throughput
3. **Hardware Acceleration**: Using specialized hardware when available
4. **Memory Management**: Reusing tensors when possible
5. **Early Stopping**: To prevent overfitting and reduce training time

Example of hardware-accelerated operations:

```racket
;; Selecting the optimal implementation in CNN.rkt
(load-optimal-ops (current-device))
```

## Future Improvements

Potential future improvements include:

1. **Full Autograd**: Implementing a dynamic computation graph for automatic differentiation
2. **More Models**: Adding support for other neural network architectures
3. **CUDA Support**: Direct CUDA support for NVIDIA GPUs
4. **Distributed Training**: Support for distributed training across multiple machines
5. **Optimizers**: Implementing more optimizers like Adam, RMSProp, etc.
6. **Broadcasting**: More flexible tensor broadcasting
7. **Serialization**: Better model saving/loading with metadata

## Code Examples

### LeNet Forward Pass

```racket
(define (forward-pass input)
  (let* ([input-with-channels (if (= (length (dt:shape input)) 3)
                                   ;; Add batch dimension if missing
                                   (dt:reshape input (list 1 (car (dt:shape input))
                                                          (cadr (dt:shape input))
                                                          (caddr (dt:shape input))))
                                   input)]
         
         ;; Compute convolution output
         [conv_out (conv2d input-with-channels conv1-filters 1 2)]
         
         ;; Apply bias manually without broadcasting
         [conv_shape (dt:shape conv_out)]
         [batch-size (car conv_shape)]
         [channels (cadr conv_shape)]
         [height (caddr conv_shape)]
         [width (cadddr conv_shape)]
         [conv_data (vector->list (dt:data conv_out))]
         [bias_data (vector->list (dt:data conv1-bias))]
         
         ;; Apply bias to each channel
         [conv_with_bias (for/vector ([i (in-range (* batch-size channels height width))])
                          (let* ([batch-idx (quotient i (* channels height width))]
                                [within-batch-idx (remainder i (* channels height width))]
                                [channel-idx (quotient within-batch-idx (* height width))]
                                [bias-val (list-ref bias_data channel-idx)])
                            (+ (list-ref conv_data i) bias-val)))]
         
         ;; Create tensor from biased data
         [conv1 (dt:create conv_shape conv_with_bias (dt:device conv_out))]
         [relu1 (relu conv1)]
         [pool1 (max-pool-2x2 relu1)]
         
         ;; Second convolutional layer
         [conv2_out (conv2d pool1 conv2-filters 1 0)]
         
         ;; Apply bias manually
         [conv2_shape (dt:shape conv2_out)]
         [batch-size2 (car conv2_shape)]
         [channels2 (cadr conv2_shape)]
         [height2 (caddr conv2_shape)]
         [width2 (cadddr conv2_shape)]
         [conv2_data (vector->list (dt:data conv2_out))]
         [bias2_data (vector->list (dt:data conv2-bias))]
         [conv2_with_bias (for/vector ([i (in-range (* batch-size2 channels2 height2 width2))])
                          (let* ([batch-idx (quotient i (* channels2 height2 width2))]
                                [within-batch-idx (remainder i (* channels2 height2 width2))]
                                [channel-idx (quotient within-batch-idx (* height2 width2))]
                                [bias-val (list-ref bias2_data channel-idx)])
                            (+ (list-ref conv2_data i) bias-val)))]
         
         [conv2 (dt:create conv2_shape conv2_with_bias (dt:device conv2_out))]
         [relu2 (relu conv2)]
         [pool2 (max-pool-2x2 relu2)]
         
         ;; Flatten and fully connected layers
         [flat (flatten pool2)]
         
         [fc1 (fc-layer flat fc1-weights fc1-bias relu)]
         [fc2 (fc-layer fc1 fc2-weights fc2-bias relu)]
         [fc3 (fc-layer fc2 fc3-weights fc3-bias)]
         
         ;; Softmax for classification
         [output (softmax fc3)])
    
    output))
```

### Tensor Add Implementation

```racket
(define (t:add t1 t2)
  (let* ([shape1 (t:shape t1)]
         [shape2 (t:shape t2)]
         [broadcasted-shapes? (not (equal? shape1 shape2))]
         [result-shape (if broadcasted-shapes?
                         (broadcast-shapes shape1 shape2)
                         shape1)])
    
    (if broadcasted-shapes?
        ;; Handle broadcasting
        (let* ([size (apply * result-shape)]
               [result-data (make-vector size 0.0)])
          ;; Implement broadcasting here
          (t:create result-shape result-data))
        
        ;; Non-broadcasting case (shapes match)
        (let* ([size (apply * shape1)]
               [result-data (make-vector size 0.0)])
          (for ([i (in-range size)])
            (vector-set! result-data i
                         (+ (vector-ref (t:data t1) i)
                            (vector-ref (t:data t2) i))))
          (t:create shape1 result-data)))))
```

This detailed documentation should give you a comprehensive understanding of the implementation details of RacoGrad.