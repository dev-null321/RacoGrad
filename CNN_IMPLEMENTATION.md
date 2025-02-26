# Implementing a CNN with Optimized RacoGrad

This guide shows how to implement a Convolutional Neural Network (CNN) using the optimized RacoGrad library components.

## Prerequisites

- Compiled C extensions (`matrix_multiplication.so`, `simd_ops.so`, `parallel_ops.so`)
- OpenCL setup (optional, for GPU acceleration)
- Optimized tensor implementation (`tensor_optimized.rkt`)

## Step 1: Import Required Modules

```racket
#lang racket

(require "tensor_optimized.rkt"  ; Optimized tensor operations
         "ffi_ops.rkt"          ; C extension bindings
         "load-mnist.rkt")      ; For MNIST dataset loading
```

## Step 2: Implement Optimized Convolution Layer

The convolution layer is critical for CNN performance. Here's an implementation using C extensions:

```racket
;; Forward pass through a convolution layer
(define (conv2d-forward input-tensor filter-tensor stride padding)
  (let* ([input-shape (t-opt:shape input-tensor)]
         [filter-shape (t-opt:shape filter-tensor)]
         [batch-size (car input-shape)]
         [in-channels (cadr input-shape)]
         [in-height (caddr input-shape)]
         [in-width (cadddr input-shape)]
         [out-channels (car filter-shape)]
         [filter-height (caddr filter-shape)]
         [filter-width (cadddr filter-shape)]
         
         ;; Calculate output dimensions
         [out-height (add1 (quotient (- (+ in-height (* 2 padding)) filter-height) stride))]
         [out-width (add1 (quotient (- (+ in-width (* 2 padding)) filter-width) stride))]
         
         ;; Create output tensor
         [output-tensor (t-opt:create (list batch-size out-channels out-height out-width)
                                     (make-vector (* batch-size out-channels out-height out-width) 0.0))])
    
    ;; Call the C extension for convolution
    (c:conv2d-forward 
      batch-size in-channels in-height in-width
      out-channels filter-height filter-width
      stride padding
      (t-opt:data input-tensor)
      (t-opt:data filter-tensor)
      (t-opt:data output-tensor))
    
    output-tensor))
```

## Step 3: Implement Pooling Layer

Max pooling is another critical operation for CNNs:

```racket
;; Max pooling layer (2x2 pooling with stride 2)
(define (max-pool-2x2 input-tensor)
  (let* ([input-shape (t-opt:shape input-tensor)]
         [batch-size (car input-shape)]
         [channels (cadr input-shape)]
         [in-height (caddr input-shape)]
         [in-width (cadddr input-shape)]
         
         ;; Output dimensions
         [out-height (quotient in-height 2)]
         [out-width (quotient in-width 2)]
         
         ;; Create output tensor
         [output-tensor (t-opt:create (list batch-size channels out-height out-width)
                                     (make-vector (* batch-size channels out-height out-width) 0.0))])
    
    ;; Call the C extension for max pooling
    (c:max-pool-2x2 
      batch-size channels in-height in-width
      (t-opt:data input-tensor)
      (t-opt:data output-tensor))
    
    output-tensor))
```

## Step 4: Implement the Fully Connected Layer

Using our optimized matrix multiplication:

```racket
;; Fully connected layer using optimized operations
(define (fc-layer input-tensor weights-tensor bias-tensor activation-fn)
  (let* ([input-flat (if (> (length (t-opt:shape input-tensor)) 2)
                         ;; Flatten input if it's coming from conv layer
                         (flatten-tensor input-tensor)
                         input-tensor)]
         [z (t-opt:add (t-opt:mul input-flat weights-tensor) bias-tensor)])
    (activation-fn z)))
```

## Step 5: Activation Functions

ReLU is the most common activation in CNNs:

```racket
;; ReLU activation using C extension
(define (relu-opt tensor)
  (let* ([shape (t-opt:shape tensor)]
         [size (apply * shape)]
         [result (make-f64vector size 0.0)])
    (c:relu-forward size (t-opt:data tensor) result)
    (t-opt:create shape result)))
```

## Step 6: Building the Complete CNN

Put everything together to build a CNN for MNIST:

```racket
;; Define model parameters
(define batch-size 64)
(define learning-rate 0.01)

;; Create model (LeNet-5 architecture)
(define (create-lenet)
  (let ([conv1-filters (t-opt:random (list 6 1 5 5) 0.1)]      ; 6 5x5 filters
        [conv1-bias (t-opt:random (list 6) 0.1)]
        
        [conv2-filters (t-opt:random (list 16 6 5 5) 0.1)]     ; 16 5x5 filters
        [conv2-bias (t-opt:random (list 16) 0.1)]
        
        [fc1-weights (t-opt:random (list 400 120) 0.1)]        ; 400 -> 120
        [fc1-bias (t-opt:random (list 120) 0.1)]
        
        [fc2-weights (t-opt:random (list 120 84) 0.1)]         ; 120 -> 84
        [fc2-bias (t-opt:random (list 84) 0.1)]
        
        [fc3-weights (t-opt:random (list 84 10) 0.1)]          ; 84 -> 10
        [fc3-bias (t-opt:random (list 10) 0.1)])
    
    (values conv1-filters conv1-bias
            conv2-filters conv2-bias
            fc1-weights fc1-bias
            fc2-weights fc2-bias
            fc3-weights fc3-bias)))

;; Forward pass through the network
(define (forward-pass x
                      conv1-filters conv1-bias
                      conv2-filters conv2-bias
                      fc1-weights fc1-bias
                      fc2-weights fc2-bias
                      fc3-weights fc3-bias)
  
  ;; First conv layer + relu + pool
  (let* ([conv1 (conv2d-forward x conv1-filters 1 2)]
         [relu1 (relu-opt conv1)]
         [pool1 (max-pool-2x2 relu1)]
         
         ;; Second conv layer + relu + pool
         [conv2 (conv2d-forward pool1 conv2-filters 1 0)]
         [relu2 (relu-opt conv2)]
         [pool2 (max-pool-2x2 relu2)]
         
         ;; Flatten and fully connected layers
         [flatten (flatten-tensor pool2)]
         
         [fc1 (fc-layer flatten fc1-weights fc1-bias relu-opt)]
         [fc2 (fc-layer fc1 fc2-weights fc2-bias relu-opt)]
         [fc3 (fc-layer fc2 fc3-weights fc3-bias identity)])
    
    fc3))
```

## Step 7: Using GPU Acceleration (Optional)

If OpenCL is configured:

```racket
;; Check if GPU is available
(define gpu-available? (c:check-opencl-available))

;; Select appropriate implementation based on hardware
(define conv2d-impl 
  (if gpu-available?
      conv2d-forward-gpu    ; Use GPU implementation
      conv2d-forward))      ; Use CPU implementation
```

## Step 8: Training Loop

An optimized training loop:

```racket
(define (train-epoch! dataset
                      conv1-filters conv1-bias
                      conv2-filters conv2-bias
                      fc1-weights fc1-bias
                      fc2-weights fc2-bias
                      fc3-weights fc3-bias)
  
  ;; Process mini-batches in parallel when possible
  (for ([batch (in-dataset-batches dataset batch-size)])
    (let-values ([(inputs labels) (batch->tensors batch)])
      ;; Forward pass
      (define outputs 
        (forward-pass inputs
                      conv1-filters conv1-bias
                      conv2-filters conv2-bias
                      fc1-weights fc1-bias
                      fc2-weights fc2-bias
                      fc3-weights fc3-bias))
      
      ;; Compute loss (cross-entropy)
      (define loss (cross-entropy-loss outputs labels))
      
      ;; Backward pass and update weights
      ;; ... (omitted for brevity)
      )))
```

## Performance Considerations

1. **Batch Processing**: Use the parallel C extensions to process batches
2. **Memory Reuse**: Use in-place operations when possible
3. **GPU Acceleration**: Identify operations that benefit most from GPU

## Example Usage

```racket
;; Load MNIST data
(define mnist-data (load-mnist))

;; Create model
(let-values ([(conv1-filters conv1-bias
              conv2-filters conv2-bias
              fc1-weights fc1-bias
              fc2-weights fc2-bias
              fc3-weights fc3-bias) (create-lenet)])
  
  ;; Train for multiple epochs
  (for ([epoch 10])
    (printf "Epoch ~a\n" epoch)
    (train-epoch! mnist-data
                 conv1-filters conv1-bias
                 conv2-filters conv2-bias
                 fc1-weights fc1-bias
                 fc2-weights fc2-bias
                 fc3-weights fc3-bias)))
```