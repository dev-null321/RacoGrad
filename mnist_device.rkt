#lang racket

(require "tensor_device.rkt"
         "device.rkt"
         "autograd.rkt")

;; MNIST Loading functions
(define (read-idx3-ubyte filename)
  (let* ([p (open-input-file filename #:mode 'binary)]
         [magic-number (integer-bytes->integer (read-bytes 4 p) #f #t)]
         [num-images (integer-bytes->integer (read-bytes 4 p) #f #t)]
         [num-rows (integer-bytes->integer (read-bytes 4 p) #f #t)]
         [num-cols (integer-bytes->integer (read-bytes 4 p) #f #t)]
         [data (make-vector (* num-images num-rows num-cols) 0)])
    (for ([i (in-range (vector-length data))])
      (vector-set! data i (read-byte p)))
    (close-input-port p)
    (dt:create (list num-images (* num-rows num-cols)) data)))

(define (read-idx1-ubyte filename)
  (let* ([p (open-input-file filename #:mode 'binary)]
         [magic-number (integer-bytes->integer (read-bytes 4 p) #f #t)]
         [num-items (integer-bytes->integer (read-bytes 4 p) #f #t)]
         [data (make-vector num-items 0)])
    (for ([i (in-range num-items)])
      (vector-set! data i (read-byte p)))
    (close-input-port p)
    (dt:create (list num-items 1) data)))

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
    (printf "Looking for images-file at: ~a~n" images-file)
    (printf "Looking for labels-file at: ~a~n" labels-file)

    ;; Error checking
    (unless (file-exists? images-file)
      (error 'load-mnist-data 
             (format "MNIST images file not found: ~a" images-file)))
    (unless (file-exists? labels-file)
      (error 'load-mnist-data 
             (format "MNIST labels file not found: ~a" labels-file)))
    (values (read-idx3-ubyte images-file)
            (read-idx1-ubyte labels-file))))

;; Normalize the data
(define (normalize X)
  (dt:scale X (/ 1.0 255.0)))

;; One-hot encode labels
(define (one-hot y num-classes)
  (let* ([num-samples (car (dt:shape y))]
         [encoded (make-vector (* num-samples num-classes) 0.0)])
    (for ([i (in-range num-samples)])
      (vector-set! encoded (+ (* i num-classes) (vector-ref (dt:data y) i)) 1.0))
    (dt:create (list num-samples num-classes) encoded)))

;; Get a batch of data given indices
(define (get-batch X indices)
  (let* ([batch-size (length indices)]
         [feature-size (cadr (dt:shape X))]
         [batch-data (make-vector (* batch-size feature-size) 0.0)])
    (for ([i (range batch-size)]
          [idx indices])
      (for ([j (range feature-size)])
        (vector-set! batch-data 
                     (+ (* i feature-size) j)
                     (vector-ref (dt:data X) (+ (* idx feature-size) j)))))
    (dt:create (list batch-size feature-size) batch-data)))

;; Softmax function
(define (softmax z)
  (let* ([shape (dt:shape z)]
         [data (dt:data z)]
         [max-vals (for/vector ([i (in-range (car shape))])
                    (apply max (for/list ([j (in-range (cadr shape))])
                               (vector-ref data (+ (* i (cadr shape)) j)))))]
         [exp-vals (for/vector ([i (in-range (vector-length data))])
                    (exp (- (vector-ref data i) 
                           (vector-ref max-vals (quotient i (cadr shape))))))]
         [sum-vals (for/vector ([i (in-range (car shape))])
                    (for/sum ([j (in-range (cadr shape))])
                      (vector-ref exp-vals (+ (* i (cadr shape)) j))))])
    (dt:create shape
              (for/vector ([i (in-range (vector-length exp-vals))])
                (/ (vector-ref exp-vals i)
                   (vector-ref sum-vals (quotient i (cadr shape))))))))

;; Forward pass
(define (broadcast-bias b batch-size)
  (let* ([bias-shape (dt:shape b)]
         [num-classes (cadr bias-shape)]
         [expanded-data (make-vector (* batch-size num-classes) 0.0)])
    (for ([i (in-range batch-size)])
      (for ([j (in-range num-classes)])
        (vector-set! expanded-data (+ (* i num-classes) j)
                     (vector-ref (dt:data b) j))))
    (dt:create (list batch-size num-classes) expanded-data)))

(define (forward X w b)
  (let* ([z (dt:add (dt:mul X w) (broadcast-bias b (car (dt:shape X))))]) ; Broadcast b
    (softmax z)))

;; Cross-entropy loss
(define (cross-entropy y-pred y-true)
  (let* ([m (car (dt:shape y-true))]
         [epsilon 1e-15]
         [y-pred-clipped (dt:create (dt:shape y-pred)
                                 (for/list ([p (vector->list (dt:data y-pred))])
                                   (max (min p (- 1.0 epsilon)) epsilon)))]
         [loss-vec (for/list ([i (in-range (vector-length (dt:data y-true)))])
                    (* (vector-ref (dt:data y-true) i)
                       (log (vector-ref (dt:data y-pred-clipped) i))))])
    (- (/ (apply + loss-vec) m))))

;; Initialize parameters
(define input-size 784)  ; 28x28 pixels
(define num-classes 10)  ; digits 0-9

;; Run the training process with a specified device
(provide train-mnist)

(define (train-mnist device-type batch-size [epochs 10])
  (printf "~nStarting MNIST training on device: ~a~n" device-type)
  
  ;; Set the device for computation
  (cond
    [(eq? device-type 'cpu)
     (set-current-device! (cpu))]
    [(eq? device-type 'mlx)
     (if (device-available? 'mlx)
         (set-current-device! (mlx))
         (begin 
           (printf "MLX not available. Falling back to CPU.\n")
           (set-current-device! (cpu))))]
    [(eq? device-type 'cuda)
     (if (device-available? 'cuda)
         (set-current-device! (cuda))
         (begin 
           (printf "CUDA not available. Falling back to CPU.\n")
           (set-current-device! (cpu))))]
    [(eq? device-type 'opencl)
     (if (device-available? 'opencl)
         (set-current-device! (opencl))
         (begin 
           (printf "OpenCL not available. Falling back to CPU.\n")
           (set-current-device! (cpu))))]
    [(eq? device-type 'gpu)
     (if (gpu-available?)
         (set-current-device! (gpu))
         (begin 
           (printf "No GPU available. Falling back to CPU.\n")
           (set-current-device! (cpu))))]
    [else 
     (printf "Unknown device type: ~a. Using CPU.\n" device-type)
     (set-current-device! (cpu))])
  
  (printf "Using device: ~a~n" (get-device-type (current-device)))
  
  ;; Create model parameters on the selected device
  (define weights (dt:random (list input-size num-classes) 0.01))  ; (input-size, num-classes)
  (define bias (dt:random (list 1 num-classes) 0.01))              ; (1, num-classes)
  
  ;; Training hyperparameters
  (define learning-rate 0.1)
  
  ;; Load and initialize MNIST data
  (printf "Loading MNIST data...~n")
  (define-values (X-train y-train) (load-mnist-data "train")) ; Load training data
  (define-values (X-test y-test) (load-mnist-data "test"))    ; Load test data
  
  (printf "Normalizing data...~n")
  (set! X-train (normalize X-train)) ; Normalize X-train
  (set! X-test (normalize X-test))   ; Normalize X-test
  
  (printf "One-hot encoding labels...~n")
  (set! y-train (one-hot y-train 10)) ; Convert y-train to one-hot
  (set! y-test (one-hot y-test 10))   ; Convert y-test to one-hot
  
  ;; Move data to device if using GPU
  (when (eq? device-type 'gpu)
    (printf "Moving data to GPU...~n")
    (set! X-train (dt:to X-train (current-device)))
    (set! y-train (dt:to y-train (current-device)))
    (set! X-test (dt:to X-test (current-device)))
    (set! y-test (dt:to y-test (current-device))))
  
  ;; Training loop
  (define (train-batch X-batch y-batch)
    (let* ([y-pred (forward X-batch weights bias)]
           [loss (cross-entropy y-pred y-batch)]
           [error (dt:sub y-pred y-batch)]
           [gradient-w (dt:scale (dt:mul (dt:transpose X-batch) error) 
                              (/ 1.0 (car (dt:shape X-batch))))]
           [gradient-b (dt:create (dt:shape bias)
                               (for/list ([j (in-range num-classes)])
                                 (/ (for/sum ([i (in-range (car (dt:shape error)))])
                                      (vector-ref (dt:data error) (+ (* i num-classes) j)))
                                    (car (dt:shape error)))))])
      ;; Update parameters
      (set! weights (dt:sub weights (dt:scale gradient-w learning-rate)))
      (set! bias (dt:sub bias (dt:scale gradient-b learning-rate)))
      loss))
  
  (define (get-test-accuracy X y)
    (let* ([predictions (forward X weights bias)]
           [num-samples (car (dt:shape X))]
           [num-classes (cadr (dt:shape y))]
           [pred-data (dt:data predictions)]
           [true-data (dt:data y)]
           [correct-count 0])
      
      ; Count correct predictions
      (for ([i (range num-samples)])
        (let* ([start-idx (* i num-classes)]
               ; Find predicted class (max probability index)
               [pred-vals (for/list ([j (range num-classes)])
                           (vector-ref pred-data (+ start-idx j)))]
               [pred-class (argmax (lambda (j) (list-ref pred-vals j))
                                 (range num-classes))]
               ; Find true class 
               [true-class (for/first ([j (range num-classes)]
                                     #:when (= 1.0 (vector-ref true-data (+ start-idx j))))
                            j)])
          ; Increment counter if prediction matches truth
          (when (= pred-class true-class)
            (set! correct-count (add1 correct-count)))))
      
      ; Return accuracy as percentage
      (exact->inexact (* 100.0 (/ correct-count num-samples)))))
  
  ;; Track time for benchmarking
  (define start-time (current-inexact-milliseconds))
  
  ;; Main training loop
  (printf "Starting training...~n")
  (printf "Training on ~a samples, validating on ~a samples~n~n" 
          (car (dt:shape X-train)) 
          (car (dt:shape X-test)))
  
  (define best-accuracy 0.0)
  
  (for ([epoch (in-range epochs)])
    (printf "Epoch ~a/~a:~n" (add1 epoch) epochs)
    
    (let* ([indices (shuffle (range (car (dt:shape X-train))))]
           [num-batches (quotient (length indices) batch-size)]
           [epoch-losses '()])
      
      ; Train on batches
      (for ([batch (in-range num-batches)])
        (let* ([batch-indices (take (drop indices (* batch batch-size)) batch-size)]
               [X-batch (get-batch X-train batch-indices)]
               [y-batch (get-batch y-train batch-indices)]
               [loss (train-batch X-batch y-batch)])
          
          (set! epoch-losses (cons loss epoch-losses))
          
          (when (= (modulo batch 50) 0)
            (printf "  Batch ~a/~a - Loss: ~a~n" 
                    batch 
                    num-batches 
                    (real->decimal-string loss 4)))))
      
      ; Epoch evaluation
      (let* ([avg-loss (/ (apply + epoch-losses) (length epoch-losses))]
             [test-accuracy (get-test-accuracy X-test y-test)])
        
        (when (> test-accuracy best-accuracy)
          (set! best-accuracy test-accuracy))
        
        (printf "~nEpoch Summary:~n")
        (printf "  Average Loss: ~a~n" (real->decimal-string avg-loss 4))
        (printf "  Test Accuracy: ~a%~n" (real->decimal-string test-accuracy 2))
        (printf "  Best Accuracy So Far: ~a%~n~n" (real->decimal-string best-accuracy 2)))))
  
  (define end-time (current-inexact-milliseconds))
  (define total-time (/ (- end-time start-time) 1000.0))
  
  (printf "Training Complete!~n")
  (printf "Final Test Accuracy: ~a%~n" 
          (real->decimal-string (get-test-accuracy X-test y-test) 2))
  (printf "Total training time: ~a seconds~n" (real->decimal-string total-time 2))
  
  ;; Return benchmark results
  (hash 'accuracy (get-test-accuracy X-test y-test)
        'time total-time
        'device device-type))

;; When executed directly, run on both CPU and GPU (if available)
(module+ main
  (printf "Starting CPU training...~n")
  (define cpu-result (train-mnist 'cpu 128 2))
  
  (if (gpu-available?)
      (let ([gpu-result (train-mnist 'gpu 128 2)])
        (printf "~n~n===== PERFORMANCE COMPARISON =====~n")
        (printf "CPU Training Time: ~a seconds~n" (real->decimal-string (hash-ref cpu-result 'time) 2))
        (printf "GPU Training Time: ~a seconds~n" (real->decimal-string (hash-ref gpu-result 'time) 2))
        (printf "Speedup: ~a times faster~n" 
                (real->decimal-string (/ (hash-ref cpu-result 'time) (hash-ref gpu-result 'time)) 2))
        (printf "CPU Accuracy: ~a%~n" (real->decimal-string (hash-ref cpu-result 'accuracy) 2))
        (printf "GPU Accuracy: ~a%~n" (real->decimal-string (hash-ref gpu-result 'accuracy) 2))
        (printf "===============================~n"))
      (printf "~nGPU acceleration not available on this system. Skipping GPU training.~n")))