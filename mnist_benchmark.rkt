#lang racket

(require "mnist_device.rkt"
         "device.rkt"
         "hardware_detection.rkt")

;; Benchmark on all available devices
(define (benchmark-mnist epochs batch-size)
  (printf "====================================\n")
  (printf "  RACOGRAD MNIST BENCHMARKS\n")
  (printf "====================================\n")
  
  (print-hardware-info)
  
  ;; Run on CPU
  (printf "\n\nRunning MNIST benchmark on CPU...\n")
  (define cpu-result (train-mnist 'cpu batch-size epochs))
  
  (define results (list (cons 'cpu cpu-result)))
  
  ;; Run on MLX if available
  (when (has-mlx-support?)
    (printf "\n\nRunning MNIST benchmark on MLX (Apple Silicon)...\n")
    (define mlx-result (train-mnist 'mlx batch-size epochs))
    (set! results (cons (cons 'mlx mlx-result) results)))
  
  ;; Run on CUDA if available
  (when (has-cuda-support?)
    (printf "\n\nRunning MNIST benchmark on CUDA (NVIDIA GPU)...\n")
    (define cuda-result (train-mnist 'cuda batch-size epochs))
    (set! results (cons (cons 'cuda cuda-result) results)))
  
  ;; Run on OpenCL if available
  (when (has-opencl?)
    (printf "\n\nRunning MNIST benchmark on OpenCL...\n")
    (define opencl-result (train-mnist 'opencl batch-size epochs))
    (set! results (cons (cons 'opencl opencl-result) results)))
  
  ;; Compare results
  (printf "\n\n====================================\n")
  (printf "  PERFORMANCE COMPARISON\n")
  (printf "====================================\n")
  
  (define cpu-time (hash-ref (cdr (assoc 'cpu results)) 'time))
  (define cpu-accuracy (hash-ref (cdr (assoc 'cpu results)) 'accuracy))
  
  (printf "CPU Training Time: ~a seconds\n" (real->decimal-string cpu-time 2))
  (printf "CPU Accuracy: ~a%\n\n" (real->decimal-string cpu-accuracy 2))
  
  (for ([result (in-list results)])
    (unless (eq? (car result) 'cpu)
      (let* ([device-type (car result)]
             [device-result (cdr result)]
             [device-time (hash-ref device-result 'time)]
             [device-accuracy (hash-ref device-result 'accuracy)]
             [speedup (/ cpu-time device-time)])
        (printf "~a Training Time: ~a seconds\n" 
                device-type
                (real->decimal-string device-time 2))
        (printf "~a Accuracy: ~a%\n" 
                device-type
                (real->decimal-string device-accuracy 2))
        (printf "Speedup vs CPU: ~a times faster\n\n" 
                (real->decimal-string speedup 2)))))
  
  (printf "====================================\n"))

;; Run the benchmark when executed directly
(module+ main
  ;; Use smaller values for quick testing
  (benchmark-mnist 2 128))