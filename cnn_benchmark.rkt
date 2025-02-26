#lang racket

(require "CNN.rkt"
         "device.rkt"
         "hardware_detection.rkt")

;; Benchmark CNN on different devices
(define (benchmark-cnn)
  (printf "====================================\n")
  (printf "  RACOGRAD CNN BENCHMARKS\n")
  (printf "====================================\n")
  
  (print-hardware-info)
  
  ;; Set parameters for quick benchmarking
  (define epochs 2)
  (define batch-size 32)
  
  ;; Run on CPU
  (printf "\n\nRunning CNN benchmark on CPU...\n")
  (define cpu-result (train-cnn 'cpu epochs batch-size))
  
  (define results (list (cons 'cpu cpu-result)))
  
  ;; Run on MLX if available
  (when (device-available? 'mlx)
    (printf "\n\nRunning CNN benchmark on MLX (Apple Silicon)...\n")
    (define mlx-result (train-cnn 'mlx epochs batch-size))
    (set! results (cons (cons 'mlx mlx-result) results)))
  
  ;; Run on CUDA if available
  (when (device-available? 'cuda)
    (printf "\n\nRunning CNN benchmark on CUDA (NVIDIA GPU)...\n")
    (define cuda-result (train-cnn 'cuda epochs batch-size))
    (set! results (cons (cons 'cuda cuda-result) results)))
  
  ;; Run on OpenCL if available
  (when (device-available? 'opencl)
    (printf "\n\nRunning CNN benchmark on OpenCL...\n")
    (define opencl-result (train-cnn 'opencl epochs batch-size))
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

;; Run the benchmark from the command line
(module+ main
  (benchmark-cnn))