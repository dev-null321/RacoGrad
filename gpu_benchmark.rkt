#lang racket

(require "tensor_device.rkt"
         "device.rkt"
         "hardware_detection.rkt")

(define (benchmark-matrix-multiplication sizes [iterations 5])
  (printf "===== Matrix Multiplication Benchmark =====~n")
  
  (for ([size sizes])
    (printf "Matrix size: ~a x ~a~n" size size)
    
    ;; Create random matrices on CPU
    (define A-cpu (dt:random (list size size) 1.0 (cpu)))
    (define B-cpu (dt:random (list size size) 1.0 (cpu)))
    
    ;; Time CPU multiplication
    (define cpu-times
      (for/list ([i (in-range iterations)])
        (let ([start (current-inexact-milliseconds)])
          (dt:mul A-cpu B-cpu)
          (- (current-inexact-milliseconds) start))))
    
    (define avg-cpu-time (/ (apply + cpu-times) iterations))
    (printf "  CPU time: ~a ms~n" (real->decimal-string avg-cpu-time 2))
    
    ;; If GPU is available, benchmark on GPU
    (when (gpu-available?)
      ;; Move matrices to GPU
      (define A-gpu (dt:to A-cpu (gpu)))
      (define B-gpu (dt:to B-cpu (gpu)))
      
      ;; Time GPU multiplication
      (define gpu-times
        (for/list ([i (in-range iterations)])
          (let ([start (current-inexact-milliseconds)])
            (dt:mul A-gpu B-gpu)
            (- (current-inexact-milliseconds) start))))
      
      (define avg-gpu-time (/ (apply + gpu-times) iterations))
      (printf "  GPU time: ~a ms~n" (real->decimal-string avg-gpu-time 2))
      
      ;; Calculate speedup
      (define speedup (/ avg-cpu-time avg-gpu-time))
      (printf "  Speedup: ~a times faster~n" (real->decimal-string speedup 2)))
    
    (printf "~n")))

(define (benchmark-elementwise-operations size [iterations 5])
  (printf "===== Element-wise Operations Benchmark =====~n")
  (printf "Vector size: ~a~n" size)
  
  ;; Create random vectors on CPU
  (define A-cpu (dt:random (list size) 1.0 (cpu)))
  (define B-cpu (dt:random (list size) 1.0 (cpu)))
  
  ;; Time CPU addition
  (define cpu-add-times
    (for/list ([i (in-range iterations)])
      (let ([start (current-inexact-milliseconds)])
        (dt:add A-cpu B-cpu)
        (- (current-inexact-milliseconds) start))))
  
  (define avg-cpu-add-time (/ (apply + cpu-add-times) iterations))
  (printf "  CPU add time: ~a ms~n" (real->decimal-string avg-cpu-add-time 2))
  
  ;; If GPU is available, benchmark on GPU
  (when (gpu-available?)
    ;; Move vectors to GPU
    (define A-gpu (dt:to A-cpu (gpu)))
    (define B-gpu (dt:to B-cpu (gpu)))
    
    ;; Time GPU addition
    (define gpu-add-times
      (for/list ([i (in-range iterations)])
        (let ([start (current-inexact-milliseconds)])
          (dt:add A-gpu B-gpu)
          (- (current-inexact-milliseconds) start))))
    
    (define avg-gpu-add-time (/ (apply + gpu-add-times) iterations))
    (printf "  GPU add time: ~a ms~n" (real->decimal-string avg-gpu-add-time 2))
    
    ;; Calculate speedup
    (define add-speedup (/ avg-cpu-add-time avg-gpu-add-time))
    (printf "  Add speedup: ~a times faster~n" (real->decimal-string add-speedup 2)))
  
  (printf "~n"))

(define (run-all-benchmarks)
  (printf "~n======================================~n")
  (printf "  RACOGRAD GPU PERFORMANCE BENCHMARKS~n")
  (printf "======================================~n")
  
  (print-hardware-info)
  
  (benchmark-matrix-multiplication '(100 500 1000))
  (benchmark-elementwise-operations 1000000)
  
  (printf "======================================~n")
  (printf "Benchmarks complete!~n")
  (printf "======================================~n"))

;; Run benchmarks when executed directly
(module+ main
  (printf "Checking GPU availability...~n")
  (if (gpu-available?)
      (run-all-benchmarks)
      (begin
        (printf "GPU acceleration not available on this system.~n")
        (printf "Running CPU-only benchmarks...~n")
        (benchmark-matrix-multiplication '(100 500 1000)))))