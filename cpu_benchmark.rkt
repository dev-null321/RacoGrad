#lang racket

(require "tensor.rkt"
         "tensor_optimized.rkt"
         "hardware_detection.rkt")

(define (benchmark-matrix-multiplication sizes [iterations 5])
  (printf "===== Matrix Multiplication Benchmark =====~n")
  
  (for ([size sizes])
    (printf "Matrix size: ~a x ~a~n" size size)
    
    ;; Create random matrices
    (define A (t:random (list size size) 1.0))
    (define B (t:random (list size size) 1.0))
    
    ;; Create optimized matrices
    (define A-opt (t-opt:random (list size size) 1.0))
    (define B-opt (t-opt:random (list size size) 1.0))
    
    ;; Time standard Racket multiplication
    (define racket-times
      (for/list ([i (in-range iterations)])
        (let ([start (current-inexact-milliseconds)])
          (t:mul A B)
          (- (current-inexact-milliseconds) start))))
    
    (define avg-racket-time (/ (apply + racket-times) iterations))
    (printf "  Standard Racket time: ~a ms~n" (real->decimal-string avg-racket-time 2))
    
    ;; Time optimized C multiplication
    (define c-times
      (for/list ([i (in-range iterations)])
        (let ([start (current-inexact-milliseconds)])
          (t-opt:mul A-opt B-opt)
          (- (current-inexact-milliseconds) start))))
    
    (define avg-c-time (/ (apply + c-times) iterations))
    (printf "  Optimized C time: ~a ms~n" (real->decimal-string avg-c-time 2))
    
    ;; Calculate speedup
    (define speedup (/ avg-racket-time avg-c-time))
    (printf "  Speedup: ~a times faster~n" (real->decimal-string speedup 2))
    
    (printf "~n")))

(define (benchmark-elementwise-operations size [iterations 5])
  (printf "===== Element-wise Operations Benchmark =====~n")
  (printf "Vector size: ~a~n" size)
  
  ;; Create random vectors
  (define A (t:random (list size) 1.0))
  (define B (t:random (list size) 1.0))
  
  ;; Create optimized vectors
  (define A-opt (t-opt:random (list size) 1.0))
  (define B-opt (t-opt:random (list size) 1.0))
  
  ;; Time standard Racket addition
  (define racket-add-times
    (for/list ([i (in-range iterations)])
      (let ([start (current-inexact-milliseconds)])
        (t:add A B)
        (- (current-inexact-milliseconds) start))))
  
  (define avg-racket-add-time (/ (apply + racket-add-times) iterations))
  (printf "  Standard Racket add time: ~a ms~n" (real->decimal-string avg-racket-add-time 2))
  
  ;; Time optimized C addition
  (define c-add-times
    (for/list ([i (in-range iterations)])
      (let ([start (current-inexact-milliseconds)])
        (t-opt:add A-opt B-opt)
        (- (current-inexact-milliseconds) start))))
  
  (define avg-c-add-time (/ (apply + c-add-times) iterations))
  (printf "  Optimized C add time: ~a ms~n" (real->decimal-string avg-c-add-time 2))
  
  ;; Calculate speedup
  (define add-speedup (/ avg-racket-add-time avg-c-add-time))
  (printf "  Add speedup: ~a times faster~n" (real->decimal-string add-speedup 2))
  
  (printf "~n"))

(define (run-all-benchmarks)
  (printf "~n======================================~n")
  (printf "  RACOGRAD CPU PERFORMANCE BENCHMARKS~n")
  (printf "======================================~n")
  
  (print-hardware-info)
  
  (benchmark-matrix-multiplication '(100 500 1000))
  (benchmark-elementwise-operations 1000000)
  
  (printf "======================================~n")
  (printf "Benchmarks complete!~n")
  (printf "======================================~n"))

;; Run benchmarks when executed directly
(module+ main
  (run-all-benchmarks))