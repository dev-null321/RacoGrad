#lang racket

(require "tensor.rkt"
         "tensor_optimized.rkt"
         "ffi_ops.rkt")

(provide run-benchmarks)

;; Run the benchmarks automatically when this file is executed directly
(module+ main
  (run-benchmarks))

;; Utility to time function execution
(define (time-execution func)
  (let ([start (current-inexact-milliseconds)])
    (func)
    (- (current-inexact-milliseconds) start)))

;; Run a benchmark multiple times and return average
(define (run-benchmark name func iterations)
  (printf "Running benchmark: ~a\n" name)
  (let* ([times (for/list ([i iterations])
                  (time-execution func))]
         [avg (/ (apply + times) iterations)])
    (printf "  Average time over ~a iterations: ~a ms\n" iterations avg)
    (cons name avg)))

;; Helper function to create random matrices of given dimensions
(define (create-random-matrices rows-a cols-a cols-b)
  (let ([a (t:random (list rows-a cols-a) 1.0)]
        [b (t:random (list cols-a cols-b) 1.0)])
    (values a b)))

;; Helper to create optimized random matrices
(define (create-random-optimized-matrices rows-a cols-a cols-b)
  (let ([a (t-opt:random (list rows-a cols-a) 1.0)]
        [b (t-opt:random (list cols-a cols-b) 1.0)])
    (values a b)))

;; Benchmark matrix multiplication
(define (benchmark-matrix-mul iterations)
  (printf "\n=== Matrix Multiplication Benchmark ===\n")
  
  (define rows-a 100)
  (define cols-a 100)
  (define cols-b 100)
  
  (let-values ([(a b) (create-random-matrices rows-a cols-a cols-b)]
               [(a-opt b-opt) (create-random-optimized-matrices rows-a cols-a cols-b)])
    
    ;; Standard Racket implementation
    (define racket-result 
      (run-benchmark "Racket Matrix Multiply" 
                     (lambda () (t:mul a b))
                     iterations))
    
    ;; C implementation via optimized tensors
    (define c-result 
      (run-benchmark "C Matrix Multiply" 
                     (lambda () (t-opt:mul a-opt b-opt))
                     iterations))
    
    ;; Print speedup
    (printf "\nSpeedup: ~a times faster\n" 
            (exact->inexact (/ (cdr racket-result) (cdr c-result))))))

;; Benchmark element-wise operations
(define (benchmark-elementwise iterations)
  (printf "\n=== Element-wise Operations Benchmark ===\n")
  
  (define size 1000000)
  
  (let ([a (t:random (list size) 1.0)]
        [b (t:random (list size) 1.0)]
        [a-opt (t-opt:random (list size) 1.0)]
        [b-opt (t-opt:random (list size) 1.0)])
    
    ;; Addition
    (define racket-add 
      (run-benchmark "Racket Add" 
                     (lambda () (t:add a b))
                     iterations))
    
    (define c-add 
      (run-benchmark "C Add" 
                     (lambda () (t-opt:add a-opt b-opt))
                     iterations))
    
    (printf "\nAdd Speedup: ~a times faster\n" 
            (exact->inexact (/ (cdr racket-add) (cdr c-add))))
    
    ;; Multiplication
    (define racket-mul 
      (run-benchmark "Racket Element-wise Mul" 
                     (lambda () (t:mul a b))
                     iterations))
    
    (define c-mul 
      (run-benchmark "C Element-wise Mul" 
                     (lambda () (t-opt:mul a-opt b-opt))
                     iterations))
    
    (printf "\nMultiply Speedup: ~a times faster\n" 
            (exact->inexact (/ (cdr racket-mul) (cdr c-mul))))))

;; Run all benchmarks
(define (run-benchmarks [iterations 10])
  (printf "\n====================================\n")
  (printf "  RACOGRAD PERFORMANCE BENCHMARKS\n")
  (printf "====================================\n")
  
  (benchmark-matrix-mul iterations)
  (benchmark-elementwise iterations)
  
  (printf "\n====================================\n")
  (printf "Benchmarks complete!\n")
  (printf "See OPTIMIZATION_STRATEGY.md for more details on optimizations.\n")
  (printf "====================================\n"))