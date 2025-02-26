#lang racket

(require "mnist.rkt"
         "CNN.rkt"
         "visualization.rkt"
         "tensor.rkt"
         "device.rkt"
         "hardware_detection.rkt")

(provide run-mnist-demo run-cnn-demo run-full-demo)

;; Function to run MNIST demo
(define (run-mnist-demo [epochs 5] [batch-size 64])
  (printf "Running MNIST Logistic Regression Demo~n")
  (printf "=====================================~n")
  
  ;; Here we would customize mnist.rkt parameters and run it
  ;; For demo purposes, we'll just use the default behavior
  (system "racket -t mnist.rkt -m")
  
  (printf "MNIST Demo Complete!~n~n"))

;; Function to run CNN demo
(define (run-cnn-demo [device 'cpu] [epochs 3] [batch-size 32])
  (printf "Running CNN Demo on device: ~a~n" device)
  (printf "==================================~n")
  
  ;; Run CNN training with specified parameters
  (let ([results (train-cnn device epochs batch-size)])
    (printf "CNN Training Results:~n")
    (printf "  Accuracy: ~a%~n" (hash-ref results 'accuracy))
    (printf "  Training Time: ~a seconds~n" (hash-ref results 'time))
    (printf "  Device: ~a~n" (hash-ref results 'device))
    
    results))

;; Function to run hardware detection demo
(define (run-hardware-demo)
  (printf "Hardware Detection Demo~n")
  (printf "======================~n")
  
  (printf "CPU: Available~n")
  
  (printf "GPU: ~a~n" 
          (if (gpu-available?) "Available" "Not available"))
  
  (printf "MLX (Apple Silicon): ~a~n"
          (if (device-available? 'mlx) "Available" "Not available"))
  
  (printf "Current device: ~a~n" (get-device-type (current-device)))
  
  (printf "Hardware Demo Complete!~n~n"))

;; Function to run full demo
(define (run-full-demo)
  (printf "RacoGrad Full Demo~n")
  (printf "=================~n~n")
  
  ;; Run hardware detection
  (run-hardware-demo)
  
  ;; Run MNIST demo
  (run-mnist-demo 3 64)
  
  ;; Run CNN demo on best available device
  (let ([device (cond
                  [(device-available? 'mlx) 'mlx]
                  [(gpu-available?) 'gpu]
                  [else 'cpu])])
    (run-cnn-demo device 2 32))
  
  (printf "Full Demo Complete!~n"))

;; Run the demo when this file is executed directly
(module+ main
  (define mode (if (> (vector-length (current-command-line-arguments)) 0)
                  (string->symbol (vector-ref (current-command-line-arguments) 0))
                  'full))
  
  (case mode
    [(mnist) (run-mnist-demo)]
    [(cnn) (run-cnn-demo)]
    [(hardware) (run-hardware-demo)]
    [else (run-full-demo)]))