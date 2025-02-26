#lang racket

(require "hardware_detection.rkt")

(provide my-device?
         cpu-device?
         gpu-device?
         mlx-device?
         cuda-device?
         opencl-device?
         
         make-device
         cpu
         gpu
         mlx
         cuda
         opencl
         
         current-device
         set-current-device!
         device-synchronize
         
         get-device-type
         device-available?
         gpu-available?)

;; Define device struct
(struct device (type properties) #:transparent #:mutable)

;; Define device types
(define CPU 'cpu)
(define GPU 'gpu)
(define MLX 'mlx)
(define CUDA 'cuda)
(define OPENCL 'opencl)

;; Helper predicates
(define (my-device? obj)
  (device? obj))

(define (cpu-device? dev)
  (and (my-device? dev) (eq? (device-type dev) CPU)))

(define (gpu-device? dev)
  (and (my-device? dev) 
       (or (eq? (device-type dev) GPU)
           (eq? (device-type dev) MLX)
           (eq? (device-type dev) CUDA)
           (eq? (device-type dev) OPENCL))))

(define (mlx-device? dev)
  (and (my-device? dev) (eq? (device-type dev) MLX)))

(define (cuda-device? dev)
  (and (my-device? dev) (eq? (device-type dev) CUDA)))

(define (opencl-device? dev)
  (and (my-device? dev) (eq? (device-type dev) OPENCL)))

;; Default devices
(define cpu-device (device CPU (hash 'cores (get-optimal-num-threads))))

;; Choose the best available GPU device
(define gpu-device 
  (cond
    [(has-mlx-support?)
     (device MLX (hash 'backend 'mlx))]
    [(has-cuda-support?)
     (device CUDA (hash 'backend 'cuda))]
    [(has-opencl?)
     (device OPENCL (hash 'backend 'opencl))]
    [else #f]))

;; Specific GPU type devices
(define mlx-device
  (if (has-mlx-support?)
      (device MLX (hash 'backend 'mlx))
      #f))

(define cuda-device
  (if (has-cuda-support?)
      (device CUDA (hash 'backend 'cuda))
      #f))

(define opencl-device
  (if (has-opencl?)
      (device OPENCL (hash 'backend 'opencl))
      #f))

;; Factory function
(define (make-device type [properties (hash)])
  (device type properties))

;; Device accessors
(define (cpu)
  cpu-device)

(define (gpu)
  (if gpu-device
      gpu-device
      (error "No GPU device available on this system")))

(define (mlx)
  (if mlx-device
      mlx-device
      (error "MLX not available - requires Apple Silicon")))

(define (cuda)
  (if cuda-device
      cuda-device
      (error "CUDA not available - requires NVIDIA GPU")))

(define (opencl)
  (if opencl-device
      opencl-device
      (error "OpenCL not available on this system")))

;; Current device management
(define current-device-param (make-parameter cpu-device))

(define (current-device)
  (current-device-param))

(define (set-current-device! dev)
  (when (not (my-device? dev))
    (error "set-current-device!: expected a device, got ~a" dev))
  (current-device-param dev))

;; Device synchronization (no-op for CPU, will wait for GPU operations to complete)
(define (device-synchronize [dev (current-device)])
  (when (gpu-device? dev)
    (printf "Synchronizing GPU operations~n")))

;; Get device type
(define (get-device-type [dev (current-device)])
  (device-type dev))

;; Check if a device is available
(define (device-available? type)
  (case type
    [(cpu) #t]
    [(gpu) (or (has-mlx-support?) (has-cuda-support?) (has-opencl?))]
    [(mlx) (has-mlx-support?)]
    [(cuda) (has-cuda-support?)]
    [(opencl) (has-opencl?)]
    [else #f]))

;; Helper to check if any GPU is available
(define (gpu-available?)
  (or (has-mlx-support?) (has-cuda-support?) (has-opencl?)))

;; Print hardware info on load
(print-hardware-info)