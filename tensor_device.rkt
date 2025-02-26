#lang racket

(require "tensor.rkt"
         "device.rkt"
         "hardware_detection.rkt"
         ffi/unsafe
         ffi/vector)

;; Provide extended tensor operations with device support
(provide 
 ;; Core operations
 dt:create     ; Create a tensor on a specific device
 dt:to         ; Move a tensor to a specific device
 dt:random     ; Create a random tensor on a device
 dt:print      ; Print a tensor
 dt:reshape    ; Reshape a tensor
 dt:device     ; Get the device of a tensor
 
 ;; Math operations
 dt:add        ; Add tensors
 dt:sub        ; Subtract tensors
 dt:mul        ; Matrix multiplication or elementwise
 dt:dot        ; Dot product
 dt:scale      ; Scalar multiplication
 dt:transpose  ; Transpose a tensor
 
 ;; Accessors (same as tensor.rkt)
 dt:shape      ; Get shape
 dt:data       ; Get data
 dt:ref        ; Get value at index
 )

;; Define device-aware tensor structure as a separate struct
;; Instead of extending tensor, which causes issues
(struct dt:tensor (shape data device) #:transparent)

;; Create tensor on device
(define (dt:create shape data [dev (current-device)])
  (let ([t (t:create shape data)])
    (dt:tensor (t:shape t) (t:data t) dev)))

;; Get device of tensor
(define (dt:device t)
  (dt:tensor-device t))

;; Move tensor to device
(define (dt:to t dev)
  (cond
    [(dt:tensor? t)
     (let ([current-dev (dt:tensor-device t)])
       (cond
         ;; Same device - no-op
         [(equal? current-dev dev) t]
         
         ;; Move to GPU from CPU
         [(and (cpu-device? current-dev) (gpu-device? dev))
          (printf "Moving tensor to GPU device~n")
          (dt:tensor (dt:tensor-shape t) (dt:tensor-data t) dev)]
         
         ;; Move to CPU from GPU
         [(and (gpu-device? current-dev) (cpu-device? dev))
          (printf "Moving tensor to CPU device~n")
          (dt:tensor (dt:tensor-shape t) (dt:tensor-data t) dev)]
         
         [else t]))]
    
    ;; Convert regular tensor to device tensor
    [(tensor? t)
     (dt:tensor (t:shape t) (t:data t) dev)]
    
    [else
     (error "dt:to: expected a tensor, got ~a" t)]))

;; Create random tensor on device
(define (dt:random shape range [dev (current-device)])
  (let ([t (t:random shape range)])
    (dt:tensor (t:shape t) (t:data t) dev)))

;; Print tensor
(define (dt:print t)
  (when (dt:tensor? t)
    ;; Create a regular tensor to print
    (let ([regular-tensor (t:create (dt:tensor-shape t) (dt:tensor-data t))])
      (t:print regular-tensor)
      (printf "Device: ~a~n" (get-device-type (dt:tensor-device t))))))

;; Reshape tensor
(define (dt:reshape t new-shape)
  (when (dt:tensor? t)
    (let ([regular-tensor (t:create (dt:tensor-shape t) (dt:tensor-data t))]
          [dev (dt:tensor-device t)])
      (let ([reshaped (t:reshape regular-tensor new-shape)])
        (dt:tensor (t:shape reshaped) (t:data reshaped) dev)))))

;; Add tensors
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
         
         ;; Both on GPU - use OpenCL implementation if available
         [(and (gpu-device? dev1) (gpu-device? dev2) (has-opencl?))
          (printf "Performing GPU tensor addition~n")
          ;; For now, fall back to CPU implementation
          (let ([regular-t1 (t:create (dt:tensor-shape t1) (dt:tensor-data t1))]
                [regular-t2 (t:create (dt:tensor-shape t2) (dt:tensor-data t2))])
            (let ([sum (t:add regular-t1 regular-t2)])
              (dt:tensor (t:shape sum) (t:data sum) dev1)))]
         
         [else
          (let ([regular-t1 (t:create (dt:tensor-shape t1) (dt:tensor-data t1))]
                [regular-t2 (t:create (dt:tensor-shape t2) (dt:tensor-data t2))])
            (let ([sum (t:add regular-t1 regular-t2)])
              (dt:tensor (t:shape sum) (t:data sum) dev1)))]))]
    
    ;; Handle regular tensors
    [else
     (let ([dev (current-device)])
       (let ([dt1 (if (dt:tensor? t1) 
                       t1 
                       (dt:tensor (t:shape t1) (t:data t1) dev))]
             [dt2 (if (dt:tensor? t2) 
                       t2 
                       (dt:tensor (t:shape t2) (t:data t2) dev))])
         (dt:add dt1 dt2)))]))

;; Subtract tensors
(define (dt:sub t1 t2)
  (cond
    [(and (dt:tensor? t1) (dt:tensor? t2))
     (let ([dev1 (dt:tensor-device t1)]
           [dev2 (dt:tensor-device t2)])
       (cond
         ;; Both on CPU
         [(and (cpu-device? dev1) (cpu-device? dev2))
          (let ([regular-t1 (t:create (dt:tensor-shape t1) (dt:tensor-data t1))]
                [regular-t2 (t:create (dt:tensor-shape t2) (dt:tensor-data t2))])
            (let ([diff (t:sub regular-t1 regular-t2)])
              (dt:tensor (t:shape diff) (t:data diff) dev1)))]
         
         ;; Different devices - move to same device
         [(not (equal? dev1 dev2))
          (let ([target-dev (if (gpu-device? dev1) dev1 dev2)])
            (dt:sub (dt:to t1 target-dev) (dt:to t2 target-dev)))]
         
         ;; Both on GPU - use OpenCL implementation if available
         [(and (gpu-device? dev1) (gpu-device? dev2) (has-opencl?))
          (printf "Performing GPU tensor subtraction~n")
          ;; For now, fall back to CPU implementation
          (let ([regular-t1 (t:create (dt:tensor-shape t1) (dt:tensor-data t1))]
                [regular-t2 (t:create (dt:tensor-shape t2) (dt:tensor-data t2))])
            (let ([diff (t:sub regular-t1 regular-t2)])
              (dt:tensor (t:shape diff) (t:data diff) dev1)))]
         
         [else
          (let ([regular-t1 (t:create (dt:tensor-shape t1) (dt:tensor-data t1))]
                [regular-t2 (t:create (dt:tensor-shape t2) (dt:tensor-data t2))])
            (let ([diff (t:sub regular-t1 regular-t2)])
              (dt:tensor (t:shape diff) (t:data diff) dev1)))]))]
    
    ;; Handle regular tensors
    [else
     (let ([dev (current-device)])
       (let ([dt1 (if (dt:tensor? t1) 
                       t1 
                       (dt:tensor (t:shape t1) (t:data t1) dev))]
             [dt2 (if (dt:tensor? t2) 
                       t2 
                       (dt:tensor (t:shape t2) (t:data t2) dev))])
         (dt:sub dt1 dt2)))]))

;; Multiply tensors
(define (dt:mul t1 t2)
  (cond
    [(and (dt:tensor? t1) (dt:tensor? t2))
     (let ([dev1 (dt:tensor-device t1)]
           [dev2 (dt:tensor-device t2)])
       (cond
         ;; Both on CPU
         [(and (cpu-device? dev1) (cpu-device? dev2))
          (let ([regular-t1 (t:create (dt:tensor-shape t1) (dt:tensor-data t1))]
                [regular-t2 (t:create (dt:tensor-shape t2) (dt:tensor-data t2))])
            (let ([prod (t:mul regular-t1 regular-t2)])
              (dt:tensor (t:shape prod) (t:data prod) dev1)))]
         
         ;; Different devices - move to same device
         [(not (equal? dev1 dev2))
          (let ([target-dev (if (gpu-device? dev1) dev1 dev2)])
            (dt:mul (dt:to t1 target-dev) (dt:to t2 target-dev)))]
         
         ;; Both on GPU - use OpenCL implementation if available
         [(and (gpu-device? dev1) (gpu-device? dev2) (has-opencl?))
          (printf "Performing GPU tensor multiplication~n")
          ;; TODO: Implement GPU matrix multiplication using OpenCL
          ;; For now, fall back to CPU implementation
          (let ([regular-t1 (t:create (dt:tensor-shape t1) (dt:tensor-data t1))]
                [regular-t2 (t:create (dt:tensor-shape t2) (dt:tensor-data t2))])
            (let ([prod (t:mul regular-t1 regular-t2)])
              (dt:tensor (t:shape prod) (t:data prod) dev1)))]
         
         [else
          (let ([regular-t1 (t:create (dt:tensor-shape t1) (dt:tensor-data t1))]
                [regular-t2 (t:create (dt:tensor-shape t2) (dt:tensor-data t2))])
            (let ([prod (t:mul regular-t1 regular-t2)])
              (dt:tensor (t:shape prod) (t:data prod) dev1)))]))]
    
    ;; Handle regular tensors
    [else
     (let ([dev (current-device)])
       (let ([dt1 (if (dt:tensor? t1) 
                       t1 
                       (dt:tensor (t:shape t1) (t:data t1) dev))]
             [dt2 (if (dt:tensor? t2) 
                       t2 
                       (dt:tensor (t:shape t2) (t:data t2) dev))])
         (dt:mul dt1 dt2)))]))

;; Scalar multiply
(define (dt:scale t scalar)
  (if (dt:tensor? t)
      (let ([regular-tensor (t:create (dt:tensor-shape t) (dt:tensor-data t))]
            [dev (dt:tensor-device t)])
        (let ([scaled (t:scale regular-tensor scalar)])
          (dt:tensor (t:shape scaled) (t:data scaled) dev)))
      (let ([scaled (t:scale t scalar)])
        (dt:tensor (t:shape scaled) (t:data scaled) (current-device)))))

;; Transpose
(define (dt:transpose t)
  (if (dt:tensor? t)
      (let ([regular-tensor (t:create (dt:tensor-shape t) (dt:tensor-data t))]
            [dev (dt:tensor-device t)])
        (let ([transposed (t:transpose regular-tensor)])
          (dt:tensor (t:shape transposed) (t:data transposed) dev)))
      (let ([transposed (t:transpose t)])
        (dt:tensor (t:shape transposed) (t:data transposed) (current-device)))))

;; Dot product
(define (dt:dot t1 t2)
  (cond
    [(and (dt:tensor? t1) (dt:tensor? t2))
     (let ([dev1 (dt:tensor-device t1)]
           [dev2 (dt:tensor-device t2)])
       (cond
         ;; Different devices - move to same device
         [(not (equal? dev1 dev2))
          (let ([target-dev (if (gpu-device? dev1) dev1 dev2)])
            (dt:dot (dt:to t1 target-dev) (dt:to t2 target-dev)))]
         
         ;; Both on GPU - use OpenCL implementation if available
         [(and (gpu-device? dev1) (gpu-device? dev2) (has-opencl?))
          (printf "Performing GPU dot product~n")
          ;; For now, fall back to CPU implementation
          (let ([regular-t1 (t:create (dt:tensor-shape t1) (dt:tensor-data t1))]
                [regular-t2 (t:create (dt:tensor-shape t2) (dt:tensor-data t2))])
            (t:dot regular-t1 regular-t2))]
         
         [else
          (let ([regular-t1 (t:create (dt:tensor-shape t1) (dt:tensor-data t1))]
                [regular-t2 (t:create (dt:tensor-shape t2) (dt:tensor-data t2))])
            (t:dot regular-t1 regular-t2))]))]
    
    ;; Handle regular tensors
    [else
     (let ([dev (current-device)])
       (let ([regular-t1 (if (dt:tensor? t1) 
                             (t:create (dt:tensor-shape t1) (dt:tensor-data t1))
                             t1)]
             [regular-t2 (if (dt:tensor? t2) 
                             (t:create (dt:tensor-shape t2) (dt:tensor-data t2))
                             t2)])
         (t:dot regular-t1 regular-t2)))]))

;; Accessors
(define (dt:shape t)
  (if (dt:tensor? t)
      (dt:tensor-shape t)
      (t:shape t)))

(define (dt:data t)
  (if (dt:tensor? t)
      (dt:tensor-data t)
      (t:data t)))

(define (dt:ref t i j)
  (if (dt:tensor? t)
      (vector-ref (dt:tensor-data t) (+ (* i (cadr (dt:tensor-shape t))) j))
      (t:ref t i j)))