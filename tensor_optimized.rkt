#lang racket

(require "ffi_ops.rkt")
(require ffi/vector)

(provide (struct-out tensor-opt) 
         t-opt:create
         t-opt:random
         t-opt:reshape
         t-opt:print
         t-opt:add
         t-opt:add!
         t-opt:sub
         t-opt:sub!
         t-opt:mul
         t-opt:transpose
         t-opt:shape
         t-opt:data
         t-opt:ref
         t-opt:scale
         t-opt:scale!)

;; Optimized tensor structure
;; Uses a f64vector for better FFI compatibility and memory layout
(struct tensor-opt (shape data) #:transparent)

;; Accessors
(define (t-opt:shape t)
  (tensor-opt-shape t))

(define (t-opt:data t)
  (tensor-opt-data t))

;; Create tensor from data
(define (t-opt:create shape data)
  (let ((vec-data (cond
                    [(f64vector? data) data]
                    [(vector? data) (list->f64vector (vector->list data))]
                    [else (list->f64vector data)])))
    (cond
      [(= (apply * shape) (f64vector-length vec-data))
       (tensor-opt shape vec-data)]
      [else
       (error "t-opt:create: Data does not match shape")])))

;; Random tensor
(define (t-opt:random shape range)
  (let* ((size (apply * shape))
         (max-value (inexact->exact (floor (* range 10000)))))
    (tensor-opt shape
                (list->f64vector
                  (for/list ([i size])
                    (/ (random max-value) 10000.0))))))

;; Reshape tensor
(define (t-opt:reshape t new-shape)
  (let ([original-size (apply * (tensor-opt-shape t))]
        [new-size (apply * new-shape)])
    (if (= original-size new-size)
        (tensor-opt new-shape (tensor-opt-data t))
        (error "t-opt:reshape: New shape must have the same number of elements"))))

;; Print tensor
(define (t-opt:print t)
  (let ([shape (tensor-opt-shape t)]
        [data (tensor-opt-data t)])
    (cond
      [(= (length shape) 1)
       (display "[")
       (for ([i (in-range (car shape))])
         (display (f64vector-ref data i))
         (display " "))
       (display "]")
       (newline)]
      [(= (length shape) 2)
       (for ([i (in-range (car shape))])
         (display "[")
         (for ([j (in-range (cadr shape))])
           (display (f64vector-ref data (+ (* i (cadr shape)) j)))
           (display " "))
         (display "]")
         (newline))]
      [else (error "t-opt:print: Unsupported tensor shape")])))

;; Add tensors - out of place
(define (t-opt:add t1 t2)
  (let ([shape1 (tensor-opt-shape t1)]
        [shape2 (tensor-opt-shape t2)])
    (cond
      [(equal? shape1 shape2)
       (let* ([size (apply * shape1)]
              [result (make-f64vector size 0.0)])
         (c:tensor-add size 
                      (tensor-opt-data t1) 
                      (tensor-opt-data t2) 
                      result)
         (tensor-opt shape1 result))]
      [else
       (error "t-opt:add: Tensors must have the same shape")])))

;; Add tensors - in-place version (t1 += t2)
(define (t-opt:add! t1 t2)
  (let ([shape1 (tensor-opt-shape t1)]
        [shape2 (tensor-opt-shape t2)])
    (cond
      [(equal? shape1 shape2)
       (let ([size (apply * shape1)])
         (c:tensor-add size 
                      (tensor-opt-data t1) 
                      (tensor-opt-data t2) 
                      (tensor-opt-data t1))
         t1)]
      [else
       (error "t-opt:add!: Tensors must have the same shape")])))

;; Subtract tensors - out of place
(define (t-opt:sub t1 t2)
  (let ([shape1 (tensor-opt-shape t1)]
        [shape2 (tensor-opt-shape t2)])
    (cond
      [(equal? shape1 shape2)
       (let* ([size (apply * shape1)]
              [result (make-f64vector size 0.0)])
         (c:tensor-sub size 
                      (tensor-opt-data t1) 
                      (tensor-opt-data t2) 
                      result)
         (tensor-opt shape1 result))]
      [else
       (error "t-opt:sub: Tensors must have the same shape")])))

;; Subtract tensors - in-place version (t1 -= t2)
(define (t-opt:sub! t1 t2)
  (let ([shape1 (tensor-opt-shape t1)]
        [shape2 (tensor-opt-shape t2)])
    (cond
      [(equal? shape1 shape2)
       (let ([size (apply * shape1)])
         (c:tensor-sub size 
                      (tensor-opt-data t1) 
                      (tensor-opt-data t2) 
                      (tensor-opt-data t1))
         t1)]
      [else
       (error "t-opt:sub!: Tensors must have the same shape")])))

;; Scale tensor - out of place
(define (t-opt:scale t scalar)
  (let* ([shape (tensor-opt-shape t)]
         [size (apply * shape)]
         [result (make-f64vector size 0.0)])
    (c:tensor-scale size 
                   (tensor-opt-data t) 
                   scalar 
                   result)
    (tensor-opt shape result)))

;; Scale tensor - in-place version (t *= scalar)
(define (t-opt:scale! t scalar)
  (let* ([shape (tensor-opt-shape t)]
         [size (apply * shape)])
    (c:tensor-scale size 
                   (tensor-opt-data t) 
                   scalar 
                   (tensor-opt-data t))
    t))

;; Matrix multiplication
(define (t-opt:mul t1 t2)
  (let ([shape1 (tensor-opt-shape t1)]
        [shape2 (tensor-opt-shape t2)])
    (cond
      ;; Matrix multiplication: (A: MxN) * (B: NxP) -> (C: MxP)
      [(and (= (length shape1) 2) (= (length shape2) 2) (= (cadr shape1) (car shape2)))
       (let* ([rows-a (car shape1)]
              [cols-a (cadr shape1)]
              [cols-b (cadr shape2)]
              [result (make-f64vector (* rows-a cols-b) 0.0)])
         (c:matrix-multiply rows-a cols-a cols-b 
                           (tensor-opt-data t1) 
                           (tensor-opt-data t2) 
                           result)
         (tensor-opt (list rows-a cols-b) result))]
      
      ;; Elementwise multiplication if shapes match
      [(equal? shape1 shape2)
       (let* ([size (apply * shape1)]
              [result (make-f64vector size 0.0)])
         (c:tensor-mul-elementwise size 
                                  (tensor-opt-data t1) 
                                  (tensor-opt-data t2) 
                                  result)
         (tensor-opt shape1 result))]
      
      [else
       (error "t-opt:mul: Tensors must have compatible shapes")])))

;; Reference element at (i, j)
(define (t-opt:ref t i j)
  (f64vector-ref (tensor-opt-data t) (+ (* i (cadr (tensor-opt-shape t))) j)))

;; Transpose a matrix (2D only)
(define (t-opt:transpose t)
  (let* ([shape (tensor-opt-shape t)]
         [rows (car shape)]
         [cols (cadr shape)]
         [data (tensor-opt-data t)]
         [new-data (make-f64vector (apply * (reverse shape)) 0.0)])
    (for* ([i rows]
           [j cols])
      (f64vector-set! new-data (+ (* j rows) i) (f64vector-ref data (+ (* i cols) j))))
    (tensor-opt (reverse shape) new-data)))