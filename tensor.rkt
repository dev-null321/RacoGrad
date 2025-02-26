#lang racket

(provide (struct-out tensor) ; Export the tensor struct and its accessors
         ;; core operations
         t:create ; creates a tensor
         t:random ; random tensor
         t:reshape
         t:print

         ;; math operations
         t:add
         t:sub
         t:mul ; Matrix multiplication and elementwise
         t:dot ; Dot Product
         t:scale ; Scalar Multiplication
         t:transpose

         ;; Accessors
         t:shape ; Get shape
         t:data  ; Get Data
         t:ref
         ; t:at    ; Uncomment if implemented
         ; t:slice ; Uncomment if implemented
         )

(struct tensor (shape data) #:transparent)

;; Wrapper accessors
(define (t:shape t)
  (tensor-shape t))

(define (t:data t)
  (tensor-data t))

;; Create a tensor
(define (t:create shape data)
  (let ((vec-data (if (vector? data) data (list->vector data))))
    (cond
      [(= (apply * shape) (vector-length vec-data))
       (tensor shape vec-data)]
      [else
       (begin
         (println "Error: Data does not match, please check the size")
         #f)])))

;; Print tensor
(define (t:print t)
  (let ([shape (tensor-shape t)]
        [data (tensor-data t)])
    (cond
      [(= (length shape) 1)
       (display "[")
       (for ([i (in-range (car shape))])
         (display (vector-ref data i))
         (display " "))
       (display "]")
       (newline)]
      [(= (length shape) 2)
       (for ([i (in-range (car shape))])
         (display "[")
         (for ([j (in-range (cadr shape))])
           (display (vector-ref data (+ (* i (cadr shape)) j)))
           (display " "))
         (display "]")
         (newline))]
      [else
       (error "t:print: unsupported tensor shape")])))

;; Random tensor
(define (t:random shape range)
  (let* ((size (apply * shape))
         (max-value (inexact->exact (floor (* range 10000)))))
    (tensor shape
            (for/vector ([i size])
              (/ (random max-value) 10000.0)))))

;; Reshape tensor
(define (t:reshape t new-shape)
  (let ([original-size (apply * (tensor-shape t))]
        [new-size (apply * new-shape)])
    (if (= original-size new-size)
        (tensor new-shape (tensor-data t))
        (error "t:reshape: New shape must have the same number of elements as the original shape"))))

;; Add tensors
(define (t:add t1 t2)
  (let ([shape1 (tensor-shape t1)]
        [shape2 (tensor-shape t2)])
    (cond
      [(equal? shape1 shape2)
       (tensor shape1
               (for/vector ([i (vector-length (tensor-data t1))])
                 (+ (vector-ref (tensor-data t1) i)
                    (vector-ref (tensor-data t2) i))))]
      [(= (length shape1) 1)
       (let ([scalar-val (vector-ref (tensor-data t1) 0)])
         (tensor shape2
                 (for/vector ([i (vector-length (tensor-data t2))])
                   (+ scalar-val (vector-ref (tensor-data t2) i)))))]

      [(= (length shape2) 1)
       (let ([scalar-val (vector-ref (tensor-data t2) 0)])
         (tensor shape1
                 (for/vector ([i (vector-length (tensor-data t1))])
                   (+ (vector-ref (tensor-data t1) i) scalar-val))))]
      [else
       (error "t:add: Tensors must have the same shape or be broadcastable for addition")])))

;; Subtract tensors
(define (t:sub t1 t2)
  (let ([shape1 (tensor-shape t1)]
        [shape2 (tensor-shape t2)])
    (cond
      [(equal? shape1 shape2)
       (tensor shape1
               (for/vector ([i (vector-length (tensor-data t1))])
                 (- (vector-ref (tensor-data t1) i)
                    (vector-ref (tensor-data t2) i))))]
      [else
       (error "t:sub: Tensors must have the same shape for subtraction")])))

;; Multiply tensors (Matrix multiply or elementwise)
(define (t:mul t1 t2)
  (let ([shape1 (tensor-shape t1)]
        [shape2 (tensor-shape t2)])
    (cond
      ;; Matrix multiplication: (A: MxN) * (B: NxP) -> (C: MxP)
      [(and (= (length shape1) 2) (= (length shape2) 2) (= (cadr shape1) (car shape2)))
       (let* ([rows-a (car shape1)]
              [cols-a (cadr shape1)]
              [cols-b (cadr shape2)]
              [result (make-vector (* rows-a cols-b) 0.0)])
         (for ([i (in-range rows-a)])
           (for ([j (in-range cols-b)])
             (for ([k (in-range cols-a)])
               (vector-set! result (+ (* i cols-b) j)
                            (+ (vector-ref result (+ (* i cols-b) j))
                               (* (t:ref t1 i k) (t:ref t2 k j)))))))
         (tensor (list rows-a cols-b) result))]

      ;; Vector (1D) * Matrix (2D) multiplication when shapes align
      [(and (= (length shape1) 1) (= (length shape2) 2) (= (car shape1) (car shape2)))
       (let* ([rows-b (car shape2)]
              [cols-b (cadr shape2)]
              [result (make-vector cols-b 0.0)])
         (for ([j (in-range cols-b)])
           (for ([i (in-range rows-b)])
             (vector-set! result j
                          (+ (vector-ref result j)
                             (* (vector-ref (tensor-data t1) i)
                                (t:ref t2 i j))))))
         (tensor (list cols-b) result))]

      ;; Elementwise multiplication if shapes match and are both 2D (or same dimension)
      [(equal? shape1 shape2)
       (tensor shape1 (vector-map * (tensor-data t1) (tensor-data t2)))]

      [else
       (error "t:mul: Tensors must have compatible shapes for multiplication")]))) 

;; Reference element at (i, j)
(define (t:ref t i j)
  (vector-ref (tensor-data t) (+ (* i (cadr (tensor-shape t))) j)))

;; Transpose a matrix (2D only)
(define (t:transpose t)
  (let* ([shape (tensor-shape t)]
         [rows (car shape)]
         [cols (cadr shape)]
         [data (tensor-data t)]
         [new-data (make-vector (apply * (reverse shape)) 0)])
    (for* ([i rows]
           [j cols])
      (vector-set! new-data (+ (* j rows) i) (vector-ref data (+ (* i cols) j))))
    (tensor (reverse shape) new-data)))

;; Scalar multiply a tensor
(define (t:scale t scalar)
  (let ([data (tensor-data t)])
    (tensor (tensor-shape t)
            (for/vector ([v data])
              (* v scalar)))))

;; Dot product (1D only)
(define (t:dot t1 t2)
  (let ([data1 (tensor-data t1)]
        [data2 (tensor-data t2)])
    (if (not (= (vector-length data1) (vector-length data2)))
        (error "t:dot: Tensors must have the same length for dot product")
        (apply + (map * data1 data2)))))
