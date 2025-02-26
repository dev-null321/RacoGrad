#lang racket

(require ffi/unsafe
         ffi/unsafe/define
         ffi/unsafe/cvector)

; Define the C function
(define-ffi-definer define-c (ffi-lib "./matrix_multiplication.so"))
(define-c matrix_multiply (_fun _pointer _pointer _pointer _int _int _int -> _void))

(provide tensor create-tensor tensor-add tensor-subtract tensor-multiply dense-forward mean-squared-error dense-backward print-tensor random-tensor reshape-tensor 
  transpose scalar-multiply dot-product relu relu-derivative tensor-shape tensor-data initialize-fnn)

(struct tensor (shape data) #:transparent)

(define (create-tensor shape data)
  (let ((vec-data (if (vector? data) data (list->vector data))))
    (cond
      [(= (apply * shape) (vector-length vec-data))
       (tensor shape vec-data)]
      [else
       (begin
         (println "Error: Data does not match, please check the size")
         #f)])))

(define (print-tensor t)
  (let ([shape (tensor-shape t)]
        [data (tensor-data t)])
    (for ([i (in-range (car shape))])
      (display "[")
      (for ([j (in-range (cadr shape))])
        (display (vector-ref data (+ (* i (cadr shape)) j)))
        (display " "))
      (display "]")
      (newline))))

; Create random tensor
(define (random-tensor shape range)
  (let* ((size (apply * shape))
         (max-value (inexact->exact (floor (* range 10000)))))
    (tensor shape
            (for/vector ([i size])
              (/ (random max-value) 10000.0)))))

; Reshaping tensors
(define (reshape-tensor t new-shape)
  (let ([original-size (apply * (tensor-shape t))]
        [new-size (apply * new-shape)])
    (if (= original-size new-size)
        (tensor new-shape (tensor-data t))
        (error "New shape must have the same number of elements as the original shape"))))

; Element-wise operations
(define (tensor-add t1 t2)
  (let ([shape1 (tensor-shape t1)]
        [shape2 (tensor-shape t2)])
    (displayln (string-append "Adding tensors with shapes: " (format "~a" shape1) " and " (format "~a" shape2)))
    (cond
      [(equal? shape1 shape2)
       (tensor shape1 (for/vector ([i (vector-length (tensor-data t1))])
                        (+ (vector-ref (tensor-data t1) i)
                           (vector-ref (tensor-data t2) i))))]
      [else
       (error "Tensors must have the same shape for addition")])))

(define (tensor-subtract t1 t2)
  (let ([shape1 (tensor-shape t1)]
        [shape2 (tensor-shape t2)])
    (displayln (string-append "Subtracting tensors with shapes: " (format "~a" shape1) " and " (format "~a" shape2)))
    (cond
      [(equal? shape1 shape2)
       (tensor shape1 (for/vector ([i (vector-length (tensor-data t1))])
                        (- (vector-ref (tensor-data t1) i)
                           (vector-ref (tensor-data t2) i))))]
      [else
       (error "Tensors must have the same shape for subtraction")])))

(define (tensor-multiply t1 t2)
  (let ([shape1 (tensor-shape t1)]
        [shape2 (tensor-shape t2)])
    (displayln (string-append "Multiplying tensors with shapes: " (format "~a" shape1) " and " (format "~a" shape2)))
    (cond
      [(and (= (length shape1) 2) (= (length shape2) 2) (= (cadr shape1) (car shape2)))
       (let* ([rows-a (car shape1)]
              [cols-a (cadr shape1)]
              [cols-b (cadr shape2)]
              [a (tensor-data t1)]
              [b (tensor-data t2)]
              [c (make-vector (* rows-a cols-b) 0.0)]
              [ptr-a (_cvector _double (vector->list a))]
              [ptr-b (_cvector _double (vector->list b))]
              [ptr-c (_cvector _double (vector->list c))])
         (matrix_multiply ptr-a ptr-b ptr-c rows-a cols-a cols-b)
         (tensor (list rows-a cols-b) (list->vector (cvector->list ptr-c))))]
      [else
       (error "Tensors must have compatible shapes for multiplication")])))
; Activation function (ReLU)
(define (relu x)
  (tensor (tensor-shape x) (for/vector ([v (tensor-data x)]) (max 0 v))))

; Derivative of ReLU
(define (relu-derivative x)
  (tensor (tensor-shape x) (for/vector ([v (tensor-data x)]) (if (> v 0) 1 0))))

; Dense layer forward propagation
(define (dense-forward input weights biases)
  (let* ([z (tensor-add (tensor-multiply input weights) biases)]
         [activation-output (relu z)])
    activation-output))

; Loss function
(define (mean-squared-error y-true y-pred)
  (let* ([diff (tensor-subtract y-true y-pred)]
         [squared-diff (tensor-multiply diff diff)]
         [sum (apply + (tensor-data squared-diff))])
    (/ sum (length (tensor-data y-true)))))

; Backpropagation for dense layer
(define (dense-backward input weights biases output grad-output learning-rate)
  (let* ([grad-activation (relu-derivative output)]
         [grad-z (tensor-multiply grad-output grad-activation)]
         [grad-biases (tensor (tensor-shape biases)
                              (for/vector ([i (tensor-data grad-z)])
                                (apply + (tensor-data i))))]
         [grad-weights (tensor-multiply (transpose input) grad-z)]
         [grad-input (tensor-multiply grad-z (transpose weights))])
    (let* ([new-weights (tensor-subtract weights (scalar-multiply grad-weights learning-rate))]
           [new-biases (tensor-subtract biases (scalar-multiply grad-biases learning-rate))])
      (values new-weights new-biases grad-input))))

(define (transpose t)
  (let* ([shape (tensor-shape t)]
         [rows (car shape)]
         [cols (cadr shape)]
         [data (tensor-data t)]
         [new-data (make-vector (apply * (reverse shape)) 0)])
    (for* ([i rows]
           [j cols])
      (vector-set! new-data (+ (* j rows) i) (vector-ref data (+ (* i cols) j))))
    (tensor (reverse shape) new-data)))

(define (scalar-multiply t scalar)
  (tensor (tensor-shape t) (for/vector ([v (tensor-data t)]) (* v scalar))))

; Example of initializing a simple FNN
(define (initialize-fnn batch-size input-dim output-dim)
  (let* ([input-data (make-list (* batch-size input-dim) 0)]
         [input-tensor (create-tensor (list batch-size input-dim) input-data)]
         [weight-shape (list input-dim output-dim)]
         [bias-shape (list output-dim)]
         [weights (random-tensor weight-shape 1.0)]
         [biases (random-tensor bias-shape 1.0)])
    (values input-tensor weights biases)))

; Dot product
(define (dot-product t1 t2)
  (let ([data1 (tensor-data t1)]
        [data2 (tensor-data t2)])
    (if (not (= (length data1) (length data2)))
        (error "Tensors must have the same length for dot product")
        (apply + (map * data1 data2)))))