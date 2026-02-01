#lang racket

(provide (struct-out tensor) ; Export the tensor struct and its accessors
         ;; core operations
         t:create ; creates a tensor
         t:random ; random tensor
         t:zeros  ; zero tensor
         t:ones   ; ones tensor
         t:fill   ; fill with value
         t:eye    ; identity matrix
         t:reshape
         t:print
         t:map    ; elementwise function application
         t:slice  ; slice rows from a 2D tensor
         t:concat ; concatenate tensors along first axis

         ;; math operations
         t:add
         t:sub
         t:mul  ; Matrix multiplication
         t:emul ; Elementwise multiplication (Hadamard product)
         t:dot ; Dot Product
         t:scale ; Scalar Multiplication
         t:transpose
         t:negate  ; elementwise negation
         t:abs     ; elementwise absolute value
         t:exp     ; elementwise exp
         t:log     ; elementwise natural log
         t:sqrt    ; elementwise sqrt
         t:square  ; elementwise square
         t:sum     ; sum all elements (returns number)
         t:mean    ; mean of all elements (returns number)
         t:max-val ; max element (returns number)
         t:min-val ; min element (returns number)
         t:clip    ; clip values to [lo, hi]

         ;; Accessors
         t:shape ; Get shape
         t:data  ; Get Data
         t:ref
         t:size  ; total number of elements
         t:rank  ; number of dimensions
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

;; Print tensor with shape info
(define (t:print t)
  (let ([shape (tensor-shape t)]
        [data (tensor-data t)])
    (printf "tensor(shape=~a)\n" shape)
    (cond
      [(= (length shape) 1)
       (display "  [")
       (for ([i (in-range (car shape))])
         (when (> i 0) (display ", "))
         (display (real->decimal-string (exact->inexact (vector-ref data i)) 4)))
       (display "]")
       (newline)]
      [(= (length shape) 2)
       (for ([i (in-range (car shape))])
         (display "  [")
         (for ([j (in-range (cadr shape))])
           (when (> j 0) (display ", "))
           (display (real->decimal-string (exact->inexact (vector-ref data (+ (* i (cadr shape)) j))) 4)))
         (display "]")
         (newline))]
      [else
       ;; For higher-dimensional tensors, print flattened with shape info
       (printf "  data: [")
       (let ([len (min 20 (vector-length data))])
         (for ([i (in-range len)])
           (when (> i 0) (display ", "))
           (display (real->decimal-string (exact->inexact (vector-ref data i)) 4)))
         (when (> (vector-length data) 20)
           (printf ", ... (~a more)" (- (vector-length data) 20))))
       (display "]\n")])))

;; Random tensor
(define (t:random shape range)
  (let* ((size (apply * shape))
         (max-value (inexact->exact (floor (* range 10000)))))
    (tensor shape
            (for/vector ([i size])
              (/ (random max-value) 10000.0)))))

;; Zero tensor
(define (t:zeros shape)
  (tensor shape (make-vector (apply * shape) 0.0)))

;; Ones tensor
(define (t:ones shape)
  (tensor shape (make-vector (apply * shape) 1.0)))

;; Fill tensor with a constant value
(define (t:fill shape val)
  (tensor shape (make-vector (apply * shape) val)))

;; Identity matrix (2D square)
(define (t:eye n)
  (let ([data (make-vector (* n n) 0.0)])
    (for ([i (in-range n)])
      (vector-set! data (+ (* i n) i) 1.0))
    (tensor (list n n) data)))

;; Reshape tensor
(define (t:reshape t new-shape)
  (let ([original-size (apply * (tensor-shape t))]
        [new-size (apply * new-shape)])
    (if (= original-size new-size)
        (tensor new-shape (tensor-data t))
        (error "t:reshape: New shape must have the same number of elements as the original shape"))))

;; Add tensors (with row-wise broadcasting for 1D + 2D)
(define (t:add t1 t2)
  (let ([shape1 (tensor-shape t1)]
        [shape2 (tensor-shape t2)])
    (cond
      [(equal? shape1 shape2)
       (tensor shape1
               (for/vector ([i (vector-length (tensor-data t1))])
                 (+ (vector-ref (tensor-data t1) i)
                    (vector-ref (tensor-data t2) i))))]
      ;; Row-wise broadcast: 2D + 1D where 1D length matches columns
      [(and (= (length shape1) 2)
            (= (length shape2) 1)
            (= (car shape2) (cadr shape1)))
       (let ([cols (cadr shape1)]
             [data1 (tensor-data t1)]
             [data2 (tensor-data t2)])
         (tensor shape1
                 (for/vector ([i (vector-length data1)])
                   (+ (vector-ref data1 i)
                      (vector-ref data2 (modulo i cols))))))]
      [(and (= (length shape2) 2)
            (= (length shape1) 1)
            (= (car shape1) (cadr shape2)))
       (let ([cols (cadr shape2)]
             [data1 (tensor-data t1)]
             [data2 (tensor-data t2)])
         (tensor shape2
                 (for/vector ([i (vector-length data2)])
                   (+ (vector-ref data1 (modulo i cols))
                      (vector-ref data2 i)))))]
      ;; Scalar-like 1D broadcast (single element)
      [(and (= (length shape1) 1) (= (car shape1) 1))
       (let ([scalar-val (vector-ref (tensor-data t1) 0)])
         (tensor shape2
                 (for/vector ([i (vector-length (tensor-data t2))])
                   (+ scalar-val (vector-ref (tensor-data t2) i)))))]
      [(and (= (length shape2) 1) (= (car shape2) 1))
       (let ([scalar-val (vector-ref (tensor-data t2) 0)])
         (tensor shape1
                 (for/vector ([i (vector-length (tensor-data t1))])
                   (+ (vector-ref (tensor-data t1) i) scalar-val))))]
      ;; 2D + 2D row broadcast: (M,N) + (1,N) -> (M,N)
      [(and (= (length shape1) 2) (= (length shape2) 2)
            (= (car shape2) 1) (= (cadr shape1) (cadr shape2)))
       (let ([cols (cadr shape1)]
             [data1 (tensor-data t1)]
             [data2 (tensor-data t2)])
         (tensor shape1
                 (for/vector ([i (vector-length data1)])
                   (+ (vector-ref data1 i)
                      (vector-ref data2 (modulo i cols))))))]
      [(and (= (length shape1) 2) (= (length shape2) 2)
            (= (car shape1) 1) (= (cadr shape1) (cadr shape2)))
       (let ([cols (cadr shape2)]
             [data1 (tensor-data t1)]
             [data2 (tensor-data t2)])
         (tensor shape2
                 (for/vector ([i (vector-length data2)])
                   (+ (vector-ref data1 (modulo i cols))
                      (vector-ref data2 i)))))]
      [else
       (error "t:add: Tensors must have the same shape or be broadcastable for addition")])))

;; Subtract tensors (with broadcasting support matching t:add)
(define (t:sub t1 t2)
  (let ([shape1 (tensor-shape t1)]
        [shape2 (tensor-shape t2)])
    (cond
      [(equal? shape1 shape2)
       (tensor shape1
               (for/vector ([i (vector-length (tensor-data t1))])
                 (- (vector-ref (tensor-data t1) i)
                    (vector-ref (tensor-data t2) i))))]
      ;; Broadcast 1D over 2D: subtract each row by the 1D tensor
      [(and (= (length shape2) 1)
            (= (length shape1) 2)
            (= (car shape2) (cadr shape1)))
       (let ([cols (cadr shape1)]
             [data1 (tensor-data t1)]
             [data2 (tensor-data t2)])
         (tensor shape1
                 (for/vector ([i (vector-length data1)])
                   (- (vector-ref data1 i)
                      (vector-ref data2 (modulo i cols))))))]
      [(and (= (length shape1) 1)
            (= (length shape2) 2)
            (= (car shape1) (cadr shape2)))
       (let ([cols (cadr shape2)]
             [data1 (tensor-data t1)]
             [data2 (tensor-data t2)])
         (tensor shape2
                 (for/vector ([i (vector-length data2)])
                   (- (vector-ref data1 (modulo i cols))
                      (vector-ref data2 i)))))]
      ;; Scalar broadcasting (1-element tensor)
      [(and (= (length shape2) 1) (= (car shape2) 1))
       (let ([scalar-val (vector-ref (tensor-data t2) 0)])
         (tensor shape1
                 (for/vector ([v (in-vector (tensor-data t1))])
                   (- v scalar-val))))]
      [(and (= (length shape1) 1) (= (car shape1) 1))
       (let ([scalar-val (vector-ref (tensor-data t1) 0)])
         (tensor shape2
                 (for/vector ([v (in-vector (tensor-data t2))])
                   (- scalar-val v))))]
      [else
       (error "t:sub: Tensors must have compatible shapes for subtraction")])))

;; Multiply tensors (Matrix multiply or elementwise)
(define (t:mul t1 t2)
  (let ([shape1 (tensor-shape t1)]
        [shape2 (tensor-shape t2)])
    (cond
      ;; Matrix multiplication: (A: MxN) * (B: NxP) -> (C: MxP)
      ;; Uses ikj loop order for better cache locality
      [(and (= (length shape1) 2) (= (length shape2) 2) (= (cadr shape1) (car shape2)))
       (let* ([rows-a (car shape1)]
              [cols-a (cadr shape1)]
              [cols-b (cadr shape2)]
              [data1 (tensor-data t1)]
              [data2 (tensor-data t2)]
              [result (make-vector (* rows-a cols-b) 0.0)])
         ;; ikj order: for each row of A, iterate over shared dim k,
         ;; then scatter the product across columns of B.
         ;; This keeps result[i,*] and B[k,*] in cache together.
         (for ([i (in-range rows-a)])
           (for ([k (in-range cols-a)])
             (let ([a-ik (vector-ref data1 (+ (* i cols-a) k))])
               (for ([j (in-range cols-b)])
                 (let ([idx (+ (* i cols-b) j)])
                   (vector-set! result idx
                                (+ (vector-ref result idx)
                                   (* a-ik (vector-ref data2 (+ (* k cols-b) j))))))))))
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

;; Elementwise (Hadamard) multiplication
(define (t:emul t1 t2)
  (let ([shape1 (tensor-shape t1)]
        [shape2 (tensor-shape t2)]
        [data1 (tensor-data t1)]
        [data2 (tensor-data t2)])
    (unless (equal? shape1 shape2)
      (error "t:emul: Tensors must have the same shape for elementwise multiplication"))
    (tensor shape1
            (for/vector ([a (in-vector data1)]
                         [b (in-vector data2)])
              (* a b)))))

;; Dot product (1D only)
(define (t:dot t1 t2)
  (let ([data1 (tensor-data t1)]
        [data2 (tensor-data t2)])
    (if (not (= (vector-length data1) (vector-length data2)))
        (error "t:dot: Tensors must have the same length for dot product")
        (for/sum ([a (in-vector data1)]
                  [b (in-vector data2)])
          (* a b)))))

;; ============================================================
;; Utility operations
;; ============================================================

;; Total number of elements
(define (t:size t)
  (apply * (tensor-shape t)))

;; Number of dimensions
(define (t:rank t)
  (length (tensor-shape t)))

;; Apply a function elementwise to a tensor
(define (t:map f t)
  (tensor (tensor-shape t)
          (for/vector ([v (in-vector (tensor-data t))])
            (f v))))

;; Elementwise negation
(define (t:negate t)
  (t:map - t))

;; Elementwise absolute value
(define (t:abs t)
  (t:map (lambda (x) (if (< x 0) (- x) x)) t))

;; Elementwise exp
(define (t:exp t)
  (t:map exp t))

;; Elementwise natural log (with small epsilon to avoid log(0))
(define (t:log t)
  (t:map (lambda (x) (log (max x 1e-15))) t))

;; Elementwise square root
(define (t:sqrt t)
  (t:map sqrt t))

;; Elementwise square
(define (t:square t)
  (t:map (lambda (x) (* x x)) t))

;; Sum all elements
(define (t:sum t)
  (for/sum ([v (in-vector (tensor-data t))]) v))

;; Mean of all elements
(define (t:mean t)
  (/ (t:sum t) (t:size t)))

;; Max element value
(define (t:max-val t)
  (for/fold ([mx -inf.0])
            ([v (in-vector (tensor-data t))])
    (max mx v)))

;; Min element value
(define (t:min-val t)
  (for/fold ([mn +inf.0])
            ([v (in-vector (tensor-data t))])
    (min mn v)))

;; Clip values to [lo, hi]
(define (t:clip t lo hi)
  (t:map (lambda (x) (max lo (min hi x))) t))

;; Slice rows [start, end) from a 2D tensor
(define (t:slice t start end)
  (let* ([shape (tensor-shape t)]
         [rows (car shape)]
         [cols (cadr shape)]
         [actual-end (min end rows)]
         [num-rows (- actual-end start)]
         [data (tensor-data t)]
         [new-data (make-vector (* num-rows cols) 0.0)])
    (for ([i (in-range num-rows)])
      (for ([j (in-range cols)])
        (vector-set! new-data (+ (* i cols) j)
                     (vector-ref data (+ (* (+ start i) cols) j)))))
    (tensor (list num-rows cols) new-data)))

;; Concatenate tensors along first axis (all must have same remaining dims)
(define (t:concat tensors)
  (let* ([shapes (map tensor-shape tensors)]
         [first-shape (car shapes)]
         [rest-dims (cdr first-shape)]
         ;; Validate all tensors have same dims except first
         [_ (for ([s (cdr shapes)])
              (unless (equal? (cdr s) rest-dims)
                (error "t:concat: All tensors must have the same shape except the first dimension")))]
         [total-rows (apply + (map car shapes))]
         [row-size (apply * rest-dims)]
         [new-data (make-vector (* total-rows row-size) 0.0)]
         [offset 0])
    (for ([t tensors])
      (let ([data (tensor-data t)]
            [n (vector-length (tensor-data t))])
        (for ([i (in-range n)])
          (vector-set! new-data (+ offset i) (vector-ref data i)))
        (set! offset (+ offset n))))
    (tensor (cons total-rows rest-dims) new-data)))
