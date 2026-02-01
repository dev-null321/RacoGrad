#lang racket

;; ============================================================
;; RacoGrad Comprehensive Test Suite
;; ============================================================
;; Tests for tensor operations, autograd, and neural network layers.
;; Run with: racket test-suite.rkt

(require "tensor.rkt"
         "autograd.rkt")

(define pass-count 0)
(define fail-count 0)
(define test-section "")

(define (set-section! name)
  (set! test-section name)
  (printf "\n━━━ ~a ━━━\n" name))

(define (check name condition)
  (if condition
      (begin
        (set! pass-count (add1 pass-count))
        (printf "  ✓ ~a\n" name))
      (begin
        (set! fail-count (add1 fail-count))
        (printf "  ✗ FAIL: ~a\n" name))))

(define (approx= a b [epsilon 1e-6])
  (< (abs (- a b)) epsilon))

(define (tensor-approx= t1 t2 [epsilon 1e-6])
  (and (equal? (t:shape t1) (t:shape t2))
       (for/and ([a (in-vector (t:data t1))]
                 [b (in-vector (t:data t2))])
         (approx= a b epsilon))))

;; ============================================================
;; 1. Tensor Creation
;; ============================================================
(set-section! "Tensor Creation")

(let ([t (t:create '(2 3) '(1 2 3 4 5 6))])
  (check "create from list" (and (equal? (t:shape t) '(2 3))
                                  (equal? (vector->list (t:data t)) '(1 2 3 4 5 6)))))

(let ([t (t:create '(2 3) #(1 2 3 4 5 6))])
  (check "create from vector" (equal? (t:shape t) '(2 3))))

(let ([t (t:create '(3) '(10 20 30))])
  (check "create 1D tensor" (and (= (t:size t) 3) (= (t:rank t) 1))))

(check "create with wrong size returns #f"
       (not (t:create '(2 2) '(1 2 3))))

(let ([t (t:zeros '(2 3))])
  (check "t:zeros" (and (equal? (t:shape t) '(2 3))
                         (= (t:sum t) 0.0))))

(let ([t (t:ones '(3 2))])
  (check "t:ones" (and (equal? (t:shape t) '(3 2))
                        (= (t:sum t) 6.0))))

(let ([t (t:fill '(2 2) 5.0)])
  (check "t:fill" (= (t:sum t) 20.0)))

(let ([eye (t:eye 3)])
  (check "t:eye" (and (= (vector-ref (t:data eye) 0) 1.0)
                       (= (vector-ref (t:data eye) 1) 0.0)
                       (= (vector-ref (t:data eye) 4) 1.0)
                       (= (vector-ref (t:data eye) 8) 1.0)
                       (= (t:sum eye) 3.0))))

(let ([t (t:random '(5 5) 1.0)])
  (check "t:random shape" (equal? (t:shape t) '(5 5)))
  (check "t:random values in range" (and (>= (t:min-val t) 0.0)
                                          (<= (t:max-val t) 1.0))))

;; ============================================================
;; 2. Tensor Accessors & Utilities
;; ============================================================
(set-section! "Tensor Accessors & Utilities")

(let ([t (t:create '(2 3) '(1 2 3 4 5 6))])
  (check "t:ref" (and (= (t:ref t 0 0) 1) (= (t:ref t 1 2) 6)))
  (check "t:size" (= (t:size t) 6))
  (check "t:rank" (= (t:rank t) 2)))

(let ([t (t:create '(2 3) '(1 2 3 4 5 6))])
  (check "t:reshape" (equal? (t:shape (t:reshape t '(3 2))) '(3 2))))

;; ============================================================
;; 3. Elementwise Math Operations
;; ============================================================
(set-section! "Elementwise Math")

(let ([t (t:create '(3) '(-2.0 0.0 3.0))])
  (check "t:negate" (equal? (vector->list (t:data (t:negate t))) '(2.0 -0.0 -3.0)))
  (check "t:abs" (equal? (vector->list (t:data (t:abs t))) '(2.0 0.0 3.0)))
  (check "t:square" (equal? (vector->list (t:data (t:square t))) '(4.0 0.0 9.0))))

(let ([t (t:create '(3) '(0.0 1.0 2.0))])
  (check "t:exp" (approx= (vector-ref (t:data (t:exp t)) 0) 1.0))
  (check "t:exp e^1" (approx= (vector-ref (t:data (t:exp t)) 1) (exp 1))))

(let ([t (t:create '(3) '(1.0 2.718281828 1.0))])
  (check "t:log(1)=0" (approx= (vector-ref (t:data (t:log t)) 0) 0.0)))

(let ([t (t:create '(3) '(0.0 1.0 4.0))])
  (check "t:sqrt" (and (approx= (vector-ref (t:data (t:sqrt t)) 0) 0.0)
                        (approx= (vector-ref (t:data (t:sqrt t)) 2) 2.0))))

(let ([t (t:create '(4) '(-1.0 0.5 3.0 10.0))])
  (check "t:clip" (let ([c (t:clip t 0.0 5.0)])
                    (and (approx= (vector-ref (t:data c) 0) 0.0)
                         (approx= (vector-ref (t:data c) 1) 0.5)
                         (approx= (vector-ref (t:data c) 2) 3.0)
                         (approx= (vector-ref (t:data c) 3) 5.0)))))

(let ([t (t:create '(2) '(10.0 20.0))])
  (check "t:map" (let ([doubled (t:map (lambda (x) (* 2 x)) t)])
                   (equal? (vector->list (t:data doubled)) '(20.0 40.0)))))

;; ============================================================
;; 4. Reduction Operations
;; ============================================================
(set-section! "Reduction Operations")

(let ([t (t:create '(2 3) '(1 2 3 4 5 6))])
  (check "t:sum" (= (t:sum t) 21))
  (check "t:mean" (approx= (t:mean t) 3.5))
  (check "t:max-val" (= (t:max-val t) 6))
  (check "t:min-val" (= (t:min-val t) 1)))

;; ============================================================
;; 5. Tensor Arithmetic
;; ============================================================
(set-section! "Tensor Arithmetic")

(let ([a (t:create '(2 2) '(1 2 3 4))]
      [b (t:create '(2 2) '(5 6 7 8))])
  (check "t:add same shape"
         (equal? (vector->list (t:data (t:add a b))) '(6 8 10 12)))
  (check "t:sub same shape"
         (equal? (vector->list (t:data (t:sub a b))) '(-4 -4 -4 -4))))

(let ([a (t:create '(2 3) '(1 2 3 4 5 6))]
      [b (t:create '(3) '(10 20 30))])
  (check "t:add 2D + 1D row broadcast"
         (equal? (vector->list (t:data (t:add a b))) '(11 22 33 14 25 36)))
  (check "t:sub 2D - 1D row broadcast"
         (equal? (vector->list (t:data (t:sub a b))) '(-9 -18 -27 -6 -15 -24))))

(let ([a (t:create '(2 3) '(1 2 3 4 5 6))]
      [b (t:create '(1 3) '(10 20 30))])
  (check "t:add 2D + (1,N) broadcast"
         (equal? (vector->list (t:data (t:add a b))) '(11 22 33 14 25 36))))

(let ([t (t:create '(2 3) '(1 2 3 4 5 6))])
  (check "t:scale"
         (equal? (vector->list (t:data (t:scale t 2))) '(2 4 6 8 10 12))))

;; ============================================================
;; 6. Matrix Multiplication
;; ============================================================
(set-section! "Matrix Multiplication")

(let ([a (t:create '(2 3) '(1 2 3 4 5 6))]
      [b (t:create '(3 2) '(7 8 9 10 11 12))])
  (let ([c (t:mul a b)])
    (check "matmul shape" (equal? (t:shape c) '(2 2)))
    ;; [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    ;; [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    (check "matmul values"
           (and (approx= (t:ref c 0 0) 58.0)
                (approx= (t:ref c 0 1) 64.0)
                (approx= (t:ref c 1 0) 139.0)
                (approx= (t:ref c 1 1) 154.0)))))

;; Identity matrix multiplication: A * I = A
(let ([a (t:create '(2 3) '(1 2 3 4 5 6))]
      [eye (t:eye 3)])
  (check "matmul by identity"
         (tensor-approx= (t:mul a eye) a)))

;; Elementwise multiplication
(let ([a (t:create '(2 2) '(1 2 3 4))]
      [b (t:create '(2 2) '(5 6 7 8))])
  (check "elementwise mul"
         (equal? (vector->list (t:data (t:emul a b))) '(5 12 21 32))))

;; ============================================================
;; 7. Dot Product
;; ============================================================
(set-section! "Dot Product")

(let ([a (t:create '(3) '(1 2 3))]
      [b (t:create '(3) '(4 5 6))])
  (check "t:dot" (= (t:dot a b) 32))) ;; 1*4 + 2*5 + 3*6

(let ([a (t:create '(3) '(1 0 0))]
      [b (t:create '(3) '(0 1 0))])
  (check "t:dot orthogonal" (= (t:dot a b) 0)))

;; ============================================================
;; 8. Transpose
;; ============================================================
(set-section! "Transpose")

(let* ([a (t:create '(2 3) '(1 2 3 4 5 6))]
       [at (t:transpose a)])
  (check "transpose shape" (equal? (t:shape at) '(3 2)))
  (check "transpose values"
         (and (= (t:ref at 0 0) 1) (= (t:ref at 0 1) 4)
              (= (t:ref at 1 0) 2) (= (t:ref at 1 1) 5)
              (= (t:ref at 2 0) 3) (= (t:ref at 2 1) 6))))

;; Double transpose = original
(let ([a (t:create '(3 4) (for/list ([i 12]) i))])
  (check "double transpose = identity"
         (tensor-approx= (t:transpose (t:transpose a)) a)))

;; ============================================================
;; 9. Slice & Concat
;; ============================================================
(set-section! "Slice & Concat")

(let* ([t (t:create '(4 3) '(1 2 3 4 5 6 7 8 9 10 11 12))]
       [s (t:slice t 1 3)])
  (check "slice shape" (equal? (t:shape s) '(2 3)))
  (check "slice values" (equal? (vector->list (t:data s)) '(4 5 6 7 8 9))))

(let* ([a (t:create '(2 3) '(1 2 3 4 5 6))]
       [b (t:create '(1 3) '(7 8 9))]
       [c (t:concat (list a b))])
  (check "concat shape" (equal? (t:shape c) '(3 3)))
  (check "concat values" (equal? (vector->list (t:data c)) '(1 2 3 4 5 6 7 8 9))))

;; ============================================================
;; 10. Activation Functions
;; ============================================================
(set-section! "Activation Functions")

(let ([t (t:create '(4) '(-2.0 -0.5 0.0 3.0))])
  (let ([r (relu t)])
    (check "relu" (equal? (vector->list (t:data r)) '(0.0 0.0 0.0 3.0))))
  (let ([r (relu-derivative t)])
    (check "relu-derivative" (equal? (vector->list (t:data r)) '(0 0 0 1)))))

(let ([t (t:create '(1) '(0.0))])
  (check "sigmoid(0) = 0.5" (approx= (vector-ref (t:data (sigmoid t)) 0) 0.5)))

(let ([t (t:create '(4) '(-2.0 -0.5 0.0 3.0))])
  (let ([lr (leaky-relu t)])
    (check "leaky-relu negative" (approx= (vector-ref (t:data lr) 0) -0.02))
    (check "leaky-relu positive" (approx= (vector-ref (t:data lr) 3) 3.0))))

(let ([t (t:create '(3) '(-1.0 0.0 1.0))])
  (let ([e (elu t)])
    (check "elu negative" (approx= (vector-ref (t:data e) 0) (- (exp -1.0) 1)))
    (check "elu positive" (approx= (vector-ref (t:data e) 2) 1.0))))

(let ([t (t:create '(3) '(0.0 1.0 20.0))])
  (let ([sp (softplus t)])
    (check "softplus(0) = ln(2)" (approx= (vector-ref (t:data sp) 0) (log 2)))
    (check "softplus large x ≈ x" (approx= (vector-ref (t:data sp) 2) 20.0))))

(let ([t (t:create '(2) '(0.0 1.0))])
  (let ([sw (swish t)])
    ;; swish(0) = 0 * sigmoid(0) = 0
    (check "swish(0) = 0" (approx= (vector-ref (t:data sw) 0) 0.0))
    ;; swish(1) = 1 * sigmoid(1) ≈ 0.7311
    (check "swish(1)" (approx= (vector-ref (t:data sw) 1) 
                                (/ 1.0 (+ 1 (exp -1.0)))))))

;; ============================================================
;; 11. Dense Layer Forward/Backward
;; ============================================================
(set-section! "Dense Layer")

(let* ([input (t:create '(1 3) '(1.0 2.0 3.0))]
       [weights (t:create '(3 2) '(0.1 0.2 0.3 0.4 0.5 0.6))]
       [biases (t:create '(2) '(0.0 0.0))]
       [output (dense-forward input weights biases relu)])
  (check "dense-forward shape" (equal? (t:shape output) '(1 2)))
  ;; [1*0.1+2*0.3+3*0.5, 1*0.2+2*0.4+3*0.6] = [2.2, 2.8]
  (check "dense-forward values" 
         (and (approx= (t:ref output 0 0) 2.2)
              (approx= (t:ref output 0 1) 2.8))))

;; ============================================================
;; 12. Mean Squared Error
;; ============================================================
(set-section! "Loss Functions")

(let ([pred (t:create '(3) '(1.0 2.0 3.0))]
      [true (t:create '(3) '(1.0 2.0 3.0))])
  (check "MSE perfect prediction = 0" (approx= (mean-squared-error true pred) 0.0)))

(let ([pred (t:create '(3) '(0.0 0.0 0.0))]
      [true (t:create '(3) '(1.0 1.0 1.0))])
  (check "MSE all off by 1" (approx= (mean-squared-error true pred) 1.0)))

;; ============================================================
;; Summary
;; ============================================================
(printf "\n══════════════════════════════════\n")
(printf "Results: ~a passed, ~a failed out of ~a total\n" 
        pass-count fail-count (+ pass-count fail-count))
(when (> fail-count 0)
  (printf "⚠ SOME TESTS FAILED\n"))
(when (= fail-count 0)
  (printf "✓ ALL TESTS PASSED\n"))
(printf "══════════════════════════════════\n")

(exit (if (= fail-count 0) 0 1))
