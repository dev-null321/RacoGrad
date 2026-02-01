#lang racket

(require "tensor.rkt")

(provide dense-forward 
         mean-squared-error 
         dense-backward 
         relu 
         relu-derivative 
         initialize-fnn 
         sigmoid 
         sigmoid-derivative
         leaky-relu
         leaky-relu-derivative
         elu
         elu-derivative
         softplus
         softplus-derivative
         swish
         swish-derivative)

;; Activation functions
(define (relu x)
  (t:create (t:shape x) 
            (for/vector ([v (t:data x)]) (max 0 v))))

(define (relu-derivative x)
  (t:create (t:shape x) 
            (for/vector ([v (t:data x)]) (if (> v 0) 1 0))))

(define (sigmoid x)
  (t:create (t:shape x)
            (for/vector ([v (t:data x)]) (/ 1 (+ 1 (exp (- v)))))))

(define (sigmoid-derivative x)
  (let ([sig (sigmoid x)])
    (t:create (t:shape x) 
              (for/vector ([v (t:data sig)]) (* v (- 1 v))))))

(define (tanh x)
  (t:create (t:shape x)
            (for/vector ([v (t:data x)]) 
              (let ([e^v (exp v)]
                    [e^-v (exp (- v))])
                (/ (- e^v e^-v) (+ e^v e^-v))))))

(define (tanh-derivative x)
  (let ([t (tanh x)])
    (t:create (t:shape x)
              (for/vector ([v (t:data t)]) (- 1 (* v v))))))

;; Leaky ReLU: max(alpha*x, x) with default alpha=0.01
(define (leaky-relu x [alpha 0.01])
  (t:create (t:shape x)
            (for/vector ([v (t:data x)])
              (if (> v 0) v (* alpha v)))))

(define (leaky-relu-derivative x [alpha 0.01])
  (t:create (t:shape x)
            (for/vector ([v (t:data x)])
              (if (> v 0) 1 alpha))))

;; ELU: x if x > 0, alpha*(exp(x)-1) otherwise
(define (elu x [alpha 1.0])
  (t:create (t:shape x)
            (for/vector ([v (t:data x)])
              (if (> v 0) v (* alpha (- (exp v) 1))))))

(define (elu-derivative x [alpha 1.0])
  (t:create (t:shape x)
            (for/vector ([v (t:data x)])
              (if (> v 0) 1 (* alpha (exp v))))))

;; Softplus: log(1 + exp(x)) — smooth approximation of ReLU
(define (softplus x)
  (t:create (t:shape x)
            (for/vector ([v (t:data x)])
              ;; Numerically stable: for large v, softplus(v) ≈ v
              (if (> v 20) v (log (+ 1 (exp v)))))))

(define (softplus-derivative x)
  ;; derivative of softplus is sigmoid
  (sigmoid x))

;; Swish: x * sigmoid(x) — self-gated activation
(define (swish x)
  (let ([sig (sigmoid x)])
    (t:create (t:shape x)
              (for/vector ([v (t:data x)]
                           [s (t:data sig)])
                (* v s)))))

(define (swish-derivative x)
  (let ([sig (sigmoid x)])
    (t:create (t:shape x)
              (for/vector ([v (t:data x)]
                           [s (t:data sig)])
                (+ s (* v s (- 1 s)))))))

;; Forward pass through a dense layer
(define (dense-forward input weights biases activation-fn)
  (let* ([mul-result (t:mul input weights)]
         [output-dim (cadr (t:shape mul-result))]
         [reshaped-biases (t:reshape biases (list output-dim))]
         [z (t:add mul-result reshaped-biases)]
         [activation-output (activation-fn z)])
    activation-output))

;; Mean Squared Error
(define (mean-squared-error y-true y-pred)
  (let* ([diff (t:sub y-true y-pred)]
         [squared-diff (t:mul diff diff)]
         [sum (apply + (vector->list (t:data squared-diff)))])
    (/ sum (length (vector->list (t:data y-true))))))

;; Backward pass for a dense layer
(define (dense-backward input weights biases output grad-output activation-derivative learning-rate)
  (let* ([grad-activation (activation-derivative output)]
         [grad-z (t:mul grad-output grad-activation)]
         [grad-weights (t:mul (t:transpose input) grad-z)]
         [bias-len (vector-length (t:data biases))]
         ;; Compute grad-biases by summing each column of grad-z
         [grad-biases (t:create (list bias-len)
                                (for/vector ([j bias-len])
                                  (apply +
                                         (for/list ([i (car (t:shape grad-z))])
                                           (vector-ref (t:data grad-z)
                                                       (+ (* i bias-len) j))))))]
         [grad-input (t:mul grad-z (t:transpose weights))])
    (values grad-weights grad-biases grad-input)))

;; Initialize a fully-connected neural network layer (input-tensor, weights, biases)
(define (initialize-fnn batch-size input-dim output-dim)
  (let* ([input-data (make-list (* batch-size input-dim) 0)]
         [input-tensor (t:create (list batch-size input-dim) input-data)]
         [weight-shape (list input-dim output-dim)]
         [bias-shape (list output-dim)]
         [weights (t:random weight-shape 1.0)]
         [biases (t:random bias-shape 1.0)])
    (values input-tensor weights biases)))
