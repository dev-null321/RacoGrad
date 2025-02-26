#lang racket

(require "tensor.rkt")

(provide dense-forward 
         mean-squared-error 
         dense-backward 
         relu 
         relu-derivative 
         initialize-fnn 
         sigmoid 
         sigmoid-derivative)

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
