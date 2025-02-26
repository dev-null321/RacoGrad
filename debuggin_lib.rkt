#lang racket

(require "tensor.rkt")
(provide dense-forward mean-squared-error dense-backward relu relu-derivative initialize-fnn sigmoid sigmoid-derivative)

(define (relu x)
  (tensor (tensor-shape x) (for/vector ([v (tensor-data x)]) (max 0 v))))

(define (relu-derivative x)
  (tensor (tensor-shape x) (for/vector ([v (tensor-data x)]) (if (> v 0) 1 0))))

(define (sigmoid x)
  (tensor (tensor-shape x) (for/vector ([v (tensor-data x)]) (/ 1 (+ 1 (exp (- v)))))))

(define (sigmoid-derivative x)
  (let ([sig (sigmoid x)])
    (tensor (tensor-shape x) (for/vector ([v (tensor-data sig)]) (* v (- 1 v))))))

(define (tanh x)
  (tensor (tensor-shape x) (for/vector ([v (tensor-data x)]) (/ (- (exp v) (exp (- v))) (+ (exp v) (exp (- v)))))))

(define (tanh-derivative x)
  (let ([t (tanh x)])
    (tensor (tensor-shape x) (for/vector ([v (tensor-data t)]) (- 1 (* v v))))))

(define (dense-forward input weights biases activation-fn)
  (let* ([mul-result (tensor-multiply input weights)]
         [mul-result-shape (tensor-shape mul-result)]
         [output-dim (cadr mul-result-shape)]
         [reshaped-biases (reshape-tensor biases (list output-dim))]
         [z (tensor-add mul-result reshaped-biases)]
         [activation-output (activation-fn z)])
    activation-output))

(define (mean-squared-error y-true y-pred)
  (let* ([diff (tensor-subtract y-true y-pred)]
         [squared-diff (tensor-multiply diff diff)]
         [sum (apply + (vector->list (tensor-data squared-diff)))])
    (/ sum (length (vector->list (tensor-data y-true))))))

(define (dense-backward input weights biases output grad-output activation-derivative learning-rate)
  (displayln "dense-backward: Starting")
  (displayln (string-append "Input shape: " (format "~a" (tensor-shape input))))
  (displayln (string-append "Weights shape: " (format "~a" (tensor-shape weights))))
  (displayln (string-append "Biases shape: " (format "~a" (tensor-shape biases))))
  (displayln (string-append "Output shape: " (format "~a" (tensor-shape output))))
  (displayln (string-append "Grad-output shape: " (format "~a" (tensor-shape grad-output))))
  
  (let* ([grad-activation (activation-derivative output)]
         [_ (displayln (string-append "Grad-activation shape: " (format "~a" (tensor-shape grad-activation))))]
         [grad-z (tensor-multiply grad-output grad-activation)]
         [_ (displayln (string-append "Grad-z shape: " (format "~a" (tensor-shape grad-z))))]
         [grad-weights (tensor-multiply (transpose input) grad-z)]
         [_ (displayln (string-append "Grad-weights shape: " (format "~a" (tensor-shape grad-weights))))]
         [grad-biases (tensor (list (vector-length (tensor-data biases)))
                              (for/vector ([j (vector-length (tensor-data biases))])
                                (apply + (for/list ([i (car (tensor-shape grad-z))])
                                           (vector-ref (tensor-data grad-z) (+ (* i (vector-length (tensor-data biases))) j))))))]
         [_ (displayln (string-append "Grad-biases shape: " (format "~a" (tensor-shape grad-biases))))]
         [grad-input (tensor-multiply grad-z (transpose weights))]
         [_ (displayln (string-append "Grad-input shape: " (format "~a" (tensor-shape grad-input))))])
    (displayln "dense-backward: Finished")
    (values grad-weights grad-biases grad-input)))

(define (initialize-fnn batch-size input-dim output-dim)
  (let* ([input-data (make-list (* batch-size input-dim) 0)]
         [input-tensor (create-tensor (list batch-size input-dim) input-data)]
         [weight-shape (list input-dim output-dim)]
         [bias-shape (list output-dim)]  ; Ensure same size as output dim
         [weights (random-tensor weight-shape 1.0)]
         [biases (random-tensor bias-shape 1.0)])
    (values input-tensor weights biases)))
