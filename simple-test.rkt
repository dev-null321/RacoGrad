#lang racket

;; Simple test for basic tensor operations and autograd
;; This is a quick smoke test — see test-suite.rkt for comprehensive tests.

(require "tensor.rkt")
(require "autograd.rkt")

(define (test-fnn)
  (let* ([input-dim 3]
         [hidden-dim 4]
         [output-dim 2]
         [batch-size 2]
         [learning-rate 0.01]
         [num-epochs 20]
         [input-data (list (list 0.1 0.2 0.3) (list 0.4 0.5 0.6))]
         [output-data (list (list 0.7 0.8) (list 0.9 0.1))])

    (define (create-data-tensor data)
      (let ([flattened-data (apply append data)])
        (t:create (list (length data) (length (car data))) flattened-data)))

    (let* ([input-tensor (create-data-tensor input-data)]
           [output-tensor (create-data-tensor output-data)]
           [hidden-weights (t:random (list input-dim hidden-dim) 0.1)]
           [hidden-biases (t:random (list hidden-dim) 0.1)]
           [output-weights (t:random (list hidden-dim output-dim) 0.1)]
           [output-biases (t:random (list output-dim) 0.1)])

      (displayln "Initial hidden weights:")
      (t:print hidden-weights)
      (displayln "Initial hidden biases:")
      (t:print hidden-biases)
      (displayln "Initial output weights:")
      (t:print output-weights)
      (displayln "Initial output biases:")
      (t:print output-biases)
      (newline)

      (for ([epoch (in-range num-epochs)])
        (let* ([hidden-output (dense-forward input-tensor hidden-weights hidden-biases relu)]
               [output (dense-forward hidden-output output-weights output-biases relu)]
               [loss (mean-squared-error output-tensor output)]
               [output-grad (t:sub output output-tensor)])
          (displayln (string-append "Epoch: " (number->string epoch) ", Loss: " (number->string loss)))

          (let-values ([(output-grad-weights output-grad-biases output-grad-input)
                        (dense-backward hidden-output output-weights output-biases output output-grad relu-derivative learning-rate)]
                       [(hidden-grad-weights hidden-grad-biases _)
                        (dense-backward input-tensor hidden-weights hidden-biases hidden-output
                                        (t:mul output-grad (t:transpose output-weights))
                                        relu-derivative learning-rate)])
            ;; Update weights using gradient descent
            (set! output-weights (t:sub output-weights (t:scale output-grad-weights learning-rate)))
            (set! output-biases (t:sub output-biases (t:scale output-grad-biases learning-rate)))
            (set! hidden-weights (t:sub hidden-weights (t:scale hidden-grad-weights learning-rate)))
            (set! hidden-biases (t:sub hidden-biases (t:scale hidden-grad-biases learning-rate))))))

      (displayln "\nFinal hidden weights:")
      (t:print hidden-weights)
      (displayln "Final hidden biases:")
      (t:print hidden-biases)
      (displayln "Final output weights:")
      (t:print output-weights)
      (displayln "Final output biases:")
      (t:print output-biases))))

(test-fnn)
