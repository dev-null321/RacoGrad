#lang racket

(require "tensor.rkt")
(require "grad.rkt")

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
        (create-tensor (list (length data) (length (car data))) flattened-data)))

    (let* ([input-tensor (create-data-tensor input-data)]
           [output-tensor (create-data-tensor output-data)]
           [hidden-weights (random-tensor (list input-dim hidden-dim) 0.1)]
           [hidden-biases (random-tensor (list hidden-dim) 0.1)]
           [output-weights (random-tensor (list hidden-dim output-dim) 0.1)]
           [output-biases (random-tensor (list output-dim) 0.1)])

      (displayln "Initial hidden weights:")
      (print-tensor hidden-weights)
      (displayln "Initial hidden biases:")
      (print-tensor hidden-biases)
      (displayln "Initial output weights:")
      (print-tensor output-weights)
      (displayln "Initial output biases:")
      (print-tensor output-biases)
      (newline)

      (for ([epoch (in-range num-epochs)])
        (let* ([hidden-output (dense-forward input-tensor hidden-weights hidden-biases)]
               [output (dense-forward hidden-output output-weights output-biases)]
               [loss (mean-squared-error output-tensor output)]
               [output-grad (tensor-subtract output output-tensor)])
          (displayln (string-append "Epoch: " (number->string epoch) ", Loss: " (number->string loss)))

          (let-values ([(output-grad-weights output-grad-biases output-grad-input)
                        (dense-backward hidden-output output-weights output-biases output output-grad learning-rate)]
                       [(hidden-grad-weights hidden-grad-biases _)
                        (dense-backward input-tensor hidden-weights hidden-biases hidden-output
                                        (tensor-multiply output-grad (transpose output-weights))
                                        learning-rate)])
            (set! output-weights output-grad-weights)
            (set! output-biases output-grad-biases)
            (set! hidden-weights hidden-grad-weights)
            (set! hidden-biases hidden-grad-biases))))

      (displayln "Final hidden weights:")
      (print-tensor hidden-weights)
      (displayln "Final hidden biases:")
      (print-tensor hidden-biases)
      (displayln "Final output weights:")
      (print-tensor output-weights)
      (displayln "Final output biases:")
      (print-tensor output-biases))))

(test-fnn)
