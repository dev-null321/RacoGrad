#lang racket
(provide load-mnist-data)
(require "tensor.rkt")

(define (read-idx-file filename)
  (let* ([data (file->bytes filename)])
    ;; Parse IDX file format
    (let* ([magic-number (integer-bytes->integer data #f #t 0 4)]
           [data-type (bitwise-and magic-number #xFF)]
           [dimension-count (bitwise-and (arithmetic-shift magic-number -8) #xFF)]
           [dimensions (for/list ([i (in-range dimension-count)])
                        (integer-bytes->integer data #f #t (+ 4 (* i 4)) (+ 8 (* i 4))))]
           [data-start (+ 4 (* dimension-count 4))]
           [data-length (- (bytes-length data) data-start)])
      (values dimensions (subbytes data data-start)))))

(define (load-mnist-data type)
  (let* ([images-file (build-path "mnist-data" 
                                 (format "~a-images.idx3-ubyte" type))]
         [labels-file (build-path "mnist-data" 
                                 (format "~a-labels.idx1-ubyte" type))]
         [images-dims+data (read-idx-file images-file)]
         [labels-dims+data (read-idx-file labels-file)])
    ;; Convert to tensors and return
    (values 
     (t:create (list (car (car images-dims+data)) 784)
               (bytes->list (cdr images-dims+data)))
     (t:create (list (car (car labels-dims+data)))
               (bytes->list (cdr labels-dims+data))))))