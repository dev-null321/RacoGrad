#lang racket
(require rackunit
         "gpt2.rkt"
         "device_pytorch.rkt")

(displayln "[test] GPT-2 small libtorch forward shape")
(define model (make-gpt2 50257 #:d-model 768 #:num-heads 12 #:num-layers 12))
(define ids (to-long (to-cuda (tensor (list (list 464 2068 7586 21831 318))))))
(define logits (gpt2-forward model ids))
(check-equal? (shape logits) '(1 5 50257))
(displayln "[ok] forward shape")

(displayln "[test] greedy step works")
(define last-logits (slice-dim logits 1 4 5))
(define next-token-t (argmax (squeeze last-logits 1) -1))
(check-equal? (shape next-token-t) '(1))
(displayln "[ok] decode step")

(displayln "ALL TESTS PASSED")
