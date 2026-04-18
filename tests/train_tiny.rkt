#lang racket

;; ============================================================
;; Tiny training smoke test: prove backward + optimizer actually
;; update weights and drive the loss down on a toy seq2seq task.
;;
;; Not a benchmark — just verifies the full training path works:
;;   forward → cross-entropy → backward → adam-step → (repeat)
;;
;; Expected: loss at step 0 ≈ ln(vocab_size) ≈ 3.5, trends down.
;; ============================================================

(require "../device_pytorch.rkt"
         "../nn.rkt"
         "../transformer.rkt"
         "../training.rkt")

(define VOCAB    32)
(define D-MODEL  32)
(define N-HEADS  2)
(define N-LAYERS 1)
(define D-FF     64)
(define MAX-LEN  16)
(define STEPS    20)

(printf "=== RacoGrad tiny transformer training ===~n")
(printf "config: vocab=~a d=~a heads=~a layers=~a~n~n"
        VOCAB D-MODEL N-HEADS N-LAYERS)

(define model (make-transformer VOCAB VOCAB
                                #:d-model D-MODEL
                                #:num-heads N-HEADS
                                #:num-layers N-LAYERS
                                #:d-ff D-FF
                                #:max-len MAX-LEN))
(define mod (transformer-module model))

(define all-params (parameters mod))
(printf "parameters: ~a tensors, ~a total weights~n~n"
        (length all-params) (num-parameters mod))

;; Tiny fixed batch: batch=2, seq=8. Memorization task — same src = tgt.
(define src (reshape (arange 0 16) (list 2 8)))
(define tgt src)

(define optimizer (make-adam-from-params all-params #:lr 0.01))

(for ([step (in-range STEPS)])
  (define loss-val (train-step model optimizer cross-entropy-loss src tgt))
  (when (or (< step 3) (= 0 (modulo step 5)) (= step (sub1 STEPS)))
    (printf "step ~a  loss = ~a~n"
            (~a step #:min-width 3 #:align 'right)
            (real->decimal-string loss-val 4))))

(adam-free optimizer)
(printf "~nDone.~n")
