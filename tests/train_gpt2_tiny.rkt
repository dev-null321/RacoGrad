#lang racket

;; ============================================================
;; Tiny GPT-2 training smoke test on GB10.
;;
;; Random-initialized 2-layer GPT-2 learns to memorize a fixed token
;; sequence. Proves the training path works end-to-end for GPT-2:
;;   gpt2-forward → cross-entropy → backward → adam-step.
;;
;; Not a real pretraining run — this is a correctness check. For
;; language modeling loss shape: logits are [B, T, V]; targets are
;; the tokens shifted left by 1. Here we just use src=tgt (next-token
;; prediction on the same sequence) since it's a memorization task.
;; ============================================================

(require "../device_pytorch.rkt"
         "../nn.rkt"
         "../gpt2.rkt"
         "../training.rkt")

(define VOCAB    64)
(define D-MODEL  64)
(define N-HEADS  2)
(define N-LAYERS 2)
(define SEQ-LEN  8)
(define STEPS    30)

(printf "=== RacoGrad GPT-2 tiny training ===~n")
(printf "vocab=~a d=~a heads=~a layers=~a seq=~a~n~n"
        VOCAB D-MODEL N-HEADS N-LAYERS SEQ-LEN)

(define model-list (make-gpt2 VOCAB
                              #:d-model D-MODEL
                              #:num-heads N-HEADS
                              #:num-layers N-LAYERS
                              #:max-len SEQ-LEN))
(define mod (gpt2-module model-list))
(printf "parameters: ~a tensors, ~a total weights~n~n"
        (length (parameters mod)) (num-parameters mod))

;; Toy dataset: two batches of fixed token sequences.
;; Batch 1: [0..7], Batch 2: [7..14].  Model should memorize both.
(define tokens (to-long (to-cuda (tensor (list (range 0 SEQ-LEN)
                                                (range SEQ-LEN (* 2 SEQ-LEN)))))))
(printf "input shape: ~a~n" (shape tokens))

(define optimizer (make-adam-from-params (parameters mod) #:lr 0.005))

;; Custom train-step: we're doing next-token prediction, so targets
;; are the inputs themselves (standard LM loss pattern when src=tgt).
(define (gpt2-train-step)
  (adam-zero-grad optimizer)
  (define logits (gpt2-forward model-list tokens))
  (define loss (cross-entropy-loss logits tokens))
  (backward loss)
  (adam-step optimizer)
  (get-item loss))

(printf "~nstep  loss~n")
(printf "----  ------~n")
(define t0 (current-inexact-milliseconds))
(for ([step (in-range STEPS)])
  (define loss (gpt2-train-step))
  (when (or (< step 3)
            (= 0 (modulo step 5))
            (= step (sub1 STEPS)))
    (printf " ~a   ~a~n"
            (~a step #:min-width 3 #:align 'right)
            (real->decimal-string loss 4))))
(define elapsed (/ (- (current-inexact-milliseconds) t0) 1000.0))

(printf "~nTotal: ~as (~a steps/sec)~n"
        (real->decimal-string elapsed 2)
        (real->decimal-string (/ STEPS elapsed) 1))

(adam-free optimizer)
