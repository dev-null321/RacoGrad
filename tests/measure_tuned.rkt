#lang racket

;; Compile-time-tuned pipeline: n_ctx / n_batch derived from
;; declared prompt + gen bounds, emitted directly into the call
;; via `define-inference-pipeline`. No runtime option plumbing.

(require "../inference.rkt")

(define path
  (expand-user-path "~/.multivac/models/gemma-4-31B-it-f16.gguf"))

(define-inference-pipeline (decode-once path prompt)
  #:max-prompt-tokens 16
  #:max-gen-tokens    16
  (let* ([m (load-model path)]
         [c (create-context m)]
         [t (tokenize m prompt)])
    (decode c t)))

(decode-once path "Hello, world")
(printf "measure_tuned: done\n")
