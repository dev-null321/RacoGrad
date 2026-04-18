#lang racket

;; Baseline: default llama-context-create settings (n_ctx=2048, n_batch=512).
;; Used for memory head-to-head against measure_tuned.rkt.

(require "../inference.rkt")

(define path
  (expand-user-path "~/.multivac/models/gemma-4-31B-it-f16.gguf"))

(define m (llama-model-load path #:n-gpu-layers 99))
(define c (llama-context-create m)) ;; defaults: n_ctx=2048, n_batch=512
(define t (llama-tokenize m "Hello, world"))
(llama-decode c t)
(llama-context-free c)
(llama-model-free m)
(printf "measure_default: done\n")
