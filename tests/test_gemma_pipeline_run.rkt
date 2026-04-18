#lang racket

;; ============================================================
;; Runtime smoke test: invoke a macro-emitted pipeline against
;; the real Gemma model. Kept minimal on purpose — load + tokenize
;; + free, nothing more. This proves the compile-time-generated
;; code actually runs real FFI calls end-to-end.
;; ============================================================

(require "../inference.rkt")

(define gemma-path
  (expand-user-path "~/.multivac/models/gemma-4-31B-it-f16.gguf"))

(define-inference-pipeline (tokenize-via-pipeline path prompt)
  (let* ([m (load-model path)])
    (tokenize m prompt)))

(printf "Calling pipeline-emitted function against Gemma...\n")
(define tokens (tokenize-via-pipeline gemma-path "Hello, world"))
(printf "Token ids: ~a\n" tokens)
(printf "Token count: ~a\n" (length tokens))
(printf "OK — macro-emitted pipeline ran end-to-end.\n")
