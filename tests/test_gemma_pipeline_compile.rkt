#lang racket

;; ============================================================
;; Compile-time wiring test against the real Gemma GGUF.
;;
;; This file ONLY defines a pipeline and prints its expansion.
;; It does NOT call the pipeline — no model is loaded, no
;; inference runs. Safe to execute regardless of watchdog state.
;; ============================================================

(require "../inference.rkt")

(define gemma-path
  (expand-user-path "~/.multivac/models/gemma-4-31B-it-f16.gguf"))

(unless (file-exists? gemma-path)
  (error 'test "Gemma model not found at ~a" gemma-path))
(printf "Gemma model present: ~a\n" gemma-path)

;; Pipeline with a shared model handle via let*. The macro should
;; emit exactly ONE llama-model-load and reuse its handle in both
;; llama-context-create and llama-tokenize.
(define-inference-pipeline (gemma-one-shot path prompt)
  (let* ([m (load-model path)])
    (sample (create-sampler-greedy)
            (decode (create-context m)
                    (tokenize m prompt)))))

(printf "\nPipeline defined: ~a\n" gemma-one-shot)
(printf "(not called — this test only proves compile-time wiring)\n")

(define expanded
  (syntax->datum
   (expand-once
    #'(define-inference-pipeline (gemma-one-shot path prompt)
        (let* ([m (load-model path)])
          (sample (create-sampler-greedy)
                  (decode (create-context m)
                          (tokenize m prompt))))))))

(printf "\n--- Expanded form ---\n")
(pretty-print expanded)

;; Assert only one llama-model-load appears in the expansion.
(define (count-calls datum name)
  (cond
    [(pair? datum)
     (+ (if (eq? (car datum) name) 1 0)
        (apply + (map (lambda (x) (count-calls x name)) datum)))]
    [else 0]))

(define n-loads (count-calls expanded 'llama-model-load))
(define n-frees (count-calls expanded 'llama-model-free))

(printf "\n--- Invariants ---\n")
(printf "llama-model-load calls in expansion: ~a (expected 1)\n" n-loads)
(printf "llama-model-free calls in expansion: ~a (expected 1)\n" n-frees)
(unless (and (= n-loads 1) (= n-frees 1))
  (error 'test "expansion is not sharing the model handle"))
(printf "OK — single load, single free.\n")
