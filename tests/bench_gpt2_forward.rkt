#lang racket

;; ============================================================
;; GPT-2 forward-pass throughput benchmark (random weights).
;;
;; Times a warm set of forward passes at fixed shape to get a
;; ms/forward + approximate tokens/sec number that's comparable
;; across hardware. Uses random weights — this measures kernel
;; + FFI + libtorch throughput, not generation quality.
;;
;; Usage:
;;   racket tests/bench_gpt2_forward.rkt
;; ============================================================

(require "../gpt2.rkt"
         "../device_pytorch.rkt")

(define VOCAB     50257)
(define D-MODEL   768)
(define N-HEADS   12)
(define N-LAYERS  12)
(define SEQ-LEN   128)
(define WARMUP    3)
(define ITERS     20)

(printf "=== GPT-2 small forward-pass benchmark ===~n")
(printf "Config: d_model=~a, heads=~a, layers=~a, seq_len=~a~n"
        D-MODEL N-HEADS N-LAYERS SEQ-LEN)
(printf "Iterations: ~a (after ~a warmup)~n~n" ITERS WARMUP)

(define model (make-gpt2 VOCAB #:d-model D-MODEL #:num-heads N-HEADS #:num-layers N-LAYERS))

;; Random token IDs in [0, VOCAB), shape [1, SEQ-LEN], on GPU.
(define ids
  (to-long
   (to-cuda
    (tensor (list (for/list ([_ (in-range SEQ-LEN)])
                    (random VOCAB)))))))

(printf "Warmup...~n")
(for ([_ (in-range WARMUP)])
  (gpt2-forward model ids))

(printf "Benchmarking ~a iterations...~n" ITERS)
(define t0 (current-inexact-milliseconds))
(for ([_ (in-range ITERS)])
  (gpt2-forward model ids))
(define t1 (current-inexact-milliseconds))

(define total-ms (- t1 t0))
(define ms-per-forward (/ total-ms ITERS))
(define forwards-per-sec (/ 1000.0 ms-per-forward))
(define tokens-per-sec   (* forwards-per-sec SEQ-LEN))

(printf "~n--- Results ---~n")
(printf "Total wall time:  ~a ms~n"     (real->decimal-string total-ms 2))
(printf "Per forward:      ~a ms~n"     (real->decimal-string ms-per-forward 2))
(printf "Forwards/sec:     ~a~n"        (real->decimal-string forwards-per-sec 2))
(printf "Effective tok/s:  ~a (seq_len=~a × forwards/sec)~n"
        (real->decimal-string tokens-per-sec 1) SEQ-LEN)
