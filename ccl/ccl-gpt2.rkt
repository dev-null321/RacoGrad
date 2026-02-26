#lang racket
;; ccl-gpt2.rkt — Racket wrapper for CCL on GPT-2 124M
;; Integrates with the existing CCL framework via Python FFI

(require racket/system racket/port racket/string)

(provide run-ccl-gpt2
         run-ccl-gpt2-comparison
         load-ccl-results)

;; Path to the Python CCL script
(define ccl-gpt2-script
  (build-path (current-directory) "ccl" "ccl_gpt2.py"))

;; Run CCL experiment on GPT-2
;; mode: "frozen" | "live" | "both"
(define (run-ccl-gpt2 #:mode [mode "both"]
                       #:rank [rank 16]
                       #:epochs [epochs 3]
                       #:lr [lr #f])
  (define cmd
    (string-append
     "python3 " (path->string ccl-gpt2-script)
     " --mode " mode
     " --rank " (number->string rank)
     " --epochs " (number->string epochs)
     (if lr (string-append " --lr " (number->string lr)) "")))
  (printf "Running: ~a\n" cmd)
  (define result (with-output-to-string (λ () (system cmd))))
  (displayln result)
  result)

;; Convenience: run full comparison
(define (run-ccl-gpt2-comparison)
  (run-ccl-gpt2 #:mode "both"))

;; Load results JSON
(define (load-ccl-results [mode "live"])
  (define path (build-path "ccl_results" mode "metrics.json"))
  (if (file-exists? path)
      (call-with-input-file path
        (λ (in) (read-json in)))
      (error 'load-ccl-results "Results not found at ~a" path)))

(module+ main
  (run-ccl-gpt2-comparison))
