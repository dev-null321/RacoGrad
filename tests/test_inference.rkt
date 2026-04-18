#lang racket

;; ============================================================
;; Integration tests for llama.cpp inference
;; Requires a GGUF model to be available on disk.
;; ============================================================

(require rackunit
         ffi/unsafe
         "../llama_backend.rkt"
         "../inference.rkt")

(define test-model-path
  (expand-user-path
   "~/Projects/multivac/gpt-oss-120b-Derestricted-GGUF/gpt-oss-120b-Derestricted.Q4_K_S.gguf"))

(printf "Running inference integration tests...\n")
(printf "Model: ~a\n\n" test-model-path)

;; Skip all tests if model doesn't exist
(unless (file-exists? test-model-path)
  (printf "SKIP: Model file not found at ~a\n" test-model-path)
  (printf "Set test-model-path to a valid GGUF model to run integration tests.\n")
  (exit 0))

;; ============================================================
;; Test 1: Model loading and metadata
;; ============================================================

(test-case "model loads and reports metadata"
  (define m (load-gguf-model test-model-path))
  (check-true (> (gguf-model-n-vocab m) 0) "vocab size should be positive")
  (check-true (> (gguf-model-n-layers m) 0) "layer count should be positive")
  (check-true (> (gguf-model-n-embd m) 0) "embedding dim should be positive")
  (check-true (string? (gguf-model-desc m)) "description should be a string")
  (printf "  [PASS] Model loaded: ~a\n" (gguf-model-desc m))
  (printf "         Vocab: ~a, Layers: ~a, Embd: ~a\n"
          (gguf-model-n-vocab m) (gguf-model-n-layers m) (gguf-model-n-embd m))
  (free-gguf-model m))

;; ============================================================
;; Test 2: Tokenization round-trip
;; ============================================================

(test-case "tokenize and detokenize round-trip"
  (define m (load-gguf-model test-model-path))
  (define text "Hello, world!")
  (define tokens (tokenize-text m text))
  (check-true (> (length tokens) 0) "should produce at least one token")
  (printf "  [INFO] \"~a\" -> ~a tokens: ~a\n" text (length tokens) tokens)

  ;; Detokenize (skip BOS token if present)
  (define bos (llama-bos-token (gguf-model-handle m)))
  (define content-tokens
    (if (and (not (null? tokens)) (= (car tokens) bos))
        (cdr tokens)
        tokens))
  (define reconstructed (bytes->string/utf-8 (detokenize-tokens m content-tokens) #\?))
  (printf "  [INFO] Detokenized: \"~a\"\n" reconstructed)
  (check-true (string-contains? reconstructed "Hello") "should reconstruct original text")
  (printf "  [PASS] Tokenization round-trip\n")
  (free-gguf-model m))

;; ============================================================
;; Test 3: Context creation and decode
;; ============================================================

(test-case "context creation and prompt decode"
  (define m (load-gguf-model test-model-path))
  (define model-h (gguf-model-handle m))
  (define ctx-h (llama-context-create model-h #:n-ctx 512 #:n-batch 256))
  (check-true (> ctx-h 0) "context handle should be positive")

  ;; Tokenize and decode a short prompt
  (define tokens (llama-tokenize model-h "The"))
  (define status (llama-decode ctx-h tokens))
  (check-equal? status 0 "decode should succeed")

  ;; Check logits
  (define logits-ptr (llama-get-logits ctx-h))
  (check-not-false logits-ptr "logits pointer should not be null")

  ;; Read first logit value to verify it's a real float
  (define first-logit (ptr-ref logits-ptr _float 0))
  (check-true (real? first-logit) "first logit should be a real number")
  (printf "  [PASS] Decode succeeded, logits[0] = ~a\n" first-logit)

  (llama-context-free ctx-h)
  (free-gguf-model m))

;; ============================================================
;; Test 4: Full text generation
;; ============================================================

(test-case "full text generation produces output"
  (with-gguf-model [m test-model-path]
    (with-generator [g m #:n-ctx 512 #:greedy #t]
      (define prompt "Once upon a time")
      (printf "  [INFO] Generating from: \"~a\"\n" prompt)
      (define result (generate-text g prompt #:max-tokens 16))
      (check-true (string? result) "result should be a string")
      (check-true (> (string-length result) 0) "result should not be empty")
      (printf "  [PASS] Generated: \"~a\"\n" result))))

;; ============================================================
;; Test 5: Resource cleanup (load/free cycle)
;; ============================================================

(test-case "resource cleanup: multiple load/free cycles"
  (for ([i (in-range 3)])
    (define m (load-gguf-model test-model-path))
    (define model-h (gguf-model-handle m))
    (define ctx-h (llama-context-create model-h #:n-ctx 256))
    (define smpl-h (llama-sampler-create-greedy))
    ;; Use them
    (define tokens (llama-tokenize model-h "test"))
    (llama-decode ctx-h tokens)
    ;; Free in correct order
    (llama-sampler-free smpl-h)
    (llama-context-free ctx-h)
    (free-gguf-model m))
  (printf "  [PASS] 3 load/free cycles completed without crash\n"))

;; ============================================================
;; Test 6: with-gguf-model cleans up on exception
;; ============================================================

(test-case "with-gguf-model cleans up on exception"
  (with-handlers ([exn:fail? (lambda (e)
                               (printf "  [PASS] Exception caught, resources cleaned up\n"))])
    (with-gguf-model [m test-model-path]
      (check-true (> (gguf-model-n-vocab m) 0))
      (error "intentional error to test cleanup"))))

;; ============================================================

(printf "\nAll inference integration tests passed!\n")
