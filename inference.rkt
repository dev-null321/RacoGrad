#lang racket

;; ============================================================
;; RacoGrad Inference — High-level GGUF model inference API
;;
;; Load a GGUF model and generate text from Racket.
;;
;; Usage:
;;   (require "inference.rkt")
;;   (with-gguf-model [m "path/to/model.gguf"]
;;     (with-generator [g m]
;;       (displayln (generate-text g "Once upon a time"))))
;; ============================================================

(require "llama_backend.rkt"
         "graph.rkt")

(provide
 ;; Re-export the raw backend names so anything expanded by
 ;; define-inference-pipeline at the caller site resolves.
 (all-from-out "llama_backend.rkt")

 ;; Structs
 (struct-out gguf-model)
 (struct-out generator)

 ;; Model lifecycle
 load-gguf-model
 free-gguf-model
 with-gguf-model

 ;; Generator lifecycle
 make-generator
 free-generator
 with-generator

 ;; Inference
 generate-text
 generate-tokens

 ;; Utilities
 tokenize-text
 detokenize-tokens

 ;; Compile-time pipeline + RAII helpers (re-exported from graph.rkt
 ;; so inference callers don't need a second require).
 define-inference-pipeline
 with-llama-resources
 define-generation-loop)

;; ============================================================
;; Data types
;; ============================================================

(struct gguf-model (handle path desc n-vocab n-layers n-embd) #:transparent)
(struct generator (model ctx-h sampler-h) #:transparent)

;; ============================================================
;; Model lifecycle
;; ============================================================

(define (load-gguf-model path #:n-gpu-layers [ngl 99])
  (printf "Loading GGUF model: ~a\n" path)
  (define model-h (llama-model-load path #:n-gpu-layers ngl))
  (define desc     (llama-model-desc model-h))
  (define n-vocab  (llama-n-vocab model-h))
  (define n-layers (llama-model-n-layer model-h))
  (define n-embd   (llama-model-n-embd model-h))
  (printf "  Model: ~a\n" desc)
  (printf "  Vocab: ~a tokens\n" n-vocab)
  (printf "  Layers: ~a, Embedding: ~a\n" n-layers n-embd)
  (gguf-model model-h path desc n-vocab n-layers n-embd))

(define (free-gguf-model m)
  (llama-model-free (gguf-model-handle m)))

(define-syntax-rule (with-gguf-model [var path opts ...] body ...)
  (let ([var (load-gguf-model path opts ...)])
    (dynamic-wind
     void
     (lambda () body ...)
     (lambda () (free-gguf-model var)))))

;; ============================================================
;; Generator lifecycle
;; ============================================================

(define (make-generator model
                        #:n-ctx        [n-ctx 2048]
                        #:n-batch      [n-batch 512]
                        #:cache-type-k [cache-type-k 'f16]
                        #:cache-type-v [cache-type-v 'f16]
                        #:temp         [temp 0.8]
                        #:top-k        [top-k 40]
                        #:top-p        [top-p 0.9]
                        #:seed         [seed #xFFFFFFFF]
                        #:greedy       [greedy #f])
  (define model-h (gguf-model-handle model))
  (define ctx-h (llama-context-create model-h
                                      #:n-ctx        n-ctx
                                      #:n-batch      n-batch
                                      #:cache-type-k cache-type-k
                                      #:cache-type-v cache-type-v))
  (define sampler-h
    (if greedy
        (llama-sampler-create-greedy)
        (llama-sampler-create #:seed seed #:top-k top-k #:top-p top-p #:temp temp)))
  (generator model ctx-h sampler-h))

(define (free-generator g)
  (llama-sampler-free (generator-sampler-h g))
  (llama-context-free (generator-ctx-h g)))

(define-syntax-rule (with-generator [var model opts ...] body ...)
  (let ([var (make-generator model opts ...)])
    (dynamic-wind
     void
     (lambda () body ...)
     (lambda () (free-generator var)))))

;; ============================================================
;; Tokenization utilities
;; ============================================================

(define (tokenize-text model text #:add-bos [add-bos #t])
  (llama-tokenize (gguf-model-handle model) text #:add-bos add-bos))

(define (detokenize-tokens model tokens)
  (llama-detokenize (gguf-model-handle model) tokens))

;; ============================================================
;; Text generation
;; ============================================================

(define (generate-tokens gen prompt-tokens
                         #:max-tokens [max-tokens 256]
                         #:callback [callback #f])
  (define model (generator-model gen))
  (define ctx-h (generator-ctx-h gen))
  (define sampler-h (generator-sampler-h gen))
  (define model-h (gguf-model-handle model))

  ;; Decode the prompt
  (llama-decode ctx-h prompt-tokens)

  ;; Auto-regressive generation loop
  (let loop ([generated '()]
             [n 0])
    (if (>= n max-tokens)
        (reverse generated)
        (let ([new-token (llama-sampler-sample sampler-h ctx-h #:idx -1)])
          (cond
            [(llama-is-eog? model-h new-token)
             (reverse generated)]
            [else
             (when callback
               (callback new-token
                         (bytes->string/utf-8
                          (llama-detokenize-token model-h new-token)
                          #\?)))
             ;; Decode the new token for next iteration
             (llama-decode ctx-h (list new-token))
             (loop (cons new-token generated) (add1 n))])))))

(define (generate-text gen prompt
                       #:max-tokens [max-tokens 256]
                       #:callback [callback #f])
  (define model (generator-model gen))
  (define model-h (gguf-model-handle model))

  ;; Tokenize prompt
  (define prompt-tokens (llama-tokenize model-h prompt))
  (printf "Prompt tokens: ~a\n" (length prompt-tokens))

  ;; Print prompt tokens
  (for ([tok prompt-tokens])
    (define piece (llama-detokenize-token model-h tok))
    (display (bytes->string/utf-8 piece #\?)))
  (flush-output)

  ;; Generate
  (define gen-tokens
    (generate-tokens gen prompt-tokens
                     #:max-tokens max-tokens
                     #:callback (or callback
                                    (lambda (tok text)
                                      (display text)
                                      (flush-output)))))

  (newline)

  ;; Return full generated text
  (bytes->string/utf-8
   (llama-detokenize model-h gen-tokens)
   #\?))

;; ============================================================
;; Main — run directly for quick test
;; ============================================================

(module+ main
  (require racket/cmdline)

  (define model-path
    (make-parameter
     (expand-user-path
      "~/Projects/multivac/gpt-oss-120b-Derestricted-GGUF/gpt-oss-120b-Derestricted.Q4_K_S.gguf")))
  (define prompt-text      (make-parameter "Hello, my name is"))
  (define max-tok          (make-parameter 64))
  (define n-ctx-param      (make-parameter 2048))
  (define n-batch-param    (make-parameter 512))
  (define cache-type-k-param (make-parameter 'f16))
  (define cache-type-v-param (make-parameter 'f16))
  (define n-gpu            (make-parameter 99))
  (define greedy?          (make-parameter #f))

  ;; Normalize CLI string → Racket symbol accepted by llama-context-create.
  ;; Accepts f16, q8_0, q4_0, plus friendly aliases q8 and q4.
  ;; Raises a clear error on anything else so users get a fast-fail.
  (define (parse-cache-type s flag-name)
    (case (string->symbol s)
      [(f16)       'f16]
      [(q8_0 q8)   'q8_0]
      [(q4_0 q4)   'q4_0]
      [else (error flag-name
                   "unsupported value ~a (expected: f16 q8_0 q4_0; aliases q8 q4)"
                   s)]))

  (command-line
   #:program "racograd-inference"
   #:once-each
   [("-m" "--model")      path "Path to GGUF model"
                               (model-path path)]
   [("-p" "--prompt")     p    "Prompt text"
                               (prompt-text p)]
   [("-n" "--max-tokens") n    "Max tokens to generate"
                               (max-tok (string->number n))]
   [("-c" "--n-ctx")      c    "KV-cache context size (smaller = less VRAM; default 2048)"
                               (n-ctx-param (string->number c))]
   [("-b" "--n-batch")    b    "Compute batch size (smaller = less VRAM for compute buffer; default 512)"
                               (n-batch-param (string->number b))]
   [("--cache-type-k")    ct   "KV cache K precision: f16 | q8_0 | q4_0 (aliases q8/q4); default f16"
                               (cache-type-k-param (parse-cache-type ct '--cache-type-k))]
   [("--cache-type-v")    ct   "KV cache V precision: f16 | q8_0 | q4_0 (aliases q8/q4); default f16"
                               (cache-type-v-param (parse-cache-type ct '--cache-type-v))]
   [("--ngl")             ngl  "GPU layers to offload"
                               (n-gpu (string->number ngl))]
   [("--greedy")               "Use greedy sampling"
                               (greedy? #t)])

  (with-gguf-model [m (model-path) #:n-gpu-layers (n-gpu)]
    (with-generator [g m
                     #:n-ctx        (n-ctx-param)
                     #:n-batch      (n-batch-param)
                     #:cache-type-k (cache-type-k-param)
                     #:cache-type-v (cache-type-v-param)
                     #:greedy       (greedy?)]
      (define result (generate-text g (prompt-text) #:max-tokens (max-tok)))
      (printf "\n--- Generated ~a tokens ---\n" (length (tokenize-text m result #:add-bos #f))))))
