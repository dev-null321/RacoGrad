#lang racket

;; ============================================================
;; RacoGrad llama.cpp Backend — GGUF Inference
;;
;; Racket FFI → libracograd_llama_ffi.so → llama.cpp → CUDA
;;
;; Follows the same patterns as libtorch_backend.rkt
;; ============================================================

(require ffi/unsafe
         ffi/unsafe/define)

(provide
 ;; Backend lifecycle
 llama-backend-init
 llama-backend-free

 ;; Model
 llama-model-load
 llama-model-free
 llama-n-vocab
 llama-model-desc
 llama-model-n-layer
 llama-model-n-embd
 llama-bos-token
 llama-eos-token
 llama-is-eog?

 ;; Context
 llama-context-create
 llama-context-free

 ;; Tokenization
 llama-tokenize
 llama-detokenize
 llama-detokenize-token

 ;; Decode
 llama-decode

 ;; Logits
 llama-get-logits
 llama-get-logits-ith

 ;; Sampling
 llama-sampler-create
 llama-sampler-create-greedy
 llama-sampler-sample
 llama-sampler-free

 ;; Perf
 llama-perf-context-print
 llama-perf-sampler-print)

;; ============================================================
;; Load the shared library
;; ============================================================

(define lib-path
  (let ([paths (list
                "ffi/libracograd_llama_ffi.so"
                "./libracograd_llama_ffi.so"
                (build-path (current-directory) "ffi" "libracograd_llama_ffi.so")
                (expand-user-path "~/Projects/RacoGrad/ffi/libracograd_llama_ffi.so"))])
    (for/first ([p paths] #:when (file-exists? p)) p)))

(define-ffi-definer define-llama (ffi-lib (or lib-path "ffi/libracograd_llama_ffi")))

;; ============================================================
;; Type aliases
;; ============================================================

(define _handle _int64)
(define _llama-token _int32)

;; ============================================================
;; Backend lifecycle
;; ============================================================

(define-llama rg_llama_backend_init (_fun -> _void))
(define-llama rg_llama_backend_free (_fun -> _void))

(define (llama-backend-init) (rg_llama_backend_init))
(define (llama-backend-free) (rg_llama_backend_free))

;; ============================================================
;; Model
;; ============================================================

(define-llama rg_llama_model_load (_fun _string _int32 -> _handle))
(define-llama rg_llama_model_free (_fun _handle -> _void))
(define-llama rg_llama_n_vocab (_fun _handle -> _int32))
(define-llama rg_llama_model_desc (_fun _handle _pointer _int32 -> _int32))
(define-llama rg_llama_model_n_layer (_fun _handle -> _int32))
(define-llama rg_llama_model_n_embd (_fun _handle -> _int32))
(define-llama rg_llama_bos_token (_fun _handle -> _int32))
(define-llama rg_llama_eos_token (_fun _handle -> _int32))
(define-llama rg_llama_is_eog (_fun _handle _int32 -> _int32))

(define (llama-model-load path #:n-gpu-layers [ngl 99])
  (define h (rg_llama_model_load path ngl))
  (when (= h -1)
    (error 'llama-model-load "failed to load model: ~a" path))
  h)

(define (llama-model-free h) (rg_llama_model_free h))

(define (llama-n-vocab model-h) (rg_llama_n_vocab model-h))

(define (llama-model-desc model-h)
  (define buf (malloc 256 'atomic))
  (define n (rg_llama_model_desc model-h buf 256))
  (if (> n 0)
      (cast buf _pointer _string)
      "unknown"))

(define (llama-model-n-layer model-h) (rg_llama_model_n_layer model-h))
(define (llama-model-n-embd model-h) (rg_llama_model_n_embd model-h))

(define (llama-bos-token model-h) (rg_llama_bos_token model-h))
(define (llama-eos-token model-h) (rg_llama_eos_token model-h))
(define (llama-is-eog? model-h token) (= 1 (rg_llama_is_eog model-h token)))

;; ============================================================
;; Context
;; ============================================================

(define-llama rg_llama_ctx_create
  (_fun _handle _uint32 _uint32 _int32 _int32 -> _handle))
(define-llama rg_llama_ctx_free (_fun _handle -> _void))

;; GGML_TYPE_* integer constants (from llama.cpp/ggml/include/ggml.h).
;; Kept as a small local table rather than a full mirror of the enum —
;; only the values we actually accept as KV cache types appear here.
(define ggml-type-f16  1)
(define ggml-type-q4_0 2)
(define ggml-type-q8_0 8)

;; Normalize user-facing cache-type symbols to GGML_TYPE_* ints.
;; Accepts: 'f16, 'q8_0, 'q4_0, plus friendly aliases 'q8 and 'q4.
(define (cache-type-symbol->int sym who)
  (case sym
    [(f16)       ggml-type-f16]
    [(q8_0 q8)   ggml-type-q8_0]
    [(q4_0 q4)   ggml-type-q4_0]
    [else
     (error who
            "unsupported cache type: ~a (expected one of: f16 q8_0 q4_0; aliases q8 q4)"
            sym)]))

(define (llama-context-create model-h
                              #:n-ctx        [n-ctx 2048]
                              #:n-batch      [n-batch 512]
                              #:cache-type-k [cache-type-k 'f16]
                              #:cache-type-v [cache-type-v 'f16])
  (define k-int (cache-type-symbol->int cache-type-k 'llama-context-create))
  (define v-int (cache-type-symbol->int cache-type-v 'llama-context-create))
  (define h (rg_llama_ctx_create model-h n-ctx n-batch k-int v-int))
  (when (= h -1)
    (error 'llama-context-create "failed to create context"))
  h)

(define (llama-context-free h) (rg_llama_ctx_free h))

;; ============================================================
;; Tokenization
;; ============================================================

(define-llama rg_llama_tokenize (_fun _handle _string _int32 _pointer _int32 _int32 -> _int32))
(define-llama rg_llama_detokenize (_fun _handle _int32 _pointer _int32 -> _int32))

(define (llama-tokenize model-h text #:add-bos [add-bos #t])
  (define text-len (string-length text))
  ;; First pass: get required token count
  (define n-est (- (rg_llama_tokenize model-h text text-len #f 0 (if add-bos 1 0))))
  (define max-tokens (max n-est 128))
  ;; Allocate and tokenize
  (define buf (malloc _int32 max-tokens))
  (define n (rg_llama_tokenize model-h text text-len buf max-tokens (if add-bos 1 0)))
  (when (< n 0)
    (error 'llama-tokenize "tokenization failed (overflow)"))
  ;; Convert to Racket list
  (for/list ([i (in-range n)])
    (ptr-ref buf _int32 i)))

(define (llama-detokenize-token model-h token)
  (define buf (malloc 128 'atomic))
  (define n (rg_llama_detokenize model-h token buf 128))
  (cond
    [(> n 0)
     ;; Racket CS does not support `make-sized-byte-string` aliasing a
     ;; raw pointer; copy the bytes into a fresh bytes instead.
     (define bs (make-bytes n))
     (memcpy bs buf n)
     bs]
    [else #""]))

(define (llama-detokenize model-h tokens)
  (apply bytes-append
         (for/list ([tok tokens])
           (llama-detokenize-token model-h tok))))

;; ============================================================
;; Decode
;; ============================================================

(define-llama rg_llama_decode (_fun _handle _pointer _int32 -> _int32))

(define (llama-decode ctx-h tokens)
  (define n (length tokens))
  (define buf (malloc _int32 n))
  (for ([tok tokens] [i (in-naturals)])
    (ptr-set! buf _int32 i tok))
  (define status (rg_llama_decode ctx-h buf n))
  (when (< status 0)
    (error 'llama-decode "decode failed with status ~a" status))
  status)

;; ============================================================
;; Logits
;; ============================================================

(define-llama rg_llama_get_logits (_fun _handle -> _pointer))
(define-llama rg_llama_get_logits_ith (_fun _handle _int32 -> _pointer))

(define (llama-get-logits ctx-h) (rg_llama_get_logits ctx-h))
(define (llama-get-logits-ith ctx-h i) (rg_llama_get_logits_ith ctx-h i))

;; ============================================================
;; Sampling
;; ============================================================

(define-llama rg_llama_sampler_create (_fun _uint32 _int32 _float _float -> _handle))
(define-llama rg_llama_sampler_create_greedy (_fun -> _handle))
(define-llama rg_llama_sampler_sample (_fun _handle _handle _int32 -> _int32))
(define-llama rg_llama_sampler_free (_fun _handle -> _void))

(define (llama-sampler-create #:seed [seed #xFFFFFFFF]
                              #:top-k [top-k 40]
                              #:top-p [top-p 0.9]
                              #:temp [temp 0.8])
  (rg_llama_sampler_create seed top-k top-p temp))

(define (llama-sampler-create-greedy)
  (rg_llama_sampler_create_greedy))

(define (llama-sampler-sample sampler-h ctx-h #:idx [idx -1])
  (rg_llama_sampler_sample sampler-h ctx-h idx))

(define (llama-sampler-free h) (rg_llama_sampler_free h))

;; ============================================================
;; Performance
;; ============================================================

(define-llama rg_llama_perf_context_print (_fun _handle -> _void))
(define-llama rg_llama_perf_sampler_print (_fun _handle -> _void))

(define (llama-perf-context-print ctx-h) (rg_llama_perf_context_print ctx-h))
(define (llama-perf-sampler-print sampler-h) (rg_llama_perf_sampler_print sampler-h))

;; ============================================================
;; Auto-init on load
;; ============================================================

(llama-backend-init)
(printf "RacoGrad llama.cpp backend loaded\n")
