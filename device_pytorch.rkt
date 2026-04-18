#lang racket

;; ============================================================
;; Compatibility shim: re-exports libtorch_backend.rkt
;; with the same API as the old device_pytorch.rkt
;;
;; Modules that (require "device_pytorch.rkt") will get the
;; libtorch backend transparently.
;; ============================================================

(require "libtorch_backend.rkt")

;; Re-export everything from libtorch_backend
(provide (all-from-out "libtorch_backend.rkt"))

;; Additional compat names that old code might use
(provide
 device-synchronize
 pt-fn)

(define (device-synchronize) (rg-sync))

;; `pt-fn` resolves a named function from pytorch_backend.rkt via
;; dynamic-require. This is the same mechanism `nn.rkt` uses for
;; `copy-weight!`, and it's how the weight loaders (gpt2.rkt's
;; `pt:gpt2-wte`, etc.) are reached without a hard module dependency.
;;
;; Tensor ops themselves run through the libtorch backend; pytorch_backend
;; is only touched for pretrained-weight fetching (transformers via pyffi)
;; and tensor↔tensor copy. So Python is in the "load weights once" path,
;; not in the forward / generation path.
(define (pt-fn name)
  (dynamic-require "pytorch_backend.rkt" name))
