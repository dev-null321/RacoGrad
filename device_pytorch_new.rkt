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

;; pt-fn was used to dynamically load ops from pytorch_backend.rkt
;; With libtorch, all ops are directly available. This is a stub
;; for any code that still references it.
(define (pt-fn name)
  (error 'pt-fn "pt-fn is deprecated in libtorch backend. Use direct function calls. Requested: ~a" name))
