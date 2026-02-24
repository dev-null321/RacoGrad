#lang racket

;; ============================================================
;; RacoGrad Training Utilities — libtorch backend (zero Python)
;; ============================================================

(require "device_pytorch.rkt")  ;; → libtorch_backend.rkt
(require "nn.rkt")

(provide
 ;; Optimizer (wraps libtorch Adam)
 make-adam-from-params
 optimizer-step
 optimizer-zero-grad
 
 ;; Training
 train-step
 
 ;; Utilities
 get-loss-value)

;; ============================================================
;; Optimizer
;; ============================================================

(define (make-adam-from-params param-structs #:lr [lr 0.001] #:weight-decay [wd 0.0])
  ;; param-structs: list of (param name tensor) structs
  ;; Extract raw tensor handles for libtorch Adam
  (define handles (map param-tensor param-structs))
  (make-adam handles #:lr lr #:weight-decay wd))

(define (optimizer-step opt)
  (adam-step opt))

(define (optimizer-zero-grad opt)
  (adam-zero-grad opt))

;; ============================================================
;; Loss / Training
;; ============================================================

(define (get-loss-value loss)
  (get-item loss))

(define (train-step model optimizer loss-fn src tgt)
  (optimizer-zero-grad optimizer)
  (define logits (model src tgt))
  (define loss (loss-fn logits tgt))
  (backward loss)
  (optimizer-step optimizer)
  (get-loss-value loss))
