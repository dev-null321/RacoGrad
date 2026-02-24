#lang racket

;; ============================================================
;; RacoGrad Training Utilities
;; ============================================================

(require "device_pytorch.rkt")
(require "nn.rkt")
(require (only-in pyffi run* run))

(provide
 make-adam-from-params
 optimizer-step
 optimizer-zero-grad
 train-step
 get-loss-value)

;; ============================================================
;; Python helper for creating optimizer from tensor list
;; ============================================================

(run* "
def _make_adam_from_list(tensor_list, lr=0.001, wd=0.0):
    import torch
    params = list(tensor_list)
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)

def _opt_step(opt):
    opt.step()

def _opt_zero(opt):
    opt.zero_grad()
")

(define py-make-adam-from-list (run "_make_adam_from_list"))
(define py-opt-step (run "_opt_step"))
(define py-opt-zero (run "_opt_zero"))

;; ============================================================
;; Optimizer API
;; ============================================================

(define (make-adam-from-params param-structs #:lr [lr 0.001] #:weight-decay [wd 0.0])
  ;; param-structs is a list of (param name tensor) structs
  ;; Extract the raw tensors for PyTorch
  (define tensors (map param-tensor param-structs))
  (py-make-adam-from-list tensors lr wd))

(define (optimizer-step opt)
  (py-opt-step opt))

(define (optimizer-zero-grad opt)
  (py-opt-zero opt))

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
