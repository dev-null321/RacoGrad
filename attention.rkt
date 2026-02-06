#lang racket

;; ============================================================
;; RacoGrad Attention Mechanisms
;; Scaled dot-product and multi-head attention
;; ============================================================

(require "device_pytorch.rkt")
(require "nn.rkt")

(provide
 ;; Core attention
 scaled-dot-product-attention
 
 ;; Multi-head attention module
 make-multi-head-attention
 
 ;; Masks
 make-causal-mask
 
 ;; Helpers
 split-heads
 merge-heads)

;; ============================================================
;; Scaled Dot-Product Attention
;; ============================================================

(define (scaled-dot-product-attention Q K V 
                                       #:mask [mask #f]
                                       #:dropout-p [dropout-p 0.0]
                                       #:training [training #t])
  (define d-k (last (shape Q)))
  (define scale (/ 1.0 (sqrt d-k)))
  
  (define scores (einsum "bhqd,bhkd->bhqk" Q K))
  (define scaled-scores (mul scores (tensor scale)))
  
  (define masked-scores
    (if mask
        (add scaled-scores mask)
        scaled-scores))
  
  (define attn-weights (softmax masked-scores #:dim -1))
  
  (define dropped-weights
    (if (> dropout-p 0.0)
        (dropout attn-weights #:p dropout-p #:training training)
        attn-weights))
  
  (define output (einsum "bhqk,bhkd->bhqd" dropped-weights V))
  
  (values output attn-weights))

;; ============================================================
;; Causal Mask
;; ============================================================

(define (make-causal-mask seq-len #:device [dev #f])
  (causal-mask seq-len #:device dev))

;; ============================================================
;; Head splitting and merging
;; ============================================================

(define (split-heads x num-heads)
  (define shp (shape x))
  (define batch (car shp))
  (define seq-len (cadr shp))
  (define d-model (caddr shp))
  (define d-k (/ d-model num-heads))
  (define reshaped (reshape x (list batch seq-len num-heads d-k)))
  (transpose reshaped 1 2))

(define (merge-heads x)
  (define shp (shape x))
  (define batch (car shp))
  (define num-heads (cadr shp))
  (define seq-len (caddr shp))
  (define d-k (cadddr shp))
  (define d-model (* num-heads d-k))
  (define transposed (transpose x 1 2))
  (reshape transposed (list batch seq-len d-model)))

;; ============================================================
;; Multi-Head Attention Module
;; ============================================================

(define (make-multi-head-attention d-model num-heads
                                    #:dropout [dropout-p 0.0]
                                    #:bias [use-bias #t])
  (unless (= 0 (modulo d-model num-heads))
    (error "d-model must be divisible by num-heads"))
  
  (define d-k (/ d-model num-heads))
  
  (define W-q (make-linear d-model d-model #:bias use-bias))
  (define W-k (make-linear d-model d-model #:bias use-bias))
  (define W-v (make-linear d-model d-model #:bias use-bias))
  (define W-o (make-linear d-model d-model #:bias use-bias))
  
  (define (forward-fn input)
    (define Q-proj (forward W-q input))
    (define K-proj (forward W-k input))
    (define V-proj (forward W-v input))
    
    (define Q (split-heads Q-proj num-heads))
    (define K (split-heads K-proj num-heads))
    (define V (split-heads V-proj num-heads))
    
    (define-values (attn-output attn-weights)
      (scaled-dot-product-attention Q K V 
                                     #:dropout-p dropout-p
                                     #:training #t))
    
    (define merged (merge-heads attn-output))
    (forward W-o merged))
  
  (nn-module "MultiHeadAttention"
             forward-fn
             '()
             (list W-q W-k W-v W-o)))
