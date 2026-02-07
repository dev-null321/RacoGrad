#lang racket

;; ============================================================
;; Transformer Training Test
;; Verify forward/loss works (backward needs requires_grad)
;; ============================================================

(require "device_pytorch.rkt")
(require "nn.rkt")  
(require "transformer.rkt")

(displayln "=== RacoGrad Training Test ===")
(displayln "")

;; 1. Create model
(displayln "1. Creating transformer (vocab=32, d=32, 1 layer)...")
(define model (make-transformer 32 32
                                #:d-model 32
                                #:num-heads 2
                                #:num-layers 1
                                #:d-ff 64
                                #:max-len 16))
(displayln "   Done")

;; 2. Create dummy data
(displayln "2. Creating dummy data (batch=2, seq=8)...")
(define src (reshape (arange 0 16) (list 2 8)))
(define tgt (reshape (arange 0 16) (list 2 8)))
(displayln (format "   src shape: ~a" (shape src)))
(displayln (format "   tgt shape: ~a" (shape tgt)))

;; 3. Forward pass
(displayln "3. Forward pass...")
(define logits (model src tgt))
(displayln (format "   logits shape: ~a" (shape logits)))

;; 4. Compute loss
(displayln "4. Computing loss...")
(define loss (cross-entropy-loss logits tgt))
(define loss-val (get-item loss))
(displayln (format "   loss: ~a" loss-val))

(displayln "")
(displayln "=== Forward pass and loss work! ===")
(displayln "")
(displayln "Note: Backward pass requires requires_grad=True on model params.")
(displayln "This is a parameter registration issue to fix in nn.rkt")
