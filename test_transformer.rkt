#lang racket

;; ============================================================
;; Transformer Integration Test
;; ============================================================

(require "device_pytorch.rkt")
(require "nn.rkt")
(require "transformer.rkt")

(displayln "=== RacoGrad Transformer Test ===")
(displayln "")

;; Test 1: Create a small transformer
(displayln "1. Creating transformer (vocab=100, d_model=64, 2 layers, 4 heads)...")
(define model (make-transformer 100 100
                                #:d-model 64
                                #:num-heads 4
                                #:num-layers 2
                                #:d-ff 256
                                #:max-len 32))
(displayln "   ✓ Model created")

;; Test 2: Create dummy input
(displayln "2. Creating dummy input (batch=2, seq_len=8)...")
(define src-tokens (arange 0 16))  ; 2*8 tokens
(define src-input (reshape src-tokens (list 2 8)))
(define tgt-tokens (arange 0 16))
(define tgt-input (reshape tgt-tokens (list 2 8)))
(displayln (format "   src shape: ~a" (shape src-input)))
(displayln (format "   tgt shape: ~a" (shape tgt-input)))
(displayln "   ✓ Inputs created")

;; Test 3: Forward pass
(displayln "3. Running forward pass...")
(define logits (model src-input tgt-input))
(displayln (format "   logits shape: ~a" (shape logits)))
(displayln "   ✓ Forward pass complete")

;; Test 4: Compute loss
(displayln "4. Computing cross-entropy loss...")
(define targets tgt-input)  ; Using input as targets for test
(define loss (cross-entropy-loss logits targets))
(displayln (format "   loss: ~a" loss))
(displayln "   ✓ Loss computed")

(displayln "")
(displayln "=== All tests passed! ===")
