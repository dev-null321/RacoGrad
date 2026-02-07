#lang racket

;; ============================================================
;; RacoGrad GPT-2 Implementation
;; Decoder-only transformer in pure Racket
;; ============================================================

(require "device_pytorch.rkt")
(require "nn.rkt")
(require "attention.rkt")

(provide
 make-gpt2-attention
 make-gpt2-mlp
 make-gpt2-block
 make-gpt2
 gpt2-forward
 gpt2-generate)

;; ============================================================
;; GPT-2 Attention (Causal Self-Attention)
;; ============================================================

(define (make-gpt2-attention d-model num-heads #:dropout [dropout-p 0.1])
  ;; Combined QKV projection for efficiency
  (define c-attn (make-linear d-model (* 3 d-model)))
  (define c-proj (make-linear d-model d-model))
  (define head-dim (quotient d-model num-heads))
  
  (nn-module "GPT2Attention"
             (lambda (x)
               (define shp (shape x))
               (define batch (car shp))
               (define seq-len (cadr shp))
               
               ;; QKV projection
               (define qkv (forward c-attn x))
               
               ;; Split into Q, K, V
               (define q (slice-dim qkv 2 0 d-model))
               (define k (slice-dim qkv 2 d-model (* 2 d-model)))
               (define v (slice-dim qkv 2 (* 2 d-model) (* 3 d-model)))
               
               ;; Reshape for multi-head: (batch, seq, heads, head_dim)
               (define q-heads (reshape q (list batch seq-len num-heads head-dim)))
               (define k-heads (reshape k (list batch seq-len num-heads head-dim)))
               (define v-heads (reshape v (list batch seq-len num-heads head-dim)))
               
               ;; Transpose to (batch, heads, seq, head_dim)
               (define q-t (transpose q-heads 1 2))
               (define k-t (transpose k-heads 1 2))
               (define v-t (transpose v-heads 1 2))
               
               ;; Attention scores: Q @ K^T / sqrt(d_k)
               (define scale (tensor (/ 1.0 (sqrt (exact->inexact head-dim)))))
               (define scores (mul (matmul q-t (transpose k-t 2 3)) scale))
               
               ;; Causal mask
               (define mask (causal-mask seq-len))
               (define masked-scores (add scores mask))
               
               ;; Softmax and apply to values
               (define attn-weights (softmax masked-scores #:dim -1))
               (define attn-out (matmul attn-weights v-t))
               
               ;; Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, d_model)
               (define attn-transposed (transpose attn-out 1 2))
               (define attn-concat (reshape attn-transposed (list batch seq-len d-model)))
               
               ;; Output projection
               (forward c-proj attn-concat))
             '()
             (list c-attn c-proj)))

;; ============================================================
;; GPT-2 MLP (Feed-Forward with GELU)
;; ============================================================

(define (make-gpt2-mlp d-model d-ff #:dropout [dropout-p 0.1])
  (define c-fc (make-linear d-model d-ff))
  (define c-proj (make-linear d-ff d-model))
  
  (nn-module "GPT2MLP"
             (lambda (x)
               (forward c-proj (gelu (forward c-fc x))))
             '()
             (list c-fc c-proj)))

;; ============================================================
;; GPT-2 Block (Pre-LN Transformer Block)
;; ============================================================

(define (make-gpt2-block d-model num-heads #:dropout [dropout-p 0.1])
  (define d-ff (* 4 d-model))  ;; GPT-2 uses 4x expansion
  
  (define ln-1 (make-layer-norm d-model))
  (define attn (make-gpt2-attention d-model num-heads #:dropout dropout-p))
  (define ln-2 (make-layer-norm d-model))
  (define mlp (make-gpt2-mlp d-model d-ff #:dropout dropout-p))
  
  (nn-module "GPT2Block"
             (lambda (x)
               ;; Pre-LN: norm before attention/MLP
               (define x1 (add x (forward attn (forward ln-1 x))))
               (add x1 (forward mlp (forward ln-2 x1))))
             '()
             (list ln-1 attn ln-2 mlp)))

;; ============================================================
;; GPT-2 Model
;; ============================================================

(define (make-gpt2 vocab-size
                   #:d-model [d-model 768]
                   #:num-heads [num-heads 12]
                   #:num-layers [num-layers 12]
                   #:max-len [max-len 1024]
                   #:dropout [dropout-p 0.1])
  
  ;; Token and position embeddings
  (define wte (make-embedding vocab-size d-model))  ;; Token embeddings
  (define wpe (make-embedding max-len d-model))     ;; Position embeddings
  
  ;; Transformer blocks
  (define blocks 
    (for/list ([_ (in-range num-layers)])
      (make-gpt2-block d-model num-heads #:dropout dropout-p)))
  
  ;; Final layer norm
  (define ln-f (make-layer-norm d-model))
  
  ;; Output projection (weight-tied with wte in real GPT-2)
  (define lm-head (make-linear d-model vocab-size))
  
  ;; Store config
  (define config (hash 'vocab-size vocab-size
                       'd-model d-model
                       'num-heads num-heads
                       'num-layers num-layers
                       'max-len max-len))
  
  (list 'gpt2 config wte wpe blocks ln-f lm-head))

;; ============================================================
;; GPT-2 Forward Pass
;; ============================================================

(define (gpt2-forward model input-ids)
  (match-define (list 'gpt2 config wte wpe blocks ln-f lm-head) model)
  
  (define shp (shape input-ids))
  (define batch (car shp))
  (define seq-len (cadr shp))
  
  ;; Position indices
  (define positions (arange 0 seq-len))
  
  ;; Embeddings
  (define tok-emb (forward wte input-ids))
  (define pos-emb (forward wpe positions))
  (define x (add tok-emb pos-emb))
  
  ;; Transformer blocks
  (define hidden
    (for/fold ([h x]) ([block blocks])
      (forward block h)))
  
  ;; Final norm and projection
  (define normed (forward ln-f hidden))
  (forward lm-head normed))

;; ============================================================
;; GPT-2 Text Generation (Greedy)
;; ============================================================

(define (gpt2-generate model input-ids max-new-tokens 
                       #:temperature [temperature 1.0])
  (define current-ids input-ids)
  
  (for ([_ (in-range max-new-tokens)])
    ;; Forward pass
    (define logits (gpt2-forward model current-ids))
    
    ;; Get logits for last position
    (define last-logits (slice-dim logits 1 
                                   (sub1 (cadr (shape logits)))
                                   (cadr (shape logits))))
    
    ;; Apply temperature
    (define scaled-logits (div last-logits (tensor temperature)))
    
    ;; Greedy: argmax
    (define next-token (t:max (squeeze scaled-logits #:dim 1) #:dim -1))
    
    ;; Append to sequence (simplified - just track logits for now)
    ;; Full implementation would concatenate tokens
    (void))
  
  current-ids)

;; ============================================================
;; GPT-2 Configurations
;; ============================================================

(define (make-gpt2-small)
  (make-gpt2 50257 #:d-model 768 #:num-heads 12 #:num-layers 12))

(define (make-gpt2-medium)
  (make-gpt2 50257 #:d-model 1024 #:num-heads 16 #:num-layers 24))

(define (make-gpt2-large)
  (make-gpt2 50257 #:d-model 1280 #:num-heads 20 #:num-layers 36))

(define (make-gpt2-xl)
  (make-gpt2 50257 #:d-model 1600 #:num-heads 25 #:num-layers 48))

;; ============================================================
;; Load Pretrained Weights
;; ============================================================

(provide load-gpt2-weights gpt2-tokenize gpt2-decode)

(define (gpt2-tokenize text [model-name "gpt2"])
  ((pt-fn 'pt:gpt2-tokenize) text model-name))

(define (gpt2-decode ids [model-name "gpt2"])
  ((pt-fn 'pt:gpt2-decode) ids model-name))

(define (load-gpt2-weights model model-name)
  (match-define (list 'gpt2 config wte wpe blocks ln-f lm-head) model)
  (define num-layers (hash-ref config 'num-layers))
  
  (displayln (format "Loading ~a weights..." model-name))
  
  ;; Load embeddings
  (set-embedding-weight! wte ((pt-fn 'pt:gpt2-wte) model-name))
  (set-embedding-weight! wpe ((pt-fn 'pt:gpt2-wpe) model-name))
  
  ;; Load final layer norm
  (set-layer-norm-weight! ln-f ((pt-fn 'pt:gpt2-ln-f-weight) model-name))
  (set-layer-norm-bias! ln-f ((pt-fn 'pt:gpt2-ln-f-bias) model-name))
  
  ;; Load each block
  (for ([i (in-range num-layers)]
        [block blocks])
    (load-gpt2-block-weights block model-name i))
  
  ;; lm_head shares weights with wte (weight tying)
  (set-linear-weight! lm-head ((pt-fn 'pt:gpt2-wte) model-name))
  
  (displayln "Weights loaded!")
  model)

(define (load-gpt2-block-weights block model-name layer)
  ;; block structure: (nn-module "GPT2Block" forward-fn params submodules)
  ;; submodules: (ln-1 attn ln-2 mlp)
  (define submodules (nn-module-submodules block))
  (define ln-1 (first submodules))
  (define attn (second submodules))
  (define ln-2 (third submodules))
  (define mlp (fourth submodules))
  
  ;; Load layer norms
  (set-layer-norm-weight! ln-1 ((pt-fn 'pt:gpt2-block-ln1-weight) model-name layer))
  (set-layer-norm-bias! ln-1 ((pt-fn 'pt:gpt2-block-ln1-bias) model-name layer))
  (set-layer-norm-weight! ln-2 ((pt-fn 'pt:gpt2-block-ln2-weight) model-name layer))
  (set-layer-norm-bias! ln-2 ((pt-fn 'pt:gpt2-block-ln2-bias) model-name layer))
  
  ;; Load attention
  (define attn-subs (nn-module-submodules attn))
  (define c-attn (first attn-subs))
  (define c-proj (second attn-subs))
  (set-linear-weight! c-attn ((pt-fn 'pt:gpt2-block-attn-c-attn-weight) model-name layer))
  (set-linear-bias! c-attn ((pt-fn 'pt:gpt2-block-attn-c-attn-bias) model-name layer))
  (set-linear-weight! c-proj ((pt-fn 'pt:gpt2-block-attn-c-proj-weight) model-name layer))
  (set-linear-bias! c-proj ((pt-fn 'pt:gpt2-block-attn-c-proj-bias) model-name layer))
  
  ;; Load MLP
  (define mlp-subs (nn-module-submodules mlp))
  (define c-fc (first mlp-subs))
  (define c-proj-mlp (second mlp-subs))
  (set-linear-weight! c-fc ((pt-fn 'pt:gpt2-block-mlp-c-fc-weight) model-name layer))
  (set-linear-bias! c-fc ((pt-fn 'pt:gpt2-block-mlp-c-fc-bias) model-name layer))
  (set-linear-weight! c-proj-mlp ((pt-fn 'pt:gpt2-block-mlp-c-proj-weight) model-name layer))
  (set-linear-bias! c-proj-mlp ((pt-fn 'pt:gpt2-block-mlp-c-proj-bias) model-name layer)))

;; ============================================================
;; Test
;; ============================================================

(module+ main
  (displayln "=== RacoGrad GPT-2 ===")
  (displayln "Creating GPT-2 small (124M params)...")
  
  (define model (make-gpt2-small))
  (displayln "Model created!")
  
  ;; Load pretrained weights
  (load-gpt2-weights model "gpt2")
  
  ;; Test with real text
  (define prompt "The meaning of life is")
  (displayln (format "Prompt: ~a" prompt))
  
  (define input-ids (gpt2-tokenize prompt))
  (displayln (format "Tokens: ~a" (to-list input-ids)))
  
  (define logits (gpt2-forward model input-ids))
  (printf "Output shape: ~a\n" (shape logits))
  
  ;; Get prediction for next token
  (define last-logits (slice-dim logits 1 (sub1 (cadr (shape logits))) (cadr (shape logits))))
  (displayln "GPT-2 with pretrained weights works!"))
