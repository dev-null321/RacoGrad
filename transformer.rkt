#lang racket

;; ============================================================
;; RacoGrad Transformer
;; Full encoder-decoder transformer with PyTorch backend
;; ============================================================

(require "device_pytorch.rkt")
(require "nn.rkt")
(require "attention.rkt")

(provide
 make-sinusoidal-positional-encoding
 make-positional-encoding
 make-feed-forward
 make-transformer-encoder-layer
 make-transformer-decoder-layer
 make-transformer-encoder
 make-transformer-decoder
 make-transformer)

;; ============================================================
;; Sinusoidal Positional Encoding (Vaswani et al.)
;; PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
;; PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
;; ============================================================

(define (make-sinusoidal-positional-encoding max-len d-model)
  ;; Pre-compute positional encodings using PyTorch ops
  ;; Computed once at construction, stored in closure
  
  (define half-d (quotient d-model 2))
  
  ;; Position indices: (max-len, 1) - cast to float for matmul
  (define pos (to-float (unsqueeze (arange 0 max-len) 1)))
  
  ;; Frequency divisors: exp(arange(0, d_model, 2) * -log(10000)/d_model)
  ;; Shape: (1, d_model/2)
  (define indices (to-float (arange 0 d-model #:step 2)))
  (define log-term (mul (tensor -1.0) (div (t:log (tensor 10000.0)) (tensor (exact->inexact d-model)))))
  (define div-term (unsqueeze (t:exp (mul indices log-term)) 0))
  
  ;; Angles: pos * div_term -> (max-len, d_model/2)
  (define angles (matmul pos div-term))
  
  ;; Apply sin to even dims, cos to odd dims
  (define sin-vals (t:sin angles))  ;; (max-len, d_model/2)
  (define cos-vals (t:cos angles))  ;; (max-len, d_model/2)
  
  ;; Interleave using stack and reshape:
  ;; stack([sin, cos], dim=2) -> (max-len, d/2, 2)
  ;; reshape to (max-len, d_model)
  (define stacked (stack (list sin-vals cos-vals) #:dim 2))
  (define pe (reshape stacked (list max-len d-model)))
  
  ;; Return module that adds pe to input
  (nn-module "SinusoidalPositionalEncoding"
             (lambda (x)
               ;; x: (batch, seq_len, d_model)
               ;; pe: (max-len, d_model) - slice to seq_len
               (define seq-len (cadr (shape x)))
               (define pe-sliced (slice-dim pe 0 0 seq-len))
               ;; Broadcasting handles batch dim automatically
               (add x pe-sliced))
             '()
             '()))

;; ============================================================
;; Learned Positional Encoding (fallback)
;; ============================================================

(define (make-positional-encoding max-len d-model)
  (define emb (make-embedding max-len d-model))
  (nn-module "PositionalEncoding"
             (lambda (x)
               (define seq-len (cadr (shape x)))
               (define positions (arange 0 seq-len))
               (add x (forward emb positions)))
             '()
             (list emb)))

;; ============================================================
;; Feed-Forward Network
;; ============================================================

(define (make-feed-forward d-model d-ff #:dropout [dropout-p 0.1])
  (define linear1 (make-linear d-model d-ff))
  (define linear2 (make-linear d-ff d-model))
  
  (nn-module "FeedForward"
             (lambda (x)
               (forward linear2 (gelu (forward linear1 x))))
             '() 
             (list linear1 linear2)))

;; ============================================================
;; Transformer Encoder Layer (Pre-LN variant)
;; ============================================================

(define (make-transformer-encoder-layer d-model num-heads d-ff 
                                         #:dropout [dropout-p 0.1])
  (define self-attn (make-multi-head-attention d-model num-heads #:dropout dropout-p))
  (define ffn (make-feed-forward d-model d-ff #:dropout dropout-p))
  (define norm1 (make-layer-norm d-model))
  (define norm2 (make-layer-norm d-model))
  
  (nn-module "TransformerEncoderLayer"
             (lambda (x)
               ;; Pre-LN: norm before attention/FFN
               (define x1 (add x (forward self-attn (forward norm1 x))))
               (add x1 (forward ffn (forward norm2 x1))))
             '()
             (list self-attn ffn norm1 norm2)))

;; ============================================================
;; Transformer Decoder Layer (Pre-LN variant)
;; ============================================================

(define (make-transformer-decoder-layer d-model num-heads d-ff
                                         #:dropout [dropout-p 0.1])
  (define self-attn (make-multi-head-attention d-model num-heads #:dropout dropout-p))
  (define cross-attn (make-multi-head-attention d-model num-heads #:dropout dropout-p))
  (define ffn (make-feed-forward d-model d-ff #:dropout dropout-p))
  (define norm1 (make-layer-norm d-model))
  (define norm2 (make-layer-norm d-model))
  (define norm3 (make-layer-norm d-model))
  
  ;; Returns a function that takes (x, encoder-output)
  (lambda (x encoder-output)
    (define x1 (add x (forward self-attn (forward norm1 x))))
    (define x2 (add x1 (forward cross-attn (forward norm2 x1))))
    (add x2 (forward ffn (forward norm3 x2)))))

;; ============================================================
;; Transformer Encoder
;; ============================================================

(define (make-transformer-encoder num-layers d-model num-heads d-ff
                                   #:max-len [max-len 512]
                                   #:dropout [dropout-p 0.1]
                                   #:sinusoidal [sinusoidal #t])
  (define pos-enc (if sinusoidal
                      (make-sinusoidal-positional-encoding max-len d-model)
                      (make-positional-encoding max-len d-model)))
  (define layers (for/list ([_ (in-range num-layers)])
                   (make-transformer-encoder-layer d-model num-heads d-ff 
                                                    #:dropout dropout-p)))
  (define final-norm (make-layer-norm d-model))
  
  (nn-module "TransformerEncoder"
             (lambda (x)
               (define x-pos (forward pos-enc x))
               (define encoded (for/fold ([h x-pos]) ([layer layers])
                                 (forward layer h)))
               (forward final-norm encoded))
             '() 
             (cons pos-enc (append layers (list final-norm)))))

;; ============================================================
;; Transformer Decoder
;; ============================================================

(define (make-transformer-decoder num-layers d-model num-heads d-ff
                                   #:max-len [max-len 512]
                                   #:dropout [dropout-p 0.1]
                                   #:sinusoidal [sinusoidal #t])
  (define pos-enc (if sinusoidal
                      (make-sinusoidal-positional-encoding max-len d-model)
                      (make-positional-encoding max-len d-model)))
  (define layer-fns (for/list ([_ (in-range num-layers)])
                      (make-transformer-decoder-layer d-model num-heads d-ff
                                                       #:dropout dropout-p)))
  (define final-norm (make-layer-norm d-model))
  
  ;; Decoder returns a function taking (x, encoder-output)
  (lambda (x encoder-output)
    (define x-pos (forward pos-enc x))
    (define decoded (for/fold ([h x-pos]) ([layer-fn layer-fns])
                      (layer-fn h encoder-output)))
    (forward final-norm decoded)))

;; ============================================================
;; Full Transformer
;; ============================================================

(define (make-transformer src-vocab tgt-vocab
                          #:d-model [d-model 256]
                          #:num-heads [num-heads 8]
                          #:num-layers [num-layers 4]
                          #:d-ff [d-ff 1024]
                          #:max-len [max-len 256]
                          #:dropout [dropout-p 0.1]
                          #:sinusoidal [sinusoidal #t])
  
  (define src-embed (make-embedding src-vocab d-model))
  (define tgt-embed (make-embedding tgt-vocab d-model))
  (define encoder (make-transformer-encoder num-layers d-model num-heads d-ff
                                             #:max-len max-len 
                                             #:dropout dropout-p
                                             #:sinusoidal sinusoidal))
  (define decoder-fn (make-transformer-decoder num-layers d-model num-heads d-ff
                                                #:max-len max-len 
                                                #:dropout dropout-p
                                                #:sinusoidal sinusoidal))
  (define output-proj (make-linear d-model tgt-vocab))
  
  ;; Returns logits: (batch, seq_len, vocab_size)
  (lambda (src-tokens tgt-tokens)
    (define src-emb (forward src-embed src-tokens))
    (define tgt-emb (forward tgt-embed tgt-tokens))
    (define enc-out (forward encoder src-emb))
    (define dec-out (decoder-fn tgt-emb enc-out))
    (forward output-proj dec-out)))

;; ============================================================
;; Test
;; ============================================================

(module+ main
  (displayln "=== RacoGrad Transformer ===")
  (displayln "Sinusoidal positional encoding: enabled")
  (displayln "Use (make-transformer src-vocab tgt-vocab) to create a model."))
