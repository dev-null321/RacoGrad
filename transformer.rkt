#lang racket

;; ============================================================
;; RacoGrad Transformer Implementation
;; ============================================================
;;
;; Implements the core transformer architecture from
;; "Attention Is All You Need" (Vaswani et al., 2017)
;;
;; Components:
;;   - Scaled dot-product attention
;;   - Multi-head attention
;;   - Positional encoding (sinusoidal)
;;   - Layer normalization
;;   - Feed-forward network
;;   - Transformer encoder layer
;;   - Transformer encoder stack
;;
;; ============================================================

(require "tensor.rkt")
(require "autograd.rkt")

(provide
 ;; Attention
 scaled-dot-product-attention
 multi-head-attention
 multi-head-attention-forward
 
 ;; Positional encoding
 sinusoidal-positional-encoding
 
 ;; Layer normalization
 layer-norm
 layer-norm-forward
 
 ;; Feed-forward
 feed-forward-network
 feed-forward-forward
 
 ;; Transformer layers
 transformer-encoder-layer
 transformer-encoder-forward
 
 ;; Full encoder
 transformer-encoder
 
 ;; Initialization
 initialize-transformer-encoder
 
 ;; Utilities
 create-causal-mask
 create-padding-mask)


;; ============================================================
;; Scaled Dot-Product Attention
;; ============================================================
;;
;; Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
;;
;; Q: queries  (batch, seq_len, d_k)
;; K: keys     (batch, seq_len, d_k)
;; V: values   (batch, seq_len, d_v)
;; mask: optional attention mask (0 = attend, -inf = ignore)
;;
;; Returns: (batch, seq_len, d_v)

(define (scaled-dot-product-attention Q K V [mask #f])
  (let* ([d_k (last (t:shape K))]
         [scale (sqrt (exact->inexact d_k))]
         ;; QK^T: (batch, seq_q, d_k) × (batch, d_k, seq_k) → (batch, seq_q, seq_k)
         [K-T (t:transpose K)]
         [scores (t:scale (t:mul Q K-T) (/ 1.0 scale))])
    ;; Apply mask if provided (add large negative to masked positions)
    (let* ([masked-scores (if mask
                              (t:add scores mask)
                              scores)]
           ;; Softmax over last dimension (keys)
           [attention-weights (softmax-last-dim masked-scores)]
           ;; Weighted sum of values
           [output (t:mul attention-weights V)])
      (values output attention-weights))))

;; Softmax over the last dimension of a tensor
(define (softmax-last-dim x)
  (let* ([shape (t:shape x)]
         [rank (length shape)])
    (cond
      [(= rank 1)
       ;; Simple 1D softmax
       (let* ([max-val (t:max-val x)]
              [shifted (t:add x (t:scale (t:ones (t:shape x)) (- max-val)))]
              [exp-vals (t:exp shifted)]
              [sum-exp (t:sum exp-vals)])
         (t:scale exp-vals (/ 1.0 sum-exp)))]
      [(= rank 2)
       ;; 2D: softmax each row
       (let* ([rows (car shape)]
              [cols (cadr shape)]
              [data (t:data x)]
              [result (make-vector (* rows cols) 0.0)])
         (for ([i (in-range rows)])
           (let* ([row-start (* i cols)]
                  [row-vals (for/list ([j (in-range cols)])
                              (vector-ref data (+ row-start j)))]
                  [max-val (apply max row-vals)]
                  [shifted (map (lambda (v) (- v max-val)) row-vals)]
                  [exp-vals (map exp shifted)]
                  [sum-exp (apply + exp-vals)]
                  [softmax-vals (map (lambda (v) (/ v sum-exp)) exp-vals)])
             (for ([j (in-range cols)]
                   [v (in-list softmax-vals)])
               (vector-set! result (+ row-start j) v))))
         (t:create shape (vector->list result)))]
      [else
       ;; Higher dimensions: reshape, apply, reshape back
       ;; For now, flatten to 2D and apply row-wise
       (let* ([last-dim (last shape)]
              [other-dims (take shape (- rank 1))]
              [flat-rows (apply * other-dims)]
              [flat-shape (list flat-rows last-dim)]
              [flat-x (t:reshape x flat-shape)]
              [flat-result (softmax-last-dim flat-x)])
         (t:reshape flat-result shape))])))


;; ============================================================
;; Multi-Head Attention
;; ============================================================
;;
;; MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
;; where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
;;
;; Parameters:
;;   W_Q, W_K, W_V: (d_model, d_model) - split into heads internally
;;   W_O: (d_model, d_model) - output projection

(struct multi-head-attention 
  (num-heads d-model d-k d-v W-Q W-K W-V W-O b-Q b-K b-V b-O)
  #:transparent)

(define (initialize-multi-head-attention d-model num-heads)
  (let* ([d-k (quotient d-model num-heads)]
         [d-v (quotient d-model num-heads)]
         ;; Xavier initialization
         [scale-qkv (sqrt (/ 2.0 (+ d-model d-model)))]
         [scale-o (sqrt (/ 2.0 (+ d-model d-model)))]
         ;; Weight matrices
         [W-Q (t:scale (t:random (list d-model d-model) 1.0) scale-qkv)]
         [W-K (t:scale (t:random (list d-model d-model) 1.0) scale-qkv)]
         [W-V (t:scale (t:random (list d-model d-model) 1.0) scale-qkv)]
         [W-O (t:scale (t:random (list d-model d-model) 1.0) scale-o)]
         ;; Biases (zeros)
         [b-Q (t:zeros (list d-model))]
         [b-K (t:zeros (list d-model))]
         [b-V (t:zeros (list d-model))]
         [b-O (t:zeros (list d-model))])
    (multi-head-attention num-heads d-model d-k d-v 
                          W-Q W-K W-V W-O b-Q b-K b-V b-O)))

(define (multi-head-attention-forward mha Q K V [mask #f])
  (let* ([num-heads (multi-head-attention-num-heads mha)]
         [d-model (multi-head-attention-d-model mha)]
         [d-k (multi-head-attention-d-k mha)]
         [W-Q (multi-head-attention-W-Q mha)]
         [W-K (multi-head-attention-W-K mha)]
         [W-V (multi-head-attention-W-V mha)]
         [W-O (multi-head-attention-W-O mha)]
         [b-Q (multi-head-attention-b-Q mha)]
         [b-K (multi-head-attention-b-K mha)]
         [b-V (multi-head-attention-b-V mha)]
         [b-O (multi-head-attention-b-O mha)]
         ;; Project Q, K, V
         [Q-proj (t:add (t:mul Q W-Q) b-Q)]
         [K-proj (t:add (t:mul K W-K) b-K)]
         [V-proj (t:add (t:mul V W-V) b-V)]
         ;; Split into heads and compute attention
         ;; For simplicity, we process heads sequentially and concatenate
         [seq-len (car (t:shape Q))]
         [head-outputs
          (for/list ([h (in-range num-heads)])
            (let* ([start (* h d-k)]
                   [end (* (+ h 1) d-k)]
                   ;; Extract head slices (simplified - assumes 2D input for now)
                   [Q-head (extract-head Q-proj start end)]
                   [K-head (extract-head K-proj start end)]
                   [V-head (extract-head V-proj start end)])
              (let-values ([(out weights) 
                            (scaled-dot-product-attention Q-head K-head V-head mask)])
                out)))]
         ;; Concatenate heads
         [concat-heads (concat-heads-horizontal head-outputs)]
         ;; Output projection
         [output (t:add (t:mul concat-heads W-O) b-O)])
    output))

;; Helper: extract columns [start, end) from a 2D tensor
(define (extract-head tensor start end)
  (let* ([shape (t:shape tensor)]
         [rows (car shape)]
         [cols (cadr shape)]
         [new-cols (- end start)]
         [data (t:data tensor)]
         [result (make-vector (* rows new-cols) 0.0)])
    (for ([i (in-range rows)])
      (for ([j (in-range new-cols)])
        (vector-set! result (+ (* i new-cols) j)
                     (vector-ref data (+ (* i cols) start j)))))
    (t:create (list rows new-cols) (vector->list result))))

;; Helper: concatenate list of 2D tensors horizontally
(define (concat-heads-horizontal tensors)
  (if (= (length tensors) 1)
      (car tensors)
      (let* ([first-t (car tensors)]
             [rows (car (t:shape first-t))]
             [total-cols (apply + (map (lambda (t) (cadr (t:shape t))) tensors))]
             [result (make-vector (* rows total-cols) 0.0)]
             [col-offset 0])
        (for ([tensor (in-list tensors)])
          (let* ([t-cols (cadr (t:shape tensor))]
                 [t-data (t:data tensor)])
            (for ([i (in-range rows)])
              (for ([j (in-range t-cols)])
                (vector-set! result (+ (* i total-cols) col-offset j)
                             (vector-ref t-data (+ (* i t-cols) j)))))
            (set! col-offset (+ col-offset t-cols))))
        (t:create (list rows total-cols) (vector->list result)))))


;; ============================================================
;; Positional Encoding
;; ============================================================
;;
;; PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
;; PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

(define (sinusoidal-positional-encoding max-len d-model)
  (let* ([result (make-vector (* max-len d-model) 0.0)]
         [base 10000.0])
    (for ([pos (in-range max-len)])
      (for ([i (in-range 0 d-model 2)])
        (let* ([angle (/ pos (expt base (/ i d-model)))]
               [sin-val (sin angle)]
               [cos-val (cos angle)])
          (vector-set! result (+ (* pos d-model) i) sin-val)
          (when (< (+ i 1) d-model)
            (vector-set! result (+ (* pos d-model) i 1) cos-val)))))
    (t:create (list max-len d-model) (vector->list result))))


;; ============================================================
;; Layer Normalization
;; ============================================================
;;
;; LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β
;;
;; Applied over the last dimension (features)

(struct layer-norm (gamma beta eps) #:transparent)

(define (initialize-layer-norm d-model [eps 1e-6])
  (layer-norm (t:ones (list d-model))   ; gamma (scale)
              (t:zeros (list d-model))  ; beta (shift)
              eps))

(define (layer-norm-forward ln x)
  (let* ([gamma (layer-norm-gamma ln)]
         [beta (layer-norm-beta ln)]
         [eps (layer-norm-eps ln)]
         [shape (t:shape x)]
         [rank (length shape)]
         [d-model (last shape)])
    (cond
      [(= rank 1)
       ;; 1D: normalize the whole vector
       (let* ([mean-val (t:mean x)]
              [centered (t:add x (t:scale (t:ones shape) (- mean-val)))]
              [var-val (/ (t:sum (t:square centered)) d-model)]
              [std (sqrt (+ var-val eps))]
              [normalized (t:scale centered (/ 1.0 std))])
         (t:add (t:emul normalized gamma) beta))]
      [(= rank 2)
       ;; 2D: normalize each row independently
       (let* ([rows (car shape)]
              [cols (cadr shape)]
              [data (t:data x)]
              [gamma-data (t:data gamma)]
              [beta-data (t:data beta)]
              [result (make-vector (* rows cols) 0.0)])
         (for ([i (in-range rows)])
           (let* ([row-start (* i cols)]
                  [row-vals (for/list ([j (in-range cols)])
                              (vector-ref data (+ row-start j)))]
                  [mean-val (/ (apply + row-vals) cols)]
                  [centered (map (lambda (v) (- v mean-val)) row-vals)]
                  [var-val (/ (apply + (map (lambda (v) (* v v)) centered)) cols)]
                  [std (sqrt (+ var-val eps))]
                  [normalized (map (lambda (v) (/ v std)) centered)])
             (for ([j (in-range cols)]
                   [n (in-list normalized)])
               (let ([g (vector-ref gamma-data j)]
                     [b (vector-ref beta-data j)])
                 (vector-set! result (+ row-start j) (+ (* n g) b))))))
         (t:create shape (vector->list result)))]
      [else
       (error 'layer-norm-forward "Unsupported rank: ~a" rank)])))


;; ============================================================
;; Feed-Forward Network
;; ============================================================
;;
;; FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
;;
;; Typically d_ff = 4 * d_model

(struct feed-forward-network (W1 b1 W2 b2) #:transparent)

(define (initialize-feed-forward d-model d-ff)
  (let* ([scale1 (sqrt (/ 2.0 (+ d-model d-ff)))]
         [scale2 (sqrt (/ 2.0 (+ d-ff d-model)))]
         [W1 (t:scale (t:random (list d-model d-ff) 1.0) scale1)]
         [b1 (t:zeros (list d-ff))]
         [W2 (t:scale (t:random (list d-ff d-model) 1.0) scale2)]
         [b2 (t:zeros (list d-model))])
    (feed-forward-network W1 b1 W2 b2)))

(define (feed-forward-forward ffn x)
  (let* ([W1 (feed-forward-network-W1 ffn)]
         [b1 (feed-forward-network-b1 ffn)]
         [W2 (feed-forward-network-W2 ffn)]
         [b2 (feed-forward-network-b2 ffn)]
         ;; First linear + ReLU
         [hidden (relu (t:add (t:mul x W1) b1))]
         ;; Second linear
         [output (t:add (t:mul hidden W2) b2)])
    output))


;; ============================================================
;; Transformer Encoder Layer
;; ============================================================
;;
;; EncoderLayer(x) = LayerNorm(x + MultiHeadAttention(x, x, x))
;;                 → LayerNorm(y + FFN(y))

(struct transformer-encoder-layer 
  (self-attn ffn norm1 norm2 dropout-rate)
  #:transparent)

(define (initialize-transformer-encoder-layer d-model num-heads d-ff [dropout 0.1])
  (transformer-encoder-layer
   (initialize-multi-head-attention d-model num-heads)
   (initialize-feed-forward d-model d-ff)
   (initialize-layer-norm d-model)
   (initialize-layer-norm d-model)
   dropout))

(define (transformer-encoder-forward layer x [mask #f])
  (let* ([self-attn (transformer-encoder-layer-self-attn layer)]
         [ffn (transformer-encoder-layer-ffn layer)]
         [norm1 (transformer-encoder-layer-norm1 layer)]
         [norm2 (transformer-encoder-layer-norm2 layer)]
         ;; Self-attention with residual + layer norm
         [attn-out (multi-head-attention-forward self-attn x x x mask)]
         [x1 (layer-norm-forward norm1 (t:add x attn-out))]
         ;; Feed-forward with residual + layer norm
         [ffn-out (feed-forward-forward ffn x1)]
         [x2 (layer-norm-forward norm2 (t:add x1 ffn-out))])
    x2))


;; ============================================================
;; Transformer Encoder Stack
;; ============================================================

(struct transformer-encoder 
  (layers pos-encoding d-model max-len)
  #:transparent)

(define (initialize-transformer-encoder 
         num-layers d-model num-heads d-ff max-len [dropout 0.1])
  (let* ([layers (for/list ([_ (in-range num-layers)])
                   (initialize-transformer-encoder-layer d-model num-heads d-ff dropout))]
         [pos-enc (sinusoidal-positional-encoding max-len d-model)])
    (transformer-encoder layers pos-enc d-model max-len)))

(define (transformer-encoder-stack-forward encoder x [mask #f])
  (let* ([layers (transformer-encoder-layers encoder)]
         [pos-enc (transformer-encoder-pos-encoding encoder)]
         ;; Add positional encoding
         [seq-len (car (t:shape x))]
         [pos-slice (t:slice-rows pos-enc 0 seq-len)]
         [x-pos (t:add x pos-slice)])
    ;; Pass through all encoder layers
    (for/fold ([h x-pos])
              ([layer (in-list layers)])
      (transformer-encoder-forward layer h mask))))

;; Helper to slice first n rows of a 2D tensor
(define (t:slice-rows tensor start end)
  (let* ([shape (t:shape tensor)]
         [rows (car shape)]
         [cols (cadr shape)]
         [new-rows (- end start)]
         [data (t:data tensor)]
         [result (make-vector (* new-rows cols) 0.0)])
    (for ([i (in-range new-rows)])
      (for ([j (in-range cols)])
        (vector-set! result (+ (* i cols) j)
                     (vector-ref data (+ (* (+ start i) cols) j)))))
    (t:create (list new-rows cols) (vector->list result))))


;; ============================================================
;; Attention Masks
;; ============================================================

;; Causal mask for autoregressive models (decoder)
;; Upper triangle = -inf, lower triangle + diagonal = 0
(define (create-causal-mask seq-len)
  (let* ([result (make-vector (* seq-len seq-len) 0.0)]
         [neg-inf -1e9])
    (for ([i (in-range seq-len)])
      (for ([j (in-range seq-len)])
        (when (> j i)  ; Future positions
          (vector-set! result (+ (* i seq-len) j) neg-inf))))
    (t:create (list seq-len seq-len) (vector->list result))))

;; Padding mask (1 at padding positions, 0 elsewhere)
;; Input: list of actual lengths for each sequence in batch
(define (create-padding-mask seq-len actual-lengths)
  (let* ([batch-size (length actual-lengths)]
         [result (make-vector (* batch-size seq-len) 0.0)]
         [neg-inf -1e9])
    (for ([b (in-range batch-size)]
          [actual-len (in-list actual-lengths)])
      (for ([j (in-range actual-len seq-len)])
        (vector-set! result (+ (* b seq-len) j) neg-inf)))
    (t:create (list batch-size seq-len) (vector->list result))))


;; ============================================================
;; Transformer Decoder Layer
;; ============================================================
;;
;; DecoderLayer has:
;;   1. Masked self-attention (causal)
;;   2. Cross-attention (attends to encoder output)
;;   3. Feed-forward network

(struct transformer-decoder-layer
  (self-attn cross-attn ffn norm1 norm2 norm3 dropout-rate)
  #:transparent)

(define (initialize-transformer-decoder-layer d-model num-heads d-ff [dropout 0.1])
  (transformer-decoder-layer
   (initialize-multi-head-attention d-model num-heads)  ; self-attention
   (initialize-multi-head-attention d-model num-heads)  ; cross-attention
   (initialize-feed-forward d-model d-ff)
   (initialize-layer-norm d-model)
   (initialize-layer-norm d-model)
   (initialize-layer-norm d-model)
   dropout))

(define (transformer-decoder-forward layer x encoder-output [self-mask #f] [cross-mask #f])
  (let* ([self-attn (transformer-decoder-layer-self-attn layer)]
         [cross-attn (transformer-decoder-layer-cross-attn layer)]
         [ffn (transformer-decoder-layer-ffn layer)]
         [norm1 (transformer-decoder-layer-norm1 layer)]
         [norm2 (transformer-decoder-layer-norm2 layer)]
         [norm3 (transformer-decoder-layer-norm3 layer)]
         ;; Masked self-attention
         [self-attn-out (multi-head-attention-forward self-attn x x x self-mask)]
         [x1 (layer-norm-forward norm1 (t:add x self-attn-out))]
         ;; Cross-attention (query from decoder, key/value from encoder)
         [cross-attn-out (multi-head-attention-forward cross-attn x1 encoder-output encoder-output cross-mask)]
         [x2 (layer-norm-forward norm2 (t:add x1 cross-attn-out))]
         ;; Feed-forward
         [ffn-out (feed-forward-forward ffn x2)]
         [x3 (layer-norm-forward norm3 (t:add x2 ffn-out))])
    x3))

(provide transformer-decoder-layer
         initialize-transformer-decoder-layer
         transformer-decoder-forward)


;; ============================================================
;; Full Transformer (Encoder-Decoder)
;; ============================================================

(struct transformer
  (enc-layers dec-layers 
   src-pos-encoding tgt-pos-encoding
   d-model src-max-len tgt-max-len)
  #:transparent)

(define (initialize-transformer
         num-encoder-layers num-decoder-layers
         d-model num-heads d-ff
         src-max-len tgt-max-len
         [dropout 0.1])
  (let* ([enc-layers (for/list ([_ (in-range num-encoder-layers)])
                       (initialize-transformer-encoder-layer d-model num-heads d-ff dropout))]
         [dec-layers (for/list ([_ (in-range num-decoder-layers)])
                       (initialize-transformer-decoder-layer d-model num-heads d-ff dropout))]
         [src-pe (sinusoidal-positional-encoding src-max-len d-model)]
         [tgt-pe (sinusoidal-positional-encoding tgt-max-len d-model)])
    (transformer enc-layers dec-layers src-pe tgt-pe d-model src-max-len tgt-max-len)))

(define (transformer-forward model src tgt [src-mask #f] [tgt-mask #f])
  (let* ([enc-layers (transformer-enc-layers model)]
         [dec-layers (transformer-dec-layers model)]
         [src-pe (transformer-src-pos-encoding model)]
         [tgt-pe (transformer-tgt-pos-encoding model)]
         ;; Encode source
         [src-len (car (t:shape src))]
         [src-pos (t:slice-rows src-pe 0 src-len)]
         [enc-input (t:add src src-pos)]
         [enc-output (for/fold ([h enc-input])
                               ([layer (in-list enc-layers)])
                       (transformer-encoder-forward layer h src-mask))]
         ;; Decode target
         [tgt-len (car (t:shape tgt))]
         [tgt-pos (t:slice-rows tgt-pe 0 tgt-len)]
         [causal-mask (create-causal-mask tgt-len)]
         [combined-mask (if tgt-mask
                            (t:add causal-mask tgt-mask)
                            causal-mask)]
         [dec-input (t:add tgt tgt-pos)]
         [dec-output (for/fold ([h dec-input])
                               ([layer (in-list dec-layers)])
                       (transformer-decoder-forward layer h enc-output combined-mask src-mask))])
    dec-output))

(provide transformer
         initialize-transformer
         transformer-forward)


;; ============================================================
;; Embedding Layer (for token sequences)
;; ============================================================

(struct embedding (weights vocab-size d-model) #:transparent)

(define (initialize-embedding vocab-size d-model)
  (let ([scale (sqrt (/ 1.0 d-model))]
        [weights (t:scale (t:random (list vocab-size d-model) 1.0) 
                          (sqrt (/ 1.0 d-model)))])
    (embedding weights vocab-size d-model)))

(define (embedding-forward emb token-ids)
  ;; token-ids is a list of integers
  ;; Returns (seq-len, d-model) tensor
  (let* ([weights (embedding-weights emb)]
         [d-model (embedding-d-model emb)]
         [seq-len (length token-ids)]
         [w-data (t:data weights)]
         [result (make-vector (* seq-len d-model) 0.0)])
    (for ([i (in-range seq-len)]
          [token-id (in-list token-ids)])
      (for ([j (in-range d-model)])
        (vector-set! result (+ (* i d-model) j)
                     (vector-ref w-data (+ (* token-id d-model) j)))))
    (t:create (list seq-len d-model) (vector->list result))))

(provide embedding
         initialize-embedding
         embedding-forward)


;; ============================================================
;; Output Projection (logits over vocabulary)
;; ============================================================

(define (output-projection hidden embed-weights)
  ;; hidden: (seq-len, d-model)
  ;; embed-weights: (vocab-size, d-model)
  ;; output: (seq-len, vocab-size) = hidden @ embed_weights^T
  (t:mul hidden (t:transpose embed-weights)))

(provide output-projection)


;; ============================================================
;; Test / Demo
;; ============================================================

(module+ main
  (displayln "=== RacoGrad Transformer Test ===\n")
  
  ;; Test positional encoding
  (displayln "1. Positional Encoding (4 positions, 8 dims):")
  (define pe (sinusoidal-positional-encoding 4 8))
  (t:print pe)
  (newline)
  
  ;; Test layer norm
  (displayln "2. Layer Normalization:")
  (define ln (initialize-layer-norm 4))
  (define x-ln (t:create '(2 4) '(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)))
  (displayln "Input:")
  (t:print x-ln)
  (displayln "After LayerNorm:")
  (t:print (layer-norm-forward ln x-ln))
  (newline)
  
  ;; Test feed-forward
  (displayln "3. Feed-Forward Network:")
  (define ffn (initialize-feed-forward 4 16))
  (define x-ffn (t:create '(2 4) '(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)))
  (displayln "Input shape: (2, 4)")
  (displayln "Output shape:")
  (displayln (t:shape (feed-forward-forward ffn x-ffn)))
  (newline)
  
  ;; Test multi-head attention
  (displayln "4. Multi-Head Attention (4 dims, 2 heads):")
  (define mha (initialize-multi-head-attention 4 2))
  (define x-mha (t:create '(3 4) '(0.1 0.2 0.3 0.4 
                                    0.5 0.6 0.7 0.8 
                                    0.9 1.0 1.1 1.2)))
  (displayln "Input shape: (3, 4)")
  (define mha-out (multi-head-attention-forward mha x-mha x-mha x-mha))
  (displayln "Output shape:")
  (displayln (t:shape mha-out))
  (newline)
  
  ;; Test full encoder layer
  (displayln "5. Transformer Encoder Layer:")
  (define enc-layer (initialize-transformer-encoder-layer 8 2 32))
  (define x-enc (t:random '(4 8) 1.0))
  (displayln "Input shape: (4, 8)")
  (define enc-out (transformer-encoder-forward enc-layer x-enc))
  (displayln "Output shape:")
  (displayln (t:shape enc-out))
  (newline)
  
  ;; Test causal mask
  (displayln "6. Causal Mask (4x4):")
  (t:print (create-causal-mask 4))
  (newline)
  
  ;; Test full encoder stack
  (displayln "7. Transformer Encoder Stack (2 layers, 16 dims, 4 heads):")
  (define encoder (initialize-transformer-encoder 2 16 4 64 100))
  (define x-full (t:random '(10 16) 1.0))
  (displayln "Input shape: (10, 16)")
  (define full-out (transformer-encoder-stack-forward encoder x-full))
  (displayln "Output shape:")
  (displayln (t:shape full-out))
  (newline)
  
  ;; Test decoder layer
  (displayln "8. Transformer Decoder Layer:")
  (define dec-layer (initialize-transformer-decoder-layer 16 4 64))
  (define tgt-input (t:random '(5 16) 1.0))
  (define enc-output (t:random '(8 16) 1.0))
  (displayln "Target input shape: (5, 16)")
  (displayln "Encoder output shape: (8, 16)")
  (define dec-out (transformer-decoder-forward dec-layer tgt-input enc-output))
  (displayln "Decoder output shape:")
  (displayln (t:shape dec-out))
  (newline)
  
  ;; Test full encoder-decoder transformer
  (displayln "9. Full Transformer (Encoder-Decoder):")
  (define full-transformer (initialize-transformer 2 2 32 4 128 50 50))
  (define src-seq (t:random '(8 32) 1.0))
  (define tgt-seq (t:random '(5 32) 1.0))
  (displayln "Source shape: (8, 32)")
  (displayln "Target shape: (5, 32)")
  (define trans-out (transformer-forward full-transformer src-seq tgt-seq))
  (displayln "Output shape:")
  (displayln (t:shape trans-out))
  (newline)
  
  ;; Test embedding layer
  (displayln "10. Embedding Layer:")
  (define emb (initialize-embedding 1000 32))  ; vocab=1000, d_model=32
  (define token-ids '(5 23 42 100 7))
  (displayln "Token IDs: (5 23 42 100 7)")
  (define emb-out (embedding-forward emb token-ids))
  (displayln "Embedding output shape:")
  (displayln (t:shape emb-out))
  (newline)
  
  ;; Test output projection
  (displayln "11. Output Projection (logits):")
  (define hidden (t:random '(5 32) 1.0))
  (define logits (output-projection hidden (embedding-weights emb)))
  (displayln "Hidden shape: (5, 32)")
  (displayln "Logits shape (seq_len, vocab_size):")
  (displayln (t:shape logits))
  
  (displayln "\n=== All tests passed! ==="))
