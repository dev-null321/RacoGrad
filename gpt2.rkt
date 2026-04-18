#lang racket

;; ============================================================
;; RacoGrad GPT-2 Implementation
;; Decoder-only transformer in pure Racket
;; ============================================================

(require "device_pytorch.rkt")
(require "nn.rkt")
(require "attention.rkt")
(require "safetensors.rkt")
(require "bpe.rkt")

(provide
 make-gpt2-attention
 make-gpt2-mlp
 make-gpt2-block
 make-gpt2
 make-gpt2-small
 make-gpt2-medium
 gpt2-forward
 gpt2-generate
 gpt2-module)

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

;; Build an nn-module view over an existing gpt2 model-list so that
;; parameter collection (and therefore optimization) works against it.
;;
;; We DON'T change make-gpt2's return type because load-gpt2-weights and
;; gpt2-forward both match-define the list — too many callers to refactor.
;; Instead, `gpt2-module` wraps the submodules in an nn-module whose
;; children are wte, wpe, every block, ln-f, and lm-head.
;;
;; Weight tying note: GPT-2 ties lm_head.weight ← wte.weight. After
;; `load-gpt2-weights` runs, lm-head has the same values as wte but is
;; still a distinct tensor. For training they'll get independent
;; gradient updates; fully tying them would need a shared-param pathway
;; through make-linear, which is a separate piece of work.
(define (gpt2-module model)
  (match-define (list 'gpt2 _config wte wpe blocks ln-f lm-head) model)
  (define children
    (filter nn-module? (append (list wte wpe) blocks (list ln-f lm-head))))
  (nn-module "GPT2"
             (lambda (input-ids) (gpt2-forward model input-ids))
             '()
             children))

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
    (define next-token (argmax (squeeze scaled-logits 1) -1))
    
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

;; Lazily build one BPE tokenizer per model-name by reading the HF
;; tokenizer files from the standard cache. Cached after first call.
(define bpe-cache (make-hash))

(define (resolve-tokenizer-files model-name)
  (define snapshots-dir
    (build-path (find-system-path 'home-dir)
                ".cache" "huggingface" "hub"
                (string-append "models--" model-name)
                "snapshots"))
  (unless (directory-exists? snapshots-dir)
    (error 'gpt2-tokenize
           "no HF cache for ~a at ~a" model-name snapshots-dir))
  (define commits (directory-list snapshots-dir))
  (when (null? commits)
    (error 'gpt2-tokenize "no snapshots under ~a" snapshots-dir))
  (define commit-dir (build-path snapshots-dir (first commits)))
  (values (build-path commit-dir "vocab.json")
          (build-path commit-dir "merges.txt")))

(define (get-bpe model-name)
  (hash-ref! bpe-cache model-name
             (lambda ()
               (define-values (vp mp) (resolve-tokenizer-files model-name))
               (make-bpe vp mp))))

(define (gpt2-tokenize text [model-name "gpt2"])
  ;; Return a [1, seq_len] long tensor on CUDA so it's ready for gpt2-forward.
  (define ids (bpe-encode (get-bpe model-name) text))
  (to-long (to-cuda (tensor (list ids)))))

(define (gpt2-decode ids-tensor [model-name "gpt2"])
  ;; Accept either a libtorch tensor (shape [1, seq]) or a plain list of ints.
  (define ids
    (cond
      [(list? ids-tensor) ids-tensor]
      [else
       (define flat (get-item ids-tensor))
       ;; get-item on a [1, N] tensor returns a list of lists
       (cond
         [(and (list? flat) (pair? flat) (list? (first flat))) (first flat)]
         [(list? flat) flat]
         [else (list flat)])]))
  ;; Filter out any trailing EOS if present
  (bpe-decode (get-bpe model-name) (map inexact->exact ids)))

;; Resolve a HuggingFace model name to its on-disk safetensors path.
;; Uses the standard HF cache layout: ~/.cache/huggingface/hub/
;; models--<org>--<name>/snapshots/<commit>/model.safetensors.
;; `model-name` may also be a direct path to a .safetensors file.
(define (resolve-safetensors-path model-name)
  (cond
    [(and (path-string? model-name)
          (file-exists? model-name))
     model-name]
    [else
     (define repo-dir
       (build-path (find-system-path 'home-dir)
                   ".cache" "huggingface" "hub"
                   (string-append "models--" model-name)))
     (define snapshots-dir (build-path repo-dir "snapshots"))
     (unless (directory-exists? snapshots-dir)
       (error 'load-gpt2-weights
              "no HF cache for ~a at ~a — run `huggingface-cli download ~a model.safetensors` or pass an absolute .safetensors path"
              model-name snapshots-dir model-name))
     (define commits (directory-list snapshots-dir))
     (when (null? commits)
       (error 'load-gpt2-weights "no snapshots under ~a" snapshots-dir))
     (define candidate
       (build-path snapshots-dir (first commits) "model.safetensors"))
     (unless (file-exists? candidate)
       (error 'load-gpt2-weights
              "no model.safetensors at ~a — only tokenizer files are cached"
              candidate))
     candidate]))

(define (load-gpt2-weights model model-name)
  (match-define (list 'gpt2 config wte wpe blocks ln-f lm-head) model)
  (define num-layers (hash-ref config 'num-layers))
  (define path (resolve-safetensors-path model-name))

  (displayln (format "Loading ~a weights from ~a..." model-name path))
  (define s (open-safetensors path))

  ;; Helper: pull a named tensor from safetensors (already on CUDA).
  (define (w name) (safetensors-tensor s name))

  ;; Embeddings
  (set-embedding-weight! wte (w "wte.weight"))
  (set-embedding-weight! wpe (w "wpe.weight"))

  ;; Final layer norm
  (set-layer-norm-weight! ln-f (w "ln_f.weight"))
  (set-layer-norm-bias!   ln-f (w "ln_f.bias"))

  ;; Each block
  (for ([i (in-range num-layers)]
        [block blocks])
    (load-gpt2-block-weights block s i))

  ;; LM head is tied to wte (same weight matrix)
  (set-linear-weight! lm-head (w "wte.weight"))

  (close-safetensors s)
  (displayln "Weights loaded!")
  model)

(define (load-gpt2-block-weights block s layer)
  ;; HuggingFace GPT-2 stores the four Conv1D-style projections
  ;; (c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj) as (in, out) rather
  ;; than PyTorch Linear's (out, in). We transpose on load so the
  ;; downstream Linear module sees weights in its expected layout.
  (define (w name)  (safetensors-tensor s name))
  (define (wt name) (transpose (safetensors-tensor s name) 0 1))
  (define (lname key) (format "h.~a.~a" layer key))

  ;; block structure: submodules = (ln-1 attn ln-2 mlp)
  (define submodules (nn-module-submodules block))
  (define ln-1 (first submodules))
  (define attn (second submodules))
  (define ln-2 (third submodules))
  (define mlp  (fourth submodules))

  ;; Layer norms (1D, no transpose needed)
  (set-layer-norm-weight! ln-1 (w (lname "ln_1.weight")))
  (set-layer-norm-bias!   ln-1 (w (lname "ln_1.bias")))
  (set-layer-norm-weight! ln-2 (w (lname "ln_2.weight")))
  (set-layer-norm-bias!   ln-2 (w (lname "ln_2.bias")))

  ;; Attention (c_attn = fused QKV; c_proj = output projection) — transpose
  (define attn-subs (nn-module-submodules attn))
  (define c-attn (first attn-subs))
  (define c-proj (second attn-subs))
  (set-linear-weight! c-attn (wt (lname "attn.c_attn.weight")))
  (set-linear-bias!   c-attn (w  (lname "attn.c_attn.bias")))
  (set-linear-weight! c-proj (wt (lname "attn.c_proj.weight")))
  (set-linear-bias!   c-proj (w  (lname "attn.c_proj.bias")))

  ;; MLP — also Conv1D-style, transpose both projections
  (define mlp-subs (nn-module-submodules mlp))
  (define c-fc       (first mlp-subs))
  (define c-proj-mlp (second mlp-subs))
  (set-linear-weight! c-fc       (wt (lname "mlp.c_fc.weight")))
  (set-linear-bias!   c-fc       (w  (lname "mlp.c_fc.bias")))
  (set-linear-weight! c-proj-mlp (wt (lname "mlp.c_proj.weight")))
  (set-linear-bias!   c-proj-mlp (w  (lname "mlp.c_proj.bias"))))

;; ============================================================
;; Test
;; ============================================================


;; ============================================================
;; ============================================================
;; ============================================================
;; Text Generation (autoregressive loop in Racket)
;; ============================================================

(provide gpt2-generate-text)

(define (gpt2-generate-text model prompt
                            #:max-tokens [max-tokens 50]
                            #:temperature [temperature 0.8]
                            #:top-k [top-k 40])
  ;; Encode the prompt on the Racket side so we keep a flat list of
  ;; token IDs around for decoding at the end — libtorch has no
  ;; tensor→list FFI yet, and `get-item` only works on scalars.
  (define tok (get-bpe "gpt2"))
  (define prompt-ids (bpe-encode tok prompt))
  (define current-ids (to-long (to-cuda (tensor (list prompt-ids)))))

  (let loop ([i 0] [ids current-ids] [all-ids prompt-ids])
    (cond
      [(>= i max-tokens)
       (bpe-decode tok all-ids)]
      [else
       ;; Forward pass, greedy sample the last position
       (define logits (gpt2-forward model ids))
       (define last-logits
         (slice-dim logits 1 (sub1 (cadr (shape logits))) (cadr (shape logits))))
       (define scaled-logits (div last-logits (tensor temperature)))
       (define next-token-t (argmax (squeeze scaled-logits 1) -1))
       (define next-raw (get-item next-token-t))
       (define next-token
         (inexact->exact
          (if (list? next-raw) (car next-raw) next-raw)))

       (cond
         [(= next-token 50256)    ; <|endoftext|>
          (bpe-decode tok all-ids)]
         [else
          (define next-tensor (to-cuda (to-long (tensor (list next-token)))))
          (define next-2d (unsqueeze next-tensor 0))
          (define new-ids (cat (list ids next-2d) #:dim 1))
          (loop (+ i 1) new-ids (append all-ids (list next-token)))])])))

;; ============================================================
;; Interactive Demo
;; ============================================================

(module+ main
  (displayln "=== RacoGrad GPT-2 Text Generation ===")
  (displayln "Loading model...")

  (define model (make-gpt2-small))
  ;; Discard return value to keep the REPL from dumping the whole struct.
  (void (load-gpt2-weights model "gpt2"))

  (displayln "Model ready!")
  (displayln "")

  (define prompt "The meaning of life is")
  (printf "Prompt: ~a\n" prompt)
  (displayln "Generating...")

  (define t0 (current-inexact-milliseconds))
  (define output (gpt2-generate-text model prompt #:max-tokens 30))
  (define t1 (current-inexact-milliseconds))

  (displayln "")
  (displayln output)
  (define elapsed-s (/ (- t1 t0) 1000.0))
  (printf "~n(generated 30 tokens in ~as = ~a tok/s)~n"
          (real->decimal-string elapsed-s 2)
          (real->decimal-string (/ 30.0 elapsed-s) 2)))
