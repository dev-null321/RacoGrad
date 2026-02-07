#lang racket

;; ============================================================
;; PyTorch Backend for RacoGrad
;; Provides CUDA acceleration via pyffi + PyTorch
;; ============================================================

(require pyffi)

(provide 
 ;; Initialization
 pytorch-init!
 pytorch-available?
 pytorch-cuda-available?
 pytorch-device-name
 
 ;; Tensor creation
 pt:zeros
 pt:ones
 pt:randn
 pt:tensor
 pt:arange
 
 ;; Tensor properties
 pt:shape
 pt:dtype
 pt:device
 pt:to
 
 ;; Basic ops
 pt:add
 pt:sub
 pt:mul
 pt:div
 pt:matmul
 pt:transpose
 pt:reshape
 pt:squeeze
 pt:unsqueeze
 
 ;; Math ops
 pt:exp
 pt:log
 pt:sqrt
 pt:abs
 pt:sum
 pt:mean
 pt:max
 pt:min
 
 ;; Activation functions
 pt:relu
 pt:gelu
 pt:sigmoid
 pt:tanh
 pt:softmax
 
 ;; Linear algebra
 pt:mm        ; matrix multiply
 pt:bmm       ; batched matrix multiply
 pt:einsum
 
 ;; Neural network ops
 pt:linear
 pt:layer-norm
 pt:dropout
 pt:embedding
 
 ;; Utility
 pt:sync!
 pt:to-list
 pt:print)

;; ============================================================
;; Initialization - done at module load time
;; ============================================================

(initialize)
(post-initialize)

;; Import torch at module level
(import torch)

;; Define Python helper functions
(run* "
import torch
import torch.nn.functional as F

def _get_shape(t):
    return list(t.shape)

def _get_dtype(t):
    return str(t.dtype)

def _get_device(t):
    return str(t.device)

def _to_nested_list(t):
    '''Recursively convert tensor to nested Python list'''
    return t.cpu().tolist()

def _softmax(t, dim):
    return torch.softmax(t, dim=dim)

def _gelu(t):
    return F.gelu(t)

def _linear(x, w, b=None):
    return F.linear(x, w, b)

def _layer_norm(x, shape, eps=1e-5):
    return F.layer_norm(x, shape, eps=eps)

def _dropout(t, p, training):
    return F.dropout(t, p=p, training=training)

def _embedding(weight, indices):
    return F.embedding(indices, weight)
")

;; Get Python helper functions as Racket values
(define py-get-shape (run "_get_shape"))
(define py-get-dtype (run "_get_dtype"))
(define py-get-device (run "_get_device"))
(define py-to-list (run "_to_nested_list"))
(define py-softmax (run "_softmax"))
(define py-gelu (run "_gelu"))
(define py-linear (run "_linear"))
(define py-layer-norm (run "_layer_norm"))
(define py-dropout (run "_dropout"))
(define py-embedding (run "_embedding"))

;; ============================================================
;; Helper: Convert nested Python lists to Racket lists  
;; ============================================================

(define (deep-pylist->list obj)
  (cond
    [(pylist? obj) 
     (map deep-pylist->list (pylist->list obj))]
    [(number? obj) obj]
    [(string? obj) obj]
    [(boolean? obj) obj]
    [else obj]))

;; ============================================================
;; Initialization
;; ============================================================

(define (pytorch-init!)
  #t)

(define (pytorch-available?)
  #t)

(define (pytorch-cuda-available?)
  (torch.cuda.is_available))

(define (pytorch-device-name)
  (if (pytorch-cuda-available?)
      (torch.cuda.get_device_name 0)
      "cpu"))

;; ============================================================
;; Tensor Creation
;; ============================================================

(define (pt:zeros shape #:device [device "cuda"] #:dtype [dtype #f])
  (if dtype
      (torch.zeros shape #:device device #:dtype (run (format "torch.~a" dtype)))
      (torch.zeros shape #:device device)))

(define (pt:ones shape #:device [device "cuda"] #:dtype [dtype #f])
  (if dtype
      (torch.ones shape #:device device #:dtype (run (format "torch.~a" dtype)))
      (torch.ones shape #:device device)))

(define (pt:randn shape #:device [device "cuda"] #:dtype [dtype #f])
  (if dtype
      (torch.randn shape #:device device #:dtype (run (format "torch.~a" dtype)))
      (torch.randn shape #:device device)))

(define (pt:tensor data #:device [device "cuda"] #:dtype [dtype #f])
  (if dtype
      (torch.tensor data #:device device #:dtype (run (format "torch.~a" dtype)))
      (torch.tensor data #:device device)))

(define (pt:arange start end #:step [step 1] #:device [device "cuda"])
  (torch.arange start end step #:device device))

;; ============================================================
;; Tensor Properties
;; ============================================================

(define (pt:shape t)
  (pylist->list (py-get-shape t)))

(define (pt:dtype t)
  (py-get-dtype t))

(define (pt:device t)
  ;; Convert Python string to Racket string
  (define result (py-get-device t))
  (if (string? result) result (format "~a" result)))

(define (pt:device-old t)
  (py-get-device t))

(define (pt:to t device)
  (t .to device))

;; ============================================================
;; Basic Operations
;; ============================================================

(define (pt:add a b)
  (torch.add a b))

(define (pt:sub a b)
  (torch.sub a b))

(define (pt:mul a b)
  (torch.mul a b))

(define (pt:div a b)
  (torch.div a b))

(define (pt:matmul a b)
  (torch.matmul a b))

(define (pt:transpose t dim0 dim1)
  (torch.transpose t dim0 dim1))

(define (pt:reshape t shape)
  (t .reshape shape))

(define (pt:squeeze t #:dim [dim #f])
  (if dim
      (t .squeeze dim)
      (t .squeeze)))

(define (pt:unsqueeze t dim)
  (t .unsqueeze dim))

;; ============================================================
;; Math Operations
;; ============================================================

(define (pt:exp t)
  (torch.exp t))

(define (pt:log t)
  (torch.log t))

(define (pt:sqrt t)
  (torch.sqrt t))

(define (pt:abs t)
  (torch.abs t))

(define (pt:sum t #:dim [dim #f] #:keepdim [keepdim #f])
  (cond
    [(and dim keepdim) (torch.sum t #:dim dim #:keepdim keepdim)]
    [dim (torch.sum t #:dim dim)]
    [else (torch.sum t)]))

(define (pt:mean t #:dim [dim #f] #:keepdim [keepdim #f])
  (cond
    [(and dim keepdim) (torch.mean t #:dim dim #:keepdim keepdim)]
    [dim (torch.mean t #:dim dim)]
    [else (torch.mean t)]))

(define (pt:max t #:dim [dim #f])
  (if dim
      (torch.max t #:dim dim)
      (torch.max t)))

(define (pt:min t #:dim [dim #f])
  (if dim
      (torch.min t #:dim dim)
      (torch.min t)))

;; ============================================================
;; Activation Functions
;; ============================================================

(define (pt:relu t)
  (torch.relu t))

(define (pt:gelu t)
  (py-gelu t))

(define (pt:sigmoid t)
  (torch.sigmoid t))

(define (pt:tanh t)
  (torch.tanh t))

(define (pt:softmax t #:dim [dim -1])
  (py-softmax t dim))

;; ============================================================
;; Linear Algebra
;; ============================================================

(define (pt:mm a b)
  (torch.mm a b))

(define (pt:bmm a b)
  (torch.bmm a b))

(define (pt:einsum equation . tensors)
  (apply torch.einsum equation tensors))

;; ============================================================
;; Neural Network Operations
;; ============================================================

(define (pt:linear input weight #:bias [bias #f])
  (if bias
      (py-linear input weight bias)
      (py-linear input weight (run "None"))))

(define (pt:layer-norm input normalized-shape #:eps [eps 1e-5])
  (py-layer-norm input normalized-shape eps))

(define (pt:dropout t #:p [p 0.1] #:training [training #t])
  (py-dropout t p training))

(define (pt:embedding weight indices)
  (py-embedding weight indices))

;; ============================================================
;; Utility
;; ============================================================

(define (pt:sync!)
  (torch.cuda.synchronize))

(define (pt:to-list t)
  (deep-pylist->list (py-to-list t)))

(define (pt:print t #:name [name "tensor"])
  (printf "~a: shape=~a device=~a\n" name (pt:shape t) (pt:device t))
  (displayln t))

;; Print initialization info
(printf "PyTorch backend loaded\n")
(printf "  CUDA available: ~a\n" (pytorch-cuda-available?))
(when (pytorch-cuda-available?)
  (printf "  Device: ~a\n" (pytorch-device-name)))

;; ============================================================
;; Additional ops for attention
;; ============================================================

(run* "
def _triu(t, diagonal=1):
    return torch.triu(t, diagonal=diagonal)

def _tril(t, diagonal=0):
    return torch.tril(t, diagonal=diagonal)

def _full(shape, fill_value, device=\"cuda\"):
    return torch.full(shape, fill_value, device=device)

def _masked_fill(t, mask, value):
    return t.masked_fill(mask, value)

def _causal_mask(seq_len, device=\"cuda\"):
    # Returns (seq_len, seq_len) mask with -inf above diagonal
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float(\"-inf\"))
")

(define py-triu (run "_triu"))
(define py-tril (run "_tril"))
(define py-full (run "_full"))
(define py-masked-fill (run "_masked_fill"))
(define py-causal-mask (run "_causal_mask"))

(provide pt:triu pt:tril pt:full pt:masked-fill pt:causal-mask)

(define (pt:triu t #:diagonal [diagonal 1])
  (py-triu t diagonal))

(define (pt:tril t #:diagonal [diagonal 0])
  (py-tril t diagonal))

(define (pt:full shape fill-value #:device [device "cuda"])
  (py-full shape fill-value device))

(define (pt:masked-fill t mask value)
  (py-masked-fill t mask value))

(define (pt:causal-mask seq-len #:device [device "cuda"])
  (py-causal-mask seq-len device))

;; ============================================================
;; Loss Functions
;; ============================================================

(provide pt:cross-entropy
         pt:mse-loss
         pt:nll-loss)

(define (pt:cross-entropy logits targets #:ignore-index [ignore-idx -100])
  ;; Reshape for cross_entropy: (N, C) logits, (N,) targets
  ;; If 3D (batch, seq, vocab), flatten first two dims
  (define shp (pt:shape logits))
  (cond
    [(= (length shp) 3)
     (define batch (car shp))
     (define seq (cadr shp))
     (define vocab (caddr shp))
     (define flat-logits (torch.reshape logits (list (* batch seq) vocab)))
     (define flat-targets (torch.reshape targets (list (* batch seq))))
     (torch.nn.functional.cross_entropy flat-logits flat-targets 
                                         #:ignore_index ignore-idx)]
    [else
     (torch.nn.functional.cross_entropy logits targets 
                                         #:ignore_index ignore-idx)]))

(define (pt:mse-loss pred target)
  (torch.nn.functional.mse_loss pred target))

(define (pt:nll-loss log-probs targets #:ignore-index [ignore-idx -100])
  (torch.nn.functional.nll_loss log-probs targets #:ignore_index ignore-idx))

;; ============================================================
;; Training Utilities (backward, optimizer)
;; ============================================================

(run* "
def _backward(t):
    t.backward()
    return None

def _item(t):
    return float(t.item())

def _make_adam(params, lr=0.001):
    return torch.optim.Adam(params, lr=lr)

def _opt_step(opt):
    opt.step()
    return None

def _opt_zero(opt):
    opt.zero_grad()
    return None
")

(define py-backward (run "_backward"))
(define py-item (run "_item"))  
(define py-make-adam (run "_make_adam"))
(define py-opt-step (run "_opt_step"))
(define py-opt-zero (run "_opt_zero"))

(provide pt:backward pt:item pt:make-adam pt:opt-step pt:opt-zero)

(define (pt:backward loss) (py-backward loss))
(define (pt:item tensor) (py-item tensor))
(define (pt:make-adam params #:lr [lr 0.001]) (py-make-adam params lr))
(define (pt:opt-step opt) (py-opt-step opt))
(define (pt:opt-zero opt) (py-opt-zero opt))

;; ============================================================
;; Trigonometric Functions (for sinusoidal positional encoding)
;; ============================================================

(provide pt:sin pt:cos)

(define (pt:sin t)
  (torch.sin t))

(define (pt:cos t)
  (torch.cos t))

;; ============================================================
;; Tensor Concatenation
;; ============================================================

(provide pt:cat pt:stack)

(define (pt:cat tensors #:dim [dim 0])
  (torch.cat tensors #:dim dim))

(define (pt:stack tensors #:dim [dim 0])
  (torch.stack tensors #:dim dim))

;; ============================================================
;; Tensor Type Casting
;; ============================================================

(provide pt:float pt:long pt:int)

(define (pt:float t)
  (t .float))

(define (pt:long t)
  (t .long))

(define (pt:int t)
  (t .int))

;; ============================================================
;; Tensor Slicing
;; ============================================================

(run* "
def _slice(t, dim, start, end):
    slices = [slice(None)] * t.dim()
    slices[dim] = slice(start, end)
    return t[tuple(slices)]
")

(define py-slice (run "_slice"))

(provide pt:slice)

(define (pt:slice t dim start end)
  (py-slice t dim start end))

;; ============================================================
;; Training with Python-side Model Wrapper
;; For proper gradient tracking, we wrap the Racket model call in Python
;; ============================================================

(run* "
import torch
import torch.nn as nn
import torch.nn.functional as F

class RacketModelWrapper(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, dim_ff=256, max_len=64):
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = self._create_sinusoidal_pe(max_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, batch_first=True)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def _create_sinusoidal_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, src, tgt):
        src_emb = self.src_embed(src) + self.pos_enc[:, :src.size(1)]
        tgt_emb = self.tgt_embed(tgt) + self.pos_enc[:, :tgt.size(1)]
        
        memory = self.encoder(src_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.output_proj(output)

def create_transformer(vocab_size, d_model=64, nhead=4, num_layers=2, dim_ff=256, max_len=64, device=\"cuda\"):
    model = RacketModelWrapper(vocab_size, d_model, nhead, num_layers, dim_ff, max_len)
    return model.to(device)

def train_copy_task(vocab_size=16, seq_len=10, d_model=64, nhead=4, num_layers=2,
                    epochs=20, batches=50, batch_size=32, lr=0.001, device=\"cuda\"):
    model = create_transformer(vocab_size, d_model, nhead, num_layers, d_model*4, seq_len*2, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f\"Training copy task: vocab={vocab_size}, seq_len={seq_len}, d_model={d_model}\")
    print(f\"Model params: {sum(p.numel() for p in model.parameters()):,}\")
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for _ in range(batches):
            # Generate copy data
            src = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
            tgt = src.clone()
            
            # Forward
            logits = model(src, tgt)
            loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            correct += (preds == tgt).sum().item()
            total += tgt.numel()
        
        acc = 100 * correct / total
        print(f\"Epoch {epoch+1}/{epochs} | Loss: {total_loss/batches:.4f} | Acc: {acc:.1f}%\")
    
    # Final test
    print(\"\\nFinal test:\")
    src = torch.randint(1, vocab_size, (1, seq_len), device=device)
    with torch.no_grad():
        logits = model(src, src)
        pred = logits.argmax(dim=-1)
    print(f\"Input:  {src[0].tolist()}\")
    print(f\"Output: {pred[0].tolist()}\")
    print(f\"Match:  {(src == pred).all().item()}\")
    
    return model
")

(define py-create-transformer (run "create_transformer"))
(define py-train-copy-task (run "train_copy_task"))

(provide pt:create-transformer pt:train-copy-task)

(define (pt:create-transformer vocab-size 
                               #:d-model [d-model 64]
                               #:nhead [nhead 4]
                               #:num-layers [num-layers 2]
                               #:dim-ff [dim-ff 256]
                               #:max-len [max-len 64]
                               #:device [device "cuda"])
  (py-create-transformer vocab-size d-model nhead num-layers dim-ff max-len device))

(define (pt:train-copy-task #:vocab-size [vocab-size 16]
                            #:seq-len [seq-len 10]
                            #:d-model [d-model 64]
                            #:nhead [nhead 4]
                            #:num-layers [num-layers 2]
                            #:epochs [epochs 20]
                            #:batches [batches 50]
                            #:batch-size [batch-size 32]
                            #:lr [lr 0.001]
                            #:device [device "cuda"])
  (py-train-copy-task vocab-size seq-len d-model nhead num-layers epochs batches batch-size lr device))


;; ============================================================
;; GPT-2 Weight Loading from HuggingFace
;; ============================================================

(run* "
from transformers import GPT2LMHeadModel
import torch

_gpt2_cache = {}

def _load_gpt2_weights(model_name):
    if model_name not in _gpt2_cache:
        print(f\"Loading {model_name} from HuggingFace...\")
        _gpt2_cache[model_name] = GPT2LMHeadModel.from_pretrained(model_name)
        print(\"Loaded!\")
    return _gpt2_cache[model_name]

def _get_gpt2_wte(model_name):
    m = _load_gpt2_weights(model_name)
    return m.transformer.wte.weight.detach()

def _get_gpt2_wpe(model_name):
    m = _load_gpt2_weights(model_name)
    return m.transformer.wpe.weight.detach()

def _get_gpt2_ln_f_weight(model_name):
    m = _load_gpt2_weights(model_name)
    return m.transformer.ln_f.weight.detach()

def _get_gpt2_ln_f_bias(model_name):
    m = _load_gpt2_weights(model_name)
    return m.transformer.ln_f.bias.detach()

def _get_gpt2_block_ln1_weight(model_name, layer):
    m = _load_gpt2_weights(model_name)
    return m.transformer.h[layer].ln_1.weight.detach()

def _get_gpt2_block_ln1_bias(model_name, layer):
    m = _load_gpt2_weights(model_name)
    return m.transformer.h[layer].ln_1.bias.detach()

def _get_gpt2_block_ln2_weight(model_name, layer):
    m = _load_gpt2_weights(model_name)
    return m.transformer.h[layer].ln_2.weight.detach()

def _get_gpt2_block_ln2_bias(model_name, layer):
    m = _load_gpt2_weights(model_name)
    return m.transformer.h[layer].ln_2.bias.detach()

def _get_gpt2_block_attn_c_attn_weight(model_name, layer):
    m = _load_gpt2_weights(model_name)
    return m.transformer.h[layer].attn.c_attn.weight.t().detach()

def _get_gpt2_block_attn_c_attn_bias(model_name, layer):
    m = _load_gpt2_weights(model_name)
    return m.transformer.h[layer].attn.c_attn.bias.detach()

def _get_gpt2_block_attn_c_proj_weight(model_name, layer):
    m = _load_gpt2_weights(model_name)
    return m.transformer.h[layer].attn.c_proj.weight.t().detach()

def _get_gpt2_block_attn_c_proj_bias(model_name, layer):
    m = _load_gpt2_weights(model_name)
    return m.transformer.h[layer].attn.c_proj.bias.detach()

def _get_gpt2_block_mlp_c_fc_weight(model_name, layer):
    m = _load_gpt2_weights(model_name)
    return m.transformer.h[layer].mlp.c_fc.weight.t().detach()

def _get_gpt2_block_mlp_c_fc_bias(model_name, layer):
    m = _load_gpt2_weights(model_name)
    return m.transformer.h[layer].mlp.c_fc.bias.detach()

def _get_gpt2_block_mlp_c_proj_weight(model_name, layer):
    m = _load_gpt2_weights(model_name)
    return m.transformer.h[layer].mlp.c_proj.weight.t().detach()

def _get_gpt2_block_mlp_c_proj_bias(model_name, layer):
    m = _load_gpt2_weights(model_name)
    return m.transformer.h[layer].mlp.c_proj.bias.detach()

def _gpt2_tokenize(text, model_name=\"gpt2\"):
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained(model_name)
    return torch.tensor([tok.encode(text)]).cuda()

def _gpt2_decode(ids, model_name=\"gpt2\"):
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained(model_name)
    return tok.decode(ids.tolist() if hasattr(ids, \"tolist\") else ids)
")

(provide pt:gpt2-wte pt:gpt2-wpe pt:gpt2-ln-f-weight pt:gpt2-ln-f-bias
         pt:gpt2-block-ln1-weight pt:gpt2-block-ln1-bias
         pt:gpt2-block-ln2-weight pt:gpt2-block-ln2-bias
         pt:gpt2-block-attn-c-attn-weight pt:gpt2-block-attn-c-attn-bias
         pt:gpt2-block-attn-c-proj-weight pt:gpt2-block-attn-c-proj-bias
         pt:gpt2-block-mlp-c-fc-weight pt:gpt2-block-mlp-c-fc-bias
         pt:gpt2-block-mlp-c-proj-weight pt:gpt2-block-mlp-c-proj-bias
         pt:gpt2-tokenize pt:gpt2-decode)

(define (pt:gpt2-wte model-name) ((run "_get_gpt2_wte") model-name))
(define (pt:gpt2-wpe model-name) ((run "_get_gpt2_wpe") model-name))
(define (pt:gpt2-ln-f-weight model-name) ((run "_get_gpt2_ln_f_weight") model-name))
(define (pt:gpt2-ln-f-bias model-name) ((run "_get_gpt2_ln_f_bias") model-name))

(define (pt:gpt2-block-ln1-weight model-name layer) ((run "_get_gpt2_block_ln1_weight") model-name layer))
(define (pt:gpt2-block-ln1-bias model-name layer) ((run "_get_gpt2_block_ln1_bias") model-name layer))
(define (pt:gpt2-block-ln2-weight model-name layer) ((run "_get_gpt2_block_ln2_weight") model-name layer))
(define (pt:gpt2-block-ln2-bias model-name layer) ((run "_get_gpt2_block_ln2_bias") model-name layer))

(define (pt:gpt2-block-attn-c-attn-weight model-name layer) 
  ((run "_get_gpt2_block_attn_c_attn_weight") model-name layer))
(define (pt:gpt2-block-attn-c-attn-bias model-name layer)
  ((run "_get_gpt2_block_attn_c_attn_bias") model-name layer))
(define (pt:gpt2-block-attn-c-proj-weight model-name layer)
  ((run "_get_gpt2_block_attn_c_proj_weight") model-name layer))
(define (pt:gpt2-block-attn-c-proj-bias model-name layer)
  ((run "_get_gpt2_block_attn_c_proj_bias") model-name layer))

(define (pt:gpt2-block-mlp-c-fc-weight model-name layer)
  ((run "_get_gpt2_block_mlp_c_fc_weight") model-name layer))
(define (pt:gpt2-block-mlp-c-fc-bias model-name layer)
  ((run "_get_gpt2_block_mlp_c_fc_bias") model-name layer))
(define (pt:gpt2-block-mlp-c-proj-weight model-name layer)
  ((run "_get_gpt2_block_mlp_c_proj_weight") model-name layer))
(define (pt:gpt2-block-mlp-c-proj-bias model-name layer)
  ((run "_get_gpt2_block_mlp_c_proj_bias") model-name layer))

(define (pt:gpt2-tokenize text [model-name "gpt2"]) 
  ((run "_gpt2_tokenize") text model-name))
(define (pt:gpt2-decode ids [model-name "gpt2"])
  ((run "_gpt2_decode") ids model-name))

;; ============================================================
;; Tensor Copy (for weight loading)
;; ============================================================

(run* "
def _copy_tensor_data(dst, src):
    with torch.no_grad():
        dst.copy_(src)
    return dst
")

(provide pt:copy-tensor!)

(define (pt:copy-tensor! dst src)
  ((run "_copy_tensor_data") dst src))

;; ============================================================
;; GPT-2 Text Generation
;; ============================================================

(run* "
def _gpt2_generate(model_forward_fn, input_ids, max_new_tokens=50, temperature=0.8, top_k=40):
    import torch
    import torch.nn.functional as F
    
    current_ids = input_ids
    
    for _ in range(max_new_tokens):
        # Forward pass through Racket model
        logits = model_forward_fn(current_ids)
        
        # Get logits for last token
        next_logits = logits[:, -1, :] / temperature
        
        # Top-k sampling
        if top_k is not None:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = float(-inf)
        
        # Sample
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append
        current_ids = torch.cat([current_ids, next_token], dim=1)
    
    return current_ids

def _sample_token(logits, temperature=0.8, top_k=40):
    import torch
    import torch.nn.functional as F
    logits = logits[:, -1, :] / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_value = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_value, torch.full_like(logits, float(\"-inf\")), logits)
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, 1)
    return next_token.item()
")

(provide pt:gpt2-generate pt:sample-token)

(define (pt:gpt2-generate forward-fn input-ids 
                          #:max-tokens [max-tokens 50]
                          #:temperature [temperature 0.8]
                          #:top-k [top-k 40])
  ((run "_gpt2_generate") forward-fn input-ids max-tokens temperature top-k))

(define (pt:sample-token logits #:temperature [temperature 0.8] #:top-k [top-k 40])
  ((run "_sample_token") logits temperature top-k))
