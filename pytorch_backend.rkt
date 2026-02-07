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
