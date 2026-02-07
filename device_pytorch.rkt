#lang racket

;; ============================================================
;; Device System with PyTorch CUDA Backend
;; Extends the original device.rkt with PyTorch integration
;; ============================================================

(require "hardware_detection.rkt")

(provide 
 ;; Device types and predicates
 my-device?
 cpu-device?
 gpu-device?
 cuda-device?
 
 ;; Device constructors
 make-device
 cpu
 cuda
 
 ;; Device management
 current-device
 set-current-device!
 device-synchronize
 get-device-type
 device-available?
 gpu-available?
 
 ;; PyTorch CUDA detection
 pytorch-cuda-available?
 
 ;; Unified Tensor API (dispatches based on current-device)
 tensor
 zeros
 ones
 randn
 arange
 
 ;; Tensor properties
 shape
 dtype
 device-of
 to-device
 to-list
 
 ;; Basic operations
 add
 sub
 mul
 div
 matmul
 mm
 bmm
 transpose
 reshape
 squeeze
 unsqueeze
 
 ;; Math operations
 t:exp
 t:log
 t:sqrt
 t:abs
 t:sum
 t:mean
 t:max
 t:min
 t:sin
 t:cos
 cat
 stack
 
 ;; Activation functions
 relu
 gelu
 sigmoid
 t:tanh
 softmax
 
 ;; Neural network operations
 linear
 layer-norm
 dropout
 embedding
 einsum
 
 ;; Utility
 sync!)

;; ============================================================
;; Conditionally load PyTorch backend
;; ============================================================

(define *pytorch-loaded* #f)
(define *pytorch-module* #f)

(define (ensure-pytorch!)
  (unless *pytorch-loaded*
    (set! *pytorch-module* (dynamic-require "pytorch_backend.rkt" #f))
    (set! *pytorch-loaded* #t)))

;; Get a function from pytorch_backend
(define (pt-fn name)
  (ensure-pytorch!)
  (dynamic-require "pytorch_backend.rkt" name))

;; ============================================================
;; Device Structure
;; ============================================================

(struct device (type properties) #:transparent #:mutable)

(define CPU 'cpu)
(define CUDA 'cuda)

(define (my-device? obj) (device? obj))
(define (cpu-device? dev) (and (my-device? dev) (eq? (device-type dev) CPU)))
(define (cuda-device? dev) (and (my-device? dev) (eq? (device-type dev) CUDA)))
(define (gpu-device? dev) (cuda-device? dev))

;; ============================================================
;; PyTorch CUDA Detection
;; ============================================================

(define (pytorch-cuda-available?)
  (with-handlers ([exn:fail? (lambda (e) #f)])
    (ensure-pytorch!)
    ((pt-fn 'pytorch-cuda-available?))))

;; ============================================================
;; Device Instances
;; ============================================================

(define cpu-device-instance (device CPU (hash 'backend 'racket)))

(define cuda-device-instance
  (if (pytorch-cuda-available?)
      (device CUDA (hash 'backend 'pytorch 
                         'device-name ((pt-fn 'pytorch-device-name))))
      #f))

;; ============================================================
;; Device Constructors
;; ============================================================

(define (make-device type [properties (hash)])
  (device type properties))

(define (cpu) cpu-device-instance)

(define (cuda)
  (if cuda-device-instance
      cuda-device-instance
      (error "CUDA not available - requires NVIDIA GPU with PyTorch")))

;; ============================================================
;; Current Device Management
;; ============================================================

(define current-device-param 
  (make-parameter (if cuda-device-instance cuda-device-instance cpu-device-instance)))

(define (current-device) (current-device-param))

(define (set-current-device! dev)
  (unless (my-device? dev)
    (error "set-current-device!: expected a device, got ~a" dev))
  (current-device-param dev))

(define (get-device-type [dev (current-device)])
  (device-type dev))

(define (device-available? type)
  (case type
    [(cpu) #t]
    [(cuda gpu) (pytorch-cuda-available?)]
    [else #f]))

(define (gpu-available?) (pytorch-cuda-available?))

(define (device-synchronize [dev (current-device)])
  (when (cuda-device? dev)
    ((pt-fn 'pt:sync!))))

(define (sync!) (device-synchronize))

;; ============================================================
;; Device String Helper
;; ============================================================

(define (device-string [dev (current-device)])
  (if (cuda-device? dev) "cuda" "cpu"))

;; ============================================================
;; Unified Tensor API
;; Dispatches to PyTorch for CUDA, native Racket for CPU
;; ============================================================

;; For now, we use PyTorch for both CPU and CUDA when available
;; This gives us a consistent API and autograd support

(define (tensor data #:device [dev #f])
  (define d (or dev (current-device)))
  ((pt-fn 'pt:tensor) data #:device (device-string d)))

(define (zeros shape #:device [dev #f])
  (define d (or dev (current-device)))
  ((pt-fn 'pt:zeros) shape #:device (device-string d)))

(define (ones shape #:device [dev #f])
  (define d (or dev (current-device)))
  ((pt-fn 'pt:ones) shape #:device (device-string d)))

(define (randn shape #:device [dev #f])
  (define d (or dev (current-device)))
  ((pt-fn 'pt:randn) shape #:device (device-string d)))

(define (arange start end #:step [step 1] #:device [dev #f])
  (define d (or dev (current-device)))
  ((pt-fn 'pt:arange) start end #:step step #:device (device-string d)))

;; ============================================================
;; Tensor Properties
;; ============================================================

(define (shape t) ((pt-fn 'pt:shape) t))
(define (dtype t) ((pt-fn 'pt:dtype) t))
(define (device-of t) ((pt-fn 'pt:device) t))
(define (to-device t dev) ((pt-fn 'pt:to) t (device-string dev)))
(define (to-list t) ((pt-fn 'pt:to-list) t))

;; ============================================================
;; Basic Operations
;; ============================================================

(define (add a b) ((pt-fn 'pt:add) a b))
(define (sub a b) ((pt-fn 'pt:sub) a b))
(define (mul a b) ((pt-fn 'pt:mul) a b))
(define (div a b) ((pt-fn 'pt:div) a b))
(define (matmul a b) ((pt-fn 'pt:matmul) a b))
(define (mm a b) ((pt-fn 'pt:mm) a b))
(define (bmm a b) ((pt-fn 'pt:bmm) a b))
(define (transpose t dim0 dim1) ((pt-fn 'pt:transpose) t dim0 dim1))
(define (reshape t new-shape) ((pt-fn 'pt:reshape) t new-shape))
(define (squeeze t #:dim [dim #f]) ((pt-fn 'pt:squeeze) t #:dim dim))
(define (unsqueeze t dim) ((pt-fn 'pt:unsqueeze) t dim))

;; ============================================================
;; Math Operations
;; ============================================================

(define (t:exp t) ((pt-fn 'pt:exp) t))
(define (t:log t) ((pt-fn 'pt:log) t))
(define (t:sqrt t) ((pt-fn 'pt:sqrt) t))
(define (t:abs t) ((pt-fn 'pt:abs) t))
(define (t:sum t #:dim [dim #f] #:keepdim [keepdim #f]) 
  ((pt-fn 'pt:sum) t #:dim dim #:keepdim keepdim))
(define (t:mean t #:dim [dim #f] #:keepdim [keepdim #f])
  ((pt-fn 'pt:mean) t #:dim dim #:keepdim keepdim))
(define (t:max t #:dim [dim #f]) ((pt-fn 'pt:max) t #:dim dim))
(define (t:min t #:dim [dim #f]) ((pt-fn 'pt:min) t #:dim dim))

;; ============================================================
;; Activation Functions
;; ============================================================

(define (relu t) ((pt-fn 'pt:relu) t))
(define (gelu t) ((pt-fn 'pt:gelu) t))
(define (sigmoid t) ((pt-fn 'pt:sigmoid) t))
(define (t:tanh t) ((pt-fn 'pt:tanh) t))
(define (softmax t #:dim [dim -1]) ((pt-fn 'pt:softmax) t #:dim dim))

;; ============================================================
;; Neural Network Operations
;; ============================================================

(define (linear input weight #:bias [bias #f])
  ((pt-fn 'pt:linear) input weight #:bias bias))

(define (layer-norm input normalized-shape #:eps [eps 1e-5])
  ((pt-fn 'pt:layer-norm) input normalized-shape #:eps eps))

(define (dropout t #:p [p 0.1] #:training [training #t])
  ((pt-fn 'pt:dropout) t #:p p #:training training))

(define (embedding weight indices)
  ((pt-fn 'pt:embedding) weight indices))

(define (einsum equation . tensors)
  (apply (pt-fn 'pt:einsum) equation tensors))

;; ============================================================
;; Initialization Message
;; ============================================================

(printf "RacoGrad Device System loaded\n")
(printf "  PyTorch CUDA: ~a\n" (if (pytorch-cuda-available?) "available" "not available"))
(printf "  Default device: ~a\n" (device-type (current-device)))
(when cuda-device-instance
  (printf "  GPU: ~a\n" (hash-ref (device-properties cuda-device-instance) 'device-name)))

;; ============================================================
;; Additional ops for attention (added for Step 4)
;; ============================================================

(provide triu tril full masked-fill causal-mask)

(define (triu t #:diagonal [diagonal 1])
  ((pt-fn 'pt:triu) t #:diagonal diagonal))

(define (tril t #:diagonal [diagonal 0])
  ((pt-fn 'pt:tril) t #:diagonal diagonal))

(define (full shape fill-value #:device [dev #f])
  (define d (or dev (current-device)))
  ((pt-fn 'pt:full) shape fill-value #:device (device-string d)))

(define (masked-fill t mask value)
  ((pt-fn 'pt:masked-fill) t mask value))

(define (causal-mask seq-len #:device [dev #f])
  (define d (or dev (current-device)))
  ((pt-fn 'pt:causal-mask) seq-len #:device (device-string d)))

;; ============================================================
;; Loss Functions
;; ============================================================

(provide cross-entropy-loss

         )

(define (cross-entropy-loss logits targets #:ignore-index [ignore-idx -100])
  ;; logits: (batch, seq_len, vocab_size)
  ;; targets: (batch, seq_len) - integer class indices
  ;; Returns scalar loss
  ((pt-fn 'pt:cross-entropy) logits targets #:ignore-index ignore-idx))


;; ============================================================
;; Training Utilities
;; ============================================================

(provide backward
         get-item
         make-adam
         opt-step
         opt-zero)

(define (backward loss) ((pt-fn 'pt:backward) loss))
(define (get-item tensor) ((pt-fn 'pt:item) tensor))
(define (make-adam params #:lr [lr 0.001]) ((pt-fn 'pt:make-adam) params #:lr lr))
(define (opt-step opt) ((pt-fn 'pt:opt-step) opt))
(define (opt-zero opt) ((pt-fn 'pt:opt-zero) opt))

;; ============================================================
;; Trigonometric Functions  
;; ============================================================

(define (t:sin t) ((pt-fn 'pt:sin) t))
(define (t:cos t) ((pt-fn 'pt:cos) t))

;; ============================================================
;; Tensor Concatenation
;; ============================================================

(define (cat tensors #:dim [dim 0]) ((pt-fn 'pt:cat) tensors #:dim dim))
(define (stack tensors #:dim [dim 0]) ((pt-fn 'pt:stack) tensors #:dim dim))

;; ============================================================
;; Tensor Type Casting
;; ============================================================

(provide to-float to-long to-int)

(define (to-float t) ((pt-fn 'pt:float) t))
(define (to-long t) ((pt-fn 'pt:long) t))
(define (to-int t) ((pt-fn 'pt:int) t))

;; ============================================================
;; Tensor Slicing
;; ============================================================

(provide slice-dim)

(define (slice-dim t dim start end) ((pt-fn 'pt:slice) t dim start end))
