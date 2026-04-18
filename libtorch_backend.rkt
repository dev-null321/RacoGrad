#lang racket

;; ============================================================
;; RacoGrad libtorch Backend — Zero Python
;;
;; Racket FFI → libracograd_ffi.so → libtorch → CUDA
;;
;; Drop-in replacement for device_pytorch.rkt + pytorch_backend.rkt
;; ============================================================

(require ffi/unsafe
         ffi/unsafe/define)

(provide
 ;; Lifecycle
 rg-init
 rg-cuda-available?
 rg-cuda-device-name
 rg-sync
 rg-cuda-empty-cache
 tensor-free
 rg-scope-begin
 rg-scope-end
 
 ;; Tensor creation
 zeros ones randn full arange
 tensor
 tensor-from-list
 tensor-from-bytes
 
 ;; Tensor info
 shape ndim numel get-item
 
 ;; Device / type
 to-cuda to-cpu to-float to-long to-int
 cuda?
 
 ;; Requires grad
 set-requires-grad! requires-grad?
 
 ;; Math
 add sub mul div neg
 t:abs t:sqrt t:exp t:log t:sin t:cos t:tanh t:sigmoid
 t:pow
 
 ;; Reduction
 sum sum-dim mean mean-dim argmax
 
 ;; Matrix
 matmul mm bmm transpose
 
 ;; Shape
 reshape unsqueeze squeeze cat stack slice-dim contiguous
 
 ;; Mask
 triu tril masked-fill eq
 
 ;; NN
 relu gelu softmax dropout
 layer-norm embedding linear
 
 ;; Loss
 cross-entropy-loss mse-loss nll-loss
 
 ;; Autograd
 backward grad zero-grad! detach
 no-grad-begin no-grad-end
 
 ;; Optimizer
 make-adam adam-step adam-zero-grad adam-free adam-set-lr!
 
 ;; Misc
 clone t:copy t:print einsum clamp
 free-tensor
 
 ;; Compat
 causal-mask)

;; ============================================================
;; Load the shared library
;; ============================================================

(define lib-path
  (let ([paths (list
                "ffi/libracograd_ffi.so"
                "./libracograd_ffi.so"
                (build-path (current-directory) "ffi" "libracograd_ffi.so")
                (expand-user-path "~/Projects/RacoGrad/ffi/libracograd_ffi.so"))])
    (for/first ([p paths] #:when (file-exists? p)) p)))

(define-ffi-definer define-rg (ffi-lib (or lib-path "ffi/libracograd_ffi")))

;; ============================================================
;; Type aliases
;; ============================================================

(define _handle _int64)

;; ============================================================
;; Lifecycle
;; ============================================================

(define-rg rg_init (_fun -> _int))
(define-rg rg_cuda_available (_fun -> _int))
(define-rg rg_cuda_device_name (_fun -> _string))
(define-rg rg_sync (_fun -> _void))
(define-rg rg_free_tensor (_fun _handle -> _void))
(define-rg rg_cuda_empty_cache (_fun -> _void))
(define-rg rg_scope_begin (_fun -> _int64))
(define-rg rg_scope_end   (_fun _int64 _pointer _int -> _void))

(define (rg-scope-begin) (rg_scope_begin))
(define (rg-scope-end watermark protect-handles)
  ;; protect-handles is a list of int64 handle values to keep
  (define n (length protect-handles))
  (define arr (malloc (* n 8) 'atomic))
  (for ([h protect-handles] [i (in-naturals)])
    (ptr-set! arr _int64 i h))
  (rg_scope_end watermark arr n))

(define (rg-init) (= 1 (rg_init)))
(define (rg-cuda-available?) (= 1 (rg_cuda_available)))
(define (rg-cuda-device-name) (rg_cuda_device_name))
(define (rg-sync) (rg_sync))
(define (rg-cuda-empty-cache) (rg_cuda_empty_cache))
(define (tensor-free h) (rg_free_tensor h))
(define free-tensor tensor-free)

;; ============================================================
;; Tensor Creation
;; ============================================================

(define-rg rg_zeros (_fun _int _pointer _int -> _handle))
(define-rg rg_ones (_fun _int _pointer _int -> _handle))
(define-rg rg_randn (_fun _int _pointer _int -> _handle))
(define-rg rg_full (_fun _int _pointer _double _int -> _handle))
(define-rg rg_arange (_fun _int64 _int64 _int64 _int -> _handle))
(define-rg rg_tensor_from_float (_fun _double -> _handle))
(define-rg rg_tensor_from_data (_fun _int _pointer _pointer _int _int -> _handle))
(define-rg rg_tensor_from_long_data (_fun _int _pointer _pointer _int _int -> _handle))
(define-rg rg_tensor_from_f32_bytes  (_fun _int _pointer _pointer _int _int -> _handle))
(define-rg rg_tensor_from_f16_bytes  (_fun _int _pointer _pointer _int _int -> _handle))
(define-rg rg_tensor_from_bf16_bytes (_fun _int _pointer _pointer _int _int -> _handle))

(define use-cuda 1)  ;; Default: use CUDA if available

(define (make-dims-ptr dims)
  (define n (length dims))
  (define ptr (malloc _int64 n))
  (for ([d dims] [i (in-naturals)])
    (ptr-set! ptr _int64 i d))
  (values ptr n))

(define (zeros shape)
  (define-values (ptr n) (make-dims-ptr shape))
  (rg_zeros n ptr use-cuda))

(define (ones shape)
  (define-values (ptr n) (make-dims-ptr shape))
  (rg_ones n ptr use-cuda))

(define (randn shape #:device [dev #f])
  (define-values (ptr n) (make-dims-ptr shape))
  (rg_randn n ptr use-cuda))

(define (full shape val)
  (define-values (ptr n) (make-dims-ptr shape))
  (rg_full n ptr (exact->inexact val) use-cuda))

(define (arange start end #:step [step 1] #:device [dev #f])
  (rg_arange start end step use-cuda))

;; Create tensor from a scalar or nested list
(define (tensor data #:device [dev #f])
  (cond
    [(number? data)
     (rg_tensor_from_float (exact->inexact data))]
    [(list? data)
     (define flat (flatten-list data))
     (define shape (infer-shape data))
     (define n (length flat))
     ;; Check if all integers (for long tensor)
     (if (andmap exact-integer? flat)
         (let ()
           (define-values (dims-ptr ndims) (make-dims-ptr shape))
           (define data-ptr (malloc _int64 n))
           (for ([v flat] [i (in-naturals)])
             (ptr-set! data-ptr _int64 i v))
           (rg_tensor_from_long_data ndims dims-ptr data-ptr n use-cuda))
         (let ()
           (define-values (dims-ptr ndims) (make-dims-ptr shape))
           (define data-ptr (malloc _double n))
           (for ([v flat] [i (in-naturals)])
             (ptr-set! data-ptr _double i (exact->inexact v)))
           (rg_tensor_from_data ndims dims-ptr data-ptr n use-cuda)))]))

(define (tensor-from-list data) (tensor data))

;; Create a tensor directly from a raw byte buffer.
;; `bs` is a Racket bytes? with the tensor's packed contents in row-major
;; layout, little-endian. `dtype` ∈ {'f32 'f16 'bf16}. `shape` is a list
;; of positive integers. Returned tensor lives on CUDA if CUDA is enabled.
;;
;; Used by the native safetensors loader so we don't pay Racket↔FFI
;; per-element marshalling cost on hundred-MB weight files.
(define (tensor-from-bytes bs dtype shape)
  (define n (for/product ([d shape]) d))
  (define bytes-per-elem
    (case dtype [(f32) 4] [(f16) 2] [(bf16) 2]
                [else (error 'tensor-from-bytes "unsupported dtype: ~a" dtype)]))
  (unless (= (bytes-length bs) (* n bytes-per-elem))
    (error 'tensor-from-bytes
           "byte count ~a does not match shape ~a × dtype ~a (expected ~a bytes)"
           (bytes-length bs) shape dtype (* n bytes-per-elem)))
  (define-values (dims-ptr ndims) (make-dims-ptr shape))
  ;; `bs` is a Racket bytes which _pointer-has-type? treats as a raw pointer.
  (case dtype
    [(f32)  (rg_tensor_from_f32_bytes  ndims dims-ptr bs n use-cuda)]
    [(f16)  (rg_tensor_from_f16_bytes  ndims dims-ptr bs n use-cuda)]
    [(bf16) (rg_tensor_from_bf16_bytes ndims dims-ptr bs n use-cuda)]))

(define (flatten-list lst)
  (cond
    [(not (list? lst)) (list lst)]
    [else (apply append (map flatten-list lst))]))

(define (infer-shape data)
  (cond
    [(not (list? data)) '()]
    [(null? data) '(0)]
    [else (cons (length data) (infer-shape (car data)))]))

;; ============================================================
;; Tensor Info
;; ============================================================

(define-rg rg_ndim (_fun _handle -> _int))
(define-rg rg_shape (_fun _handle _int -> _int64))
(define-rg rg_item (_fun _handle -> _double))
(define-rg rg_numel (_fun _handle -> _int64))

(define (ndim h) (rg_ndim h))
(define (shape h)
  (for/list ([i (in-range (ndim h))])
    (rg_shape h i)))
(define (get-item h) (rg_item h))
(define (numel h) (rg_numel h))

;; ============================================================
;; Device / Type
;; ============================================================

(define-rg rg_to_cuda (_fun _handle -> _handle))
(define-rg rg_to_cpu (_fun _handle -> _handle))
(define-rg rg_to_float (_fun _handle -> _handle))
(define-rg rg_to_long (_fun _handle -> _handle))
(define-rg rg_to_int (_fun _handle -> _handle))
(define-rg rg_is_cuda (_fun _handle -> _int))

(define (to-cuda h) (rg_to_cuda h))
(define (to-cpu h) (rg_to_cpu h))
(define (to-float h) (rg_to_float h))
(define (to-long h) (rg_to_long h))
(define (to-int h) (rg_to_int h))
(define (cuda? h) (= 1 (rg_is_cuda h)))

;; ============================================================
;; Requires Grad
;; ============================================================

(define-rg rg_set_requires_grad (_fun _handle _int -> _handle))
(define-rg rg_requires_grad (_fun _handle -> _int))

(define (set-requires-grad! h [val #t])
  (rg_set_requires_grad h (if val 1 0)))
(define (requires-grad? h) (= 1 (rg_requires_grad h)))

;; ============================================================
;; Math
;; ============================================================

(define-rg rg_add (_fun _handle _handle -> _handle))
(define-rg rg_sub (_fun _handle _handle -> _handle))
(define-rg rg_mul (_fun _handle _handle -> _handle))
(define-rg rg_div (_fun _handle _handle -> _handle))
(define-rg rg_neg (_fun _handle -> _handle))
(define-rg rg_abs (_fun _handle -> _handle))
(define-rg rg_sqrt (_fun _handle -> _handle))
(define-rg rg_exp (_fun _handle -> _handle))
(define-rg rg_log (_fun _handle -> _handle))
(define-rg rg_sin (_fun _handle -> _handle))
(define-rg rg_cos (_fun _handle -> _handle))
(define-rg rg_tanh (_fun _handle -> _handle))
(define-rg rg_sigmoid (_fun _handle -> _handle))
(define-rg rg_pow (_fun _handle _double -> _handle))

(define (add a b) (rg_add a b))
(define (sub a b) (rg_sub a b))
(define (mul a b) (rg_mul a b))
(define (div a b) (rg_div a b))
(define (neg h) (rg_neg h))
(define t:abs rg_abs)
(define t:sqrt rg_sqrt)
(define t:exp rg_exp)
(define t:log rg_log)
(define t:sin rg_sin)
(define t:cos rg_cos)
(define t:tanh rg_tanh)
(define t:sigmoid rg_sigmoid)
(define (t:pow h exp) (rg_pow h (exact->inexact exp)))

;; ============================================================
;; Reduction
;; ============================================================

(define-rg rg_sum (_fun _handle -> _handle))
(define-rg rg_sum_dim (_fun _handle _int _int -> _handle))
(define-rg rg_mean (_fun _handle -> _handle))
(define-rg rg_mean_dim (_fun _handle _int _int -> _handle))
(define-rg rg_argmax (_fun _handle _int -> _handle))

(define (sum h) (rg_sum h))
(define (sum-dim h dim #:keepdim [kd #f]) (rg_sum_dim h dim (if kd 1 0)))
(define (mean h) (rg_mean h))
(define (mean-dim h dim #:keepdim [kd #f]) (rg_mean_dim h dim (if kd 1 0)))
(define (argmax h dim) (rg_argmax h dim))

;; ============================================================
;; Matrix
;; ============================================================

(define-rg rg_matmul (_fun _handle _handle -> _handle))
(define-rg rg_mm (_fun _handle _handle -> _handle))
(define-rg rg_bmm (_fun _handle _handle -> _handle))
(define-rg rg_transpose (_fun _handle _int _int -> _handle))

(define (matmul a b) (rg_matmul a b))
(define (mm a b) (rg_mm a b))
(define (bmm a b) (rg_bmm a b))
(define (transpose h d0 d1) (rg_transpose h d0 d1))

;; ============================================================
;; Shape
;; ============================================================

(define-rg rg_reshape (_fun _handle _int _pointer -> _handle))
(define-rg rg_unsqueeze (_fun _handle _int -> _handle))
(define-rg rg_squeeze (_fun _handle _int -> _handle))
(define-rg rg_cat (_fun _pointer _int _int -> _handle))
(define-rg rg_stack (_fun _pointer _int _int -> _handle))
(define-rg rg_slice (_fun _handle _int _int64 _int64 -> _handle))
(define-rg rg_contiguous (_fun _handle -> _handle))

(define (reshape h new-shape)
  (define-values (ptr n) (make-dims-ptr new-shape))
  (rg_reshape h n ptr))

(define (unsqueeze h dim) (rg_unsqueeze h dim))
(define (squeeze h dim) (rg_squeeze h dim))

(define (cat tensors #:dim [dim 0])
  (define n (length tensors))
  (define ptr (malloc _handle n))
  (for ([t tensors] [i (in-naturals)])
    (ptr-set! ptr _handle i t))
  (rg_cat ptr n dim))

(define (stack tensors #:dim [dim 0])
  (define n (length tensors))
  (define ptr (malloc _handle n))
  (for ([t tensors] [i (in-naturals)])
    (ptr-set! ptr _handle i t))
  (rg_stack ptr n dim))

(define (slice-dim h dim start end)
  (rg_slice h dim start end))

(define (contiguous h) (rg_contiguous h))

;; ============================================================
;; Mask
;; ============================================================

(define-rg rg_triu (_fun _handle _int -> _handle))
(define-rg rg_tril (_fun _handle _int -> _handle))
(define-rg rg_masked_fill (_fun _handle _handle _double -> _handle))
(define-rg rg_eq (_fun _handle _handle -> _handle))

(define (triu h #:diagonal [d 0]) (rg_triu h d))
(define (tril h #:diagonal [d 0]) (rg_tril h d))
(define (masked-fill h mask val) (rg_masked_fill h mask (exact->inexact val)))
(define (eq a b) (rg_eq a b))

;; ============================================================
;; NN Ops
;; ============================================================

(define-rg rg_relu (_fun _handle -> _handle))
(define-rg rg_gelu (_fun _handle -> _handle))
(define-rg rg_softmax (_fun _handle _int -> _handle))
(define-rg rg_dropout (_fun _handle _double _int -> _handle))
(define-rg rg_layer_norm (_fun _handle _handle _handle _int _double -> _handle))
(define-rg rg_layer_norm_simple (_fun _handle _int _double -> _handle))
(define-rg rg_embedding (_fun _handle _handle -> _handle))
(define-rg rg_linear (_fun _handle _handle _handle -> _handle))

(define (relu h) (rg_relu h))
(define (gelu h) (rg_gelu h))
(define (softmax h #:dim [dim -1]) (rg_softmax h dim))
(define (dropout h #:p [p 0.1] #:training [training #t]) (rg_dropout h (exact->inexact p) (if training 1 0)))
;; nn.rkt calls (layer-norm input normalized-shape #:eps eps) — just normalize, no gamma/beta
;; The gamma/beta are applied manually by nn.rkt via mul+add
(define (layer-norm h normalized-shape #:eps [eps 1e-5])
  (define norm-size (if (list? normalized-shape) (car normalized-shape) normalized-shape))
  (rg_layer_norm_simple h norm-size (exact->inexact eps)))
(define (embedding weight indices) (rg_embedding weight indices))
(define (linear input weight #:bias [bias 0]) (rg_linear input weight bias))

;; ============================================================
;; Loss
;; ============================================================

(define-rg rg_cross_entropy_loss (_fun _handle _handle _int64 -> _handle))
(define-rg rg_mse_loss (_fun _handle _handle -> _handle))
(define-rg rg_nll_loss (_fun _handle _handle -> _handle))

(define (cross-entropy-loss logits targets #:ignore-index [idx -100])
  (rg_cross_entropy_loss logits targets idx))
(define (mse-loss a b) (rg_mse_loss a b))
(define (nll-loss a b) (rg_nll_loss a b))

;; ============================================================
;; Autograd
;; ============================================================

(define-rg rg_backward (_fun _handle -> _void))
(define-rg rg_grad (_fun _handle -> _handle))
(define-rg rg_zero_grad (_fun _handle -> _void))
(define-rg rg_detach (_fun _handle -> _handle))
(define-rg rg_no_grad_begin (_fun -> _int64))
(define-rg rg_no_grad_end (_fun _int64 -> _void))

(define (backward h) (rg_backward h))
(define (grad h) (rg_grad h))
(define (zero-grad! h) (rg_zero_grad h))
(define (detach h) (rg_detach h))
(define (no-grad-begin) (rg_no_grad_begin))
(define (no-grad-end guard) (rg_no_grad_end guard))

;; ============================================================
;; Optimizer
;; ============================================================

(define-rg rg_adam_create (_fun _pointer _int _double _double _double _double -> _handle))
(define-rg rg_adam_step (_fun _handle -> _void))
(define-rg rg_adam_zero_grad (_fun _handle -> _void))
(define-rg rg_adam_free   (_fun _handle -> _void))
(define-rg rg_adam_set_lr (_fun _handle _double -> _void))

(define (make-adam param-handles #:lr [lr 0.001] #:beta1 [b1 0.9] #:beta2 [b2 0.999] #:weight-decay [wd 0.0])
  (define n (length param-handles))
  (define ptr (malloc _handle n))
  (for ([h param-handles] [i (in-naturals)])
    (ptr-set! ptr _handle i h))
  (rg_adam_create ptr n (exact->inexact lr) (exact->inexact b1) (exact->inexact b2) (exact->inexact wd)))

(define (adam-step h) (rg_adam_step h))
(define (adam-zero-grad h) (rg_adam_zero_grad h))
(define (adam-free h) (rg_adam_free h))
(define (adam-set-lr! h lr) (rg_adam_set_lr h (exact->inexact lr)))

;; ============================================================
;; Misc
;; ============================================================

(define-rg rg_clone (_fun _handle -> _handle))
(define-rg rg_copy (_fun _handle _handle -> _handle))
(define-rg rg_print (_fun _handle -> _void))
(define-rg rg_einsum (_fun _string _pointer _int -> _handle))
(define-rg rg_clamp (_fun _handle _double _double -> _handle))

(define (clone h) (rg_clone h))
(define (t:copy dst src) (rg_copy dst src))
(define (t:print h) (rg_print h))
(define (clamp h min-v max-v) (rg_clamp h (exact->inexact min-v) (exact->inexact max-v)))

(define (einsum equation . tensors)
  (define n (length tensors))
  (define ptr (malloc _handle n))
  (for ([t tensors] [i (in-naturals)])
    (ptr-set! ptr _handle i t))
  (rg_einsum equation ptr n))

;; ============================================================
;; Auto-init on load
;; ============================================================

(define cuda-ok (rg-init))
(printf "RacoGrad libtorch backend loaded\n")
(printf "  CUDA: ~a\n" (if (rg-cuda-available?) "available" "not available"))
(when (rg-cuda-available?)
  (printf "  GPU: ~a\n" (rg-cuda-device-name)))

;; ============================================================
;; Compat: causal-mask
;; ============================================================

(define (causal-mask seq-len #:device [dev #f])
  ;; Upper triangular boolean mask, then masked_fill with -inf
  (define o (ones (list seq-len seq-len)))
  (define upper (triu o #:diagonal 1))
  (define upper-mask (eq upper (ones (list seq-len seq-len))))
  (define neg-inf (- (/ 1.0 0.0)))
  (masked-fill (zeros (list seq-len seq-len)) upper-mask neg-inf))
