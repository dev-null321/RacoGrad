#lang racket

;; ============================================================
;; Native safetensors loader for RacoGrad.
;;
;; Format:
;;   [0:8]   uint64 LE  — header length N
;;   [8:8+N] UTF-8 JSON — { "tensor-name": { dtype, shape, data_offsets }, ... }
;;   [8+N:]  raw bytes — concatenated tensor payloads, indexed by data_offsets
;;
;; This module opens the file as a memory-mapped-ish handle (we read the
;; bytes we need on demand, not the whole file into memory), parses the
;; header, and exposes:
;;
;;   (open-safetensors path)        → st handle
;;   (safetensors-keys st)          → (listof string) tensor names
;;   (safetensors-info st name)     → (hash 'dtype 'shape 'offsets)
;;   (safetensors-raw st name)      → bytes (raw tensor bytes, LE)
;;   (safetensors-tensor st name)   → libtorch tensor on CUDA
;;   (close-safetensors st)         → void
;;
;; Supported dtypes: F32, F16, BF16. Others raise a clear error.
;; ============================================================

(require racket/format
         json
         "libtorch_backend.rkt")

(provide open-safetensors
         close-safetensors
         safetensors-keys
         safetensors-info
         safetensors-raw
         safetensors-tensor)

(struct st (port header data-start) #:transparent)

(define (read-header-length in)
  ;; 8 bytes little-endian unsigned 64-bit
  (integer-bytes->integer (read-bytes 8 in) #f #f))

(define (parse-dtype sym)
  (case sym
    [(F32) 'f32]
    [(F16) 'f16]
    [(BF16) 'bf16]
    [else (error 'safetensors
                 "unsupported dtype ~a (supported: F32 F16 BF16)" sym)]))

(define (open-safetensors path)
  (define in (open-input-file path #:mode 'binary))
  (define hdr-len (read-header-length in))
  (define hdr-bytes (read-bytes hdr-len in))
  (define hdr-json (bytes->jsexpr hdr-bytes))
  ;; Header includes an optional "__metadata__" key; drop it from tensor list.
  (define tensors
    (for/hash ([(k v) (in-hash hdr-json)]
               #:unless (equal? k '__metadata__))
      (values (symbol->string k)
              (hash 'dtype   (parse-dtype (string->symbol (hash-ref v 'dtype)))
                    'shape   (hash-ref v 'shape)
                    'offsets (hash-ref v 'data_offsets)))))
  (st in tensors (+ 8 hdr-len)))

(define (close-safetensors s)
  (close-input-port (st-port s)))

(define (safetensors-keys s)
  (sort (hash-keys (st-header s)) string<?))

(define (safetensors-info s name)
  (hash-ref (st-header s) name
            (lambda () (error 'safetensors-info "no such tensor: ~a" name))))

;; Read the raw payload bytes for a given tensor.
;; Jumps to the data region at (data-start + begin) and reads (end - begin).
(define (safetensors-raw s name)
  (define info (safetensors-info s name))
  (match-define (list begin end) (hash-ref info 'offsets))
  (define len (- end begin))
  (file-position (st-port s) (+ (st-data-start s) begin))
  (read-bytes len (st-port s)))

;; Load a tensor directly into a libtorch tensor on CUDA.
(define (safetensors-tensor s name)
  (define info (safetensors-info s name))
  (define bs (safetensors-raw s name))
  (tensor-from-bytes bs (hash-ref info 'dtype) (hash-ref info 'shape)))
