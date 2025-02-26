#lang racket

(require ffi/unsafe
         ffi/vector)

(provide c:matrix-multiply
         c:tensor-add
         c:tensor-sub
         c:tensor-mul-elementwise
         c:tensor-scale
         c:relu-forward
         c:relu-backward
         c:sigmoid-forward
         c:sigmoid-backward)

;; Load the shared library
(define matrix-lib (ffi-lib "matrix_multiplication"))

;; Define the FFI functions
(define c:matrix-multiply
  (get-ffi-obj "matrix_multiply" matrix-lib
               (_fun _int _int _int _f64vector _f64vector _f64vector -> _void)))

(define c:tensor-add
  (get-ffi-obj "tensor_add" matrix-lib
               (_fun _int _f64vector _f64vector _f64vector -> _void)))

(define c:tensor-sub
  (get-ffi-obj "tensor_sub" matrix-lib
               (_fun _int _f64vector _f64vector _f64vector -> _void)))

(define c:tensor-mul-elementwise
  (get-ffi-obj "tensor_mul_elementwise" matrix-lib
               (_fun _int _f64vector _f64vector _f64vector -> _void)))

(define c:tensor-scale
  (get-ffi-obj "tensor_scale" matrix-lib
               (_fun _int _f64vector _double _f64vector -> _void)))

(define c:relu-forward
  (get-ffi-obj "relu_forward" matrix-lib
               (_fun _int _f64vector _f64vector -> _void)))

(define c:relu-backward
  (get-ffi-obj "relu_backward" matrix-lib
               (_fun _int _f64vector _f64vector -> _void)))

(define c:sigmoid-forward
  (get-ffi-obj "sigmoid_forward" matrix-lib
               (_fun _int _f64vector _f64vector -> _void)))

(define c:sigmoid-backward
  (get-ffi-obj "sigmoid_backward" matrix-lib
               (_fun _int _f64vector _f64vector -> _void)))