#lang racket

(require ffi/unsafe
         ffi/vector
         "device.rkt"
         "hardware_detection.rkt")

;; Provide CNN operations
(provide
 ;; CNN operations
 c:conv2d-forward
 c:max-pool-2x2
 c:flatten-tensor
 c:softmax
 c:cross-entropy-loss
 
 ;; Load functions based on available hardware
 load-optimal-ops)

;; Load CPU implementations
(define cnn-lib (ffi-lib "cnn_ops"))

;; Try to load MLX implementations if on Apple Silicon
(define mlx-cnn-lib
  (with-handlers ([exn:fail? (lambda (e) #f)])
    (ffi-lib "mlx_cnn_ops")))

;; Default implementations using CPU
(define c:conv2d-forward
  (get-ffi-obj "conv2d_forward" cnn-lib
               (_fun _int _int _int _int    ; batch_size, in_channels, in_height, in_width
                     _int _int _int         ; out_channels, filter_height, filter_width
                     _int _int              ; stride, padding
                     _f64vector _f64vector _f64vector -> _void)))

(define c:max-pool-2x2
  (get-ffi-obj "max_pool_2x2" cnn-lib
               (_fun _int _int _int _int    ; batch_size, channels, in_height, in_width
                     _f64vector _f64vector -> _void)))

(define c:flatten-tensor
  (get-ffi-obj "flatten_tensor" cnn-lib
               (_fun _int _int _int _int    ; batch_size, channels, height, width
                     _f64vector _f64vector -> _void)))

(define c:softmax
  (get-ffi-obj "softmax" cnn-lib
               (_fun _int _int              ; batch_size, num_classes
                     _f64vector _f64vector -> _void)))

(define c:cross-entropy-loss
  (get-ffi-obj "cross_entropy_loss" cnn-lib
               (_fun _int _int              ; batch_size, num_classes
                     _f64vector _f64vector -> _double)))

;; MLX implementations (if available)
(define mlx:conv2d-forward
  (if mlx-cnn-lib
      (with-handlers ([exn:fail? (lambda (e) #f)])
        (get-ffi-obj "mlx_conv2d_forward" mlx-cnn-lib
                     (_fun _int _int _int _int
                           _int _int _int
                           _int _int
                           _f64vector _f64vector _f64vector -> _void)))
      #f))

(define mlx:max-pool-2x2
  (if mlx-cnn-lib
      (with-handlers ([exn:fail? (lambda (e) #f)])
        (get-ffi-obj "mlx_max_pool_2x2" mlx-cnn-lib
                     (_fun _int _int _int _int
                           _f64vector _f64vector -> _void)))
      #f))

(define mlx:flatten-tensor
  (if mlx-cnn-lib
      (with-handlers ([exn:fail? (lambda (e) #f)])
        (get-ffi-obj "mlx_flatten_tensor" mlx-cnn-lib
                     (_fun _int _int _int _int
                           _f64vector _f64vector -> _void)))
      #f))

(define mlx:softmax
  (if mlx-cnn-lib
      (with-handlers ([exn:fail? (lambda (e) #f)])
        (get-ffi-obj "mlx_softmax" mlx-cnn-lib
                     (_fun _int _int
                           _f64vector _f64vector -> _void)))
      #f))

;; Function to reload operations based on device
(define (load-optimal-ops [dev (current-device)])
  (cond
    [(mlx-device? dev)
     (printf "Loading MLX optimized CNN operations for Apple Silicon~n")
     (when mlx:conv2d-forward
       (set! c:conv2d-forward mlx:conv2d-forward))
     (when mlx:max-pool-2x2
       (set! c:max-pool-2x2 mlx:max-pool-2x2))
     (when mlx:flatten-tensor
       (set! c:flatten-tensor mlx:flatten-tensor))
     (when mlx:softmax
       (set! c:softmax mlx:softmax))]
    
    [else
     (printf "Using standard CPU CNN operations~n")
     ; Keep the default CPU implementations
     ]))