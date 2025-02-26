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

;; Load CPU implementations with error handling
(define cnn-lib 
  (with-handlers ([exn:fail? (lambda (e) 
                             (printf "Warning: cnn_ops library not found. ~a~n" (exn-message e))
                             #f)])
    (ffi-lib "cnn_ops" '("" "0"))))

;; Try to load MLX implementations if on Apple Silicon
(define mlx-cnn-lib
  (with-handlers ([exn:fail? (lambda (e) #f)])
    (ffi-lib "mlx_cnn_ops" '("" "0"))))

;; Dummy implementations when library is not available
(define (dummy-conv2d batch-size in-channels in-height in-width
                    out-channels filter-height filter-width
                    stride padding
                    input filters output)
  (printf "Using dummy conv2d implementation~n"))

(define (dummy-max-pool batch-size channels in-height in-width
                       input output)
  (printf "Using dummy max-pool implementation~n"))

(define (dummy-flatten batch-size channels height width
                      input output)
  (printf "Using dummy flatten implementation~n"))

(define (dummy-softmax batch-size num-classes
                      input output)
  (printf "Using dummy softmax implementation~n"))

(define (dummy-cross-entropy batch-size num-classes
                           predictions targets)
  (printf "Using dummy cross-entropy implementation~n")
  0.1) ; Return a dummy loss

;; Default implementations using CPU (or dummy if not available)
(define c:conv2d-forward
  (if cnn-lib
      (with-handlers ([exn:fail? (lambda (e) dummy-conv2d)])
        (get-ffi-obj "conv2d_forward" cnn-lib
                    (_fun _int _int _int _int    ; batch_size, in_channels, in_height, in_width
                          _int _int _int         ; out_channels, filter_height, filter_width
                          _int _int              ; stride, padding
                          _f64vector _f64vector _f64vector -> _void)))
      dummy-conv2d))

(define c:max-pool-2x2
  (if cnn-lib
      (with-handlers ([exn:fail? (lambda (e) dummy-max-pool)])
        (get-ffi-obj "max_pool_2x2" cnn-lib
                    (_fun _int _int _int _int    ; batch_size, channels, in_height, in_width
                          _f64vector _f64vector -> _void)))
      dummy-max-pool))

(define c:flatten-tensor
  (if cnn-lib
      (with-handlers ([exn:fail? (lambda (e) dummy-flatten)])
        (get-ffi-obj "flatten_tensor" cnn-lib
                    (_fun _int _int _int _int    ; batch_size, channels, height, width
                          _f64vector _f64vector -> _void)))
      dummy-flatten))

(define c:softmax
  (if cnn-lib
      (with-handlers ([exn:fail? (lambda (e) dummy-softmax)])
        (get-ffi-obj "softmax" cnn-lib
                    (_fun _int _int              ; batch_size, num_classes
                          _f64vector _f64vector -> _void)))
      dummy-softmax))

(define c:cross-entropy-loss
  (if cnn-lib
      (with-handlers ([exn:fail? (lambda (e) dummy-cross-entropy)])
        (get-ffi-obj "cross_entropy_loss" cnn-lib
                    (_fun _int _int              ; batch_size, num_classes
                          _f64vector _f64vector -> _double)))
      dummy-cross-entropy))

;; MLX implementations (if available)
(define mlx:conv2d-forward
  (if mlx-cnn-lib
      (with-handlers ([exn:fail? (lambda (e) 
                                  (printf "MLX conv2d not available: ~a~n" (exn-message e))
                                  #f)])
        (get-ffi-obj "mlx_conv2d_forward" mlx-cnn-lib
                     (_fun _int _int _int _int
                           _int _int _int
                           _int _int
                           _f64vector _f64vector _f64vector -> _void)))
      ;; If MLX not available, use dummy or CPU implementation
      (if cnn-lib c:conv2d-forward dummy-conv2d)))

(define mlx:max-pool-2x2
  (if mlx-cnn-lib
      (with-handlers ([exn:fail? (lambda (e) #f)])
        (get-ffi-obj "mlx_max_pool_2x2" mlx-cnn-lib
                     (_fun _int _int _int _int
                           _f64vector _f64vector -> _void)))
      ;; If MLX not available, use dummy or CPU implementation
      (if cnn-lib c:max-pool-2x2 dummy-max-pool)))

(define mlx:flatten-tensor
  (if mlx-cnn-lib
      (with-handlers ([exn:fail? (lambda (e) #f)])
        (get-ffi-obj "mlx_flatten_tensor" mlx-cnn-lib
                     (_fun _int _int _int _int
                           _f64vector _f64vector -> _void)))
      ;; If MLX not available, use dummy or CPU implementation
      (if cnn-lib c:flatten-tensor dummy-flatten)))

(define mlx:softmax
  (if mlx-cnn-lib
      (with-handlers ([exn:fail? (lambda (e) #f)])
        (get-ffi-obj "mlx_softmax" mlx-cnn-lib
                     (_fun _int _int
                           _f64vector _f64vector -> _void)))
      ;; If MLX not available, use dummy or CPU implementation
      (if cnn-lib c:softmax dummy-softmax)))

;; Function to reload operations based on device
(define (load-optimal-ops [dev (current-device)])
  (cond
    [(mlx-device? dev)
     (printf "Loading MLX optimized CNN operations for Apple Silicon~n")
     (if mlx-cnn-lib
         (begin
           (set! c:conv2d-forward mlx:conv2d-forward)
           (set! c:max-pool-2x2 mlx:max-pool-2x2)
           (set! c:flatten-tensor mlx:flatten-tensor)
           (set! c:softmax mlx:softmax))
         (printf "Warning: MLX library not available. Using CPU fallback.~n"))]
    
    [(gpu-device? dev)
     (printf "GPU acceleration not yet implemented, using CPU fallback~n")]
    
    [else
     (printf "Using standard CPU CNN operations~n")
     (if (not cnn-lib)
         (printf "Warning: CNN operations library not found. Using dummy implementations.~n"))]))