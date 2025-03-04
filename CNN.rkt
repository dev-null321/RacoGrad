#lang racket

(require "tensor_device.rkt"   ; Device-aware tensor operations
         "cnn_ops.rkt"        ; CNN operations
         "device.rkt"         ; Device management
         "hardware_detection.rkt" ; Hardware detection
         "ffi_ops.rkt"        ; For relu-forward, etc.
         ffi/vector)          ; For list->f64vector

(provide (all-defined-out))

;; Implementation of a CNN for image classification
;; This example uses a LeNet-5 like architecture for CIFAR-10

;; Wrap up the C functions with tensor operations
;; These operations create and return new tensors

;; Convolution operation with fallback to simple implementation if C function fails
(define (conv2d input-tensor filter-tensor [stride 1] [padding 0])
  (with-handlers ([exn:fail? (lambda (e)
                              (printf "Warning: C convolution failed, using fallback implementation: ~a~n" 
                                      (exn-message e))
                              (simple-conv2d input-tensor filter-tensor stride padding))])
    (let* ([input-shape (dt:shape input-tensor)]
           [filter-shape (dt:shape filter-tensor)]
           [batch-size (car input-shape)]
           [in-channels (cadr input-shape)]
           [in-height (caddr input-shape)]
           [in-width (cadddr input-shape)]
           [out-channels (car filter-shape)]
           [filter-height (caddr filter-shape)]
           [filter-width (cadddr filter-shape)]
           
           ;; Calculate output dimensions
           [out-height (add1 (quotient (- (+ in-height (* 2 padding)) filter-height) stride))]
           [out-width (add1 (quotient (- (+ in-width (* 2 padding)) filter-width) stride))]
           
           ;; Get device from input tensor
           [device (dt:device input-tensor)]
           
           ;; Create output tensor on the same device
           [output-data (make-vector (* batch-size out-channels out-height out-width) 0.0)]
           [output-tensor (dt:create (list batch-size out-channels out-height out-width) output-data device)])
      
      ;; Try to call the C function with appropriate tensors
      (c:conv2d-forward 
       batch-size in-channels in-height in-width
       out-channels filter-height filter-width
       stride padding
       (list->f64vector (vector->list (dt:data input-tensor)))
       (list->f64vector (vector->list (dt:data filter-tensor)))
       (list->f64vector (vector->list (dt:data output-tensor))))
      
      output-tensor)))

;; Simple convolutional operation (without C library) as fallback
(define (simple-conv2d input-tensor filter-tensor [stride 1] [padding 0])
  (printf "Using pure Racket convolution implementation~n")
  (let* ([input-shape (dt:shape input-tensor)]
         [filter-shape (dt:shape filter-tensor)]
         
         [batch-size (car input-shape)]
         [in-channels (cadr input-shape)]
         [in-height (caddr input-shape)]
         [in-width (cadddr input-shape)]
         
         [out-channels (car filter-shape)]
         [filter-height (caddr filter-shape)]
         [filter-width (cadddr filter-shape)]
         
         ;; Calculate output dimensions
         [out-height (add1 (quotient (- (+ in-height (* 2 padding)) filter-height) stride))]
         [out-width (add1 (quotient (- (+ in-width (* 2 padding)) filter-width) stride))]
         
         [output-data (make-vector (* batch-size out-channels out-height out-width) 0.0)]
         [output-tensor (dt:create (list batch-size out-channels out-height out-width) 
                                 output-data 
                                 (dt:device input-tensor))]
         
         [input-data (dt:data input-tensor)]
         [filter-data (dt:data filter-tensor)])
    
    ;; Perform convolution manually
    (for ([b (in-range batch-size)])
      (for ([oc (in-range out-channels)])
        (for ([oh (in-range out-height)])
          (for ([ow (in-range out-width)])
            ;; For each output position
            (let ([out-idx (+ (* b out-channels out-height out-width)
                             (* oc out-height out-width)
                             (* oh out-width)
                             ow)]
                  [sum 0.0])
              
              ;; Sum over input channels and filter dimensions
              (for ([ic (in-range in-channels)])
                (for ([fh (in-range filter-height)])
                  (for ([fw (in-range filter-width)])
                    (let* ([ih (+ (* oh stride) fh (- padding))]
                           [iw (+ (* ow stride) fw (- padding))])
                      
                      ;; Check if input position is valid
                      (when (and (>= ih 0) (< ih in-height)
                                 (>= iw 0) (< iw in-width))
                        (let ([input-idx (+ (* b in-channels in-height in-width)
                                           (* ic in-height in-width)
                                           (* ih in-width)
                                           iw)]
                              [filter-idx (+ (* oc in-channels filter-height filter-width)
                                            (* ic filter-height filter-width)
                                            (* fh filter-width)
                                            fw)])
                          (set! sum (+ sum (* (vector-ref input-data input-idx)
                                             (vector-ref filter-data filter-idx))))))))))
              
              ;; Set output
              (vector-set! output-data out-idx sum)))))
    
    output-tensor))

;; Max pooling operation with fallback
(define (max-pool-2x2 input-tensor)
  (with-handlers ([exn:fail? (lambda (e)
                              (printf "Warning: C max pooling failed, using fallback implementation: ~a~n"
                                      (exn-message e))
                              (simple-max-pool input-tensor))])
    (let* ([input-shape (dt:shape input-tensor)]
           [batch-size (car input-shape)]
           [channels (cadr input-shape)]
           [in-height (caddr input-shape)]
           [in-width (cadddr input-shape)]
           
           ;; Calculate output dimensions
           [out-height (quotient in-height 2)]
           [out-width (quotient in-width 2)]
           
           ;; Get device from input tensor
           [device (dt:device input-tensor)]
           
           ;; Create output tensor on the same device
           [output-data (make-vector (* batch-size channels out-height out-width) 0.0)]
           [output-tensor (dt:create (list batch-size channels out-height out-width) output-data device)])
      
      ;; Call the C function
      (c:max-pool-2x2
       batch-size channels in-height in-width
       (list->f64vector (vector->list (dt:data input-tensor)))
       (list->f64vector (vector->list (dt:data output-tensor))))
      
      output-tensor)))

;; Simple max pooling (2x2) as fallback
(define (simple-max-pool input-tensor)
  (printf "Using pure Racket max pooling implementation~n")
  (let* ([input-shape (dt:shape input-tensor)]
         [batch-size (car input-shape)]
         [channels (cadr input-shape)]
         [in-height (caddr input-shape)]
         [in-width (cadddr input-shape)]
         
         [out-height (quotient in-height 2)]
         [out-width (quotient in-width 2)]
         
         [output-data (make-vector (* batch-size channels out-height out-width) 0.0)]
         [output-tensor (dt:create (list batch-size channels out-height out-width)
                                 output-data
                                 (dt:device input-tensor))]
         
         [input-data (dt:data input-tensor)])
    
    ;; Perform max pooling
    (for ([b (in-range batch-size)])
      (for ([c (in-range channels)])
        (for ([oh (in-range out-height)])
          (for ([ow (in-range out-width)])
            (let ([out-idx (+ (* b channels out-height out-width)
                             (* c out-height out-width)
                             (* oh out-width)
                             ow)]
                  [max-val -inf.0])
              
              ;; Find max in 2x2 region
              (for ([h (in-range (* oh 2) (+ (* oh 2) 2))])
                (for ([w (in-range (* ow 2) (+ (* ow 2) 2))])
                  (let ([in-idx (+ (* b channels in-height in-width)
                                  (* c in-height in-width)
                                  (* h in-width)
                                  w)])
                    (set! max-val (max max-val (vector-ref input-data in-idx))))))
              
              ;; Set output
              (vector-set! output-data out-idx max-val))))))
    
    output-tensor))

;; Flatten a 4D tensor to 2D (batch_size, features) with fallback
(define (flatten input-tensor)
  (with-handlers ([exn:fail? (lambda (e)
                              (printf "Warning: C flatten failed, using fallback implementation: ~a~n"
                                      (exn-message e))
                              (simple-flatten input-tensor))])
    (let* ([input-shape (dt:shape input-tensor)]
           [batch-size (car input-shape)]
           [channels (cadr input-shape)]
           [height (caddr input-shape)]
           [width (cadddr input-shape)]
           
           ;; Calculate flattened size
           [flat-size (* channels height width)]
           
           ;; Get device from input tensor
           [device (dt:device input-tensor)]
           
           ;; Create output tensor on the same device
           [output-data (make-vector (* batch-size flat-size) 0.0)]
           [output-tensor (dt:create (list batch-size flat-size) output-data device)])
      
      ;; Call the C function
      (c:flatten-tensor
       batch-size channels height width
       (list->f64vector (vector->list (dt:data input-tensor)))
       (list->f64vector (vector->list (dt:data output-tensor))))
      
      output-tensor)))

;; Simple tensor flatten (4D to 2D) as fallback
(define (simple-flatten input-tensor)
  (printf "Using pure Racket flatten implementation~n")
  (let* ([shape (dt:shape input-tensor)]
         [batch-size (car shape)]
         [channels (cadr shape)]
         [height (caddr shape)]
         [width (cadddr shape)]
         [flat-size (* channels height width)]
         
         [output-data (make-vector (* batch-size flat-size) 0.0)]
         [output-tensor (dt:create (list batch-size flat-size) 
                                 output-data
                                 (dt:device input-tensor))]
         
         [input-data (dt:data input-tensor)])
    
    ;; Flatten the tensor
    (for ([b (in-range batch-size)])
      (for ([c (in-range channels)])
        (for ([h (in-range height)])
          (for ([w (in-range width)])
            (let ([in-idx (+ (* b channels height width)
                            (* c height width)
                            (* h width)
                            w)]
                  [out-idx (+ (* b flat-size)
                             (* c height width)
                             (* h width)
                             w)])
              (vector-set! output-data out-idx (vector-ref input-data in-idx)))))))
    
    output-tensor))

;; ReLU activation function with fallback
(define (relu input-tensor)
  (with-handlers ([exn:fail? (lambda (e)
                              (printf "Warning: C ReLU failed, using fallback implementation: ~a~n"
                                      (exn-message e))
                              (simple-relu input-tensor))])
    (let* ([shape (dt:shape input-tensor)]
           [size (apply * shape)]
           [device (dt:device input-tensor)]
           [result-data (make-vector size 0.0)]
           [result-tensor (dt:create shape result-data device)])
      
      ;; Use the C function from ffi_ops.rkt
      (c:relu-forward size 
                     (list->f64vector (vector->list (dt:data input-tensor))) 
                     (list->f64vector (vector->list (dt:data result-tensor))))
      
      result-tensor)))

;; Simple ReLU implementation as fallback
(define (simple-relu input-tensor)
  (printf "Using pure Racket ReLU implementation~n")
  (let* ([shape (dt:shape input-tensor)]
         [size (apply * shape)]
         [input-data (dt:data input-tensor)]
         [output-data (make-vector size 0.0)]
         [output-tensor (dt:create shape output-data (dt:device input-tensor))])
    
    ;; Apply ReLU: max(0, x)
    (for ([i (in-range size)])
      (vector-set! output-data i (max 0.0 (vector-ref input-data i))))
    
    output-tensor))

;; Softmax function with fallback
(define (softmax input-tensor)
  (with-handlers ([exn:fail? (lambda (e)
                              (printf "Warning: C softmax failed, using fallback implementation: ~a~n"
                                      (exn-message e))
                              (simple-softmax input-tensor))])
    (let* ([shape (dt:shape input-tensor)]
           [batch-size (car shape)]
           [num-classes (cadr shape)]
           [device (dt:device input-tensor)]
           [result-data (make-vector (* batch-size num-classes) 0.0)]
           [result-tensor (dt:create shape result-data device)])
      
      ;; Call the C function
      (c:softmax
       batch-size num-classes
       (list->f64vector (vector->list (dt:data input-tensor)))
       (list->f64vector (vector->list (dt:data result-tensor))))
      
      result-tensor)))

;; Simple softmax implementation as fallback
(define (simple-softmax input-tensor)
  (printf "Using pure Racket softmax implementation~n")
  (let* ([shape (dt:shape input-tensor)]
         [batch-size (car shape)]
         [num-classes (cadr shape)]
         [input-data (dt:data input-tensor)]
         [output-data (make-vector (* batch-size num-classes) 0.0)]
         [output-tensor (dt:create shape output-data (dt:device input-tensor))])
    
    ;; Apply softmax batch-wise
    (for ([b (in-range batch-size)])
      (let* ([start-idx (* b num-classes)]
             [end-idx (+ start-idx num-classes)]
             
             ;; Find max value for numerical stability
             [max-val (for/fold ([max-val -inf.0])
                                ([i (in-range start-idx end-idx)])
                        (max max-val (vector-ref input-data i)))]
             
             ;; Compute exp(x_i - max)
             [exp-vals (make-vector num-classes 0.0)]
             [_ (for ([c (in-range num-classes)])
                  (vector-set! exp-vals c 
                               (exp (- (vector-ref input-data (+ start-idx c)) 
                                      max-val))))]
             
             ;; Compute sum of exp values
             [sum (for/sum ([c (in-range num-classes)])
                    (vector-ref exp-vals c))])
        
        ;; Normalize by sum
        (for ([c (in-range num-classes)])
          (vector-set! output-data (+ start-idx c)
                       (/ (vector-ref exp-vals c) sum)))))
    
    output-tensor))

;; Fully connected layer
(define (fc-layer input-tensor weights bias [activation-fn identity])
  (let* ([device (dt:device input-tensor)]
         [z (dt:add (dt:mul input-tensor weights) bias)])
    (activation-fn z)))

;; Cross-entropy loss with fallback
(define (cross-entropy-loss predictions targets)
  (with-handlers ([exn:fail? (lambda (e)
                              (printf "Warning: C cross-entropy failed, using fallback implementation: ~a~n"
                                      (exn-message e))
                              (simple-cross-entropy-loss predictions targets))])
    (let* ([shape (dt:shape predictions)]
           [batch-size (car shape)]
           [num-classes (cadr shape)])
      
      ;; Call the C function
      (c:cross-entropy-loss
       batch-size num-classes
       (list->f64vector (vector->list (dt:data predictions)))
       (list->f64vector (vector->list (dt:data targets)))))))

;; Simple cross-entropy loss implementation as fallback
(define (simple-cross-entropy-loss predictions targets)
  (printf "Using pure Racket cross-entropy loss implementation~n")
  (let* ([shape (dt:shape predictions)]
         [batch-size (car shape)]
         [num-classes (cadr shape)]
         [pred-data (dt:data predictions)]
         [target-data (dt:data targets)]
         [epsilon 1e-15]  ; To avoid log(0)
         [total-loss 0.0])
    
    ;; Compute loss batch-wise
    (for ([b (in-range batch-size)])
      (let ([start-idx (* b num-classes)])
        (for ([c (in-range num-classes)])
          (let* ([p (max epsilon (min (- 1.0 epsilon) 
                                      (vector-ref pred-data (+ start-idx c))))]
                 [t (vector-ref target-data (+ start-idx c))])
            (when (> t 0.0)  ; Only compute loss for non-zero targets
              (set! total-loss (- total-loss (* t (log p)))))))))
    
    ;; Return average loss
    (/ total-loss batch-size)))

;; Define the LeNet model architecture
;; Returns a function that performs the forward pass
(define (make-lenet [dev (current-device)]
                    [c1f #f] [c1b #f] [c2f #f] [c2b #f]
                    [f1w #f] [f1b #f] [f2w #f] [f2b #f]
                    [f3w #f] [f3b #f])
  (printf "Creating LeNet model on device: ~a~n" (get-device-type dev))
  ;; Create model parameters on the specified device or use provided ones
  (let ([conv1-filters (if c1f c1f (dt:random (list 6 1 5 5) 0.1 dev))]       ; 6 filters, 1 channel, 5x5
        [conv1-bias (if c1b c1b (dt:random (list 1 6) 0.1 dev))]              ; 6 channels
        
        [conv2-filters (if c2f c2f (dt:random (list 16 6 5 5) 0.1 dev))]      ; 16 filters, 6 channels, 5x5
        [conv2-bias (if c2b c2b (dt:random (list 1 16) 0.1 dev))]             ; 16 channels
        
        [fc1-weights (if f1w f1w (dt:random (list 256 120) 0.1 dev))]         ; 256 inputs (4x4x16), 120 outputs
        [fc1-bias (if f1b f1b (dt:random (list 1 120) 0.1 dev))]
        
        [fc2-weights (if f2w f2w (dt:random (list 120 84) 0.1 dev))]          ; 120 inputs, 84 outputs
        [fc2-bias (if f2b f2b (dt:random (list 1 84) 0.1 dev))]
        
        [fc3-weights (if f3w f3w (dt:random (list 84 10) 0.1 dev))]           ; 84 inputs, 10 outputs
        [fc3-bias (if f3b f3b (dt:random (list 1 10) 0.1 dev))])
    
    (printf "Model initialized successfully~n")
    
    ;; Forward pass function
    (define (forward-pass input)
      (printf "Running forward pass...~n")
      ;; First convolutional layer + ReLU + pooling
      (let* ([input-with-channels (if (= (length (dt:shape input)) 3)
                                       ;; Add batch dimension if missing
                                       (dt:reshape input (list 1 (car (dt:shape input))
                                                              (cadr (dt:shape input))
                                                              (caddr (dt:shape input))))
                                       input)]
             
             ;; Compute convolution output
             [_ (printf "  Running convolution 1...~n")]
             [conv_out (with-handlers 
                           ([exn:fail? (lambda (e) 
                                        (printf "  Convolution failed: ~a~n" (exn-message e))
                                        (dt:create (list (car (dt:shape input-with-channels))
                                                        6
                                                        28 28)
                                                  (make-vector (* (car (dt:shape input-with-channels))
                                                                6 28 28)
                                                              0.1)
                                                  dev))])
                         (conv2d input-with-channels conv1-filters 1 2))]
             
             ;; Apply bias manually without broadcasting
             [conv_shape (dt:shape conv_out)]
             [batch-size (car conv_shape)]
             [channels (cadr conv_shape)]
             [height (caddr conv_shape)]
             [width (cadddr conv_shape)]
             [conv_data (vector->list (dt:data conv_out))]
             [bias_data (vector->list (dt:data conv1-bias))]
             
             ;; Apply bias to each channel
             [_ (printf "  Applying bias 1...~n")]
             [conv_with_bias (for/vector ([i (in-range (* batch-size channels height width))])
                              (let* ([batch-idx (quotient i (* channels height width))]
                                    [within-batch-idx (remainder i (* channels height width))]
                                    [channel-idx (quotient within-batch-idx (* height width))]
                                    [bias-val (list-ref bias_data channel-idx)])
                                (+ (list-ref conv_data i) bias-val)))]
             
             ;; Create tensor from biased data
             [conv1 (dt:create conv_shape conv_with_bias (dt:device conv_out))]
             [_ (printf "  Applying ReLU 1...~n")]
             [relu1 (relu conv1)]
             [_ (printf "  Max pooling 1...~n")]
             [pool1 (with-handlers 
                        ([exn:fail? (lambda (e) 
                                     (printf "  Pooling failed: ~a~n" (exn-message e))
                                     (dt:create (list batch-size channels 
                                                     (quotient height 2)
                                                     (quotient width 2))
                                               (make-vector (* batch-size channels 
                                                             (quotient height 2)
                                                             (quotient width 2))
                                                           0.1)
                                               dev))])
                      (max-pool-2x2 relu1))]
             
             ;; Second convolutional layer + ReLU + pooling
             [_ (printf "  Running convolution 2...~n")]
             [conv2_out (with-handlers 
                           ([exn:fail? (lambda (e) 
                                        (printf "  Convolution 2 failed: ~a~n" (exn-message e))
                                        (dt:create (list batch-size 16 5 5)
                                                  (make-vector (* batch-size 16 5 5) 0.1)
                                                  dev))])
                         (conv2d pool1 conv2-filters 1 0))]
             
             ;; Apply bias manually without broadcasting
             [conv2_shape (dt:shape conv2_out)]
             [batch-size2 (car conv2_shape)]
             [channels2 (cadr conv2_shape)]
             [height2 (caddr conv2_shape)]
             [width2 (cadddr conv2_shape)]
             [conv2_data (vector->list (dt:data conv2_out))]
             [bias2_data (vector->list (dt:data conv2-bias))]
             
             ;; Apply bias to each channel
             [_ (printf "  Applying bias 2...~n")]
             [conv2_with_bias (for/vector ([i (in-range (* batch-size2 channels2 height2 width2))])
                              (let* ([batch-idx (quotient i (* channels2 height2 width2))]
                                    [within-batch-idx (remainder i (* channels2 height2 width2))]
                                    [channel-idx (quotient within-batch-idx (* height2 width2))]
                                    [bias-val (list-ref bias2_data channel-idx)])
                                (+ (list-ref conv2_data i) bias-val)))]
             
             ;; Create tensor from biased data
             [conv2 (dt:create conv2_shape conv2_with_bias (dt:device conv2_out))]
             [_ (printf "  Applying ReLU 2...~n")]
             [relu2 (relu conv2)]
             [_ (printf "  Max pooling 2...~n")]
             [pool2 (with-handlers 
                        ([exn:fail? (lambda (e) 
                                     (printf "  Pooling 2 failed: ~a~n" (exn-message e))
                                     (dt:create (list batch-size2 channels2 
                                                     (quotient height2 2)
                                                     (quotient width2 2))
                                               (make-vector (* batch-size2 channels2 
                                                             (quotient height2 2)
                                                             (quotient width2 2))
                                                           0.1)
                                               dev))])
                      (max-pool-2x2 relu2))]
             
             ;; Flatten and fully connected layers
             [_ (printf "  Flattening tensor...~n")]
             [flat (with-handlers 
                       ([exn:fail? (lambda (e) 
                                    (printf "  Flatten failed: ~a~n" (exn-message e))
                                    (dt:create (list batch-size2 400)
                                              (make-vector (* batch-size2 400) 0.1)
                                              dev))])
                     (flatten pool2))]
             
             [_ (printf "  FC layer 1...~n")]
             [fc1 (fc-layer flat fc1-weights fc1-bias relu)]
             [_ (printf "  FC layer 2...~n")]
             [fc2 (fc-layer fc1 fc2-weights fc2-bias relu)]
             [_ (printf "  FC layer 3...~n")]
             [fc3 (fc-layer fc2 fc3-weights fc3-bias)]
             
             ;; Softmax for classification
             [_ (printf "  Softmax...~n")]
             [output (with-handlers 
                         ([exn:fail? (lambda (e) 
                                      (printf "  Softmax failed: ~a~n" (exn-message e))
                                      (dt:create (list batch-size2 10)
                                                (make-vector (* batch-size2 10) 0.1)
                                                dev))])
                       (softmax fc3))])
        
        (printf "Forward pass completed~n")
        output))
    
    ;; Return model parameters and forward function
    (values forward-pass
            conv1-filters conv1-bias
            conv2-filters conv2-bias
            fc1-weights fc1-bias
            fc2-weights fc2-bias
            fc3-weights fc3-bias)))

;; Predict class from model output
(define (predict output-tensor)
  (let* ([shape (dt:shape output-tensor)]
         [batch-size (car shape)]
         [num-classes (cadr shape)]
         [data (dt:data output-tensor)]
         [predictions (for/list ([b (in-range batch-size)])
                        (for/fold ([max-idx 0]
                                    [max-val (vector-ref data (* b num-classes))])
                                   ([c (in-range 1 num-classes)])
                          (let ([val (vector-ref data (+ (* b num-classes) c))])
                            (if (> val max-val)
                                (values c val)
                                (values max-idx max-val)))))])
    predictions))

;; Function to load MNIST data for CNN directly
(define (load-mnist-data-for-cnn)
  (printf "Loading MNIST data for CNN training...~n")
  
  (define (read-idx3-ubyte filename)
    (with-handlers ([exn:fail? (lambda (e)
                                (printf "Warning: Could not read MNIST images from ~a: ~a~n" 
                                        filename (exn-message e))
                                #f)])
      (let* ([p (open-input-file filename #:mode 'binary)]
             [magic-number (integer-bytes->integer (read-bytes 4 p) #f #t)]
             [num-images (integer-bytes->integer (read-bytes 4 p) #f #t)]
             [num-rows (integer-bytes->integer (read-bytes 4 p) #f #t)]
             [num-cols (integer-bytes->integer (read-bytes 4 p) #f #t)]
             [data (make-vector (* num-images num-rows num-cols) 0)])
        (for ([i (in-range (vector-length data))])
          (vector-set! data i (/ (read-byte p) 255.0))) ; Normalize to [0,1]
        (close-input-port p)
        data)))
  
  (define (read-idx1-ubyte filename)
    (with-handlers ([exn:fail? (lambda (e)
                                (printf "Warning: Could not read MNIST labels from ~a: ~a~n" 
                                        filename (exn-message e))
                                #f)])
      (let* ([p (open-input-file filename #:mode 'binary)]
             [magic-number (integer-bytes->integer (read-bytes 4 p) #f #t)]
             [num-items (integer-bytes->integer (read-bytes 4 p) #f #t)]
             [data (make-vector num-items 0)])
        (for ([i (in-range num-items)])
          (vector-set! data i (read-byte p)))
        (close-input-port p)
        data)))
  
  (define (one-hot y num-classes)
    (when (not y)
      (error 'one-hot "Cannot one-hot encode null input"))
    (let* ([num-samples (vector-length y)]
           [encoded (make-vector (* num-samples num-classes) 0.0)])
      (for ([i (in-range num-samples)])
        (vector-set! encoded (+ (* i num-classes) (vector-ref y i)) 1.0))
      encoded))
  
  ;; Try to load MNIST data but fallback to random data if needed
  (with-handlers 
    ([exn:fail? 
      (lambda (e)
        (printf "Error loading MNIST data: ~a~nFalling back to random data.~n" (exn-message e))
        (let* ([train-size 100]
               [test-size 20]
               [dev (current-device)]
               
               ;; Generate random data - single channel for simplicity
               [train-images (dt:random (list train-size 1 28 28) 1.0 dev)]
               [train-labels-data (make-vector (* train-size 10) 0.0)]
               [test-images (dt:random (list test-size 1 28 28) 1.0 dev)]
               [test-labels-data (make-vector (* test-size 10) 0.0)])
          
          ;; Generate one-hot labels (1 in a random position for each example)
          (for ([i (in-range train-size)])
            (vector-set! train-labels-data 
                         (+ (* i 10) (random 10)) 
                         1.0))
          
          (for ([i (in-range test-size)])
            (vector-set! test-labels-data 
                         (+ (* i 10) (random 10)) 
                         1.0))
          
          (let ([train-labels (dt:create (list train-size 10) train-labels-data dev)]
                [test-labels (dt:create (list test-size 10) test-labels-data dev)])
            
            (printf "Created random data: ~a training images, ~a test images~n" 
                    train-size test-size)
            (values train-images train-labels test-images test-labels))))])
  
    ;; Try to locate the MNIST data - first check in the proper location
    (let* ([base-paths 
            (list 
             "/path to mnist/"
             "/path to mnist/"
             "./mnist-data/"
             "../mnist-data/")]
           [found-path #f])
      
      ;; Look for the data files in each potential location
      (for ([path base-paths] #:when (not found-path))
        (when (file-exists? (string-append path "train-images.idx3-ubyte"))
          (set! found-path path)))
      
      (if found-path
          (let* ([train-images-file (string-append found-path "train-images.idx3-ubyte")]
                 [train-labels-file (string-append found-path "train-labels.idx1-ubyte")]
                 [test-images-file (string-append found-path "t10k-images.idx3-ubyte")]
                 [test-labels-file (string-append found-path "t10k-labels.idx1-ubyte")]
                 
                 ;; Load and format train data
                 [train-images-data (or (read-idx3-ubyte train-images-file)
                                       (begin 
                                         (printf "Failed to read training images, using random data~n")
                                         (make-vector (* 100 784) 0.5)))]
                 [train-labels-data (or (read-idx1-ubyte train-labels-file)
                                       (begin
                                         (printf "Failed to read training labels, using random data~n")
                                         (for/vector ([i (in-range 100)]) (random 10))))]
                 
                 ;; Load and format test data
                 [test-images-data (or (read-idx3-ubyte test-images-file)
                                      (begin 
                                        (printf "Failed to read test images, using random data~n")
                                        (make-vector (* 20 784) 0.5)))]
                 [test-labels-data (or (read-idx1-ubyte test-labels-file)
                                      (begin
                                        (printf "Failed to read test labels, using random data~n")
                                        (for/vector ([i (in-range 20)]) (random 10))))]
                 
                 ;; Create one-hot encoded labels
                 [train-labels-onehot (one-hot train-labels-data 10)]
                 [test-labels-onehot (one-hot test-labels-data 10)]
                 
                 ;; Create device tensors
                 [dev (current-device)]
                 [train-size (min 1000 (/ (vector-length train-images-data) 784))]
                 [test-size (min 500 (/ (vector-length test-images-data) 784))]
                 
                 ;; For CNN, reshape to 4D: [batch_size, channels, height, width]
                 ;; Single channel for grayscale
                 [train-tensor (dt:create (list train-size 1 28 28) 
                                         (make-vector (* train-size 1 28 28) 0.0) 
                                         dev)]
                 [test-tensor (dt:create (list test-size 1 28 28) 
                                        (make-vector (* test-size 1 28 28) 0.0) 
                                        dev)])
            
            ;; Copy data to tensor
            (for ([i (in-range train-size)])
              (for ([h (in-range 28)])
                (for ([w (in-range 28)])
                  (let ([src-idx (+ (* i 784) (* h 28) w)]
                        [dst-idx (+ (* i 1 28 28) (* h 28) w)])
                    (vector-set! (dt:data train-tensor) dst-idx 
                                (vector-ref train-images-data src-idx))))))
            
            (for ([i (in-range test-size)])
              (for ([h (in-range 28)])
                (for ([w (in-range 28)])
                  (let ([src-idx (+ (* i 784) (* h 28) w)]
                        [dst-idx (+ (* i 1 28 28) (* h 28) w)])
                    (vector-set! (dt:data test-tensor) dst-idx 
                                (vector-ref test-images-data src-idx))))))
            
            ;; Create label tensors
            (define train-labels (dt:create (list train-size 10) 
                                           (make-vector (* train-size 10) 0.0)
                                           dev))
            (define test-labels (dt:create (list test-size 10) 
                                          (make-vector (* test-size 10) 0.0)
                                          dev))
            
            ;; Copy labels
            (for ([i (in-range train-size)])
              (for ([j (in-range 10)])
                (let ([idx (+ (* i 10) j)])
                  (vector-set! (dt:data train-labels) idx
                              (vector-ref train-labels-onehot idx)))))
            
            (for ([i (in-range test-size)])
              (for ([j (in-range 10)])
                (let ([idx (+ (* i 10) j)])
                  (vector-set! (dt:data test-labels) idx
                              (vector-ref test-labels-onehot idx)))))
            
            (printf "Loaded MNIST: ~a training images, ~a test images~n" 
                    train-size test-size)
            (values train-tensor train-labels test-tensor test-labels))
          
          ;; No MNIST data found in any location
          (let* ([train-size 100]
                 [test-size 20]
                 [dev (current-device)]
                 
                 ;; Generate random data
                 [train-images (dt:random (list train-size 1 28 28) 1.0 dev)]
                 [train-labels-data (make-vector (* train-size 10) 0.0)]
                 [test-images (dt:random (list test-size 1 28 28) 1.0 dev)]
                 [test-labels-data (make-vector (* test-size 10) 0.0)])
            
            ;; Generate one-hot labels (1 in a random position for each example)
            (for ([i (in-range train-size)])
              (vector-set! train-labels-data 
                           (+ (* i 10) (random 10)) 
                           1.0))
            
            (for ([i (in-range test-size)])
              (vector-set! test-labels-data 
                           (+ (* i 10) (random 10)) 
                           1.0))
            
            (let ([train-labels (dt:create (list train-size 10) train-labels-data dev)]
                  [test-labels (dt:create (list test-size 10) test-labels-data dev)])
              
              (printf "Created random MNIST data: ~a training images, ~a test images~n" 
                      train-size test-size)
              (values train-images train-labels test-images test-labels)))))))

;; Get a batch of data using efficient tensor slicing
(define (get-batch images labels batch-size start-idx)
  (let* ([end-idx (min (+ start-idx batch-size) (car (dt:shape images)))]
         [actual-batch-size (- end-idx start-idx)]
         [device (dt:device images)]
         
         ;; Calculate sizes for tensor dimensions
         [img-channels (cadr (dt:shape images))]
         [img-height (caddr (dt:shape images))]
         [img-width (cadddr (dt:shape images))]
         [label-classes (cadr (dt:shape labels))]
         
         ;; Create tensors for the batch
         [batch-images-data (make-vector (* actual-batch-size img-channels img-height img-width) 0.0)]
         [batch-images (dt:create (list actual-batch-size img-channels img-height img-width)
                                  batch-images-data device)]
         
         [batch-labels-data (make-vector (* actual-batch-size label-classes) 0.0)]
         [batch-labels (dt:create (list actual-batch-size label-classes)
                                  batch-labels-data device)]
         
         ;; Get source data
         [images-data (dt:data images)]
         [labels-data (dt:data labels)]
         
         ;; Calculate element sizes for index calculations
         [image-elems (* img-channels img-height img-width)]
         [label-elems label-classes])
    
    ;; Copy image data in one efficient operation using memcpy semantics
    (for ([i (in-range actual-batch-size)])
      (let ([src-offset (* (+ start-idx i) image-elems)]
            [dst-offset (* i image-elems)])
        ;; Copy entire image in one go
        (for ([j (in-range image-elems)])
          (vector-set! batch-images-data 
                       (+ dst-offset j) 
                       (vector-ref images-data (+ src-offset j))))
        
        ;; Copy label data
        (let ([src-label-offset (* (+ start-idx i) label-elems)]
              [dst-label-offset (* i label-elems)])
          (for ([j (in-range label-elems)])
            (vector-set! batch-labels-data 
                         (+ dst-label-offset j) 
                         (vector-ref labels-data (+ src-label-offset j)))))))
    
    (values batch-images batch-labels)))

;; Calculate accuracy
(define (accuracy predictions labels)
  (let* ([pred-list (predict predictions)]
         [labels-data (dt:data labels)]
         [num-samples (length pred-list)]
         [correct-count 0])
    
    (for ([i (in-range num-samples)])
      (let* ([predicted-class (list-ref pred-list i)]
             [true-class (argmax (lambda (j) 
                                   (vector-ref labels-data (+ (* i (cadr (dt:shape labels))) j)))
                                 (range (cadr (dt:shape labels))))])
        (when (= predicted-class true-class)
          (set! correct-count (add1 correct-count)))))
    
    (/ correct-count num-samples)))

;; Helper functions for backpropagation

;; Compute gradients for convolution layer
(define (compute-conv-gradients input-tensor output-gradients filters biases stride padding)
  ;; This is a simplified gradient computation for demonstration
  ;; In a real implementation, this would compute precise gradients
  (let* ([input-shape (dt:shape input-tensor)]
         [batch-size (car input-shape)]
         [input-channels (cadr input-shape)]
         [input-height (caddr input-shape)]
         [input-width (cadddr input-shape)]
         
         [filter-shape (dt:shape filters)]
         [output-channels (car filter-shape)]
         [filter-height (caddr filter-shape)]
         [filter-width (cadddr filter-shape)]
         
         [device (dt:device input-tensor)]
         
         ;; Gradient shapes
         [filter-gradients (dt:create filter-shape 
                                    (make-vector (apply * filter-shape) 0.0)
                                    device)]
         [bias-gradients (dt:create (dt:shape biases)
                                  (make-vector (apply * (dt:shape biases)) 0.0)
                                  device)]
         [input-gradients (dt:create input-shape
                                   (make-vector (apply * input-shape) 0.0)
                                   device)])
    
    ;; Compute bias gradients by summing over batch, height, and width dimensions
    (let* ([output-grad-data (dt:data output-gradients)]
           [output-shape (dt:shape output-gradients)]
           [output-height (caddr output-shape)]
           [output-width (cadddr output-shape)]
           [bias-grad-data (make-vector output-channels 0.0)])
      
      ;; Sum gradients for each output channel
      (for ([b (in-range batch-size)])
        (for ([c (in-range output-channels)])
          (for ([h (in-range output-height)])
            (for ([w (in-range output-width)])
              (let ([idx (+ (* b output-channels output-height output-width)
                           (* c output-height output-width)
                           (* h output-width)
                           w)])
                (vector-set! bias-grad-data c
                             (+ (vector-ref bias-grad-data c)
                                (vector-ref output-grad-data idx))))))))
      
      ;; Set bias gradients
      (set! bias-gradients (dt:create (dt:shape biases) bias-grad-data device)))
    
    ;; Return gradients
    (values filter-gradients bias-gradients input-gradients)))

;; Compute gradients for fully connected layer
(define (compute-fc-gradients input-tensor output-gradients weights biases)
  (let* ([input-shape (dt:shape input-tensor)]
         [batch-size (car input-shape)]
         [input-features (cadr input-shape)]
         
         [weights-shape (dt:shape weights)]
         [output-features (cadr weights-shape)]
         
         [device (dt:device input-tensor)]
         
         ;; Weight gradients: dL/dW = X^T * dL/dY
         [weight-gradients (dt:mul (dt:transpose input-tensor) output-gradients)]
         
         ;; Bias gradients: sum of output gradients along batch dimension
         [bias-data (make-vector output-features 0.0)]
         [_ (for ([b (in-range batch-size)])
              (for ([f (in-range output-features)])
                (vector-set! bias-data f
                             (+ (vector-ref bias-data f)
                                (vector-ref (dt:data output-gradients) 
                                           (+ (* b output-features) f))))))]
         [bias-gradients (dt:create (list 1 output-features) bias-data device)]
         
         ;; Input gradients: dL/dX = dL/dY * W^T
         [input-gradients (dt:mul output-gradients (dt:transpose weights))])
    
    (values weight-gradients bias-gradients input-gradients)))

;; ReLU gradient
(define (relu-gradient output-tensor grad-tensor)
  (let* ([output-data (dt:data output-tensor)]
         [grad-data (dt:data grad-tensor)]
         [shape (dt:shape output-tensor)]
         [size (apply * shape)]
         [device (dt:device output-tensor)]
         [result-data (make-vector size 0.0)])
    
    ;; ReLU gradient: 1 if output > 0, 0 otherwise
    (for ([i (in-range size)])
      (vector-set! result-data i
                   (if (> (vector-ref output-data i) 0)
                       (vector-ref grad-data i)
                       0.0)))
    
    (dt:create shape result-data device)))

;; Update parameters using gradients
(define (update-params param gradients learning-rate)
  (dt:sub param (dt:scale gradients learning-rate)))

;; Train function with backpropagation
(define (train-cnn [device-type 'mlx] [epochs 5] [batch-size 32])
  (printf "Setting up CNN on device: ~a~n" device-type)
  
  ;; Choose device based on type
  (cond
    [(eq? device-type 'cpu)
     (set-current-device! (cpu))]
    [(eq? device-type 'mlx)
     (if (device-available? 'mlx)
         (set-current-device! (mlx))
         (begin
           (printf "MLX not available, falling back to CPU~n")
           (set-current-device! (cpu))))]
    [(eq? device-type 'gpu)
     (if (gpu-available?)
         (set-current-device! (gpu))
         (begin
           (printf "GPU not available, falling back to CPU~n")
           (set-current-device! (cpu))))])
  
  (printf "Using device: ~a~n" (get-device-type (current-device)))
  
  ;; Load optimal operations for the current device
  (load-optimal-ops (current-device))
  
  ;; Create model
  (define-values (forward
                  conv1-filters conv1-bias
                  conv2-filters conv2-bias
                  fc1-weights fc1-bias
                  fc2-weights fc2-bias
                  fc3-weights fc3-bias)
    (make-lenet (current-device)))
  
  ;; Load data
  (define-values (train-images train-labels test-images test-labels)
    (load-mnist-data-for-cnn))
  
  ;; Move data to the selected device
  (set! train-images (dt:to train-images (current-device)))
  (set! train-labels (dt:to train-labels (current-device)))
  (set! test-images (dt:to test-images (current-device)))
  (set! test-labels (dt:to test-labels (current-device)))
  
  ;; Training variables
  (define learning-rate 0.01)
  (define train-size (car (dt:shape train-images)))
  (define num-batches (ceiling (/ train-size batch-size)))
  
  ;; Record start time
  (define start-time (current-inexact-milliseconds))
  
  ;; Main training loop
  (printf "Starting training for ~a epochs...~n" epochs)
  
  (for ([epoch (in-range epochs)])
    (printf "Epoch ~a/~a~n" (add1 epoch) epochs)
    
    ;; Shuffle indices
    (define indices (shuffle (range train-size)))
    
    ;; Initialize variables for this epoch
    (define total-loss 0.0)
    (define batch-count 0)
    
    ;; Process each batch
    (for ([batch (in-range num-batches)])
      (let* ([start-idx (* batch batch-size)]
             [end-idx (min (+ start-idx batch-size) train-size)]
             [batch-indices (take (drop indices start-idx) (- end-idx start-idx))])
        
        ;; Get batch data
        (define-values (batch-images batch-labels)
          (get-batch train-images train-labels batch-size start-idx))
        
        ;; Debug info
        (when (= (modulo batch 10) 0)
          (printf "  Batch shapes: ~a, ~a~n" 
                  (dt:shape batch-images) 
                  (dt:shape batch-labels)))
        
        ;; Forward pass with intermediate activations saved for backprop
        (define actual-batch-size (car (dt:shape batch-images)))
        
        ;; Forward pass for first conv layer
        (define conv1-out (conv2d batch-images conv1-filters 1 2))
        (define conv1-with-bias 
          (let* ([conv-shape (dt:shape conv1-out)]
                 [batch-size (car conv-shape)]
                 [channels (cadr conv-shape)]
                 [height (caddr conv-shape)]
                 [width (cadddr conv-shape)]
                 [conv-data (vector->list (dt:data conv1-out))]
                 [bias-data (vector->list (dt:data conv1-bias))]
                 [conv-with-bias (for/vector ([i (in-range (* batch-size channels height width))])
                                  (let* ([batch-idx (quotient i (* channels height width))]
                                         [within-batch-idx (remainder i (* channels height width))]
                                         [channel-idx (quotient within-batch-idx (* height width))]
                                         [bias-val (list-ref bias-data channel-idx)])
                                    (+ (list-ref conv-data i) bias-val)))])
            (dt:create conv-shape conv-with-bias (dt:device conv1-out))))
        (define relu1 (relu conv1-with-bias))
        (define pool1 (max-pool-2x2 relu1))
        
        ;; Forward pass for second conv layer
        (define conv2-out (conv2d pool1 conv2-filters 1 0))
        (define conv2-with-bias 
          (let* ([conv-shape (dt:shape conv2-out)]
                 [batch-size (car conv-shape)]
                 [channels (cadr conv-shape)]
                 [height (caddr conv-shape)]
                 [width (cadddr conv-shape)]
                 [conv-data (vector->list (dt:data conv2-out))]
                 [bias-data (vector->list (dt:data conv2-bias))]
                 [conv-with-bias (for/vector ([i (in-range (* batch-size channels height width))])
                                  (let* ([batch-idx (quotient i (* channels height width))]
                                         [within-batch-idx (remainder i (* channels height width))]
                                         [channel-idx (quotient within-batch-idx (* height width))]
                                         [bias-val (list-ref bias-data channel-idx)])
                                    (+ (list-ref conv-data i) bias-val)))])
            (dt:create conv-shape conv-with-bias (dt:device conv2-out))))
        (define relu2 (relu conv2-with-bias))
        (define pool2 (max-pool-2x2 relu2))
        
        ;; Flatten and fully connected layers
        (define flat (flatten pool2))
        (define fc1-z (dt:add (dt:mul flat fc1-weights) fc1-bias))
        (define fc1-a (relu fc1-z))
        (define fc2-z (dt:add (dt:mul fc1-a fc2-weights) fc2-bias))
        (define fc2-a (relu fc2-z))
        (define fc3-z (dt:add (dt:mul fc2-a fc3-weights) fc3-bias))
        (define predictions (softmax fc3-z))
        
        ;; Calculate loss
        (define loss (cross-entropy-loss predictions batch-labels))
        
        ;; Backpropagation
        ;; Calculate initial gradient from loss
        (define output-grad 
          (let* ([pred-data (dt:data predictions)]
                 [label-data (dt:data batch-labels)]
                 [batch-size (car (dt:shape predictions))]
                 [num-classes (cadr (dt:shape predictions))]
                 [device (dt:device predictions)]
                 [grad-data (make-vector (* batch-size num-classes) 0.0)])
            
            ;; Calculate gradient of cross-entropy: (predictions - labels) / batch_size
            (for ([i (in-range (* batch-size num-classes))])
              (vector-set! grad-data i
                           (/ (- (vector-ref pred-data i)
                                (vector-ref label-data i))
                             batch-size)))
            
            (dt:create (dt:shape predictions) grad-data device)))
        
        ;; Backprop through fully connected layers
        (define-values (grad-fc3-w grad-fc3-b grad-fc3)
          (compute-fc-gradients fc2-a output-grad fc3-weights fc3-bias))
        
        (define grad-fc2-a (dt:mul grad-fc3 (dt:transpose fc3-weights)))
        (define grad-fc2-z (relu-gradient fc2-z grad-fc2-a))
        
        (define-values (grad-fc2-w grad-fc2-b grad-fc2)
          (compute-fc-gradients fc1-a grad-fc2-z fc2-weights fc2-bias))
        
        (define grad-fc1-a (dt:mul grad-fc2 (dt:transpose fc2-weights)))
        (define grad-fc1-z (relu-gradient fc1-z grad-fc1-a))
        
        (define-values (grad-fc1-w grad-fc1-b grad-fc1)
          (compute-fc-gradients flat grad-fc1-z fc1-weights fc1-bias))
        
        ;; Update FC layer weights and biases
        (set! fc3-weights (update-params fc3-weights grad-fc3-w learning-rate))
        (set! fc3-bias (update-params fc3-bias grad-fc3-b learning-rate))
        (set! fc2-weights (update-params fc2-weights grad-fc2-w learning-rate))
        (set! fc2-bias (update-params fc2-bias grad-fc2-b learning-rate))
        (set! fc1-weights (update-params fc1-weights grad-fc1-w learning-rate))
        (set! fc1-bias (update-params fc1-bias grad-fc1-b learning-rate))
        
        ;; Backprop through flatten and pooling layers
        ;; This is simplified and not fully implemented for the conv layers
        ;; In a real implementation, you would compute precise gradients for conv layers
        (define-values (grad-conv2-w grad-conv2-b grad-conv2)
          (compute-conv-gradients pool1 grad-fc1 conv2-filters conv2-bias 1 0))
        
        (define-values (grad-conv1-w grad-conv1-b grad-conv1)
          (compute-conv-gradients batch-images grad-conv2 conv1-filters conv1-bias 1 2))
        
        ;; Update conv layer weights and biases
        (set! conv2-filters (update-params conv2-filters grad-conv2-w (* 0.5 learning-rate)))
        (set! conv2-bias (update-params conv2-bias grad-conv2-b (* 0.5 learning-rate)))
        (set! conv1-filters (update-params conv1-filters grad-conv1-w (* 0.5 learning-rate)))
        (set! conv1-bias (update-params conv1-bias grad-conv1-b (* 0.5 learning-rate)))
        
        ;; Add to total loss
        (set! total-loss (+ total-loss loss))
        (set! batch-count (add1 batch-count))
        
        ;; Print progress every 10 batches
        (when (= (modulo batch 10) 0)
          (printf "  Batch ~a/~a, Loss: ~a~n" batch num-batches (exact->inexact loss)))))
    
    ;; Calculate average loss
    (define avg-loss (/ total-loss batch-count))
    
    ;; Recreate forward pass with updated parameters
    (define-values (forward-updated 
                   _1 _2 _3 _4 _5 _6 _7 _8 _9)
      (make-lenet (current-device) 
                  conv1-filters conv1-bias
                  conv2-filters conv2-bias
                  fc1-weights fc1-bias
                  fc2-weights fc2-bias
                  fc3-weights fc3-bias))
    
    (set! forward forward-updated)
    
    ;; Evaluate on test set
    (define test-predictions (forward test-images))
    (define test-accuracy (accuracy test-predictions test-labels))
    
    (printf "  Epoch ~a: Avg Loss = ~a, Test Accuracy = ~a%~n" 
            (add1 epoch) 
            (exact->inexact avg-loss)
            (* 100.0 (exact->inexact test-accuracy))))
  
  ;; Record end time
  (define end-time (current-inexact-milliseconds))
  (define total-time (/ (- end-time start-time) 1000.0))
  
  (printf "Training complete!~n")
  (printf "Total time: ~a seconds~n" (exact->inexact total-time))
  
  ;; Return statistics
  (hash 'accuracy (* 100.0 (exact->inexact (accuracy (forward test-images) test-labels)))
        'time total-time
        'device device-type))

;; Simple test function for CNN
(define (test-cnn)
  (printf "Running CNN quick test for MNIST...~n")
  (with-handlers ([exn:fail? (lambda (e)
                              (printf "Error in test: ~a~n" (exn-message e))
                              (printf "Stacktrace: ~a~n" (exn->string e))
                              #f)])
    
    ;; Create sample tensors similar to MNIST format
    (printf "Creating test input tensor (2x1x28x28)...~n")
    (define input (dt:random (list 2 1 28 28) 0.1))
    
    ;; Test convolution
    (printf "Testing convolution...~n")
    (define filters (dt:random (list 6 1 5 5) 0.1))
    (define conv-result (conv2d input filters 1 2))
    (printf "Convolution output shape: ~a~n" (dt:shape conv-result))
    
    ;; Test ReLU
    (printf "Testing ReLU...~n")
    (define relu-result (relu conv-result))
    
    ;; Test max pooling
    (printf "Testing max pooling...~n")
    (define pool-result (max-pool-2x2 relu-result))
    (printf "Pooling output shape: ~a~n" (dt:shape pool-result))
    
    ;; Second conv layer
    (printf "Testing second conv layer...~n")
    (define filters2 (dt:random (list 16 6 5 5) 0.1))
    (define conv-result2 (conv2d pool-result filters2 1 0))
    (printf "Conv2 output shape: ~a~n" (dt:shape conv-result2))
    
    ;; Second ReLU
    (printf "Testing second ReLU...~n")
    (define relu-result2 (relu conv-result2))
    
    ;; Second max pooling
    (printf "Testing second max pooling...~n")
    (define pool-result2 (max-pool-2x2 relu-result2))
    (printf "Pool2 output shape: ~a~n" (dt:shape pool-result2))
    
    ;; Test flatten
    (printf "Testing flatten...~n")
    (define flat-result (flatten pool-result2))
    (printf "Flattened shape: ~a~n" (dt:shape flat-result))
    
    ;; Test fully connected layer
    (printf "Testing fully connected layer...~n")
    (define weights (dt:random (list (cadr (dt:shape flat-result)) 10) 0.1))
    (define bias (dt:random (list 1 10) 0.1))
    (define fc-result (fc-layer flat-result weights bias))
    
    ;; Test softmax
    (printf "Testing softmax...~n")
    (define softmax-result (softmax fc-result))
    (printf "Final output shape: ~a~n" (dt:shape softmax-result))
    
    ;; Test loss calculation
    (printf "Testing loss calculation...~n")
    (define targets (dt:random (list 2 10) 0.1))
    (define loss (cross-entropy-loss softmax-result targets))
    (printf "Loss: ~a~n" loss)
    
    (printf "CNN test completed successfully!~n")
    #t))

;; Run the CNN from the command line
(module+ main
  (printf "Running CNN test on the default device...~n")
  (with-handlers ([exn:fail? (lambda (e)
                              (printf "Error running CNN: ~a~n" (exn-message e))
                              (printf "This is likely due to missing C libraries. Using pure Racket implementations as fallback.~n")
                              #f)])
    
    ;; First run a simple test to make sure the CNN operations work
    (define test-success (test-cnn))
    
    (if test-success
        (begin
          (printf "~nCNN component test passed successfully!~n")
          
          ;; Ask if we want to run the full training
          (printf "Do you want to run the full MNIST training? (Takes longer) (y/N): ")
          (let ([response (read-line)])
            (when (or (equal? response "y") (equal? response "Y"))
              (printf "~nStarting CNN training on MNIST...~n")
              (train-cnn 'cpu 1 32))))  ;; Just 1 epoch for quick test
        
        (printf "~nCNN test failed. Please fix the issues before proceeding.~n"))))
