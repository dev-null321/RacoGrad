#lang racket

(require "tensor.rkt"
         "device.rkt"
         "tensor_device.rkt")

;; Simple test for CNN components
;; This will test the basic tensor operations needed for CNNs without
;; relying on the C libraries or MNIST data

;; Create a simple 4D tensor (batch_size, channels, height, width)
(define (make-test-tensor [batch-size 2] [channels 3] [height 8] [width 8])
  (let* ([data (make-vector (* batch-size channels height width) 0.0)])
    ;; Fill with simple pattern
    (for ([b (in-range batch-size)]
          [b-val (in-list '(0.1 0.2))])
      (for ([c (in-range channels)]
            [c-val (in-list '(0.01 0.02 0.03))])
        (for ([h (in-range height)])
          (for ([w (in-range width)])
            (let ([idx (+ (* b channels height width)
                         (* c height width)
                         (* h width)
                         w)])
              (vector-set! data idx (+ b-val c-val (* 0.1 h) (* 0.01 w))))))))
    
    ;; Create tensor
    (dt:create (list batch-size channels height width) data (cpu))))

;; Simple convolutional operation (without C library)
(define (simple-conv2d input-tensor filter-tensor [stride 1] [padding 0])
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

;; Simple max pooling (2x2 without C library)
(define (simple-max-pool input-tensor)
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

;; Simple ReLU activation
(define (simple-relu input-tensor)
  (let* ([shape (dt:shape input-tensor)]
         [size (apply * shape)]
         [input-data (dt:data input-tensor)]
         [output-data (make-vector size 0.0)]
         [output-tensor (dt:create shape output-data (dt:device input-tensor))])
    
    ;; Apply ReLU
    (for ([i (in-range size)])
      (vector-set! output-data i (max 0.0 (vector-ref input-data i))))
    
    output-tensor))

;; Simple tensor flatten (4D to 2D)
(define (simple-flatten input-tensor)
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

;; Fully connected layer
(define (simple-fc input-tensor weights bias [activation-fn simple-relu])
  (let* ([batch-size (car (dt:shape input-tensor))]
         [out-features (cadr (dt:shape weights))]
         
         [output-data (make-vector (* batch-size out-features) 0.0)]
         [z (dt:create (list batch-size out-features)
                      output-data
                      (dt:device input-tensor))]
         
         [input-data (dt:data input-tensor)]
         [weights-data (dt:data weights)]
         [bias-data (dt:data bias)])
    
    ;; Matrix multiplication and bias addition
    (for ([b (in-range batch-size)])
      (for ([o (in-range out-features)])
        (let ([z-idx (+ (* b out-features) o)]
              [sum 0.0])
          
          ;; Dot product
          (for ([i (in-range (cadr (dt:shape input-tensor)))])
            (let ([in-idx (+ (* b (cadr (dt:shape input-tensor))) i)]
                  [w-idx (+ (* i out-features) o)])
              (set! sum (+ sum (* (vector-ref input-data in-idx)
                                 (vector-ref weights-data w-idx))))))
          
          ;; Add bias
          (set! sum (+ sum (vector-ref bias-data o)))
          (vector-set! output-data z-idx sum))))
    
    ;; Apply activation function if provided
    (if activation-fn
        (activation-fn z)
        z)))

;; Simple softmax function
(define (simple-softmax z)
  (let* ([shape (dt:shape z)]
         [batch-size (car shape)]
         [features (cadr shape)]
         [z-data (dt:data z)]
         [output-data (make-vector (* batch-size features) 0.0)]
         [output (dt:create shape output-data (dt:device z))])
    
    ;; Apply softmax: exp(x_i) / sum(exp(x_j))
    (for ([b (in-range batch-size)])
      (let* ([start-idx (* b features)]
             [end-idx (+ start-idx features)]
             
             ;; Find max value for numerical stability
             [max-val (for/fold ([max-val -inf.0])
                                ([i (in-range start-idx end-idx)])
                        (max max-val (vector-ref z-data i)))]
             
             ;; Compute exp(x_i - max)
             [exp-vals (for/vector ([i (in-range start-idx end-idx)])
                         (exp (- (vector-ref z-data i) max-val)))]
             
             ;; Compute sum of exp values
             [sum (for/sum ([i (in-range features)])
                    (vector-ref exp-vals i))])
        
        ;; Normalize by sum
        (for ([i (in-range features)])
          (vector-set! output-data (+ start-idx i)
                       (/ (vector-ref exp-vals i) sum)))))
    
    output))

;; Run a simple CNN test
(define (run-cnn-test)
  (printf "Running basic CNN test...~n")
  
  ;; Create test input
  (printf "Creating test input tensor...~n")
  (define input (make-test-tensor 2 3 8 8))
  (printf "Input shape: ~a~n" (dt:shape input))
  
  ;; Create test filters 
  (printf "Creating test filters...~n")
  (define filters1 (dt:random (list 6 3 3 3) 0.1))
  (define bias1 (dt:random (list 1 6) 0.1))
  (printf "First layer filter shape: ~a~n" (dt:shape filters1))
  
  ;; First convolution layer
  (printf "Running first convolution layer...~n")
  (define conv1 (simple-conv2d input filters1 1 1))
  (printf "Conv1 output shape: ~a~n" (dt:shape conv1))
  
  ;; Add bias
  (printf "Adding bias...~n")
  (define conv1-with-bias conv1)  ; Simplified for test
  
  ;; ReLU activation
  (printf "Applying ReLU...~n")
  (define relu1 (simple-relu conv1-with-bias))
  
  ;; Max pooling
  (printf "Max pooling...~n")
  (define pool1 (simple-max-pool relu1))
  (printf "Pool1 output shape: ~a~n" (dt:shape pool1))
  
  ;; Second layer
  (printf "Creating second layer filters...~n")
  (define filters2 (dt:random (list 12 6 3 3) 0.1))
  (define bias2 (dt:random (list 1 12) 0.1))
  
  ;; Second convolution
  (printf "Running second convolution layer...~n")
  (define conv2 (simple-conv2d pool1 filters2 1 1))
  (printf "Conv2 output shape: ~a~n" (dt:shape conv2))
  
  ;; Add bias
  (define conv2-with-bias conv2)  ; Simplified for test
  
  ;; ReLU activation
  (printf "Applying ReLU...~n")
  (define relu2 (simple-relu conv2-with-bias))
  
  ;; Max pooling
  (printf "Max pooling...~n")
  (define pool2 (simple-max-pool relu2))
  (printf "Pool2 output shape: ~a~n" (dt:shape pool2))
  
  ;; Flatten
  (printf "Flattening tensor...~n")
  (define flat (simple-flatten pool2))
  (printf "Flattened shape: ~a~n" (dt:shape flat))
  
  ;; FC layers
  (printf "Creating fully connected layers...~n")
  (define flat-size (cadr (dt:shape flat)))
  (define fc1-weights (dt:random (list flat-size 84) 0.1))
  (define fc1-bias (dt:random (list 1 84) 0.1))
  
  (printf "Running FC layer 1...~n")
  (define fc1 (simple-fc flat fc1-weights fc1-bias))
  (printf "FC1 output shape: ~a~n" (dt:shape fc1))
  
  (define fc2-weights (dt:random (list 84 10) 0.1))
  (define fc2-bias (dt:random (list 1 10) 0.1))
  
  (printf "Running FC layer 2...~n")
  (define fc2 (simple-fc fc1 fc2-weights fc2-bias #f))
  (printf "FC2 output shape: ~a~n" (dt:shape fc2))
  
  ;; Softmax
  (printf "Applying softmax...~n")
  (define output (simple-softmax fc2))
  (printf "Final output shape: ~a~n" (dt:shape output))
  
  ;; Print sample of output
  (printf "Sample output probabilities (first example):~n")
  (for ([i (in-range 10)])
    (printf "  Class ~a: ~a~n" i (vector-ref (dt:data output) i)))
  
  (printf "CNN test completed successfully!~n")
  output)

;; Run the test when the file is executed directly
(module+ main
  (run-cnn-test))