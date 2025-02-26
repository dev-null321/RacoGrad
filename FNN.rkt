#lang racket

(require "tensor.rkt"
         "autograd.rkt"
         "device.rkt")

(provide (all-defined-out))

;; Feedforward Neural Network for classification tasks

;; Initialize a multi-layer feedforward neural network
;; Example: (initialize-neural-network (list 784 256 128 10))
(define (initialize-neural-network layer-sizes [scale 0.01])
  (let* ([num-layers (length layer-sizes)]
         [weights (for/list ([i (in-range (sub1 num-layers))])
                    (t:random (list (list-ref layer-sizes i) 
                                   (list-ref layer-sizes (add1 i)))
                             scale))]
         [biases (for/list ([i (in-range (sub1 num-layers))])
                   (t:random (list 1 (list-ref layer-sizes (add1 i)))
                            scale))])
    (values weights biases)))

;; Forward pass through the entire network
(define (forward-pass input weights biases activation-fn [final-activation softmax])
  (let* ([num-layers (add1 (length weights))]
         [layer-outputs '()]
         [layer-inputs '()]
         [current-activations input])
    
    ;; Process each layer
    (for ([i (in-range (length weights))])
      (let* ([w (list-ref weights i)]
             [b (list-ref biases i)]
             [is-final-layer? (= i (sub1 (length weights)))]
             [z (t:add (t:mul current-activations w) b)]
             [a (if is-final-layer?
                    (final-activation z)
                    (activation-fn z))])
        (set! layer-inputs (cons z layer-inputs))
        (set! layer-outputs (cons a layer-outputs))
        (set! current-activations a)))
    
    ;; Return final output and intermediate values for backprop
    (values (car layer-outputs)               ; Final output
            (reverse (cdr layer-outputs))     ; Hidden layer activations
            (reverse layer-inputs))))         ; Pre-activation values

;; Calculate cross-entropy loss for classification
(define (cross-entropy-loss y-pred y-true)
  (let* ([m (car (t:shape y-true))]
         [epsilon 1e-15]
         [y-pred-clipped (t:create (t:shape y-pred)
                                 (for/list ([p (vector->list (t:data y-pred))])
                                   (max (min p (- 1.0 epsilon)) epsilon)))]
         [loss-vec (for/list ([i (in-range (vector-length (t:data y-true)))])
                    (* (vector-ref (t:data y-true) i)
                       (log (vector-ref (t:data y-pred-clipped) i))))])
    (- (/ (apply + loss-vec) m))))

;; Backpropagation and parameter update
(define (backpropagation input y-true y-pred hidden-activations z-values weights biases learning-rate)
  (let* ([num-layers (length weights)]
         [batch-size (car (t:shape input))]
         
         ;; Calculate initial gradient (for output layer)
         [output-grad (t:sub y-pred y-true)]  ; For softmax + cross-entropy, this is the gradient
         
         ;; Initialize lists to store gradients
         [weight-gradients '()]
         [bias-gradients '()]
         
         ;; Add input to start of activations for backprop calculations
         [all-activations (cons input hidden-activations)]
         
         ;; Backpropagate through the network
         [current-grad output-grad])
    
    ;; Process each layer in reverse
    (for ([layer-idx (in-range (sub1 num-layers) -1 -1)])
      (let* ([activation (list-ref all-activations layer-idx)]
             [z (list-ref z-values layer-idx)]
             [w (list-ref weights layer-idx)]
             
             ;; For hidden layers, apply activation derivative to gradients
             [gradient (if (= layer-idx (sub1 num-layers))
                           current-grad  ; Output layer gradient
                           ;; Hidden layer gradient
                           (let ([grad-times-weights (t:mul current-grad (t:transpose (list-ref weights (add1 layer-idx))))]
                                 [activation-deriv (relu-derivative z)])
                             (t:mul grad-times-weights activation-deriv)))]
             
             ;; Calculate gradients for weights and biases
             [weight-grad (t:scale (t:mul (t:transpose activation) gradient) 
                                 (/ 1.0 batch-size))]
             [bias-grad (t:create (t:shape (list-ref biases layer-idx))
                               (for/list ([j (in-range (cadr (t:shape gradient)))])
                                 (/ (for/sum ([i (in-range batch-size)])
                                      (vector-ref (t:data gradient) (+ (* i (cadr (t:shape gradient))) j)))
                                    batch-size)))])
        
        ;; Store gradients
        (set! weight-gradients (cons weight-grad weight-gradients))
        (set! bias-gradients (cons bias-grad bias-gradients))
        
        ;; Update current gradient for next layer
        (set! current-grad gradient)))
    
    ;; Update weights and biases using gradients
    (let ([new-weights (map (lambda (w gradient) 
                             (t:sub w (t:scale gradient learning-rate))) 
                           weights weight-gradients)]
          [new-biases (map (lambda (b gradient) 
                            (t:sub b (t:scale gradient learning-rate))) 
                          biases bias-gradients)])
      
      (values new-weights new-biases))))

;; Softmax function
(define (softmax z)
  (let* ([shape (t:shape z)]
         [data (t:data z)]
         [max-vals (for/vector ([i (in-range (car shape))])
                    (apply max (for/list ([j (in-range (cadr shape))])
                               (vector-ref data (+ (* i (cadr shape)) j)))))]
         [exp-vals (for/vector ([i (in-range (vector-length data))])
                    (exp (- (vector-ref data i) 
                           (vector-ref max-vals (quotient i (cadr shape))))))]
         [sum-vals (for/vector ([i (in-range (car shape))])
                    (for/sum ([j (in-range (cadr shape))])
                      (vector-ref exp-vals (+ (* i (cadr shape)) j))))])
    (t:create shape
              (for/vector ([i (in-range (vector-length exp-vals))])
                (/ (vector-ref exp-vals i)
                   (vector-ref sum-vals (quotient i (cadr shape))))))))

;; Predict classes from model output
(define (predict output-tensor)
  (let* ([shape (t:shape output-tensor)]
         [batch-size (car shape)]
         [num-classes (cadr shape)]
         [data (t:data output-tensor)]
         [predictions (for/list ([b (in-range batch-size)])
                        (for/fold ([max-idx 0]
                                    [max-val (vector-ref data (* b num-classes))])
                                   ([c (in-range 1 num-classes)])
                          (let ([val (vector-ref data (+ (* b num-classes) c))])
                            (if (> val max-val)
                                (values c val)
                                (values max-idx max-val)))))])
    predictions))

;; Calculate accuracy
(define (accuracy predictions y-true)
  (let* ([num-samples (length predictions)]
         [true-labels (let ([shape (t:shape y-true)]
                            [data (t:data y-true)])
                        (for/list ([i (in-range (car shape))])
                          (for/first ([j (in-range (cadr shape))]
                                     #:when (= 1.0 (vector-ref data (+ (* i (cadr shape)) j))))
                            j)))]
         [correct-count (for/sum ([i (in-range num-samples)])
                          (if (= (list-ref predictions i) (list-ref true-labels i))
                              1
                              0))])
    (/ correct-count num-samples)))

;; Create batch data
(define (get-batch X y indices)
  (let* ([batch-size (length indices)]
         [X-features (cadr (t:shape X))]
         [y-classes (cadr (t:shape y))]
         
         [X-batch-data (make-vector (* batch-size X-features) 0.0)]
         [y-batch-data (make-vector (* batch-size y-classes) 0.0)])
    
    ;; Copy data for each example in the batch
    (for ([i (in-range batch-size)]
          [idx (in-list indices)])
      ;; Copy features
      (for ([j (in-range X-features)])
        (vector-set! X-batch-data 
                     (+ (* i X-features) j)
                     (vector-ref (t:data X) (+ (* idx X-features) j))))
      
      ;; Copy labels
      (for ([j (in-range y-classes)])
        (vector-set! y-batch-data 
                     (+ (* i y-classes) j)
                     (vector-ref (t:data y) (+ (* idx y-classes) j)))))
    
    (values (t:create (list batch-size X-features) X-batch-data)
            (t:create (list batch-size y-classes) y-batch-data))))

;; Simple training function for demonstration
(define (train-fnn X-train y-train X-val y-val 
                  [hidden-layers (list 128 64)] 
                  [learning-rate 0.01] 
                  [epochs 10] 
                  [batch-size 32])
  (printf "Initializing neural network...~n")
  
  ;; Determine network architecture
  (define input-size (cadr (t:shape X-train)))
  (define output-size (cadr (t:shape y-train)))
  (define layer-sizes (append (list input-size) hidden-layers (list output-size)))
  
  ;; Initialize network parameters
  (define-values (weights biases) 
    (initialize-neural-network layer-sizes))
  
  ;; Early stopping parameters
  (define patience 3)
  (define best-val-accuracy 0.0)
  (define wait 0)
  (define best-weights weights)
  (define best-biases biases)
  
  (printf "Neural network architecture: ~a~n" layer-sizes)
  (printf "Starting training for ~a epochs...~n" epochs)
  
  ;; Training loop
  (for/fold ([stop? #f]) 
            ([epoch (in-range epochs)] #:break stop?)
    (printf "Epoch ~a/~a:~n" (add1 epoch) epochs)
    
    ;; Shuffle training data
    (define train-size (car (t:shape X-train)))
    (define indices (shuffle (range train-size)))
    (define num-batches (quotient train-size batch-size))
    
    ;; Initialize tracking variables
    (define epoch-losses '())
    
    ;; Process each batch
    (for ([batch (in-range num-batches)])
      (let* ([start-idx (* batch batch-size)]
             [end-idx (min (+ start-idx batch-size) train-size)]
             [batch-indices (take (drop indices start-idx) (- end-idx start-idx))])
        
        ;; Get batch data
        (define-values (X-batch y-batch) 
          (get-batch X-train y-train batch-indices))
        
        ;; Forward pass
        (define-values (y-pred hidden-activations z-values) 
          (forward-pass X-batch weights biases relu))
        
        ;; Calculate loss
        (define loss (cross-entropy-loss y-pred y-batch))
        (set! epoch-losses (cons loss epoch-losses))
        
        ;; Backpropagation and parameter update
        (define-values (new-weights new-biases) 
          (backpropagation X-batch y-batch y-pred 
                          hidden-activations z-values 
                          weights biases learning-rate))
        
        ;; Update parameters
        (set! weights new-weights)
        (set! biases new-biases)
        
        ;; Print progress
        (when (= (modulo batch 10) 0)
          (printf "  Batch ~a/~a: Loss = ~a~n" 
                  batch 
                  num-batches 
                  (real->decimal-string loss 4)))))
    
    ;; Evaluate on validation set
    (define-values (val-pred _ __) 
      (forward-pass X-val weights biases relu))
    
    (define val-predictions (predict val-pred))
    (define val-accuracy (* 100.0 (accuracy val-predictions y-val)))
    
    ;; Calculate average loss for epoch
    (define avg-loss 
      (/ (apply + epoch-losses) (length epoch-losses)))
    
    (printf "Epoch Summary:~n")
    (printf "  Average Loss: ~a~n" (real->decimal-string avg-loss 4))
    (printf "  Validation Accuracy: ~a%~n" (real->decimal-string val-accuracy 2))
    
    ;; Early stopping check
    (if (> val-accuracy best-val-accuracy)
        (begin
          (set! best-val-accuracy val-accuracy)
          (set! best-weights weights)
          (set! best-biases biases)
          (set! wait 0)
          (printf "  New best model saved!~n"))
        (begin
          (set! wait (add1 wait))
          (printf "  No improvement for ~a epochs.~n" wait)))
    
    ;; Check if we should stop
    (if (>= wait patience)
        (begin
          (printf "Early stopping triggered after ~a epochs without improvement.~n" 
                  patience)
          #t)  ; Stop training
        #f))   ; Continue training
  
  ;; Restore best model
  (set! weights best-weights)
  (set! biases best-biases)
  
  (printf "Training Complete!~n")
  (printf "Best Validation Accuracy: ~a%~n" 
          (real->decimal-string best-val-accuracy 2))
  
  ;; Return the trained model
  (values weights biases))

;; Example usage in a main module
(module+ main
  (require "mnist.rkt")
  
  (printf "Loading MNIST data...~n")
  
  ;; Data loading functions are in mnist.rkt
  (define-values (X-train-full y-train-full) (load-mnist-data "train"))
  (define-values (X-test y-test) (load-mnist-data "test"))
  
  ;; Normalize and one-hot encode
  (define X-train-norm (normalize X-train-full))
  (define X-test-norm (normalize X-test))
  (define y-train-onehot (one-hot y-train-full 10))
  (define y-test-onehot (one-hot y-test 10))
  
  ;; Create validation split
  (define train-size (car (t:shape X-train-norm)))
  (define validation-split 0.1)
  (define val-size (inexact->exact (floor (* train-size validation-split))))
  (define train-size-actual (- train-size val-size))
  
  ;; Shuffle indices
  (define all-indices (shuffle (range train-size)))
  (define train-indices (take all-indices train-size-actual))
  (define val-indices (drop all-indices train-size-actual))
  
  ;; Create training and validation sets
  (define-values (X-train y-train) (get-batch X-train-norm y-train-onehot train-indices))
  (define-values (X-val y-val) (get-batch X-train-norm y-train-onehot val-indices))
  
  (printf "Training FNN on MNIST...~n")
  (printf "Train set: ~a examples~n" (car (t:shape X-train)))
  (printf "Validation set: ~a examples~n" (car (t:shape X-val)))
  (printf "Test set: ~a examples~n" (car (t:shape X-test-norm)))
  
  ;; Train the model
  (define-values (trained-weights trained-biases) 
    (train-fnn X-train y-train X-val y-val 
              (list 128)  ; One hidden layer with 128 units
              0.01        ; Learning rate
              5           ; Epochs
              64))        ; Batch size
  
  ;; Evaluate on test set
  (define-values (test-pred _ __) 
    (forward-pass X-test-norm trained-weights trained-biases relu))
  
  (define test-predictions (predict test-pred))
  (define test-accuracy (* 100.0 (accuracy test-predictions y-test-onehot)))
  
  (printf "Final Test Accuracy: ~a%~n" 
          (real->decimal-string test-accuracy 2)))