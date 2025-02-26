#lang racket

(require "tensor.rkt")
(require "deep_learn_library.rkt")

;; Create sample data
;; X: 2 features, 4 samples
(define X (t:create '(4 2) '(1 2   
                            2 3
                            3 4
                            4 5)))

;; y: binary labels (0 or 1)
(define y (t:create '(4 1) '(0.0 0.0 1.0 1.0))) 

;; Initialize weights and bias
(define weights (t:random '(2 1) 1.0))  ; 2 features -> 1 output
(define bias (t:random '(1) 1.0))

;; Sigmoid function
(define (sigmoid z)
  (t:create (t:shape z)
            (for/list ([x (vector->list (t:data z))])
              (/ 1.0 (+ 1.0 (exp (- x)))))))

;; Forward pass
(define (forward X w b)
  (let* ([z (t:add (t:mul X w) b)])
    (sigmoid z)))

;; Binary Cross Entropy Loss
(define (binary-cross-entropy y-pred y-true)
  (let* ([m (car (t:shape y-true))]
         [epsilon 1e-15]  ; To avoid log(0)
         [y-pred-clipped (t:create (t:shape y-pred)
                                 (for/list ([p (vector->list (t:data y-pred))])
                                   (max (min p (- 1.0 epsilon)) epsilon)))]
         [loss-vec (for/list ([p (vector->list (t:data y-pred-clipped))]
                             [t (vector->list (t:data y-true))])
                    (+ (* t (log p))
                       (* (- 1 t) (log (- 1.0 p)))))])
    (- (/ (apply + loss-vec) m))))

;; Training loop
(define learning-rate 0.1)
(define epochs 100)

(for ([epoch (in-range epochs)])
  (let* ([;; Forward pass
          y-pred (forward X weights bias)]
         [;; Calculate loss
          loss (binary-cross-entropy y-pred y)])
    
    ;; Print progress every 10 epochs
    (when (= (modulo epoch 10) 0)
      (printf "Epoch ~a, Loss: ~a~n" epoch loss))
    
    ;; Calculate gradients (simplified for this example)
    (let* ([error (t:sub y-pred y)]
           [gradient-w (t:scale (t:mul (t:transpose X) error) (/ 1.0 (car (t:shape X))))]
           [gradient-b (t:create '(1) 
                               (list (/ (for/sum ([e (vector->list (t:data error))]) e)
                                      (car (t:shape X)))))])
      
      ;; Update weights and bias
      (set! weights (t:sub weights (t:scale gradient-w learning-rate)))
      (set! bias (t:sub bias (t:scale gradient-b learning-rate))))))

;; Test the model
(printf "~nFinal weights:~n")
(t:print weights)
(printf "~nFinal bias:~n")
(t:print bias)

;; Make predictions
(printf "~nFinal predictions:~n")
(t:print (forward X weights bias))