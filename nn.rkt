#lang racket

;; ============================================================
;; RacoGrad Neural Network Module System
;; Lispy nn.Module-style layer abstractions
;; ============================================================

(require "device_pytorch.rkt")

(provide
 ;; Module struct and interface
 nn-module
 nn-module?
 nn-module-name
 forward
 parameters
 
 ;; Layer constructors
 make-linear
 make-embedding
 make-layer-norm
 make-dropout
 
 ;; Containers
 sequential
 
 ;; Initialization schemes
 init-xavier-uniform
 init-xavier-normal
 init-kaiming-uniform
 init-kaiming-normal
 init-zeros
 init-ones
 
 ;; Parameter helpers
 param
 param?
 param-tensor
 param-name
 all-parameters
 num-parameters
 print-module)

;; ============================================================
;; Parameter Wrapper
;; ============================================================

(struct param (name tensor) #:transparent)

;; ============================================================
;; Module Structure
;; ============================================================

(struct nn-module (name forward-fn params children) #:transparent)

;; Forward pass - works with modules or bare functions
(define (forward m input)
  (cond
    [(nn-module? m) ((nn-module-forward-fn m) input)]
    [(procedure? m) (m input)]
    [else (error "forward: expected module or procedure, got ~a" m)]))

;; Get parameters from a module (recursively)
(define (parameters m)
  (cond
    [(nn-module? m)
     (append (nn-module-params m)
             (apply append (map parameters (nn-module-children m))))]
    [else '()]))

;; ============================================================
;; Initialization Schemes
;; ============================================================

(define (init-xavier-uniform shape #:gain [gain 1.0])
  (define fan-in (car shape))
  (define fan-out (if (> (length shape) 1) (cadr shape) 1))
  (define std (* gain (sqrt (/ 2.0 (+ fan-in fan-out)))))
  (define bound (* std (sqrt 3.0)))
  (mul (randn shape) (tensor (* bound 2))))

(define (init-xavier-normal shape #:gain [gain 1.0])
  (define fan-in (car shape))
  (define fan-out (if (> (length shape) 1) (cadr shape) 1))
  (define std (* gain (sqrt (/ 2.0 (+ fan-in fan-out)))))
  (mul (randn shape) (tensor std)))

(define (init-kaiming-uniform shape #:a [a 0] #:mode [mode 'fan-in])
  (define fan (if (eq? mode 'fan-in) (car shape) (cadr shape)))
  (define gain (sqrt (/ 2.0 (+ 1 (* a a)))))
  (define std (/ gain (sqrt fan)))
  (define bound (* std (sqrt 3.0)))
  (mul (randn shape) (tensor (* bound 2))))

(define (init-kaiming-normal shape #:a [a 0] #:mode [mode 'fan-in])
  (define fan (if (eq? mode 'fan-in) (car shape) (cadr shape)))
  (define gain (sqrt (/ 2.0 (+ 1 (* a a)))))
  (define std (/ gain (sqrt fan)))
  (mul (randn shape) (tensor std)))

(define (init-zeros shape) (zeros shape))
(define (init-ones shape) (ones shape))

;; ============================================================
;; Linear Layer
;; ============================================================

(define (make-linear in-features out-features 
                     #:bias [use-bias #t]
                     #:init [init-fn init-kaiming-uniform])
  (define W (param "weight" (init-fn (list out-features in-features))))
  (define b (if use-bias (param "bias" (zeros (list out-features))) #f))
  
  (define (forward-fn input)
    (if b
        (linear input (param-tensor W) #:bias (param-tensor b))
        (linear input (param-tensor W))))
  
  (nn-module "Linear" forward-fn (if b (list W b) (list W)) '()))

;; ============================================================
;; Embedding Layer
;; ============================================================

(define (make-embedding num-embeddings embedding-dim
                        #:init [init-fn init-xavier-normal])
  (define E (param "embedding" (init-fn (list num-embeddings embedding-dim))))
  (define (forward-fn indices) (embedding (param-tensor E) indices))
  (nn-module "Embedding" forward-fn (list E) '()))

;; ============================================================
;; Layer Normalization
;; ============================================================

(define (make-layer-norm normalized-shape #:eps [eps 1e-5])
  (define shape (if (list? normalized-shape) normalized-shape (list normalized-shape)))
  (define gamma (param "gamma" (ones shape)))
  (define beta (param "beta" (zeros shape)))
  
  (define (forward-fn input)
    (define normed (layer-norm input shape #:eps eps))
    (add (mul normed (param-tensor gamma)) (param-tensor beta)))
  
  (nn-module "LayerNorm" forward-fn (list gamma beta) '()))

;; ============================================================
;; Dropout Layer
;; ============================================================

(define (make-dropout p #:training [training #t])
  (nn-module "Dropout" (lambda (input) (dropout input #:p p #:training training)) '() '()))

;; ============================================================
;; Sequential Container
;; ============================================================

(define (sequential . layers)
  (define (forward-fn input)
    (for/fold ([x input]) ([layer layers])
      (forward layer x)))
  (define child-modules (filter nn-module? layers))
  (nn-module "Sequential" forward-fn '() child-modules))

;; ============================================================
;; Parameter Utilities
;; ============================================================

(define (all-parameters model) (map param-tensor (parameters model)))

(define (num-parameters model)
  (for/sum ([p (parameters model)])
    (apply * (shape (param-tensor p)))))

(define (print-module m [indent 0])
  (define prefix (make-string indent #\space))
  (printf "~a~a(\n" prefix (nn-module-name m))
  (for ([p (nn-module-params m)])
    (printf "~a  ~a: ~a\n" prefix (param-name p) (shape (param-tensor p))))
  (for ([child (nn-module-children m)])
    (print-module child (+ indent 2)))
  (printf "~a)\n" prefix))
