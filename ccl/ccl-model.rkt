#lang racket

;; ============================================================
;; CCL: Model Architecture — Frozen & Live Modes
;; ============================================================

(require "../device_pytorch.rkt")
(require "../nn.rkt")
(require (only-in pyffi run run*))

(provide
 make-ccl-linear ccl-linear-forward
 ccl-linear-skill-u ccl-linear-skill-v
 ccl-linear-base-weight ccl-linear-base-bias
 ccl-linear-in-dim ccl-linear-out-dim ccl-linear-rank
 ensure-adapter!
 make-ccl-model ccl-forward
 ccl-get-skill-params ccl-get-all-params
 ccl-model-layers ccl-model-activation ccl-model-num-skills
 ccl-model-frozen?
 ccl-mse-loss ccl-mse-loss-val
 ccl-make-adam ccl-make-sgd ccl-zero-grad ccl-opt-step ccl-backward
 py-item)

(run* "
def _init_w(od, id_, sc, frozen):
    w = (torch.randn(od, id_, device='cuda') * sc).detach()
    w.requires_grad_(not frozen)
    return w

def _init_b(od, frozen):
    b = torch.zeros(od, device='cuda')
    b.requires_grad_(not frozen)
    return b

def _init_u(od, r):
    sc = 1.0 / (r ** 0.5)
    return (torch.randn(od, r, device='cuda') * sc).detach().requires_grad_(True)

def _init_v(id_, r):
    return torch.zeros(id_, r, device='cuda', requires_grad=True)

def _fwd(x, bw, bb, u, v):
    return x @ (bw + u @ v.T).T + bb

def _mse(p, t):
    return torch.nn.functional.mse_loss(p, t)

def _item(t):
    return float(t.item()) if hasattr(t, 'item') else float(t)

def _adam(params, lr):
    return torch.optim.Adam(params, lr=lr)

def _sgd(params, lr):
    return torch.optim.SGD(params, lr=lr)

def _zero(opt):
    opt.zero_grad()

def _step(opt):
    opt.step()

def _bwd(loss):
    loss.backward()
")

(define py-init-w (run "_init_w"))
(define py-init-b (run "_init_b"))
(define py-init-u (run "_init_u"))
(define py-init-v (run "_init_v"))
(define py-fwd    (run "_fwd"))
(define py-mse    (run "_mse"))
(define py-item   (run "_item"))
(define py-adam   (run "_adam"))
(define py-sgd    (run "_sgd"))
(define py-zero   (run "_zero"))
(define py-step   (run "_step"))
(define py-bwd    (run "_bwd"))

;; ============================================================
(struct ccl-linear
  (base-weight base-bias in-dim out-dim rank skill-u skill-v)
  #:mutable #:transparent)

(define (make-ccl-linear in-dim out-dim rank #:frozen [frozen #t])
  (define sc (sqrt (/ 2.0 (+ in-dim out-dim))))
  (ccl-linear (py-init-w out-dim in-dim sc frozen)
              (py-init-b out-dim frozen)
              in-dim out-dim rank (make-hash) (make-hash)))

(define (ensure-adapter! layer skill-idx)
  (unless (hash-has-key? (ccl-linear-skill-u layer) skill-idx)
    (hash-set! (ccl-linear-skill-u layer) skill-idx
               (py-init-u (ccl-linear-out-dim layer) (ccl-linear-rank layer)))
    (hash-set! (ccl-linear-skill-v layer) skill-idx
               (py-init-v (ccl-linear-in-dim layer) (ccl-linear-rank layer)))))

(define (ccl-linear-forward layer x skill-idx)
  (ensure-adapter! layer skill-idx)
  (py-fwd x (ccl-linear-base-weight layer) (ccl-linear-base-bias layer)
          (hash-ref (ccl-linear-skill-u layer) skill-idx)
          (hash-ref (ccl-linear-skill-v layer) skill-idx)))

;; ============================================================
(struct ccl-model (layers activation num-skills frozen?) #:mutable #:transparent)

(define (make-ccl-model dims rank #:activation [act "gelu"] #:num-skills [n 0] #:frozen [frozen #t])
  (ccl-model
   (for/list ([i (in-range (sub1 (length dims)))])
     (make-ccl-linear (list-ref dims i) (list-ref dims (add1 i)) rank #:frozen frozen))
   act n frozen))

(define (ccl-forward model x skill-idx)
  (define layers (ccl-model-layers model))
  (define n (length layers))
  (for/fold ([h x]) ([layer (in-list layers)] [i (in-naturals)])
    (define out (ccl-linear-forward layer h skill-idx))
    (if (< i (sub1 n))
        (cond [(equal? (ccl-model-activation model) "relu") (relu out)]
              [(equal? (ccl-model-activation model) "gelu") (gelu out)]
              [else out])
        out)))

(define (ccl-get-skill-params model skill-idx)
  (for/fold ([ps '()]) ([layer (in-list (ccl-model-layers model))])
    (ensure-adapter! layer skill-idx)
    (append ps (list (hash-ref (ccl-linear-skill-u layer) skill-idx)
                     (hash-ref (ccl-linear-skill-v layer) skill-idx)))))

(define (ccl-get-all-params model)
  (for/fold ([ps '()]) ([layer (in-list (ccl-model-layers model))])
    (define base (list (ccl-linear-base-weight layer) (ccl-linear-base-bias layer)))
    (define adapters
      (for/fold ([ap '()]) ([(idx u) (in-hash (ccl-linear-skill-u layer))])
        (append ap (list u (hash-ref (ccl-linear-skill-v layer) idx)))))
    (append ps base adapters)))

(define (ccl-mse-loss pred target) (py-mse pred target))
(define (ccl-mse-loss-val pred target) (py-item (py-mse pred target)))
(define (ccl-make-adam params lr) (py-adam params lr))
(define (ccl-make-sgd params lr) (py-sgd params lr))
(define (ccl-zero-grad opt) (py-zero opt))
(define (ccl-opt-step opt) (py-step opt))
(define (ccl-backward loss) (py-bwd loss))
