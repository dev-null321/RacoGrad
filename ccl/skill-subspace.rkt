#lang racket

;; ============================================================
;; CCL: Skill Subspace Decomposition
;; ============================================================

(require "../device_pytorch.rkt")
(require (only-in pyffi run run*))

(provide
 make-skill-subspace-manager
 skill-mgr-adapters skill-mgr-param-dim skill-mgr-rank skill-mgr-base
 set-skill-mgr-adapters!
 num-skills get-projection project-vector
 orthogonality-loss-val get-skill-delta compose-params
 add-skill! measure-orthogonality-deviation)

(run* "
def _mk_u(d, r, scale):
    return torch.randn(d, r, device='cuda', requires_grad=True) * scale

def _mk_v(d, r):
    return torch.zeros(d, r, device='cuda', requires_grad=True)

def _proj_mat(u):
    gram = u.T @ u + 1e-6 * torch.eye(u.shape[1], device='cuda')
    return u @ torch.linalg.inv(gram) @ u.T

def _proj_vec(proj, v):
    return proj @ v

def _delta(u, v):
    return u @ v.T

def _ortho_dev(ui, uj):
    c = torch.norm(ui.T @ uj, p='fro').item()
    n = (torch.norm(ui, p='fro') * torch.norm(uj, p='fro')).item()
    return c / (n + 1e-8)

def _ortho_loss_val(ui, uj):
    return torch.sum((ui.T @ uj) ** 2).item()
")

(define mk-u (run "_mk_u"))
(define mk-v (run "_mk_v"))
(define proj-mat (run "_proj_mat"))
(define proj-vec (run "_proj_vec"))
(define py-delta (run "_delta"))
(define py-ortho-dev (run "_ortho_dev"))
(define py-ortho-loss (run "_ortho_loss_val"))

(struct skill-mgr (param-dim rank adapters base) #:mutable #:transparent)

(define (num-skills m) (length (skill-mgr-adapters m)))

(define (make-adapter d r)
  (define s (/ 1.0 (sqrt (* d r))))
  (cons (mk-u d r s) (mk-v d r)))

(define (make-skill-subspace-manager pd r bp #:num-skills [n 0])
  (skill-mgr pd r (for/list ([_ (in-range n)]) (make-adapter pd r)) bp))

(define (add-skill! m)
  (define a (make-adapter (skill-mgr-param-dim m) (skill-mgr-rank m)))
  (set-skill-mgr-adapters! m (append (skill-mgr-adapters m) (list a)))
  (sub1 (num-skills m)))

(define (get-projection m j)
  (proj-mat (car (list-ref (skill-mgr-adapters m) j))))

(define (project-vector m j v)
  (proj-vec (get-projection m j) v))

(define (measure-orthogonality-deviation m)
  (define as (skill-mgr-adapters m))
  (define n (length as))
  (if (< n 2) 0.0
      (apply max (for*/list ([i (in-range n)] [j (in-range (add1 i) n)])
                   (py-ortho-dev (car (list-ref as i)) (car (list-ref as j)))))))

(define (orthogonality-loss-val m)
  (define as (skill-mgr-adapters m))
  (define n (length as))
  (if (< n 2) 0.0
      (for*/sum ([i (in-range n)] [j (in-range (add1 i) n)])
        (py-ortho-loss (car (list-ref as i)) (car (list-ref as j))))))

(define (get-skill-delta m j)
  (define a (list-ref (skill-mgr-adapters m) j))
  (py-delta (car a) (cdr a)))

(define (compose-params m j)
  (add (skill-mgr-base m) (get-skill-delta m j)))
