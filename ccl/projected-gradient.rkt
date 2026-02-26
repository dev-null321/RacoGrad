#lang racket

;; ============================================================
;; CCL: Projected Gradient Descent — Live Mode
;; ============================================================
;;
;; Live: all params updatable, grad projected into skill V_j via P_j
;; Frozen: only adapter params updated (baseline)
;; ============================================================

(require "../device_pytorch.rkt")
(require (only-in pyffi run run*))
(require "ccl-model.rkt")

(provide
 projected-train-step!
 frozen-train-step!
 reorthogonalize-adapters!
 project-gradients!)

(run* "
def _project_grads(base_params, u):
    gram = u.T @ u + 1e-6 * torch.eye(u.shape[1], device=u.device)
    P = u @ torch.linalg.inv(gram) @ u.T
    for p in base_params:
        if p.grad is not None:
            if p.grad.dim() == 1:
                p.grad.copy_((P @ p.grad.unsqueeze(1)).squeeze(1))
            else:
                p.grad.copy_(P @ p.grad)

def _gram_schmidt(u_list):
    result = []
    for i, u in enumerate(u_list):
        u_new = u.clone()
        for j in range(i):
            uj = result[j]
            gram = uj.T @ uj + 1e-6 * torch.eye(uj.shape[1], device=uj.device)
            proj = uj @ torch.linalg.inv(gram) @ uj.T @ u_new
            u_new = u_new - proj
        norms = torch.norm(u_new, dim=0, keepdim=True) + 1e-8
        u_new = u_new / norms * torch.norm(u, dim=0, keepdim=True)
        result.append(u_new)
    return result

def _copy_data(dst, src):
    with torch.no_grad():
        dst.copy_(src)

def _getitem(lst, i):
    return lst[i]
")

(define py-project-grads (run "_project_grads"))
(define py-gram-schmidt (run "_gram_schmidt"))
(define py-copy-data (run "_copy_data"))
(define py-getitem (run "_getitem"))

;; Frozen-base step: only adapter params in optimizer
(define (frozen-train-step! model skill-idx x y opt)
  (ccl-zero-grad opt)
  (define pred (ccl-forward model x skill-idx))
  (define loss (ccl-mse-loss pred y))
  (ccl-backward loss)
  (ccl-opt-step opt)
  (py-item loss))

;; Project base param gradients into skill j's subspace
(define (project-gradients! model skill-idx)
  (for ([layer (in-list (ccl-model-layers model))])
    (ensure-adapter! layer skill-idx)
    (py-project-grads
     (list (ccl-linear-base-weight layer) (ccl-linear-base-bias layer))
     (hash-ref (ccl-linear-skill-u layer) skill-idx))))

;; Live projected step: all params in optimizer, but grads projected
(define (projected-train-step! model skill-idx x y opt)
  (ccl-zero-grad opt)
  (define pred (ccl-forward model x skill-idx))
  (define loss (ccl-mse-loss pred y))
  (ccl-backward loss)
  (project-gradients! model skill-idx)
  (ccl-opt-step opt)
  (py-item loss))

;; Modified Gram-Schmidt re-orthogonalization on U matrices
(define (reorthogonalize-adapters! model num-skills-val)
  (for ([layer (in-list (ccl-model-layers model))])
    (define u-list
      (for/list ([j (in-range num-skills-val)])
        (ensure-adapter! layer j)
        (hash-ref (ccl-linear-skill-u layer) j)))
    (define new-u-list (py-gram-schmidt u-list))
    (for ([j (in-range num-skills-val)])
      (py-copy-data (list-ref u-list j) (py-getitem new-u-list j)))))
