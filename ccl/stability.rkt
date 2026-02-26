#lang racket

;; ============================================================
;; CCL: Stability Verification & Metrics Tracking
;; ============================================================

(require "../device_pytorch.rkt")
(require (only-in pyffi run run*))

(provide
 make-metrics-tracker
 record-metric!
 get-metric
 get-all-metrics
 stability-report
 print-interference-matrix
 print-scaling-table)

;; Python helper
(run* "
def _ccl_item_safe(t):
    if hasattr(t, 'item'):
        return float(t.item())
    return float(t)
")
(define py-item-safe (run "_ccl_item_safe"))

;; ============================================================
;; Metrics Tracker
;; ============================================================
;; Tracks time series of: per-skill loss, epsilon, interference

(struct metrics-tracker (data) #:mutable #:transparent)

(define (make-metrics-tracker)
  (metrics-tracker (make-hash)))

(define (record-metric! tracker key step value)
  (define h (metrics-tracker-data tracker))
  (define series (hash-ref h key '()))
  (hash-set! h key (append series (list (cons step value)))))

(define (get-metric tracker key)
  (hash-ref (metrics-tracker-data tracker) key '()))

(define (get-all-metrics tracker)
  (metrics-tracker-data tracker))

;; ============================================================
;; Reports
;; ============================================================

(define (stability-report tracker num-skills-val)
  (printf "\n═══════════════════════════════════════════════════\n")
  (printf "  CCL Stability Report\n")
  (printf "═══════════════════════════════════════════════════\n")
  
  ;; Per-skill final losses
  (printf "\n  Final losses per skill:\n")
  (for ([j (in-range num-skills-val)])
    (define series (get-metric tracker (format "loss-~a" j)))
    (when (not (null? series))
      (printf "    Skill ~a: ~a\n" j (~r (cdar (last-pair series)) #:precision '(= 6)))))
  
  ;; Epsilon over time
  (define eps-series (get-metric tracker "epsilon"))
  (when (not (null? eps-series))
    (printf "\n  Orthogonality deviation (ε) over time:\n")
    (for ([pt (in-list eps-series)])
      (printf "    Step ~a: ε = ~a\n" (car pt) (~r (cdr pt) #:precision '(= 6)))))
  
  (printf "═══════════════════════════════════════════════════\n\n"))

(define (print-interference-matrix losses-before losses-after num-tasks label)
  (printf "\n  ~a Interference Matrix:\n" label)
  (printf "  (positive = forgetting, negative = transfer)\n\n")
  (printf "  Trained ↓  Eval→")
  (for ([e (in-range num-tasks)]) (printf "    Task ~a  " e))
  (printf "\n")
  (for ([t (in-range num-tasks)])
    (printf "  Skill ~a:        " t)
    (for ([e (in-range num-tasks)])
      (define before (hash-ref losses-before (cons t e) 0.0))
      (define after (hash-ref losses-after (cons t e) 0.0))
      (printf " ~a " (~r (- after before) #:precision '(= 6))))
    (printf "\n")))

(define (print-scaling-table results)
  (printf "\n  η          Interference     η²\n")
  (printf "  ─────────  ──────────────   ──────────\n")
  (for ([r (in-list results)])
    (define eta (first r))
    (define interf (second r))
    (printf "  ~a    ~a   ~a\n"
            (~r eta #:precision '(= 5))
            (~r interf #:precision '(= 8))
            (~r (* eta eta) #:precision '(= 8)))))

;; Helper
(define (last-pair lst) (if (null? (cdr lst)) lst (last-pair (cdr lst))))
