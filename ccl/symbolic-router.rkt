#lang racket

;; ============================================================
;; CCL: Symbolic Router — Task → Skill Index
;; ============================================================

(provide
 define-skill-registry
 register-skill!
 route-task
 with-skill
 skill-registry?
 skill-registry-skills
 skill-entry-name
 skill-entry-index
 skill-entry-keywords
 describe-skills)

(struct skill-entry (index name keywords description) #:transparent)
(struct skill-registry (skills name->idx keyword->idx) #:mutable #:transparent)

(define (define-skill-registry)
  (skill-registry '() (make-hash) (make-hash)))

(define (register-skill! registry name keywords [description ""])
  (define idx (length (skill-registry-skills registry)))
  (define entry (skill-entry idx name keywords description))
  (set-skill-registry-skills! registry
    (append (skill-registry-skills registry) (list entry)))
  (hash-set! (skill-registry-name->idx registry) name idx)
  (for ([kw (in-list keywords)])
    (hash-set! (skill-registry-keyword->idx registry) kw idx))
  idx)

(define (route-task registry task-description)
  (define desc (if (symbol? task-description)
                   (symbol->string task-description)
                   task-description))
  (define words (string-split (string-downcase desc)))
  (define scores
    (for/list ([entry (in-list (skill-registry-skills registry))])
      (define score
        (for/sum ([w (in-list words)])
          (if (member w (skill-entry-keywords entry)) 1 0)))
      (cons (skill-entry-index entry) score)))
  (define best (argmax cdr scores))
  (if (> (cdr best) 0) (car best) 0))

(define-syntax-rule (with-skill registry task-desc body ...)
  (let ([current-skill-idx (route-task registry task-desc)])
    body ...))

(define (describe-skills registry)
  (printf "╔═══════════════════════════════════════╗\n")
  (printf "║       CCL Skill Registry              ║\n")
  (printf "╠═══════════════════════════════════════╣\n")
  (for ([entry (in-list (skill-registry-skills registry))])
    (printf "║ [~a] ~a\n" (skill-entry-index entry) (skill-entry-name entry))
    (printf "║     keywords: ~a\n"
            (string-join (skill-entry-keywords entry) ", "))
    (when (not (equal? (skill-entry-description entry) ""))
      (printf "║     ~a\n" (skill-entry-description entry))))
  (printf "╚═══════════════════════════════════════╝\n"))
