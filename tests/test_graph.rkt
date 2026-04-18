#lang racket

;; ============================================================
;; Tests for graph.rkt — pure Racket, no model needed
;; ============================================================

(require rackunit
         "../graph.rkt")

(printf "Running graph tests...\n\n")

;; ============================================================
;; Test 1: Graph construction from S-expressions
;; ============================================================

(test-case "graph-from-sexpr builds correct tree"
  (define g (graph-from-sexpr
             '(decode (tokenize (load-model "model.gguf") "hello"))))
  (check-true (graph-node? g))
  (check-equal? (graph-node-op g) 'decode)
  (check-equal? (length (graph-node-inputs g)) 1)
  (define tok-node (car (graph-node-inputs g)))
  (check-true (graph-node? tok-node))
  (check-equal? (graph-node-op tok-node) 'tokenize)
  (printf "  [PASS] graph-from-sexpr builds correct tree\n"))

;; ============================================================
;; Test 2: Node counting
;; ============================================================

(test-case "graph-node-count counts all nodes"
  (define g (graph-from-sexpr
             '(sample (decode (tokenize (load-model "m.gguf") "hi")))))
  (check-equal? (graph-node-count g) 4)
  (printf "  [PASS] graph-node-count = 4 for 4-node graph\n"))

(test-case "graph-node-count single node"
  (define g (make-node 'load-model '("path")))
  (check-equal? (graph-node-count g) 1)
  (printf "  [PASS] graph-node-count = 1 for single node\n"))

;; ============================================================
;; Test 3: Topological sort
;; ============================================================

(test-case "topological sort puts dependencies first"
  (define g (graph-from-sexpr
             '(sample (decode (tokenize (load-model "m") "hello")))))
  (define sorted (graph-topological-sort g))
  (define ops (map graph-node-op sorted))
  ;; load-model must come before tokenize, tokenize before decode, etc.
  (check-true (< (index-of ops 'load-model) (index-of ops 'tokenize)))
  (check-true (< (index-of ops 'tokenize) (index-of ops 'decode)))
  (check-true (< (index-of ops 'decode) (index-of ops 'sample)))
  (printf "  [PASS] topological sort: load-model < tokenize < decode < sample\n"))

;; ============================================================
;; Test 4: Last-use analysis
;; ============================================================

(test-case "last-use analysis identifies correct last users"
  (define g (graph-from-sexpr
             '(sample (decode (tokenize (load-model "m") "hi")))))
  (define all-nodes (graph-collect-nodes g))
  (define last-use (graph-last-use g))

  ;; load-model is last used by tokenize
  (define load-node (findf (lambda (n) (eq? (graph-node-op n) 'load-model)) all-nodes))
  (define tok-node  (findf (lambda (n) (eq? (graph-node-op n) 'tokenize)) all-nodes))
  (check-equal? (hash-ref last-use (graph-node-id load-node))
                (graph-node-id tok-node))

  ;; tokenize is last used by decode
  (define dec-node (findf (lambda (n) (eq? (graph-node-op n) 'decode)) all-nodes))
  (check-equal? (hash-ref last-use (graph-node-id tok-node))
                (graph-node-id dec-node))

  (printf "  [PASS] last-use: load-model->tokenize, tokenize->decode\n"))

;; ============================================================
;; Test 5: Resource balance check
;; ============================================================

(test-case "resource-balanced? detects unbalanced graph"
  ;; Graph with load-model but no free-model
  (define g (graph-from-sexpr '(decode (load-model "m"))))
  (check-false (graph-resource-balanced? g))
  (printf "  [PASS] unbalanced graph (alloc without free) detected\n"))

(test-case "resource-balanced? accepts balanced graph"
  ;; Manually build a balanced graph
  (define load-n (make-node 'load-model '("path")))
  (define use-n (make-node 'decode (list load-n)))
  (define free-n (make-node 'free-model (list load-n)))
  ;; Root that depends on both use and free
  (define root (make-node 'seq (list use-n free-n)))
  (check-true (graph-resource-balanced? root))
  (printf "  [PASS] balanced graph (alloc + free) accepted\n"))

;; ============================================================
;; Test 6: Graph optimization inserts frees
;; ============================================================

(test-case "graph-optimize inserts free nodes after last use"
  (define g (graph-from-sexpr
             '(sample (decode (create-context (load-model "m"))))))
  (define optimized (graph-optimize g))
  (define ops (map graph-node-op optimized))

  ;; Should have original 4 nodes + 2 free nodes (free-model, free-context)
  (check-not-false (member 'free-model ops) "free-model should be inserted")
  (check-not-false (member 'free-context ops) "free-context should be inserted")

  ;; free-model should come after create-context (its last user)
  (check-true (< (index-of ops 'create-context) (index-of ops 'free-model)))

  ;; free-context should come after decode (its last user)
  (check-true (< (index-of ops 'decode) (index-of ops 'free-context)))

  (printf "  [PASS] optimization inserts free-model and free-context at correct points\n"))

;; ============================================================
;; Test 7: Optimization idempotency
;; ============================================================

(test-case "optimizing an already-optimized graph is stable"
  (define g (graph-from-sexpr
             '(decode (create-context (load-model "m")))))
  (define opt1 (graph-optimize g))
  ;; Optimizing the list again should produce same op sequence
  ;; (since frees are already inserted, no new ones should appear)
  (define ops1 (map graph-node-op opt1))
  ;; The free nodes don't contain resource allocs, so re-optimizing
  ;; the original graph should give same result
  (define opt2 (graph-optimize g))
  (define ops2 (map graph-node-op opt2))
  (check-equal? ops1 ops2)
  (printf "  [PASS] optimization is idempotent\n"))

;; ============================================================
;; Test 8: Graph to S-expression round trip
;; ============================================================

(test-case "graph->sexpr produces readable output"
  (define g (graph-from-sexpr '(decode (tokenize (load-model "m") "hi"))))
  (define sexpr (graph->sexpr g))
  (check-equal? (car sexpr) 'decode)
  (check-true (list? sexpr))
  (printf "  [PASS] graph->sexpr round trip produces valid S-expression\n"))

;; ============================================================
;; Test 9: with-llama-resources macro expands correctly
;; ============================================================

(test-case "with-llama-resources cleans up on normal exit"
  (define freed? (box #f))
  (with-llama-resources ([x 42 (set-box! freed? #t)])
    (check-equal? x 42))
  (check-true (unbox freed?))
  (printf "  [PASS] with-llama-resources cleans up on normal exit\n"))

(test-case "with-llama-resources cleans up on exception"
  (define freed? (box #f))
  (with-handlers ([exn:fail? (lambda (e) 'caught)])
    (with-llama-resources ([x 42 (set-box! freed? #t)])
      (error "boom")))
  (check-true (unbox freed?))
  (printf "  [PASS] with-llama-resources cleans up on exception\n"))

(test-case "with-llama-resources handles multiple bindings"
  (define order '())
  (with-llama-resources ([a 1 (set! order (cons 'a order))]
                         [b 2 (set! order (cons 'b order))])
    (check-equal? (+ a b) 3))
  ;; Cleanup should happen in reverse order (LIFO)
  (check-equal? order '(a b))
  (printf "  [PASS] with-llama-resources LIFO cleanup for multiple bindings\n"))

;; ============================================================
;; Test 10: define-generation-loop macro
;; ============================================================

(test-case "define-generation-loop produces working loop"
  ;; Mock functions
  (define decoded-tokens '())
  (define call-count 0)

  (define (mock-decode tokens)
    (set! decoded-tokens (append decoded-tokens tokens)))

  (define (mock-sample)
    (set! call-count (add1 call-count))
    (if (> call-count 5) 999 call-count))  ;; 999 = EOG

  (define (mock-eog? tok) (= tok 999))
  (define (mock-detok tok) (format "t~a" tok))

  (define-generation-loop test-loop
    #:decode-fn mock-decode
    #:sample-fn mock-sample
    #:eog-fn mock-eog?
    #:detok-fn mock-detok)

  (define result (test-loop '(100 101 102) 10 #f))
  ;; Should generate tokens 1-5, then stop at 999 (EOG)
  (check-equal? result '(1 2 3 4 5))
  ;; Prompt tokens should have been decoded first
  (check-true (equal? (take decoded-tokens 3) '(100 101 102)))
  (printf "  [PASS] define-generation-loop generates tokens and stops at EOG\n"))

;; ============================================================
;; Test 11: define-inference-pipeline — compile-time graph opt
;;
;; The macro emits unqualified identifiers like `llama-model-load`
;; with the use-site's lexical context. We stub them here so this
;; test stays pure-Racket (no FFI / no llama.cpp init).
;; ============================================================

(define call-log (box '()))
(define (logcall! tag . args)
  (set-box! call-log (cons (cons tag args) (unbox call-log))))

(define (reset-log!) (set-box! call-log '()))
(define (log-ops)    (reverse (map car (unbox call-log))))

(define model-counter 0)
(define ctx-counter   0)
(define sampler-counter 0)

(define (llama-model-load path)
  (set! model-counter (add1 model-counter))
  (define h (string->symbol (format "model-~a" model-counter)))
  (logcall! 'load-model path h) h)
(define (llama-model-free h)        (logcall! 'free-model h))
(define (llama-context-create m)
  (set! ctx-counter (add1 ctx-counter))
  (define h (string->symbol (format "ctx-~a" ctx-counter)))
  (logcall! 'create-context m h) h)
(define (llama-context-free h)      (logcall! 'free-context h))
(define (llama-sampler-create-greedy)
  (set! sampler-counter (add1 sampler-counter))
  (define h (string->symbol (format "smp-~a" sampler-counter)))
  (logcall! 'create-sampler-greedy h) h)
(define (llama-sampler-free h)      (logcall! 'free-sampler h))
(define (llama-tokenize m text)     (logcall! 'tokenize m text) '(1 2 3))
(define (llama-decode c toks)       (logcall! 'decode c toks) 'decoded)
(define (llama-sampler-sample s c)  (logcall! 'sample s c) 42)

(test-case "define-inference-pipeline compiles and runs with stubs"
  (reset-log!)
  (set! model-counter 0) (set! ctx-counter 0) (set! sampler-counter 0)

  (define-inference-pipeline (run path prompt)
    (sample (create-sampler-greedy)
            (decode (create-context (load-model path))
                    (tokenize (load-model path) prompt))))

  (define result (run "m.gguf" "hello"))
  (check-equal? result 42)
  (printf "  [PASS] pipeline returns final op's result\n"))

(test-case "pipeline inserts free ops after each alloc's last use"
  (reset-log!)
  (set! model-counter 0) (set! ctx-counter 0) (set! sampler-counter 0)

  (define-inference-pipeline (run2 path prompt)
    (sample (create-sampler-greedy)
            (decode (create-context (load-model path))
                    (tokenize (load-model path) prompt))))

  (run2 "m.gguf" "hi")
  (define ops (log-ops))
  ;; Every alloc has a matching free.
  (check-equal? (length (filter (lambda (o) (eq? o 'load-model))    ops)) 2)
  (check-equal? (length (filter (lambda (o) (eq? o 'free-model))    ops)) 2)
  (check-equal? (length (filter (lambda (o) (eq? o 'create-context)) ops)) 1)
  (check-equal? (length (filter (lambda (o) (eq? o 'free-context))   ops)) 1)
  (check-equal? (length (filter (lambda (o) (eq? o 'create-sampler-greedy)) ops)) 1)
  (check-equal? (length (filter (lambda (o) (eq? o 'free-sampler))   ops)) 1)
  (printf "  [PASS] every alloc has a matching free in the call log\n"))

(test-case "pipeline frees context after its last use (decode)"
  (reset-log!)
  (set! model-counter 0) (set! ctx-counter 0) (set! sampler-counter 0)

  (define-inference-pipeline (run3 path prompt)
    (decode (create-context (load-model path))
            (tokenize (load-model path) prompt)))

  (run3 "m.gguf" "hi")
  (define ops (log-ops))
  ;; decode must come before free-context; free-context must come
  ;; before any tail ops (there are none here, but ordering is what
  ;; we care about).
  (check-true (< (index-of ops 'decode)
                 (index-of ops 'free-context)))
  (printf "  [PASS] free-context appears after decode (its last use)\n"))

(test-case "pipeline let* shares an alloc — one load, two references"
  (reset-log!)
  (set! model-counter 0) (set! ctx-counter 0) (set! sampler-counter 0)

  (define-inference-pipeline (run-shared path prompt)
    (let* ([m (load-model path)])
      (decode (create-context m)
              (tokenize m prompt))))

  (run-shared "m.gguf" "hi")
  (define ops (log-ops))
  ;; Exactly one load + one free for the shared model.
  (check-equal? (length (filter (lambda (o) (eq? o 'load-model)) ops)) 1)
  (check-equal? (length (filter (lambda (o) (eq? o 'free-model)) ops)) 1)
  ;; Both uses (create-context + tokenize) should reference the same model handle.
  (define load-entry    (findf (lambda (e) (eq? (car e) 'load-model))    (reverse (unbox call-log))))
  (define model-handle  (third load-entry))
  (define ctx-entry     (findf (lambda (e) (eq? (car e) 'create-context)) (reverse (unbox call-log))))
  (define tok-entry     (findf (lambda (e) (eq? (car e) 'tokenize))       (reverse (unbox call-log))))
  (check-equal? (second ctx-entry) model-handle)
  (check-equal? (second tok-entry) model-handle)
  ;; free-model appears after the LAST use (tokenize comes after create-context in topo order).
  (check-true (< (index-of ops 'tokenize) (index-of ops 'free-model)))
  (check-true (< (index-of ops 'create-context) (index-of ops 'free-model)))
  (printf "  [PASS] let*-shared model is loaded once, both uses see same handle, freed after last use\n"))

(test-case "create-context captures model — free-model extended to decode"
  (define expanded
    (syntax->datum
     (expand-once
      #'(define-inference-pipeline (decode-once path prompt)
          (let* ([m (load-model path)]
                 [c (create-context m)]
                 [t (tokenize m prompt)])
            (decode c t))))))
  (define body (caddr expanded))
  (define bindings (cadr body))
  (define ops-in-order
    (for/list ([b bindings])
      (car (cadr b))))
  ;; llama-model-free must come AFTER llama-decode. Naively (without
  ;; capture propagation) it would appear after llama-tokenize and
  ;; point at freed memory during decode.
  (define free-model-idx (index-of ops-in-order 'llama-model-free))
  (define decode-idx     (index-of ops-in-order 'llama-decode))
  (check-not-false free-model-idx "expected llama-model-free in expansion")
  (check-true (> free-model-idx decode-idx)
              "llama-model-free must follow llama-decode when ctx captures model")
  (printf "  [PASS] capture propagation extends model liveness past decode\n"))

(test-case "pipeline options inject #:n-ctx / #:n-batch on create-context"
  (define expanded
    (syntax->datum
     (expand-once
      #'(define-inference-pipeline (run-small path prompt)
          #:max-prompt-tokens 16
          #:max-gen-tokens 8
          (let* ([m (load-model path)]
                 [c (create-context m)])
            (tokenize m prompt))))))
  ;; Find the binding whose RHS calls llama-context-create and
  ;; check that it was emitted with keyword args.
  (define body (caddr expanded))
  (define ctx-binding
    (findf (lambda (b)
             (and (pair? (cadr b))
                  (eq? (car (cadr b)) 'llama-context-create)))
           (cadr body)))
  (check-not-false ctx-binding "expected an llama-context-create binding")
  (define rhs (cadr ctx-binding))
  ;; Expected: (llama-context-create m #:n-ctx 24 #:n-batch 24)
  (check-not-false (member '#:n-ctx   rhs) "missing #:n-ctx kwarg in emission")
  (check-not-false (member '#:n-batch rhs) "missing #:n-batch kwarg in emission")
  (define n-ctx-val (cadr (member '#:n-ctx rhs)))
  ;; Derivation floors at 32 (llama.cpp practical minimum), so
  ;; max-prompt=16 + max-gen=8 → 24, floored to 32.
  (check-equal? n-ctx-val 32 "n-ctx should be max(32, mpt+mgt)")
  (printf "  [PASS] options inject derived n-ctx/n-batch into create-context\n"))

(test-case "pipeline without options emits plain create-context call"
  (define expanded
    (syntax->datum
     (expand-once
      #'(define-inference-pipeline (run-default path prompt)
          (let* ([m (load-model path)]
                 [c (create-context m)])
            (tokenize m prompt))))))
  (define body (caddr expanded))
  (define ctx-binding
    (findf (lambda (b)
             (and (pair? (cadr b))
                  (eq? (car (cadr b)) 'llama-context-create)))
           (cadr body)))
  (define rhs (cadr ctx-binding))
  ;; Expected: (llama-context-create m) — no kwargs
  (check-false (member '#:n-ctx   rhs) "unexpected #:n-ctx on default pipeline")
  (check-false (member '#:n-batch rhs) "unexpected #:n-batch on default pipeline")
  (printf "  [PASS] no options → no kwargs on create-context (falls back to llama defaults)\n"))

(test-case "pipeline explicit #:n-ctx overrides derived value"
  (define expanded
    (syntax->datum
     (expand-once
      #'(define-inference-pipeline (run-override path prompt)
          #:max-prompt-tokens 16
          #:max-gen-tokens   8
          #:n-ctx            256
          (let* ([m (load-model path)]
                 [c (create-context m)])
            (tokenize m prompt))))))
  (define body (caddr expanded))
  (define ctx-binding
    (findf (lambda (b)
             (and (pair? (cadr b))
                  (eq? (car (cadr b)) 'llama-context-create)))
           (cadr body)))
  (define rhs (cadr ctx-binding))
  (check-equal? (cadr (member '#:n-ctx rhs)) 256
                "explicit #:n-ctx should win over derived")
  (printf "  [PASS] explicit #:n-ctx wins over max-prompt-tokens+max-gen-tokens\n"))

(test-case "pipeline rejects unknown options at compile time"
  (check-exn exn:fail:syntax?
             (lambda ()
               (expand-once
                #'(define-inference-pipeline (bad path)
                    #:this-is-not-a-real-option 42
                    (load-model path)))))
  (printf "  [PASS] unknown option raises syntax error\n"))

(test-case "pipeline emits a flat let* (no nested function calls on alloc results)"
  (define expanded
    (syntax->datum
     (expand-once
      #'(define-inference-pipeline (pipe-shape m p)
          (decode (create-context (load-model m))
                  (tokenize (load-model m) p))))))
  ;; Expansion should be: (define (pipe-shape m p) (let* ([...] ...) <id>))
  (check-equal? (car expanded) 'define)
  (define body (caddr expanded))
  (check-equal? (car body) 'let*)
  ;; Every let* binding's RHS is a flat (fn arg ...) call — not a nested
  ;; composition like (fn (other-fn ...)).
  (for ([b (in-list (cadr body))])
    (define rhs (cadr b))
    (for ([arg (in-list (cdr rhs))])
      (check-false (pair? arg)
                   (format "binding ~a has nested call in arg: ~a" (car b) arg))))
  (printf "  [PASS] expansion is a flat let* with no nested op calls\n"))

;; ============================================================
;; KV precision compile-time emission tests
;; ============================================================

;; Helper: pull the llama-context-create binding out of an expansion.
(define (find-context-rhs expanded-datum)
  (define body (caddr expanded-datum))
  (define bindings (cadr body))
  (define ctx-binding
    (findf (lambda (b)
             (and (pair? (cadr b))
                  (eq? (car (cadr b)) 'llama-context-create)))
           bindings))
  (and ctx-binding (cadr ctx-binding)))

(test-case "pipeline #:kv-precision 'q8 emits both #:cache-type-k and #:cache-type-v as q8_0"
  (define expanded
    (syntax->datum
     (expand-once
      #'(define-inference-pipeline (run-q8 path prompt)
          #:kv-precision 'q8
          (let* ([m (load-model path)]
                 [c (create-context m)])
            (tokenize m prompt))))))
  (define rhs (find-context-rhs expanded))
  (check-not-false rhs "expected an llama-context-create binding")
  (check-not-false (member '#:cache-type-k rhs) "missing #:cache-type-k kwarg")
  (check-not-false (member '#:cache-type-v rhs) "missing #:cache-type-v kwarg")
  ;; Values are emitted as (quote q8_0) so the symbol is a literal, not a ref.
  (check-equal? (cadr (member '#:cache-type-k rhs)) '(quote q8_0))
  (check-equal? (cadr (member '#:cache-type-v rhs)) '(quote q8_0))
  (printf "  [PASS] #:kv-precision 'q8 → both cache types normalized to q8_0\n"))

(test-case "pipeline split-form #:kv-precision-k / -v emits asymmetric cache types"
  (define expanded
    (syntax->datum
     (expand-once
      #'(define-inference-pipeline (run-split path prompt)
          #:kv-precision-k 'q4
          #:kv-precision-v 'q8
          (let* ([m (load-model path)]
                 [c (create-context m)])
            (tokenize m prompt))))))
  (define rhs (find-context-rhs expanded))
  (check-equal? (cadr (member '#:cache-type-k rhs)) '(quote q4_0)
                "split form should emit q4_0 for K")
  (check-equal? (cadr (member '#:cache-type-v rhs)) '(quote q8_0)
                "split form should emit q8_0 for V")
  (printf "  [PASS] split #:kv-precision-k/-v emits asymmetric q4_0/q8_0\n"))

(test-case "pipeline rejects unknown kv-precision value at compile time"
  (check-exn exn:fail:syntax?
             (lambda ()
               (expand-once
                #'(define-inference-pipeline (bad-kv path)
                    #:kv-precision 'q3
                    (let* ([m (load-model path)]
                           [c (create-context m)])
                      (tokenize m path))))))
  (printf "  [PASS] unknown #:kv-precision value (q3) raises syntax error\n"))

(test-case "pipeline explicit #:cache-type-k overrides #:kv-precision"
  (define expanded
    (syntax->datum
     (expand-once
      #'(define-inference-pipeline (run-override-kv path prompt)
          #:kv-precision   'q8
          #:cache-type-k   'q4
          (let* ([m (load-model path)]
                 [c (create-context m)])
            (tokenize m prompt))))))
  (define rhs (find-context-rhs expanded))
  ;; Explicit #:cache-type-k should beat shared #:kv-precision for K.
  (check-equal? (cadr (member '#:cache-type-k rhs)) '(quote q4_0)
                "explicit #:cache-type-k should win over #:kv-precision")
  ;; V has no explicit override, so it should still fall through to kv-precision.
  (check-equal? (cadr (member '#:cache-type-v rhs)) '(quote q8_0)
                "#:cache-type-v without explicit override should come from #:kv-precision")
  (printf "  [PASS] explicit #:cache-type-k wins over #:kv-precision, V falls through\n"))

(test-case "pipeline #:cache-type-k accepts long form 'q8_0 as well as alias 'q8"
  (define expanded
    (syntax->datum
     (expand-once
      #'(define-inference-pipeline (run-long path prompt)
          #:cache-type-k 'q8_0
          #:cache-type-v 'f16
          (let* ([m (load-model path)]
                 [c (create-context m)])
            (tokenize m prompt))))))
  (define rhs (find-context-rhs expanded))
  (check-equal? (cadr (member '#:cache-type-k rhs)) '(quote q8_0))
  (check-equal? (cadr (member '#:cache-type-v rhs)) '(quote f16))
  (printf "  [PASS] long-form 'q8_0 and 'f16 accepted at explicit cache-type layer\n"))

(test-case "pipeline with only sizing options does NOT emit cache-type kwargs"
  (define expanded
    (syntax->datum
     (expand-once
      #'(define-inference-pipeline (run-sizing-only path prompt)
          #:n-ctx   512
          #:n-batch 128
          (let* ([m (load-model path)]
                 [c (create-context m)])
            (tokenize m prompt))))))
  (define rhs (find-context-rhs expanded))
  (check-not-false (member '#:n-ctx   rhs))
  (check-not-false (member '#:n-batch rhs))
  (check-false (member '#:cache-type-k rhs)
               "cache-type-k should be absent when no KV options given")
  (check-false (member '#:cache-type-v rhs)
               "cache-type-v should be absent when no KV options given")
  (printf "  [PASS] sizing-only pipeline omits cache-type kwargs (backend default applies)\n"))

(test-case "pipeline with only cache-type does NOT emit sizing kwargs"
  (define expanded
    (syntax->datum
     (expand-once
      #'(define-inference-pipeline (run-kv-only path prompt)
          #:kv-precision 'q8
          (let* ([m (load-model path)]
                 [c (create-context m)])
            (tokenize m prompt))))))
  (define rhs (find-context-rhs expanded))
  (check-false (member '#:n-ctx   rhs)
               "n-ctx should be absent when no sizing options given")
  (check-false (member '#:n-batch rhs)
               "n-batch should be absent when no sizing options given")
  (check-not-false (member '#:cache-type-k rhs))
  (check-not-false (member '#:cache-type-v rhs))
  (printf "  [PASS] kv-only pipeline omits sizing kwargs (incremental kwarg emission)\n"))

;; ============================================================

(printf "\nAll graph tests passed!\n")
