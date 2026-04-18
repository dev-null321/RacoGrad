#lang racket

;; ============================================================
;; RacoGrad Computation Graph — S-Expression Inference Graphs
;;
;; Two layers, at two different phases:
;;
;;   1. Runtime IR (phase 0): `graph-from-sexpr`, `graph-optimize`,
;;      `graph-last-use`, `graph-resource-balanced?` — pure Racket
;;      functions that operate on `graph-node` values. Useful for
;;      introspection and unit testing. NOT a compile-time pass.
;;
;;   2. Compile-time macro (phase 1): `define-inference-pipeline`
;;      parses a pipeline s-expression at expansion time, runs the
;;      same last-use / free-insertion pass, and emits a flat `let*`
;;      of real FFI calls into `llama_backend.rkt`. This *is* the
;;      compile-time optimization.
;;
;;   3. RAII helper macro (phase 1): `with-llama-resources` — expands
;;      to nested `dynamic-wind` so cleanup runs LIFO on any exit.
;;
;;   4. Inlining macro (phase 1): `define-generation-loop` — emits a
;;      monomorphic auto-regressive loop so the decode / sample /
;;      eog / detok functions are lexically bound at the use site
;;      rather than resolved through a dispatch table.
;;
;; The graph layer sits above the FFI backend and below the
;; user-facing inference API.
;; ============================================================

(require (for-syntax racket/base
                     racket/list
                     racket/syntax
                     syntax/parse))

(provide
 ;; Core data structures
 (struct-out graph-node)

 ;; Graph construction
 make-node
 graph-from-sexpr

 ;; Graph analysis (pure, testable without models)
 graph-node-count
 graph-collect-nodes
 graph-topological-sort
 graph-last-use
 graph-resource-balanced?
 graph-optimize
 graph->sexpr

 ;; Compile-time macros
 with-llama-resources
 define-generation-loop
 define-inference-pipeline)

;; ============================================================
;; Core data structures (phase 0 — runtime IR)
;; ============================================================

;; A node in the computation graph.
;; id:     unique symbol
;; op:     operation symbol ('load-model, 'create-context, ...)
;; inputs: list of graph-node or literal values this depends on
;; attrs:  hash of keyword arguments
(struct graph-node (id op inputs attrs) #:transparent)

;; Resource operations that require cleanup
(define resource-alloc-ops '(load-model create-context create-sampler))
(define resource-free-ops  '(free-model free-context free-sampler))

;; Maps alloc ops to their corresponding free ops
(define alloc->free
  (hash 'load-model     'free-model
        'create-context 'free-context
        'create-sampler 'free-sampler))

;; ============================================================
;; Graph construction
;; ============================================================

(define node-counter 0)

(define (make-node op inputs #:attrs [attrs (hash)] #:id [id #f])
  (define node-id (or id (begin (set! node-counter (add1 node-counter))
                                (string->symbol (format "~a-~a" op node-counter)))))
  (graph-node node-id op inputs attrs))

(define (graph-from-sexpr sexpr)
  (set! node-counter 0)
  (parse-sexpr sexpr))

(define (parse-sexpr expr)
  (match expr
    [(list op args ...)
     (define parsed-args
       (for/list ([a args])
         (if (list? a) (parse-sexpr a) a)))
     (make-node op parsed-args)]
    [atom atom]))

;; ============================================================
;; Graph analysis — pure functions
;; ============================================================

(define (graph-node-count root)
  (length (graph-collect-nodes root)))

;; Collect all nodes in topological (children-before-parents) order.
(define (graph-collect-nodes root)
  (define seen (mutable-set))
  (define result '())
  (define (walk node)
    (when (and (graph-node? node) (not (set-member? seen (graph-node-id node))))
      (set-add! seen (graph-node-id node))
      (for ([input (graph-node-inputs node)])
        (walk input))
      (set! result (cons node result))))
  (walk root)
  (reverse result))

(define (graph-topological-sort root)
  (graph-collect-nodes root))

;; Last-use analysis: for each node, find the last node that references it.
(define (graph-last-use root)
  (define all-nodes (graph-collect-nodes root))
  (define last-use (make-hash))
  (for ([node all-nodes])
    (for ([input (graph-node-inputs node)])
      (when (graph-node? input)
        (hash-set! last-use (graph-node-id input) (graph-node-id node)))))
  (for ([node all-nodes])
    (unless (hash-has-key? last-use (graph-node-id node))
      (hash-set! last-use (graph-node-id node) (graph-node-id node))))
  last-use)

(define (graph-resource-balanced? root)
  (define all-nodes (graph-collect-nodes root))
  (define allocs
    (for/list ([n all-nodes] #:when (member (graph-node-op n) resource-alloc-ops))
      (graph-node-op n)))
  (define frees
    (for/list ([n all-nodes] #:when (member (graph-node-op n) resource-free-ops))
      (graph-node-op n)))
  (define alloc-counts (count-ops allocs))
  (define free-counts (count-ops frees))
  (for/and ([(op count) (in-hash alloc-counts)])
    (define free-op (hash-ref alloc->free op #f))
    (and free-op (= count (hash-ref free-counts free-op 0)))))

(define (count-ops ops)
  (for/fold ([h (hash)]) ([op ops])
    (hash-set h op (add1 (hash-ref h op 0)))))

;; Insert free nodes after last use of each resource.
(define (graph-optimize root)
  (define all-nodes (graph-collect-nodes root))
  (define last-use-map (graph-last-use root))
  (define resource-nodes
    (for/list ([n all-nodes] #:when (member (graph-node-op n) resource-alloc-ops))
      n))
  (define optimized '())
  (for ([node all-nodes])
    (set! optimized (cons node optimized))
    (for ([rn resource-nodes])
      (when (eq? (hash-ref last-use-map (graph-node-id rn) #f) (graph-node-id node))
        (define free-op (hash-ref alloc->free (graph-node-op rn)))
        (define free-node (make-node free-op (list rn)))
        (set! optimized (cons free-node optimized)))))
  (reverse optimized))

(define (graph->sexpr node)
  (cond
    [(graph-node? node)
     (cons (graph-node-op node)
           (for/list ([input (graph-node-inputs node)])
             (graph->sexpr input)))]
    [else node]))

;; ============================================================
;; Compile-time pipeline machinery (phase 1)
;;
;; These helpers mirror the phase-0 ones but operate directly on
;; syntax so `define-inference-pipeline` can run the last-use /
;; insert-frees pass during macro expansion.
;; ============================================================

(begin-for-syntax

  ;; Compile-time op table.
  ;;   op-symbol → (list emitted-fn-name-symbol kind [free-op-symbol])
  ;; kind ∈ {alloc, free, pure}.
  ;;
  ;; The emitted name is stored as a plain symbol. At emission time we
  ;; wrap it with `datum->syntax ctx` where ctx is the use-site stx,
  ;; so the identifier resolves against bindings visible at the use
  ;; site (e.g. `llama-model-load` imported from llama_backend.rkt).
  ;; If we stored a `#'foo` literal instead, it would carry graph.rkt's
  ;; lexical context and show up unbound.
  ;; Each entry is (list fn-name kind [free-op [captures]]).
  ;; `captures` is a list of arg-indices whose liveness is extended
  ;; to the creating node's own last-use — used for parent→child
  ;; resource aliasing that the dataflow graph can't otherwise see.
  ;; Example: llama_context internally references the model handle,
  ;; so freeing the model while the context is live would dangle.
  (define pipeline-op-table
    (hasheq
     'load-model            (list 'llama-model-load             'alloc 'free-model)
     'free-model            (list 'llama-model-free             'free)
     'create-context        (list 'llama-context-create         'alloc 'free-context '(0))
     'free-context          (list 'llama-context-free           'free)
     'create-sampler        (list 'llama-sampler-create         'alloc 'free-sampler)
     'create-sampler-greedy (list 'llama-sampler-create-greedy  'alloc 'free-sampler)
     'free-sampler          (list 'llama-sampler-free           'free)
     'tokenize              (list 'llama-tokenize               'pure)
     'decode                (list 'llama-decode                 'pure)
     'sample                (list 'llama-sampler-sample         'pure)
     'detokenize            (list 'llama-detokenize             'pure)
     'detokenize-token      (list 'llama-detokenize-token       'pure)
     'is-eog?               (list 'llama-is-eog?                'pure)
     'get-logits            (list 'llama-get-logits             'pure)
     'get-logits-ith        (list 'llama-get-logits-ith         'pure)))

  (define (pipeline-op-known? sym)
    (and (symbol? sym) (hash-has-key? pipeline-op-table sym)))

  (define (pipeline-op-fn sym)    (car  (hash-ref pipeline-op-table sym)))
  (define (pipeline-op-kind sym)  (cadr (hash-ref pipeline-op-table sym)))

  (define (pipeline-op-free sym)
    (define e (hash-ref pipeline-op-table sym))
    (and (>= (length e) 3) (eq? 'alloc (cadr e)) (caddr e)))

  (define (pipeline-op-captures sym)
    (define e (hash-ref pipeline-op-table sym))
    (cond
      [(and (>= (length e) 4) (list? (list-ref e 3))) (list-ref e 3)]
      [else '()]))

  ;; Compile-time node representation. Distinct from the phase-0
  ;; `graph-node` struct — we can't cross the phase barrier with
  ;; runtime struct instances, and this version carries a hygienic
  ;; binding identifier rather than a symbol.
  (struct pnode (id op args bind-id) #:transparent)

  (define pnode-counter (box 0))
  (define (fresh-pnode-id! op-sym)
    (set-box! pnode-counter (add1 (unbox pnode-counter)))
    (string->symbol (format "~a-~a" op-sym (unbox pnode-counter))))

  ;; Accumulator of every pnode created during one macro expansion.
  ;; Populated by `track-pnode!`, reset by `pipeline-parse-top`. We use
  ;; this instead of walking the result tree because a let*-bound node
  ;; that the body never references would otherwise be dropped.
  (define all-parsed-pnodes (box '()))
  (define (track-pnode! p)
    (set-box! all-parsed-pnodes (cons p (unbox all-parsed-pnodes)))
    p)

  ;; Walk syntax and produce either a pnode (for op applications) or
  ;; the original syntax (for opaque expressions / identifiers).
  ;;
  ;; `env` maps symbol → pnode for `let*`-bound intermediates, so the
  ;; user can name an alloc once and reference it from multiple places
  ;; without duplicating the emitted call.
  ;;
  ;; `bind-id` lets the caller override the pnode's emitted binding
  ;; identifier — used by let* handling to reuse the user's name.
  (define (pipeline-parse stx [env (hasheq)] #:bind-id [bind-id #f])
    (syntax-case stx (let*)
      ;; (let* () body) — empty bindings, just descend into body.
      [(let* () body)
       (pipeline-parse #'body env)]
      ;; (let* ([name init] rest ...) body)
      ;; Parse init with the current env and mark its pnode with the
      ;; user's identifier as bind-id. Then register name→pnode and
      ;; recurse with the remaining bindings.
      [(let* ([name init] rest ...) body)
       (identifier? #'name)
       (let ([init-parsed (pipeline-parse #'init env #:bind-id #'name)])
         (unless (pnode? init-parsed)
           (raise-syntax-error
            'define-inference-pipeline
            "let* binding in a pipeline must be an op application"
            stx #'init))
         (pipeline-parse
          #'(let* (rest ...) body)
          (hash-set env (syntax-e #'name) init-parsed)))]
      ;; (op arg ...) with a known op → pnode.
      [(op arg ...)
       (and (identifier? #'op)
            (pipeline-op-known? (syntax-e #'op)))
       (let* ([op-sym (syntax-e #'op)]
              [id     (fresh-pnode-id! op-sym)])
         (track-pnode!
          (pnode id
                 op-sym
                 (map (lambda (a) (pipeline-parse a env)) (syntax->list #'(arg ...)))
                 (or bind-id (generate-temporary id)))))]
      ;; Bare identifier bound by an enclosing let*: reuse its pnode.
      [x
       (and (identifier? #'x)
            (hash-has-key? env (syntax-e #'x)))
       (hash-ref env (syntax-e #'x))]
      ;; Anything else (literals, user arg identifiers, sub-expressions
      ;; using non-op functions) passes through as opaque syntax.
      [_ stx]))

  ;; Returns every pnode seen during parse, in topological order
  ;; (children before parents — guaranteed by post-order creation
  ;; during recursive descent in `pipeline-parse`). Covers orphan
  ;; let*-bound pnodes that the body never references.
  (define (pipeline-collect _tree)
    (reverse (unbox all-parsed-pnodes)))

  ;; Top-level entry: resets per-expansion state and parses stx.
  (define (pipeline-parse-top stx)
    (set-box! pnode-counter 0)
    (set-box! all-parsed-pnodes '())
    (pipeline-parse stx))

  ;; id → id of last pnode that references it.
  ;;
  ;; Two passes:
  ;;   1. Naive dataflow: each arg's last-use = the latest node that
  ;;      mentions it as a direct input.
  ;;   2. Capture propagation: if node N is registered as capturing
  ;;      arg i, the captured pnode's last-use is extended to N's
  ;;      last-use. Done in reverse-topological order so transitive
  ;;      captures (A captured by B, B captured by C) resolve in one
  ;;      pass without a fixpoint.
  (define (pipeline-last-use all)
    (define m (make-hash))
    (for ([n all])
      (for ([a (pnode-args n)])
        (when (pnode? a)
          (hash-set! m (pnode-id a) (pnode-id n)))))
    (for ([n all])
      (unless (hash-has-key? m (pnode-id n))
        (hash-set! m (pnode-id n) (pnode-id n))))
    ;; Position index for "is X later than Y" comparisons.
    (define pos (make-hash))
    (for ([n all] [i (in-naturals)])
      (hash-set! pos (pnode-id n) i))
    (define (later? a b) (> (hash-ref pos a) (hash-ref pos b)))
    (for ([n (reverse all)])
      (define caps (pipeline-op-captures (pnode-op n)))
      (define n-lu (hash-ref m (pnode-id n)))
      (for ([i caps]
            #:when (< i (length (pnode-args n))))
        (define arg (list-ref (pnode-args n) i))
        (when (pnode? arg)
          (define arg-lu (hash-ref m (pnode-id arg)))
          (when (later? n-lu arg-lu)
            (hash-set! m (pnode-id arg) n-lu)))))
    m)

  ;; An arg is either a pnode (use its bind-id) or opaque syntax (emit as-is).
  (define (pipeline-arg->stx a)
    (if (pnode? a) (pnode-bind-id a) a))

  ;; Convert an op-name symbol into a syntax identifier with the
  ;; use-site's lexical context, so it resolves against bindings
  ;; visible at the use site.
  (define (op-fn-id op-sym ctx)
    (datum->syntax ctx (pipeline-op-fn op-sym)))

  (define (free-fn-id-for op-sym ctx)
    (datum->syntax ctx (pipeline-op-fn (pipeline-op-free op-sym))))

  ;; Derive a concrete n_ctx from pipeline options. Priority:
  ;;   1. Explicit #:n-ctx N wins.
  ;;   2. Otherwise n-ctx = max-prompt-tokens + max-gen-tokens,
  ;;      floored at 32 (llama.cpp minimum practical KV slot count).
  ;;   3. If neither, return #f → emit without the kwarg and let
  ;;      llama-context-create use its default (2048).
  (define (options-compute-n-ctx opts)
    (cond
      [(hash-ref opts '#:n-ctx #f) => values]
      [else
       (define mpt (hash-ref opts '#:max-prompt-tokens #f))
       (define mgt (hash-ref opts '#:max-gen-tokens #f))
       (cond
         [(and mpt mgt) (max 32 (+ mpt mgt))]
         [else #f])]))

  (define (options-compute-n-batch opts n-ctx)
    (cond
      [(hash-ref opts '#:n-batch #f) => values]
      [n-ctx (min n-ctx 512)]
      [else #f]))

  ;; Normalize a cache-type symbol into the backend's canonical form.
  ;; Returns 'f16 | 'q8_0 | 'q4_0 for recognized inputs; #f otherwise.
  ;; The short aliases 'q8 / 'q4 are accepted both at the explicit
  ;; cache-type layer and at the convenience kv-precision layer.
  (define (normalize-cache-type sym)
    (case sym
      [(f16)       'f16]
      [(q8_0 q8)   'q8_0]
      [(q4_0 q4)   'q4_0]
      [else        #f]))

  ;; Precedence for K and V cache types:
  ;;   1. explicit #:cache-type-k / #:cache-type-v
  ;;   2. split convenience #:kv-precision-k / #:kv-precision-v
  ;;   3. shared convenience #:kv-precision
  ;;   4. #f  → omit kwarg, backend default ('f16) applies
  ;; Values at this point are already normalized by parse-pipeline-options,
  ;; so they're one of 'f16 | 'q8_0 | 'q4_0 or absent.
  (define (options-compute-cache-type-k opts)
    (or (hash-ref opts '#:cache-type-k   #f)
        (hash-ref opts '#:kv-precision-k #f)
        (hash-ref opts '#:kv-precision   #f)))

  (define (options-compute-cache-type-v opts)
    (or (hash-ref opts '#:cache-type-v   #f)
        (hash-ref opts '#:kv-precision-v #f)
        (hash-ref opts '#:kv-precision   #f)))

  ;; Build a (possibly empty) list of #'[#:kw stx-val ...] pairs
  ;; for the create-context call. Each kwarg appears iff its
  ;; resolver returned a value; absent kwargs are simply not emitted.
  (define (create-context-kwarg-stxs ctx opts)
    (define n-ctx-val       (options-compute-n-ctx opts))
    (define n-batch-val     (options-compute-n-batch opts n-ctx-val))
    (define cache-k-val     (options-compute-cache-type-k opts))
    (define cache-v-val     (options-compute-cache-type-v opts))
    ;; Integer options → emit the number directly.
    ;; Symbol options → emit (quote sym) so the symbol is a literal,
    ;; not a variable reference, at the call site.
    (define (num-stx n) (datum->syntax ctx n))
    (define (sym-stx s)
      (with-syntax ([literal (datum->syntax ctx s)])
        #'(quote literal)))
    (define (kw-stx key) (datum->syntax ctx key))
    (apply
     append
     (filter
      values
      (list
       (and n-ctx-val   (list (kw-stx '#:n-ctx)        (num-stx n-ctx-val)))
       (and n-batch-val (list (kw-stx '#:n-batch)      (num-stx n-batch-val)))
       (and cache-k-val (list (kw-stx '#:cache-type-k) (sym-stx cache-k-val)))
       (and cache-v-val (list (kw-stx '#:cache-type-v) (sym-stx cache-v-val)))))))

  ;; Build the RHS of an alloc-binding. For most ops that's just
  ;; (fn arg ...). For create-context we splice any compile-time-derived
  ;; kwargs — each one independently emitted (or omitted) based on the
  ;; resolver outcome.
  (define (emit-alloc-rhs n ctx opts)
    (define op-sym (pnode-op n))
    (define fn-stx (op-fn-id op-sym ctx))
    (define arg-stxs (map pipeline-arg->stx (pnode-args n)))
    (define kwarg-stxs
      (cond
        [(eq? op-sym 'create-context) (create-context-kwarg-stxs ctx opts)]
        [else '()]))
    (with-syntax ([fn       fn-stx]
                  [(a ...)  arg-stxs]
                  [(kw ...) kwarg-stxs])
      #'(fn a ... kw ...)))

  ;; Given a parsed body tree, emit syntax of shape (let* ([...] ...) root-id).
  ;; `ctx`  — use-site stx, whose lexical context is transferred to emitted fn ids.
  ;; `opts` — compile-time options hash keyed on keywords (#:n-ctx, etc.).
  (define (pipeline-emit tree ctx opts)
    (cond
      [(pnode? tree)
       (set-box! pnode-counter 0)
       (define all (pipeline-collect tree))
       (define last-use (pipeline-last-use all))
       (define allocs
         (filter (lambda (n) (eq? 'alloc (pipeline-op-kind (pnode-op n)))) all))
       (define bindings
         (apply
          append
          (for/list ([n all])
            (define alloc-binding
              (with-syntax ([id  (pnode-bind-id n)]
                            [rhs (emit-alloc-rhs n ctx opts)])
                (list #'[id rhs])))
            (define frees-here
              (for/list ([a allocs]
                         #:when (eq? (hash-ref last-use (pnode-id a)) (pnode-id n)))
                (with-syntax ([tmp     (generate-temporary '_free)]
                              [free-fn (free-fn-id-for (pnode-op a) ctx)]
                              [id      (pnode-bind-id a)])
                  #'[tmp (free-fn id)])))
            (append alloc-binding frees-here))))
       (with-syntax ([(b ...) bindings]
                     [root    (pnode-bind-id tree)])
         #'(let* (b ...) root))]
      [else
       ;; Body wasn't a pipeline form — pass through unchanged.
       tree]))

  ;; Parse keyword/value pairs at macro-use into an options hash.
  ;; Rejects unknown option names at compile time.
  ;;
  ;; Options fall into two categories:
  ;;   - Integer options (sizing): must be a non-negative integer literal.
  ;;   - Symbol options (KV precision): must normalize via
  ;;     `normalize-cache-type` to 'f16 | 'q8_0 | 'q4_0. Short aliases
  ;;     'q8 / 'q4 are accepted. The normalized form is stored so the
  ;;     resolver doesn't need to re-normalize later.
  (define integer-pipeline-options
    '(#:max-prompt-tokens #:max-gen-tokens #:n-ctx #:n-batch))

  (define symbol-pipeline-options
    '(#:cache-type-k #:cache-type-v
      #:kv-precision #:kv-precision-k #:kv-precision-v))

  ;; Accept symbol options in either quoted or unquoted source form.
  ;; User writes `#:kv-precision 'q8`, so syntax->datum returns `(quote q8)`;
  ;; unwrap to `q8` for downstream validation.
  (define (unwrap-quoted-symbol datum)
    (cond
      [(and (pair? datum)
            (eq? 'quote (car datum))
            (pair? (cdr datum))
            (null? (cddr datum))
            (symbol? (cadr datum)))
       (cadr datum)]
      [else datum]))

  (define (parse-pipeline-options kws vs stx)
    (for/fold ([h (hasheq)])
              ([k (in-list kws)]
               [v (in-list vs)])
      (define key (syntax-e k))
      (define val (syntax->datum v))
      (cond
        [(memq key integer-pipeline-options)
         (unless (exact-nonnegative-integer? val)
           (raise-syntax-error 'define-inference-pipeline
                               (format "option ~a expects a non-negative integer literal" key)
                               stx v))
         (hash-set h key val)]
        [(memq key symbol-pipeline-options)
         (define sym (unwrap-quoted-symbol val))
         (define normalized (and (symbol? sym) (normalize-cache-type sym)))
         (unless normalized
           (raise-syntax-error 'define-inference-pipeline
                               (format "option ~a expects one of: f16 q8_0 q4_0 (aliases q8 q4)" key)
                               stx v))
         (hash-set h key normalized)]
        [else
         (raise-syntax-error 'define-inference-pipeline
                             (format "unknown pipeline option ~a" key)
                             stx k)]))))

;; ============================================================
;; Compile-time macros
;; ============================================================

;; with-llama-resources: RAII-style resource management.
;; Guarantees cleanup via dynamic-wind, even on exceptions.
;; Cleanup runs in reverse (LIFO) order.
;;
;; Usage:
;;   (with-llama-resources
;;     ([model (load-gguf-model "path") (free-gguf-model model)]
;;      [gen   (make-generator model)   (free-generator gen)])
;;     (generate-text gen "Hello"))
(define-syntax (with-llama-resources stx)
  (syntax-parse stx
    [(_ ([name:id init:expr cleanup:expr] ...) body:expr ...+)
     (define bindings (syntax->list #'((name init cleanup) ...)))
     (foldr
      (lambda (binding body-so-far)
        (syntax-parse binding
          [(n i c)
           #`(let ([n i])
               (dynamic-wind
                void
                (lambda () #,body-so-far)
                (lambda () c)))]))
      #'(begin body ...)
      bindings)]
    [(_ ([name:id init:expr] ...) body:expr ...+)
     #'(let ([name init] ...)
         (dynamic-wind
          void
          (lambda () body ...)
          (lambda ()
            (for-each (lambda (v cleanup)
                        (when cleanup (cleanup v)))
                      (list name ...)
                      (list (detect-cleanup name) ...)))))]))

(define (detect-cleanup v)
  (cond
    [(procedure? v) #f]
    [else #f]))

;; define-generation-loop: emits an auto-regressive generation loop
;; where the four backend functions are inlined at the call site.
;;
;; What the macro actually delivers: the generated loop holds direct
;; lexical references to decode-fn / sample-fn / eog-fn / detok-fn,
;; so there's no dispatch-table lookup per iteration. It does NOT
;; magically eliminate the per-iteration list allocation inside
;; decode-fn — that's the backend's concern.
(define-syntax (define-generation-loop stx)
  (syntax-parse stx
    [(_ name:id
        #:decode-fn decode-fn:expr
        #:sample-fn sample-fn:expr
        #:eog-fn    eog-fn:expr
        #:detok-fn  detok-fn:expr)
     #'(define (name prompt-tokens max-tokens callback)
         (decode-fn prompt-tokens)
         (let loop ([generated '()]
                    [n 0])
           (if (>= n max-tokens)
               (reverse generated)
               (let ([new-token (sample-fn)])
                 (cond
                   [(eog-fn new-token)
                    (reverse generated)]
                   [else
                    (when callback
                      (callback new-token (detok-fn new-token)))
                    (decode-fn (list new-token))
                    (loop (cons new-token generated) (add1 n))])))))]))

;; define-inference-pipeline: takes a pipeline s-expression, parses
;; it into a graph at compile time, inserts free-nodes after each
;; alloc's last use, and emits a flat `let*` of FFI calls.
;;
;; The ops in the body must be registered in `pipeline-op-table`
;; above. Their emitted function names (e.g. `llama-model-load`)
;; must be in scope at the use site.
;;
;; Usage:
;;   (define-inference-pipeline (run-once path prompt)
;;     (sample
;;      (decode
;;       (create-context (load-model path))
;;       (tokenize (load-model path) prompt))
;;      (create-sampler-greedy)))
;;
;; Expands (roughly) to:
;;   (define (run-once path prompt)
;;     (let* ([m  (llama-model-load path)]
;;            [c  (llama-context-create m)]
;;            [t  (llama-tokenize m prompt)]
;;            [_1 (llama-model-free m)]        ;; inserted: last use of m is tokenize
;;            [_2 (llama-decode c t)]
;;            [_3 (llama-context-free c)]      ;; inserted: last use of c is decode
;;            [s  (llama-sampler-create-greedy)]
;;            [r  (llama-sampler-sample s c)]
;;            [_4 (llama-sampler-free s)])
;;       r))
(define-syntax (define-inference-pipeline stx)
  (syntax-parse stx
    [(_ (name:id arg:id ...)
        (~seq kw:keyword v:expr) ...
        body:expr)
     (define opts (parse-pipeline-options
                   (syntax->list #'(kw ...))
                   (syntax->list #'(v  ...))
                   stx))
     (with-syntax ([expanded (pipeline-emit (pipeline-parse-top #'body) stx opts)])
       #'(define (name arg ...) expanded))]
    [(_ name:id
        (~seq kw:keyword v:expr) ...
        body:expr)
     (define opts (parse-pipeline-options
                   (syntax->list #'(kw ...))
                   (syntax->list #'(v  ...))
                   stx))
     (with-syntax ([expanded (pipeline-emit (pipeline-parse-top #'body) stx opts)])
       #'(define name expanded))]))
