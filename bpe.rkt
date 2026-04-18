#lang racket

;; ============================================================
;; GPT-2 byte-level BPE tokenizer, native Racket.
;;
;; Implements the same algorithm as HuggingFace's GPT2Tokenizer:
;;   1. Pre-tokenize input text with the GPT-2 regex
;;      (contractions, letter runs, digit runs, punctuation runs,
;;       trailing whitespace). ASCII-simplified — the reference
;;       regex uses Unicode properties we approximate with class
;;       ranges; that's fine for English prompts.
;;   2. For each pre-token, map bytes 0..255 → a "printable" unicode
;;      character (HF's bytes-to-unicode trick, using Ġ for space).
;;   3. Run BPE merges from merges.txt, lowest-rank pair first.
;;   4. Look the final string tokens up in vocab.json.
;;
;; Exposes:
;;   (make-bpe vocab-path merges-path) → tok
;;   (bpe-encode tok text)              → (listof exact-nonnegative-integer)
;;   (bpe-decode tok ids)               → string
;; ============================================================

(require json
         racket/file
         racket/string)

(provide make-bpe
         bpe-encode
         bpe-decode)

;; ------------------------------------------------------------
;; Byte ↔ unicode mapping (GPT-2's bytes_to_unicode)
;; ------------------------------------------------------------

;; Build the reversible byte-to-unicode mapping that HF GPT-2 uses
;; to avoid whitespace / control characters in the vocab.
;;
;; Printable ASCII + Latin-1 supplement bytes map to themselves; every
;; other byte gets remapped to codepoint (256 + n) so vocab strings are
;; always "clean" unicode.
(define (build-bytes-to-unicode)
  (define (printable-byte? b)
    (or (and (>= b 33) (<= b 126))
        (and (>= b 161) (<= b 172))
        (and (>= b 174) (<= b 255))))
  (define b->u (make-hasheqv))
  (define u->b (make-hasheqv))
  (define n 0)
  ;; First pass: printable bytes map to themselves
  (for ([b (in-range 256)]
        #:when (printable-byte? b))
    (define ch (integer->char b))
    (hash-set! b->u b ch)
    (hash-set! u->b ch b))
  ;; Second pass: non-printable bytes get remapped to (256 + n)
  (for ([b (in-range 256)]
        #:unless (printable-byte? b))
    (define ch (integer->char (+ 256 n)))
    (hash-set! b->u b ch)
    (hash-set! u->b ch b)
    (set! n (add1 n)))
  (values b->u u->b))

(define-values (BYTE->UNICODE UNICODE->BYTE) (build-bytes-to-unicode))

(define (bytes->unicode-string bs)
  (apply string
         (for/list ([b (in-bytes bs)])
           (hash-ref BYTE->UNICODE b))))

(define (unicode-string->bytes str)
  (apply bytes
         (for/list ([ch (in-string str)])
           (hash-ref UNICODE->BYTE ch
                     (lambda () (error 'bpe-decode "unknown char in token: ~a" ch))))))

;; ------------------------------------------------------------
;; Pre-tokenization regex
;; ------------------------------------------------------------
;;
;; HF's canonical GPT-2 regex (Python `regex` module):
;;   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
;;
;; Racket's pregexp doesn't support \p{L} / \p{N}. We approximate with
;; [A-Za-z] / [0-9] + [\u00c0-\u024f] for common Latin letters. That
;; covers English prompts; users with accented or non-Latin input may
;; get slightly-different tokenization than HF, which propagates into
;; the model as different input tokens but still produces coherent text.

(define GPT2-REGEX
  (pregexp
   (string-append
    "'s|'t|'re|'ve|'m|'ll|'d"
    "| ?[A-Za-z\u00c0-\u024f]+"
    "| ?[0-9]+"
    "| ?[^\\s[:alnum:]]+"
    "|\\s+")))

(define (pre-tokenize text)
  (regexp-match* GPT2-REGEX text))

;; ------------------------------------------------------------
;; BPE merge table
;; ------------------------------------------------------------

;; Parse merges.txt into a hash: (cons a b) → rank (lower = earlier / preferred).
;; The first line of merges.txt is a version header and is skipped.
(define (load-merges merges-path)
  (define lines (file->lines merges-path))
  (define rules
    (cond
      [(and (pair? lines) (regexp-match? #rx"^#" (first lines)))
       (rest lines)]
      [else lines]))
  (define ranks (make-hash))
  (for ([line (in-list rules)]
        [i    (in-naturals)])
    (define parts (string-split line " "))
    (when (= 2 (length parts))
      (hash-set! ranks (cons (first parts) (second parts)) i)))
  ranks)

;; Get the pair with the lowest rank (most-preferred merge) in `pairs`.
;; Returns (values best-pair best-rank) or (values #f #f) if no pair is mergeable.
(define (best-pair pairs ranks)
  (for/fold ([best #f] [best-rank #f])
            ([p (in-list pairs)])
    (define r (hash-ref ranks p #f))
    (cond
      [(not r) (values best best-rank)]
      [(or (not best-rank) (< r best-rank)) (values p r)]
      [else (values best best-rank)])))

;; Return the list of adjacent pairs in a token sequence.
(define (adjacent-pairs toks)
  (cond
    [(< (length toks) 2) '()]
    [else
     (for/list ([a (in-list toks)]
                [b (in-list (rest toks))])
       (cons a b))]))

;; Merge every occurrence of `(cons a b)` in `toks` into "ab".
(define (merge-pair toks a b)
  (let loop ([acc '()] [rem toks])
    (cond
      [(null? rem) (reverse acc)]
      [(and (pair? (cdr rem))
            (equal? (first rem) a)
            (equal? (second rem) b))
       (loop (cons (string-append a b) acc) (cddr rem))]
      [else
       (loop (cons (first rem) acc) (cdr rem))])))

;; Apply BPE to a unicode string. Returns a list of subword strings.
;; Repeatedly finds the lowest-ranked adjacent mergeable pair and merges it.
(define (bpe-apply word ranks cache)
  (cond
    [(hash-ref cache word #f) => values]
    [else
     (define initial (map string (string->list word)))
     (define result
       (let loop ([toks initial])
         (cond
           [(< (length toks) 2) toks]
           [else
            (define-values (pair _) (best-pair (adjacent-pairs toks) ranks))
            (cond
              [(not pair) toks]
              [else (loop (merge-pair toks (car pair) (cdr pair)))])])))
     (hash-set! cache word result)
     result]))

;; ------------------------------------------------------------
;; Top-level: tokenizer struct + encode/decode
;; ------------------------------------------------------------

(struct bpe (vocab id->token ranks cache) #:transparent)

(define (make-bpe vocab-path merges-path)
  (define vocab-raw (call-with-input-file vocab-path read-json))
  (define vocab (make-hash))
  (define id->token (make-hash))
  (for ([(k v) (in-hash vocab-raw)])
    (define key (if (symbol? k) (symbol->string k) k))
    (hash-set! vocab key v)
    (hash-set! id->token v key))
  (bpe vocab id->token (load-merges merges-path) (make-hash)))

(define (bpe-encode tok text)
  (define ids '())
  (for ([chunk (in-list (pre-tokenize text))])
    ;; UTF-8 byte view → HF unicode mapping
    (define word (bytes->unicode-string (string->bytes/utf-8 chunk)))
    (define subwords (bpe-apply word (bpe-ranks tok) (bpe-cache tok)))
    (for ([sw (in-list subwords)])
      (define id (hash-ref (bpe-vocab tok) sw
                           (lambda ()
                             (error 'bpe-encode "subword not in vocab: ~a" sw))))
      (set! ids (cons id ids))))
  (reverse ids))

(define (bpe-decode tok ids)
  (define strs
    (for/list ([id (in-list ids)])
      (hash-ref (bpe-id->token tok) id
                (lambda () (error 'bpe-decode "unknown token id: ~a" id)))))
  (bytes->string/utf-8
   (unicode-string->bytes (apply string-append strs))))
