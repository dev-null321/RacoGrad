# Unified Context API + KV Cache Precision Spec

## Goal

Make `llama-context-create` the single source of truth for inference context sizing and KV-cache storage options. The compile-time macro should remain responsible for graph/liveness lowering, but it should stop owning backend-specific option semantics.

## Scope

This change should cover:

1. `llama_backend.rkt`
2. `ffi/racograd_llama_ffi.cpp`
3. `inference.rkt`
4. `graph.rkt`
5. `tests/test_graph.rkt`
6. `USAGE.md`

Do not change the last-use / capture-propagation behavior except as needed to thread new kwargs through emitted `create-context` calls.

## Source Of Truth

`llama-context-create` in `llama_backend.rkt` is the canonical API for context creation options.

The macro in `graph.rkt` may:

- compute compile-time values
- validate compile-time-only convenience options
- splice kwargs into emitted calls

The macro should not:

- know llama.cpp enum integers
- know FFI encoding details
- become the authoritative definition of context options

## Target Backend API

Target signature:

```racket
(define (llama-context-create model-h
                              #:n-ctx        [n-ctx 2048]
                              #:n-batch      [n-batch 512]
                              #:cache-type-k [cache-type-k 'f16]
                              #:cache-type-v [cache-type-v 'f16])
  ...)
```

This is the minimum required surface for this change.

Optional future extension points like rope scaling may be added later, but they are not required for this task.

## Accepted Backend Values

Accepted symbolic values for `#:cache-type-k` and `#:cache-type-v`:

- `'f16`
- `'q8_0`
- `'q4_0`

Backend normalization may also accept these aliases:

- `'q8` -> `'q8_0`
- `'q4` -> `'q4_0`

If an unsupported symbol is provided, raise a clear Racket error in `llama-context-create`.

Example error shape:

```racket
(error 'llama-context-create "unsupported cache type: ~a" cache-type-k)
```

The backend should map normalized symbols to the integer constants expected by the C FFI layer.

## FFI Contract

Extend the context-create FFI boundary from:

```text
(model_h, n_ctx, n_batch)
```

to:

```text
(model_h, n_ctx, n_batch, cache_type_k_int, cache_type_v_int)
```

Requirements:

- `racograd_llama_ffi.cpp` should set:
  - `params.type_k`
  - `params.type_v`
- The mapping must use `GGML_TYPE_*` values compatible with llama.cpp / ggml.
- The FFI layer should stay dumb: it accepts ints, not symbols or strings.

If context creation fails, preserve the existing failure behavior.

## CLI Contract

`inference.rkt` should expose runtime parity with the backend API.

Required flags:

- `-c`, `--n-ctx`
- `-b`, `--n-batch`
- `--cache-type-k`
- `--cache-type-v`

Accepted CLI values:

- `f16`
- `q8_0`
- `q4_0`

CLI may also accept friendly aliases:

- `q8` -> `q8_0`
- `q4` -> `q4_0`

CLI should normalize strings to backend symbols before calling `llama-context-create`.

Example target call shape:

```racket
(llama-context-create model-h
                      #:n-ctx n-ctx
                      #:n-batch n-batch
                      #:cache-type-k cache-type-k
                      #:cache-type-v cache-type-v)
```

If the user supplies an unknown cache type, fail fast with a clear CLI error.

## Macro Layering

`graph.rkt` should preserve its graph analysis responsibilities:

- parse pipeline s-expressions
- build pnodes
- run last-use analysis
- run capture-propagation
- emit flat `let*`
- insert `free-*` calls

That part is load-bearing and should remain intact.

The only intended config refactor is:

- `define-inference-pipeline` should lower compile-time-known options into kwargs on `llama-context-create`
- backend option semantics should live in `llama_backend.rkt`, not in the macro

## Macro Options

Split macro options into two categories.

Low-level passthrough options:

- `#:n-ctx`
- `#:n-batch`
- `#:cache-type-k`
- `#:cache-type-v`

Convenience options:

- `#:kv-precision`
- `#:kv-precision-k`
- `#:kv-precision-v`

Recommended semantics:

- `#:kv-precision 'q8` => emits `#:cache-type-k 'q8_0` and `#:cache-type-v 'q8_0`
- `#:kv-precision 'q4` => emits `#:cache-type-k 'q4_0` and `#:cache-type-v 'q4_0`
- `#:kv-precision-k 'q4` => emits `#:cache-type-k 'q4_0`
- `#:kv-precision-v 'q8` => emits `#:cache-type-v 'q8_0`

Allowed convenience values:

- `'f16`
- `'q8`
- `'q4`

Compile-time behavior:

- unknown convenience values should raise syntax errors
- explicit low-level options should win over convenience sugar if both are present

Precedence:

1. explicit `#:cache-type-k` / `#:cache-type-v`
2. split convenience `#:kv-precision-k` / `#:kv-precision-v`
3. shared convenience `#:kv-precision`
4. backend defaults in `llama-context-create`

For sizing options, preserve existing behavior:

1. explicit `#:n-ctx`
2. derived `max-prompt-tokens + max-gen-tokens`, floored as today
3. backend default

And:

1. explicit `#:n-batch`
2. derived from `n-ctx` as today
3. backend default

## Emission Guidance

`emit-alloc-rhs` should move toward incremental kwarg assembly rather than one hard-coded branch that only emits kwargs when both `n-ctx` and `n-batch` are present.

Desired behavior:

- each optional kwarg is emitted independently when present
- absence of one kwarg should not suppress other emitted kwargs

Examples:

```racket
(llama-context-create m)
```

```racket
(llama-context-create m #:n-ctx 256)
```

```racket
(llama-context-create m #:cache-type-k 'q8_0 #:cache-type-v 'q8_0)
```

```racket
(llama-context-create m
                      #:n-ctx 256
                      #:n-batch 128
                      #:cache-type-k 'q4_0
                      #:cache-type-v 'q8_0)
```

## Tests

Add or update tests in `tests/test_graph.rkt` to cover:

1. `#:kv-precision 'q8` emits both:
   - `#:cache-type-k 'q8_0`
   - `#:cache-type-v 'q8_0`

2. Split convenience form:
   - `#:kv-precision-k 'q4`
   - `#:kv-precision-v 'q8`
   emits:
   - `#:cache-type-k 'q4_0`
   - `#:cache-type-v 'q8_0`

3. Unknown convenience value like `'q3` raises a compile-time syntax error.

4. Explicit low-level values override convenience values.

5. Existing `#:n-ctx` / `#:n-batch` tests continue to pass.

If backend tests already exist or are easy to add, also verify:

- `llama-context-create` accepts `'f16`, `'q8_0`, `'q4_0`
- aliases `'q8` and `'q4` normalize correctly
- unsupported values fail clearly

## Documentation

Update `USAGE.md` and any CLI help text so they do not claim parity unless parity actually exists after this change.

Document clearly:

- `n_ctx` controls KV-cache capacity / long-lived memory
- `n_batch` controls compute-side batch buffer pressure
- cache type controls KV-cache precision

Avoid describing `n_ctx` and `n_batch` as the same optimization.

## Non-Goals

Do not:

- rewrite liveness / capture-propagation
- change resource-free insertion order
- move enum encoding logic into `graph.rkt`
- make the macro the only path to these optimizations

## Preferred Implementation Order

1. Extend `llama-context-create` and its FFI boundary.
2. Add CLI flags and normalization.
3. Refactor macro option plumbing to forward low-level kwargs cleanly.
4. Add convenience macro sugar for `#:kv-precision`.
5. Update tests and docs.
