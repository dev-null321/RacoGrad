# RacoGrad compile-time inference pipeline — next step handoff

## Where things stand

`graph.rkt` has a working `define-inference-pipeline` macro that parses an
S-expression body, builds a DAG of `pnode` structs at phase 1, runs
last-use + capture-propagation liveness analysis, and emits a flat
`let*` of real FFI calls into `llama_backend.rkt`. 24 tests pass in
`tests/test_graph.rkt`.

Pipeline options `#:max-prompt-tokens` / `#:max-gen-tokens` / `#:n-ctx` /
`#:n-batch` are threaded into `llama-context-create` as baked-in kwargs
at expansion time. The macro also emits matching `free-*` calls at each
alloc's last use, and `create-context` is registered as capturing its
model arg (see `pipeline-op-captures` in `graph.rkt`) so `free-model`
is extended past any op that uses the context.

The CLI in `inference.rkt`'s `module+ main` exposes the same knobs as
`-c` / `-b` for non-Racket users. See `USAGE.md`.

## Measured savings (same prompt, same model, vary `-c`/`-b`)

| Model                         | Baseline (c=2048 b=512) | Tuned (c=512 b=128) | Saved       |
|-------------------------------|-------------------------|---------------------|-------------|
| Gemma-4 31B f16 (dense, wide) | 2311 MiB tunable        | 254 MiB             | ~2.0 GB     |
| Qwen3.6-35B-A3B Q8 (hybrid)   | 545 MiB                 | 135 MiB             | ~0.4 GB     |
| gpt-oss-120b Q4_K_S (MoE+GQA) | 490 MiB                 | 121 MiB             | ~0.37 GB    |

Proportional reduction is ~75% in all cases. Absolute savings scale
with baseline KV cache size, which depends on n_layers × n_kv_heads ×
head_dim × dtype_bytes × (fraction of attention layers). Modern
architectures (GQA, hybrid recurrent layers, small head_dim) have
small KV caches to begin with; that's why the absolute wins shrink on
newer models.

## Next optimization: KV precision at compile time

This is the high-value, low-effort next step. llama.cpp supports a
separate KV cache precision (`cache_type_k`, `cache_type_v` — default
f16). Setting them to `q8_0` halves the cache; `q4_0` quarters it.
Currently not plumbed through the FFI or the macro.

### Scope

1. **FFI layer (`ffi/racograd_llama_ffi.cpp`):** `rg_llama_context_create`
   currently takes `(model_h, n_ctx, n_batch)`. Extend to also take
   `cache_type_k_int` / `cache_type_v_int`, map them to
   `llama_context_params.type_k` / `.type_v` via `GGML_TYPE_*` enum
   values. See llama.cpp `include/llama.h` `llama_context_params`.

2. **Racket FFI binding (`llama_backend.rkt`):** extend
   `llama-context-create` to accept `#:cache-type-k` and
   `#:cache-type-v` keyword args (symbols like `'f16`, `'q8_0`,
   `'q4_0`) and translate to the int constants the FFI expects.

3. **Compile-time macro (`graph.rkt`):**

   - Add `#:kv-precision` to `known-pipeline-options` (allowed values:
     `'f16` / `'q8` / `'q4`) and to the Q/V-separate variants
     `#:kv-precision-k` / `#:kv-precision-v` if users want them split.
   - In `emit-alloc-rhs`, when the op is `create-context` and
     `#:kv-precision` is set, splice `#:cache-type-k`
     and `#:cache-type-v` into the emitted call alongside `#:n-ctx`
     and `#:n-batch`. Exactly the same pattern we already use for
     n-ctx/n-batch — look at how that's done and mirror it.

4. **Tests (`tests/test_graph.rkt`):** three new `test-case`s:

   - `#:kv-precision 'q8` injects both `#:cache-type-k 'q8_0` and
     `#:cache-type-v 'q8_0` into the emitted `llama-context-create`.
   - Split form: `#:kv-precision-k 'q4 #:kv-precision-v 'q8`
     emits the asymmetric combo.
   - Unknown value (e.g. `'q3`) raises a compile-time syntax error.

### Expected savings

On gpt-oss-120b Q4 with our current tuned settings (c=512 b=128),
baseline KV is 18 MiB. With `#:kv-precision 'q8` that drops to ~9 MiB,
with `'q4` to ~4.5 MiB. Small absolute bytes on this one model, but
it *compounds* with the `n_ctx` shrink we already do, and on
full-attention f16-KV models (the Gemma-family case) this is the
difference between "~2 GB saved" and "~3 GB saved."

More importantly: this is the optimization the current model landscape
actually cares about. `n_ctx` tuning is 2018-era. KV precision is 2024.

## Things NOT to touch on this change

- The existing capture-propagation logic in `pipeline-last-use` is
  load-bearing: removing it causes the emitted `free-model` to run
  before `decode`, dangling the model pointer inside the context. The
  test `create-context captures model — free-model extended to decode`
  guards this; don't let it regress.
- The hygienic identifier trick in `pipeline-op-table` — stored as
  plain symbols, wrapped with `datum->syntax ctx` at emission — is
  required for emitted `llama-*` names to resolve at the caller's use
  site. Don't try to store `#'llama-foo` syntax literals in the table.
- The CLI sizing guide in `USAGE.md` references the 80→110 GB
  `MemoryMax` change we made — that's a server-protection note for the
  specific DGX this runs on, not a portable recommendation. Leave it.

## Files for codex to read first

1. `graph.rkt` — whole file, especially `pipeline-op-table` (~line 208),
   `parse-pipeline-options` (~line 414), `emit-alloc-rhs` (~line 353),
   `define-inference-pipeline` (~line 533).
2. `llama_backend.rkt` — the `llama-context-create` definition (search
   for `rg_llama_context_create`).
3. `ffi/racograd_llama_ffi.cpp` — `rg_llama_context_create` function
   body.
4. `tests/test_graph.rkt` — the existing
   `pipeline options inject #:n-ctx / #:n-batch on create-context`
   test is the exact template to copy for the new KV-precision tests.

## Build / run

- FFI rebuild: `./compile_llama.sh` (produces `ffi/libracograd_llama_ffi.so`).
- Tests: `racket tests/test_graph.rkt`.
- End-to-end: `racket inference.rkt -m /path/to/model.gguf -p "..." -c 512 -b 128 --greedy --ngl 99`.
