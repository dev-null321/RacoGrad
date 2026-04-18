# Running any GGUF through RacoGrad

No Racket editing required. All tuning happens via command-line flags.

## Prerequisites

1. Build the llama.cpp FFI bridge once:
   ```
   ./compile_llama.sh
   ```
   Produces `ffi/libracograd_llama_ffi.so`. The script expects llama.cpp
   built at `~/Projects/multivac/llama.cpp`; edit `LLAMA_DIR` in the script
   if yours lives elsewhere.

2. (Optional but strongly recommended on servers you don't sit in front of)
   Arm the hardware watchdog so a stuck job gets auto-reset after 30 s
   instead of freezing the box. Once:
   ```
   sudo sed -i.bak 's/^#RuntimeWatchdogSec=off$/RuntimeWatchdogSec=30/' \
     /etc/systemd/system.conf
   sudo systemctl daemon-reexec
   cat /sys/class/watchdog/watchdog0/state   # should print: active
   ```

## Running a model

```
racket inference.rkt \
  -m /path/to/model.gguf \
  -p "your prompt" \
  -n 64              \   # max tokens to generate
  -c 512             \   # KV-cache context size
  -b 64              \   # compute batch size
  --cache-type-k q8_0 \  # KV cache K precision (optional)
  --cache-type-v q8_0 \  # KV cache V precision (optional)
  --ngl 99           \   # GPU layers to offload (99 = as many as fit)
  --greedy               # deterministic sampling
```

The tuning flags target three **different** memory concerns. Treat them
as independent knobs, not as "two ways to shrink the same thing":

- **`-c` / `--n-ctx`** — sets the KV cache *capacity* (in tokens). KV
  cache is long-lived and scales linearly with this value. This is what
  you want to shrink when you know your prompts + generations are short.
- **`-b` / `--n-batch`** — sets the compute-side *batch buffer*. Scales
  with batch size during prompt processing; does not affect KV cache.
  Shrink this when compute buffer pressure is the problem.
- **`--cache-type-k` / `--cache-type-v`** — sets the KV cache *storage
  precision*. Defaults to `f16` (2 bytes/element). Options: `f16`,
  `q8_0` (halves), `q4_0` (quarters). Aliases `q8` / `q4` accepted.
  This halves or quarters KV cache size at the cost of a small amount
  of precision; compounds with `--n-ctx` shrinkage.

Defaults are `-c 2048 -b 512 --cache-type-k f16 --cache-type-v f16`
(matching llama.cpp defaults).

## Picking the knobs

Rough guide:

| Use case | `-c` | `-b` | `--cache-type-k/v` | Why |
|---|---|---|---|---|
| Short chat (~64 tok prompts, ~128 tok replies) | `256` | `64` | `f16` | Tiny KV cache already; precision doesn't matter |
| Code explanation (~512 tok prompts) | `1024` | `256` | `f16` or `q8_0` | Q8 KV saves another ~50% if tight on VRAM |
| RAG / long docs (~4K tok prompts) | `4096` | `512` | `q8_0` | KV cache dominates; Q8 saves the most here |
| Big gen loop (default) | `2048` | `512` | `f16` | llama.cpp defaults |
| VRAM-starved, accept quality loss | tune | tune | `q4_0` | Cuts KV cache to ~1/4; slight perplexity hit |

Going below what your prompt needs for `-c` will cause llama.cpp to
truncate — check its startup log for `n_ctx = N`; that's what it
actually allocated (it can round up from what you requested).

Quantized KV cache (`q8_0` / `q4_0`) is independent of *weight*
quantization. A Q4-weighted model still uses f16 KV by default.

## Running under a memory cap (recommended)

To guarantee an OOM kills the process instead of the machine:

```
systemd-run --user --scope --quiet -p MemoryMax=110G -- \
  racket inference.rkt -m /path/to/model.gguf -p "..." -c 512 -b 64 --greedy --ngl 99
```

Pick `MemoryMax` based on your actual workload: `(model weight bytes) +
(KV + compute buffers) + ~5 GB slack for Racket/FFI`. On a 119 GiB box,
`110G` fits anything up to ~100 GB of weights (e.g. `gpt-oss-120b` at
Q4 runs ~112 GB peak and needs the full box). Leave ~9 GB for the OS;
swap catches any short spillover. Only go lower if you genuinely have
less RAM to spare.

## Memory savings (measured)

All three buffers scale ~75% together when you halve both `-c` and `-b`.
Absolute savings depend on the model's baseline KV cache size, which is
itself a function of `n_layers × n_kv_heads × head_dim × dtype_bytes ×
(fraction of attention layers)`.

| Model (at defaults → tuned) | Baseline tunable | Tuned | Savings |
|---|---|---|---|
| Gemma-4 31B f16 (dense, wide MHA, all-attention) | 2311 MiB | 254 MiB | ~2.0 GB |
| Qwen3.6-35B-A3B Q8 (MoE + hybrid Gated Delta Net) | 545 MiB | 135 MiB | ~0.4 GB |
| gpt-oss-120b Q4_K_S (MoE + GQA) | 490 MiB | 121 MiB | ~0.37 GB |

Model **weights** are unchanged; this is inference-state memory only.
Newer architectures (GQA, hybrid layers, small head_dim) have already
shrunk the baseline, so absolute wins are smaller on them — the
proportional reduction is still the same ~75%.

### KV precision is independent

Adding `--cache-type-k q8_0 --cache-type-v q8_0` halves the *remaining*
KV cache on top of whatever `-c` already bought you. For
attention-heavy f16-KV models (Gemma-family), this can push total
savings past the numbers above.

## Compile-time pipelines (for Racket users)

If you *do* want to write Racket, `graph.rkt`'s `define-inference-pipeline`
lets you declare compile-time bounds via `#:max-prompt-tokens N` and
`#:max-gen-tokens N`. The macro derives `n_ctx`/`n_batch` at expansion
time and bakes them into the emitted call. See `tests/test_graph.rkt`
for examples; the CLI above delivers the same runtime memory profile
without needing to write Racket.
