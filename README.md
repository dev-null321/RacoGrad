# RacoGrad: Deep Learning in Racket

<p align="center">
  <b>A GPU-accelerated deep learning framework written in Racket — train custom architectures with libtorch, run 100B-parameter GGUFs with llama.cpp, <i>zero Python in the execution path</i>.</b>
</p>

---

## What is this?

RacoGrad combines Racket's functional style with two native C FFI backends. Pick the path that matches what you want to do:

| You want to… | Use | Backend |
|---|---|---|
| Train your own transformer / GPT-2 / custom model | `libtorch_backend.rkt` | libtorch + CUDA |
| Run inference on any GGUF (Qwen, Gemma, Llama-3, gpt-oss-120b, …) | `llama_backend.rkt` | llama.cpp + CUDA |
| Declaratively describe inference pipelines with compile-time memory tuning | `graph.rkt` | macro on top of llama_backend |

Both backends run through dedicated C FFI bridges. No pyffi, no subprocess calls, no embedded Python. Weights load natively from `.safetensors`, tokenization is native BPE in Racket, generation runs on the GPU.

## Highlights

- **End-to-end GPT-2** — pretrained HuggingFace weights load directly from `.safetensors`, byte-level BPE tokenizer matches the HF reference exactly, generation runs on CUDA via libtorch. Measured **50 tok/s** on GB10.
- **Transformer training** — full forward → cross-entropy → autograd backward → Adam optimizer path works on GB10. Loss curves visible, parameters actually update.
- **Compile-time inference pipelines** — `define-inference-pipeline` parses an S-expression, builds a DAG at phase 1, runs last-use + capture-propagation liveness analysis, and emits a flat `let*` of real FFI calls with `free-*` ops inserted at the latest safe point. Options (`#:max-prompt-tokens`, `#:n-ctx`, `#:kv-precision`, …) bake into the emitted code as literal kwargs.
- **GGUF inference at scale** — runs gpt-oss-120b (Q4, ~112 GB peak), Qwen3.6-35B-A3B (Q8 hybrid MoE), Gemma-4 31B f16, whatever llama.cpp accepts.
- **Tunable KV memory** — `n_ctx`, `n_batch`, and K/V cache precision (f16 / Q8_0 / Q4_0) exposed on both the macro and the CLI. Measured ~75% reduction in tunable inference buffers.
- **Native safetensors reader** — zero-copy from disk to libtorch tensors on CUDA. Supports f32, f16, bf16.
- **Native BPE tokenizer** — full GPT-2 byte-level BPE (bytes→unicode trick, the official pre-tokenization regex, merge-rank table from `merges.txt`). Round-trips reference prompts to the exact same token IDs HF produces.

## Architecture

```
                    ┌──────────────────────────────────┐
                    │   Racket user modules            │
                    │                                  │
                    │   gpt2.rkt / transformer.rkt     │
                    │   inference.rkt / graph.rkt      │
                    │   training.rkt / safetensors.rkt │
                    │   bpe.rkt / nn.rkt               │
                    └─────────────────┬────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          libtorch_backend.rkt                  llama_backend.rkt
                    │                                   │
                    ▼                                   ▼
         ffi/racograd_ffi.cpp                ffi/racograd_llama_ffi.cpp
                    │                                   │
                    ▼                                   ▼
                libtorch                            llama.cpp
             (C++ / CUDA)                         (C++ / CUDA)
```

## Quick Start

### Prerequisites

- Racket 8.0+
- C++ toolchain (clang / g++ with C++17)
- libtorch with CUDA — tested with the libtorch shipped in the PyTorch 2.10 wheel (installing `torch` via pip puts a full libtorch under `.venv/lib/python*/site-packages/torch/lib/`)
- For GGUF inference: llama.cpp built somewhere on disk (see [compile_llama.sh](compile_llama.sh))
- NVIDIA GPU with CUDA drivers

### Install

```bash
git clone https://github.com/dev-null321/RacoGrad.git
cd RacoGrad
```

### Build the libtorch FFI (for training + custom models)

If you installed torch into a venv at `./.venv`, this just works:

```bash
LIBTORCH_DIR=./.venv/lib/python3.12/site-packages/torch
g++ -O3 -std=c++17 -shared -fPIC \
  ffi/racograd_ffi.cpp \
  -I$LIBTORCH_DIR/include \
  -I$LIBTORCH_DIR/include/torch/csrc/api/include \
  -I/usr/local/cuda/targets/x86_64-linux/include \
  -L$LIBTORCH_DIR/lib \
  -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
  -Wl,-rpath,$LIBTORCH_DIR/lib \
  -o ffi/libracograd_ffi.so
```

(On ARM DGX / SBSA swap `x86_64-linux` → `sbsa-linux` in the CUDA include path.)

### Build the llama.cpp FFI (for GGUF inference)

```bash
./compile_llama.sh
```

This expects llama.cpp already built at `~/Projects/multivac/llama.cpp` — edit `LLAMA_DIR` in the script if yours is elsewhere.

## Examples

### 1. Run GPT-2 on a prompt (end-to-end, native)

```bash
racket gpt2.rkt
```

```
=== RacoGrad GPT-2 Text Generation ===
Loading model...
Loading gpt2 weights from ~/.cache/huggingface/hub/models--gpt2/.../model.safetensors...
Weights loaded!
Model ready!

Prompt: The meaning of life is
Generating...

The meaning of life is not the same as the meaning of death.
The meaning of life is not the same as the meaning of death.
The meaning of

(generated 30 tokens in 0.62s = 48.47 tok/s)
```

The first time you run this it'll complain if `model.safetensors` isn't in your HF cache — grab it with:

```bash
wget -P ~/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/ \
  https://huggingface.co/gpt2/resolve/main/model.safetensors
```

### 2. Train a tiny transformer

```bash
racket tests/train_tiny.rkt
```

```
parameters: 50 tensors, 24608 total weights

step   0  loss = 13.1987
step   1  loss = 10.9534
step   2  loss = 8.5615
step   5  loss = 6.7459
step  10  loss = 4.2929
step  15  loss = 2.9264
step  19  loss = 2.8924
```

### 3. Train a tiny GPT-2 on memorization

```bash
racket tests/train_gpt2_tiny.rkt
```

```
parameters: 30 tensors, 108864 total weights

step  loss
----  ------
   0   10.7756
   5   0.2379
  29   0.0895

Total: 0.64s (46.7 steps/sec)
```

### 4. Run any GGUF through the llama.cpp path

```bash
racket inference.rkt \
  -m /path/to/model.gguf \
  -p "Write a Python function that returns the nth Fibonacci number." \
  -n 100 \
  -c 512 \
  -b 128 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --ngl 99 \
  --greedy
```

See [USAGE.md](USAGE.md) for the full CLI guide and a `-c` / `-b` cheat-sheet by workload.

### 5. Compile-time inference pipeline

```racket
#lang racket
(require "inference.rkt")

(define-inference-pipeline (chat-pipe path prompt)
  #:max-prompt-tokens 512
  #:max-gen-tokens    256
  #:kv-precision      'q8
  (let* ([m (load-model path)]
         [c (create-context m)])
    (decode c (tokenize m prompt))))
```

Expansion (inspectable with `expand-once`):

```racket
(define (chat-pipe path prompt)
  (let* ([m (llama-model-load path)]
         [c (llama-context-create m #:n-ctx 768
                                    #:n-batch 512
                                    #:cache-type-k 'q8_0
                                    #:cache-type-v 'q8_0)]
         [t (llama-tokenize m prompt)]
         [r (llama-decode c t)]
         [_ (llama-context-free c)]
         [_ (llama-model-free m)])
    r))
```

All kwargs are baked in at expansion. `free-model` is inserted after `decode` (not after `tokenize`) because `create-context` is declared to capture its model arg. 31 unit tests cover every combination in [`tests/test_graph.rkt`](tests/test_graph.rkt).

## Module Overview

### Training / custom-model path (libtorch)

| Module | Description |
|---|---|
| `gpt2.rkt` | GPT-2 implementation, `load-gpt2-weights`, `gpt2-module` (for training), BPE-backed tokenize/decode |
| `transformer.rkt` | Encoder/decoder + per-layer transformer modules, wrapped as `nn-module`s for parameter collection |
| `attention.rkt` | Multi-head attention, causal / cross-attention masks |
| `nn.rkt` | Core NN primitives (`Linear`, `Embedding`, `LayerNorm`, `sequential`, `parameters`, `forward`) |
| `training.rkt` | `make-adam-from-params`, `train-step`, `get-loss-value` |
| `safetensors.rkt` | Native binary reader: header-length prefix, JSON metadata, on-demand payload reads, direct `tensor-from-bytes` |
| `bpe.rkt` | Byte-level BPE tokenizer (bytes→unicode, GPT-2 pre-tokenization regex, merge-rank table) |
| `libtorch_backend.rkt` | Racket FFI bindings + high-level ops (`zeros`, `matmul`, `backward`, `make-adam`, `tensor-from-bytes`) |
| `ffi/racograd_ffi.cpp` | C bridge into libtorch / CUDA (handle-based tensors, autograd, Adam, tensor-from-raw-bytes for f32 / f16 / bf16) |

### GGUF inference path (llama.cpp)

| Module | Description |
|---|---|
| `inference.rkt` | High-level inference API + CLI (`with-gguf-model`, `with-generator`, `generate-text`) |
| `graph.rkt` | Compile-time pipeline machinery: `define-inference-pipeline`, capture propagation, kwarg emission |
| `llama_backend.rkt` | Racket FFI bindings for the llama.cpp bridge, owns cache-type symbol ↔ int normalization |
| `ffi/racograd_llama_ffi.cpp` | C bridge into llama.cpp (handle-based, supports f16 / Q8_0 / Q4_0 KV cache) |
| `compile_llama.sh` | One-shot build script for the llama.cpp FFI bridge |

## Benchmarks

All measured on NVIDIA GB10 (ARM DGX Spark). Reproducible — scripts linked in each section.

### GPT-2 small forward pass (libtorch path)

Random weights, batch=1, seq_len=128, 20 iterations after warmup. Run: `racket tests/bench_gpt2_forward.rkt`.

| Metric | Value |
|---|---|
| ms / forward | 20.0 |
| Forwards / sec | 50 |
| Effective tokens / sec* | 6 400 |

\* `seq_len × forwards/sec`. Autoregressive generation is closer to the raw `forwards/sec` number (one forward per new token, growing context).

### GPT-2 end-to-end generation (native, pretrained weights)

Greedy decoding, pretrained `gpt2` small (124M params), 30 tokens on prompt "The meaning of life is". Run: `racket gpt2.rkt`.

| Metric | Value |
|---|---|
| Cold (first call, CUDA warmup) | 48 tok/s |
| Warm (subsequent calls) | 170+ tok/s |

### Training (libtorch path)

Both measured with Adam (lr=0.005–0.01, β defaults). Loss is cross-entropy over the vocab.

**Tiny transformer** (vocab=32, d=32, 1 layer, 2 heads, 24.6K params).  
Run: `racket tests/train_tiny.rkt`

| Step | Loss |
|---|---|
| 0 | 13.20 |
| 5 | 6.75 |
| 10 | 4.29 |
| 19 | 2.89 |

**Tiny GPT-2** (vocab=64, d=64, 2 layers, 2 heads, 108.9K params, seq_len=8).  
Run: `racket tests/train_gpt2_tiny.rkt`

| Step | Loss |
|---|---|
| 0 | 10.78 |
| 5 | 0.24 |
| 29 | 0.09 |

Throughput: ~47 steps/sec. The model memorizes the toy batch as expected.

### GGUF inference memory tuning (llama.cpp path)

Default `-c 2048 -b 512` (f16 KV) vs tuned `-c 512 -b 128` (still f16 KV).

| Model | Architecture | Baseline tunable | Tuned | Saved |
|---|---|---|---|---|
| Gemma-4 31B f16 | dense, wide MHA, all-attention | 2311 MiB | 254 MiB | ~2.0 GB |
| Qwen3.6-35B-A3B Q8 | MoE + hybrid Gated Delta Net | 545 MiB | 135 MiB | ~0.4 GB |
| gpt-oss-120b Q4_K_S | MoE + GQA | 490 MiB | 121 MiB | ~0.37 GB |

Proportional reduction is ~75% across architectures. Absolute savings scale with the model's baseline KV cache size — wider attention + f16 KV + all-attention layers means more to shrink. Newer architectures (GQA, hybrid layers, small head_dim) have already shrunk the baseline, so absolute wins are smaller there.

Adding `--cache-type-k q8_0 --cache-type-v q8_0` halves the remaining KV cache on top of this.

## Design Notes

### "No Python in the execution path"

The previous pyffi/Python-subprocess path was replaced wholesale:

- **Weight loading** — `safetensors.rkt` parses the binary format directly. Raw bytes go through `rg_tensor_from_f32_bytes` (and f16 / bf16 variants) into libtorch via `torch::from_blob(...).clone()`, landing on CUDA with no copies through Python tensors.
- **Weight copying into leaf params** — `rg_copy` wraps in `torch::NoGradGuard` and uses `.data().copy_()` to bypass autograd's leaf-requires-grad safety check when loading pretrained weights into parameters.
- **Tokenization** — `bpe.rkt` implements the full GPT-2 byte-level BPE: the reversible bytes-to-unicode mapping, the exact pre-tokenization regex (ASCII-approximated for `\p{L}` / `\p{N}` since Racket's regex doesn't support Unicode property escapes — fine for English prompts), merge-rank table from `merges.txt`, and greedy lowest-rank-first merging with a per-word cache.

Verified: `bpe-encode` for "The meaning of life is" returns `(464 3616 286 1204 318)`, matching the HuggingFace reference exactly. Round-trip through `bpe-decode` reproduces the original string including contractions, numbers, whitespace, and newlines.

### Compile-time inference pipelines

`define-inference-pipeline` is a phase-1 macro that does genuine graph work at expansion, not runtime setup. Concretely:

1. **Parse** the body into a DAG of `pnode` structs (each with `id`, `op`, `args`, `bind-id`). `let*` bindings are threaded through an environment so shared allocations (one `load-model` used twice) produce one emission, not two.
2. **Analyze**:
   - Last-use pass marks each node's latest consumer.
   - Capture-propagation pass extends a node's last-use to any op that holds it by reference (e.g. `create-context` captures its model arg; freeing the model before the context is used would dangle).
3. **Compute option values** — `#:max-prompt-tokens` + `#:max-gen-tokens` → `n_ctx` (floored at 32), `#:kv-precision 'q8` → `#:cache-type-k 'q8_0 #:cache-type-v 'q8_0` with precedence rules (explicit > split sugar > shared sugar > backend default).
4. **Emit** a flat `let*` where every kwarg is a literal value in the source, and `free-*` ops are inserted incrementally at each alloc's last-use point. Kwargs are emitted independently — absence of one doesn't suppress others.

At runtime Racket's compiler sees only the emitted `let*`; the pnodes, options hash, and graph exist only during expansion.

The layering is strict: the FFI takes ints, the Racket backend takes symbols, the CLI takes strings, the macro computes compile-time values. Enum-encoding details don't leak across layers.

### What's honest about the savings

The "proportional ~75% reduction in tunable inference buffers" number is real and reproducible. But:

- **Model weights themselves are unchanged.** A 61 GB Gemma-4 f16 doesn't get smaller. The savings are on inference state (KV cache + compute buffers).
- **Absolute savings depend on architecture.** Pure-attention f16-KV models (older Gemma, Llama-2) have the most to shrink. Modern GQA / hybrid / quantized-weight models start with smaller KV caches and save proportionally less in absolute terms.
- **The macro is not where the bytes come from.** A CLI user with `-c 512 -b 128 --cache-type-k q8_0` gets the same bytes back as someone who writes `#:kv-precision 'q8`. The macro's value is declarative intent + compile-time checking + graph-level liveness, not the runtime allocation sizes themselves.

## Scope / limits / "what doesn't work yet"

Being explicit so there are no surprises:

- **Training uses fp32 only.** No AMP / bf16 / fp16 mixed precision. That's a 2–3× speedup left on the table.
- **Attention is naive O(n²).** No Flash Attention yet; long contexts (>1024) will be slow and memory-hungry.
- **No data loaders for real corpora.** Tokenization is native, but there's no pipeline for WikiText / The Pile / etc. — you'd need to write the batching + shuffling yourself.
- **No weight tying for GPT-2 training.** `load-gpt2-weights` copies `wte.weight` into `lm_head.weight` but they're separate tensors during training; both get independent gradient updates. Correct would be a single shared tensor.
- **One model at a time.** The handle tables in both FFIs are shared global state; no multi-model-in-one-process isolation.
- **Pretraining GPT-2 from scratch on WikiText-103** with the current (unoptimized) path is realistic but slow — estimated 30–80 hours per epoch on GB10. Flash Attention + mixed precision would bring that into the single-digit-hours range. Fine-tuning pretrained GPT-2 is much cheaper (a few hours on a small slice).
- **Gemma-4 31B f16 loops on output** through the llama.cpp path (model-specific quirk, not our FFI — verified by running Qwen3.6-35B-A3B Q8 through the same CLI and getting coherent output).

## Roadmap

- [ ] Mixed precision (fp16 / bf16) on the libtorch training path
- [ ] Flash Attention / `scaled_dot_product_attention` via libtorch
- [ ] Shared-parameter support for weight-tied heads (proper GPT-2 training)
- [ ] WikiText / The Pile data loader with streaming + shuffling
- [ ] LoRA fine-tuning utilities
- [ ] Preflight memory estimation for GGUF inference
- [ ] Expanded model zoo (Llama / Mistral class architectures on the training path)
- [ ] Cross-device multi-model isolation (lift the global handle tables)

## License

MIT
