# RacoGrad: Deep Learning in Racket

<p align="center">
  <b>A GPU-accelerated deep learning framework written in Racket with a native C FFI → libtorch backend</b>
</p>

<p align="center">
  <i>Yes, you can run GPT-2 from a Lisp, without a Python runtime in the execution path.</i>
</p>

---

## What is this?

RacoGrad is a deep learning framework that combines Racket’s functional style with CUDA acceleration via **libtorch**. Models and training logic are written in Racket; tensor execution runs through a native C FFI bridge to libtorch.

## Highlights

- **GPT-2 Text Generation** — load pretrained HuggingFace weights and generate text
- **Full Transformer Stack** — encoder/decoder, multi-head attention, positional encoding
- **CUDA Acceleration** — native backend through `ffi/racograd_ffi.cpp` + libtorch
- **Training Utilities** — clipping, checkpointing, scheduler support
- **Autodiff** — backprop through the backend ops

## Backend Architecture

```
Racket modules (gpt2.rkt, transformer.rkt, training.rkt)
        │
        ▼
libtorch_backend.rkt
        │
        ▼
ffi/racograd_ffi.cpp  (C FFI boundary)
        │
        ▼
libtorch (C++ / CUDA)
```

## Installation

### Prerequisites

- Racket 8.0+
- C++ toolchain (clang/g++)
- libtorch with CUDA (tested with libtorch 2.5.1 + cu124)
- NVIDIA CUDA-capable GPU (for GPU acceleration)

### Setup

```bash
git clone https://github.com/dev-null321/RacoGrad.git
cd RacoGrad

# Point to your libtorch install
export LIBTORCH_DIR=~/libtorch-install/libtorch

# Build native FFI bridge (example)
g++ -O3 -std=c++17 -shared -fPIC \
  ffi/racograd_ffi.cpp \
  -I$LIBTORCH_DIR/include \
  -I$LIBTORCH_DIR/include/torch/csrc/api/include \
  -L$LIBTORCH_DIR/lib \
  -ltorch -ltorch_cpu -lc10 \
  -o ffi/libracograd_ffi.so

# Run tests/demos
racket regression-test.rkt
racket gpt2.rkt
```

> Note: `pytorch_backend.rkt` remains in-tree for historical compatibility, but the primary path is `libtorch_backend.rkt`.

## Module Overview

| Module | Description |
|---|---|
| `gpt2.rkt` | GPT-2 implementation + pretrained weight loading |
| `transformer.rkt` | Transformer encoder-decoder architecture |
| `attention.rkt` | Multi-head attention + masking |
| `nn.rkt` | Core NN primitives |
| `training.rkt` | Training loops and utilities |
| `libtorch_backend.rkt` | Primary backend interface to native libtorch FFI |
| `ffi/racograd_ffi.cpp` | C FFI bridge into libtorch/CUDA |

## Benchmarks (current)

| Task | Model | Hardware | Result |
|---|---|---|---|
| Copy (seq2seq) | Transformer 237K | RTX 4060 Ti | 100% acc, 2 epochs |
| Reverse (seq2seq) | Transformer 237K | RTX 4060 Ti | 100% acc, 2 epochs |
| Text Generation | GPT-2 124M | RTX 4060 Ti | Real-time inference |

## Roadmap

- [ ] Mixed precision (fp16/bf16)
- [ ] Flash Attention-style kernels
- [ ] LoRA fine-tuning utilities
- [ ] Expanded model zoo (LLaMA/Mistral class architectures)

## License

MIT
