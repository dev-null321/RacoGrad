# RacoGrad: Deep Learning in Racket

<p align="center">
  <b>A GPU-accelerated deep learning framework written in Racket with PyTorch backend</b>
</p>

<p align="center">
  <i>Yes, you can run GPT-2 from a Lisp.</i>
</p>

---

## What is this?

RacoGrad is a deep learning framework that combines the elegance of Racket (a Lisp dialect) with the power of PyTorch's GPU acceleration. Write neural networks in a functional style, train them on CUDA, and run pretrained language models — all from Racket.

## Highlights

- **🚀 GPT-2 Text Generation** — Load pretrained HuggingFace weights, generate text
- **🔧 Full Transformer Architecture** — Multi-head attention, positional encoding, encoder-decoder
- **⚡ CUDA Acceleration** — PyTorch FFI backend for GPU-accelerated training and inference
- **📚 Training Utilities** — Gradient clipping, checkpointing, learning rate scheduling
- **🧠 Autodiff** — Automatic differentiation for backpropagation

## Quick Examples

### Generate Text with GPT-2

```racket
#lang racket
(require "gpt2.rkt")

;; Load pretrained GPT-2 (124M params)
(define model (make-gpt2-small))
(load-gpt2-weights model "gpt2")

;; Generate text
(gpt2-generate-text model "The meaning of life is" 
                    #:max-tokens 50 
                    #:temperature 0.8)

;; => "The meaning of life is very different.
;;     Life is an evolving process, and evolution is more than 
;;     simply a process of taking something of value out of 
;;     something you already own"
```

### Train a Transformer on Sequence Tasks

```racket
#lang racket
(require "transformer.rkt" "training.rkt")

;; Create a transformer
(define model (make-transformer
               #:d-model 64
               #:num-heads 4
               #:num-encoder-layers 2
               #:num-decoder-layers 2
               #:d-ff 256
               #:vocab-size 16
               #:max-seq-len 32
               #:sinusoidal #t))

;; Training data (copy task)
(define (make-copy-batch n seq-len)
  (define src (random-integers n seq-len 1 10))
  (values src src))

;; Train
(train-model model make-copy-batch
             #:epochs 10
             #:batch-size 32
             #:lr 0.001)

;; => Epoch 1: loss=2.3451, acc=0.12
;; => Epoch 2: loss=0.8234, acc=0.67
;; => ...
;; => Epoch 10: loss=0.0012, acc=1.00
```

### Multi-Head Attention

```racket
#lang racket
(require "attention.rkt")

;; Create attention module
(define attn (make-multi-head-attention 512 8))  ; d_model=512, 8 heads

;; Forward pass (self-attention)
(define output (forward attn query key value #:mask causal-mask))
```

### Tensor Operations on GPU

```racket
#lang racket
(require "pytorch_backend.rkt")

;; Create tensors on CUDA
(define a (pt:tensor '((1 2 3) (4 5 6)) #:device "cuda"))
(define b (pt:tensor '((7 8) (9 10) (11 12)) #:device "cuda"))

;; Matrix multiply on GPU
(define c (pt:matmul a b))
;; => tensor([[ 58,  64], [139, 154]], device='cuda:0')

;; Automatic differentiation
(define x (pt:tensor '((1 2) (3 4)) #:requires-grad #t))
(define y (pt:sum (pt:mul x x)))
(pt:backward y)
(pt:grad x)
;; => tensor([[2, 4], [6, 8]])
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Racket                        │
│  ┌───────────┐ ┌───────────┐ ┌───────────────┐  │
│  │  gpt2.rkt │ │transformer│ │ training.rkt  │  │
│  │           │ │   .rkt    │ │               │  │
│  └─────┬─────┘ └─────┬─────┘ └───────┬───────┘  │
│        │             │               │          │
│        └─────────────┼───────────────┘          │
│                      ▼                          │
│  ┌───────────────────────────────────────────┐  │
│  │          pytorch_backend.rkt              │  │
│  │        (FFI bridge to Python)             │  │
│  └─────────────────────┬─────────────────────┘  │
└────────────────────────┼────────────────────────┘
                         │ FFI
┌────────────────────────┼────────────────────────┐
│                        ▼                        │
│  ┌───────────────────────────────────────────┐  │
│  │              PyTorch / CUDA               │  │
│  │     (tensor ops, autograd, GPU kernels)   │  │
│  └───────────────────────────────────────────┘  │
│                     Python                      │
└─────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Racket 8.0+
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- HuggingFace Transformers (for GPT-2 weights)

### Setup

```bash
# Clone the repository
git clone https://github.com/dev-null321/RacoGrad.git
cd RacoGrad

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install Python dependencies
pip install torch transformers

# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Run GPT-2 demo
racket gpt2.rkt
```

## Module Overview

| Module | Description |
|--------|-------------|
| `gpt2.rkt` | GPT-2 implementation with HuggingFace weight loading |
| `transformer.rkt` | Full transformer encoder-decoder architecture |
| `attention.rkt` | Multi-head attention, causal masking |
| `nn.rkt` | Neural network primitives (Linear, LayerNorm, Embedding) |
| `training.rkt` | Training loops, grad clipping, checkpointing |
| `pytorch_backend.rkt` | FFI bridge to PyTorch |

## Why Racket?

- **Homoiconicity** — Code is data. Macro-generate architectures.
- **Functional style** — Composable, testable neural network modules
- **REPL-driven development** — Interactive experimentation
- **Different** — Because the world has enough Python ML frameworks

## Benchmarks

| Task | Model | Hardware | Result |
|------|-------|----------|--------|
| Copy (seq2seq) | Transformer 237K | RTX 4060 Ti | 100% acc, 2 epochs |
| Reverse (seq2seq) | Transformer 237K | RTX 4060 Ti | 100% acc, 2 epochs |
| Text Generation | GPT-2 124M | RTX 4060 Ti | Real-time inference |

## Roadmap

- [ ] Mixed precision training (fp16/bf16)
- [ ] Flash Attention
- [ ] LoRA fine-tuning
- [ ] Distributed training
- [ ] More pretrained models (LLaMA, Mistral)

## Contributing

Contributions welcome! This is a research project exploring the intersection of functional programming and deep learning.

## License

MIT License

---

<p align="center">
  <i>"The last question was asked for the first time, half in jest..."</i><br>
  — Isaac Asimov, "The Last Question"
</p>
