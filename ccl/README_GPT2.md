# CCL on GPT-2 124M — RacoGrad

## Continual Controllable Learning at Scale

This module implements the CCL (Continual Controllable Learning) framework from RacoGrad on GPT-2 124M, demonstrating that the theoretical guarantees proven in the small-model demo hold at real language model scale.

## Key Results

### Training Performance (3 epochs, rank-16 adapters)

| Skill | Initial PPL | Final PPL (Frozen) | Final PPL (Live) |
|---|---|---|---|
| Code Generation | 37,764 | 1.12 | 1.08 |
| Question Answering | 166,854 | 1.03 | 1.03 |
| Summarization | 800 | 1.03 | 1.02 |
| Translation | 206,456 | 1.05 | 1.03 |

### Critical Finding: 14.8× Less Interference with Projected Gradients

| Metric | Frozen Base | Live Projected |
|---|---|---|
| Max cross-skill interference | 0.0269 | **0.0018** |
| Final ε (orthogonality deviation) | 0.0407 | **0.0400** |
| Re-orthogonalizations needed | N/A | **0** |

Live projected gradient training achieves **lower loss on all skills** while producing **14.8× less cross-skill interference** than the frozen baseline.

### O(η²) Interference Scaling

The theoretical bound predicts interference scales as O(η²). Verified experimentally:

| η | Interference | η² | Ratio |
|---|---|---|---|
| 0.01 | 3.05e-4 | 1.00e-4 | 3.05 |
| 0.005 | 3.61e-5 | 2.50e-5 | 1.44 |
| 0.001 | 1.11e-6 | 1.00e-6 | 1.11 |

At practical learning rates (η ≤ 0.001), the ratio stabilizes near 1, confirming O(η²) scaling.

## Architecture

```
GPT-2 124M (pretrained, 124M params)
  └── 12 transformer blocks, each with:
      ├── c_attn  + SkillAdapter(768→2304, rank=16, 4 skills)
      ├── c_proj  + SkillAdapter(768→768,  rank=16, 4 skills)
      ├── c_fc    + SkillAdapter(768→3072, rank=16, 4 skills)
      └── mlp_proj + SkillAdapter(3072→768, rank=16, 4 skills)

Adapter overhead: 9.4M params (7.58% of base)
```

Each skill j has adapters (U_j, V_j) per layer. During forward pass, hooks inject `x @ V_j @ U_j^T` into each adapted layer's output. Projected gradient descent ensures base model updates stay within the active skill's subspace.

## Skills

1. **Code Generation** — Python function templates
2. **Question Answering** — Factual Q&A pairs
3. **Summarization** — Passage → summary pairs
4. **Translation** — English → French pairs

## Usage

```bash
# Full comparison (frozen baseline + live projected)
python3 ccl/ccl_gpt2.py --mode both

# Just live projected gradients
python3 ccl/ccl_gpt2.py --mode live --rank 16 --epochs 3

# Just frozen baseline
python3 ccl/ccl_gpt2.py --mode frozen
```

## Output

Results are saved to `ccl_results/`:
- `frozen/` and `live/` — per-mode metrics, configs
- `comparison/` — side-by-side plots
- `*/figures/` — all plots (losses, perplexity, epsilon drift, interference matrix, η² scaling)
- `metrics.json` — full metric timeseries

## Plots Generated

1. **skill_losses.png** — Per-skill eval loss over training steps
2. **skill_perplexity.png** — Per-skill perplexity (log scale)
3. **epsilon_drift.png** — Subspace orthogonality deviation ε over time
4. **interference_matrix.png** — Cross-skill interference heatmap
5. **eta_squared_scaling.png** — O(η²) verification (log-log + ratio)
6. **frozen_vs_live_losses.png** — Side-by-side comparison
7. **interference_comparison.png** — Side-by-side interference matrices

## Hardware

- NVIDIA GeForce RTX 4060 Ti (16GB VRAM)
- Training time: ~2.5 min per mode (frozen + live total ~5 min)
- Peak VRAM: ~3.5GB (GPT-2 124M + rank-16 adapters)

## Connection to CCL Theory

The small-model demo (`demo.rkt`) proves CCL's theoretical properties on synthetic data. This GPT-2 experiment demonstrates:

1. **Low-rank skill subspaces** work on real transformer attention/MLP layers
2. **Projected gradient descent** eliminates cross-skill interference at scale
3. **ε stays bounded** — no re-orthogonalization needed at rank 16 over 3 epochs
4. **O(η²) scaling** confirmed on a real 124M-parameter language model
5. **Live > Frozen** — projected gradients enable better per-skill optimization while maintaining isolation
