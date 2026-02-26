# CCL LLaMA 3 8B -- v3 (Complete Rewrite)

## What Changed from v2

v2 had 6 structural bugs that made results meaningless. All fixed:

| Bug | v2 Problem | v3 Fix |
|-----|-----------|--------|
| 1. Gradient accumulation | Mixed all skill gradients, projected with last skill_idx | Sequential training: each skill trained to convergence separately. Gradients projected immediately after each backward. |
| 2. Projection only on A | Only A.grad projected; B.grad unconstrained | Full DeltaW projection: A.grad projected through P_j, B.grad projected through A's pseudoinverse to keep full update in subspace. |
| 3. Joint multi-task training | All skills interleaved per batch (not continual learning) | Sequential protocol: skill 0 → convergence → measure all → skill 1 → convergence → measure all → ... |
| 4. Trivially orthogonal subspaces | skill_rank=4 in 4096-dim space (no overlap possible) | adapter_rank=64, skill_rank=32, random init with 30% deliberate overlap. |
| 5. 1-step eta test | Single step on memorized data | 80 steps per eta value on held-out eval data. |
| 6. Train = Eval data | Same templates recycled | Completely separate template sets for train and eval. |

## Architecture

```
ccl_model.py    -- SharedSkillAdapter, CCLLlama, ProjectedGradientEngine
ccl_data.py     -- Dataset generation with separate train/eval templates
ccl_trainer.py  -- Sequential continual learning training loop
ccl_eval.py     -- Eta-squared scaling test
ccl_plots.py    -- Visualization
run_experiment.py -- Main entry point
```

## Training Protocol

```
Phase 0: Train skill 0 (code) for 5 epochs
  → Evaluate ALL 4 skills (baseline)
Phase 1: Train skill 1 (QA) for 5 epochs
  → Evaluate ALL 4 skills (measure skill 0 forgetting)
Phase 2: Train skill 2 (summarize) for 5 epochs
  → Evaluate ALL 4 skills (measure skill 0,1 forgetting)
Phase 3: Train skill 3 (translate) for 5 epochs
  → Evaluate ALL 4 skills (full interference picture)
```

**Unprojected**: shared adapters, no projection → expect catastrophic forgetting
**Projected (CCL)**: gradient projection into skill subspaces → expect bounded forgetting

## Usage

```bash
# Full comparison (recommended)
python run_experiment.py --mode both

# Just projected
python run_experiment.py --mode projected

# With overrides (if rank=64 doesn't fit in VRAM)
python run_experiment.py --mode both --rank 32 --epochs 3

# Quick test
python run_experiment.py --mode both --rank 16 --epochs 1 --eta-steps 20
```

## Key Parameters

- `adapter_rank`: 64 (large enough for real expressiveness)
- `skill_rank`: 32 (50% of adapter rank)
- `overlap_fraction`: 0.3 (30% shared basis vectors between adjacent skills)
- `num_epochs_per_skill`: 5
- `lr`: 1e-4
- `eta_steps`: 80 (for scaling test)

## Hardware

- Oracle PC: i7, RTX 4060 Ti (16GB VRAM), 92GB RAM
- Model: NousResearch/Meta-Llama-3-8B, 4-bit quantized (NF4)
- If VRAM is tight, reduce `--rank` to 48 or 32

## Honest Results Policy

This experiment tests whether CCL actually reduces interference in a hard regime:
- Overlapping subspaces (not trivially orthogonal)
- Sequential training (real forgetting scenario)
- Held-out evaluation (not memorized data)

If CCL doesn't help, the results will show that honestly.
