# Continual Controllable Learning (CCL) for RacoGrad

Continual learning with provable bounded interference via projected gradient descent. Implemented in Racket with PyTorch FFI (CUDA backend).

## Two Modes

### Frozen Base (Baseline)
- Base model θ_base is frozen; only skill adapters W_j = U_j V_j^T are updated
- **Zero cross-task interference** by construction (adapters are independent parameters)
- Proves the math works in the trivial case

### Live Projected Gradients (Research Contribution)
- **All parameters are live and updatable** — no frozen weights
- Gradient projected into skill subspace V_j before optimizer step: `θ_{k+1} = θ_k - η · P_j · ∇L(θ_k)`
- Interference bounded by cross-subspace leakage: `L_i(θ_{k+1}) - L_i(θ_k) ≤ (M/2)η²‖P_j·∇L‖² + O(ε·η)`
- Periodic **Gram-Schmidt re-orthogonalization** prevents subspace drift
- ε (orthogonality deviation) tracked over training

## Stability Theorem

When updates are restricted to subspace V_j via projection P_j:

```
L_i(θ_{k+1}) - L_i(θ_k) ≤ (M/2) · η² · ‖P_j · ∇L_{k+1}‖² + O(ε · η)
```

- **M**: Hessian bound (smoothness constant)
- **ε**: max_{i≠j} ‖U_i^T U_j‖_F / (‖U_i‖_F · ‖U_j‖_F) — orthogonality deviation
- Forgetting scales as **O(η²)** when ε ≈ 0

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Symbolic Router: "classify complex" → Skill 1  │
└─────────────────┬───────────────────────────────┘
                  │ j=1
                  ▼
┌─────────────────────────────────────────────────┐
│  Projected Gradient Descent                      │
│  1. ∇L = backprop(loss)          (full gradient) │
│  2. ∇L_proj = P_j · ∇L    (project into V_j)    │
│  3. θ ← θ - η · ∇L_proj  (update all params)    │
│                                                   │
│  Every N steps: Gram-Schmidt re-orthogonalize    │
│  U_1...U_s to prevent subspace drift             │
└─────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `ccl-model.rkt` | CCL linear layers + model (frozen & live modes) |
| `projected-gradient.rkt` | Projected SGD/Adam, gradient projection, re-orthogonalization |
| `skill-subspace.rkt` | Projection operators, orthogonality metrics |
| `stability.rkt` | Metrics tracking and reporting |
| `symbolic-router.rkt` | Keyword-based task → skill routing |
| `demo.rkt` | Full demo: frozen vs live, interference analysis, O(η²) verification |

## Usage

```bash
cd ~/Projects/RacoGrad
racket ccl/demo.rkt
```

## Demo Results

### Frozen Base
- Zero cross-task interference (by construction)
- Each skill learns independently

### Live Projected
- All 3 skills learn simultaneously via interleaved training
- ε (subspace drift) stays bounded (~0.14-0.15)
- Live model achieves **lower loss on complex tasks** (shared base helps)
- Interference scales approximately with η²

### Forgetting vs η (one-step, live mode)
```
η        Interference   η²
0.10     0.012          0.010
0.05     0.002          0.003
0.01     0.001          0.000
0.005    0.001          0.000
0.001    0.000          0.000
```

## Key Design Decisions

1. **Projection via U_j basis**: Each skill defines a subspace via U_j ∈ R^{d×r}. The projection P_j = U_j(U_j^T U_j)^{-1}U_j^T is applied to gradients of base parameters before the optimizer step.

2. **Adapter + base composition**: Forward pass uses W_base + U_j V_j^T so adapters specialize per-skill while base captures shared representations.

3. **Gram-Schmidt re-orthogonalization**: When ε exceeds threshold, modified Gram-Schmidt is applied to {U_1,...,U_s} to restore approximate orthogonality.

4. **Symbolic router**: Racket pattern matching maps task descriptions to skill indices. Extensible via keyword registration.

## Dependencies

- RacoGrad (parent project)
- PyTorch with CUDA (via `pyffi`)
- Racket 8.x+
EOF