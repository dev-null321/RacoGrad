#!/usr/bin/env python3
"""CCL Visualization"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ccl_model import SKILL_NAMES


def generate_plots(metrics, config, interference, eta_results, save_dir, phase_losses=None):
    fig_dir = os.path.join(save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    mode = "Projected" if config.use_projection else "Unprojected"

    # 1. Eval losses over phases
    if phase_losses:
        fig, ax = plt.subplots(figsize=(10, 6))
        phases = sorted(phase_losses.keys())
        for si, skill in enumerate(SKILL_NAMES):
            vals = [phase_losses[p][si] for p in phases]
            ax.plot(range(len(phases) + 1),
                   [metrics.get(f"eval_loss/{skill}")[0][1]] + vals,
                   'o-', label=skill, linewidth=2, markersize=8)
        ax.set_xlabel("Phase (after training skill N)")
        ax.set_ylabel("Eval Loss")
        ax.set_xticks(range(len(phases) + 1))
        ax.set_xticklabels(["init"] + [SKILL_NAMES[p] for p in phases])
        ax.set_title(f"LLaMA 3 8B: Sequential CL Eval Losses ({mode})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "eval_losses_sequential.png"), dpi=150)
        plt.close(fig)

    # 2. Training losses
    fig, ax = plt.subplots(figsize=(10, 6))
    for skill in SKILL_NAMES:
        series = metrics.get(f"train_loss/{skill}")
        if series:
            steps, vals = zip(*series)
            ax.plot(steps, vals, 'o-', label=skill, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title(f"LLaMA 3 8B: Training Losses ({mode})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "train_losses.png"), dpi=150)
    plt.close(fig)

    # 3. Epsilon
    fig, ax = plt.subplots(figsize=(10, 5))
    series = metrics.get("epsilon")
    if series:
        steps, vals = zip(*series)
        ax.plot(steps, vals, 'b-o', linewidth=2, markersize=4)
        ax.axhline(y=config.eps_threshold, color='r', linestyle='--', label='Threshold')
    ax.set_xlabel("Step")
    ax.set_ylabel("ε")
    ax.set_title(f"Subspace Orthogonality ({mode})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "epsilon.png"), dpi=150)
    plt.close(fig)

    # 4. Interference matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    n = len(SKILL_NAMES)
    vmax = max(np.abs(interference).max(), 1e-6)
    im = ax.imshow(interference, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(SKILL_NAMES)
    ax.set_yticklabels(SKILL_NAMES)
    ax.set_xlabel("Evaluated Skill")
    ax.set_ylabel("Trained Skill (phase)")
    ax.set_title(f"Interference Matrix ({mode})")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{interference[i,j]:.6f}", ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "interference.png"), dpi=150)
    plt.close(fig)

    # 5. Eta scaling
    if eta_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        etas = [r["eta"] for r in eta_results]
        interfs = [r["interference"] for r in eta_results]
        ax.loglog(etas, [max(x, 1e-12) for x in interfs], 'bo-', linewidth=2, markersize=8, label='Measured')
        ax.loglog(etas, [e**2 for e in etas], 'r--', linewidth=2, label='η² reference')
        ax.set_xlabel("η (learning rate)")
        ax.set_ylabel("Max Off-Diagonal Interference")
        ax.set_title(f"η² Scaling Test ({mode})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "eta_scaling.png"), dpi=150)
        plt.close(fig)

    # 6. Forgetting trajectory (how much each skill degrades over phases)
    if phase_losses:
        fig, ax = plt.subplots(figsize=(10, 6))
        init_losses = {si: metrics.get(f"eval_loss/{SKILL_NAMES[si]}")[0][1]
                      for si in range(len(SKILL_NAMES))}
        phases = sorted(phase_losses.keys())
        for si, skill in enumerate(SKILL_NAMES):
            forgetting = []
            for p in phases:
                delta = phase_losses[p][si] - init_losses[si]
                forgetting.append(delta)
            ax.plot(range(len(phases)), forgetting, 'o-', label=skill, linewidth=2, markersize=8)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel("Phase (after training skill N)")
        ax.set_ylabel("Loss Change from Baseline")
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels([SKILL_NAMES[p] for p in phases])
        ax.set_title(f"Forgetting Trajectory ({mode})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "forgetting_trajectory.png"), dpi=150)
        plt.close(fig)

    print(f"  Plots saved to {fig_dir}/")


def generate_comparison_plots(unproj_dir, proj_dir, comp_dir):
    """Generate comparison plots between unprojected and projected runs."""
    os.makedirs(os.path.join(comp_dir, "figures"), exist_ok=True)

    # Load interference matrices
    unproj_interf = np.load(os.path.join(unproj_dir, "interference_matrix.npy"))
    proj_interf = np.load(os.path.join(proj_dir, "interference_matrix.npy"))

    n = len(SKILL_NAMES)

    # Side-by-side interference
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    vmax = max(np.abs(unproj_interf).max(), np.abs(proj_interf).max(), 1e-6)
    for ax, matrix, title in [(ax1, unproj_interf, "Unprojected"), (ax2, proj_interf, "Projected")]:
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(SKILL_NAMES)
        ax.set_yticklabels(SKILL_NAMES)
        ax.set_title(title, fontsize=13)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i,j]:.6f}", ha='center', va='center', fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)

    unproj_max = max(abs(unproj_interf[i, j]) for i in range(n) for j in range(n) if i != j)
    proj_max = max(abs(proj_interf[i, j]) for i in range(n) for j in range(n) if i != j)
    ratio = unproj_max / proj_max if proj_max > 0 else float('inf')
    fig.suptitle(f"LLaMA 3 8B: {ratio:.1f}x Interference Reduction", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, "figures", "comparison.png"), dpi=150)
    plt.close(fig)

    # Forgetting comparison
    for label, d in [("unprojected", unproj_dir), ("projected", proj_dir)]:
        phase_path = os.path.join(d, "phase_losses.json")
        if os.path.exists(phase_path):
            pass  # Could add more plots

    # Save summary
    summary = {
        "unprojected_max_interference": float(unproj_max),
        "projected_max_interference": float(proj_max),
        "interference_reduction_ratio": float(ratio),
        "model": "LLaMA 3 8B (4-bit)",
        "adapter_rank": 64,
        "skill_rank": 32,
        "protocol": "sequential_continual_learning",
    }
    with open(os.path.join(comp_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Comparison: unprojected max={unproj_max:.10f}, projected max={proj_max:.10f}")
    print(f"  Ratio: {ratio:.2f}x reduction")
    print(f"  Saved to {comp_dir}/")

    return summary
