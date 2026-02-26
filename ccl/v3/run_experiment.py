#!/usr/bin/env python3
"""
CCL LLaMA 3 8B -- v3: Complete Rewrite
=======================================

Sequential continual learning with proper gradient projection.
Fixes all 6 structural bugs from v2.

Usage:
    python run_experiment.py --mode both        # Full comparison (default)
    python run_experiment.py --mode projected    # Projected only
    python run_experiment.py --mode unprojected  # Unprojected only
    python run_experiment.py --rank 32           # Override adapter rank
"""

import argparse
import gc
import os
import sys
import torch
import numpy as np

from ccl_model import CCLConfig, SKILL_NAMES
from ccl_trainer import CCLTrainer
from ccl_eval import verify_eta_squared_scaling
from ccl_plots import generate_plots, generate_comparison_plots


def run_single(mode: str, config_overrides: dict = None) -> dict:
    """Run a single experiment (projected or unprojected)."""
    use_proj = (mode == "projected")
    save_dir = f"ccl_results_llama3/v3/{mode}"

    config = CCLConfig(
        use_projection=use_proj,
        save_dir=save_dir,
    )

    # Apply overrides
    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)
                print(f"  Override: {k} = {v}")

    print(f"\n{'='*70}")
    print(f"  CCL LLaMA 3 8B v3: {mode.upper()}")
    print(f"  adapter_rank={config.adapter_rank}, skill_rank={config.skill_rank}")
    print(f"  overlap={config.overlap_fraction}, epochs/skill={config.num_epochs_per_skill}")
    print(f"  lr={config.lr}, batch_size={config.batch_size}")
    print(f"{'='*70}")

    # Train
    trainer = CCLTrainer(config)
    trainer.train()

    # Interference matrix
    interference = trainer.compute_interference_matrix()

    # Eta scaling
    eta_results = verify_eta_squared_scaling(
        trainer.model, config, trainer.grad_engine, config.device
    )

    # Save
    trainer.save_results(interference, eta_results)

    # Plots
    generate_plots(trainer.metrics, config, interference, eta_results,
                  save_dir, trainer.phase_losses)

    print(trainer.metrics.summary())

    max_off_diag = max(abs(interference[i, j])
                      for i in range(4) for j in range(4) if i != j)
    eps = trainer.grad_engine.measure_epsilon()
    print(f"\n  FINAL: max_off_diag={max_off_diag:.10f}, ε={eps:.10f}")
    print(f"  reorth_count={trainer.grad_engine.reorth_count}")

    return {
        "interference": interference,
        "eta_results": eta_results,
        "max_off_diag": max_off_diag,
        "epsilon": eps,
        "save_dir": save_dir,
    }


def run_comparison(config_overrides: dict = None):
    """Run both unprojected and projected, then compare."""
    print("=" * 70)
    print("  CCL LLaMA 3 8B v3: Sequential CL Comparison")
    print("=" * 70)

    # Unprojected
    print("\n\n  ========== UNPROJECTED ==========")
    unproj = run_single("unprojected", config_overrides)

    # Free GPU
    torch.cuda.empty_cache()
    gc.collect()

    # Projected
    print("\n\n  ========== PROJECTED ==========")
    proj = run_single("projected", config_overrides)

    # Compare
    comp_dir = "ccl_results_llama3/v3/comparison"
    summary = generate_comparison_plots(
        unproj["save_dir"], proj["save_dir"], comp_dir
    )

    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  Unprojected max interference: {unproj['max_off_diag']:.10f}")
    print(f"  Projected max interference:   {proj['max_off_diag']:.10f}")
    ratio = unproj['max_off_diag'] / proj['max_off_diag'] if proj['max_off_diag'] > 0 else float('inf')
    print(f"  Ratio: {ratio:.2f}x reduction")
    print(f"\n  Results in: ccl_results_llama3/v3/")


def main():
    parser = argparse.ArgumentParser(description="CCL LLaMA 3 8B v3")
    parser.add_argument("--mode", choices=["unprojected", "projected", "both"],
                       default="both")
    parser.add_argument("--rank", type=int, default=None, help="Adapter rank (default 64)")
    parser.add_argument("--skill-rank", type=int, default=None, help="Skill rank (default rank//2)")
    parser.add_argument("--overlap", type=float, default=None, help="Overlap fraction (default 0.3)")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per skill (default 5)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eta-steps", type=int, default=None, help="Steps per eta value")
    args = parser.parse_args()

    overrides = {}
    if args.rank is not None:
        overrides["adapter_rank"] = args.rank
        if args.skill_rank is None:
            overrides["skill_rank"] = args.rank // 2
    if args.skill_rank is not None:
        overrides["skill_rank"] = args.skill_rank
    if args.overlap is not None:
        overrides["overlap_fraction"] = args.overlap
    if args.epochs is not None:
        overrides["num_epochs_per_skill"] = args.epochs
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.eta_steps is not None:
        overrides["eta_steps"] = args.eta_steps

    if args.mode == "both":
        run_comparison(overrides or None)
    else:
        run_single(args.mode, overrides or None)


if __name__ == "__main__":
    main()
