#!/usr/bin/env python3
"""
CCL Evaluation: Eta-squared scaling test with proper methodology.
"""

import torch
import torch.nn as nn
import copy
import numpy as np
import json
import os
from typing import Dict, List
from tqdm import tqdm

from ccl_model import CCLConfig, CCLLlama, ProjectedGradientEngine, SKILL_NAMES
from ccl_data import generate_dataset, tokenize_dataset


def verify_eta_squared_scaling(model: CCLLlama, config: CCLConfig,
                                grad_engine: ProjectedGradientEngine,
                                device: str) -> List[dict]:
    """Test that interference scales as η² with projection.

    For each eta value:
    1. Save adapter state
    2. Train skill 0 for eta_steps steps at learning rate eta
    3. Measure loss change on skill 1 (interference)
    4. Restore adapter state

    Uses held-out eval data, not training data.
    """
    print(f"\n  === Eta-Squared Scaling Test ===")
    print(f"  Steps per eta: {config.eta_steps}")
    print(f"  Eta values: {config.eta_values}")

    # Prepare eval data
    eval_data = {}
    for idx, skill in enumerate(SKILL_NAMES):
        evl = generate_dataset(model.tokenizer, skill, "eval", 50, config.seq_len)
        eval_data[idx] = tokenize_dataset(evl, model.tokenizer, config.seq_len).to(device)

    # Prepare train data for skill 0
    train_0 = generate_dataset(model.tokenizer, SKILL_NAMES[0], "train", 200, config.seq_len)
    train_data_0 = tokenize_dataset(train_0, model.tokenizer, config.seq_len).to(device)

    # Save initial adapter state
    initial_state = {k: v.clone() for k, v in model.adapters.state_dict().items()}

    @torch.no_grad()
    def eval_loss(skill_idx):
        model.base_model.eval()
        data = eval_data[skill_idx]
        total, n = 0.0, 0
        bs = config.batch_size
        for start in range(0, data.shape[0], bs):
            end = min(start + bs, data.shape[0])
            ids = data[start:end]
            out = model(ids, labels=ids.clone())
            total += out.loss.item() * (end - start)
            n += (end - start)
        return total / max(n, 1)

    results = []

    for eta in tqdm(config.eta_values, desc="  Eta scaling"):
        # Restore to initial state
        model.adapters.load_state_dict(initial_state)

        # Measure baseline losses
        baseline_losses = {si: eval_loss(si) for si in range(config.num_skills)}

        # Create optimizer with this eta
        trainable = list(model.adapters.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=eta, weight_decay=0.01)

        # Train skill 0 for eta_steps steps
        model.base_model.train()
        bs = config.batch_size
        step = 0
        pbar = tqdm(total=config.eta_steps, desc=f"    η={eta:.4f}", leave=False)
        while step < config.eta_steps:
            perm = torch.randperm(train_data_0.shape[0])
            for start in range(0, train_data_0.shape[0], bs):
                if step >= config.eta_steps:
                    break
                end = min(start + bs, train_data_0.shape[0])
                ids = train_data_0[perm[start:end]]

                out = model(ids, labels=ids.clone())
                loss = out.loss
                loss.backward()

                if config.use_projection:
                    grad_engine.project_adapter_gradients(0)

                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                pbar.update(1)
        pbar.close()

        # Measure post-training losses
        post_losses = {si: eval_loss(si) for si in range(config.num_skills)}

        # Interference = loss change on OTHER skills
        interferences = {}
        for si in range(config.num_skills):
            interferences[si] = post_losses[si] - baseline_losses[si]

        # Max off-diagonal interference
        max_interf = max(abs(interferences[si]) for si in range(1, config.num_skills))

        result = {
            "eta": eta,
            "interference": float(max_interf),
            "skill_deltas": {SKILL_NAMES[si]: float(interferences[si]) for si in range(config.num_skills)},
            "baseline_losses": {SKILL_NAMES[si]: float(baseline_losses[si]) for si in range(config.num_skills)},
            "post_losses": {SKILL_NAMES[si]: float(post_losses[si]) for si in range(config.num_skills)},
        }
        results.append(result)
        print(f"    η={eta:.6f}: max_interf={max_interf:.10f}, skill0_Δ={interferences[0]:.10f}")

    # Restore original state
    model.adapters.load_state_dict(initial_state)

    # Check η² scaling
    if len(results) >= 2:
        print(f"\n  Eta-squared scaling check:")
        print(f"  {'eta':>10s}  {'interf':>14s}  {'η²':>14s}  {'ratio':>10s}")
        for r in results:
            eta = r["eta"]
            interf = r["interference"]
            eta_sq = eta ** 2
            ratio = interf / eta_sq if eta_sq > 0 else float('inf')
            print(f"  {eta:>10.6f}  {interf:>14.10f}  {eta_sq:>14.10f}  {ratio:>10.4f}")

    return results
