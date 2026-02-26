#!/usr/bin/env python3
"""
CCL Trainer: Sequential continual learning protocol.

Train skill 0 -> measure all -> train skill 1 -> measure all -> ...
This is where catastrophic forgetting shows up (or doesn't, with CCL).
"""

import torch
import torch.nn as nn
import os
import time
import json
import copy
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from ccl_model import CCLConfig, CCLLlama, ProjectedGradientEngine, SKILL_NAMES
from ccl_data import generate_dataset, tokenize_dataset


class MetricsTracker:
    def __init__(self):
        self.data: Dict[str, List[Tuple[int, float]]] = {}
        self.start_time = time.time()

    def record(self, key: str, step: int, value: float):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append((step, value))

    def get(self, key: str):
        return self.data.get(key, [])

    def save(self, path: str):
        serializable = {k: [(s, float(v)) for s, v in vs] for k, vs in self.data.items()}
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

    def summary(self) -> str:
        lines = ["\n" + "=" * 60, "  Metrics Summary", "=" * 60]
        elapsed = time.time() - self.start_time
        lines.append(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
        for key in sorted(self.data.keys()):
            vals = [v for _, v in self.data[key]]
            if vals:
                lines.append(f"  {key}: last={vals[-1]:.10f}, min={min(vals):.10f}, max={max(vals):.10f}")
        lines.append("=" * 60)
        return "\n".join(lines)


class CCLTrainer:
    def __init__(self, config: CCLConfig):
        self.config = config
        self.device = config.device
        self.model = CCLLlama(config)
        self.grad_engine = ProjectedGradientEngine(self.model, config)
        self.metrics = MetricsTracker()

        # Optimizer over shared adapter params + projection bases
        trainable = list(self.model.adapters.parameters())
        self.optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=0.01)
        n_trainable = sum(p.numel() for p in trainable)
        print(f"  Optimizer: {n_trainable:,} trainable params")

        self._prepare_datasets()
        os.makedirs(config.save_dir, exist_ok=True)

    def _prepare_datasets(self):
        print("  Preparing datasets (separate train/eval templates)...")
        self.train_data = {}
        self.eval_data = {}
        for idx, skill in enumerate(SKILL_NAMES):
            train = generate_dataset(self.model.tokenizer, skill, "train", 200, self.config.seq_len)
            self.train_data[idx] = tokenize_dataset(train, self.model.tokenizer, self.config.seq_len).to(self.device)

            evl = generate_dataset(self.model.tokenizer, skill, "eval", 50, self.config.seq_len)
            self.eval_data[idx] = tokenize_dataset(evl, self.model.tokenizer, self.config.seq_len).to(self.device)

        for idx, skill in enumerate(SKILL_NAMES):
            print(f"    {skill}: train={self.train_data[idx].shape[0]}, eval={self.eval_data[idx].shape[0]}")

    @torch.no_grad()
    def eval_skill_loss(self, skill_idx: int) -> float:
        """Evaluate loss on HELD-OUT eval data for a skill."""
        self.model.base_model.eval()
        data = self.eval_data[skill_idx]
        total_loss, n = 0.0, 0
        bs = self.config.batch_size
        for start in range(0, data.shape[0], bs):
            end = min(start + bs, data.shape[0])
            ids = data[start:end]
            out = self.model(ids, labels=ids.clone())
            total_loss += out.loss.item() * (end - start)
            n += (end - start)
        self.model.base_model.train()
        return total_loss / max(n, 1)

    @torch.no_grad()
    def eval_all_skills(self) -> Dict[int, float]:
        """Evaluate all skills, return dict of losses."""
        losses = {}
        for si in range(self.config.num_skills):
            losses[si] = self.eval_skill_loss(si)
        return losses

    def train_one_skill(self, skill_idx: int, global_step: int) -> int:
        """Train one skill to convergence (multiple epochs). Returns updated global_step."""
        skill_name = SKILL_NAMES[skill_idx]
        data = self.train_data[skill_idx]
        num_samples = data.shape[0]
        bs = self.config.batch_size

        print(f"\n  === Training skill {skill_idx}: {skill_name} ===")
        print(f"  Epochs: {self.config.num_epochs_per_skill}, Samples: {num_samples}")

        for epoch in range(self.config.num_epochs_per_skill):
            # Shuffle data each epoch
            perm = torch.randperm(num_samples)
            data_shuffled = data[perm]

            epoch_loss = 0.0
            n_batches = (num_samples + bs - 1) // bs
            accum_count = 0

            pbar = tqdm(range(0, num_samples, bs),
                       desc=f"  Skill {skill_idx} ({skill_name}) Epoch {epoch+1}/{self.config.num_epochs_per_skill}",
                       leave=False)

            for start in pbar:
                end = min(start + bs, num_samples)
                ids = data_shuffled[start:end]

                self.model.base_model.train()
                out = self.model(ids, labels=ids.clone())
                loss = out.loss / self.config.grad_accum_steps
                loss.backward()

                # Project gradients IMMEDIATELY after backward (Bug #1 fix)
                if self.config.use_projection:
                    self.grad_engine.project_adapter_gradients(skill_idx)

                epoch_loss += out.loss.item()
                accum_count += 1

                if accum_count % self.config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.adapters.parameters()), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Maybe reorthogonalize
                    if self.config.use_projection and global_step % self.config.reorth_every == 0:
                        eps = self.grad_engine.maybe_reorthogonalize()
                        self.metrics.record("epsilon", global_step, eps)

                pbar.set_postfix(loss=f"{out.loss.item():.4f}")

            # Flush remaining gradients
            if accum_count % self.config.grad_accum_steps != 0:
                if self.config.use_projection:
                    self.grad_engine.project_adapter_gradients(skill_idx)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.adapters.parameters()), self.config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                global_step += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.metrics.record(f"train_loss/{skill_name}", global_step, avg_loss)
            print(f"    Epoch {epoch+1}: avg_loss={avg_loss:.6f}")

        return global_step

    def train(self):
        """Sequential continual learning: train each skill to convergence, measure all after each."""
        mode = "PROJECTED" if self.config.use_projection else "UNPROJECTED"
        print(f"\n{'='*60}")
        print(f"  CCL LLaMA 3 8B: {mode} (Sequential Protocol)")
        print(f"{'='*60}\n")

        global_step = 0

        # Initial evaluation (baseline before any training)
        print("  Initial evaluation (pre-training):")
        init_losses = self.eval_all_skills()
        for si, loss in init_losses.items():
            self.metrics.record(f"eval_loss/{SKILL_NAMES[si]}", 0, loss)
            print(f"    {SKILL_NAMES[si]}: {loss:.10f}")

        eps = self.grad_engine.measure_epsilon()
        self.metrics.record("epsilon", 0, eps)
        print(f"    ε = {eps:.10f}")

        # Sequential training: skill 0, then 1, then 2, then 3
        # After each skill, evaluate ALL skills
        self.phase_losses = {}  # phase_losses[phase][skill_idx] = loss

        for phase, skill_idx in enumerate(range(self.config.num_skills)):
            global_step = self.train_one_skill(skill_idx, global_step)

            # Evaluate ALL skills after this phase
            print(f"\n  Post-phase {phase} evaluation (after training {SKILL_NAMES[skill_idx]}):")
            losses = self.eval_all_skills()
            self.phase_losses[phase] = losses
            for si, loss in losses.items():
                self.metrics.record(f"eval_loss/{SKILL_NAMES[si]}", global_step, loss)
                marker = ""
                if si < skill_idx:
                    # Check forgetting
                    if phase - 1 in self.phase_losses:
                        prev = self.phase_losses[phase - 1][si]
                        delta = loss - prev
                        marker = f" (Δ={delta:+.10f} {'FORGETTING' if delta > 0 else 'retained'})"
                    elif phase == 0 and si == skill_idx:
                        pass
                print(f"    {SKILL_NAMES[si]}: {loss:.10f}{marker}")

            eps = self.grad_engine.measure_epsilon()
            self.metrics.record("epsilon", global_step, eps)
            print(f"    ε = {eps:.10f}")

        print(f"\n  Training complete. Total steps: {global_step}")

    def compute_interference_matrix(self) -> np.ndarray:
        """Compute interference: how training skill i affects loss on skill j.

        Uses the phase_losses recorded during sequential training.
        interference[i][j] = loss_j(after training i) - loss_j(before training i)
        """
        print("\n  Computing interference matrix from sequential training...")
        n = self.config.num_skills
        interference = np.zeros((n, n))

        for phase in range(n):
            for j in range(n):
                after = self.phase_losses[phase][j]
                if phase == 0:
                    # Compare to initial eval
                    before_series = self.metrics.get(f"eval_loss/{SKILL_NAMES[j]}")
                    before = before_series[0][1] if before_series else after
                else:
                    before = self.phase_losses[phase - 1][j]
                interference[phase][j] = after - before

        # Print
        print(f"\n  Interference Matrix (rows=trained skill, cols=eval skill):")
        print(f"  {'':>12s}", end="")
        for name in SKILL_NAMES:
            print(f"  {name:>14s}", end="")
        print()
        for i in range(n):
            print(f"  {SKILL_NAMES[i]:>12s}", end="")
            for j in range(n):
                val = interference[i][j]
                print(f"  {val:>14.10f}", end="")
            print()

        return interference

    def save_results(self, interference: np.ndarray, eta_results=None):
        save_dir = self.config.save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.metrics.save(os.path.join(save_dir, "metrics.json"))
        np.save(os.path.join(save_dir, "interference_matrix.npy"), interference)

        # Save phase losses
        phase_losses_serializable = {
            str(k): {str(si): float(v) for si, v in vs.items()}
            for k, vs in self.phase_losses.items()
        }
        with open(os.path.join(save_dir, "phase_losses.json"), 'w') as f:
            json.dump(phase_losses_serializable, f, indent=2)

        if eta_results:
            with open(os.path.join(save_dir, "eta_results.json"), 'w') as f:
                json.dump(eta_results, f, indent=2)

        # Config
        config_dict = {k: v for k, v in self.config.__dict__.items()
                      if not k.startswith('_')}
        for k, v in config_dict.items():
            if isinstance(v, torch.dtype):
                config_dict[k] = str(v)
        with open(os.path.join(save_dir, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        print(f"  Results saved to {save_dir}/")
