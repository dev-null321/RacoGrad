#!/usr/bin/env python3
"""
CCL (Continual Controllable Learning) for LLaMA 3 8B (4-bit quantized) - v2
=============================================================================

Key insight: With quantized frozen base, interference comes from SHARED adapter
parameters across skills. The CCL contribution is projecting adapter gradients 
into skill-specific subspaces to prevent cross-skill interference.

Architecture:
- 4-bit quantized LLaMA 3 8B (frozen)
- Shared low-rank adapters on Q, K, V, O projections  
- Each skill j has a learned projection matrix P_j
- "Unprojected": all skills update the same shared adapter freely (interference)
- "Projected": skill j's gradients are projected into P_j's column space

This mirrors the GPT-2 experiment where the shared base model created interference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time
import os
import gc
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

@dataclass
class CCLLlamaConfig:
    model_name: str = "NousResearch/Meta-Llama-3-8B"
    d_model: int = 4096
    num_layers: int = 32
    adapter_rank: int = 16
    adapter_targets: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    lr: float = 3e-5
    num_epochs: int = 3
    batch_size: int = 1
    seq_len: int = 128
    grad_accum_steps: int = 8
    max_grad_norm: float = 1.0
    
    num_skills: int = 4
    reorth_every: int = 50
    eps_threshold: float = 0.3
    use_projection: bool = True  # False = unprojected baseline
    
    log_every: int = 5
    eval_every: int = 25
    save_dir: str = "ccl_results_llama3"
    device: str = "cuda"


# ============================================================
# Shared Adapter with Skill Projection 
# ============================================================

class SharedSkillAdapter(nn.Module):
    """
    Shared low-rank adapter: W_adapter = A @ B^T (shared across all skills).
    Each skill j has a projection basis P_j (columns of U_j).
    
    During forward for skill j: output += x @ B @ A^T  (same for all skills)
    During backward for skill j (projected mode): gradients of A, B are 
    projected so only the component in skill j's subspace is updated.
    """
    
    def __init__(self, in_dim: int, out_dim: int, rank: int, num_skills: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.num_skills = num_skills
        
        # Shared adapter weights
        scale = 1.0 / math.sqrt(rank)
        self.A = nn.Parameter(torch.randn(out_dim, rank) * scale)  # out_dim x rank
        self.B = nn.Parameter(torch.zeros(in_dim, rank))            # in_dim x rank
        
        # Per-skill projection bases (columns define skill subspace in adapter output space)
        # U_j: out_dim x (rank // num_skills) -- each skill gets a portion of the output space
        skill_rank = max(rank // num_skills, 4)
        self.skill_rank = skill_rank
        self.U = nn.ParameterList([
            nn.Parameter(self._init_orthogonal(out_dim, skill_rank, j))
            for j in range(num_skills)
        ])
    
    def _init_orthogonal(self, dim, rank, seed):
        """Initialize with orthogonal columns, offset per skill."""
        torch.manual_seed(42 + seed)
        Q, _ = torch.linalg.qr(torch.randn(dim, rank))
        return Q[:, :rank] * 0.1
    
    def get_projection_matrix(self, skill_idx: int) -> torch.Tensor:
        """P_j = U_j (U_j^T U_j)^{-1} U_j^T -- projects into skill j's subspace."""
        U = self.U[skill_idx].float()
        gram = U.T @ U + 1e-6 * torch.eye(self.skill_rank, device=U.device)
        return U @ torch.linalg.inv(gram) @ U.T


# ============================================================  
# CCL-Wrapped LLaMA
# ============================================================

class CCLLlama(nn.Module):
    def __init__(self, config: CCLLlamaConfig):
        super().__init__()
        self.config = config
        self._load_base_model()
        self._add_adapters()
        self._freeze_base()
    
    def _load_base_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        print(f"Loading {self.config.model_name} (4-bit)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        n_params = sum(p.numel() for p in self.base_model.parameters())
        vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"  Base: {n_params:,} params, VRAM: {vram:.2f}GB")
    
    def _add_adapters(self):
        self.adapted_layers: Dict[str, dict] = {}
        self.adapters = nn.ModuleDict()
        adapter_params = 0
        
        for layer_idx in range(self.config.num_layers):
            block = self.base_model.model.layers[layer_idx]
            attn = block.self_attn
            
            for target_name in self.config.adapter_targets:
                module = getattr(attn, target_name)
                in_dim = module.in_features
                out_dim = module.out_features
                
                key = f"layer{layer_idx}_{target_name}"
                adapter = SharedSkillAdapter(
                    in_dim, out_dim, self.config.adapter_rank, self.config.num_skills
                ).to(self.config.device)
                
                self.adapters[key] = adapter
                self.adapted_layers[key] = {
                    "module": module,
                    "adapter": adapter,
                }
                adapter_params += sum(p.numel() for p in adapter.parameters())
        
        print(f"  Adapters: {adapter_params:,} params (shared, rank={self.config.adapter_rank})")
        vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"  VRAM after adapters: {vram:.2f}GB")
    
    def _freeze_base(self):
        for p in self.base_model.parameters():
            p.requires_grad = False
        print("  Base frozen")
    
    def forward(self, input_ids, labels=None, attention_mask=None):
        """Forward pass -- adapter hooks are always active (shared adapter)."""
        handles = self._register_hooks()
        try:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        finally:
            for h in handles:
                h.remove()
        return outputs
    
    def _register_hooks(self):
        handles = []
        for key, info in self.adapted_layers.items():
            module = info["module"]
            adapter = info["adapter"]
            A = adapter.A
            B = adapter.B
            
            def make_hook(a, b):
                def hook(mod, inp, out):
                    x = inp[0]
                    # adapter: x @ B @ A^T
                    adapter_out = x.to(b.dtype) @ b @ a.T
                    return out + adapter_out.to(out.dtype)
                return hook
            
            h = module.register_forward_hook(make_hook(A, B))
            handles.append(h)
        return handles


# ============================================================
# Projected Gradient Engine 
# ============================================================

class ProjectedGradientEngine:
    def __init__(self, model: CCLLlama, config: CCLLlamaConfig):
        self.model = model
        self.config = config
        self.reorth_count = 0
    
    def project_adapter_gradients(self, skill_idx: int):
        """Project shared adapter gradients into skill j's subspace."""
        for key, info in self.model.adapted_layers.items():
            adapter = info["adapter"]
            P = adapter.get_projection_matrix(skill_idx)  # out_dim x out_dim
            
            # Project A's gradient: A is out_dim x rank
            if adapter.A.grad is not None:
                # Each column of A.grad gets projected
                adapter.A.grad.data = (P @ adapter.A.grad.data.float()).to(adapter.A.grad.dtype)
            
            # B's gradient doesn't need projection (it's in the input space)
            # But we can project to maintain consistency
    
    @torch.no_grad()
    def reorthogonalize(self):
        """Re-orthogonalize skill subspaces."""
        for key, info in self.model.adapted_layers.items():
            adapter = info["adapter"]
            U_list = [adapter.U[j].data.float() for j in range(self.config.num_skills)]
            
            new_U = []
            for i, u in enumerate(U_list):
                u_new = u.clone()
                for j in range(i):
                    uj = new_U[j]
                    gram = uj.T @ uj + 1e-6 * torch.eye(adapter.skill_rank, device=u.device)
                    proj = uj @ torch.linalg.inv(gram) @ uj.T @ u_new
                    u_new = u_new - proj
                norms = torch.norm(u_new, dim=0, keepdim=True) + 1e-8
                u_new = u_new / norms * torch.norm(u, dim=0, keepdim=True)
                new_U.append(u_new)
            
            for j in range(self.config.num_skills):
                adapter.U[j].data.copy_(new_U[j].to(adapter.U[j].dtype))
        
        self.reorth_count += 1
    
    def measure_epsilon(self) -> float:
        max_eps = 0.0
        for key, info in self.model.adapted_layers.items():
            adapter = info["adapter"]
            for i in range(self.config.num_skills):
                for j in range(i + 1, self.config.num_skills):
                    Ui = adapter.U[i].float()
                    Uj = adapter.U[j].float()
                    cross = torch.norm(Ui.T @ Uj, p='fro').item()
                    norm_prod = (torch.norm(Ui, p='fro') * torch.norm(Uj, p='fro')).item()
                    eps = cross / (norm_prod + 1e-8)
                    max_eps = max(max_eps, eps)
        return max_eps
    
    def maybe_reorthogonalize(self):
        eps = self.measure_epsilon()
        if eps > self.config.eps_threshold:
            self.reorthogonalize()
            eps = self.measure_epsilon()
        return eps


# ============================================================
# Dataset Generation
# ============================================================

SKILL_NAMES = ["code", "qa", "summarize", "translate"]

def generate_synthetic_dataset(tokenizer, skill_name, num_samples=200, seq_len=128):
    data = []
    if skill_name == "code":
        tasks = [
            "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n",
            "def sort_list(lst):\n    return sorted(lst)\n",
            "def find_max(nums):\n    return max(nums)\n",
            "def reverse_str(s):\n    return s[::-1]\n",
            "def count_words(text):\n    return len(text.split())\n",
            "def is_palindrome(s):\n    return s == s[::-1]\n",
            "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a\n",
            "def merge_dicts(d1, d2):\n    return {**d1, **d2}\n",
            "def flatten(lst):\n    return [x for sub in lst for x in sub]\n",
            "def gcd(a, b):\n    while b: a, b = b, a%b\n    return a\n",
        ]
        for i in range(num_samples):
            text = f"# Python\n{tasks[i % len(tasks)]}"
            data.append({"text": text})
    elif skill_name == "qa":
        pairs = [
            ("What is the capital of France?", "Paris"),
            ("Who wrote Romeo and Juliet?", "Shakespeare"),
            ("What is photosynthesis?", "Plants converting sunlight to energy"),
            ("How many planets?", "8 planets"),
            ("Speed of light?", "299,792,458 m/s"),
            ("What is DNA?", "Genetic information molecule"),
            ("Who painted Mona Lisa?", "Da Vinci"),
            ("What is gravity?", "Force attracting mass"),
            ("When did WWII end?", "1945"),
            ("Largest ocean?", "Pacific"),
        ]
        for i in range(num_samples):
            q, a = pairs[i % len(pairs)]
            data.append({"text": f"Q: {q}\nA: {a}\n"})
    elif skill_name == "summarize":
        passages = [
            ("The Amazon covers 5.5M sq km with 10% of species.", "Huge biodiverse forest."),
            ("AI has made progress with LLMs for text and code.", "AI advances via LLMs."),
            ("The Great Wall stretches 21,196 km.", "Ancient defensive wall."),
            ("Climate change causes extreme weather.", "Climate drives extremes."),
            ("The brain has 86 billion neurons.", "Complex neural network."),
        ]
        for i in range(num_samples):
            p, s = passages[i % len(passages)]
            data.append({"text": f"Text: {p}\nSummary: {s}\n"})
    elif skill_name == "translate":
        pairs = [
            ("Hello, how are you?", "Bonjour, comment allez-vous?"),
            ("The cat is on the table.", "Le chat est sur la table."),
            ("I love programming.", "J'adore la programmation."),
            ("What time is it?", "Quelle heure est-il?"),
            ("Good morning.", "Bonjour."),
            ("Thank you.", "Merci."),
            ("The weather is nice.", "Il fait beau."),
            ("See you tomorrow.", "A demain."),
            ("I am happy.", "Je suis heureux."),
            ("Good night.", "Bonne nuit."),
        ]
        for i in range(num_samples):
            en, fr = pairs[i % len(pairs)]
            data.append({"text": f"English: {en}\nFrench: {fr}\n"})
    return data


def tokenize_dataset(data, tokenizer, seq_len):
    all_ids = []
    for item in data:
        ids = tokenizer.encode(item["text"], add_special_tokens=True)
        if len(ids) >= seq_len:
            ids = ids[:seq_len]
        else:
            ids = ids + [tokenizer.eos_token_id] * (seq_len - len(ids))
        all_ids.append(ids)
    return torch.tensor(all_ids, dtype=torch.long)


# ============================================================
# Metrics
# ============================================================

class MetricsTracker:
    def __init__(self):
        self.data: Dict[str, List[Tuple[int, float]]] = {}
        self.start_time = time.time()
    
    def record(self, key, step, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append((step, value))
    
    def get(self, key):
        return self.data.get(key, [])
    
    def save(self, path):
        serializable = {k: [(s, float(v)) for s, v in vs] for k, vs in self.data.items()}
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def summary(self):
        lines = ["\n" + "=" * 60, "  Metrics Summary", "=" * 60]
        elapsed = time.time() - self.start_time
        lines.append(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
        for key in sorted(self.data.keys()):
            vals = [v for _, v in self.data[key]]
            if vals:
                lines.append(f"  {key}: last={vals[-1]:.6f}, min={min(vals):.6f}, max={max(vals):.6f}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# CCL Trainer
# ============================================================

class CCLTrainer:
    def __init__(self, config: CCLLlamaConfig):
        self.config = config
        self.device = config.device
        self.model = CCLLlama(config)
        self.grad_engine = ProjectedGradientEngine(self.model, config)
        self.metrics = MetricsTracker()
        
        # Optimize shared adapter params + projection bases
        trainable = list(self.model.adapters.parameters())
        self.optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=0.01)
        n_trainable = sum(p.numel() for p in trainable)
        print(f"  Optimizer: {n_trainable:,} trainable params")
        
        self._prepare_datasets()
        os.makedirs(config.save_dir, exist_ok=True)
    
    def _prepare_datasets(self):
        print("  Preparing datasets...")
        self.train_data = {}
        self.eval_data = {}
        for idx, skill in enumerate(SKILL_NAMES):
            train = generate_synthetic_dataset(self.model.tokenizer, skill, 200, self.config.seq_len)
            self.train_data[idx] = tokenize_dataset(train, self.model.tokenizer, self.config.seq_len).to(self.device)
            evl = generate_synthetic_dataset(self.model.tokenizer, skill, 30, self.config.seq_len)
            self.eval_data[idx] = tokenize_dataset(evl, self.model.tokenizer, self.config.seq_len).to(self.device)
    
    @torch.no_grad()
    def eval_skill_loss(self, skill_idx):
        self.model.base_model.eval()
        data = self.eval_data[skill_idx]
        total_loss, n = 0, 0
        for start in range(0, min(data.shape[0], 16), 4):
            end = min(start + 4, data.shape[0])
            ids = data[start:end]
            out = self.model(ids, labels=ids.clone())
            total_loss += out.loss.item()
            n += 1
        self.model.base_model.train()
        return total_loss / max(n, 1)
    
    def train(self):
        mode = "PROJECTED" if self.config.use_projection else "UNPROJECTED"
        print(f"\n{'='*60}\n  CCL LLaMA 3 8B: {mode}\n{'='*60}\n")
        
        global_step = 0
        
        # Initial eval
        for si in range(self.config.num_skills):
            loss = self.eval_skill_loss(si)
            self.metrics.record(f"loss/{SKILL_NAMES[si]}", 0, loss)
            print(f"  Init {SKILL_NAMES[si]}: {loss:.4f}")
        
        eps = self.grad_engine.measure_epsilon()
        self.metrics.record("epsilon", 0, eps)
        print(f"  Init ε = {eps:.6f}\n")
        
        for epoch in range(self.config.num_epochs):
            num_batches = min(self.train_data[i].shape[0] for i in range(self.config.num_skills))
            
            for batch_idx in range(num_batches):
                for skill_idx in range(self.config.num_skills):
                    ids = self.train_data[skill_idx][batch_idx:batch_idx+1]
                    
                    self.model.base_model.train()
                    out = self.model(ids, labels=ids.clone())
                    loss = out.loss / self.config.grad_accum_steps
                    loss.backward()
                    
                    loss_val = loss.item() * self.config.grad_accum_steps
                    
                    if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                        if self.config.use_projection:
                            self.grad_engine.project_adapter_gradients(skill_idx)
                        
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.adapters.parameters()), self.config.max_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                        
                        self.metrics.record(f"train/{SKILL_NAMES[skill_idx]}", global_step, loss_val)
                        
                        if global_step % self.config.log_every == 0:
                            vram = torch.cuda.memory_allocated() / 1024**3
                            print(f"  [{epoch+1}/{self.config.num_epochs}] step {global_step}: "
                                  f"{SKILL_NAMES[skill_idx]}={loss_val:.4f} VRAM:{vram:.1f}GB")
                        
                        if global_step % self.config.eval_every == 0:
                            for si in range(self.config.num_skills):
                                el = self.eval_skill_loss(si)
                                self.metrics.record(f"loss/{SKILL_NAMES[si]}", global_step, el)
                            eps = self.grad_engine.maybe_reorthogonalize()
                            self.metrics.record("epsilon", global_step, eps)
                
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
            
            # Epoch summary
            print(f"\n  === Epoch {epoch+1} ===")
            for si in range(self.config.num_skills):
                loss = self.eval_skill_loss(si)
                self.metrics.record(f"loss/{SKILL_NAMES[si]}", global_step, loss)
                print(f"    {SKILL_NAMES[si]}: {loss:.4f} (ppl={math.exp(min(loss,20)):.2f})")
            eps = self.grad_engine.measure_epsilon()
            self.metrics.record("epsilon", global_step, eps)
            print(f"    ε = {eps:.6f}\n")
        
        return global_step
    
    def compute_interference_matrix(self):
        """Train on skill A, measure change in skill B's loss."""
        print("\n  Computing interference matrix...")
        n = self.config.num_skills
        matrix = np.zeros((n, n))
        
        state = {k: v.clone() for k, v in self.model.adapters.state_dict().items()}
        
        for train_skill in range(n):
            self.model.adapters.load_state_dict({k: v.clone() for k, v in state.items()})
            losses_before = [self.eval_skill_loss(si) for si in range(n)]
            
            opt = torch.optim.AdamW(self.model.adapters.parameters(), lr=self.config.lr)
            for step in range(10):
                idx = step % self.train_data[train_skill].shape[0]
                ids = self.train_data[train_skill][idx:idx+1]
                opt.zero_grad()
                self.model.base_model.train()
                out = self.model(ids, labels=ids.clone())
                out.loss.backward()
                
                if self.config.use_projection:
                    self.grad_engine.project_adapter_gradients(train_skill)
                
                opt.step()
            
            losses_after = [self.eval_skill_loss(si) for si in range(n)]
            for j in range(n):
                matrix[train_skill, j] = losses_after[j] - losses_before[j]
            
            print(f"    {SKILL_NAMES[train_skill]:>12}: " + 
                  " ".join(f"{matrix[train_skill,j]:+.5f}" for j in range(n)))
        
        self.model.adapters.load_state_dict(state)
        return matrix
    
    def verify_eta_squared_scaling(self, etas=None):
        if etas is None:
            etas = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        
        print("\n  O(η²) scaling verification...")
        results = []
        state = {k: v.clone() for k, v in self.model.adapters.state_dict().items()}
        
        for eta in etas:
            self.model.adapters.load_state_dict({k: v.clone() for k, v in state.items()})
            opt = torch.optim.SGD(self.model.adapters.parameters(), lr=eta)
            
            loss_before = self.eval_skill_loss(0)  # code
            
            # Train on skill 1 (qa)
            opt.zero_grad()
            ids = self.train_data[1][:1]
            self.model.base_model.train()
            out = self.model(ids, labels=ids.clone())
            out.loss.backward()
            
            if self.config.use_projection:
                self.grad_engine.project_adapter_gradients(1)
            
            opt.step()
            
            loss_after = self.eval_skill_loss(0)
            interference = abs(loss_after - loss_before)
            
            results.append({
                "eta": eta,
                "interference": interference,
                "eta_sq": eta**2,
                "ratio": interference / (eta**2) if eta > 0 else 0
            })
            print(f"    η={eta:.5f}  interf={interference:.8f}  η²={eta**2:.8f}  ratio={results[-1]['ratio']:.4f}")
        
        self.model.adapters.load_state_dict(state)
        return results
    
    def save_results(self, interference, eta_results):
        self.metrics.save(os.path.join(self.config.save_dir, "metrics.json"))
        np.save(os.path.join(self.config.save_dir, "interference_matrix.npy"), interference)
        with open(os.path.join(self.config.save_dir, "eta_scaling.json"), 'w') as f:
            json.dump(eta_results, f, indent=2)
        config_dict = {k: str(v) if isinstance(v, torch.dtype) else v 
                       for k, v in self.config.__dict__.items()}
        with open(os.path.join(self.config.save_dir, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)


# ============================================================
# Plotting
# ============================================================

def generate_plots(metrics, config, interference, eta_results, save_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig_dir = os.path.join(save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Skill losses
    fig, ax = plt.subplots(figsize=(10, 6))
    for skill in SKILL_NAMES:
        series = metrics.get(f"loss/{skill}")
        if series:
            steps, vals = zip(*series)
            ax.plot(steps, vals, label=skill, linewidth=2)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title(f"LLaMA 3 8B: Skill Losses ({'Projected' if config.use_projection else 'Unprojected'})")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "skill_losses.png"), dpi=150); plt.close(fig)
    
    # 2. Epsilon
    fig, ax = plt.subplots(figsize=(10, 5))
    series = metrics.get("epsilon")
    if series:
        steps, vals = zip(*series)
        ax.plot(steps, vals, 'b-o', linewidth=2, markersize=4)
        ax.axhline(y=config.eps_threshold, color='r', linestyle='--', label=f'Threshold')
    ax.set_xlabel("Step"); ax.set_ylabel("ε")
    ax.set_title("Subspace Orthogonality"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "epsilon.png"), dpi=150); plt.close(fig)
    
    # 3. Interference matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = max(np.abs(interference).max(), 1e-6)
    im = ax.imshow(interference, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(SKILL_NAMES); ax.set_yticklabels(SKILL_NAMES)
    ax.set_xlabel("Evaluated"); ax.set_ylabel("Trained")
    mode = "Projected" if config.use_projection else "Unprojected"
    ax.set_title(f"Interference ({mode})")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{interference[i,j]:.4f}", ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "interference.png"), dpi=150); plt.close(fig)
    
    # 4. Eta scaling
    if eta_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        etas = [r["eta"] for r in eta_results]
        interfs = [r["interference"] for r in eta_results]
        ax.loglog(etas, [max(x, 1e-12) for x in interfs], 'bo-', linewidth=2, markersize=8, label='Measured')
        ax.loglog(etas, [e**2 for e in etas], 'r--', linewidth=2, label='η² reference')
        ax.set_xlabel("η"); ax.set_ylabel("Interference")
        ax.set_title(f"η² Scaling ({mode})"); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "eta_scaling.png"), dpi=150); plt.close(fig)
    
    print(f"  Plots saved to {fig_dir}/")


# ============================================================
# Main
# ============================================================

def run_experiment(mode, save_dir=None):
    if save_dir is None:
        save_dir = f"ccl_results_llama3/{mode}"
    
    config = CCLLlamaConfig(
        use_projection=(mode == "projected"),
        save_dir=save_dir,
        num_epochs=3,
        batch_size=1,
        seq_len=128,
        lr=3e-5,
        grad_accum_steps=8,
    )
    
    trainer = CCLTrainer(config)
    trainer.train()
    
    interference = trainer.compute_interference_matrix()
    eta_results = trainer.verify_eta_squared_scaling()
    
    trainer.save_results(interference, eta_results)
    generate_plots(trainer.metrics, config, interference, eta_results, save_dir)
    
    print(trainer.metrics.summary())
    
    max_off_diag = max(abs(interference[i,j]) for i in range(4) for j in range(4) if i != j)
    eps = trainer.grad_engine.measure_epsilon()
    print(f"\n  FINAL: max_off_diag={max_off_diag:.6f}, ε={eps:.6f}, reorth={trainer.grad_engine.reorth_count}")
    
    return trainer, interference, eta_results


def run_comparison():
    print("=" * 70)
    print("  CCL LLaMA 3 8B: Unprojected vs Projected")
    print("=" * 70)
    
    # Unprojected baseline
    print("\n\n  === UNPROJECTED ===")
    _, unproj_interf, unproj_eta = run_experiment("unprojected", "ccl_results_llama3/unprojected")
    
    # Free GPU
    torch.cuda.empty_cache(); gc.collect()
    
    # Projected  
    print("\n\n  === PROJECTED ===")
    _, proj_interf, proj_eta = run_experiment("projected", "ccl_results_llama3/projected")
    
    # Compare
    unproj_max = max(abs(unproj_interf[i,j]) for i in range(4) for j in range(4) if i != j)
    proj_max = max(abs(proj_interf[i,j]) for i in range(4) for j in range(4) if i != j)
    ratio = unproj_max / proj_max if proj_max > 0 else float('inf')
    
    print(f"\n{'='*70}")
    print(f"  COMPARISON: UNPROJECTED vs PROJECTED (LLaMA 3 8B)")
    print(f"{'='*70}")
    print(f"  Max off-diagonal interference:")
    print(f"    Unprojected: {unproj_max:.6f}")
    print(f"    Projected:   {proj_max:.6f}")
    print(f"    Ratio:       {ratio:.1f}x less interference with projection")
    print(f"  (GPT-2 124M: 14.8x reduction)")
    
    # Comparison plot
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    comp_dir = "ccl_results_llama3/comparison"
    os.makedirs(comp_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    vmax = max(np.abs(unproj_interf).max(), np.abs(proj_interf).max(), 1e-6)
    for ax, matrix, title in [(ax1, unproj_interf, "Unprojected"), (ax2, proj_interf, "Projected")]:
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(SKILL_NAMES); ax.set_yticklabels(SKILL_NAMES)
        ax.set_title(title, fontsize=13)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{matrix[i,j]:.4f}", ha='center', va='center', fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"LLaMA 3 8B: {ratio:.1f}x Interference Reduction with Projection", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, "comparison.png"), dpi=150)
    plt.close(fig)
    
    # GPT-2 comparison if available
    gpt2_interf_path = "ccl_results/live/interference_matrix.npy"
    if os.path.exists(gpt2_interf_path):
        gpt2_interf = np.load(gpt2_interf_path)
        gpt2_max = max(abs(gpt2_interf[i,j]) for i in range(4) for j in range(4) if i != j)
        
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        all_interf = [unproj_interf, proj_interf, gpt2_interf]
        titles = [f"LLaMA 8B Unprojected\n(max={unproj_max:.4f})", 
                  f"LLaMA 8B Projected\n(max={proj_max:.4f})", 
                  f"GPT-2 124M Live\n(max={gpt2_max:.4f})"]
        vmax = max(np.abs(m).max() for m in all_interf)
        for ax, matrix, title in zip(axes, all_interf, titles):
            im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(SKILL_NAMES, fontsize=9); ax.set_yticklabels(SKILL_NAMES, fontsize=9)
            ax.set_title(title, fontsize=11)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{matrix[i,j]:.3f}", ha='center', va='center', fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle("Cross-Model Comparison", fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(comp_dir, "cross_model.png"), dpi=150)
        plt.close(fig)
    
    # Save summary
    summary = {
        "unprojected_max_interference": float(unproj_max),
        "projected_max_interference": float(proj_max),
        "interference_reduction_ratio": float(ratio),
        "gpt2_ratio": 14.8,
        "model": "LLaMA 3 8B (4-bit)",
    }
    with open(os.path.join(comp_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Results saved to {comp_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["unprojected", "projected", "both"], default="both")
    args = parser.parse_args()
    
    if args.mode == "both":
        run_comparison()
    else:
        run_experiment(args.mode)
