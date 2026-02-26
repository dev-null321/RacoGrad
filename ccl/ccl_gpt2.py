#!/usr/bin/env python3
"""
CCL (Continual Controllable Learning) for GPT-2 124M
=====================================================

Implements the CCL framework from RacoGrad at GPT-2 scale:
- Low-rank skill adapters (U_j V_j^T) on attention & MLP layers
- Projected gradient descent into skill subspaces
- Gram-Schmidt re-orthogonalization
- Stability metrics and interference tracking

Skills: code generation, question answering, summarization, translation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

@dataclass
class CCLConfig:
    """CCL training configuration."""
    # Model
    model_name: str = "gpt2"
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    vocab_size: int = 50257
    max_len: int = 1024
    
    # Adapters
    adapter_rank: int = 16
    adapter_layers: str = "all"  # "all", "attn", "mlp", or list of layer indices
    adapter_targets: List[str] = field(default_factory=lambda: ["c_attn", "c_proj", "c_fc", "mlp_proj"])
    
    # Training
    lr: float = 1e-4
    num_epochs: int = 5
    batch_size: int = 4
    seq_len: int = 128
    grad_accum_steps: int = 4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # CCL specific
    num_skills: int = 4
    reorth_every: int = 50       # Re-orthogonalize every N steps
    eps_threshold: float = 0.3   # Trigger re-orth if epsilon exceeds this
    frozen_base: bool = False    # If True, only train adapters (baseline)
    
    # Logging
    log_every: int = 10
    eval_every: int = 50
    save_dir: str = "ccl_results"
    
    # Hardware
    device: str = "cuda"
    dtype: torch.dtype = torch.float32  # full precision for adapters


# ============================================================
# Low-Rank Skill Adapter
# ============================================================

class SkillAdapter(nn.Module):
    """
    Low-rank adapter: W_adapted = W_base + U_j @ V_j^T
    
    Each skill j has its own (U_j, V_j) pair.
    U_j ∈ R^{out_dim x rank}, V_j ∈ R^{in_dim x rank}
    """
    
    def __init__(self, in_dim: int, out_dim: int, rank: int, num_skills: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.num_skills = num_skills
        
        # Adapter matrices for each skill
        # U: out_dim x rank (Kaiming-like init)
        # V: in_dim x rank (zero init, so adapter starts as identity)
        scale = 1.0 / math.sqrt(rank)
        self.U = nn.ParameterList([
            nn.Parameter(torch.randn(out_dim, rank) * scale)
            for _ in range(num_skills)
        ])
        self.V = nn.ParameterList([
            nn.Parameter(torch.zeros(in_dim, rank))
            for _ in range(num_skills)
        ])
    
    def get_delta(self, skill_idx: int) -> torch.Tensor:
        """Get the adapter delta: U_j @ V_j^T"""
        return self.U[skill_idx] @ self.V[skill_idx].T
    
    def get_projection_matrix(self, skill_idx: int) -> torch.Tensor:
        """
        Projection matrix P_j = U_j (U_j^T U_j)^{-1} U_j^T
        Projects gradients into skill j's subspace.
        """
        U = self.U[skill_idx]
        gram = U.T @ U + 1e-6 * torch.eye(self.rank, device=U.device)
        return U @ torch.linalg.inv(gram) @ U.T


class AdaptedLinear(nn.Module):
    """
    Linear layer with CCL skill adapters.
    forward(x, skill_idx) = x @ (W + U_j V_j^T)^T + b
    """
    
    def __init__(self, original_layer, rank: int, num_skills: int):
        super().__init__()
        self.linear = original_layer
        # HuggingFace GPT-2 uses Conv1D (weight is [out, in] but stored as [in, out])
        from transformers.pytorch_utils import Conv1D
        if isinstance(original_layer, Conv1D):
            # Conv1D stores weight as (in_features, out_features) -- transposed
            in_dim = original_layer.weight.shape[0]
            out_dim = original_layer.weight.shape[1]
            self.is_conv1d = True
        else:
            in_dim = original_layer.in_features
            out_dim = original_layer.out_features
            self.is_conv1d = False
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.adapter = SkillAdapter(in_dim, out_dim, rank, num_skills)
    
    def forward(self, x: torch.Tensor, skill_idx: int) -> torch.Tensor:
        base_out = self.linear(x)
        # For Conv1D: weight is (in, out), forward is x @ weight + bias
        # Adapter: delta = U_j @ V_j^T has shape (in_dim, out_dim) for Conv1D
        # So adapter_out = x @ (V_j @ U_j^T) -- matching Conv1D convention
        if self.is_conv1d:
            adapter_out = x @ self.adapter.V[skill_idx] @ self.adapter.U[skill_idx].T
        else:
            adapter_out = x @ self.adapter.V[skill_idx] @ self.adapter.U[skill_idx].T
        return base_out + adapter_out


# ============================================================
# CCL-Wrapped GPT-2
# ============================================================

class CCLGPT2(nn.Module):
    """
    GPT-2 with CCL skill adapters on attention and MLP projections.
    
    Architecture:
    - Base GPT-2 weights (frozen or live)
    - Per-skill low-rank adapters on configurable layers
    - Projected gradient descent for live mode
    """
    
    def __init__(self, config: CCLConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained GPT-2
        self._load_base_model()
        
        # Add adapters
        self._add_adapters()
        
        # Freeze base if configured
        if config.frozen_base:
            self._freeze_base()
    
    def _load_base_model(self):
        """Load pretrained GPT-2 from HuggingFace."""
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        print(f"Loading {self.config.model_name} from HuggingFace...")
        self.base_model = GPT2LMHeadModel.from_pretrained(self.config.model_name)
        self.base_model.to(self.config.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        n_params = sum(p.numel() for p in self.base_model.parameters())
        print(f"  Base model: {n_params:,} parameters")
    
    def _add_adapters(self):
        """Add low-rank adapters to attention and MLP layers."""
        self.adapted_layers: Dict[str, AdaptedLinear] = {}
        adapter_params = 0
        
        for layer_idx in range(self.config.num_layers):
            block = self.base_model.transformer.h[layer_idx]
            
            # Attention projections
            if "c_attn" in self.config.adapter_targets:
                name = f"layer{layer_idx}.attn.c_attn"
                adapted = AdaptedLinear(
                    block.attn.c_attn, self.config.adapter_rank, self.config.num_skills
                ).to(self.config.device)
                self.adapted_layers[name] = adapted
                adapter_params += sum(p.numel() for p in adapted.adapter.parameters())
            
            if "c_proj" in self.config.adapter_targets:
                name = f"layer{layer_idx}.attn.c_proj"
                adapted = AdaptedLinear(
                    block.attn.c_proj, self.config.adapter_rank, self.config.num_skills
                ).to(self.config.device)
                self.adapted_layers[name] = adapted
                adapter_params += sum(p.numel() for p in adapted.adapter.parameters())
            
            # MLP projections
            if "c_fc" in self.config.adapter_targets:
                name = f"layer{layer_idx}.mlp.c_fc"
                adapted = AdaptedLinear(
                    block.mlp.c_fc, self.config.adapter_rank, self.config.num_skills
                ).to(self.config.device)
                self.adapted_layers[name] = adapted
                adapter_params += sum(p.numel() for p in adapted.adapter.parameters())
            
            if "mlp_proj" in self.config.adapter_targets:
                name = f"layer{layer_idx}.mlp.c_proj"
                adapted = AdaptedLinear(
                    block.mlp.c_proj, self.config.adapter_rank, self.config.num_skills
                ).to(self.config.device)
                self.adapted_layers[name] = adapted
                adapter_params += sum(p.numel() for p in adapted.adapter.parameters())
        
        # Register as module for proper parameter tracking
        self.adapter_module = nn.ModuleDict(
            {k.replace(".", "_"): v.adapter for k, v in self.adapted_layers.items()}
        )
        
        print(f"  Adapters: {adapter_params:,} parameters "
              f"(rank={self.config.adapter_rank}, {len(self.adapted_layers)} layers, "
              f"{self.config.num_skills} skills)")
        print(f"  Adapter overhead: {adapter_params / sum(p.numel() for p in self.base_model.parameters()) * 100:.2f}%")
    
    def _freeze_base(self):
        """Freeze base model parameters."""
        for p in self.base_model.parameters():
            p.requires_grad = False
        print("  Base model frozen (adapter-only training)")
    
    def forward(self, input_ids: torch.Tensor, skill_idx: int,
                labels: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass with skill-specific adapters.
        Uses forward hooks to inject adapter outputs (preserves autograd).
        """
        handles = self._register_adapter_hooks(skill_idx)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        for h in handles:
            h.remove()
        return outputs
    
    def _register_adapter_hooks(self, skill_idx: int) -> list:
        """Register forward hooks that add adapter outputs."""
        handles = []
        for name, adapted in self.adapted_layers.items():
            module = self._get_module(name)
            U_j = adapted.adapter.U[skill_idx]
            V_j = adapted.adapter.V[skill_idx]
            
            def make_hook(u, v):
                def hook(mod, inp, out):
                    x = inp[0]
                    return out + x @ v @ u.T
                return hook
            
            h = module.register_forward_hook(make_hook(U_j, V_j))
            handles.append(h)
        return handles
    
    def _get_module(self, name: str):
        """Get nn.Module by adapter name."""
        parts = name.split(".")
        layer_idx = int(parts[0].replace("layer", ""))
        block = self.base_model.transformer.h[layer_idx]
        if "attn.c_attn" in name:
            return block.attn.c_attn
        elif "attn.c_proj" in name:
            return block.attn.c_proj
        elif "mlp.c_fc" in name:
            return block.mlp.c_fc
        elif "mlp.c_proj" in name:
            return block.mlp.c_proj
        else:
            raise ValueError(f"Unknown layer: {name}")
    
    def _get_weight(self, name: str) -> nn.Parameter:
        """Get weight parameter by adapter name."""
        return self._get_module(name).weight
    
    def get_adapter_params(self, skill_idx: Optional[int] = None) -> List[nn.Parameter]:
        """Get adapter parameters for a specific skill or all skills."""
        params = []
        for adapted in self.adapted_layers.values():
            if skill_idx is not None:
                params.extend([adapted.adapter.U[skill_idx], adapted.adapter.V[skill_idx]])
            else:
                params.extend(adapted.adapter.parameters())
        return params
    
    def get_base_params(self) -> List[nn.Parameter]:
        """Get base model parameters."""
        return [p for p in self.base_model.parameters() if p.requires_grad]
    
    def get_all_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        params = list(self.adapter_module.parameters())
        if not self.config.frozen_base:
            params.extend(self.get_base_params())
        return params


# ============================================================
# Projected Gradient Engine
# ============================================================

class ProjectedGradientEngine:
    """
    Implements projected gradient descent for CCL.
    
    For skill j:
    1. Compute full gradient ∇L
    2. Project base param gradients: ∇L_proj = P_j · ∇L
    3. Update: θ ← θ - η · ∇L_proj
    
    Periodically re-orthogonalize U matrices via modified Gram-Schmidt.
    """
    
    def __init__(self, model: CCLGPT2, config: CCLConfig):
        self.model = model
        self.config = config
        self.step_count = 0
        self.reorth_count = 0
    
    def project_base_gradients(self, skill_idx: int):
        """
        Project base model gradients into skill j's subspace.
        
        For each adapted layer, compute P_j = U_j(U_j^T U_j)^{-1}U_j^T
        and apply it to the corresponding base weight's gradient.
        """
        if self.config.frozen_base:
            return  # No base gradients to project
        
        for name, adapted in self.model.adapted_layers.items():
            weight = self.model._get_weight(name)
            if weight.grad is None:
                continue
            
            P = adapted.adapter.get_projection_matrix(skill_idx)
            # P is (out_dim, out_dim) where out_dim matches adapter U dimension
            
            if adapted.is_conv1d:
                # Conv1D weight grad is (in_dim, out_dim)
                # Project along out_dim (columns): grad @ P^T
                weight.grad.data = weight.grad.data @ P.T
            else:
                # Linear weight grad is (out_dim, in_dim)  
                # Project along out_dim (rows): P @ grad
                weight.grad.data = P @ weight.grad.data
    
    @torch.no_grad()
    def reorthogonalize(self):
        """
        Modified Gram-Schmidt on U matrices across skills.
        Ensures skill subspaces remain approximately orthogonal.
        """
        for name, adapted in self.model.adapted_layers.items():
            adapter = adapted.adapter
            U_list = [adapter.U[j].data for j in range(self.config.num_skills)]
            
            new_U = []
            for i, u in enumerate(U_list):
                u_new = u.clone()
                for j in range(i):
                    uj = new_U[j]
                    gram = uj.T @ uj + 1e-6 * torch.eye(adapter.rank, device=u.device)
                    proj = uj @ torch.linalg.inv(gram) @ uj.T @ u_new
                    u_new = u_new - proj
                # Preserve scale
                norms = torch.norm(u_new, dim=0, keepdim=True) + 1e-8
                u_new = u_new / norms * torch.norm(u, dim=0, keepdim=True)
                new_U.append(u_new)
            
            # Copy back
            for j in range(self.config.num_skills):
                adapter.U[j].data.copy_(new_U[j])
        
        self.reorth_count += 1
    
    def measure_epsilon(self) -> float:
        """
        Measure max pairwise orthogonality deviation across all adapted layers.
        ε = max_{i≠j} ‖U_i^T U_j‖_F / (‖U_i‖_F · ‖U_j‖_F)
        """
        max_eps = 0.0
        for name, adapted in self.model.adapted_layers.items():
            adapter = adapted.adapter
            for i in range(self.config.num_skills):
                for j in range(i + 1, self.config.num_skills):
                    cross = torch.norm(adapter.U[i].T @ adapter.U[j], p='fro').item()
                    norm_prod = (torch.norm(adapter.U[i], p='fro') * 
                                torch.norm(adapter.U[j], p='fro')).item()
                    eps = cross / (norm_prod + 1e-8)
                    max_eps = max(max_eps, eps)
        return max_eps
    
    def maybe_reorthogonalize(self) -> Optional[float]:
        """Check epsilon and re-orthogonalize if needed. Returns epsilon."""
        eps = self.measure_epsilon()
        if eps > self.config.eps_threshold:
            eps_before = eps
            self.reorthogonalize()
            eps_after = self.measure_epsilon()
            return eps_after
        return eps


# ============================================================
# Metrics Tracker
# ============================================================

class MetricsTracker:
    """Track training metrics for CCL analysis."""
    
    def __init__(self, config: CCLConfig):
        self.config = config
        self.data: Dict[str, List[Tuple[int, float]]] = {}
        self.start_time = time.time()
    
    def record(self, key: str, step: int, value: float):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append((step, value))
    
    def get(self, key: str) -> List[Tuple[int, float]]:
        return self.data.get(key, [])
    
    def get_last(self, key: str) -> Optional[float]:
        series = self.get(key)
        return series[-1][1] if series else None
    
    def save(self, path: str):
        """Save metrics to JSON."""
        serializable = {}
        for k, v in self.data.items():
            serializable[k] = [(s, float(val)) for s, val in v]
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def summary(self) -> str:
        """Print summary of tracked metrics."""
        lines = ["\n" + "=" * 60, "  CCL Metrics Summary", "=" * 60]
        elapsed = time.time() - self.start_time
        lines.append(f"  Training time: {elapsed:.1f}s")
        
        for key in sorted(self.data.keys()):
            series = self.data[key]
            if series:
                vals = [v for _, v in series]
                lines.append(f"  {key}: last={vals[-1]:.6f}, min={min(vals):.6f}, "
                           f"max={max(vals):.6f}, n={len(vals)}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# Dataset Generation
# ============================================================

SKILL_NAMES = ["code", "qa", "summarize", "translate"]
SKILL_DESCRIPTIONS = {
    "code": "Code generation",
    "qa": "Question answering",
    "summarize": "Summarization",
    "translate": "Translation (English→French-style)"
}

def generate_synthetic_dataset(tokenizer, skill_name: str, num_samples: int = 500, 
                                seq_len: int = 128) -> List[Dict]:
    """
    Generate synthetic training data for each skill.
    Uses template-based generation for reproducible experiments.
    """
    data = []
    
    if skill_name == "code":
        templates = [
            "# Python function to {task}\ndef {func}({args}):\n    {body}",
            "# Calculate {task}\nresult = {expr}\nprint(result)",
            "# {task}\nfor i in range({n}):\n    {body}",
            "# Class for {task}\nclass {cls}:\n    def __init__(self):\n        {init}",
            "# {task}\nimport {module}\n{code}",
        ]
        tasks = [
            ("compute factorial", "factorial", "n", "return 1 if n <= 1 else n * factorial(n-1)"),
            ("sort a list", "sort_list", "lst", "return sorted(lst)"),
            ("find maximum", "find_max", "nums", "return max(nums)"),
            ("reverse string", "reverse_str", "s", "return s[::-1]"),
            ("count words", "count_words", "text", "return len(text.split())"),
            ("check palindrome", "is_palindrome", "s", "return s == s[::-1]"),
            ("calculate fibonacci", "fibonacci", "n", "a, b = 0, 1\\n    for _ in range(n):\\n        a, b = b, a+b\\n    return a"),
            ("merge dictionaries", "merge_dicts", "d1, d2", "return {**d1, **d2}"),
            ("flatten list", "flatten", "lst", "return [x for sub in lst for x in sub]"),
            ("binary search", "binary_search", "arr, target", "lo, hi = 0, len(arr)-1\\n    while lo <= hi:\\n        mid = (lo+hi)//2\\n        if arr[mid] == target: return mid\\n        elif arr[mid] < target: lo = mid+1\\n        else: hi = mid-1\\n    return -1"),
        ]
        for i in range(num_samples):
            task, func, args, body = tasks[i % len(tasks)]
            text = f"# Python function to {task}\ndef {func}({args}):\n    {body}\n"
            # Add variations
            if i % 3 == 0:
                text = f"\"\"\"Write a function to {task}\"\"\"\n{text}"
            elif i % 3 == 1:
                text = f"# Task: {task}\n# Solution:\n{text}"
            data.append({"text": text, "skill": "code"})
    
    elif skill_name == "qa":
        qa_pairs = [
            ("What is the capital of France?", "The capital of France is Paris."),
            ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
            ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight into energy."),
            ("How many planets are in our solar system?", "There are 8 planets in our solar system."),
            ("What is the speed of light?", "The speed of light is approximately 299,792,458 meters per second."),
            ("What is DNA?", "DNA (deoxyribonucleic acid) is the molecule that carries genetic information."),
            ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
            ("What is gravity?", "Gravity is a fundamental force that attracts objects with mass toward each other."),
            ("What year did World War II end?", "World War II ended in 1945."),
            ("What is the largest ocean?", "The Pacific Ocean is the largest ocean on Earth."),
            ("What is machine learning?", "Machine learning is a subset of AI where systems learn from data."),
            ("Who discovered penicillin?", "Alexander Fleming discovered penicillin in 1928."),
            ("What is the Pythagorean theorem?", "The Pythagorean theorem states that a² + b² = c² for right triangles."),
            ("What causes rainbows?", "Rainbows are caused by refraction and reflection of sunlight in water droplets."),
            ("What is the boiling point of water?", "Water boils at 100°C (212°F) at standard atmospheric pressure."),
        ]
        for i in range(num_samples):
            q, a = qa_pairs[i % len(qa_pairs)]
            text = f"Question: {q}\nAnswer: {a}\n"
            if i % 2 == 0:
                text = f"Q: {q}\nA: {a}\n"
            data.append({"text": text, "skill": "qa"})
    
    elif skill_name == "summarize":
        passages = [
            ("The Amazon rainforest, often referred to as the lungs of the Earth, is the world's largest tropical rainforest. It covers approximately 5.5 million square kilometers and is home to an estimated 10% of all species on Earth. The forest plays a crucial role in regulating the global climate by absorbing carbon dioxide and releasing oxygen.",
             "The Amazon rainforest is the world's largest tropical forest, covering 5.5M km², hosting 10% of Earth's species, and helping regulate global climate."),
            ("Artificial intelligence has made remarkable progress in recent years. Large language models can now generate human-like text, answer complex questions, and even write code. However, concerns remain about bias, safety, and the potential impact on employment.",
             "AI has advanced significantly with language models capable of text generation and coding, but concerns about bias, safety, and jobs persist."),
            ("The Great Wall of China is one of the most impressive architectural feats in history. Built over many centuries, it stretches approximately 21,196 kilometers. The wall was constructed to protect Chinese states against invasions and raids from various nomadic groups.",
             "The Great Wall of China stretches 21,196 km and was built over centuries to protect against nomadic invasions."),
            ("Climate change is causing significant shifts in weather patterns worldwide. Rising temperatures are leading to more frequent extreme weather events, melting ice caps, and rising sea levels. Scientists warn that without immediate action, these effects will worsen considerably.",
             "Climate change is driving extreme weather, melting ice, and rising seas, with scientists urging immediate action."),
            ("The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. This complex network enables consciousness, thought, memory, and emotion. Despite decades of research, many aspects of brain function remain mysterious.",
             "The brain's 86 billion interconnected neurons enable consciousness and thought, though much remains unknown."),
        ]
        for i in range(num_samples):
            passage, summary = passages[i % len(passages)]
            text = f"Text: {passage}\nSummary: {summary}\n"
            if i % 2 == 0:
                text = f"Summarize the following:\n{passage}\n\nSummary: {summary}\n"
            data.append({"text": text, "skill": "summarize"})
    
    elif skill_name == "translate":
        pairs = [
            ("Hello, how are you?", "Bonjour, comment allez-vous?"),
            ("The cat is on the table.", "Le chat est sur la table."),
            ("I love programming.", "J'adore la programmation."),
            ("What time is it?", "Quelle heure est-il?"),
            ("The weather is beautiful today.", "Le temps est beau aujourd'hui."),
            ("I am learning French.", "J'apprends le français."),
            ("She reads a book every day.", "Elle lit un livre chaque jour."),
            ("We are going to the park.", "Nous allons au parc."),
            ("The children are playing outside.", "Les enfants jouent dehors."),
            ("Thank you very much.", "Merci beaucoup."),
            ("Good morning, have a nice day.", "Bonjour, passez une bonne journée."),
            ("This restaurant is excellent.", "Ce restaurant est excellent."),
            ("I need help with my homework.", "J'ai besoin d'aide pour mes devoirs."),
            ("The train arrives at noon.", "Le train arrive à midi."),
            ("Music makes me happy.", "La musique me rend heureux."),
        ]
        for i in range(num_samples):
            en, fr = pairs[i % len(pairs)]
            text = f"English: {en}\nFrench: {fr}\n"
            if i % 2 == 0:
                text = f"Translate to French:\n{en}\n\nTranslation: {fr}\n"
            data.append({"text": text, "skill": "translate"})
    
    return data


def tokenize_dataset(data: List[Dict], tokenizer, seq_len: int) -> torch.Tensor:
    """Tokenize and pad/truncate dataset to fixed length."""
    all_ids = []
    for item in data:
        ids = tokenizer.encode(item["text"], add_special_tokens=True)
        # Pad or truncate
        if len(ids) >= seq_len:
            ids = ids[:seq_len]
        else:
            ids = ids + [tokenizer.eos_token_id] * (seq_len - len(ids))
        all_ids.append(ids)
    return torch.tensor(all_ids, dtype=torch.long)


# ============================================================
# CCL Trainer
# ============================================================

class CCLTrainer:
    """
    Trains GPT-2 with CCL framework.
    
    Supports both frozen-base (baseline) and live projected gradient modes.
    Tracks interference, forgetting, epsilon drift, and per-skill metrics.
    """
    
    def __init__(self, config: CCLConfig):
        self.config = config
        self.device = config.device
        
        # Create model
        self.model = CCLGPT2(config)
        
        # Gradient engine
        self.grad_engine = ProjectedGradientEngine(self.model, config)
        
        # Metrics
        self.metrics = MetricsTracker(config)
        
        # Optimizer
        trainable = self.model.get_all_trainable_params()
        self.optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=0.01)
        print(f"  Optimizer: AdamW, lr={config.lr}, {sum(p.numel() for p in trainable):,} trainable params")
        
        # Datasets
        self._prepare_datasets()
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
    
    def _prepare_datasets(self):
        """Generate and tokenize datasets for all skills."""
        print("\n  Preparing datasets...")
        self.train_data = {}
        self.eval_data = {}
        
        for idx, skill in enumerate(SKILL_NAMES):
            # Training data
            train_samples = generate_synthetic_dataset(
                self.model.tokenizer, skill, num_samples=400, seq_len=self.config.seq_len
            )
            self.train_data[idx] = tokenize_dataset(
                train_samples, self.model.tokenizer, self.config.seq_len
            ).to(self.device)
            
            # Eval data (separate samples)
            eval_samples = generate_synthetic_dataset(
                self.model.tokenizer, skill, num_samples=50, seq_len=self.config.seq_len
            )
            self.eval_data[idx] = tokenize_dataset(
                eval_samples, self.model.tokenizer, self.config.seq_len
            ).to(self.device)
            
            print(f"    {skill}: {self.train_data[idx].shape[0]} train, "
                  f"{self.eval_data[idx].shape[0]} eval samples")
    
    def eval_skill_loss(self, skill_idx: int) -> float:
        """Evaluate loss on a skill's eval set."""
        self.model.base_model.eval()
        with torch.no_grad():
            data = self.eval_data[skill_idx]
            # Use first batch
            batch_size = min(self.config.batch_size * 4, data.shape[0])
            input_ids = data[:batch_size]
            labels = input_ids.clone()
            
            outputs = self.model(input_ids, skill_idx, labels=labels)
            loss = outputs.loss.item()
        self.model.base_model.train()
        return loss
    
    def eval_perplexity(self, skill_idx: int) -> float:
        """Compute perplexity on eval set."""
        loss = self.eval_skill_loss(skill_idx)
        return math.exp(min(loss, 20))  # Cap to avoid overflow
    
    def train_step(self, skill_idx: int, input_ids: torch.Tensor) -> float:
        """
        Single training step for a skill.
        
        Frozen mode: standard backprop (only adapter grads flow)
        Live mode: backprop + project base gradients into skill subspace
        """
        self.model.base_model.train()
        labels = input_ids.clone()
        
        # Forward
        outputs = self.model(input_ids, skill_idx, labels=labels)
        loss = outputs.loss / self.config.grad_accum_steps
        
        # Backward
        loss.backward()
        
        # Project base gradients (live mode only)
        if not self.config.frozen_base:
            self.grad_engine.project_base_gradients(skill_idx)
        
        return loss.item() * self.config.grad_accum_steps
    
    def train(self):
        """
        Main training loop with interleaved skill training.
        """
        print(f"\n{'=' * 60}")
        print(f"  CCL Training: {'FROZEN' if self.config.frozen_base else 'LIVE PROJECTED'}")
        print(f"  Skills: {', '.join(SKILL_NAMES)}")
        print(f"  Epochs: {self.config.num_epochs}, Batch: {self.config.batch_size}")
        print(f"  Rank: {self.config.adapter_rank}, LR: {self.config.lr}")
        print(f"{'=' * 60}\n")
        
        global_step = 0
        
        # Record initial losses
        print("  Initial eval losses:")
        for skill_idx in range(self.config.num_skills):
            loss = self.eval_skill_loss(skill_idx)
            ppl = self.eval_perplexity(skill_idx)
            self.metrics.record(f"loss/{SKILL_NAMES[skill_idx]}", 0, loss)
            self.metrics.record(f"ppl/{SKILL_NAMES[skill_idx]}", 0, ppl)
            print(f"    {SKILL_NAMES[skill_idx]}: loss={loss:.4f}, ppl={ppl:.2f}")
        
        eps = self.grad_engine.measure_epsilon()
        self.metrics.record("epsilon", 0, eps)
        print(f"  Initial ε = {eps:.6f}")
        
        for epoch in range(self.config.num_epochs):
            epoch_losses = {i: [] for i in range(self.config.num_skills)}
            
            # Interleaved training: cycle through skills
            num_batches = min(self.train_data[i].shape[0] for i in range(self.config.num_skills)) // self.config.batch_size
            
            for batch_idx in range(num_batches):
                for skill_idx in range(self.config.num_skills):
                    # Get batch
                    start = batch_idx * self.config.batch_size
                    end = start + self.config.batch_size
                    input_ids = self.train_data[skill_idx][start:end]
                    
                    # Train step
                    loss = self.train_step(skill_idx, input_ids)
                    epoch_losses[skill_idx].append(loss)
                    
                    # Gradient accumulation
                    if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.get_all_trainable_params(),
                            self.config.max_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                        
                        # Record metrics
                        self.metrics.record(f"train_loss/{SKILL_NAMES[skill_idx]}", 
                                          global_step, loss)
                        
                        # Logging
                        if global_step % self.config.log_every == 0:
                            losses_str = " | ".join(
                                f"{SKILL_NAMES[i]}={np.mean(epoch_losses[i][-5:]):.4f}"
                                for i in range(self.config.num_skills)
                            )
                            print(f"    [{epoch+1}/{self.config.num_epochs}] "
                                  f"step {global_step}: {losses_str}")
                        
                        # Eval & epsilon check
                        if global_step % self.config.eval_every == 0:
                            for si in range(self.config.num_skills):
                                eval_loss = self.eval_skill_loss(si)
                                eval_ppl = self.eval_perplexity(si)
                                self.metrics.record(f"loss/{SKILL_NAMES[si]}", 
                                                  global_step, eval_loss)
                                self.metrics.record(f"ppl/{SKILL_NAMES[si]}", 
                                                  global_step, eval_ppl)
                            
                            eps = self.grad_engine.maybe_reorthogonalize()
                            self.metrics.record("epsilon", global_step, eps)
                            
                            if global_step % (self.config.eval_every * 2) == 0:
                                print(f"    ε = {eps:.6f}, "
                                      f"reorth count = {self.grad_engine.reorth_count}")
            
            # End of epoch eval
            print(f"\n  === Epoch {epoch+1} ===")
            for si in range(self.config.num_skills):
                loss = self.eval_skill_loss(si)
                ppl = self.eval_perplexity(si)
                self.metrics.record(f"loss/{SKILL_NAMES[si]}", global_step, loss)
                self.metrics.record(f"ppl/{SKILL_NAMES[si]}", global_step, ppl)
                print(f"    {SKILL_NAMES[si]}: loss={loss:.4f}, ppl={ppl:.2f}")
            eps = self.grad_engine.measure_epsilon()
            self.metrics.record("epsilon", global_step, eps)
            print(f"    ε = {eps:.6f}\n")
        
        return global_step
    
    def compute_interference_matrix(self) -> np.ndarray:
        """
        Compute cross-skill interference matrix.
        
        M[i,j] = loss_i(after training j) - loss_i(before training j)
        Positive = forgetting, negative = positive transfer.
        """
        print("\n  Computing interference matrix...")
        n = self.config.num_skills
        matrix = np.zeros((n, n))
        
        for train_skill in range(n):
            # Record all losses before
            losses_before = [self.eval_skill_loss(si) for si in range(n)]
            
            # Do a few training steps on train_skill
            for step in range(10):
                batch_idx = step % (self.train_data[train_skill].shape[0] // self.config.batch_size)
                start = batch_idx * self.config.batch_size
                end = start + self.config.batch_size
                input_ids = self.train_data[train_skill][start:end]
                
                self.optimizer.zero_grad()
                loss = self.train_step(train_skill, input_ids)
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_all_trainable_params(), self.config.max_grad_norm
                )
                self.optimizer.step()
            
            # Record all losses after
            losses_after = [self.eval_skill_loss(si) for si in range(n)]
            
            for eval_skill in range(n):
                matrix[train_skill, eval_skill] = losses_after[eval_skill] - losses_before[eval_skill]
        
        return matrix
    
    def verify_eta_squared_scaling(self, etas: List[float] = None) -> List[dict]:
        """
        Verify O(η²) interference scaling.
        
        For each learning rate η:
        1. Start from current model state
        2. Do one step on skill 1
        3. Measure interference on skill 0
        """
        if etas is None:
            etas = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        
        print("\n  Verifying O(η²) scaling...")
        results = []
        
        # Save current state
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        for eta in etas:
            # Restore state
            self.model.load_state_dict({k: v.clone() for k, v in state.items()})
            
            # Create optimizer with this lr
            trainable = self.model.get_all_trainable_params()
            opt = torch.optim.SGD(trainable, lr=eta)
            
            # Measure skill 0 loss before
            loss_before = self.eval_skill_loss(0)
            
            # One step on skill 1
            opt.zero_grad()
            input_ids = self.train_data[1][:self.config.batch_size]
            outputs = self.model(input_ids, 1, labels=input_ids.clone())
            outputs.loss.backward()
            if not self.config.frozen_base:
                self.grad_engine.project_base_gradients(1)
            opt.step()
            
            # Measure skill 0 loss after
            loss_after = self.eval_skill_loss(0)
            interference = abs(loss_after - loss_before)
            
            results.append({
                "eta": eta,
                "interference": interference,
                "eta_sq": eta ** 2,
                "ratio": interference / (eta ** 2) if eta > 0 else 0
            })
            print(f"    η={eta:.5f}  interference={interference:.8f}  η²={eta**2:.8f}  "
                  f"ratio={results[-1]['ratio']:.4f}")
        
        # Restore state
        self.model.load_state_dict(state)
        
        return results
    
    def save_results(self):
        """Save all results to disk."""
        save_dir = self.config.save_dir
        
        # Save metrics
        self.metrics.save(os.path.join(save_dir, "metrics.json"))
        
        # Save config
        config_dict = {k: str(v) if isinstance(v, torch.dtype) else v 
                      for k, v in self.config.__dict__.items()}
        with open(os.path.join(save_dir, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"\n  Results saved to {save_dir}/")


# ============================================================
# Plotting
# ============================================================

def generate_plots(metrics: MetricsTracker, config: CCLConfig, 
                   interference_matrix: np.ndarray,
                   eta_results: List[dict],
                   save_dir: str):
    """Generate all CCL analysis plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig_dir = os.path.join(save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Per-skill loss over training
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, skill in enumerate(SKILL_NAMES):
        series = metrics.get(f"loss/{skill}")
        if series:
            steps, vals = zip(*series)
            ax.plot(steps, vals, label=skill, linewidth=2)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Per-Skill Loss During Interleaved CCL Training", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "skill_losses.png"), dpi=150)
    plt.close(fig)
    
    # 2. Perplexity over training
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, skill in enumerate(SKILL_NAMES):
        series = metrics.get(f"ppl/{skill}")
        if series:
            steps, vals = zip(*series)
            ax.plot(steps, vals, label=skill, linewidth=2)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title("Per-Skill Perplexity During CCL Training", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "skill_perplexity.png"), dpi=150)
    plt.close(fig)
    
    # 3. Epsilon (orthogonality deviation) over time
    fig, ax = plt.subplots(figsize=(10, 5))
    series = metrics.get("epsilon")
    if series:
        steps, vals = zip(*series)
        ax.plot(steps, vals, 'b-o', linewidth=2, markersize=4)
        ax.axhline(y=config.eps_threshold, color='r', linestyle='--', 
                   label=f'Threshold (ε={config.eps_threshold})')
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("ε (Orthogonality Deviation)", fontsize=12)
    ax.set_title("Subspace Orthogonality Deviation Over Training", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "epsilon_drift.png"), dpi=150)
    plt.close(fig)
    
    # 4. Interference matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(interference_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-np.abs(interference_matrix).max(),
                   vmax=np.abs(interference_matrix).max())
    ax.set_xticks(range(config.num_skills))
    ax.set_yticks(range(config.num_skills))
    ax.set_xticklabels(SKILL_NAMES, fontsize=11)
    ax.set_yticklabels(SKILL_NAMES, fontsize=11)
    ax.set_xlabel("Evaluated Skill", fontsize=12)
    ax.set_ylabel("Trained Skill", fontsize=12)
    ax.set_title("Cross-Skill Interference Matrix\n(+ve = forgetting, -ve = transfer)", fontsize=14)
    
    # Add values
    for i in range(config.num_skills):
        for j in range(config.num_skills):
            ax.text(j, i, f"{interference_matrix[i,j]:.4f}",
                   ha='center', va='center', fontsize=10,
                   color='white' if abs(interference_matrix[i,j]) > np.abs(interference_matrix).max()/2 else 'black')
    
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "interference_matrix.png"), dpi=150)
    plt.close(fig)
    
    # 5. O(η²) scaling verification
    if eta_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        etas = [r["eta"] for r in eta_results]
        interfs = [r["interference"] for r in eta_results]
        eta_sqs = [r["eta_sq"] for r in eta_results]
        
        # Log-log plot
        ax1.loglog(etas, interfs, 'bo-', linewidth=2, markersize=8, label='Measured interference')
        ax1.loglog(etas, eta_sqs, 'r--', linewidth=2, label='η² reference')
        ax1.set_xlabel("Learning Rate η", fontsize=12)
        ax1.set_ylabel("Interference", fontsize=12)
        ax1.set_title("Interference vs Learning Rate (log-log)", fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Ratio plot (should be ~constant if O(η²))
        ratios = [r["ratio"] for r in eta_results]
        ax2.plot(etas, ratios, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel("Learning Rate η", fontsize=12)
        ax2.set_ylabel("Interference / η²", fontsize=12)
        ax2.set_title("Interference / η² (should be ~constant)", fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "eta_squared_scaling.png"), dpi=150)
        plt.close(fig)
    
    print(f"  Plots saved to {fig_dir}/")


# ============================================================
# Main
# ============================================================

def run_experiment(mode: str = "live", save_dir: str = None):
    """
    Run complete CCL experiment on GPT-2 124M.
    
    Args:
        mode: "live" (projected gradients) or "frozen" (baseline)
        save_dir: output directory
    """
    if save_dir is None:
        save_dir = f"ccl_results/{mode}"
    
    config = CCLConfig(
        frozen_base=(mode == "frozen"),
        save_dir=save_dir,
        adapter_rank=16,
        num_epochs=3,
        batch_size=4,
        seq_len=128,
        lr=5e-5 if mode == "live" else 1e-4,
        grad_accum_steps=4,
        log_every=10,
        eval_every=25,
        reorth_every=25,
    )
    
    trainer = CCLTrainer(config)
    
    # Train
    total_steps = trainer.train()
    
    # Interference matrix
    interference = trainer.compute_interference_matrix()
    print("\n  Interference Matrix:")
    for i, skill_i in enumerate(SKILL_NAMES):
        row = " ".join(f"{interference[i,j]:+.5f}" for j in range(config.num_skills))
        print(f"    {skill_i:>12}: {row}")
    
    # O(η²) verification
    eta_results = trainer.verify_eta_squared_scaling()
    
    # Save
    trainer.save_results()
    
    # Save additional results
    np.save(os.path.join(save_dir, "interference_matrix.npy"), interference)
    with open(os.path.join(save_dir, "eta_scaling.json"), 'w') as f:
        json.dump(eta_results, f, indent=2)
    
    # Plots
    generate_plots(trainer.metrics, config, interference, eta_results, save_dir)
    
    # Summary
    print(trainer.metrics.summary())
    
    return trainer, interference, eta_results


def run_comparison():
    """Run both frozen and live experiments for comparison."""
    print("=" * 70)
    print("  CCL on GPT-2 124M: Full Comparison")
    print("=" * 70)
    
    # Frozen baseline
    print("\n\n" + "=" * 70)
    print("  PHASE 1: FROZEN BASE (baseline)")
    print("=" * 70)
    frozen_trainer, frozen_interf, frozen_eta = run_experiment("frozen", "ccl_results/frozen")
    
    # Live projected
    print("\n\n" + "=" * 70)
    print("  PHASE 2: LIVE PROJECTED GRADIENTS")
    print("=" * 70)
    live_trainer, live_interf, live_eta = run_experiment("live", "ccl_results/live")
    
    # Comparison summary
    print("\n\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    
    print("\n  Final losses (frozen vs live):")
    print(f"  {'Skill':<12} {'Frozen':>10} {'Live':>10} {'Δ':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for si, skill in enumerate(SKILL_NAMES):
        fl = frozen_trainer.eval_skill_loss(si)
        ll = live_trainer.eval_skill_loss(si)
        print(f"  {skill:<12} {fl:>10.4f} {ll:>10.4f} {ll-fl:>+10.4f}")
    
    print(f"\n  Max interference (frozen): {np.abs(frozen_interf).max():.6f}")
    print(f"  Max interference (live):   {np.abs(live_interf).max():.6f}")
    
    frozen_eps = frozen_trainer.grad_engine.measure_epsilon()
    live_eps = live_trainer.grad_engine.measure_epsilon()
    print(f"\n  Final ε (frozen): {frozen_eps:.6f}")
    print(f"  Final ε (live):   {live_eps:.6f}")
    
    print(f"\n  Re-orthogonalizations (live): {live_trainer.grad_engine.reorth_count}")
    
    # Generate comparison plot
    _generate_comparison_plots(frozen_trainer, live_trainer, frozen_interf, live_interf)


def _generate_comparison_plots(frozen_trainer, live_trainer, frozen_interf, live_interf):
    """Generate side-by-side comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig_dir = "ccl_results/comparison"
    os.makedirs(fig_dir, exist_ok=True)
    
    # Side-by-side loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, skill in enumerate(SKILL_NAMES):
        series = frozen_trainer.metrics.get(f"loss/{skill}")
        if series:
            steps, vals = zip(*series)
            ax1.plot(steps, vals, label=skill, linewidth=2)
    ax1.set_title("Frozen Base (Baseline)", fontsize=14)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for idx, skill in enumerate(SKILL_NAMES):
        series = live_trainer.metrics.get(f"loss/{skill}")
        if series:
            steps, vals = zip(*series)
            ax2.plot(steps, vals, label=skill, linewidth=2)
    ax2.set_title("Live Projected Gradients", fontsize=14)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle("CCL on GPT-2 124M: Frozen vs Live", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "frozen_vs_live_losses.png"), dpi=150)
    plt.close(fig)
    
    # Side-by-side interference matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    vmax = max(np.abs(frozen_interf).max(), np.abs(live_interf).max())
    
    for ax, matrix, title in [(ax1, frozen_interf, "Frozen"), (ax2, live_interf, "Live")]:
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(SKILL_NAMES, fontsize=10)
        ax.set_yticklabels(SKILL_NAMES, fontsize=10)
        ax.set_title(f"{title} Interference", fontsize=13)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{matrix[i,j]:.4f}", ha='center', va='center', fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    
    fig.suptitle("Cross-Skill Interference: Frozen vs Live", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "interference_comparison.png"), dpi=150)
    plt.close(fig)
    
    print(f"  Comparison plots saved to {fig_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CCL on GPT-2 124M")
    parser.add_argument("--mode", choices=["frozen", "live", "both"], default="both",
                       help="Training mode")
    parser.add_argument("--rank", type=int, default=16, help="Adapter rank")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    args = parser.parse_args()
    
    if args.mode == "both":
        run_comparison()
    else:
        run_experiment(args.mode)
