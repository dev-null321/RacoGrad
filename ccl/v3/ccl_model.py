#!/usr/bin/env python3
"""
CCL Model Components: SharedSkillAdapter, CCLLlama, ProjectedGradientEngine
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Dict, List
from dataclasses import dataclass, field


# ============================================================
# Configuration
# ============================================================

@dataclass
class CCLConfig:
    model_name: str = "NousResearch/Meta-Llama-3-8B"
    d_model: int = 4096
    num_layers: int = 32
    adapter_rank: int = 64
    adapter_targets: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    lr: float = 1e-4
    num_epochs_per_skill: int = 5
    batch_size: int = 2
    seq_len: int = 128
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0

    num_skills: int = 4
    skill_rank: int = 32  # 50% of adapter_rank -- deliberate overlap
    overlap_fraction: float = 0.3  # 30% overlap between skill subspaces
    reorth_every: int = 50
    eps_threshold: float = 0.3
    use_projection: bool = True

    # Eta scaling
    eta_steps: int = 80
    eta_values: List[float] = field(default_factory=lambda: [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1])

    log_every: int = 5
    eval_every: int = 25
    save_dir: str = "ccl_results_llama3/v3"
    device: str = "cuda"


SKILL_NAMES = ["code", "qa", "summarize", "translate"]


# ============================================================
# Shared Adapter with Skill Projection
# ============================================================

class SharedSkillAdapter(nn.Module):
    """
    Shared low-rank adapter: DeltaW = A @ B^T
    Each skill j has projection basis U_j with DELIBERATE overlap.
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int, num_skills: int,
                 skill_rank: int, overlap_fraction: float = 0.3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.num_skills = num_skills
        self.skill_rank = skill_rank

        # Shared adapter weights
        scale = 1.0 / math.sqrt(rank)
        self.A = nn.Parameter(torch.randn(out_dim, rank) * scale)
        self.B = nn.Parameter(torch.zeros(in_dim, rank))

        # Per-skill projection bases with DELIBERATE OVERLAP
        # Random initialization -- no orthogonal guarantee
        self.U = nn.ParameterList([
            nn.Parameter(self._init_overlapping(out_dim, skill_rank, j, num_skills, overlap_fraction))
            for j in range(num_skills)
        ])

    def _init_overlapping(self, dim, rank, skill_idx, num_skills, overlap_frac):
        """Initialize skill subspaces with deliberate overlap.

        Strategy: each skill gets a random subspace. We ensure overlap by
        sharing a fraction of basis vectors between adjacent skills.
        """
        torch.manual_seed(1000 + skill_idx)
        # Start with random basis
        basis = torch.randn(dim, rank)

        if skill_idx > 0:
            # Share overlap_frac of columns with previous skill
            n_shared = max(1, int(rank * overlap_frac))
            torch.manual_seed(1000 + skill_idx - 1)
            prev_basis = torch.randn(dim, rank)
            # Replace first n_shared columns with columns from previous skill
            basis[:, :n_shared] = prev_basis[:, :n_shared]

        # Normalize columns
        norms = torch.norm(basis, dim=0, keepdim=True) + 1e-8
        basis = basis / norms * 0.1
        return basis

    def get_projection_matrix(self, skill_idx: int) -> torch.Tensor:
        """P_j = U_j (U_j^T U_j)^{-1} U_j^T"""
        U = self.U[skill_idx].float()
        gram = U.T @ U + 1e-6 * torch.eye(self.skill_rank, device=U.device)
        return U @ torch.linalg.inv(gram) @ U.T

    def forward_adapter(self, x):
        """x @ B @ A^T"""
        return x.to(self.B.dtype) @ self.B @ self.A.T


# ============================================================
# CCL-Wrapped LLaMA
# ============================================================

class CCLLlama(nn.Module):
    def __init__(self, config: CCLConfig):
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
                    in_dim, out_dim, self.config.adapter_rank,
                    self.config.num_skills, self.config.skill_rank,
                    self.config.overlap_fraction,
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
    def __init__(self, model: CCLLlama, config: CCLConfig):
        self.model = model
        self.config = config
        self.reorth_count = 0

    def project_adapter_gradients(self, skill_idx: int):
        """Project FULL DeltaW gradient into skill j's subspace.

        The adapter computes DeltaW = A @ B^T.
        The gradient of the loss w.r.t. DeltaW is G = dL/dA @ B + ... but
        more precisely, we need the full DeltaW gradient.

        G_DeltaW = A.grad @ B^T (from chain rule through the bilinear form)
        is not quite right. Instead:

        For DeltaW = A @ B^T applied as out += x @ B @ A^T:
        - dL/dA has shape (out_dim, rank) -- gradient w.r.t. output-side
        - dL/dB has shape (in_dim, rank) -- gradient w.r.t. input-side

        The full gradient of the adapter output w.r.t. the weight matrix is:
        G = dL/d(DeltaW) which has shape (out_dim, in_dim).
        We can reconstruct: G = A.grad @ B.data^T + A.data @ B.grad^T
        But this isn't right either -- A.grad and B.grad are computed assuming
        the other is fixed.

        Correct approach: Project A.grad in output space and B.grad in input space.
        For the output DeltaW = A @ B^T to lie in subspace P_j:
        - Project A.grad: A.grad_proj = P_j @ A.grad (projects output dimension)
        - Project B.grad: B.grad_proj = P_j_in @ B.grad where P_j_in projects input dim

        Actually, the cleanest approach for DeltaW = A @ B^T:
        The full gradient G = A.grad @ B^T is out_dim x in_dim
        G_proj = P_j @ G  (project rows into skill subspace)
        Then decompose back: A.grad_new = P_j @ A.grad, B keeps its gradient.

        But we also need the B-side. The correct projection for the FULL update:
        Delta(DeltaW) = dA @ B^T + A @ dB^T
        We want P_j @ Delta(DeltaW) @ P_j_in = Delta(DeltaW) (stays in subspace)

        Simplification: project A.grad rows AND B.grad rows into their respective
        skill subspaces. We maintain U_j for output dim. For input dim, use same
        approach (or just project A since it's the output-side bottleneck).

        IMPLEMENTATION: Project both A.grad and B.grad through the output-space
        projection P_j. Since B is in input space, we construct an input-space
        projection from the same skill basis rotated through the adapter.
        
        Pragmatic fix: Project A.grad through P_j (output space), and also
        construct input-space projection and apply to B.grad.
        """
        for key, info in self.model.adapted_layers.items():
            adapter = info["adapter"]
            P = adapter.get_projection_matrix(skill_idx)  # out_dim x out_dim

            # Project A's gradient (output space)
            if adapter.A.grad is not None:
                adapter.A.grad.data = (P @ adapter.A.grad.data.float()).to(adapter.A.grad.dtype)

            # Project B's gradient through the adapter relationship
            # DeltaW = A @ B^T. For the update to stay in skill subspace:
            # We need P @ (A @ dB^T) = (P @ A) @ dB^T
            # This is satisfied if A is already in the subspace (which projection ensures over time)
            # But we should also directly constrain B.
            # B.grad has shape (in_dim, rank). The output-space constraint on B is:
            # dB should be such that A @ dB^T has rows in span(U_j)
            # i.e., dB^T = A^+ @ P @ A @ dB_old^T where A^+ is pseudoinverse
            # Simpler: project the full gradient matrix and decompose.
            if adapter.B.grad is not None and adapter.A.grad is not None:
                # Compute full gradient of DeltaW
                # G = A.grad @ B.data^T  (not exactly right but captures the direction)
                # Actually, for separate A,B params, each grad is independent.
                # The key insight: B.grad columns should align with A's projected directions.
                # Project: B.grad_new = B.grad @ A^T @ P^T @ (A^T @ P^T)^+ 
                # This is complex. Simpler: use A to create input-space projection.
                A_data = adapter.A.data.float()
                PA = P @ A_data  # out_dim x rank -- A projected
                # Input-space projection: P_in = B @ A^T @ P @ A @ B^T / norms
                # Too expensive for 4096x4096. Instead, project B.grad through A:
                # dB should produce output in skill subspace when combined with A
                # A @ dB^T should be in span(P). So dB^T = A^+ @ P @ A @ dB_old^T
                # A^+ = (A^T A)^{-1} A^T
                AtA = A_data.T @ A_data + 1e-6 * torch.eye(adapter.rank, device=A_data.device)
                AtA_inv = torch.linalg.inv(AtA)
                A_pinv = AtA_inv @ A_data.T  # rank x out_dim
                
                # dB_new^T = A^+ @ P @ A @ dB_old^T
                B_grad_T = adapter.B.grad.data.float().T  # rank x in_dim
                B_grad_T_new = A_pinv @ P @ A_data @ B_grad_T  # rank x in_dim
                adapter.B.grad.data = B_grad_T_new.T.to(adapter.B.grad.dtype)

    @torch.no_grad()
    def reorthogonalize(self):
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
