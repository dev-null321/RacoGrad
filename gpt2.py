#!/usr/bin/env python3
"""
RacoGrad GPT-2 Implementation
Decoder-only transformer with pretrained weight loading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

class GPT2Attention(nn.Module):
    """Multi-head causal self-attention."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.c_attn = nn.Linear(d_model, 3 * d_model)  # Q, K, V combined
        self.c_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.size()
        
        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(out)


class GPT2MLP(nn.Module):
    """Feed-forward network with GELU."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff)
        self.c_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate='tanh')  # GPT-2 uses tanh approximation
        x = self.c_proj(x)
        return self.dropout(x)


class GPT2Block(nn.Module):
    """Transformer block: attention + MLP with pre-norm."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = GPT2Attention(d_model, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = GPT2MLP(d_model, d_ff, dropout)
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """GPT-2 model."""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len=1024, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.wte = nn.Embedding(vocab_size, d_model)  # Token embeddings
        self.wpe = nn.Embedding(max_len, d_model)     # Position embeddings
        self.drop = nn.Dropout(dropout)
        
        d_ff = 4 * d_model  # GPT-2 uses 4x expansion
        self.blocks = nn.ModuleList([
            GPT2Block(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.wte.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx):
        B, T = idx.size()
        
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            # Crop to max_len if needed
            idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]
            
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
            
        return idx


def load_gpt2_weights(model, model_name='gpt2'):
    """Load pretrained GPT-2 weights from HuggingFace."""
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        print("Install transformers: pip install transformers")
        return False
    
    print(f"Loading {model_name} weights from HuggingFace...")
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_sd = hf_model.state_dict()
    
    # Map HuggingFace keys to our model
    mapping = {
        'transformer.wte.weight': 'wte.weight',
        'transformer.wpe.weight': 'wpe.weight',
        'transformer.ln_f.weight': 'ln_f.weight',
        'transformer.ln_f.bias': 'ln_f.bias',
    }
    
    for i in range(len(model.blocks)):
        prefix_hf = f'transformer.h.{i}'
        prefix_our = f'blocks.{i}'
        
        mapping.update({
            f'{prefix_hf}.ln_1.weight': f'{prefix_our}.ln_1.weight',
            f'{prefix_hf}.ln_1.bias': f'{prefix_our}.ln_1.bias',
            f'{prefix_hf}.attn.c_attn.weight': f'{prefix_our}.attn.c_attn.weight',
            f'{prefix_hf}.attn.c_attn.bias': f'{prefix_our}.attn.c_attn.bias',
            f'{prefix_hf}.attn.c_proj.weight': f'{prefix_our}.attn.c_proj.weight',
            f'{prefix_hf}.attn.c_proj.bias': f'{prefix_our}.attn.c_proj.bias',
            f'{prefix_hf}.ln_2.weight': f'{prefix_our}.ln_2.weight',
            f'{prefix_hf}.ln_2.bias': f'{prefix_our}.ln_2.bias',
            f'{prefix_hf}.mlp.c_fc.weight': f'{prefix_our}.mlp.c_fc.weight',
            f'{prefix_hf}.mlp.c_fc.bias': f'{prefix_our}.mlp.c_fc.bias',
            f'{prefix_hf}.mlp.c_proj.weight': f'{prefix_our}.mlp.c_proj.weight',
            f'{prefix_hf}.mlp.c_proj.bias': f'{prefix_our}.mlp.c_proj.bias',
        })
    
    our_sd = model.state_dict()
    
    for hf_key, our_key in mapping.items():
        if hf_key in hf_sd and our_key in our_sd:
            # HuggingFace uses Conv1D which stores weights transposed
            if 'attn.c_attn' in hf_key or 'attn.c_proj' in hf_key or 'mlp.c_fc' in hf_key or 'mlp.c_proj' in hf_key:
                if 'weight' in hf_key:
                    our_sd[our_key] = hf_sd[hf_key].t()
                else:
                    our_sd[our_key] = hf_sd[hf_key]
            else:
                our_sd[our_key] = hf_sd[hf_key]
    
    model.load_state_dict(our_sd)
    print("Weights loaded successfully!")
    return True


# GPT-2 configurations
GPT2_CONFIGS = {
    'gpt2': dict(vocab_size=50257, d_model=768, num_heads=12, num_layers=12),        # 124M
    'gpt2-medium': dict(vocab_size=50257, d_model=1024, num_heads=16, num_layers=24), # 350M
    'gpt2-large': dict(vocab_size=50257, d_model=1280, num_heads=20, num_layers=36),  # 774M
    'gpt2-xl': dict(vocab_size=50257, d_model=1600, num_heads=25, num_layers=48),     # 1.5B
}


def create_gpt2(model_name='gpt2', device='cuda', load_pretrained=True):
    """Create GPT-2 model, optionally with pretrained weights."""
    config = GPT2_CONFIGS[model_name]
    model = GPT2(**config).to(device)
    
    print(f"Created {model_name}: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    if load_pretrained:
        load_gpt2_weights(model, model_name)
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2', choices=list(GPT2_CONFIGS.keys()))
    parser.add_argument('--prompt', default='The meaning of life is')
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create and load model
    model = create_gpt2(args.model, device)
    model.eval()
    
    # Tokenize
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    
    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)
    
    input_ids = tokenizer.encode(args.prompt, return_tensors='pt').to(device)
    output_ids = model.generate(input_ids, args.max_tokens, args.temperature, args.top_k)
    
    output_text = tokenizer.decode(output_ids[0])
    print(output_text)
