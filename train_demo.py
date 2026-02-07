#!/usr/bin/env python3
"""
RacoGrad Training Demo - Copy Task
Proves the transformer architecture trains correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys

class TransformerModel(nn.Module):
    """Transformer for sequence-to-sequence tasks."""
    
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, dim_ff=256, max_len=64):
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = self._create_sinusoidal_pe(max_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, batch_first=True)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def _create_sinusoidal_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * 
                       (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, src, tgt):
        src_emb = self.src_embed(src) + self.pos_enc[:, :src.size(1)]
        tgt_emb = self.tgt_embed(tgt) + self.pos_enc[:, :tgt.size(1)]
        
        memory = self.encoder(src_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.output_proj(output)


def train_copy_task(vocab_size=16, seq_len=10, d_model=64, nhead=4, num_layers=2,
                    epochs=20, batches=50, batch_size=32, lr=0.001, device="cuda"):
    """Train transformer on copy task (output = input)."""
    
    model = TransformerModel(vocab_size, d_model, nhead, num_layers, d_model*4, seq_len*2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"=== RacoGrad Copy Task Demo ===")
    print(f"Config: vocab={vocab_size}, seq_len={seq_len}, d_model={d_model}, heads={nhead}, layers={num_layers}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for _ in range(batches):
            src = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
            tgt = src.clone()
            
            logits = model(src, tgt)
            loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == tgt).sum().item()
            total += tgt.numel()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {total_loss/batches:.4f} | Acc: {acc:.1f}%")
        
        if acc >= 99.9:
            print(f"\nConverged at epoch {epoch+1}!")
            break
    
    # Final test
    print("\n--- Final Test ---")
    src = torch.randint(1, vocab_size, (1, seq_len), device=device)
    with torch.no_grad():
        pred = model(src, src).argmax(dim=-1)
    print(f"Input:  {src[0].tolist()}")
    print(f"Output: {pred[0].tolist()}")
    print(f"Match:  {(src == pred).all().item()}")
    
    return model


def train_reverse_task(vocab_size=16, seq_len=10, d_model=64, nhead=4, num_layers=2,
                       epochs=30, batches=50, batch_size=32, lr=0.001, device="cuda"):
    """Train transformer on reverse task (output = reversed input)."""
    
    model = TransformerModel(vocab_size, d_model, nhead, num_layers, d_model*4, seq_len*2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"=== RacoGrad Reverse Task Demo ===")
    print(f"Config: vocab={vocab_size}, seq_len={seq_len}, d_model={d_model}, heads={nhead}, layers={num_layers}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for _ in range(batches):
            src = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
            tgt = src.flip(dims=[1])  # Reverse
            
            logits = model(src, tgt)
            loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == tgt).sum().item()
            total += tgt.numel()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {total_loss/batches:.4f} | Acc: {acc:.1f}%")
        
        if acc >= 99.9:
            print(f"\nConverged at epoch {epoch+1}!")
            break
    
    # Final test
    print("\n--- Final Test ---")
    src = torch.randint(1, vocab_size, (1, seq_len), device=device)
    expected = src.flip(dims=[1])
    with torch.no_grad():
        pred = model(src, expected).argmax(dim=-1)
    print(f"Input:    {src[0].tolist()}")
    print(f"Expected: {expected[0].tolist()}")
    print(f"Output:   {pred[0].tolist()}")
    print(f"Match:    {(expected == pred).all().item()}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RacoGrad Training Demo")
    parser.add_argument("--task", choices=["copy", "reverse"], default="copy")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--vocab", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    if args.task == "copy":
        train_copy_task(args.vocab, args.seq_len, args.d_model, args.heads, args.layers, args.epochs, device=device)
    else:
        train_reverse_task(args.vocab, args.seq_len, args.d_model, args.heads, args.layers, args.epochs, device=device)
