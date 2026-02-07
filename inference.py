#!/usr/bin/env python3
"""
RacoGrad Inference - Greedy and Beam Search Decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def greedy_decode(model, src, max_len, start_token=0, end_token=None, device="cuda"):
    """
    Greedy decoding: always pick highest probability token.
    
    Args:
        model: Transformer model
        src: Source sequence (batch, src_len)
        max_len: Maximum output length
        start_token: Start-of-sequence token
        end_token: End-of-sequence token (optional, stops early if seen)
        
    Returns:
        Generated sequence (batch, out_len)
    """
    batch_size = src.size(0)
    
    # Start with just the start token
    tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
    
    for _ in range(max_len - 1):
        # Get logits for next token
        logits = model(src, tgt)  # (batch, tgt_len, vocab)
        next_logits = logits[:, -1, :]  # (batch, vocab)
        
        # Greedy: pick highest prob
        next_token = next_logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
        
        # Append to sequence
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # Early stop if all sequences hit end token
        if end_token is not None and (next_token == end_token).all():
            break
    
    return tgt


def beam_search_decode(model, src, max_len, beam_size=4, start_token=0, end_token=None, 
                       length_penalty=0.6, device="cuda"):
    """
    Beam search decoding: maintain top-k hypotheses.
    
    Args:
        model: Transformer model
        src: Source sequence (1, src_len) - single example only
        max_len: Maximum output length
        beam_size: Number of beams to keep
        start_token: Start-of-sequence token
        end_token: End-of-sequence token
        length_penalty: Length normalization (alpha in paper)
        
    Returns:
        Best sequence (1, out_len), score
    """
    assert src.size(0) == 1, "Beam search works on single examples"
    
    vocab_size = model.output_proj.out_features
    
    # Expand src for beam
    src_expanded = src.repeat(beam_size, 1)  # (beam, src_len)
    
    # Initialize beams: (beam_size, 1)
    beams = torch.full((beam_size, 1), start_token, dtype=torch.long, device=device)
    beam_scores = torch.zeros(beam_size, device=device)
    beam_scores[1:] = float('-inf')  # Only first beam active initially
    
    # Completed hypotheses
    completed = []
    
    for step in range(max_len - 1):
        # Forward pass
        logits = model(src_expanded, beams)  # (beam, tgt_len, vocab)
        next_logits = logits[:, -1, :]  # (beam, vocab)
        log_probs = F.log_softmax(next_logits, dim=-1)
        
        # Compute scores for all possible next tokens
        # scores: (beam, vocab)
        scores = beam_scores.unsqueeze(1) + log_probs
        
        if step == 0:
            # First step: only consider first beam
            scores = scores[0:1]
        
        # Flatten and get top-k
        scores_flat = scores.view(-1)  # (beam * vocab) or (vocab)
        top_scores, top_indices = scores_flat.topk(beam_size)
        
        # Convert flat indices back to beam and token indices
        if step == 0:
            beam_indices = torch.zeros(beam_size, dtype=torch.long, device=device)
            token_indices = top_indices
        else:
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
        
        # Update beams
        new_beams = torch.cat([
            beams[beam_indices],
            token_indices.unsqueeze(1)
        ], dim=1)
        beam_scores = top_scores
        beams = new_beams
        
        # Check for completed sequences
        if end_token is not None:
            for i in range(beam_size):
                if token_indices[i] == end_token:
                    # Length-normalized score
                    length = beams.size(1)
                    norm_score = beam_scores[i] / (length ** length_penalty)
                    completed.append((beams[i].clone(), norm_score.item()))
        
        # If we have enough completed, stop
        if len(completed) >= beam_size:
            break
    
    # Add remaining beams to completed
    for i in range(beam_size):
        length = beams.size(1)
        norm_score = beam_scores[i] / (length ** length_penalty)
        completed.append((beams[i], norm_score.item()))
    
    # Sort by score and return best
    completed.sort(key=lambda x: x[1], reverse=True)
    best_seq, best_score = completed[0]
    
    return best_seq.unsqueeze(0), best_score


# Export for Racket FFI
def create_decoder(model, method="greedy", beam_size=4, max_len=50, 
                   start_token=0, end_token=None, device="cuda"):
    """Create a decoder function for the model."""
    
    if method == "greedy":
        def decode(src):
            return greedy_decode(model, src, max_len, start_token, end_token, device)
        return decode
    else:
        def decode(src):
            return beam_search_decode(model, src, max_len, beam_size, 
                                     start_token, end_token, device=device)
        return decode


if __name__ == "__main__":
    # Demo with a trained model
    from train_demo import TransformerModel, train_copy_task
    
    print("Training model for decode demo...")
    model = train_copy_task(epochs=5, batches=30)
    
    print("\n=== Decode Demo ===")
    device = "cuda"
    
    # Test input
    src = torch.tensor([[3, 7, 2, 9, 5, 1, 8, 4, 6, 10]], device=device)
    print(f"Source: {src[0].tolist()}")
    
    # Greedy decode (for copy task, should match input)
    print("\nGreedy decode:")
    output = greedy_decode(model, src, max_len=10, start_token=3, device=device)
    print(f"Output: {output[0].tolist()}")
    
    # Beam search
    print("\nBeam search (beam=4):")
    output, score = beam_search_decode(model, src, max_len=10, beam_size=4, 
                                        start_token=3, device=device)
    print(f"Output: {output[0].tolist()}")
    print(f"Score:  {score:.4f}")
