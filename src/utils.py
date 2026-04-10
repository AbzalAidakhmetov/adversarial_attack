"""
Perplexity computation utilities using GPT-2.
Extracted from the LLM-Steer-Instruct notebook for reuse.
"""

import torch
from typing import Optional
import random 
import numpy as np

# Global variables for lazy loading
_perplx_model = None
_perplx_tok = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_perplexity_model(device: str = 'cpu'):
    """Lazy load GPT-2 model and tokenizer for perplexity computation."""
    global _perplx_model, _perplx_tok
    if _perplx_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _perplx_model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')
        _perplx_tok = AutoTokenizer.from_pretrained('openai-community/gpt2')
        _perplx_model.to(device)
        _perplx_model.eval()
    return _perplx_model, _perplx_tok


@torch.no_grad()
def compute_perplexity(text: str, device: str = 'cpu', max_len: Optional[int] = None) -> float:
    """
    Compute perplexity of text using GPT-2.
    
    Args:
        text: Input text to compute perplexity for
        device: Device to run computation on ('cpu' or 'cuda')
        max_len: Maximum length for tokenization (default: model's max length)
        
    Returns:
        Perplexity score as float. Returns inf for empty/invalid text.
    """
    model_plx, tok = _load_perplexity_model(device)
    if max_len is None:
        max_len = getattr(tok, 'model_max_length', 1024)
        if max_len is None or max_len > 4096 or max_len == int(1e30):
            max_len = 1024
    
    # Guard against empty/whitespace-only responses which cause GPT-2 to error
    if not isinstance(text, str) or len(text.strip()) == 0:
        return float('inf')
    
    enc = tok(text, return_tensors='pt', truncation=True, max_length=max_len)
    # Some tokenizers can still yield zero-length encodings
    if enc['input_ids'].numel() == 0 or enc['input_ids'].shape[-1] == 0:
        return float('inf')
    
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model_plx(**enc, labels=enc['input_ids'])
    return float(torch.exp(out.loss).item())

def evaluate_perplexity(completions: list[dict], threshold: float = 3.0) -> dict[str, float]:
    """
    Evaluate perplexity for a list of completion dictionaries.
    
    Args:
        completions: List of dicts with 'response' key containing text.
        threshold: Perplexity threshold to compute low perplexity fraction.
    Returns:
        Dictionary with mean perplexity and fraction below threshold.
    """
    perplx_scores = []
    for comp in completions:
        response = comp['response']
        perplx = compute_perplexity(response, device='cuda' if torch.cuda.is_available() else 'cpu')
        perplx_scores.append(perplx)
    perplx_scores_tensor = torch.tensor(perplx_scores)
    mean_perplx = torch.mean(perplx_scores_tensor)
    low_perplx_fraction = torch.sum(perplx_scores_tensor < threshold).item() / len(perplx_scores_tensor)
    return {
        "mean_perplexity": mean_perplx.item(),
        "low_perplexity_fraction": low_perplx_fraction,
    }

