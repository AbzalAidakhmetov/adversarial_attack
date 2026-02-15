"""
Orthogonalization utilities for steering vectors.

This module provides functions to orthogonalize steering directions against
reference activation subspaces, helping to create more targeted interventions
that don't interfere with model capabilities.
"""

from typing import Optional, Literal, Callable
import torch
from modsteer.steering.utils import compute_eoi_toks_custom, to_chat, _get_layers_module
from tqdm import tqdm


def compute_reference_activations(
    model,
    tokenizer,
    positive_prompts: list[str],
    negative_prompts: list[str],
    device: str = "cuda",
    batch_size: int = 16,
) -> torch.Tensor:
    """
    Compute activation differences (positive - negative) for reference data.
    
    Args:
        model: The language model (nnsight LanguageModel)
        tokenizer: The tokenizer
        positive_prompts: List of prompts for positive direction (e.g., harmful)
        negative_prompts: List of prompts for negative direction (e.g., harmless)
        device: Device to use (activations will be accumulated on CPU to save VRAM)
        batch_size: Batch size for processing
    
    Returns:
        sum_of_activations: (num_layers, num_prompts, num_token_positions, embedding_dim)
    """
    assert len(positive_prompts) == len(negative_prompts), "Pos/neg prompt counts must match."
    
    # Convert to chat format
    positive_prompts_chat = [to_chat(tokenizer, p) for p in positive_prompts]
    negative_prompts_chat = [to_chat(tokenizer, p) for p in negative_prompts]
    
    # Compute end-of-instruction tokens
    eoi_toks = compute_eoi_toks_custom(tokenizer)
    positions = list(range(-len(eoi_toks) - 1, 0))
    
    # Initialize activation storage
    num_layers = model.config.num_hidden_layers
    num_token_positions = len(positions)
    embedding_dim = model.config.hidden_size
    num_prompts = len(positive_prompts_chat)
    
    # Store on CPU to avoid OOM
    sum_of_activations = torch.zeros(
        (num_layers, num_prompts, num_token_positions, embedding_dim),
        dtype=torch.float64,
        device="cpu",  # Force CPU
    )
    
    layers = _get_layers_module(model)
    
    with torch.no_grad():
        for i in range(0, num_prompts, batch_size):
            batch_prompts_pos = positive_prompts_chat[i : i + batch_size]
            batch_prompts_neg = negative_prompts_chat[i : i + batch_size]
            cur_bs = len(batch_prompts_pos)
            handles_pos = []
            handles_neg = []

            
            with model.trace(batch_prompts_pos):
                for l in range(num_layers):
                    handles_pos.append(layers[l].output[0].save())
            
            with model.trace(batch_prompts_neg):
                for l in range(num_layers):
                    handles_neg.append(layers[l].output[0].save())
            
            for l, h in enumerate(handles_pos):
                activations_at_pos = h[:, positions]
                sum_of_activations[l, i:i + cur_bs] += activations_at_pos.to(dtype=sum_of_activations.dtype, device=sum_of_activations.device)
            
            for l, h in enumerate(handles_neg):
                activations_at_neg = h[:, positions]
                sum_of_activations[l, i:i + cur_bs] -= activations_at_neg.to(dtype=sum_of_activations.dtype, device=sum_of_activations.device)
            
            # Clean up to free GPU memory
            del handles_pos, handles_neg
            torch.cuda.empty_cache()

    return sum_of_activations


def _make_projector_from_SVD(S: torch.Tensor, energy_to_remove: float = 0.95, verbose: bool = False) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a projector that removes components in the span of S's top singular vectors.
    
    Args:
        S: Matrix whose column space defines the subspace to project out
        energy_to_remove: Fraction of singular value energy to remove (top components)
    
    Returns:
        A function that projects vectors orthogonal to the retained subspace
    """
    if S.numel() == 0 or S.shape[1] == 0:
        raise ValueError("Input matrix S must have at least one column.")
    
    U, s, _ = torch.linalg.svd(S, full_matrices=False)
    energy = s.square()
    total_energy = energy.sum().clamp_min(1e-12)
    cum_energy = torch.cumsum(energy, dim=0) / total_energy
    
    # Find k such that we keep components explaining 'energy_to_remove' fraction of variance
    k = max(1, int((cum_energy <= energy_to_remove + 1e-9).sum().item()))
    
    if verbose:
        tqdm.write(f"  SVD Spectrum: Top 5 vals: {s[:5].tolist()}")
        tqdm.write(f"  Removing top {k} components (Energy: {cum_energy[k-1].item():.4f})")

    U_k = U[:, :k]
    
    def projector(v: torch.Tensor):
        # Support (d,), (d,1), or (n,d) shapes
        if v.dim() == 1:  # (d,)
            return v - U_k @ (U_k.T @ v)
        if v.dim() == 2 and v.shape[0] == S.shape[0] and v.shape[1] == 1:  # (d,1)
            v1 = v.squeeze(1)
            out = v1 - U_k @ (U_k.T @ v1)
            return out.unsqueeze(1)
        if v.dim() == 2 and v.shape[1] == S.shape[0]:  # (n,d) row-batch
            return v - (v @ U_k) @ U_k.T
        raise ValueError(f"Unexpected v shape {tuple(v.shape)}")
    
    return projector


def orthogonalize_direction(
    model,
    tokenizer,
    direction: torch.Tensor,
    reference_activations: torch.Tensor,
    method: Literal["svd", "mean_null", "qr", "gram_schmidt", "whitening", "fisher", "median_null", "pca", "hard_mean"] = "svd",
    energy_to_remove: float = 0.95,
    position_indices: Optional[list[int]] = None,
    layers: Optional[list[int]] = None,
    lambda_coeff: float = 1.0,
    renorm: Literal["match_norm", "none", "cap"] = "match_norm",
    renorm_cap: float = 2.0,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Orthogonalize a direction vector against reference activations.
    
    Args:
        direction: Direction tensor to orthogonalize, shape (num_layers, embedding_dim)
        reference_activations: Pre-computed activations, shape (num_layers, num_prompts, num_positions, embedding_dim)
        method: Orthogonalization method:
            - "svd": SVD-based projection (removes top-k components of prompt-averaged differences)
            - "mean_null": Project away from mean harmful direction
            - "qr": QR decomposition (stable orthogonalization)
            - "gram_schmidt": Classical Gram-Schmidt process
            - "whitening": Covariance eigenvector-based projection
            - "fisher": LDA-style projection (whitened mean direction)
            - "median_null": Robust projection using coordinate-wise median
            - "pca": Principal Component Analysis on all differences (uncentered)
            - "hard_mean": Aggressive projection against the single mean difference vector
        energy_to_remove: Fraction of energy to remove for SVD/whitening
        position_indices: Specific position indices to use (defaults to all)
        layers: List of layer indices to orthogonalize. If None, all layers are processed.
                Layers not in this list will keep their original direction unchanged.
        lambda_coeff: Partial removal coefficient (0 = no removal, 1 = full removal)
        renorm: How to renormalize after orthogonalization ("match_norm", "none", "cap")
        renorm_cap: Max allowed gain when renorm="cap"
        verbose: Print debug info
    
    Returns:
        Orthogonalized direction tensor, same shape as input
    """
    if position_indices is None:
        eoi_toks = compute_eoi_toks_custom(tokenizer)
        position_indices = list(range(-len(eoi_toks) - 1, 0))
    
    assert direction.shape[0] == model.config.num_hidden_layers, "Direction tensor must have the same number of layers as the model"
    
    L = direction.shape[0]
    orthogonalized = torch.empty_like(direction)
    
    # Determine which layers to orthogonalize
    if layers is None:
        layers_to_process = set(range(L))
    else:
        layers_to_process = set(layers)
        # Validate layer indices
        invalid_layers = [l for l in layers_to_process if l < 0 or l >= L]
        if invalid_layers:
            raise ValueError(f"Invalid layer indices: {invalid_layers}. Must be in range [0, {L-1}]")
    
    def _apply_renorm(v: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
        v_norm = v.norm().clamp_min(1e-12)
        if renorm == "none":
            return v
        if renorm == "match_norm":
            scale = target_norm / v_norm
        elif renorm == "cap":
            scale = min((target_norm / v_norm).item(), renorm_cap)
        else:
            raise ValueError(f"Unknown renorm mode '{renorm}'")
        return v * scale

    if method == "svd":
        for l in tqdm(range(L)):
            if l not in layers_to_process or direction[l].norm() < 1e-12:
                orthogonalized[l] = direction[l]
                continue
            
            # Mean over prompts, so S_l is (positions, dim).T -> (dim, positions)
            S_l = reference_activations[l, :, position_indices, :].mean(dim=1).T.to(direction.device)
            if verbose: tqdm.write(f"Layer {l} SVD input norm: {S_l.norm():.4f}")
            
            proj_l = _make_projector_from_SVD(S_l, energy_to_remove, verbose=verbose)
            v = direction[l].to(S_l.dtype).to(S_l.device)
            v_proj_full = proj_l(v)
            proj_component = v - v_proj_full
            v_partial = v - lambda_coeff * proj_component
            v_out = _apply_renorm(v_partial, direction[l].norm().clamp_min(1e-12))
            orthogonalized[l] = v_out
    
    elif method == "mean_null":
        for l in tqdm(range(L)):
            if l not in layers_to_process or direction[l].norm() < 1e-12:
                orthogonalized[l] = direction[l]
                continue

            H = reference_activations[l, :, position_indices, :].to(direction.device)
            h_mean = H.mean(dim=(0, 1))
            
            v = direction[l].to(H.dtype).to(H.device)
            h_norm = h_mean.norm().item()
            
            if h_norm < 1e-12:
                if verbose: tqdm.write(f"Layer {l}: Mean vector is zero.")
                orthogonalized[l] = v
                continue
            
            u = h_mean / h_mean.norm().clamp_min(1e-12)
            v_proj_full = v - torch.dot(u, v) * u
            proj_component = v - v_proj_full
            v_partial = v - lambda_coeff * proj_component
            v_out = _apply_renorm(v_partial, v.norm().clamp_min(1e-12))
            orthogonalized[l] = v_out
    
    elif method == "qr":
        for l in tqdm(range(L)):
            if l not in layers_to_process or direction[l].norm() < 1e-12:
                orthogonalized[l] = direction[l]
                continue

            H = reference_activations[l, :, position_indices, :].reshape(-1, reference_activations.shape[-1]).T.to(direction.device)
            v = direction[l].to(H.dtype).to(H.device)
            
            # QR decomposition: removes components in span of H
            Q, _ = torch.linalg.qr(H, mode='reduced')
            v_proj_full = v - Q @ (Q.T @ v)
            proj_component = v - v_proj_full
            v_partial = v - lambda_coeff * proj_component
            v_out = _apply_renorm(v_partial, v.norm().clamp_min(1e-12))
            orthogonalized[l] = v_out
    
    elif method == "gram_schmidt":
        for l in tqdm(range(L)):
            if l not in layers_to_process or direction[l].norm() < 1e-12:
                orthogonalized[l] = direction[l]
                continue

            H = reference_activations[l, :, position_indices, :].reshape(-1, reference_activations.shape[-1]).to(direction.device)
            v = direction[l].to(H.dtype).to(H.device)

            # Build an orthonormal basis of span(H) via Modified Gram–Schmidt
            # Treat rows of H as vectors in R^d, form matrix A with those as columns (d x N)
            if H.numel() == 0:
                orthogonalized[l] = v
                continue
            A = H.T  # (d, N)
            d, N = A.shape
            orthonormal_columns = []
            for j in range(N):
                candidate = A[:, j]
                if candidate.norm() <= 1e-12:
                    continue
                q = candidate.clone()
                for q_prev in orthonormal_columns:
                    q = q - (q_prev @ q) * q_prev
                q_norm = q.norm()
                if q_norm <= 1e-12:
                    continue
                orthonormal_columns.append(q / q_norm)

            if len(orthonormal_columns) == 0:
                orthogonalized[l] = v
                continue

            Q = torch.stack(orthonormal_columns, dim=1)  # (d, r)
            v_proj_full = v - Q @ (Q.T @ v)
            if verbose:
                tqdm.write(
                    f"Layer {l}: v_proj norm {v_proj_full.norm().item():.4f}, "
                    f"v norm {v.norm().item():.4f}, "
                    f"percent preserved {v_proj_full.norm().item() / v.norm().item():.4f}"
                )
            proj_component = v - v_proj_full
            v_partial = v - lambda_coeff * proj_component
            v_out = _apply_renorm(v_partial, v.norm().clamp_min(1e-12))
            orthogonalized[l] = v_out
    
    elif method == "whitening":
        for l in tqdm(range(L)):
            if l not in layers_to_process or direction[l].norm() < 1e-12:
                orthogonalized[l] = direction[l]
                continue

            H = reference_activations[l, :, position_indices, :].reshape(-1, reference_activations.shape[-1]).to(direction.device)
            v = direction[l].to(H.dtype).to(H.device)
            
            if H.shape[0] <= 1:
                orthogonalized[l] = v
                continue
            
            H_centered = H - H.mean(dim=0)
            cov = (H_centered.T @ H_centered) / (H_centered.shape[0] - 1 + 1e-8)
            U, s, _ = torch.linalg.svd(cov)
            
            # Project out top-k eigenvectors
            k = max(1, int((s.cumsum(0) / s.sum() <= energy_to_remove).sum()))
            U_k = U[:, :k]
            v_proj_full = v - U_k @ (U_k.T @ v)
            proj_component = v - v_proj_full
            v_partial = v - lambda_coeff * proj_component
            v_out = _apply_renorm(v_partial, v.norm().clamp_min(1e-12))
            orthogonalized[l] = v_out

    elif method == "fisher":
        for l in tqdm(range(L)):
            if l not in layers_to_process or direction[l].norm() < 1e-12:
                orthogonalized[l] = direction[l]
                continue

            H = reference_activations[l, :, position_indices, :].reshape(-1, reference_activations.shape[-1]).to(direction.device)
            v = direction[l].to(H.dtype).to(H.device)
            
            if H.shape[0] <= 1:
                orthogonalized[l] = v
                continue
            
            h_mean = H.mean(dim=0)
            if verbose: tqdm.write(f"Layer {l}: Mean diff norm {h_mean.norm().item():.4f}")

            H_centered = H - h_mean
            # Regularized covariance approximation
            cov = (H_centered.T @ H_centered) / (H_centered.shape[0] - 1 + 1e-8)
            reg = 1e-4 * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
            
            try:
                # Solve for whitened mean: (Sigma + lambda*I)^(-1) * mu
                # Uses cholesky or LU implicitly
                w = torch.linalg.solve(cov + reg, h_mean)
            except RuntimeError:
                # Fallback to pseudo-inverse if solve fails
                w = torch.linalg.pinv(cov + reg) @ h_mean
            
            w_norm = w.norm()
            if w_norm < 1e-12:
                if verbose: tqdm.write(f"Layer {l}: Fisher direction is zero.")
                orthogonalized[l] = v
                continue
                
            w = w / w_norm
            v_proj_full = v - torch.dot(w, v) * w
            proj_component = v - v_proj_full
            v_partial = v - lambda_coeff * proj_component
            v_out = _apply_renorm(v_partial, v.norm().clamp_min(1e-12))
            orthogonalized[l] = v_out

    elif method == "median_null":
        for l in tqdm(range(L)):
            if l not in layers_to_process or direction[l].norm() < 1e-12:
                orthogonalized[l] = direction[l]
                continue

            H = reference_activations[l, :, position_indices, :].to(direction.device)
            H_flat = H.reshape(-1, H.shape[-1])
            # Coordinate-wise median
            h_median = torch.median(H_flat, dim=0).values
            
            v = direction[l].to(H.dtype).to(H.device)
            u = h_median / h_median.norm().clamp_min(1e-12)
            
            v_proj_full = v - torch.dot(u, v) * u
            proj_component = v - v_proj_full
            v_partial = v - lambda_coeff * proj_component
            v_out = _apply_renorm(v_partial, v.norm().clamp_min(1e-12))
            orthogonalized[l] = v_out

    elif method == "pca":
        for l in tqdm(range(L)):
            if l not in layers_to_process or direction[l].norm() < 1e-12:
                orthogonalized[l] = direction[l]
                continue

            # Use all differences (uncentered) for PCA to find dominant subspace
            # H shape: (dim, num_samples)
            H = reference_activations[l, :, position_indices, :].reshape(-1, reference_activations.shape[-1]).T.to(direction.device)
            
            if verbose: tqdm.write(f"Layer {l}: PCA Input shape {H.shape}")
            
            proj_l = _make_projector_from_SVD(H, energy_to_remove, verbose=(verbose and l == 12))
            v = direction[l].to(H.dtype).to(H.device)
            v_proj_full = proj_l(v)
            proj_component = v - v_proj_full
            v_partial = v - lambda_coeff * proj_component
            v_out = _apply_renorm(v_partial, direction[l].norm().clamp_min(1e-12))
            orthogonalized[l] = v_out

    elif method == "hard_mean":
        for l in tqdm(range(L)):
            if l not in layers_to_process or direction[l].norm() < 1e-12:
                orthogonalized[l] = direction[l]
                continue

            # Compute global mean difference vector across all prompts and positions
            H = reference_activations[l, :, position_indices, :].to(direction.device)
            h_mean = H.mean(dim=(0, 1)) # (dim,)
            
            v = direction[l].to(H.dtype).to(H.device)
            
            # Normalize mean vector
            if h_mean.norm() < 1e-12:
                orthogonalized[l] = v
                continue
                
            u = h_mean / h_mean.norm()
            
            # Project v onto u
            proj = torch.dot(v, u) * u
            v_proj_full = v - proj
            proj_component = v - v_proj_full
            v_partial = v - lambda_coeff * proj_component
            v_out = _apply_renorm(v_partial, v.norm().clamp_min(1e-12))
            orthogonalized[l] = v_out
    
    return orthogonalized


def orthogonalize_direction_from_data(
    direction: torch.Tensor,
    model,
    tokenizer,
    method: Literal["svd", "mean_null", "qr", "gram_schmidt", "whitening", "fisher", "median_null", "pca", "hard_mean"] = "svd",
    num_samples: int = 128,
    energy_to_remove: float = 0.95,
    device: str = "cuda",
    batch_size: int = 16,
    position_indices: Optional[list[int]] = None,
    layers: Optional[list[int]] = None,
    lambda_coeff: float = 1.0,
    renorm: Literal["match_norm", "none", "cap"] = "match_norm",
    renorm_cap: float = 2.0,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Orthogonalize an innocuous direction against the harmful/harmless subspace.
    
    Args:
        direction: Innocuous direction tensor (num_layers, embedding_dim)
        model: The language model (nnsight LanguageModel)
        tokenizer: The tokenizer
        method: Orthogonalization method (see orthogonalize_direction for details)
        num_samples: Number of harmful/harmless samples to use
        energy_to_remove: Fraction of energy to remove for SVD/whitening
        device: Device to use
        batch_size: Batch size for processing
        position_indices: Specific position indices to use (defaults to all)
        layers: List of layer indices to orthogonalize. If None, all layers are processed.
                Layers not in this list will keep their original direction unchanged.
        lambda_coeff: Partial removal coefficient (0 = no removal, 1 = full removal)
        renorm: How to renormalize after orthogonalization ("match_norm", "none", "cap")
        renorm_cap: Max allowed gain when renorm="cap"
        verbose: Print debug info
    
    Returns:
        Orthogonalized direction tensor
    """
    import random
    import json
    from pathlib import Path
    from modsteer.dataset.load_dataset import load_dataset_split
    random.seed(42)

    # 1. Load reserved test prompts to exclude
    excluded_prompts = set()
    # Paths to the files containing prompts to exclude
    exclusion_paths = [
        "./data/refusal/harmful_prompts.json",
        "./data/refusal/harmless_prompts.json"
    ]
    
    for p_str in exclusion_paths:
        p = Path(p_str)
        if p.exists():
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                    for item in data:
                        if isinstance(item, dict) and "prompt" in item:
                            excluded_prompts.add(item["prompt"])
            except Exception as e:
                if verbose:
                    print(f"Warning: Error loading exclusion file {p}: {e}")

    # 2. Gather pool from all splits (train + test)
    harmful_pool = []
    harmless_pool = []
    
    for split in ["train"]: # "train", "test"
        try:
            harmful_pool.extend(load_dataset_split(harmtype='harmful', split=split, instructions_only=True))
            harmless_pool.extend(load_dataset_split(harmtype='harmless', split=split, instructions_only=True))
        except Exception:
            raise Exception(f"Error loading dataset split {split}")

    # 3. Filter out excluded prompts and deduplicate
    harmful_pool = list(set(harmful_pool) - excluded_prompts)
    harmless_pool = list(set(harmless_pool) - excluded_prompts)
    
    # 4. Sample
    n_available = min(len(harmful_pool), len(harmless_pool))
    n_use = min(num_samples, n_available)
    
    if verbose:
        print(f"Dataset Pool: {len(harmful_pool)} harmful, {len(harmless_pool)} harmless (after exclusion).")
        print(f"Using {n_use} samples per class.")

    harmful_train = random.sample(harmful_pool, n_use)
    harmless_train = random.sample(harmless_pool, n_use)
    
    reference_activations = compute_reference_activations(
        model=model,
        tokenizer=tokenizer,
        positive_prompts=harmful_train,
        negative_prompts=harmless_train,
        device=device,
        batch_size=batch_size,
    )
    
    return orthogonalize_direction(
        model=model, 
        tokenizer=tokenizer,
        direction=direction,
        reference_activations=reference_activations,
        method=method,
        energy_to_remove=energy_to_remove,
        position_indices=position_indices,
        layers=layers,
        lambda_coeff=lambda_coeff,
        renorm=renorm,
        renorm_cap=renorm_cap,
        verbose=verbose,
    )
