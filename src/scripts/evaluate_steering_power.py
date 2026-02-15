# 0. Setup
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import omegaconf
import torch
from modsteer import PROJECT_ROOT
from modsteer.steering.utils import _get_layers_module, to_chat
from modsteer.utils import set_seed
from nnsight import LanguageModel
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer

import logging
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"nnsight\.intervention\.interleaver",
)

logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# SAE Definition (from notebook)
class SparseAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
        self.dtype = dtype
        self.to(self.device, self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of data using a linear, followed by a ReLU."""
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of data using a linear."""
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SAE forward pass. Returns the reconstruction and the encoded features."""
        f = self.encode(x)
        return self.decode(f), f

class SAELensWrapper(torch.nn.Module):
    def __init__(self, sae):
        super().__init__()
        self.sae = sae
        self.device = sae.device
        self.dtype = sae.dtype

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae.encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae.decode(x)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sae(x)


def load_sae(
    path: str,
    d_model: int,
    expansion_factor: int,
    device: torch.device,
    model_name: str,
) -> torch.nn.Module:
    model_name = model_name.lower()
    path_lower = path.lower()

    # Gemma: always pull Layer 12 SAE from sae_lens for consistent experiments
    if "gemma" in model_name or "gemma" in path_lower:
        try:
            from sae_lens import SAE

            print(
                "Loading Gemma SAE (layer_12/width_16k/canonical) from HuggingFace..."
            )
            sae_obj = SAE.from_pretrained(
                release="gemma-scope-2b-pt-res-canonical",
                sae_id="layer_12/width_16k/canonical",
                device=str(device),
            )
            # sae_lens returns (sae, cfg, sparsity) in some versions
            sae_model = sae_obj[0] if isinstance(sae_obj, tuple) else sae_obj
            return SAELensWrapper(sae_model)
        except ImportError:
            print("sae_lens not installed")
        except Exception as e:
            print(f"Failed to load Gemma SAE: {e}")

    # Directory containing saved sae_lens weights
    if os.path.isdir(path):
        try:
            from sae_lens import SAE

            sae_obj = SAE.load_from_disk(path, device=str(device))
            sae_model = sae_obj[0] if isinstance(sae_obj, tuple) else sae_obj
            return SAELensWrapper(sae_model)
        except ImportError:
            print("sae_lens not installed or failed to load from directory, falling back to standard load")
        except Exception as e:
            print(f"Failed to load with sae_lens: {e}")

    sae = SparseAutoEncoder(
        d_model,
        d_model * expansion_factor,
        device,
    )
    
    if os.path.isdir(path):
        # Fallback for directory if sae_lens failed but maybe it's our format?
        pass
    else:
        sae_dict = torch.load(
            path, weights_only=True, map_location=device
        )
        sae.load_state_dict(sae_dict)

    return sae

def measure_feature_activation_preservation(
    prompts: List[Dict],
    direction_original: torch.Tensor,
    direction_safe: torch.Tensor,
    layer_idx: int,
    feature_idx: int,
    weight: float,
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    sae: SparseAutoEncoder,
) -> Dict[str, float]:
    """
    Measure how well the safe steering vector preserves the original feature activation.
    
    Args:
        prompts: List of prompts to test
        direction_original: Original steering direction
        direction_safe: Safe (orthogonalized) steering direction
        layer_idx: Layer index to steer
        feature_idx: SAE feature index to measure
        weight: Steering weight
        model: The language model
        tokenizer: The tokenizer
        sae: The SAE model
    
    Returns:
        dict with mean_delta_orig, mean_delta_safe, preserve_ratio
    """
    delta_orig_list = []
    delta_safe_list = []

    # Ensure directions are on the correct device
    direction_original = direction_original.to(model.device)
    direction_safe = direction_safe.to(model.device)

    for prompt in tqdm(prompts, desc="Measuring activation", leave=False):
        prompt_text = prompt['prompt']
        prompt_chat = to_chat(tokenizer, prompt_text)
        
        # We need to process one by one or batch. The snippet does one by one.
        # To work with nnsight properly, we should use .trace() context.
        
        # 1. Baseline (no steering)
        with torch.no_grad():
            with model.trace(prompt_chat):
                layers = _get_layers_module(model)
                # Accessing output[0] of the layer. 
                # note: .save() in nnsight saves the proxy.
                baseline_acts_proxy = layers[layer_idx].output[0].save()
            
            # .value or .save() result needs to be accessed after trace
            # Since we are iterating, we should fetch the value.
            baseline_acts = baseline_acts_proxy
            baseline_acts = baseline_acts.squeeze(0)  # [seq_len, d_model]
        
        # 2. Original brand steering
        with torch.no_grad():
            with model.trace(prompt_chat):
                layers = _get_layers_module(model)
                # Add steering vector
                # We need to be careful with nnsight syntax for inplace modification or addition
                # The snippet uses: layers[layer_idx].output[0][:] += direction_original * weight
                layers[layer_idx].output[0][:] += direction_original * weight
                orig_acts_proxy = layers[layer_idx].output[0].save()
            
            orig_acts = orig_acts_proxy.squeeze(0)
        
        # 3. Safe brand steering
        with torch.no_grad():
            with model.trace(prompt_chat):
                layers = _get_layers_module(model)
                layers[layer_idx].output[0][:] += direction_safe * weight
                safe_acts_proxy = layers[layer_idx].output[0].save()
            
            safe_acts = safe_acts_proxy.squeeze(0)
        
        # Encode with SAE to get feature activations
        # Note: SAE is a torch module, we can run it on the tensor values
        baseline_features = sae.encode(baseline_acts)  # [seq_len, d_hidden]
        orig_features = sae.encode(orig_acts)
        safe_features = sae.encode(safe_acts)
        
        # Feature activation (average over tokens)
        # The snippet takes mean over tokens
        delta_orig = (orig_features[:, feature_idx] - baseline_features[:, feature_idx]).mean().item()
        delta_safe = (safe_features[:, feature_idx] - baseline_features[:, feature_idx]).mean().item()
        
        delta_orig_list.append(delta_orig)
        delta_safe_list.append(delta_safe)

    # Average across prompts
    mean_delta_orig = np.mean(delta_orig_list)
    mean_delta_safe = np.mean(delta_safe_list)

    preserve_ratio = mean_delta_safe / mean_delta_orig if mean_delta_orig != 0 else 0

    print(f"\nAverage Δ feature activation (original): {mean_delta_orig:.4f}")
    print(f"Average Δ feature activation (safe): {mean_delta_safe:.4f}")
    print(f"Preserved brand power ratio: {preserve_ratio:.4f} ({preserve_ratio*100:.1f}%)")
    
    return {
        "mean_delta_orig": mean_delta_orig,
        "mean_delta_safe": mean_delta_safe,
        "preserve_ratio": preserve_ratio
    }


def evaluate_configs(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    direction_original: torch.Tensor,
    direction_safe: Dict[str, torch.Tensor] | torch.Tensor,
    prompts: List[Dict],
    sae: torch.nn.Module,
    cfg: omegaconf.DictConfig,
) -> List[Dict]:
    
    results = []

    # If direction_safe is just a tensor, wrap it in a dict
    if isinstance(direction_safe, torch.Tensor):
        safe_directions = {"safe": direction_safe}
    elif isinstance(direction_safe, dict):
        safe_directions = direction_safe
    else:
        raise ValueError(f"Unsupported type for direction_safe: {type(direction_safe)}")

    layer = 12  # experiment focuses on layer 12

    if direction_original.ndim > 1:
        current_direction_orig = direction_original[layer]
    else:
        current_direction_orig = direction_original

    for method_name, d_safe in safe_directions.items():
        if d_safe.ndim > 1:
            current_direction_safe = d_safe[layer]
        else:
            current_direction_safe = d_safe

        for steering_weight in tqdm(cfg.steering_weights, desc=f"Weights@L{layer}-{method_name}", leave=False):
            
            metrics = measure_feature_activation_preservation(
                prompts=prompts,
                direction_original=current_direction_orig,
                direction_safe=current_direction_safe,
                layer_idx=layer,
                feature_idx=cfg.feature_idx,
                weight=steering_weight,
                model=model,
                tokenizer=tokenizer,
                sae=sae,
            )

            rec = {
                "layer": int(layer),
                "method": method_name,
                "weight": float(steering_weight),
                "mean_delta_orig": float(metrics["mean_delta_orig"]),
                "mean_delta_safe": float(metrics["mean_delta_safe"]),
                "preserve_ratio": float(metrics["preserve_ratio"]),
            }

            results.append(rec)
    
    with open(Path(cfg.results_path) / "results_steering_power.json", "w") as f:
        json.dump(results, f, indent=4)
        print(f"Results saved to {Path(cfg.results_path) / 'results_steering_power.json'}")
    return results


def run(cfg: omegaconf.DictConfig):

    set_seed(cfg.seed)

    Path(cfg.results_path).mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {cfg.model}")
    model = LanguageModel(cfg.model, device_map=cfg.device, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    direction_safe, direction_original = _load_direction_tensors(cfg)

    print(f"Loaded {len(direction_safe)} safe directions: {list(direction_safe.keys())}")

    if direction_original is None:
        raise ValueError("No raw/original direction found. Provide a file without 'safe' in its name or set original_directions_path.")

    d_model = model.config.hidden_size
    sae = load_sae(
        path=cfg.sae_path,
        d_model=d_model,
        expansion_factor=cfg.sae_expansion_factor,
        device=torch.device(cfg.device),
        model_name=cfg.model_name,
    )

    with open(cfg.prompts_path, "r") as f:
        prompts = json.load(f)

    num_eval_samples = min(cfg.val_samples, len(prompts)) if cfg.val_samples is not None else len(prompts)
    prompts = prompts[:num_eval_samples]

    print(f"Evaluation prompts: {prompts[:3]} ...")
    results = evaluate_configs(
        model=model,
        tokenizer=tokenizer,
        direction_original=direction_original,
        direction_safe=direction_safe,
        prompts=prompts,
        sae=sae,
        cfg=cfg,
    )

    return results


def _load_direction_tensors(cfg: omegaconf.DictConfig) -> Tuple[Dict[str, Tensor], Tensor | None]:
    """
    Load safe and raw directions based on filenames. Filenames containing 'safe'
    are treated as safe vectors, otherwise as the raw/original direction.
    """
    def expand_paths(entry: str) -> List[str]:
        entry_path = Path(entry)
        if entry_path.is_dir():
            return sorted(str(p) for p in entry_path.glob("*.pt"))
        return [str(entry_path)]

    # Collect every path
    path_entries = cfg.directions_path
    if isinstance(path_entries, (list, omegaconf.listconfig.ListConfig)):
        candidate_paths: List[str] = []
        for entry in path_entries:
            candidate_paths.extend(expand_paths(entry))
    else:
        candidate_paths = expand_paths(path_entries)

    safe_vectors: Dict[str, Tensor] = {}
    raw_direction: Tensor | None = None

    for path_str in candidate_paths:
        tensor = torch.load(path_str, map_location="cpu")
        name = Path(path_str).stem
        if "safe" in name:
            method = name.split("_safe_")[-1] if "_safe_" in name else name
            safe_vectors[method] = tensor
        else:
            raw_direction = tensor

    # Fallback to explicit path if provided
    original_path = cfg.get("original_directions_path")
    if raw_direction is None and original_path:
        raw_direction = torch.load(original_path, map_location="cpu")

    if not safe_vectors:
        raise ValueError(
            "No safe directions were loaded. Provide files containing 'safe' in their name."
        )

    return safe_vectors, raw_direction


@hydra.main(config_path=str(PROJECT_ROOT / "config"), config_name="evaluate_steering_power.yaml", version_base="1.2")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
