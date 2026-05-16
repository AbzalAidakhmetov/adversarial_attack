import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import omegaconf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import hydra

from classifiers import evaluate_perplexity, set_seed
from steering import generate_steered_batched, evaluate_steering
from classifiers import evaluate_jailbreak

PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging

logging.getLogger("LiteLLM").setLevel(logging.WARNING)

def load_directions(cfg):
    """Load steering directions from a .pt file.

    Supports two formats:
      1. Raw tensor indexed by layer (e.g. shape [num_layers, d_model]).
         Uses cfg.steering_layers as-is.
      2. Dict from build_adv_stealth.py with keys
         'steering_vector_poisoned' (1-D) and 'layer' (int).
         Builds a sparse tensor so that directions[layer] works,
         and overrides steering_layers to [layer].

    For dict-format files, two booleans pick which 1-D vector to load:
      - use_clean   : clean vs. poisoned
      - use_defended: original vs. refusal-orthogonalized (from
                      defense/orthogonalize_steering.py)
    """
    data = torch.load(cfg.directions_path, map_location="cpu", weights_only=False)

    if isinstance(data, dict) and "steering_vector_poisoned" in data:
        use_clean    = getattr(cfg, "use_clean", False)
        use_defended = getattr(cfg, "use_defended", False)
        base = "steering_vector_clean" if use_clean else "steering_vector_poisoned"
        key  = f"{base}_defended" if use_defended else base
        if key not in data:
            raise KeyError(
                f"directions_path {cfg.directions_path} has no '{key}'. "
                f"Run defense/orthogonalize_steering.py first to produce defended vectors."
            )
        vec = data[key]
        layer = data["layer"]
        # Build a tensor that can be indexed as directions[layer]
        directions = torch.zeros(layer + 1, vec.shape[0])
        directions[layer] = vec
        steering_layers = [layer]
        print(f"Loaded {key} from dict format (layer={layer}, norm={vec.norm():.4f})")
        cli_layers = getattr(cfg, "steering_layers", None)
        if cli_layers and list(cli_layers) != [layer]:
            print(f"  WARNING: CLI steering_layers={list(cli_layers)} ignored — "
                  f"saved vector is at layer {layer}.")
        return directions, steering_layers

    # Standard tensor format: must be 2-D [num_layers, d_model] so that
    # `direction[layer_idx]` returns the per-layer vector. A 1-D save would
    # silently broadcast a scalar across the whole residual stream.
    if not isinstance(data, torch.Tensor) or data.ndim != 2:
        raise ValueError(
            f"raw-tensor directions_path must be 2-D [num_layers, d_model]; "
            f"got {type(data).__name__}{getattr(data, 'shape', '')}"
        )
    return data, None


def run(cfg: omegaconf.DictConfig):

    set_seed(cfg.seed)

    Path(cfg.results_path).mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=torch.bfloat16, device_map=cfg.device
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    directions, steering_layers = load_directions(cfg)

    with open(cfg.prompts_path, "r") as f:
        prompts = json.load(f)

    num_eval_samples = min(cfg.val_samples, len(prompts)) if cfg.val_samples is not None else len(prompts)
    prompts = prompts[:num_eval_samples]

    print(f'Evaluation prompts: {prompts[:3]} ...')
    results = evaluate_configs(
        model=model,
        tokenizer=tokenizer,
        directions=directions,
        prompts=prompts,
        cfg=cfg,
        steering_layers_override=steering_layers,
    )

    return results


def evaluate_configs(
    model,
    tokenizer,
    directions: torch.Tensor,
    prompts: List[Dict],
    cfg,
    steering_layers_override: Optional[List[int]] = None,
) -> List[Dict]:

    results = []
    steering_layers = steering_layers_override if steering_layers_override is not None else cfg.steering_layers

    protocol = getattr(cfg, "protocol", "prefill")
    if protocol not in ("prefill", "all_steps"):
        raise ValueError(f"Unknown protocol '{protocol}'. Supported: ['all_steps', 'prefill']")
    batch_size = int(getattr(cfg, "batch_size", 8))
    print(f"Using steering protocol: {protocol} (batch_size={batch_size})")

    for layer in tqdm(steering_layers, desc="Layers"):

        for steering_weight in tqdm(cfg.steering_weights, desc=f"Weights@L{layer}", leave=False):

            prompts_with_completions = generate_completions(
                model,
                tokenizer,
                prompts,
                layer,
                steering_weight,
                directions,
                cfg.max_new_tokens,
                batch_size=batch_size,
                protocol=protocol,
            )

            steering_success = evaluate_steering(
                completions=prompts_with_completions, attribute=cfg.attribute
            )

            judge_cfg = omegaconf.OmegaConf.to_container(cfg.judge, resolve=True) if "judge" in cfg else None
            jailbreak_results = evaluate_jailbreak(
                completions=prompts_with_completions,
                methodologies=cfg.eval_methods,
                evaluation_path=Path(cfg.results_path) / f"refusal_{cfg.eval_source}_eval_L{layer}_w{steering_weight}.json",
                device=cfg.device,
                judge=judge_cfg,
            )
            
            perplexity_results = evaluate_perplexity(prompts_with_completions)

            if "judge" in cfg.eval_methods:
                jsr = jailbreak_results.get("judge_success_rate")
                judge_success_rate = float(jsr) if jsr is not None else None
                judge_skipped_count = int(jailbreak_results.get("judge_skipped_count", 0))
                judge_n_classified = int(jailbreak_results.get("judge_n_classified", 0))
            else:
                judge_success_rate = None
                judge_skipped_count = 0
                judge_n_classified = 0

            rec = {
                "layer": int(layer),
                "weight": float(steering_weight),
                "judge_success_rate": judge_success_rate,
                "judge_skipped_count": judge_skipped_count,
                "judge_n_classified": judge_n_classified,
                "steering_success_rate": steering_success,
                "mean_perplexity": float(perplexity_results["mean_perplexity"]),
                "low_perplexity_fraction": float(perplexity_results["low_perplexity_fraction"]),
            }

            results.append(rec)
    
    with open(Path(cfg.results_path) / "results", "w") as f:
        json.dump(results, f, indent=4)
        print(results)
    return results

def generate_completions(
    model,
    tokenizer,
    prompts: List[Dict],
    layer: int,
    weight: float,
    directions: torch.Tensor,
    max_new_tokens: int,
    batch_size: int = 8,
    protocol: str = "prefill",
) -> List[Dict]:

    n = len(prompts)
    pbar = tqdm(total=n, desc=f"Generating@L{layer}_w{weight}", leave=False)
    for start in range(0, n, batch_size):
        batch = prompts[start:start + batch_size]
        completions = generate_steered_batched(
            model=model,
            tokenizer=tokenizer,
            prompts=[p["prompt"] for p in batch],
            direction=directions,
            layer_idx=layer,
            weight=weight,
            max_new_tokens=max_new_tokens,
            protocol=protocol,
        )
        for p, c in zip(batch, completions):
            p["response"] = c
        pbar.update(len(batch))
    pbar.close()
    return prompts
        

@hydra.main(config_path=str(PROJECT_ROOT / "config"), config_name="evaluate_jailbreak.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
