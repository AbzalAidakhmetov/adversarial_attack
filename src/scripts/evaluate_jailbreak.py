# 0. Setup 
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from modsteer import PROJECT_ROOT
import numpy as np
import omegaconf
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from nnsight import LanguageModel

from modsteer.dataset.load_dataset import load_dataset_split
from modsteer.utils import evaluate_perplexity
from modsteer.steering.utils import (
    compute_mean_activations,
    to_chat,
    generate_with_steered_model,
    select_candidate_layers,
)

import hydra
from hydra import initialize, compose
from typing import Dict, List
from modsteer.utils import set_seed
from modsteer.steering.utils import evaluate_steering

from modsteer.pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
import logging

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"nnsight\.intervention\.interleaver",
)

logging.getLogger("LiteLLM").setLevel(logging.WARNING)

def run(cfg: omegaconf.DictConfig):

    set_seed(cfg.seed)

    Path(cfg.results_path).mkdir(parents=True, exist_ok=True)

    model = LanguageModel(cfg.model, device_map=cfg.device, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    directions: torch.Tensor = torch.load(cfg.directions_path)
    
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
    )

    results


def evaluate_configs(
    model: LanguageModel,
    tokenizer,
    directions: Dict[int, Dict[int, torch.Tensor]],
    prompts: List[Dict],
    cfg,
) -> List[Dict]:
    
    results = []

    for layer in tqdm(cfg.steering_layers, desc="Layers"):

        for steering_weight in tqdm(cfg.steering_weights, desc=f"Weights@L{layer}", leave=False):
            
            prompts_with_completions = generate_completions(
                model,
                tokenizer,
                prompts,
                layer,
                steering_weight,
                directions,
                cfg.max_new_tokens,
            )

            steering_success = evaluate_steering(
                completions=prompts_with_completions, attribute=cfg.attribute
            )

            jailbreak_results = evaluate_jailbreak(
                completions=prompts_with_completions,
                methodologies=cfg.eval_methods,
                evaluation_path=Path(cfg.results_path) / f"refusal_{cfg.eval_source}_eval_L{layer}_w{steering_weight}.json",
            )
            
            perplexity_results = evaluate_perplexity(prompts_with_completions)

            rec = {
                "layer": int(layer),
                "weight": float(steering_weight),
                "llamaguard2_success_rate": float(jailbreak_results["llamaguard2_success_rate"]) if "llamaguard2" in cfg.eval_methods else None,
                "llama33_success_rate": float(jailbreak_results["llama33_success_rate"]) if "llama33" in cfg.eval_methods else None,
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
) -> List[str]:
    
    for prompt in tqdm(prompts, desc=f"Generating@L{layer}_w{weight}", leave=False):

        prompt_text = prompt['prompt']
        
        completion = generate_with_steered_model(
            model,
            tokenizer,
            prompt_text,
            layer_idx=layer,
            direction=directions,
            weight=weight,
            max_new_tokens=max_new_tokens,
        )

        prompt['response'] = completion

    return prompts
        

@hydra.main(config_path=str(PROJECT_ROOT / "config"), config_name="evaluate_jailbreak.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
