# 0. Setup 
from __future__ import annotations

from pathlib import Path

from modsteer import PROJECT_ROOT
import omegaconf
import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer

import hydra
from modsteer.utils import set_seed
from modsteer.steering.orthogonalization import orthogonalize_direction_from_data

def run(cfg: omegaconf.DictConfig):
    set_seed(cfg.seed)

    Path(cfg.results_path).mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    model = LanguageModel(cfg.model, device_map=cfg.device, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)

    directions_raw: torch.Tensor = torch.load(cfg.directions_path)
    assert directions_raw.shape[0] == model.config.num_hidden_layers and directions_raw.shape[1] == model.config.hidden_size, \
        f"Expected shape (num_layers, d_model), got shape {directions_raw.shape}"

    for method in cfg.methods:
        directions_safe = orthogonalize_direction_from_data(directions_raw, model, tokenizer, method, num_samples=cfg.num_samples)
        torch.save(directions_safe, f"{cfg.results_path}/year_n_samples/direction_raw_{cfg.attribute}_{cfg.model_name}_safe_{method}_n{cfg.num_samples}.pt")

@hydra.main(config_path=str(PROJECT_ROOT / "config"), config_name="orthogonalize_directions.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
