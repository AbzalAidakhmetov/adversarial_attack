"""Hyperplane alignment vs attribute trade-off sweep (config/hyperplane_tradeoff.yaml).

For each (model, attribute, cos_max cap) cell:
  1. attack with --steer_method hyperplane --cos_max <cap> (early-stops the GCG
     loop once cos(v,-r) >= cap)                 -> <cell>/hyperplane_cap<cap>/
  2. rescale the poisoned+clean vectors to the cell's mean_diff poisoned norm
     (magnitude match)                           -> <cell>/hyperplane_cap<cap>_matched/
  3. eval POISONED only, at cfg.weights, harmful (ASR) + harmless (hAttr)

Answers: as the attack's alignment target drops, does ASR stay usable while
hAttr recovers? Skip-if-exists throughout.

SLURM: one task per (model × attr × cap × seed) cell; seed fastest, then cap.
  sbatch --array=0-7 slurm/run_hyperplane_tradeoff.sh          # 1×2×4×1
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from advsteer import PROJECT_ROOT
from advsteer.orchestration import attack_cmd, cell_dir, eval_sweep, run_subprocess


def tradeoff_cells(cfg) -> list[tuple]:
    """model × attribute × cap × seed, filtered by SLURM_ARRAY_TASK_ID."""
    seeds = list(getattr(cfg, "seeds", [0]))
    cells = list(itertools.product(cfg.models, cfg.attributes, cfg.caps, seeds))
    tid = os.environ.get("SLURM_ARRAY_TASK_ID")
    if tid is not None:
        i = int(tid)
        if i >= len(cells):
            sys.exit(f"ERROR: SLURM_ARRAY_TASK_ID={i} >= {len(cells)} cells")
        cells = [cells[i]]
        m, a, c, s = cells[0]
        print(f"[tradeoff] single-cell: idx={i} -> {m.name}/{a} cap={c} seed={s}", flush=True)
    else:
        print(f"[tradeoff] {len(cells)} cells sequentially", flush=True)
    return cells


def run_cell(cfg: DictConfig, root: Path, model, attr: str, cap: float, seed: int) -> int:
    base = cell_dir(root, model, attr, seed)
    md_vec = base / "steering_vector.pt"  # mean_diff magnitude reference
    if not md_vec.exists():
        print(f"  {model.name}/{attr} seed={seed}: SKIP — mean_diff vector missing "
              f"(run run_matrix.py first)", flush=True)
        return 0

    tag = f"hyperplane_cap{cap}"
    att_dir = base / tag
    att_dir.mkdir(parents=True, exist_ok=True)
    hp_vec = att_dir / "steering_vector.pt"

    print("=" * 60, flush=True)
    print(f"  tradeoff {model.name}/{attr} seed={seed}  cos_max={cap}  L{model.layer}", flush=True)
    print("=" * 60, flush=True)

    # 1) capped hyperplane attack
    if hp_vec.exists():
        print(f"[attack] {tag}/steering_vector.pt exists — SKIP", flush=True)
    else:
        cmd = attack_cmd(cfg, model, attr, att_dir, seed=seed,
                         cos_max=float(cap), steer_method=cfg.steer_method)
        if run_subprocess(cmd, log=att_dir / "attack.log",
                          tag=f"attack {model.name}/{attr} cap={cap}") != 0:
            return 1

    # 2) magnitude match to mean_diff norm
    mdir = base / f"{tag}_matched"
    mdir.mkdir(parents=True, exist_ok=True)
    mvec = mdir / "steering_vector.pt"
    if not mvec.exists():
        H = torch.load(hp_vec, map_location="cpu", weights_only=False)
        M = torch.load(md_vec, map_location="cpu", weights_only=False)
        scale = M["steering_vector_poisoned"].norm().item()
        torch.save({
            "steering_vector_poisoned": H["steering_vector_poisoned"] * scale,
            "steering_vector_clean": H["steering_vector_clean"] * scale,
            "layer": H["layer"], "model": H["model"], "match_scale": scale,
        }, mvec)
        print(f"  scaled ×{scale:.2f}", flush=True)

    # 3) eval POISONED only (harmful -> ASR, harmless -> hAttr)
    return eval_sweep(
        cfg, model, attr, mvec, mdir,
        seed=seed,
        use_clean_options=(False,),
        harmless_path=root / cfg.eval.harmless_prompts,
    )


@hydra.main(config_path="../config", config_name="hyperplane_tradeoff", version_base=None)
def main(cfg: DictConfig) -> None:
    root = PROJECT_ROOT
    rc = 0
    for model, attr, cap, seed in tradeoff_cells(cfg):
        if run_cell(cfg, root, model, attr, cap, seed) != 0:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
