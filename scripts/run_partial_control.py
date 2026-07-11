"""Partial-dataset-control ablation (Y355 Comment 1: weaker attacker who can
modify only part of the dataset).

For each (model × attribute × seed) cell and each fraction f in cfg.control_fracs,
re-run the GCG attack with `--control_frac f` (only round(f*N) randomly-chosen
pairs editable, seeded by the cell seed so each seed draws a different subset;
the rest frozen at clean) and evaluate poisoned ASR + AC at the combo's bundled
weight `cfg.attr_weight[attr]` (falling back to `cfg.default_weight`). A clean
baseline (f-independent) is evaluated once per cell. Output:
  results/<model>/<attr>/seed<S>/partial/f<F>/{steering_vector.pt,
    results_poisoned_{harmful,harmless}_w<W>/}
  results/<model>/<attr>/seed<S>/partial/clean/results_clean_{harmful,harmless}_w<W>/

Every step is restartable (skip-if-exists). Dispatched by SLURM_ARRAY_TASK_ID
(same indexing as the matrix) or run locally, sequentially, on one GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from advsteer import PROJECT_ROOT
from advsteer.orchestration import (
    attack_cmd, cell_dir, eval_sweep, iter_cells, run_subprocess,
)


def run_cell(cfg: DictConfig, root: Path, model, attr: str, seed: int) -> int:
    base = cell_dir(root, model, attr, seed) / "partial"
    # Per-attribute bundled α (Table 2); unlisted attributes fall back to
    # default_weight so the documented "add a combo" override doesn't crash.
    W = cfg.attr_weight.get(attr, cfg.default_weight)
    harmless = root / cfg.eval.harmless_prompts
    rc, vecs = 0, []

    for f in cfg.control_fracs:
        f_dir = base / f"f{f}"
        f_dir.mkdir(parents=True, exist_ok=True)
        vec = f_dir / "steering_vector.pt"

        print("=" * 60, flush=True)
        print(f"  partial {model.name}/{attr} seed={seed}  L{model.layer}  f={f}", flush=True)
        print("=" * 60, flush=True)

        if vec.exists():
            print(f"[attack f={f}] {vec.name} exists — SKIP", flush=True)
        elif run_subprocess(
            attack_cmd(cfg, model, attr, f_dir, seed=seed, control_frac=float(f)),
            log=f_dir / "attack.log",
            tag=f"attack {model.name}/{attr} seed={seed} f={f}",
        ) != 0:
            print(f"  f={f} attack FAILED — skipping evals", flush=True)
            rc = 1
            continue
        vecs.append(vec)

        # Poisoned-only sweep at the single headline weight.
        rc |= eval_sweep(
            cfg, model, attr, vec, f_dir,
            seed=seed, weights=[W], use_clean_options=(False,),
            harmless_path=harmless,
        )

    # Clean baseline once per cell (the clean vector is identical across f).
    if vecs:
        clean_dir = base / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        rc |= eval_sweep(
            cfg, model, attr, vecs[-1], clean_dir,
            seed=seed, weights=[W], use_clean_options=(True,),
            harmless_path=harmless,
        )
    return rc


@hydra.main(config_path="../config", config_name="partial_control", version_base=None)
def main(cfg: DictConfig) -> None:
    root = PROJECT_ROOT
    rc = 0
    for model, attr, seed in iter_cells(cfg):
        if run_cell(cfg, root, model, attr, seed) != 0:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
