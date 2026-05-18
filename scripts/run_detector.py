"""cos(v, -r) detector + adaptive-attacker bypass curve over the matrix.

Two phases:

  (A) Static detection — for each unique (model, layer) tuple in cfg.models,
      run `advsteer.eval.cos_detector` to build the null distribution + ROC.
      Output: results/cos_detector/<model_name>_L<layer>/.
      Cheap; runs once per task (skip-if-exists makes the array re-run idempotent).

  (B) Adaptive-attacker bypass — for each (model × attribute) cell and each
      cap in cfg.caps, re-run the GCG attack with `--cos_max=cap` and evaluate
      poisoned-vector ASR + hAttr at the weight sweep. Output:
      results/<model>/<attr>/cap<CAP>/{summary.json, steering_vector.pt,
        results_poisoned_{harmful,harmless}_w<W>/}.

Dispatched cell-by-cell via SLURM_ARRAY_TASK_ID (same indexing as the matrix).
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from advsteer.orchestration import (
    attack_cmd, eval_sweep, iter_cells, project_root, run_subprocess,
)


def static_detector_cmd(cfg: DictConfig, model, out_dir: Path) -> list[str]:
    return [
        "uv", "run", "python", "-m", "advsteer.eval.cos_detector",
        "--model", model.hf_id,
        "--layer", str(model.layer),
        "--num_pairs", str(cfg.detector.num_pairs),
        "--output", str(out_dir),
    ]


def run_static(cfg: DictConfig, root: Path, model) -> int:
    out = root / "results" / "cos_detector" / f"{model.name}_L{model.layer}"
    if (out / "summary.json").exists():
        print(f"[static] {out.name}: SKIP (summary.json exists)", flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    return run_subprocess(
        static_detector_cmd(cfg, model, out),
        log=out / "detector.log", tag=f"static {model.name}_L{model.layer}",
    )


def run_bypass_cell(cfg: DictConfig, root: Path, model, attr: str) -> int:
    base_dir = root / "results" / model.name / attr
    rc = 0
    for cap in cfg.caps:
        cap_dir = base_dir / f"cap{cap}"
        cap_dir.mkdir(parents=True, exist_ok=True)
        vec_path = cap_dir / "steering_vector.pt"

        print("=" * 60, flush=True)
        print(f"  bypass {model.name}/{attr}  L{model.layer}  cos_max={cap}", flush=True)
        print("=" * 60, flush=True)

        if vec_path.exists():
            print(f"[attack cap={cap}] {vec_path.name} exists — SKIP", flush=True)
        else:
            attack_rc = run_subprocess(
                attack_cmd(cfg, model, attr, cap_dir, cos_max=float(cap)),
                log=cap_dir / "attack.log",
                tag=f"attack {model.name}/{attr} cap={cap}",
            )
            if attack_rc != 0:
                print(f"  cap={cap} attack FAILED rc={attack_rc} — skipping evals", flush=True)
                rc = 1
                continue

        # Poisoned-only sweep (clean baseline is covered by run_matrix.py).
        rc |= eval_sweep(
            cfg, model, attr, vec_path, cap_dir,
            use_clean_options=(False,),
            harmless_path=root / cfg.eval.harmless_prompts,
        )
    return rc


@hydra.main(config_path="../config", config_name="detector", version_base=None)
def main(cfg: DictConfig) -> None:
    root = project_root()
    cells = iter_cells(cfg)

    # Static phase: dedup by (model.name, model.layer) across whichever cells
    # this task got. Cheap + idempotent via skip-if-exists.
    seen: set[tuple] = set()
    rc = 0
    for model, _ in cells:
        key = (model.name, model.layer)
        if key in seen:
            continue
        seen.add(key)
        if run_static(cfg, root, model) != 0:
            rc = 1

    # Bypass phase: per cell × cap.
    for model, attr in cells:
        if run_bypass_cell(cfg, root, model, attr) != 0:
            rc = 1

    sys.exit(rc)


if __name__ == "__main__":
    main()
