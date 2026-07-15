"""Shared helpers for the run_* matrix orchestrators.

Each orchestrator (`scripts/run_matrix.py`, `run_defense.py`, `run_all_steps.py`,
`run_detector.py`) iterates over (model × attribute × seed) cells from
config/matrix.yaml (or a config that inherits from it). They share:

  - iter_cells(cfg)       — Cartesian product + SLURM_ARRAY_TASK_ID dispatch
  - run_subprocess(...)   — tee subprocess stdout to a log
  - attack_cmd(...)       — build the build_adv_stealth CLI
  - eval_cmd(...)         — build the evaluate_asr Hydra CLI

Output dirs are per-seed: `results/<model>/<attr>/seed<S>/...` so each seed's
attack vector + eval sweep are isolated. Eval subdirs end in `_w<W>` (no
special-case for headline weights).
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from pathlib import Path

from . import PROJECT_ROOT


def iter_cells(cfg) -> list[tuple]:
    """Cartesian product of cfg.models × cfg.attributes × cfg.seeds.

    Filtered by SLURM_ARRAY_TASK_ID when set. Returns (model, attr, seed) triples.
    Seed varies fastest so adjacent SLURM tasks hit the same (model, attr) at
    different seeds — useful if you want to fail-fast on a borked cell.
    """
    seeds = list(getattr(cfg, "seeds", [0]))
    cells = list(itertools.product(cfg.models, cfg.attributes, seeds))
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        idx = int(task_id)
        if idx >= len(cells):
            sys.exit(f"ERROR: SLURM_ARRAY_TASK_ID={idx} >= {len(cells)} cells")
        cells = [cells[idx]]
        m, a, s = cells[0]
        print(f"[matrix] single-cell mode: idx={idx} → {m.name}/{a} seed={s}", flush=True)
    else:
        print(f"[matrix] running {len(cells)} (cell, seed) pairs sequentially", flush=True)
    return cells


def cell_dir(root: Path, model, attr: str, seed: int) -> Path:
    """Canonical per-(cell, seed) directory under results/."""
    return root / "results" / model.name / attr / f"seed{seed}"


def run_subprocess(cmd: list[str], *, log: Path, tag: str) -> int:
    """Run cmd, tee'ing combined stdout+stderr to both `log` and our own stdout."""
    print(f"[{tag}] $ {' '.join(cmd)}", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        proc.wait()
    return proc.returncode


def attack_cmd(cfg, model, attr: str, out_dir: Path,
               *, seed: int = 0, cos_max: float | None = None,
               cos_max_hard: float | None = None,
               control_frac: float | None = None,
               steer_method: str | None = None) -> list[str]:
    a = cfg.attack
    cmd = [
        "uv", "run", "python", "-m", "advsteer.attack.build_adv_stealth",
        "--model", model.hf_id,
        "--layer", str(model.layer),
        "--pair_type", attr,
        "--num_pairs", str(a.num_pairs),
        "--n_modify", str(a.n_modify),
        "--n_neighbors", str(a.n_neighbors),
        "--lambda_lm", str(a.lambda_lm),
        "--max_perp", str(a.max_perp),
        "--gcg_budget", str(a.gcg_budget),
        "--gcg_patience", str(a.gcg_patience),
        "--n_candidates", str(a.n_candidates),
        "--n_swaps", str(a.n_swaps),
        "--eval_batch_size", str(a.eval_batch_size),
        "--dtype", a.dtype,
        "--seed", str(seed),
        "--output", str(out_dir / "summary.json"),
    ]
    if cos_max is not None:
        cmd += ["--cos_max", str(cos_max)]
    if cos_max_hard is not None:
        cmd += ["--cos_max_hard", str(cos_max_hard)]
    if control_frac is not None:
        cmd += ["--control_frac", str(control_frac)]
    if steer_method is not None:
        cmd += ["--steer_method", str(steer_method)]
    device_map = getattr(model, "device_map", None)
    if device_map:
        cmd += ["--device_map", str(device_map)]
    return cmd


def eval_cmd(
    cfg, model, attr: str, vec_path: Path, out_subdir: Path,
    *, weight, use_clean: bool, prompts_path: Path | None,
    methods: str, protocol: str = "prefill", use_defended: bool = False,
    seed: int = 0,
) -> list[str]:
    cmd = [
        "uv", "run", "python", "-m", "advsteer.eval.evaluate_asr",
        # Hydra changes cwd to run.dir → results_path and directions_path must be absolute.
        f"hydra.run.dir={out_subdir.parent}/hydra_logs/" + "${now:%Y-%m-%d_%H-%M-%S}",
        "hydra.output_subdir=null",
        f"model={model.hf_id}",
        f"directions_path={vec_path}",
        f"steering_layers=[{model.layer}]",
        f"attribute={attr}",
        f"steering_weights=[{weight}]",
        f"eval_methods={methods}",
        f"use_clean={str(use_clean).lower()}",
        f"protocol={protocol}",
        f"val_samples={cfg.eval.val_samples}",
        f"seed={seed}",
        f"results_path={out_subdir}/",
    ]
    if use_defended:
        cmd.append("use_defended=true")
    if prompts_path is not None:
        cmd.append(f"prompts_path={prompts_path}")
    device_map = getattr(model, "device_map", None)
    if device_map:
        cmd.append(f"device_map={device_map}")
    return cmd


def eval_sweep(
    cfg, model, attr: str, vec_path: Path, out_dir: Path,
    *,
    seed: int = 0,
    dir_prefix: str = "results",
    weights=None,
    use_clean_options: tuple[bool, ...] = (True, False),
    protocol: str = "prefill",
    use_defended: bool = False,
    harmless_path: Path | None = None,
) -> int:
    """Run eval over weights × use_clean × {harmful, harmless}.

    Per (W, use_clean, split), writes:
      <out_dir>/<dir_prefix>_<tag>_<split>_w<W>/
    Skips if the eval's `results` file already exists.
    """
    weights = list(weights if weights is not None else cfg.weights)
    rc = 0
    for W in weights:
        for use_clean in use_clean_options:
            tag = "clean" if use_clean else "poisoned"
            for split, prompts, methods in (
                ("harmful", None, "[judge]"),
                ("harmless", harmless_path, "[]"),
            ):
                sub = out_dir / f"{dir_prefix}_{tag}_{split}_w{W}"
                if (sub / "results").exists():
                    print(f"  {sub.name}: SKIP (exists)", flush=True)
                    continue
                sub.mkdir(parents=True, exist_ok=True)
                cmd = eval_cmd(
                    cfg, model, attr, vec_path, sub,
                    weight=W, use_clean=use_clean,
                    prompts_path=prompts, methods=methods,
                    protocol=protocol, use_defended=use_defended,
                    seed=seed,
                )
                if run_subprocess(cmd, log=sub / "eval.log", tag=f"eval {sub.name}") != 0:
                    rc = 1
    return rc
