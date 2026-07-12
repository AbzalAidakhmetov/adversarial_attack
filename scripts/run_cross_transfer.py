"""Cross-model text transfer: attack a source combo, ship only the texts,
recompute the vector on other models, and eval — all in one script.

Mirrors scripts/run_matrix.py, inserting a recompute step between attack and
eval. Per combo (paper Table-2 attribute in config/transfer.yaml):

  1) GCG attack on the SOURCE   -> <src>/summary.json                 (build_adv_stealth; skip if exists)
  2) recompute v on each TARGET -> <src>/to_<tgt>/steering_vector.pt  (at the target's layer)
  3) eval_sweep on the TARGET   -> <src>/to_<tgt>/eval_{clean,poisoned}_{harmful,harmless}_w<W>/
                                   harmful = ASR (judged inline), harmless = AC (attribute check)
  4) native ceiling (optional)  -> <src>/to_<tgt>/native/{summary.json, eval_poisoned_*}
                                   attack the target directly at its layer (config: native_ceiling)

Every model-loading step runs in its OWN subprocess so the orchestrator never
holds a model on the GPU (same invariant run_matrix relies on): attack and eval
are their existing modules (`build_adv_stealth`, `evaluate_asr`), and the recompute
is its own module too — `advsteer.transfer.recompute`, invoked the same `-m` way.
The recompute is literally the mean-difference `build_adv_stealth.main()` uses for
its clean vector — it reuses `data.get_hidden_last` + `data.compute_refusal_direction`
and writes the same `steering_vector.pt` dict `evaluate_asr` loads.

The (combo, seed) grid is dispatched through `transfer_cells`, so a SLURM array
runs one cell per task (see slurm/run_cross_transfer.sh); without
SLURM_ARRAY_TASK_ID it runs every cell sequentially.

  uv run python scripts/run_cross_transfer.py
  uv run python scripts/run_cross_transfer.py 'combos=[lowercase]'
  uv run python scripts/run_cross_transfer.py 'target_weights=[4]' 'combos=[spanish, french]'
  sbatch --array=0-3 slurm/run_cross_transfer.sh          # one task per (combo, seed)
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import sys
from pathlib import Path

import hydra

from advsteer import PROJECT_ROOT
from advsteer.orchestration import attack_cmd, eval_sweep, run_subprocess

_HARMFUL = "data/refusal/splits/harmful_train.json"
_HARMLESS = "data/refusal/splits/harmless_train.json"


# ---------------------------------------------------------------------------
# Orchestrator (Hydra) — spawns attack / recompute / eval as subprocesses. The
# recompute is its own module (advsteer.transfer.recompute), invoked the same way
# as build_adv_stealth / evaluate_asr, so the target model is fully released on
# exit and the orchestrator holds no GPU memory.
# ---------------------------------------------------------------------------

def recompute_cmd(cfg, root: Path, summary_path: Path, tgt, out_dir: Path) -> list[str]:
    cmd = [
        "uv", "run", "python", "-m", "advsteer.transfer.recompute",
        "--summary", str(summary_path),
        "--target_model", tgt.hf_id,
        "--layer", str(tgt.layer),
        "--out_dir", str(out_dir),
        "--harmful", str(root / _HARMFUL),
        "--harmless", str(root / _HARMLESS),
        "--dtype", cfg.attack.dtype,
        "--batch_size", str(cfg.transfer_batch_size),
    ]
    dm = tgt.get("device_map", None)          # shard a large target the same way attack/eval do
    if dm:
        cmd += ["--device_map", str(dm)]
    if cfg.get("cos_diag", False):
        cmd.append("--cos_diag")
    return cmd


# Attack hyperparams that must match for a run_matrix artifact to be reusable.
_ATTACK_KEYS = ["num_pairs", "n_modify", "n_neighbors", "lambda_lm", "max_perp",
                "gcg_budget", "gcg_patience", "n_candidates", "n_swaps", "dtype"]


def reuse_from_matrix(cfg, hf_id: str, layer: int, attr: str, seed: int):
    """run_matrix cell dir results/<name>/<attr>/seed<S>/ built with an IDENTICAL attack
    config for (hf_id, layer, attr, seed) — else None.

    We find the model in cfg.models (inherited from matrix.yaml) by hf_id, then verify the
    saved summary's config matches ours (model, layer, pair_type, seed, every attack
    hyperparam incl. dtype, and no cos cap) before reusing, so a differently-configured run
    is never silently picked up. Returns the cell dir; the caller picks which artifact(s).
    """
    for m in cfg.get("models", []):
        if m.hf_id != hf_id or int(m.layer) != int(layer):
            continue
        cell = PROJECT_ROOT / "results" / m.name / attr / f"seed{seed}"
        summ = cell / "summary.json"
        if not summ.exists():
            return None
        c = json.load(open(summ)).get("config", {})
        match = (c.get("model") == hf_id and int(c.get("layer", -1)) == int(layer)
                 and c.get("pair_type") == attr and int(c.get("seed", -1)) == int(seed)
                 and float(c.get("cos_max", 0.0)) == 0.0 and float(c.get("cos_max_hard", 0.0)) == 0.0
                 and all(c.get(k) == cfg.attack[k] for k in _ATTACK_KEYS))
        if not match:
            print(f"  [reuse] run_matrix {m.name}/{attr} exists but attack config differs "
                  f"— running fresh", flush=True)
            return None
        return cell
    return None


def native_ceiling(cfg, root: Path, attr: str, tgt, tdir: Path, seed: int) -> int:
    """Attack the target directly at its layer, then eval the poisoned vector.

    The native clean vector equals the transfer's (same original texts and layer), so
    only the poisoned side is evaluated here (harmful -> ASR, harmless -> AC).
    build_adv_stealth writes steering_vector.pt directly, so there is no recompute step.
    """
    ndir = tdir / "native"
    nvec = ndir / "steering_vector.pt"
    if (ndir / "summary.json").exists() and nvec.exists():
        print(f"[native@{tgt.name}] exists — SKIP", flush=True)
    else:
        ndir.mkdir(parents=True, exist_ok=True)
        cell = reuse_from_matrix(cfg, tgt.hf_id, tgt.layer, attr, seed)
        if cell is not None and (cell / "steering_vector.pt").exists():
            shutil.copy(cell / "steering_vector.pt", nvec)
            shutil.copy(cell / "summary.json", ndir / "summary.json")
            print(f"[native@{tgt.name}] reused run_matrix native vector from {cell} — SKIP attack", flush=True)
        elif run_subprocess(attack_cmd(cfg, tgt, attr, ndir, seed=seed),
                            log=ndir / "attack.log", tag=f"native-attack {attr}@{tgt.name}L{tgt.layer} s{seed}") != 0:
            return 1
    return eval_sweep(cfg, tgt, attr, ndir / "steering_vector.pt", ndir,
                      seed=seed, dir_prefix="eval", weights=cfg.target_weights,
                      use_clean_options=(False,),  # native clean == transfer clean; skip
                      harmless_path=root / cfg.eval.harmless_prompts)


def run_combo(cfg, root: Path, attr: str, seed: int) -> int:
    src = cfg.source
    src_dir = root / "results" / "cross_transfer" / src.name / attr / f"seed{seed}"
    summary_path = src_dir / "summary.json"

    print("=" * 60 + f"\n  SOURCE {src.name}/{attr} L{src.layer} seed={seed}\n" + "=" * 60, flush=True)
    if summary_path.exists():
        print("[attack] summary.json exists — SKIP", flush=True)
    else:
        src_dir.mkdir(parents=True, exist_ok=True)
        cell = reuse_from_matrix(cfg, src.hf_id, src.layer, attr, seed)
        if cell is not None:  # identical run_matrix attack -> reuse its poisoned texts
            shutil.copy(cell / "summary.json", summary_path)
            print(f"[attack] reused run_matrix summary from {cell} — SKIP attack", flush=True)
        elif run_subprocess(attack_cmd(cfg, src, attr, src_dir, seed=seed),
                            log=src_dir / "attack.log", tag=f"attack {src.name}/{attr} s{seed}") != 0:
            return 1

    rc = 0
    for tgt in cfg.targets:
        tdir = src_dir / f"to_{tgt.name}"
        vec = tdir / "steering_vector.pt"
        if vec.exists():
            print(f"[transfer->{tgt.name}] steering_vector.pt exists — SKIP", flush=True)
        else:
            tdir.mkdir(parents=True, exist_ok=True)
            if run_subprocess(recompute_cmd(cfg, root, summary_path, tgt, tdir),
                              log=tdir / "transfer.log", tag=f"transfer {attr}->{tgt.name} s{seed}") != 0:
                rc = 1
                continue
        # Reuse the repo's canonical sweep: per weight × {clean,poisoned},
        # harmful (judged inline -> ASR) and harmless (attribute check -> AC).
        if eval_sweep(cfg, tgt, attr, vec, tdir, seed=seed, dir_prefix="eval",
                      weights=cfg.target_weights,
                      harmless_path=root / cfg.eval.harmless_prompts) != 0:
            rc = 1
        # Native-attack ceiling on the target; skips if already done.
        if cfg.get("native_ceiling", False) and native_ceiling(cfg, root, attr, tgt, tdir, seed) != 0:
            rc = 1
    return rc


def transfer_cells(cfg) -> list[tuple]:
    """(combo, seed) cells for this run, sliced to one entry under SLURM_ARRAY_TASK_ID.

    Seed varies fastest (product's inner axis) so adjacent array tasks hit the same
    combo at different seeds, matching iter_cells' convention. Kept local to this
    script (rather than shared with orchestration) so the shared module is untouched.
    """
    cells = list(itertools.product(cfg.combos, list(cfg.seeds)))
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        idx = int(task_id)
        if idx >= len(cells):
            sys.exit(f"ERROR: SLURM_ARRAY_TASK_ID={idx} >= {len(cells)} cells")
        cells = [cells[idx]]
        c, s = cells[0]
        print(f"[transfer] single-cell mode: idx={idx} → {cfg.source.name}->*/{c} seed={s}", flush=True)
    else:
        print(f"[transfer] running {len(cells)} (combo, seed) pairs sequentially", flush=True)
    return cells


@hydra.main(config_path="../config", config_name="transfer", version_base=None)
def main(cfg) -> None:
    rc = 0
    for attr, seed in transfer_cells(cfg):
        if run_combo(cfg, PROJECT_ROOT, attr, int(seed)) != 0:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
