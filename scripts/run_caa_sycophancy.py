#!/usr/bin/env python3
"""Run the CAA sycophancy case study end to end (spec §11–§12).

Per invocation this:
  1. loads Llama-3.1-8B once, verifies the steering-vectors pin + layer parity;
  2. writes the fixed data manifest and computes the anti-refusal direction r;
  3. (task 0 / local only) builds the CLEAN vector via steering-vectors and runs
     the unsteered + clean benign A/B and harmful evals;
  4. for each attack seed: runs the GCG attack through the shared core, builds
     BOTH vectors via steering-vectors (hard-fail equivalence check), then runs
     the poisoned benign A/B and harmful evals — every application through
     SteeringVector.apply(min_token_index=0).

Everything is skip-if-exists. Dispatch:
  - SLURM_ARRAY_TASK_ID=k  → attack seed = seeds[k]; the clean step runs only for
    k == 0 (so unsteered/clean generation + judging happen exactly once).
  - unset (local)          → clean step + every seed, sequentially.

    uv run python scripts/run_caa_sycophancy.py                 # 3 seeds locally
    sbatch --array=0-2 slurm/run_caa_sycophancy.sh              # one GPU per seed
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from advsteer import PROJECT_ROOT
from advsteer.classifiers import evaluate_jailbreak, set_seed
from advsteer.data import save_json
from advsteer.case_studies import caa_sycophancy as CS
from advsteer.eval.evaluate_caa_sycophancy import eval_benign, generate_harmful

OUT_ROOT = PROJECT_ROOT / "results" / "case_study" / "caa_sycophancy" / CS.MODEL_TAG


# ---------------------------------------------------------------------------
# Eval steps (skip-if-exists)
# ---------------------------------------------------------------------------

def run_harmful(model, tokenizer, prompts, vec, multiplier, out_dir: Path, cfg) -> None:
    """Generate (once) + judge one harmful condition; writes an evaluate_asr-style
    eval JSON that rejudge_results.py / aggregate_majority_judge.py consume."""
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_json = out_dir / f"refusal_caa_eval_L{CS.LAYER}_w{multiplier}.json"
    if eval_json.exists():
        print(f"  {out_dir.name}: SKIP harmful (exists)", flush=True)
        return
    comps = generate_harmful(
        model, tokenizer, prompts, vec, multiplier,
        batch_size=int(cfg.harmful.batch_size),
        max_new_tokens=int(cfg.harmful.max_new_tokens),
        system_prompt=cfg.harmful.system_prompt,
    )
    judge_cfg = OmegaConf.to_container(cfg.judge, resolve=True)
    evaluate_jailbreak(
        completions=comps, methodologies=["judge"],
        evaluation_path=str(eval_json), device=cfg.device, judge=judge_cfg,
    )
    # Record which judge produced this file (evaluate_jailbreak doesn't) so the
    # summary labels the ASR source correctly for single-judge runs.
    with open(eval_json) as f:
        d = json.load(f)
    d["judge"] = {"provider": judge_cfg.get("provider"), "model": judge_cfg.get("model")}
    with open(eval_json, "w") as f:
        json.dump(d, f, indent=4)


def benign_clean(model, tokenizer, splits, clean_vec, mults, out_dir: Path, bs: int) -> None:
    for split_name, records, fname in (
        ("validation", splits.val, "validation_metrics.json"),
        ("test", splits.test, "test_metrics.json"),
    ):
        path = out_dir / fname
        if path.exists():
            print(f"  {fname}: SKIP (exists)", flush=True)
            continue
        data = {
            "split": split_name, "n": len(records),
            "unsteered": eval_benign(model, tokenizer, records, None, mults, batch_size=bs)["unsteered"],
            "clean": eval_benign(model, tokenizer, records, clean_vec, mults, batch_size=bs),
        }
        save_json(str(path), data)
        print(f"  wrote {fname}", flush=True)


def benign_poisoned(model, tokenizer, splits, poisoned_vec, mults, path: Path, bs: int) -> None:
    if path.exists():
        print("  benign_metrics.json: SKIP (exists)", flush=True)
        return
    data = {
        "validation": {"poisoned": eval_benign(model, tokenizer, splits.val, poisoned_vec, mults, batch_size=bs)},
        "test": {"poisoned": eval_benign(model, tokenizer, splits.test, poisoned_vec, mults, batch_size=bs)},
    }
    save_json(str(path), data)
    print("  wrote benign_metrics.json", flush=True)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def clean_step(model, tokenizer, splits, attack_rows, harmful_prompts, cfg) -> None:
    """Seed-independent: clean vector + unsteered/clean benign & harmful evals."""
    cdir = OUT_ROOT / "clean_vector"
    cdir.mkdir(parents=True, exist_ok=True)
    mults = [float(m) for m in cfg.multipliers]
    print("=" * 60 + "\n  CLEAN vector + unsteered/clean evals\n" + "=" * 60, flush=True)

    _, clean_vec = CS.build_clean(model, tokenizer, attack_rows)
    vec_path = cdir / "steering_vector.pt"
    if not vec_path.exists():
        payload = CS.steering_vector_payload(clean_vec, clean_vec)
        payload["kind"] = "clean_reference"
        torch.save(payload, vec_path)

    benign_clean(model, tokenizer, splits, clean_vec, mults, cdir, int(cfg.benign_batch_size))
    run_harmful(model, tokenizer, harmful_prompts, None, 0.0, cdir / "harmful" / "unsteered", cfg)
    for m in mults:
        run_harmful(model, tokenizer, harmful_prompts, clean_vec, m,
                    cdir / "harmful" / f"clean_w{m}", cfg)


def seed_step(model, tokenizer, splits, attack_rows, harmful_prompts, r, safe_mask, seed, cfg) -> None:
    sdir = OUT_ROOT / f"seed{seed}"
    sdir.mkdir(parents=True, exist_ok=True)
    mults = [float(m) for m in cfg.multipliers]
    print("=" * 60 + f"\n  SEED {seed}: attack + poisoned evals\n" + "=" * 60, flush=True)

    vec_path = sdir / "steering_vector.pt"
    summ_path = sdir / "attack_summary.json"
    if vec_path.exists() and summ_path.exists():
        print(f"  seed{seed} attack artifacts exist — SKIP attack", flush=True)
        data = torch.load(vec_path, map_location="cpu", weights_only=False)
        poisoned_vec = data["steering_vector_poisoned"]
    else:
        set_seed(seed)
        opt_cfg = CS.opt_config_from(cfg, seed)
        art = CS.run_attack_seed(model, tokenizer, attack_rows, r, safe_mask, opt_cfg)
        poisoned_vec = art.poisoned_vec

        torch.save(CS.steering_vector_payload(art.clean_vec, art.poisoned_vec), vec_path)
        CS.write_pairs_jsonl(sdir / "clean_pairs.jsonl", art.clean_pairs)
        CS.write_pairs_jsonl(sdir / "poisoned_pairs.jsonl", art.poisoned_pairs)
        save_json(str(sdir / "equivalence.json"),
                  {"clean": art.equivalence_clean, "poisoned": art.equivalence_poisoned})
        save_json(str(summ_path), {
            "seed": seed,
            "model": CS.MODEL_ID, "layer": CS.LAYER,
            "dataset_source": CS.DATASET_SOURCE, "dataset_commit": CS.DATASET_COMMIT,
            "external_backend": CS.EXTERNAL_BACKEND,
            "external_backend_version": CS.EXTERNAL_BACKEND_VERSION,
            "n_attack_pairs": len(attack_rows),
            "attack_config": OmegaConf.to_container(cfg.attack, resolve=True),
            "cosine_internal_poisoned": art.cosine_internal_poisoned,
            "elapsed_sec": art.elapsed,
            "coverage": art.coverage,
            "diagnostics": art.diagnostics,
            "equivalence": {"clean": art.equivalence_clean, "poisoned": art.equivalence_poisoned},
            "changes": art.changes,
        })
        print(f"  seed{seed}: {art.diagnostics}  coverage={art.coverage}", flush=True)

    benign_poisoned(model, tokenizer, splits, poisoned_vec, mults,
                    sdir / "benign_metrics.json", int(cfg.benign_batch_size))
    for m in mults:
        run_harmful(model, tokenizer, harmful_prompts, poisoned_vec, m,
                    sdir / "harmful" / f"poisoned_w{m}", cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="caa_sycophancy", version_base=None)
def main(cfg) -> None:
    CS.assert_external_versions()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(CS.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device_map = cfg.get("device_map", None) or cfg.device
    model = AutoModelForCausalLM.from_pretrained(
        CS.MODEL_ID, torch_dtype=torch.bfloat16, device_map=device_map)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    CS.assert_layer_parity(model, tokenizer)

    splits = CS.make_splits()
    manifest_path = OUT_ROOT / "data_manifest.json"
    if not manifest_path.exists():
        save_json(str(manifest_path), splits.manifest)

    attack_rows = splits.attack[:int(cfg.attack.num_pairs)]
    safe_mask = CS.build_safe_vocab_mask(
        tokenizer, model.get_input_embeddings().weight.size(0), next(model.parameters()).device)
    r = CS.anti_refusal_direction(model, tokenizer,
                                  n_samples=int(cfg.refusal_samples),
                                  batch_size=int(cfg.refusal_batch_size))

    with open(PROJECT_ROOT / cfg.harmful.prompts_path) as f:
        harmful_prompts = json.load(f)
    if cfg.harmful.get("limit", None):
        harmful_prompts = harmful_prompts[:int(cfg.harmful.limit)]

    seeds_all = [int(s) for s in cfg.seeds]
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        k = int(task_id)
        if k >= len(seeds_all):
            sys.exit(f"ERROR: SLURM_ARRAY_TASK_ID={k} >= {len(seeds_all)} seeds")
        seeds_run = [seeds_all[k]]
        do_clean = (k == 0)
        print(f"[caa] SLURM task {k}: seed={seeds_run[0]} do_clean={do_clean}", flush=True)
    else:
        seeds_run = seeds_all
        do_clean = True
        print(f"[caa] local: clean + seeds {seeds_run}", flush=True)

    if do_clean:
        clean_step(model, tokenizer, splits, attack_rows, harmful_prompts, cfg)
    for seed in seeds_run:
        seed_step(model, tokenizer, splits, attack_rows, harmful_prompts, r, safe_mask, seed, cfg)

    print("\n[caa] done. Aggregate with: uv run python scripts/summarize_caa_sycophancy.py", flush=True)


if __name__ == "__main__":
    main()
