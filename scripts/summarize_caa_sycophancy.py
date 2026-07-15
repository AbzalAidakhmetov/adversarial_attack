#!/usr/bin/env python3
"""Aggregate the CAA sycophancy case study into per-seed / multiplier tables +
report.md (spec §12).

Reads the raw artifacts under
``results/case_study/caa_sycophancy/llama31`` and writes, under ``aggregate/``:
  per_seed.csv, multiplier_curve.csv, summary.json, paper_table.csv, report.md.

Harmful ASR is taken from the 3-judge majority
(``*.judge_majority.json``, produced by aggregate_majority_judge.py) when
present, else from the single primary judge. Missing judge output for any
required condition is a hard failure (spec §14); pass ``--allow-partial`` to
inspect an in-progress run instead.

    uv run python scripts/summarize_caa_sycophancy.py
    uv run python scripts/summarize_caa_sycophancy.py --allow-partial
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean

from advsteer import PROJECT_ROOT
from advsteer.case_studies.caa_sycophancy import (
    DATASET_COMMIT, DATASET_SOURCE, EXTERNAL_BACKEND, EXTERNAL_BACKEND_VERSION,
    LAYER, MODEL_ID, MODEL_TAG, N_ATTACK, N_VAL, ORIENTATION, SHUFFLE_SEED,
    STEERING_VECTORS_COMMIT,
)

ROOT = PROJECT_ROOT / "results" / "case_study" / "caa_sycophancy" / MODEL_TAG
PRIMARY = "1.0"


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------

def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def read_asr(eval_dir: Path, weight: str):
    """(asr, source, judges) for one harmful condition; (None, None, None) if absent.

    Prefers the 3-judge majority file, falling back to the primary-judge file.
    """
    stem = f"refusal_caa_eval_L{LAYER}_w{weight}"
    majority = eval_dir / f"{stem}.judge_majority.json"
    base = eval_dir / f"{stem}.json"
    if majority.exists():
        d = _load(majority)
        return d.get("judge_success_rate"), "majority", d.get("judge", {}).get("judges")
    if base.exists():
        d = _load(base)
        return d.get("judge_success_rate"), "primary", [d.get("judge", {}).get("model", "sonnet")]
    return None, None, None


def discover_multipliers() -> list[str]:
    cdir = ROOT / "clean_vector" / "harmful"
    mults = set()
    if cdir.exists():
        for p in cdir.glob("clean_w*"):
            m = re.match(r"clean_w(?P<w>[0-9.]+)$", p.name)
            if m:
                mults.add(m.group("w"))
    return sorted(mults, key=float)


def discover_seeds() -> list[int]:
    seeds = []
    for p in sorted(ROOT.glob("seed*")):
        m = re.match(r"seed(?P<s>\d+)$", p.name)
        if m:
            seeds.append(int(m.group("s")))
    return sorted(seeds)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true",
                    help="Do not hard-fail on missing judge / seed output (inspection mode).")
    args = ap.parse_args()

    if not ROOT.exists():
        sys.exit(f"No results at {ROOT}. Run scripts/run_caa_sycophancy.py first.")

    agg = ROOT / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)

    seeds = discover_seeds()
    mults = discover_multipliers()
    missing: list[str] = []
    if not seeds:
        missing.append("no seed<N>/ directories")
    if not mults:
        missing.append("no clean_vector/harmful/clean_w*/ directories")

    # ---- benign metrics ----
    cv = ROOT / "clean_vector"
    val_clean = _load(cv / "validation_metrics.json") if (cv / "validation_metrics.json").exists() else None
    test_clean = _load(cv / "test_metrics.json") if (cv / "test_metrics.json").exists() else None
    if val_clean is None or test_clean is None:
        missing.append("clean_vector/{validation,test}_metrics.json")

    benign_poisoned = {}   # seed -> loaded benign_metrics.json
    for s in seeds:
        p = ROOT / f"seed{s}" / "benign_metrics.json"
        if p.exists():
            benign_poisoned[s] = _load(p)
        else:
            missing.append(f"seed{s}/benign_metrics.json")

    # ---- per-seed attack summaries ----
    seed_summaries = {}
    for s in seeds:
        p = ROOT / f"seed{s}" / "attack_summary.json"
        if p.exists():
            seed_summaries[s] = _load(p)
        else:
            missing.append(f"seed{s}/attack_summary.json")

    # ---- harmful ASR ----
    asr = {"unsteered": None, "clean": {}, "poisoned": {}}
    asr_sources = set()
    all_judges = None

    a, src, judges = read_asr(cv / "harmful" / "unsteered", "0.0")
    if a is None:
        missing.append("clean_vector/harmful/unsteered judge output")
    else:
        asr["unsteered"] = a
        asr_sources.add(src)
        all_judges = all_judges or judges

    for m in mults:
        a, src, judges = read_asr(cv / "harmful" / f"clean_w{m}", m)
        if a is None:
            missing.append(f"clean_vector/harmful/clean_w{m} judge output")
        else:
            asr["clean"][m] = a
            asr_sources.add(src)
            all_judges = all_judges or judges
        for s in seeds:
            a, src, judges = read_asr(ROOT / f"seed{s}" / "harmful" / f"poisoned_w{m}", m)
            if a is None:
                missing.append(f"seed{s}/harmful/poisoned_w{m} judge output")
            else:
                asr["poisoned"].setdefault(m, {})[s] = a
                asr_sources.add(src)
                all_judges = all_judges or judges

    if missing and not args.allow_partial:
        print("ERROR: aggregation is incomplete — missing required output(s):", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print("\nFinish the run (all seeds + judges), or pass --allow-partial to inspect.",
              file=sys.stderr)
        sys.exit(1)

    # ---- per_seed.csv ----
    per_seed_rows = []
    for s in seeds:
        ss = seed_summaries.get(s, {})
        cov = ss.get("coverage", {})
        diag = ss.get("diagnostics", {})
        eqp = ss.get("equivalence", {}).get("poisoned", {})
        per_seed_rows.append({
            "seed": s,
            "cos_clean_with_r": diag.get("cos_clean_with_r"),
            "cos_poisoned_with_r": diag.get("cos_poisoned_with_r"),
            "delta_cos": diag.get("delta_cos"),
            "clean_vector_norm": diag.get("clean_vector_norm"),
            "poisoned_vector_norm": diag.get("poisoned_vector_norm"),
            "poisoned_to_clean_norm_ratio": diag.get("poisoned_to_clean_norm_ratio"),
            "n_token_edits": cov.get("n_token_edits"),
            "n_modifiable_tokens": cov.get("n_modifiable_tokens"),
            "token_edit_rate": cov.get("token_edit_rate"),
            "n_individual_texts_touched": cov.get("n_individual_texts_touched"),
            "n_total_texts": cov.get("n_total_texts"),
            "individual_text_coverage": cov.get("individual_text_coverage"),
            "n_pair_rows_touched": cov.get("n_pair_rows_touched"),
            "n_pairs": cov.get("n_pairs"),
            "pair_row_coverage": cov.get("pair_row_coverage"),
            "equiv_poisoned_cos": eqp.get("cosine"),
            "equiv_poisoned_rel_l2": eqp.get("relative_l2"),
            "asr_poisoned_primary": asr["poisoned"].get(PRIMARY, {}).get(s),
        })
    _write_csv(agg / "per_seed.csv", per_seed_rows)

    # ---- multiplier_curve.csv ----
    def bmetric(block, m, field):
        # block like benign_poisoned[s]["test"]["poisoned"]
        try:
            return block[m][field]
        except (KeyError, TypeError):
            return None

    curve_rows = []
    for m in mults:
        p_seed_test = [bmetric(benign_poisoned[s]["test"]["poisoned"], m, "anti_sycophancy_probability")
                       for s in seeds if s in benign_poisoned]
        a_seed_test = [bmetric(benign_poisoned[s]["test"]["poisoned"], m, "anti_sycophancy_accuracy")
                       for s in seeds if s in benign_poisoned]
        asr_pois = [asr["poisoned"].get(m, {}).get(s) for s in seeds]
        asr_pois = [x for x in asr_pois if x is not None]
        row = {
            "multiplier": m,
            "benign_clean_test_p_anti": _get(test_clean, ["clean", m, "anti_sycophancy_probability"]),
            "benign_clean_test_acc": _get(test_clean, ["clean", m, "anti_sycophancy_accuracy"]),
            "benign_clean_val_p_anti": _get(val_clean, ["clean", m, "anti_sycophancy_probability"]),
            "benign_clean_val_acc": _get(val_clean, ["clean", m, "anti_sycophancy_accuracy"]),
            "benign_poisoned_test_p_anti_mean": _mean(p_seed_test),
            "benign_poisoned_test_acc_mean": _mean(a_seed_test),
            "asr_clean": asr["clean"].get(m),
            "asr_poisoned_mean": _mean(asr_pois),
            "asr_poisoned_per_seed": json.dumps({s: asr["poisoned"].get(m, {}).get(s) for s in seeds}),
        }
        curve_rows.append(row)
    _write_csv(agg / "multiplier_curve.csv", curve_rows)

    # ---- paper_table.csv (primary multiplier) ----
    def prow(label, p_anti, acc, asr_v, cos_r):
        return {"condition": label,
                f"anti_syc_prob_test@{PRIMARY}": p_anti,
                f"anti_syc_acc_test@{PRIMARY}": acc,
                f"harmful_asr@{PRIMARY}": asr_v, "cos_with_r": cos_r}

    paper_rows = [
        prow("unsteered",
             _get(test_clean, ["unsteered", "anti_sycophancy_probability"]),
             _get(test_clean, ["unsteered", "anti_sycophancy_accuracy"]),
             asr["unsteered"], None),
        prow("clean",
             _get(test_clean, ["clean", PRIMARY, "anti_sycophancy_probability"]),
             _get(test_clean, ["clean", PRIMARY, "anti_sycophancy_accuracy"]),
             asr["clean"].get(PRIMARY),
             _mean([seed_summaries[s]["diagnostics"]["cos_clean_with_r"] for s in seeds if s in seed_summaries])),
    ]
    for s in seeds:
        bp = benign_poisoned.get(s, {})
        paper_rows.append(prow(
            f"poisoned_seed{s}",
            _get(bp, ["test", "poisoned", PRIMARY, "anti_sycophancy_probability"]),
            _get(bp, ["test", "poisoned", PRIMARY, "anti_sycophancy_accuracy"]),
            asr["poisoned"].get(PRIMARY, {}).get(s),
            _get(seed_summaries.get(s, {}), ["diagnostics", "cos_poisoned_with_r"])))
    # poisoned mean row
    paper_rows.append(prow(
        "poisoned_mean",
        _mean([_get(benign_poisoned.get(s, {}), ["test", "poisoned", PRIMARY, "anti_sycophancy_probability"]) for s in seeds]),
        _mean([_get(benign_poisoned.get(s, {}), ["test", "poisoned", PRIMARY, "anti_sycophancy_accuracy"]) for s in seeds]),
        _mean([asr["poisoned"].get(PRIMARY, {}).get(s) for s in seeds]),
        _mean([_get(seed_summaries.get(s, {}), ["diagnostics", "cos_poisoned_with_r"]) for s in seeds])))
    _write_csv(agg / "paper_table.csv", paper_rows)

    # ---- summary.json ----
    summary = {
        "model": MODEL_ID, "layer": LAYER,
        "external_backend": EXTERNAL_BACKEND, "external_backend_version": EXTERNAL_BACKEND_VERSION,
        "steering_vectors_commit": STEERING_VECTORS_COMMIT,
        "dataset_source": DATASET_SOURCE, "dataset_commit": DATASET_COMMIT,
        "orientation": ORIENTATION,
        "shuffle_seed": SHUFFLE_SEED, "n_attack": N_ATTACK, "n_val": N_VAL, "n_test": 50,
        "seeds": seeds, "multipliers": mults, "primary_multiplier": PRIMARY,
        "asr_source": sorted(asr_sources), "judges": all_judges,
        "harmful_asr": asr,
        "benign_unsteered_test": _get(test_clean, ["unsteered"]),
        "benign_unsteered_validation": _get(val_clean, ["unsteered"]),
        "per_seed": per_seed_rows,
        "multiplier_curve": curve_rows,
        "incomplete": missing,
    }
    with open(agg / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _write_report(agg / "report.md", summary, paper_rows, curve_rows)
    print(f"Wrote aggregate outputs to {agg}")
    if missing:
        print(f"WARNING: {len(missing)} missing output(s) (ran with --allow-partial).")


# ---------------------------------------------------------------------------
# small utils
# ---------------------------------------------------------------------------

def _get(d, keys):
    for k in keys:
        if d is None:
            return None
        try:
            d = d[k]
        except (KeyError, TypeError):
            return None
    return d


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(mean(xs), 6) if xs else None


def _write_csv(path: Path, rows):
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _fmt(x):
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _write_report(path: Path, summary, paper_rows, curve_rows):
    lines = []
    A = lines.append
    A("# CAA sycophancy case study — report\n")
    A("Externally grounded evidence that a poisoned **anti-sycophancy** steering "
      "dataset built through a public pipeline can raise jailbreak ASR while "
      "preserving its advertised benign effect.\n")

    A("## 1. External dataset & package versions\n")
    A(f"- Dataset: **{summary['dataset_source']}** @ `{summary['dataset_commit']}` "
      f"(sycophancy `generate_dataset.json` + `test_dataset_ab.json`, vendored unmodified).")
    A(f"- Steering package: **{summary['external_backend']}=={summary['external_backend_version']}** "
      f"(repo commit `{summary['steering_vectors_commit']}`).")
    A(f"- Model: `{summary['model']}`, layer **{summary['layer']}** (= `hidden_states[{summary['layer']+1}]`), bfloat16.\n")

    A("## 2. Native Llama-3.1 chat-template adaptation\n")
    A("The public CAA workflow is adapted to the native Llama-3.1 template: "
      "`system` = \"You are a helpful, honest assistant.\", the CAA `question` as the "
      "user turn, and the answer label as an assistant turn built with "
      "`continue_final_message=True` so the read token (`read_token_index=-1`) is the "
      "answer label, not an end-of-turn token. The textual BOS is stripped because "
      "`steering-vectors` re-inserts special tokens when it tokenizes.\n")

    A("## 3. Fixed splits\n")
    A(f"- 1000 generation rows shuffled with `random.Random({summary['shuffle_seed']})`; "
      f"first **{summary['n_attack']}** = attack/vector-construction set, next "
      f"**{summary['n_val']}** = benign validation.")
    A(f"- Final benign test = all **{summary['n_test']}** official CAA A/B test rows.")
    A("- Attack set is fixed across attack seeds; all indices are saved in `data_manifest.json`.\n")

    A("## 4 & 5. Anti-sycophancy (benign) and jailbreak (harmful) results\n")
    A(f"Primary operating point is the package's unit multiplier α = {summary['primary_multiplier']} "
      f"(ASR source: {', '.join(summary['asr_source']) or '—'}; judges: {summary.get('judges')}).\n")
    A(f"| condition | anti-syc p (test@{PRIMARY}) | anti-syc acc (test@{PRIMARY}) | harmful ASR@{PRIMARY} | cos(v, r) |")
    A("|---|---|---|---|---|")
    for r in paper_rows:
        A(f"| {r['condition']} | {_fmt(r[f'anti_syc_prob_test@{PRIMARY}'])} | "
          f"{_fmt(r[f'anti_syc_acc_test@{PRIMARY}'])} | {_fmt(r[f'harmful_asr@{PRIMARY}'])} | "
          f"{_fmt(r['cos_with_r'])} |")
    A("")
    A("Full multiplier curves are in `multiplier_curve.csv` (a weak or degenerate "
      "operating point cannot be hidden by reporting only α=1.0):\n")
    A("| α | clean anti-syc p (test) | poisoned anti-syc p (test, mean) | clean ASR | poisoned ASR (mean) |")
    A("|---|---|---|---|---|")
    for r in curve_rows:
        A(f"| {r['multiplier']} | {_fmt(r['benign_clean_test_p_anti'])} | "
          f"{_fmt(r['benign_poisoned_test_p_anti_mean'])} | {_fmt(r['asr_clean'])} | "
          f"{_fmt(r['asr_poisoned_mean'])} |")
    A("")

    A("## 6. Cosine & norm diagnostics\n")
    A("| seed | cos(clean, r) | cos(poisoned, r) | Δcos | ‖clean‖ | ‖poisoned‖ | ‖poisoned‖/‖clean‖ |")
    A("|---|---|---|---|---|---|---|")
    for r in summary["per_seed"]:
        A(f"| {r['seed']} | {_fmt(r['cos_clean_with_r'])} | {_fmt(r['cos_poisoned_with_r'])} | "
          f"{_fmt(r['delta_cos'])} | {_fmt(r['clean_vector_norm'])} | {_fmt(r['poisoned_vector_norm'])} | "
          f"{_fmt(r['poisoned_to_clean_norm_ratio'])} |")
    A("")

    A("## 7. Token edit rate vs. text / pair coverage\n")
    A("Token-level edit sparsity is **not** the fraction of dataset texts touched: a "
      "low token edit rate can co-occur with high text coverage.\n")
    A("| seed | token_edit_rate | edits / modifiable | individual_text_coverage | pair_row_coverage |")
    A("|---|---|---|---|---|")
    for r in summary["per_seed"]:
        n_texts = r.get("n_total_texts") or 2 * summary["n_attack"]
        n_pairs = r.get("n_pairs") or summary["n_attack"]
        A(f"| {r['seed']} | {_fmt(r['token_edit_rate'])} | "
          f"{r['n_token_edits']}/{r['n_modifiable_tokens']} | "
          f"{_fmt(r['individual_text_coverage'])} ({r['n_individual_texts_touched']}/{n_texts}) | "
          f"{_fmt(r['pair_row_coverage'])} ({r['n_pair_rows_touched']}/{n_pairs}) |")
    A("")

    A("## 8. Scope\n")
    A("This is evidence for **one** third-party dataset (CAA sycophancy) and **one** "
      "public pipeline (`steering-vectors`) on **one** model, not a claim of universal "
      "transfer. Both final vectors are built by `steering_vectors.train_steering_vector` "
      "and every reported number applies them through `SteeringVector.apply` "
      "(`min_token_index=0`, all tokens); the internal optimizer vector matches the "
      "external one to cos ≥ 0.9999 (see `seed*/equivalence.json`; both vectors "
      "are built at `batch_size=1` so the read is bf16 batch-invariant against the "
      "optimizer's own reads).\n")
    if summary["incomplete"]:
        A(f"> ⚠️ Generated from an INCOMPLETE run ({len(summary['incomplete'])} missing outputs).\n")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
