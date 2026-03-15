#!/usr/bin/env python3
"""
Summarize clean baselines and poisoned runs for a behavior family.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_eval_rates(results_path: Path):
    rows = load_json(results_path)
    by_weight = {}
    for row in rows:
        weight = row.get("weight")
        if weight is None:
            continue
        by_weight[float(weight)] = {
            "llama33_success_rate": row.get("llama33_success_rate"),
            "steering_success_rate": row.get("steering_success_rate"),
        }
    return by_weight


def summarize_values(values):
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(clean),
        "mean": mean(clean),
        "std": pstdev(clean) if len(clean) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-baseline", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--run-prefix", type=str, required=True)
    args = ap.parse_args()

    clean_eval = load_eval_rates(args.clean_baseline)
    run_dirs = sorted(
        path for path in args.run_root.iterdir()
        if path.is_dir() and path.name.startswith(args.run_prefix)
    )

    runs = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        eval_path = run_dir / "llama33_eval_100_256tok" / "results"
        if not summary_path.exists() or not eval_path.exists():
            continue
        summary = load_json(summary_path)
        eval_rates = load_eval_rates(eval_path)
        runs.append(
            {
                "name": run_dir.name,
                "k_adv": summary["config"]["k_adv"],
                "seed": summary["config"]["seed"],
                "cos": summary["evaluation"]["resulting_cos_sim_with_neg_refusal"],
                "dot": summary["evaluation"]["resulting_dot_with_neg_refusal_unit"],
                "w2_llama33": eval_rates.get(2.0, {}).get("llama33_success_rate"),
                "w4_llama33": eval_rates.get(4.0, {}).get("llama33_success_rate"),
                "w2_steering": eval_rates.get(2.0, {}).get("steering_success_rate"),
                "w4_steering": eval_rates.get(4.0, {}).get("steering_success_rate"),
            }
        )

    grouped = {}
    for run in runs:
        grouped.setdefault(run["k_adv"], []).append(run)

    payload = {
        "clean_baseline": {
            "w2_llama33": clean_eval.get(2.0, {}).get("llama33_success_rate"),
            "w4_llama33": clean_eval.get(4.0, {}).get("llama33_success_rate"),
            "w2_steering": clean_eval.get(2.0, {}).get("steering_success_rate"),
            "w4_steering": clean_eval.get(4.0, {}).get("steering_success_rate"),
        },
        "runs": runs,
        "by_k": {},
    }

    for k_adv, items in sorted(grouped.items()):
        payload["by_k"][str(k_adv)] = {
            "n_runs": len(items),
            "cos": summarize_values([item["cos"] for item in items]),
            "dot": summarize_values([item["dot"] for item in items]),
            "w2_llama33": summarize_values([item["w2_llama33"] for item in items]),
            "w4_llama33": summarize_values([item["w4_llama33"] for item in items]),
            "w2_steering": summarize_values([item["w2_steering"] for item in items]),
            "w4_steering": summarize_values([item["w4_steering"] for item in items]),
        }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
