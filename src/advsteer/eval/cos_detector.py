#!/usr/bin/env python3
"""Cos-based detector for poisoned steering vectors.

For a fixed (model, layer):
  - Compute the refusal direction r once from harmful_train/harmless_train.
  - For every attribute in data/pair_specs.yaml whose pairs file exists,
    compute the clean steering vector v = mean(POS) - mean(NEG) at the
    target layer from --num_pairs pairs and record cos(v, -r).
    This is the null distribution of "legitimate steering vectors".
  - For every results/<dir>/steering_vector.pt whose 'model' and
    'layer' fields match the current run, record cos for both
    'steering_vector_clean' and 'steering_vector_poisoned'.
  - Sweep thresholds, plot strip-plot + ROC, dump CSV + JSON.

The defense the paper proposes (Sec. 5, "Possible defenses"): flag a
vector iff cos(v, -r) > tau. This script measures how well that flag
separates poisoned from clean vectors and at what FPR.

Usage:
    uv run python -m advsteer.eval.cos_detector \
        --model google/gemma-2-2b-it --layer 14 \
        --output results/cos_detector/gemma_L14/
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from advsteer.data import (
    PAIR_TYPE_SPECS,
    compute_refusal_direction,
    get_hidden_last,
    load_pairs,
)
from advsteer.classifiers import set_seed

_ROOT = str(Path(__file__).resolve().parents[3])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--num_pairs", type=int, default=20,
                    help="Pairs per attribute when computing the clean null vector. "
                         "Matches the attack default for apples-to-apples comparison.")
    ap.add_argument("--refusal_harmful_path",
                    default=os.path.join(_ROOT, "data", "refusal", "splits", "harmful_train.json"))
    ap.add_argument("--refusal_harmless_path",
                    default=os.path.join(_ROOT, "data", "refusal", "splits", "harmless_train.json"))
    ap.add_argument("--refusal_samples", type=int, default=128)
    ap.add_argument("--data_dir", default=os.path.join(_ROOT, "data", "pairs"))
    ap.add_argument("--experiments_dir", default=os.path.join(_ROOT, "results"))
    ap.add_argument("--output", required=True, help="Output dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--batch_size", type=int, default=8)
    return ap.parse_args()


def cos1d(v: torch.Tensor, u: torch.Tensor) -> float:
    return F.cosine_similarity(v.unsqueeze(0).float(), u.unsqueeze(0).float()).item()


def attributes_with_data(args) -> list:
    out = []
    for name, spec in PAIR_TYPE_SPECS.items():
        path = os.path.join(args.data_dir, *spec["path_parts"])
        if os.path.exists(path):
            out.append(name)
    return sorted(out)


def collect_clean_attrs(args, model, tokenizer, hf_layer, neg_refusal_cpu) -> list:
    records = []
    attrs = attributes_with_data(args)
    print(f"\nSweeping {len(attrs)} attributes for clean cos(v_attr, -r)...")
    for pt in attrs:
        try:
            pos, neg, _ = load_pairs(pt, args.num_pairs, args.data_dir)
        except Exception as e:
            print(f"  {pt:30s} skipped ({type(e).__name__})")
            continue
        if len(pos) < 5:
            print(f"  {pt:30s} skipped (only {len(pos)} pairs)")
            continue
        h_pos = get_hidden_last(model, tokenizer, pos, hf_layer, args.batch_size)
        h_neg = get_hidden_last(model, tokenizer, neg, hf_layer, args.batch_size)
        v = (h_pos.mean(0) - h_neg.mean(0)).cpu()
        c = cos1d(v, neg_refusal_cpu)
        records.append({
            "source": "clean_attr", "pair_type": pt, "exp_dir": "",
            "n_pairs": len(pos), "cos": c, "norm": float(v.norm().item()),
        })
        print(f"  {pt:30s} cos(v,-r) = {c:+.4f}  norm = {v.norm().item():7.2f}  ({len(pos)} pairs)")
    return records


def collect_saved_vectors(args, neg_refusal_cpu) -> list:
    records = []
    exp_files = sorted(glob.glob(os.path.join(args.experiments_dir, "**", "steering_vector.pt"),
                                 recursive=True))
    print(f"\nScanning {len(exp_files)} steering_vector.pt files...")
    for fp in exp_files:
        try:
            d = torch.load(fp, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  {fp}: load failed ({e})")
            continue
        if not (isinstance(d, dict) and "steering_vector_poisoned" in d):
            continue
        if d.get("model") != args.model or d.get("layer") != args.layer:
            continue
        exp = os.path.basename(os.path.dirname(fp))
        v_c, v_p = d["steering_vector_clean"], d["steering_vector_poisoned"]
        c_c = cos1d(v_c, neg_refusal_cpu)
        c_p = cos1d(v_p, neg_refusal_cpu)
        records.append({
            "source": "saved_clean", "pair_type": exp, "exp_dir": exp,
            "n_pairs": None, "cos": c_c, "norm": float(v_c.norm().item()),
        })
        records.append({
            "source": "poisoned", "pair_type": exp, "exp_dir": exp,
            "n_pairs": None, "cos": c_p, "norm": float(v_p.norm().item()),
        })
        print(f"  {exp:40s} saved_clean={c_c:+.4f}  poisoned={c_p:+.4f}")
    return records


def threshold_sweep(clean_cos, poison_cos):
    import numpy as np
    if clean_cos.size == 0 or poison_cos.size == 0:
        return None
    lo = min(clean_cos.min(), poison_cos.min()) - 0.05
    hi = max(clean_cos.max(), poison_cos.max()) + 0.05
    taus = np.linspace(lo, hi, 400)
    fpr = np.array([(clean_cos > t).mean() for t in taus])
    tpr = np.array([(poison_cos > t).mean() for t in taus])

    tau_safe = float(clean_cos.max()) + 1e-6
    tau_p99 = float(np.quantile(clean_cos, 0.99))
    mu, sd = float(clean_cos.mean()), float(clean_cos.std(ddof=1))
    tau_3sigma = mu + 3.0 * sd

    def at(t):
        return {"tau": float(t),
                "tpr": float((poison_cos > t).mean()),
                "fpr": float((clean_cos > t).mean())}

    return {
        "taus":  taus.tolist(), "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "operating_points": {
            "max_clean":  at(tau_safe),
            "p99_clean":  at(tau_p99),
            "mu_plus_3sigma": at(tau_3sigma),
        },
    }


def plot_strip(records, sweep, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    clean = np.array([r["cos"] for r in records if r["source"] == "clean_attr"])
    saved = np.array([r["cos"] for r in records if r["source"] == "saved_clean"])
    poison = np.array([r["cos"] for r in records if r["source"] == "poisoned"])
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    if clean.size:
        ax.scatter(clean, rng.uniform(-0.18, 0.18, clean.size),
                   c="tab:blue", alpha=0.6, s=24,
                   label=f"clean attribute vectors (n={clean.size})")
    if saved.size:
        ax.scatter(saved, rng.uniform(0.82, 1.18, saved.size),
                   c="tab:green", alpha=0.8, marker="o", s=60, edgecolor="black",
                   label=f"clean from saved attacks (n={saved.size})")
    if poison.size:
        ax.scatter(poison, rng.uniform(1.82, 2.18, poison.size),
                   c="tab:red", alpha=0.95, marker="X", s=90, edgecolor="black",
                   label=f"poisoned (n={poison.size})")
    if sweep:
        tau = sweep["operating_points"]["max_clean"]["tau"]
        ax.axvline(tau, color="black", linestyle="--", alpha=0.6,
                   label=f"τ = max(clean) = {tau:.3f}")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["clean attrs", "saved-clean", "poisoned"])
    ax.set_xlabel("cos(v, -r)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_roc(sweep, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(sweep["fpr"], sweep["tpr"], "-", label="ROC")
    for name, op in sweep["operating_points"].items():
        ax.scatter([op["fpr"]], [op["tpr"]], s=60, zorder=5,
                   label=f"{name}: τ={op['tau']:.3f}, FPR={op['fpr']:.2f}, TPR={op['tpr']:.2f}")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.3, label="chance")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("FPR (legitimate vectors flagged)")
    ax.set_ylabel("TPR (poisoned vectors caught)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)

    print(f"=== cos-detector | {args.model} @ L{args.layer} ===")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # nnsight-free: data.py's hf_layer convention is (user_layer + 1)
    hf_layer = args.layer + 1

    neg_refusal = -compute_refusal_direction(
        model, tokenizer, hf_layer,
        args.refusal_harmful_path, args.refusal_harmless_path,
        args.refusal_samples, args.batch_size,
    )
    neg_refusal_cpu = neg_refusal.detach().cpu()
    print(f"  -r at L{args.layer}: norm={neg_refusal_cpu.norm().item():.4f}")

    records = []
    records += collect_clean_attrs(args, model, tokenizer, hf_layer, neg_refusal_cpu)
    records += collect_saved_vectors(args, neg_refusal_cpu)

    # CSV
    csv_path = os.path.join(args.output, "cos_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "pair_type", "exp_dir", "n_pairs", "cos", "norm"])
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    # Sweep
    import numpy as np
    clean_cos = np.array([r["cos"] for r in records if r["source"] == "clean_attr"])
    poison_cos = np.array([r["cos"] for r in records if r["source"] == "poisoned"])

    summary = {
        "model": args.model, "layer": args.layer,
        "n_clean_attrs": int(clean_cos.size),
        "n_poisoned": int(poison_cos.size),
        "clean_cos": {
            "mean": float(clean_cos.mean()) if clean_cos.size else None,
            "std":  float(clean_cos.std(ddof=1)) if clean_cos.size > 1 else None,
            "min":  float(clean_cos.min()) if clean_cos.size else None,
            "max":  float(clean_cos.max()) if clean_cos.size else None,
        },
        "poisoned_cos": {
            "mean": float(poison_cos.mean()) if poison_cos.size else None,
            "min":  float(poison_cos.min()) if poison_cos.size else None,
            "max":  float(poison_cos.max()) if poison_cos.size else None,
        },
    }
    sweep = threshold_sweep(clean_cos, poison_cos)
    if sweep is not None:
        summary["operating_points"] = sweep["operating_points"]
        tag = f"{args.model.split('/')[-1]} L{args.layer}"
        plot_strip(records, sweep, os.path.join(args.output, "cos_strip.png"), tag)
        plot_roc(sweep, os.path.join(args.output, "cos_roc.png"), f"ROC — {tag}")

    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
