#!/usr/bin/env python3
"""Defense: project the refusal direction out of an attribute steering vector.

Given a steering_vector.pt (clean + poisoned at layer L), compute the refusal
direction r at the same layer from the train splits and produce defended
variants:

    v_def = v - (v . r_hat) * r_hat

By construction cos(v_def, r) = 0, so the defended vector cannot push
activations along the refusal axis. The component of v orthogonal to r --
which is what the attribute relies on -- is preserved, so attribute
effectiveness on harmless prompts should be largely retained while ASR drops
back toward the clean baseline.

Writes a new .pt containing the original keys plus
`steering_vector_{clean,poisoned}_defended` and `refusal_direction`, and a
small JSON sidecar reporting the cosines before/after for the record.

Usage:
    uv run python -m advsteer.defense.orthogonalize_steering \
        --input results/<model>/<attr>/steering_vector.pt \
        --model google/gemma-2-2b-it
"""

import argparse, gc, json, os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from advsteer.data import compute_refusal_direction
from advsteer.classifiers import set_seed


_ROOT = str(Path(__file__).resolve().parents[3])


def parse_args():
    ap = argparse.ArgumentParser(description="Orthogonalize steering vector against refusal direction")
    ap.add_argument("--input", required=True, help="Path to steering_vector.pt from build_adv_stealth.py")
    ap.add_argument("--output", default=None, help="Output .pt path (default: <input_dir>/steering_vector_defended.pt)")
    ap.add_argument("--model", default=None, help="Override model id (default: read from input .pt)")
    ap.add_argument("--refusal_harmful_path",  default=os.path.join(_ROOT, "data", "refusal", "splits", "harmful_train.json"))
    ap.add_argument("--refusal_harmless_path", default=os.path.join(_ROOT, "data", "refusal", "splits", "harmless_train.json"))
    ap.add_argument("--refusal_samples", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data = torch.load(args.input, map_location="cpu", weights_only=False)
    if not (isinstance(data, dict) and "steering_vector_poisoned" in data):
        raise ValueError(f"{args.input} is not a build_adv_stealth.py dict (missing steering_vector_poisoned).")

    layer = int(data["layer"])
    model_id = args.model or data.get("model")
    if model_id is None:
        raise ValueError("No model id in .pt and --model not provided.")
    print(f"Loaded steering vectors @ layer {layer} for model {model_id}")

    v_clean = data["steering_vector_clean"].float()
    v_pois  = data["steering_vector_poisoned"].float()

    # ── Load model and compute refusal direction at the steering layer ────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # HF hidden_states[0] is the embedding output; layer L's residual stream
    # output is at hidden_states[L+1] -- matches the convention used by
    # build_adv_stealth.py (hf_layer = args.layer + 1).
    hf_layer = layer + 1
    r = compute_refusal_direction(
        model, tokenizer, hf_layer,
        args.refusal_harmful_path, args.refusal_harmless_path,
        args.refusal_samples, args.batch_size,
    ).float().cpu()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Orthogonalize ─────────────────────────────────────────────────────────
    r_hat = r / r.norm().clamp_min(1e-12)
    def project_out(v: torch.Tensor) -> torch.Tensor:
        return v - torch.dot(v, r_hat) * r_hat

    v_clean_def = project_out(v_clean)
    v_pois_def  = project_out(v_pois)

    def cos_neg_r(v: torch.Tensor) -> float:
        return F.cosine_similarity(v.unsqueeze(0), (-r).unsqueeze(0)).item()
    def cos_pos_r(v: torch.Tensor) -> float:
        return F.cosine_similarity(v.unsqueeze(0), r.unsqueeze(0)).item()

    report = {
        "model": model_id,
        "layer": layer,
        "refusal_dir_norm": float(r.norm()),
        "clean": {
            "cos_neg_r":     cos_neg_r(v_clean),
            "cos_neg_r_def": cos_neg_r(v_clean_def),
            "norm":          float(v_clean.norm()),
            "norm_def":      float(v_clean_def.norm()),
            "delta_norm_pct":   100.0 * (v_clean_def.norm().item() / v_clean.norm().item() - 1.0),
        },
        "poisoned": {
            "cos_neg_r":     cos_neg_r(v_pois),
            "cos_neg_r_def": cos_neg_r(v_pois_def),
            "norm":          float(v_pois.norm()),
            "norm_def":      float(v_pois_def.norm()),
            "delta_norm_pct":   100.0 * (v_pois_def.norm().item() / v_pois.norm().item() - 1.0),
        },
        # Sanity: post-defense cos with r itself should be ~0 (numerical zero).
        "residual_cos_with_r": {
            "clean_def":    cos_pos_r(v_clean_def),
            "poisoned_def": cos_pos_r(v_pois_def),
        },
    }

    print("\n── Orthogonalization report ─────────────────────────────────────")
    print(f"  refusal direction norm: {report['refusal_dir_norm']:.4f}")
    print(f"  clean    cos(v,-r): {report['clean']['cos_neg_r']:+.4f}  →  {report['clean']['cos_neg_r_def']:+.4f}"
          f"   norm: {report['clean']['norm']:.3f} → {report['clean']['norm_def']:.3f}"
          f"  ({report['clean']['delta_norm_pct']:+.2f}%)")
    print(f"  poisoned cos(v,-r): {report['poisoned']['cos_neg_r']:+.4f}  →  {report['poisoned']['cos_neg_r_def']:+.4f}"
          f"   norm: {report['poisoned']['norm']:.3f} → {report['poisoned']['norm_def']:.3f}"
          f"  ({report['poisoned']['delta_norm_pct']:+.2f}%)")
    print(f"  residual cos(v_def, r): clean={report['residual_cos_with_r']['clean_def']:+.2e}"
          f"  poisoned={report['residual_cos_with_r']['poisoned_def']:+.2e}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = args.output or os.path.join(os.path.dirname(args.input), "steering_vector_defended.pt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({
        "steering_vector_clean":             v_clean,
        "steering_vector_poisoned":          v_pois,
        "steering_vector_clean_defended":    v_clean_def,
        "steering_vector_poisoned_defended": v_pois_def,
        "refusal_direction":                 r,
        "layer":                             layer,
        "model":                             model_id,
    }, out_path)

    sidecar = os.path.splitext(out_path)[0] + "_report.json"
    with open(sidecar, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  saved: {out_path}")
    print(f"  report: {sidecar}")


if __name__ == "__main__":
    main()
