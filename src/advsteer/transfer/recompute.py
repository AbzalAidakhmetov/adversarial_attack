"""Cross-model recompute worker (a subprocess step of scripts/run_cross_transfer.py).

Recomputes a steering vector on a TARGET model from the source's poisoned/clean
pair texts — the same mean-difference `build_adv_stealth` uses for its clean
vector, with no re-optimization:

    v = mean h_tgt(POS) - mean h_tgt(NEG)   (poisoned from the final texts,
                                             clean from the originals)

It is its own module — invoked as `uv run python -m advsteer.transfer.recompute`,
mirroring `advsteer.attack.build_adv_stealth` / `advsteer.eval.evaluate_asr` — so
it runs in a fresh process (the target model is fully released on exit, leaving the
orchestrator holding no GPU memory) and the invocation matches the rest of the
pipeline rather than a re-entrant sentinel on the orchestrator's own argv.

Writes `<out_dir>/steering_vector.pt` (the dict format `evaluate_asr` loads) and
`<out_dir>/transfer_stats.json` (the `cos(v, -r)` diagnostics, only with --cos_diag).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from advsteer.data import get_hidden_last, compute_refusal_direction, save_json

_REFUSAL_SAMPLES = 128  # matches build_adv_stealth's default (the caller doesn't override it)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m advsteer.transfer.recompute")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--target_model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--harmful", required=True)
    ap.add_argument("--harmless", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device_map", default="cuda")   # 'auto' to shard a large target
    ap.add_argument("--cos_diag", action="store_true",
                    help="also record cos(v,-r) (costs a 256-sample refusal forward); off by default")
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    bs = a.batch_size
    hf = a.layer + 1  # hidden_states index (0 = embeddings), matching the attack's hf_layer

    summary = json.load(open(a.summary))
    tok = AutoTokenizer.from_pretrained(a.target_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.target_model,
        torch_dtype=torch.bfloat16 if a.dtype == "bfloat16" else torch.float32,
        device_map=a.device_map).eval()

    # steering vector = mean h(POS) - mean h(NEG): poisoned from the final texts, clean
    # from the originals (the same build build_adv_stealth uses for its clean vector).
    h_pos_p = get_hidden_last(model, tok, summary["final_pos_texts"], hf, bs)
    h_neg_p = get_hidden_last(model, tok, summary["final_neg_texts"], hf, bs)
    v_p = h_pos_p.mean(0) - h_neg_p.mean(0)
    h_pos_c = get_hidden_last(model, tok, summary["original_pos_texts"], hf, bs)
    h_neg_c = get_hidden_last(model, tok, summary["original_neg_texts"], hf, bs)
    v_c = h_pos_c.mean(0) - h_neg_c.mean(0)

    stats = {"target_model": a.target_model, "layer": a.layer, "norm_poisoned": v_p.norm().item()}
    if a.cos_diag:  # optional: cos(v,-r) needs a 256-sample refusal forward the vector doesn't use
        neg_r = -compute_refusal_direction(model, tok, hf, a.harmful, a.harmless, _REFUSAL_SAMPLES, bs)
        cos = lambda v: F.cosine_similarity(v.unsqueeze(0), neg_r.unsqueeze(0)).item()
        stats["cos_poisoned"], stats["cos_clean"] = cos(v_p), cos(v_c)

    # Atomic write: save to a temp path then rename, so a kill mid-save never leaves a
    # truncated steering_vector.pt that the skip-guard would later feed to torch.load.
    tmp = out_dir / "steering_vector.pt.tmp"
    torch.save({"steering_vector_clean": v_c.cpu(), "steering_vector_poisoned": v_p.cpu(),
                "layer": a.layer, "model": a.target_model}, tmp)
    os.replace(tmp, out_dir / "steering_vector.pt")
    save_json(str(out_dir / "transfer_stats.json"), stats)
    print(f"  recompute @ L{a.layer}: |v_pois|={stats['norm_poisoned']:.2f}"
          + (f"  cos(v_pois,-r)={stats['cos_poisoned']:.4f}" if a.cos_diag else ""))


if __name__ == "__main__":
    main()
