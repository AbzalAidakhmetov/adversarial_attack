#!/usr/bin/env python3
"""
Split a snapshots.pt file (from build_adv_stealth.py --snapshot_every) into
individual steering_vector .pt files compatible with evaluate_asr.py.

For each snapshot iter k saves:
  <out_dir>/snap_iter<k>.pt   {'steering_vector_poisoned', 'steering_vector_clean', 'layer', ...}

Also writes a manifest.json listing the snapshots.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", required=True, help="Path to snapshots.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stride", type=int, default=1, help="Take every Nth snapshot")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.snapshots, map_location="cpu", weights_only=False)
    layer = int(data["layer"])
    model = data.get("model", "")
    pair_type = data.get("pair_type", "")
    snaps = data["snapshots"]
    steer_clean = data["steering_vector_clean"]
    steer_poisoned = data["steering_vector_poisoned"]

    manifest = {
        "model": model,
        "pair_type": pair_type,
        "layer": layer,
        "n_snapshots": len(snaps),
        "stride": args.stride,
        "files": [],
    }

    # Always save the clean and final-poisoned vectors
    clean_path = out / "vec_clean.pt"
    torch.save({
        "steering_vector_clean": steer_clean,
        "steering_vector_poisoned": steer_clean,  # alias so eval can load either key
        "layer": layer,
        "tag": "clean",
        "iter": -1,
    }, clean_path)
    manifest["files"].append({"tag": "clean", "iter": -1, "path": "vec_clean.pt", "cos": None, "n_mods": 0})

    for i, s in enumerate(snaps):
        if i % args.stride != 0 and i != len(snaps) - 1:
            continue
        it = int(s["iter"])
        v = s["vector"]
        p = out / f"vec_snap_iter{it:05d}.pt"
        torch.save({
            "steering_vector_poisoned": v,
            "steering_vector_clean": steer_clean,
            "layer": layer,
            "tag": f"snap_iter{it}",
            "iter": it,
            "cos": float(s.get("cos", 0.0)),
            "n_mods": int(s.get("n_mods", 0)),
        }, p)
        manifest["files"].append({
            "tag": f"snap_iter{it}",
            "iter": it,
            "path": p.name,
            "cos": float(s.get("cos", 0.0)),
            "n_mods": int(s.get("n_mods", 0)),
        })

    final_path = out / "vec_poisoned_fresh.pt"
    torch.save({
        "steering_vector_poisoned": steer_poisoned,
        "steering_vector_clean": steer_clean,
        "layer": layer,
        "tag": "poisoned_fresh",
        "iter": -2,
    }, final_path)
    manifest["files"].append({
        "tag": "poisoned_fresh",
        "iter": -2,
        "path": "vec_poisoned_fresh.pt",
        "cos": None,
        "n_mods": None,
    })

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest['files'])} vectors to {out}/")
    for e in manifest["files"]:
        print(f"  {e['tag']:<22s} iter={e['iter']:>5d}  -> {e['path']}")


if __name__ == "__main__":
    main()
