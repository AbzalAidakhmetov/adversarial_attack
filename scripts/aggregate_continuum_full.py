#!/usr/bin/env python3
"""
Aggregate continuum_full evaluation results into a summary table.

Walks experiments/<exp>/continuum_full/<tag>/ subdirs and reads results from
either direct path or nested-under-hydra-logs path.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def find_results(base_dir: Path, kind: str):
    """kind in {'harmful', 'harmless'}. Returns dict from `results` JSON or None."""
    direct = base_dir / kind / "results"
    if direct.is_file():
        return json.loads(direct.read_text())
    matches = list(base_dir.glob(f"hydra_logs/**/{kind}/results"))
    for m in matches:
        try:
            return json.loads(m.read_text())
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="Path to experiments/<name>/")
    args = ap.parse_args()

    base = Path(args.exp_dir) / "continuum_full"
    manifest_path = base / "vectors" / "manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: {manifest_path} missing")
        sys.exit(1)
    m = json.loads(manifest_path.read_text())

    print(f"  CONTINUUM-FULL  {Path(args.exp_dir).name}  {m.get('pair_type')}@L{m.get('layer')}")
    print(f"  {'tag':<22s} {'iter':>6s} {'cos':>7s} {'cASR':>5s} {'cAttr':>5s} {'hAttr':>5s} {'pPerp':>7s}")

    rows = []
    for e in m["files"]:
        tag = e["tag"]
        d = base / tag
        h = find_results(d, "harmful")
        l = find_results(d, "harmless")
        asr = cattr = hattr = pperp = "----"
        if h and isinstance(h, list) and h:
            asr_val = h[0].get('llama33_success_rate')
            asr = f"{(asr_val or 0):.2f}"
            cattr = f"{h[0].get('steering_success_rate') or 0:.2f}"
        if l and isinstance(l, list) and l:
            hattr = f"{l[0].get('steering_success_rate') or 0:.2f}"
            mp = l[0].get('mean_perplexity') or 0
            pperp = "inf" if mp == float("inf") else (f"{mp:.0f}" if isinstance(mp, (int, float)) else "?")
        cs = "-" if e.get("cos") is None else f"{e['cos']:+.3f}"
        print(f"  {tag:<22s} {e['iter']:>6d} {cs:>7s} {asr:>5s} {cattr:>5s} {hattr:>5s} {pperp:>7s}")
        rows.append({"tag": tag, "iter": e["iter"], "cos": e.get("cos"),
                     "harmful_asr": h[0].get('llama33_success_rate') if h else None,
                     "harmful_attr": h[0].get('steering_success_rate') if h else None,
                     "harmless_attr": l[0].get('steering_success_rate') if l else None,
                     "harmless_perp": l[0].get('mean_perplexity') if l else None})

    out = base / "summary.json"
    out.write_text(json.dumps({"results": rows}, indent=2))
    print(f"\n  Saved structured: {out}")


if __name__ == "__main__":
    main()
