"""Print every ASR / hAttr number the paper cites, computed from the *current*
``results`` files (which carry the majority-vote ``judge_success_rate`` after
``scripts/aggregate_majority_judge.py``). Used to refresh the in-text claims
after switching judges.
"""
from __future__ import annotations

from pathlib import Path

import sys
from advsteer import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "plots"))
from plot_main_results import COMBOS, load_combo_metrics  # type: ignore[import-not-found]
from _seeds import aggregate_from_seeds, metric_extractor  # type: ignore[import-not-found]


RESULTS = PROJECT_ROOT / "results"


def main():
    rows = [load_combo_metrics(c, RESULTS) for c in COMBOS]

    print("\n== Per-combo ASR (majority judge) ==")
    print(f"{'combo':<24}  {'asr_clean':>14}  {'asr_pois':>14}  {'dASR':>8}  "
          f"{'hattr_c':>10}  {'hattr_p':>10}")
    for r in rows:
        c = r["asr_clean"]; p = r["asr_poisoned"]
        hc = r["hattr_clean"]; hp = r["hattr_poisoned"]
        name = f"{r['model']} {r['label']}"
        print(f"{name:<24}  {c.mean:.3f}±{c.std:.3f}  {p.mean:.3f}±{p.std:.3f}  "
              f"{r['delta_asr']:+.3f}  {hc.mean:.3f}±{hc.std:.3f}  {hp.mean:.3f}±{hp.std:.3f}")

    asr_pois = [r["asr_poisoned"].mean for r in rows]
    asr_clean = [r["asr_clean"].mean for r in rows]
    dasr = [r["delta_asr"] for r in rows]
    sig_pois = [r["asr_poisoned"].std for r in rows]
    print(f"\nPoisoned ASR range: {min(asr_pois):.3f} – {max(asr_pois):.3f}")
    print(f"Clean ASR range:    {min(asr_clean):.3f} – {max(asr_clean):.3f}")
    print(f"ΔASR range:         {min(dasr):+.3f} – {max(dasr):+.3f}  (in pp: +{min(dasr)*100:.0f} to +{max(dasr)*100:.0f})")
    print(f"σ (poisoned ASR) ≤ 0.07 on: {sum(1 for s in sig_pois if s <= 0.07)} / {len(sig_pois)}")
    print(f"σ (poisoned ASR) ≤ 0.10 on: {sum(1 for s in sig_pois if s <= 0.10)} / {len(sig_pois)}")
    sig_max = max(sig_pois)
    sig_max_combo = rows[sig_pois.index(sig_max)]
    print(f"max σ: {sig_max:.3f} on {sig_max_combo['model']} {sig_max_combo['label']}")
    print(f"ratio poisoned/clean ASR: " +
          ", ".join(f"{r['asr_poisoned'].mean / r['asr_clean'].mean:.1f}x"
                    if r['asr_clean'].mean > 0 else "inf"
                    for r in rows))

    print("\n== Defense (orthogonalization) ==")
    drows = []
    for c in COMBOS:
        base = RESULTS / c.directory
        w = c.weight
        try:
            asr_def = aggregate_from_seeds(
                base, metric_extractor(f"results_defense_poisoned_harmful_w{w}", "judge_success_rate"),
                seeds=(0,),
            )
            asr_clean_def = aggregate_from_seeds(
                base, metric_extractor(f"results_clean_harmful_w{w}", "judge_success_rate"),
            )
            asr_pois_def = aggregate_from_seeds(
                base, metric_extractor(f"results_poisoned_harmful_w{w}", "judge_success_rate"),
            )
        except Exception as e:
            print(f"  {c.model} {c.label}: skip ({e})")
            continue
        gap = asr_pois_def.mean - asr_clean_def.mean
        recovered = (asr_pois_def.mean - asr_def.mean) / gap if gap > 0 else float("nan")
        drow = {"name": f"{c.model} {c.label}",
                "asr_def": asr_def.mean, "asr_clean": asr_clean_def.mean,
                "asr_pois": asr_pois_def.mean,
                "delta": asr_def.mean - asr_pois_def.mean,
                "recovered_pct": recovered * 100}
        drows.append(drow)
        print(f"  {drow['name']:<24}  clean={drow['asr_clean']:.3f}  "
              f"pois={drow['asr_pois']:.3f}  def={drow['asr_def']:.3f}  "
              f"Δdef={drow['delta']:+.3f}  recov={drow['recovered_pct']:.1f}%")
    if drows:
        import statistics
        deltas = [d["delta"] for d in drows]
        recs = [d["recovered_pct"] for d in drows if d["recovered_pct"] == d["recovered_pct"]]
        print(f"\nΔdef range:         {min(deltas):+.3f} – {max(deltas):+.3f}")
        print(f"mean Δdef:          {statistics.mean(deltas):+.3f}")
        print(f"mean recovered:     {statistics.mean(recs):.1f}%")
        print(f"median recovered:   {statistics.median(recs):.1f}%")
        print(f"defended ≤ clean on: " + ", ".join(d['name'] for d in drows if d['asr_def'] <= d['asr_clean']))

    print("\n== Constrained-bypass sweep (Llama lowercase w=2) ==")
    base = RESULTS / "llama31/lowercase"
    asr_clean = aggregate_from_seeds(
        base, metric_extractor("results_clean_harmful_w2", "judge_success_rate"),
    )
    asr_pois = aggregate_from_seeds(
        base, metric_extractor("results_poisoned_harmful_w2", "judge_success_rate"),
    )
    print(f"  clean    ASR = {asr_clean.mean:.3f} ± {asr_clean.std:.3f}")
    print(f"  poisoned ASR = {asr_pois.mean:.3f} ± {asr_pois.std:.3f}")
    for cap in (0.05, 0.10, 0.15, 0.20, 0.30):
        cap_str = f"cap_hard{cap}"
        try:
            asr_cap = aggregate_from_seeds(
                base,
                lambda sd, c=cap_str: metric_extractor("results_poisoned_harmful_w2",
                                                       "judge_success_rate")(sd / c),
            )
        except Exception as e:
            print(f"  τ={cap}: skip ({e})")
            continue
        gap = asr_pois.mean - asr_clean.mean
        bypass = (asr_cap.mean - asr_clean.mean) / gap if gap > 0 else float("nan")
        print(f"  τ={cap:.2f}: ASR = {asr_cap.mean:.3f} ± {asr_cap.std:.3f}  bypass={bypass*100:.1f}%")


if __name__ == "__main__":
    main()
