#!/usr/bin/env python3
"""Compare results/<model>/<attr>/results_* to the CLAUDE.md headline table.

Reads:
  results/<model>/<attr>/results_clean_harmful/results     -> ASR  clean
  results/<model>/<attr>/results_clean_harmless/results    -> hAttr clean
  results/<model>/<attr>/results_poisoned_harmful/results  -> ASR  poisoned
  results/<model>/<attr>/results_poisoned_harmless/results -> hAttr poisoned

Each `results` file is a JSON list with one record per (layer, weight); we
pick the record matching the combo's (layer, weight) and pull:
  ASR   -> judge_success_rate
  hAttr -> steering_success_rate

Run from project root:
    uv run python slurm/verify_best.py
"""
import json
from pathlib import Path
from typing import Optional

# (exp_dir, layer, weight, expected_hAttr_clean, expected_hAttr_poisoned,
#  expected_ASR_clean, expected_ASR_poisoned)
HEADLINES = [
    ("gemma/spanish",        14, 3, 0.84, 0.94, 0.03, 0.51),
    ("gemma/french",         14, 3, 0.87, 0.86, 0.09, 0.44),
    ("gemma/has_bold_only",  14, 4, 0.73, 0.72, 0.05, 0.21),
    ("llama31/lowercase",    18, 2, 0.84, 0.91, 0.06, 0.39),
    ("llama31/spanish",      18, 3, 0.87, 0.82, 0.01, 0.20),
]

ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = ROOT / "results"


def _pick(records, layer: int, weight: float, key: str) -> Optional[float]:
    if not records:
        return None
    for rec in records:
        if int(rec.get("layer", -1)) == layer and abs(float(rec.get("weight", -1)) - weight) < 1e-6:
            val = rec.get(key)
            return float(val) if val is not None else None
    # fallback: single record
    if len(records) == 1:
        val = records[0].get(key)
        return float(val) if val is not None else None
    return None


def _load(path: Path) -> Optional[list]:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _fmt(x: Optional[float]) -> str:
    return "  -  " if x is None else f"{x:5.2f}"


def main() -> int:
    print(f"{'combo':<28} {'L':>3} {'w':>4}  {'hAttr_c':>8} {'hAttr_p':>8} {'ΔhAttr':>7} {'exp_Δ':>6}   {'ASR_c':>6} {'ASR_p':>6} {'ΔASR':>6} {'exp_Δ':>6}  status")
    print("-" * 130)
    any_missing = False
    for exp, layer, weight, eh_c, eh_p, ea_c, ea_p in HEADLINES:
        d = EXP_ROOT / exp
        ch = _load(d / "results_clean_harmful" / "results")
        cm = _load(d / "results_clean_harmless" / "results")
        ph = _load(d / "results_poisoned_harmful" / "results")
        pm = _load(d / "results_poisoned_harmless" / "results")

        asr_c = _pick(ch, layer, weight, "judge_success_rate") if ch else None
        asr_p = _pick(ph, layer, weight, "judge_success_rate") if ph else None
        hat_c = _pick(cm, layer, weight, "steering_success_rate") if cm else None
        hat_p = _pick(pm, layer, weight, "steering_success_rate") if pm else None

        d_hatt = (hat_p - hat_c) if (hat_c is not None and hat_p is not None) else None
        d_asr  = (asr_p - asr_c) if (asr_c is not None and asr_p is not None) else None
        exp_d_hatt = eh_p - eh_c
        exp_d_asr  = ea_p - ea_c

        present = [ch is not None, cm is not None, ph is not None, pm is not None]
        n_done = sum(present)
        if n_done < 4:
            any_missing = True
            status = f"pending ({n_done}/4)"
        else:
            status = "done"

        print(
            f"{exp:<28} {layer:>3} {weight:>4}  "
            f"{_fmt(hat_c)} {_fmt(hat_p)} {_fmt(d_hatt):>7} {exp_d_hatt:>+6.2f}   "
            f"{_fmt(asr_c)} {_fmt(asr_p)} {_fmt(d_asr):>6} {exp_d_asr:>+6.2f}  "
            f"{status}"
        )

    return 0 if not any_missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
