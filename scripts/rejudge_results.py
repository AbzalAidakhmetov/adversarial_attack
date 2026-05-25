"""Re-judge cached eval completions with a different LLM judge.

Walks `results/**/refusal_*_eval_*.json`, extracts the cached
(prompt, response) pairs, and re-runs the judge with the provider/model
specified on the CLI. Writes a sibling file with the judge tag in the
filename so the original primary-judge output is untouched.

Restartable: skips any input whose sibling output already exists. Safe to
re-run after partial failures.

Usage:
  uv run python scripts/rejudge_results.py --provider together \\
      --model meta-llama/Llama-3.3-70B-Instruct-Turbo --tag llama70b

  uv run python scripts/rejudge_results.py --provider openai \\
      --model gpt-4.1 --tag gpt41
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from advsteer import PROJECT_ROOT
from advsteer.classifiers import judge_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rejudge")


def sibling_path(src: Path, tag: str) -> Path:
    return src.with_name(f"{src.stem}.judge_{tag}.json")


def rejudge_one(src: Path, provider: str, model: str, tag: str, concurrency: int) -> Optional[dict]:
    dst = sibling_path(src, tag)
    if dst.exists():
        log.info("skip (exists): %s", dst)
        return None

    with src.open() as f:
        data = json.load(f)
    completions = data.get("completions") or []
    if not completions:
        log.warning("no completions in %s — skipping", src)
        return None

    prompts = [c["prompt"] for c in completions]
    responses = [c["response"] for c in completions]

    results = judge_fn(
        prompts,
        responses,
        judge={"provider": provider, "model": model, "concurrency": concurrency},
    )

    skipped = 0
    kept_labels: list[int] = []
    out_completions = []
    for c, (label, reason) in zip(completions, results):
        cc = dict(c)
        cc["is_jailbreak"] = int(label) if label is not None else None
        if label is None:
            cc["judge_skip_reason"] = reason
            skipped += 1
        else:
            kept_labels.append(int(label))
        out_completions.append(cc)

    eval_out = {
        "judge": {"provider": provider, "model": model, "tag": tag},
        "judge_skipped_count": skipped,
        "judge_n_classified": len(kept_labels),
        "judge_success_rate": float(np.mean(kept_labels)) if kept_labels else None,
        "completions": out_completions,
    }

    if all("category" in c for c in completions):
        per_cat = {}
        for cat in sorted({c["category"] for c in completions}):
            cl = [cc["is_jailbreak"] for cc in out_completions
                  if cc.get("category") == cat and cc["is_jailbreak"] is not None]
            per_cat[cat] = float(np.mean(cl)) if cl else None
        eval_out["judge_per_category"] = per_cat

    with dst.open("w") as f:
        json.dump(eval_out, f, indent=4)
    log.info("wrote %s (ASR=%s, skipped=%d/%d)",
             dst, eval_out["judge_success_rate"], skipped, len(completions))
    return eval_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["anthropic", "openai", "together"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True,
                    help="short slug for filename, e.g. 'llama70b' or 'gpt41'")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--results_root", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--glob", default="**/refusal_*_eval_*.json",
                    help="glob under results_root; default matches all eval JSONs")
    args = ap.parse_args()

    root = Path(args.results_root)
    files = sorted(p for p in root.glob(args.glob)
                   if ".judge_" not in p.name)  # don't rejudge sibling outputs
    log.info("found %d input files under %s", len(files), root)

    n_done = n_skipped = 0
    for i, src in enumerate(files, 1):
        log.info("[%d/%d] %s", i, len(files), src.relative_to(root))
        out = rejudge_one(src, args.provider, args.model, args.tag, args.concurrency)
        if out is None:
            n_skipped += 1
        else:
            n_done += 1

    log.info("done: %d judged, %d already-existed-or-empty", n_done, n_skipped)


if __name__ == "__main__":
    main()
