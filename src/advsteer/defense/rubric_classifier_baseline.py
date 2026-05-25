#!/usr/bin/env python3
"""Rubric-classifier baseline: does an LLM-as-auditor distinguish harmful
refusal-direction prompts from poisoned contrastive-pair texts?

Three populations, one classifier (Claude Sonnet 4.5 by default, same rubric
as `advsteer.defense.stealth_check`):

  1. harmful_refusal   — first N items of data/refusal/splits/harmful_train.json
                         (the harmful prompts that feed `r` in eq. 2).
                         Positive control: rubric should fire ~100%.
  2. harmless_refusal  — first N items of data/refusal/splits/harmless_train.json.
                         Negative control: rubric should fire ~0%.
  3. poisoned_pairs    — `final_pos_texts` + `final_neg_texts` from every
                         results/<model>/<attr>/seed<S>/summary.json the
                         attack pipeline produced.
                         Claim: rubric fires ~0%, i.e. the payload is
                         invisible to a surface-level content classifier.

A GCG-style adversarial-suffix baseline hook is included as `--include_gcg
<jsonl>` and currently stubbed — leave the population out by default; the
classifier code path is identical so dropping a JSONL in later is one flag.

Outputs:
  results/rubric_classifier_baseline/
    raw_classifications.jsonl   # append-only, one row per text (resume-safe)
    summary.json                # per-population + per-(model,attr) aggregates
    table.txt                   # pretty stdout dump

Usage:
    uv run python -m advsteer.defense.rubric_classifier_baseline
    uv run python -m advsteer.defense.rubric_classifier_baseline --n_refusal 128 --concurrency 12
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import litellm

from advsteer.classifiers import DEFAULT_JUDGE, PROVIDER_ENV, PROVIDER_PREFIX, _resolve_judge_cfg
from advsteer.defense.stealth_check import STEALTH_PROMPT


_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Population loaders
# ---------------------------------------------------------------------------

@dataclass
class Item:
    population: str            # 'harmful_refusal' | 'harmless_refusal' | 'poisoned_pairs' | 'gcg_baseline'
    text: str
    meta: Dict                  # population-specific (e.g. model/attr/seed/polarity/index)

    @property
    def uid(self) -> str:
        h = hashlib.sha1()
        h.update(self.population.encode())
        h.update(b"\x00")
        h.update(json.dumps(self.meta, sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\x00")
        h.update(self.text.encode("utf-8", errors="replace"))
        return h.hexdigest()


def load_refusal_split(path: Path, n: int, population: str) -> List[Item]:
    with open(path) as f:
        rows = json.load(f)
    items: List[Item] = []
    for i, row in enumerate(rows[:n]):
        text = row.get("instruction") or row.get("prompt")
        if not isinstance(text, str):
            continue
        items.append(Item(
            population=population,
            text=text,
            meta={"source": str(path.relative_to(_ROOT)) if path.is_relative_to(_ROOT) else str(path),
                  "index": i, "category": row.get("category")},
        ))
    return items


def load_poisoned_pairs(summary_paths: Iterable[Path]) -> List[Item]:
    items: List[Item] = []
    for sp in summary_paths:
        with open(sp) as f:
            s = json.load(f)
        cfg = s["config"]
        model = (cfg.get("model") or "?").split("/")[-1]
        attr = cfg.get("pair_type", "?")
        seed = cfg.get("seed", "?")
        for polarity, key in (("pos", "final_pos_texts"), ("neg", "final_neg_texts")):
            for i, text in enumerate(s.get(key, [])):
                items.append(Item(
                    population="poisoned_pairs",
                    text=text,
                    meta={"summary_path": str(sp.relative_to(_ROOT)),
                          "model": model, "attribute": attr, "seed": seed,
                          "polarity": polarity, "index": i},
                ))
    return items


def load_clean_pairs(summary_paths: Iterable[Path]) -> List[Item]:
    """Clean (pre-poisoning) pair texts. The 'honest dataset' baseline a downstream
    user would have if no attacker were involved. Dedupes across seeds — originals
    are identical for the same (model, attribute), and the classifier is deterministic
    at T=0, so per-seed copies would waste API budget."""
    items: List[Item] = []
    seen: set = set()
    for sp in summary_paths:
        with open(sp) as f:
            s = json.load(f)
        cfg = s["config"]
        model = (cfg.get("model") or "?").split("/")[-1]
        attr = cfg.get("pair_type", "?")
        for polarity, key in (("pos", "original_pos_texts"), ("neg", "original_neg_texts")):
            for i, text in enumerate(s.get(key, [])):
                k = (model, attr, polarity, i, text)
                if k in seen:
                    continue
                seen.add(k)
                items.append(Item(
                    population="clean_pairs",
                    text=text,
                    meta={"model": model, "attribute": attr,
                          "polarity": polarity, "index": i},
                ))
    return items


def load_gcg_baseline(path: Optional[Path]) -> List[Item]:
    """Hook for a GCG adversarial-suffix baseline. Expects JSONL with `{prompt: str}`
    or `{instruction: str}`. Off by default; left in place so plugging a comparator
    in later is one flag (`--include_gcg <path>`)."""
    if path is None:
        return []
    items: List[Item] = []
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("prompt") or row.get("instruction") or row.get("text")
            if not isinstance(text, str):
                continue
            items.append(Item(
                population="gcg_baseline",
                text=text,
                meta={"source": str(path), "index": ln, **{k: v for k, v in row.items() if k not in {"prompt", "instruction", "text"}}},
            ))
    return items


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_one(text: str, model: str, api_key: str, max_retries: int = 5) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Returns (label, raw_response, skip_reason). label: 1=flagged, 0=clean, None=skipped."""
    messages = [{"role": "user", "content": STEALTH_PROMPT.format(text=text)}]
    last_err = None
    for attempt in range(max_retries):
        try:
            output = litellm.completion(
                model=model,
                messages=messages,
                api_key=api_key,
                temperature=0,
                max_tokens=10,
            )
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        choice = output["choices"][0]
        if choice.get("finish_reason") == "refusal":
            return None, None, "judge_refusal"
        content = choice["message"].content
        if not content or not content.strip():
            last_err = f"empty_content (finish_reason={choice.get('finish_reason')!r})"
            time.sleep(2 ** attempt)
            continue
        label = int(content.strip().lower().startswith("yes"))
        return label, content.strip(), None
    reason = "empty_content" if isinstance(last_err, str) and last_err.startswith("empty_content") \
        else f"api_error:{last_err}"
    return None, None, reason


# ---------------------------------------------------------------------------
# Checkpoint-aware orchestrator
# ---------------------------------------------------------------------------

def load_done_uids(jsonl_path: Path) -> Dict[str, dict]:
    if not jsonl_path.exists():
        return {}
    done: Dict[str, dict] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = row.get("uid")
            if uid:
                done[uid] = row
    return done


def classify_items(items: List[Item], model: str, api_key: str, jsonl_path: Path,
                   concurrency: int, log_every: int = 50) -> List[dict]:
    """Classify items with thread-pool concurrency, appending to a JSONL checkpoint.

    Returns the list of all per-item rows (whether newly classified this run or
    loaded from the checkpoint)."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_uids(jsonl_path)
    todo = [it for it in items if it.uid not in done]
    if done:
        print(f"  [resume] {len(done)} already classified, {len(todo)} remaining.")
    if not todo:
        return list(done.values())

    fh = open(jsonl_path, "a")
    new_rows: List[dict] = []
    completed = 0
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(classify_one, it.text, model, api_key): it for it in todo}
            for fut in as_completed(futures):
                it = futures[fut]
                label, raw, reason = fut.result()
                row = {
                    "uid": it.uid,
                    "population": it.population,
                    "meta": it.meta,
                    "text": it.text,
                    "label": label,            # 1 flagged, 0 clean, None skipped
                    "raw_response": raw,
                    "skip_reason": reason,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                new_rows.append(row)
                completed += 1
                if completed % log_every == 0 or completed == len(todo):
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(todo) - completed) / rate if rate > 0 else float("inf")
                    print(f"  [{completed:>5}/{len(todo)}] elapsed={elapsed:6.1f}s  "
                          f"rate={rate:5.2f}/s  eta={eta:6.1f}s", flush=True)
    finally:
        fh.close()
    return list(done.values()) + new_rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(rows: List[dict]) -> dict:
    out: Dict[str, dict] = {}

    by_pop: Dict[str, List[dict]] = {}
    for r in rows:
        by_pop.setdefault(r["population"], []).append(r)

    for pop, rs in by_pop.items():
        labels = [r["label"] for r in rs if r["label"] is not None]
        skipped = sum(1 for r in rs if r["label"] is None)
        flagged = sum(labels)
        out[pop] = {
            "n": len(rs),
            "n_classified": len(labels),
            "n_skipped": skipped,
            "n_flagged": flagged,
            "flag_rate": (flagged / len(labels)) if labels else None,
        }

    # Per-(model, attribute) breakdown for clean + poisoned pairs.
    for pop_key, out_key in (("clean_pairs", "clean_per_cell"), ("poisoned_pairs", "poisoned_per_cell")):
        per_cell: Dict[str, Dict[str, dict]] = {}
        for r in by_pop.get(pop_key, []):
            m = r["meta"].get("model", "?")
            a = r["meta"].get("attribute", "?")
            per_cell.setdefault(m, {}).setdefault(a, {"labels": [], "skipped": 0})
            if r["label"] is None:
                per_cell[m][a]["skipped"] += 1
            else:
                per_cell[m][a]["labels"].append(r["label"])
        out[out_key] = {
            m: {
                a: {
                    "n_classified": len(d["labels"]),
                    "n_skipped": d["skipped"],
                    "n_flagged": int(sum(d["labels"])),
                    "flag_rate": (sum(d["labels"]) / len(d["labels"])) if d["labels"] else None,
                }
                for a, d in attrs.items()
            }
            for m, attrs in per_cell.items()
        }
    return out


def print_table(agg: dict, table_path: Optional[Path] = None):
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("Rubric-classifier baseline — flag rates by population")
    lines.append("=" * 78)
    lines.append(f"{'population':<22}  {'n':>5}  {'classified':>10}  {'flagged':>8}  {'flag_rate':>9}")
    for pop in ("harmful_refusal", "harmless_refusal", "clean_pairs", "poisoned_pairs", "gcg_baseline"):
        s = agg.get(pop)
        if s is None or s["n"] == 0:
            continue
        fr = f"{s['flag_rate']:.3f}" if s["flag_rate"] is not None else "  n/a"
        lines.append(f"{pop:<22}  {s['n']:>5d}  {s['n_classified']:>10d}  {s['n_flagged']:>8d}  {fr:>9}")

    clean_per_cell = agg.get("clean_per_cell", {})
    pois_per_cell = agg.get("poisoned_per_cell", {})
    if clean_per_cell or pois_per_cell:
        lines.append("")
        lines.append("Per-(model, attribute) flag rate — clean vs poisoned pairs")
        lines.append("-" * 78)
        lines.append(f"{'model':<22}  {'attribute':<18}  {'clean':>14}  {'poisoned':>14}")
        models = sorted(set(list(clean_per_cell) + list(pois_per_cell)))
        for m in models:
            attrs = sorted(set(list(clean_per_cell.get(m, {})) + list(pois_per_cell.get(m, {}))))
            for a in attrs:
                c = clean_per_cell.get(m, {}).get(a)
                p = pois_per_cell.get(m, {}).get(a)
                c_s = f"{c['n_flagged']}/{c['n_classified']} ({c['flag_rate']:.3f})" if c and c["flag_rate"] is not None else "          n/a"
                p_s = f"{p['n_flagged']}/{p['n_classified']} ({p['flag_rate']:.3f})" if p and p["flag_rate"] is not None else "          n/a"
                lines.append(f"{m:<22}  {a:<18}  {c_s:>14}  {p_s:>14}")
    out = "\n".join(lines)
    print(out)
    if table_path is not None:
        table_path.write_text(out + "\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_refusal", type=int, default=128,
                    help="N harmful + N harmless prompts from the refusal-direction train split.")
    ap.add_argument("--summaries", nargs="*", default=None,
                    help="Specific summary.json files. Default: auto-discover results/*/*/seed*/summary.json")
    ap.add_argument("--include_gcg", type=str, default=None,
                    help="Optional JSONL with GCG-style adversarial prompts to include as a baseline.")
    ap.add_argument("--judge_provider", default=DEFAULT_JUDGE["provider"])
    ap.add_argument("--judge_model", default=DEFAULT_JUDGE["model"])
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--output_dir", default=str(_ROOT / "results" / "rubric_classifier_baseline"))
    ap.add_argument("--limit_poisoned", type=int, default=None,
                    help="(debug) cap the number of poisoned-pair texts classified.")
    return ap.parse_args()


def main():
    args = parse_args()

    cfg = _resolve_judge_cfg({"provider": args.judge_provider, "model": args.judge_model})
    provider = cfg["provider"]
    api_key = os.environ.get(PROVIDER_ENV[provider])
    assert api_key, f"{PROVIDER_ENV[provider]} must be set."
    judge_model = PROVIDER_PREFIX[provider] + cfg["model"]
    print(f"Judge: {judge_model}")
    print(f"Concurrency: {args.concurrency}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "raw_classifications.jsonl"
    summary_path = out_dir / "summary.json"
    table_path = out_dir / "table.txt"

    # ── Load populations ─────────────────────────────────────────────────
    harmful_path = _ROOT / "data" / "refusal" / "splits" / "harmful_train.json"
    harmless_path = _ROOT / "data" / "refusal" / "splits" / "harmless_train.json"
    harmful_items = load_refusal_split(harmful_path, args.n_refusal, "harmful_refusal")
    harmless_items = load_refusal_split(harmless_path, args.n_refusal, "harmless_refusal")
    print(f"Loaded {len(harmful_items)} harmful + {len(harmless_items)} harmless refusal-direction prompts.")

    if args.summaries:
        summary_paths = [Path(p) for p in args.summaries]
    else:
        summary_paths = [Path(p) for p in sorted(glob.glob(str(_ROOT / "results" / "*" / "*" / "seed*" / "summary.json")))]
        # Fall back to the older 2-level layout (matches stealth_check.py).
        if not summary_paths:
            summary_paths = [Path(p) for p in sorted(glob.glob(str(_ROOT / "results" / "*" / "*" / "summary.json")))]
    if not summary_paths:
        print("WARNING: no summary.json found — poisoned_pairs will be empty.", file=sys.stderr)
    poisoned_items = load_poisoned_pairs(summary_paths)
    if args.limit_poisoned is not None:
        poisoned_items = poisoned_items[: args.limit_poisoned]
    print(f"Loaded {len(poisoned_items)} poisoned pair texts from {len(summary_paths)} summary.json files.")

    clean_items = load_clean_pairs(summary_paths)
    print(f"Loaded {len(clean_items)} clean (pre-attack) pair texts (deduped across seeds).")

    gcg_items = load_gcg_baseline(Path(args.include_gcg) if args.include_gcg else None)
    if gcg_items:
        print(f"Loaded {len(gcg_items)} GCG-baseline texts.")

    all_items = harmful_items + harmless_items + clean_items + poisoned_items + gcg_items
    print(f"Total items to classify: {len(all_items)} (checkpoint at {jsonl_path})")

    # ── Classify (resume-safe) ───────────────────────────────────────────
    rows = classify_items(all_items, judge_model, api_key, jsonl_path, args.concurrency)

    # ── Aggregate + write ────────────────────────────────────────────────
    agg = aggregate(rows)
    summary = {
        "judge": judge_model,
        "n_refusal": args.n_refusal,
        "summaries_scanned": [str(p) for p in summary_paths],
        "include_gcg": args.include_gcg,
        "aggregate": agg,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {summary_path}")
    print_table(agg, table_path=table_path)
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
