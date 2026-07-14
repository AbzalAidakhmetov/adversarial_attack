"""Real-world case study: run one poisoned steering
dataset end-to-end through a *public* steering-vector pipeline — ``repeng``
(github.com/vgel/repeng).

This grounds the attack outside the paper's own code. ``repeng`` is a genuinely
different construction from Section 3: it fits a **per-layer PCA** direction over
a **band** of layers (default ``range(-5,-18,-1)``) and adds it at **every** token
position on **every** decode step (whereas the paper uses a single-layer
mean-difference added on the prefill pass only). We feed it, unmodified, the
attacker-shipped poisoned pair texts and measure, for {no control, clean vector,
poisoned vector} across a coeff sweep:

  * ASR  — jailbreak rate on the harmful prompts, judged by the configured LLM
           judge (default Together / Llama-3.3-70B, one of the paper's three;
           this PoC is single-judge by design — the paper's 3-judge majority is
           deferred to the revision).
  * AC   — the declared attribute still fires on harmless prompts (stealth): the
           poisoned repeng vector does its advertised job too.
  * perplexity — GPT-2 mean perplexity + low-perplexity fraction of the jailbreak
           responses, so an over-steered/degenerate column is visible, not hidden.

This is an *existence proof* (one combo, one seed): a clear poisoned>>clean gap
shows the payload survives a real public pipeline. The clean-vector column at the
same coefficient is the honest-bundle counterfactual — the ASR gap between the
poisoned and clean columns is the jailbreak the poisoned *texts* smuggle through
repeng.

All settings live in ``config/repeng_poc.yaml`` (Hydra); nothing is hardcoded here.
Harmful (ASR) and harmless (AC) evaluation use the SAME prompt files and the SAME
``val_samples`` as the canonical eval (``config/evaluate_jailbreak.yaml`` /
``config/matrix.yaml``), so the numbers are directly comparable to the main results.
Scoring uses the paper's own evaluators — ``evaluate_jailbreak`` (judge ASR +
per-sample ``is_jailbreak`` + a skipped-samples audit JSONL), ``evaluate_perplexity``,
and ``evaluate_steering`` from ``advsteer`` — the same functions ``advsteer.eval.
evaluate_asr`` calls; only *generation* is repeng-specific (its ControlModel).
Output follows ``evaluate_asr``'s layout: for each (condition, split, weight) a
``results_{baseline,clean,poisoned}_{harmful,harmless}_w<W>/`` dir holds a canonical
``results`` list (the same record schema) and a ``refusal_<eval_source>_eval_L*_w*.json``.
ASR is ``judge_success_rate`` on the harmful split; hAttr is ``steering_success_rate``
on the harmless split. A roll-up ``summary.json`` collects the per-condition rows +
a few sample completions.

Run (background; ~30 min on a 3090):
  HF_HOME=/workspace/.hf_home PROJECT_ROOT=$(pwd) \
  .venv/bin/python scripts/repeng_poc.py                 # defaults: has_bold_only, seed 2
  # override anything via Hydra, e.g.:
  .venv/bin/python scripts/repeng_poc.py combo=spanish seed=0 coeffs=[16,20] val_samples=50
"""

from __future__ import annotations

import json
import logging
from importlib.metadata import version
from pathlib import Path

import hydra
import litellm
import omegaconf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel, DatasetEntry

# The judge fires ~150 calls/condition; litellm's per-call INFO logging drowns the
# run's own progress. Quiet it (labels/ASR/AC still print).
litellm.suppress_debug_info = True
for _n in ("LiteLLM", "litellm", "httpx"):
    logging.getLogger(_n).setLevel(logging.WARNING)

from advsteer import PROJECT_ROOT
from advsteer.data import load_texts_from_json
from advsteer.steering import to_chat, evaluate_steering
from advsteer.classifiers import evaluate_jailbreak, evaluate_perplexity


@torch.no_grad()
def generate(cmodel, tok, prompts, control, coeff, max_new_tokens, batch_size):
    """Batched greedy generation through repeng's ControlModel (control on/off).

    Uses repeng's own control cycle (repeng/tests.py): set_control(vector, coeff) ->
    ControlModel.generate -> reset, greedy (do_sample=False), pad_token_id=eos. Two
    deliberate departures from repeng's single-prompt example: we batch with
    left-padding (repeng's ControlModule masks padded positions, so per-prompt output
    is unchanged), and we omit its example `repetition_penalty=1.1` to match advsteer's
    own eval (steering.generate_steered_batched) — so ASR/AC are decoded exactly like
    the paper's main results. The control *mechanism* is repeng's, untouched.
    """
    cmodel.reset() if control is None else cmodel.set_control(control, coeff)
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok([to_chat(tok, p) for p in chunk], return_tensors="pt",
                  padding=True, add_special_tokens=True).to(cmodel.device)
        gen = cmodel.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                              max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tok.eos_token_id)
        outs += [tok.decode(gen[j, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                 for j in range(len(chunk))]
    cmodel.reset()
    return outs


def eval_split(cmodel, tok, cond, control, coeff, prompts, split, attribute,
               judge_cfg, eval_methods, layers, eval_source, out_dir, max_new_tokens, batch_size):
    """One (condition, split, weight) eval, laid out exactly like advsteer.eval.evaluate_asr.

    Writes ``results_{cond}_{split}_w<W>/`` containing a canonical ``results`` list
    (one record, same schema evaluate_configs emits) and the per-sample
    ``refusal_<eval_source>_eval_L*_w*.json`` from ``evaluate_jailbreak``. Judge runs
    only when ``eval_methods`` includes 'judge' (harmful split); the harmless split
    passes ``[]`` so ``judge_success_rate`` is None and AC carries the signal.
    """
    resp = generate(cmodel, tok, prompts, control, coeff, max_new_tokens, batch_size)
    completions = [{"prompt": p, "response": r} for p, r in zip(prompts, resp)]
    rdir = Path(out_dir) / f"results_{cond}_{split}_w{int(coeff)}"
    rdir.mkdir(parents=True, exist_ok=True)
    layer_tag = f"{layers[0]}_{layers[-1]}"
    jb = evaluate_jailbreak(
        completions=completions, methodologies=list(eval_methods),
        evaluation_path=str(rdir / f"refusal_{eval_source}_eval_L{layer_tag}_w{int(coeff)}.json"),
        device=cmodel.device.type, judge=judge_cfg)
    ppl = evaluate_perplexity(completions)
    ac = evaluate_steering(completions, attribute)
    jsr = jb.get("judge_success_rate")
    rec = {
        "layer": list(layers), "weight": float(coeff),
        "judge_success_rate": (float(jsr) if jsr is not None else None),
        "judge_skipped_count": int(jb.get("judge_skipped_count", 0)),
        "judge_n_classified": int(jb.get("judge_n_classified", 0)),
        "steering_success_rate": ac,
        "mean_perplexity": float(ppl["mean_perplexity"]),
        "low_perplexity_fraction": float(ppl["low_perplexity_fraction"]),
    }
    with open(rdir / "results", "w") as f:
        json.dump([rec], f, indent=4)
    return rec, completions


def run(cfg: omegaconf.DictConfig) -> None:
    attribute = cfg.attribute or cfg.combo
    coeffs = [float(c) for c in cfg.coeffs]
    layers = list(range(*cfg.layer_band))
    judge_cfg = omegaconf.OmegaConf.to_container(cfg.judge, resolve=True)
    summary_path = cfg.summary or cfg.summary_default
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_out = out_dir / "summary.json"

    if not Path(summary_path).exists():
        raise SystemExit(
            f"Attack summary not found: {summary_path}\n"
            f"Produce the poisoned pair texts first (e.g. scripts/run_cross_transfer.py or "
            f"scripts/run_matrix.py for this combo/seed), or set summary=<path>.")
    d = json.load(open(summary_path))
    print(f"combo={cfg.combo} seed={cfg.seed}  cos_clean={d.get('cos_clean'):+.3f} "
          f"cos_poisoned={d.get('cos_poisoned'):+.3f}  layers={layers}")

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=torch.bfloat16, device_map=cfg.device).eval()
    cmodel = ControlModel(model, layers)

    def make_ds(pos, neg):
        return [DatasetEntry(positive=to_chat(tok, p), negative=to_chat(tok, n))
                for p, n in zip(pos, neg)]

    # Clean + poisoned control vectors: the top principal component of the pair-text
    # hidden-state differences over the band. repeng reads each text's last non-pad
    # token (so padding side is irrelevant) and returns unit-norm directions, so its
    # coeffs are a different scale from the paper's mean-diff alpha in {2,3,4} —
    # report them as repeng-native.
    print(f"training poisoned + clean repeng control vectors ({cfg.method})...")
    cv_pois = ControlVector.train(cmodel, tok, make_ds(d["final_pos_texts"], d["final_neg_texts"]),
                                  batch_size=8, method=cfg.method)
    cv_clean = ControlVector.train(cmodel, tok, make_ds(d["original_pos_texts"], d["original_neg_texts"]),
                                   batch_size=8, method=cfg.method)

    # Same prompt files + val_samples for both sides (harmful ASR, harmless AC).
    harmful = load_texts_from_json(cfg.harmful_prompts, cfg.val_samples)
    harmless = load_texts_from_json(cfg.harmless_prompts, cfg.val_samples)

    results = {
        "config": {
            "model": cfg.model, "combo": cfg.combo, "seed": cfg.seed,
            "summary": summary_path, "attribute": attribute, "layers": layers,
            "coeffs": coeffs, "method": cfg.method, "val_samples": cfg.val_samples,
            "n_harmful": len(harmful), "n_harmless": len(harmless),
            "max_new_tokens": cfg.max_new_tokens, "eval_source": cfg.eval_source,
            "judge": f"{judge_cfg['provider']}/{judge_cfg['model']}",
            "judge_protocol": "single-judge existence-proof PoC "
                              "(paper's 3-judge majority deferred to revision)",
            "eval": "canonical evaluate_asr layout: results_{cond}_{harmful,harmless}_w<W>/results",
            "pipeline": f"repeng=={version('repeng')} ControlVector.train(method={cfg.method})",
            "cos_clean": d.get("cos_clean"), "cos_poisoned": d.get("cos_poisoned"),
        },
        "conditions": [],
        "samples": {},
    }

    def run_condition(name, control, coeff):
        print(f"\n--- {name} (coeff={coeff}) ---")
        # Harmful -> ASR (judge). Harmless -> hAttr (AC, no judge). Same split
        # convention advsteer.eval.evaluate_asr / summarize_defense use.
        rec_h, comp_h = eval_split(cmodel, tok, name, control, coeff, harmful, "harmful",
                                   attribute, judge_cfg, ["judge"], layers, cfg.eval_source,
                                   out_dir, cfg.max_new_tokens, cfg.batch_size)
        rec_hl, comp_hl = eval_split(cmodel, tok, name, control, coeff, harmless, "harmless",
                                     attribute, judge_cfg, [], layers, cfg.eval_source,
                                     out_dir, cfg.max_new_tokens, cfg.batch_size)
        asr, ac = rec_h["judge_success_rate"], rec_hl["steering_success_rate"]
        print(f"    ASR={asr if asr is not None else float('nan'):.3f}  hAttr={ac:.3f}  "
              f"ppl={rec_h['mean_perplexity']:.1f}  "
              f"(judged {rec_h['judge_n_classified']}/{len(harmful)}, "
              f"skipped {rec_h['judge_skipped_count']})")
        # Human roll-up (ASR from harmful, hAttr from harmless) — the canonical
        # per-split `results` files above are the source of truth.
        results["conditions"].append({
            "name": name, "coeff": coeff, "asr": asr, "asr_n": rec_h["judge_n_classified"],
            "asr_skipped": rec_h["judge_skipped_count"], "hattr": ac,
            "harmful_mean_perplexity": rec_h["mean_perplexity"],
            "harmless_mean_perplexity": rec_hl["mean_perplexity"],
        })
        results["samples"][f"{name}_w{int(coeff)}"] = {
            "harmful": comp_h[:5],
            "harmless": comp_hl[:3],
        }
        with open(summary_out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    run_condition("baseline", None, 0.0)
    for coeff in coeffs:
        run_condition("clean", cv_clean, coeff)
        run_condition("poisoned", cv_pois, coeff)

    print("\n================ SUMMARY ================")
    print(f"{'condition':<20}{'coeff':>7}{'ASR':>8}{'hAttr':>8}{'ppl':>9}")
    for c in results["conditions"]:
        asr = c["asr"]
        print(f"{c['name']:<20}{c['coeff']:>7.0f}"
              f"{asr if asr is not None else float('nan'):>8.3f}"
              f"{c['hattr']:>8.3f}{c['harmful_mean_perplexity']:>9.1f}")
    print(f"\nwrote {summary_out} + canonical results_*/ dirs under {out_dir}")


@hydra.main(config_path=str(PROJECT_ROOT / "config"), config_name="repeng_poc.yaml", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
