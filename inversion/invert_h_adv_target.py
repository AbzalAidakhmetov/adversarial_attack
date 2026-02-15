#!/usr/bin/env python3
"""
Invert a provided hidden target vector (h_adv_target) into a short token sequence
at a chosen hidden layer using a single Straight-Through Gumbel-Softmax method,
and log results similarly to inversion/build_adv_no_comma.py.

Features:
- ST-Gumbel with vocab mask (no specials), round-trip loss on hard sequence,
  small LM prior, bounded epsilon repair at the embedding input, and EOT.
- Comprehensive grid search over token count and all core hyperparameters.
- Partial JSON logging after each run and final summary with the best result.

Notes on hidden_states indexing:
- HF models return hidden_states[0] as embeddings; the resid_post for layer L
  aligns with hidden_states[L+1]. This script accepts --layer (REFUSAL_LAYER, no +1)
  and internally uses layer_idx = layer + 1 when selecting hidden_states.
"""

import os
import json
import argparse
import random
from time import time
from typing import List, Dict, Any, Optional
import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_h_adv_target_from_path(path: str, device_: str) -> torch.Tensor:
    p = os.path.expanduser(path)
    ext = os.path.splitext(p)[1].lower()
    if ext in [".pt", ".pth", ".bin"]:
        obj = torch.load(p, map_location=device_)
        if isinstance(obj, torch.Tensor):
            t = obj.to(torch.float32).to(device_)
        elif isinstance(obj, dict):
            t = None
            for key in ["h_adv_target", "target", "vec", "vector"]:
                val = obj.get(key, None)
                if isinstance(val, torch.Tensor):
                    t = val.to(torch.float32).to(device_)
                    break
                if isinstance(val, (list, tuple)):
                    t = torch.tensor(val, dtype=torch.float32, device=device_)
                    break
            if t is None:
                raise ValueError(
                    "Could not find tensor in checkpoint dict; expected keys: h_adv_target/target/vec/vector"
                )
        elif isinstance(obj, (list, tuple)):
            t = torch.tensor(obj, dtype=torch.float32, device=device_)
        else:
            raise ValueError(f"Unsupported object type in {p}: {type(obj)}")
        return t
    elif ext in [".npy"]:
        import numpy as np  # local import to avoid hard dependency otherwise
        arr = np.load(p)
        return torch.from_numpy(arr).to(torch.float32).to(device_)
    elif ext in [".json", ".jsn", ".txt"]:
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception:
            with open(p, "r") as f:
                txt = f.read()
            nums = [float(x) for x in txt.replace(",", " ").split() if x.strip()]
            data = nums
        if isinstance(data, dict):
            for key in ["h_adv_target", "target", "vec", "vector"]:
                if key in data and isinstance(data[key], (list, tuple)):
                    data = data[key]
                    break
        if not isinstance(data, (list, tuple)):
            raise ValueError(
                "JSON/TXT file must contain a list of numbers or a dict with key h_adv_target/target/vec/vector"
            )
        return torch.tensor(data, dtype=torch.float32, device=device_)
    else:
        raise ValueError(f"Unsupported file extension for h_adv_target_path: {ext}")


def invert_soft_prompt_up_to_discretization(*args, **kwargs):
    raise NotImplementedError("Soft prompt inversion has been removed in favor of ST-Gumbel.")


def round_soft_prompt_to_tokens(*args, **kwargs):
    raise NotImplementedError("Soft prompt rounding is not supported; use ST-Gumbel outputs.")


@torch.no_grad()
def _lm_neglogp(model, ids: torch.Tensor) -> torch.Tensor:
    if ids.numel() < 2:
        return torch.tensor(0.0, device=ids.device)
    out = model(input_ids=ids.unsqueeze(0), use_cache=False)
    logits = out.logits[0, :-1, :]
    tgt = ids[1:]
    return F.cross_entropy(logits, tgt)


def invert_to_tokens_via_gumbel_st(
    model,
    tokenizer,
    layer_idx: int,
    h_target: torch.Tensor,
    n_tokens: int = 8,
    lr: float = 0.05,
    max_iters: int = 1500,
    tau_start: float = 2.0,
    tau_end: float = 0.08,
    entropy_w: float = 1e-3,
    roundtrip_w: float = 0.25,
    lm_w: float = 0.05,
    eps_budget: float = 0.5,
    eot_samples: int = 2,
    seed_ids: Optional[List[int]] = None,
    seed: int = 0,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    model.eval()

    M = model.get_input_embeddings().weight.detach()
    V, d = M.shape
    h_target = h_target.to(device).flatten()
    assert h_target.numel() == d

    specials = set(getattr(tokenizer, "all_special_ids", []) or [])
    mask = torch.zeros(V, device=device)
    if len(specials) > 0:
        mask[list(specials)] = -1e9

    logits = torch.zeros(n_tokens, V, device=device, requires_grad=True)
    if seed_ids is not None and len(seed_ids) == n_tokens:
        with torch.no_grad():
            logits.data += -8.0
            logits.data[torch.arange(n_tokens, device=device), torch.tensor(seed_ids, device=device)] = 8.0

    eps = torch.zeros(1, n_tokens, d, device=device, requires_grad=True)
    opt = torch.optim.AdamW([logits, eps], lr=lr, betas=(0.9, 0.98), weight_decay=0.0)

    best = {"score": float("inf"), "ids": None, "cos": -1.0}
    losses: List[float] = []

    def _tau_schedule_st(it: int, T: int, tau_start_: float, tau_end_: float) -> float:
        t = it / max(1, T)
        # cosine schedule, warm->cool
        return float(tau_end_ + 0.5 * (tau_start_ - tau_end_) * (1 + torch.cos(torch.tensor(t * 3.1415926535))))

    pbar = tqdm(total=max_iters, desc=f"invert-gumbel-st T={n_tokens}, lr={lr}", leave=False)
    for it in range(1, max_iters + 1):
        tau = _tau_schedule_st(it, max_iters, tau_start, tau_end)
        rt_w = roundtrip_w * (1.0 - (tau - tau_end) / max(1e-8, (tau_start - tau_end)))

        total_loss = 0.0
        total_main = 0.0
        total_ent = 0.0
        total_rt = 0.0
        total_lm = 0.0

        for _ in range(max(1, int(eot_samples))):
            g = -torch.log(-torch.log(torch.rand_like(logits)))
            probs = torch.softmax((logits + g + mask) / tau, dim=-1)

            hard_ids = probs.argmax(dim=-1)
            y_hard = F.one_hot(hard_ids, num_classes=V).type_as(probs)
            y = (y_hard - probs).detach() + probs

            soft_E = y @ M
            inp_soft = (soft_E.unsqueeze(0) + eps)
            out = model(inputs_embeds=inp_soft, output_hidden_states=True, use_cache=False)
            h_pos = out.hidden_states[layer_idx][0, -1, :]
            loss_main = F.mse_loss(h_pos, h_target)

            ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()

            hard_E = M[hard_ids]
            inp_hard = (hard_E.unsqueeze(0) + eps).detach()
            hard_out = model(inputs_embeds=inp_hard, output_hidden_states=True, use_cache=False)
            hard_hpos = hard_out.hidden_states[layer_idx][0, -1, :]
            loss_rt = F.mse_loss(hard_hpos, h_target)

            loss_lm = _lm_neglogp(model, hard_ids)

            total_main += loss_main
            total_ent  += ent
            total_rt   += loss_rt
            total_lm   += loss_lm

        e = float(max(1, int(eot_samples)))
        loss_eps_bound = 1e-3 * torch.relu(eps.norm(p=2) - eps_budget)
        loss = (total_main/e) + entropy_w*(total_ent/e) + rt_w*(total_rt/e) + lm_w*(total_lm/e) + loss_eps_bound

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        cur = float(loss.item())
        losses.append(cur)

        score = float((total_main/e + rt_w*(total_rt/e)).item())
        with torch.no_grad():
            ids_now = probs.argmax(dim=-1)
            cos = F.cosine_similarity(h_pos, h_target, dim=-1).item()
            if score < best["score"]:
                best.update(score=score, ids=ids_now.detach().cpu(), cos=cos)

        pbar.set_postfix({"loss": f"{cur:.3e}", "score": f"{score:.3e}", "tau": f"{tau:.2f}", "cos": f"{best['cos']:.3f}"})
        pbar.update(1)
    pbar.close()

    ids = best["ids"].tolist() if best["ids"] is not None else []
    text = tokenizer.decode(ids, skip_special_tokens=True) if ids else ""
    return {
        "losses": losses,
        "best_score": best["score"],
        "best_loss": best["score"],
        "best_ids": ids,
        "text": text,
        "cosine": best["cos"],
        "eps_norm": float(min(eps.norm().item(), eps_budget) if eps is not None else 0.0),
    }


def compute_last_token_embedding_all_grad_emb(*args, **kwargs):
    raise NotImplementedError("Snap-to-vocab helper removed; use ST-Gumbel.")


def invert_snap_to_vocab(*args, **kwargs):
    raise NotImplementedError("Snap-to-vocab inversion has been removed in favor of ST-Gumbel.")


def parse_args():
    ap = argparse.ArgumentParser(description="Invert a provided h_adv_target into a discrete token sequence via ST-Gumbel (HF-only)")
    ap.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=9, help="REFUSAL_LAYER (no +1)")
    ap.add_argument("--h_adv_target_path", type=str, required=True, help="Path to precomputed hidden target vector")

    # Token count grid
    ap.add_argument("--token_counts", type=int, nargs="*", default=None, help="Explicit token counts to try (overrides range)")
    ap.add_argument("--token_min", type=int, default=1, help="Minimum token count (inclusive) if --token_counts not set")
    ap.add_argument("--token_max", type=int, default=8, help="Maximum token count (inclusive) if --token_counts not set")
    ap.add_argument("--token_stride", type=int, default=1, help="Stride for token count range if --token_counts not set")

    # Hyperparameter grids (each accepts multiple values)
    ap.add_argument("--lrs", type=float, nargs="*", default=[0.05])
    ap.add_argument("--tau_starts", type=float, nargs="*", default=[2.0])
    ap.add_argument("--tau_ends", type=float, nargs="*", default=[0.08])
    ap.add_argument("--entropy_ws", type=float, nargs="*", default=[1e-3])
    ap.add_argument("--roundtrip_ws", type=float, nargs="*", default=[0.25])
    ap.add_argument("--lm_ws", type=float, nargs="*", default=[0.05])
    ap.add_argument("--eps_budgets", type=float, nargs="*", default=[0.5])
    ap.add_argument("--eot_samples_list", type=int, nargs="*", default=[2])

    ap.add_argument("--max_iters", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default="experiments/invert_h_adv/summary.json")
    return ap.parse_args()


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map=device)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True

    print(f"Loading h_adv_target from {args.h_adv_target_path}")
    h_adv_target = _load_h_adv_target_from_path(args.h_adv_target_path, device)
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is not None:
        if h_adv_target.dim() != 1 or h_adv_target.numel() != int(hidden_size):
            raise ValueError(
                f"h_adv_target shape {tuple(h_adv_target.shape)} does not match model hidden_size {hidden_size}"
            )

    out_dir = os.path.dirname(os.path.expanduser(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.token_counts and len(args.token_counts) > 0:
        token_grid = sorted(set([int(x) for x in args.token_counts if int(x) > 0]))
    else:
        token_min = max(1, int(args.token_min))
        token_max = max(token_min, int(args.token_max))
        stride = max(1, int(args.token_stride))
        token_grid = list(range(token_min, token_max + 1, stride))

    results: List[Dict[str, Any]] = []
    tmp_path = os.path.join(out_dir, "partial_invert_results.json")

    hs_index = args.layer + 1

    best_overall = None

    grid = itertools.product(
        token_grid,
        args.lrs,
        args.tau_starts,
        args.tau_ends,
        args.entropy_ws,
        args.roundtrip_ws,
        args.lm_ws,
        args.eps_budgets,
        args.eot_samples_list,
    )

    for (
        n_tokens,
        lr,
        tau_start,
        tau_end,
        entropy_w,
        roundtrip_w,
        lm_w,
        eps_budget,
        eot_samples,
    ) in grid:
        cfg = {
            "n_tokens": int(n_tokens),
            "lr": float(lr),
            "tau_start": float(tau_start),
            "tau_end": float(tau_end),
            "entropy_w": float(entropy_w),
            "roundtrip_w": float(roundtrip_w),
            "lm_w": float(lm_w),
            "eps_budget": float(eps_budget),
            "eot_samples": int(eot_samples),
            "layer": int(args.layer),
        }
        print(f"\nRunning ST-Gumbel with {cfg} ...")
        start = time()
        res = invert_to_tokens_via_gumbel_st(
            model=model,
            tokenizer=tokenizer,
            layer_idx=hs_index,
            h_target=h_adv_target.to(device),
            n_tokens=n_tokens,
            lr=lr,
            max_iters=args.max_iters,
            tau_start=tau_start,
            tau_end=tau_end,
            entropy_w=entropy_w,
            roundtrip_w=roundtrip_w,
            lm_w=lm_w,
            eps_budget=eps_budget,
            eot_samples=eot_samples,
            seed=args.seed,
        )
        run = {
            **cfg,
            "ok": True,
            "time": time() - start,
            "best_loss": float(res.get("best_loss", float("inf"))),
            "best_score": float(res.get("best_score", float("inf"))),
            "ids": res.get("best_ids"),
            "text": res.get("text"),
            "cosine": res.get("cosine"),
            "eps_norm": res.get("eps_norm"),
        }

        results.append(run)
        save_json(tmp_path, {"config": vars(args), "results": results})

        if run.get("ok", True):
            if best_overall is None or run["best_loss"] < best_overall["best_loss"]:
                best_overall = run

    print("\nGrid search completed. Best overall result:")
    if best_overall:
        print(
            f"  Tokens: {best_overall['n_tokens']} | LR: {best_overall['lr']} | Loss: {best_overall['best_loss']:.2e} | Cos: {best_overall.get('cosine', float('nan')):.3f}"
        )
        if best_overall.get("text"):
            print(f"  Text: {repr(best_overall['text'])}")
    else:
        print("  No successful inversions found")

    summary = {
        "config": vars(args),
        "target_norm": float(h_adv_target.norm().item()),
        "hidden_size": int(hidden_size) if hidden_size is not None else None,
        "best_overall": best_overall,
        "results": results,
    }
    save_json(os.path.expanduser(args.output), summary)
    print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()


