#!/usr/bin/env python3
"""
Visualize the geometry of steering vectors, adversarial hidden states, and refusal direction.

Produces:
  1. PCA scatter: clean pos/neg, adv pos/neg, harmful, harmless hidden states
  2. Vector decomposition bar chart: attribute vs -refusal vs residual components
  3. Per-sample projections onto -refusal: how much does each adv sample contribute?
"""

import os, json, sys
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ──
MODEL_NAME = "google/gemma-2-2b-it"
LAYER = 11
HF_LAYER = LAYER + 1
BATCH_SIZE = 8

EXPERIMENTS = {
    "emoji": "emoji_sufftmpl_k2_dual_t32_s0_20260321_0516",
    "no_comma": "nocomma_sufftmpl_k2_dual_t32_s0_20260321_0516",
    "lowercase": "lowercase_sufftmpl_k2_dual_t32_s0_20260321_0516",
}

_PAIR_TYPE_SPECS = {
    "emoji": {
        "path_parts": ("emoji_pairs.jsonl",),
        "instruction_id": "format:emoji",
        "exact_match": True,
    },
    "no_comma": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "punctuation:no_comma",
        "exact_match": False,
    },
    "lowercase": {
        "path_parts": ("ifeval_augmented_filtered.jsonl",),
        "instruction_id": "change_case:english_lowercase",
        "exact_match": False,
    },
}

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(_root, "data", "pairs")
REFUSAL_DIR = os.path.join(_root, "data", "refusal")
EXP_DIR = os.path.join(_root, "experiments")
OUT_DIR = os.path.join(_root, "experiments", "geometry_viz")


def _extract_ids(result):
    if isinstance(result, list): return result
    if hasattr(result, "input_ids"):
        ids = result.input_ids
        return ids[0] if isinstance(ids[0], list) else ids
    if isinstance(result, dict):
        ids = result["input_ids"]
        return ids[0] if isinstance(ids[0], list) else ids
    return list(result)


def get_hidden_last(model, tokenizer, texts, layer_idx, batch_size=8):
    device = next(model.parameters()).device
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        all_ids = [
            _extract_ids(tokenizer.apply_chat_template(
                [{"role": "user", "content": t}],
                add_generation_prompt=True, tokenize=True))
            for t in chunk
        ]
        max_len = max(len(ids) for ids in all_ids)
        pad_id = tokenizer.pad_token_id
        padded = [[pad_id]*(max_len-len(ids)) + ids for ids in all_ids]
        masks = [[0]*(max_len-len(ids)) + [1]*len(ids) for ids in all_ids]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attn_mask = torch.tensor(masks, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            all_vecs.append(out.hidden_states[layer_idx][:, -1, :].float().cpu())
    return torch.cat(all_vecs, dim=0)


def load_pairs(pair_type, num_pairs=20):
    spec = _PAIR_TYPE_SPECS[pair_type]
    path = os.path.join(DATA_DIR, *spec["path_parts"])
    iid = spec["instruction_id"]
    if spec["exact_match"]:
        filt = lambda r: r.get("single_instruction_id") == iid
    else:
        filt = lambda r: iid in str(r.get("single_instruction_id", ""))
    pos, neg = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if not filt(row): continue
            p, n = row.get("prompt"), row.get("prompt_without_instruction")
            if isinstance(p, str) and isinstance(n, str):
                pos.append(p); neg.append(n)
    return pos[:num_pairs], neg[:num_pairs]


def load_refusal_texts(n=100):
    with open(os.path.join(REFUSAL_DIR, "harmful_prompts.json")) as f:
        harmful = [r["prompt"] for r in json.load(f)][:n]
    with open(os.path.join(REFUSAL_DIR, "harmless_prompts.json")) as f:
        harmless = [r["prompt"] for r in json.load(f)][:n]
    return harmful, harmless


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Collect hidden states ──
    print("Computing refusal hidden states...")
    harmful_texts, harmless_texts = load_refusal_texts(100)
    h_harmful = get_hidden_last(model, tokenizer, harmful_texts, HF_LAYER, BATCH_SIZE)
    h_harmless = get_hidden_last(model, tokenizer, harmless_texts, HF_LAYER, BATCH_SIZE)
    refusal_dir = h_harmful.mean(0) - h_harmless.mean(0)
    neg_ref = -refusal_dir
    neg_ref_unit = neg_ref / neg_ref.norm()

    all_hidden = {}  # {attr: {label: tensor}}

    for attr, exp_name in EXPERIMENTS.items():
        print(f"\nProcessing {attr}...")

        # Load clean pairs
        pos_texts, neg_texts = load_pairs(attr, 20)
        h_pos = get_hidden_last(model, tokenizer, pos_texts, HF_LAYER, BATCH_SIZE)
        h_neg = get_hidden_last(model, tokenizer, neg_texts, HF_LAYER, BATCH_SIZE)

        # Load adversarial texts from summary
        with open(os.path.join(EXP_DIR, exp_name, "summary.json")) as f:
            summary = json.load(f)
        adv_texts = summary["best"]["texts"]
        adv_neg_texts = summary["best"].get("neg_texts", [])

        h_adv_pos = get_hidden_last(model, tokenizer, adv_texts, HF_LAYER, BATCH_SIZE)
        h_adv_neg = get_hidden_last(model, tokenizer, adv_neg_texts, HF_LAYER, BATCH_SIZE) if adv_neg_texts else torch.zeros(0, h_pos.shape[1])

        all_hidden[attr] = {
            "clean_pos": h_pos, "clean_neg": h_neg,
            "adv_pos": h_adv_pos, "adv_neg": h_adv_neg,
        }

        # Per-sample projections onto -refusal
        print(f"  Clean POS projections onto -refusal: {(h_pos @ neg_ref_unit).tolist()[:5]}")
        print(f"  Adv POS projections onto -refusal:   {(h_adv_pos @ neg_ref_unit).tolist()}")
        if h_adv_neg.shape[0] > 0:
            print(f"  Adv NEG projections onto -refusal:   {(h_adv_neg @ neg_ref_unit).tolist()}")

    if device == "cuda":
        torch.cuda.empty_cache()

    # ── PLOT 1: PCA scatter per attribute ──
    print("\nGenerating PCA plots...")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    for idx, (attr, exp_name) in enumerate(EXPERIMENTS.items()):
        ax = axes[idx]
        hd = all_hidden[attr]

        # Fit PCA on all points for this attribute + refusal
        all_pts = torch.cat([
            hd["clean_pos"], hd["clean_neg"],
            hd["adv_pos"], hd["adv_neg"] if hd["adv_neg"].shape[0] > 0 else torch.zeros(0, hd["clean_pos"].shape[1]),
            h_harmful[:50], h_harmless[:50],
        ], dim=0).numpy()

        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_pts)

        n_cp = hd["clean_pos"].shape[0]
        n_cn = hd["clean_neg"].shape[0]
        n_ap = hd["adv_pos"].shape[0]
        n_an = hd["adv_neg"].shape[0]
        n_harm = 50
        n_less = 50

        offset = 0
        cp_2d = all_2d[offset:offset+n_cp]; offset += n_cp
        cn_2d = all_2d[offset:offset+n_cn]; offset += n_cn
        ap_2d = all_2d[offset:offset+n_ap]; offset += n_ap
        an_2d = all_2d[offset:offset+n_an]; offset += n_an
        harmful_2d = all_2d[offset:offset+n_harm]; offset += n_harm
        harmless_2d = all_2d[offset:offset+n_less]; offset += n_less

        ax.scatter(*cp_2d.T, c="dodgerblue", s=40, alpha=0.7, label="Clean POS", zorder=3)
        ax.scatter(*cn_2d.T, c="steelblue", s=40, alpha=0.7, marker="x", label="Clean NEG", zorder=3)
        ax.scatter(*ap_2d.T, c="red", s=120, alpha=0.9, marker="*", label="Adv POS", zorder=5)
        if n_an > 0:
            ax.scatter(*an_2d.T, c="darkred", s=120, alpha=0.9, marker="P", label="Adv NEG", zorder=5)
        ax.scatter(*harmful_2d.T, c="orange", s=15, alpha=0.4, label="Harmful", zorder=2)
        ax.scatter(*harmless_2d.T, c="green", s=15, alpha=0.4, label="Harmless", zorder=2)

        # Project directions into PCA space
        neg_ref_2d = pca.transform(neg_ref_unit.numpy().reshape(1, -1))[0]
        origin = np.mean(all_2d, axis=0)
        scale_arrow = 15
        ax.annotate("", xy=origin + neg_ref_2d * scale_arrow, xytext=origin,
                     arrowprops=dict(arrowstyle="->", color="black", lw=2.5))
        ax.text(*(origin + neg_ref_2d * scale_arrow * 1.1), "-refusal", fontsize=9, fontweight="bold")

        # Clean and poisoned steering vector directions
        sv = torch.load(os.path.join(EXP_DIR, exp_name, "steering_vector.pt"), weights_only=False)
        clean_unit = sv["steering_vector_clean"] / sv["steering_vector_clean"].norm()
        pois_unit = sv["steering_vector_poisoned"] / sv["steering_vector_poisoned"].norm()
        clean_2d = pca.transform(clean_unit.numpy().reshape(1, -1))[0]
        pois_2d = pca.transform(pois_unit.numpy().reshape(1, -1))[0]
        ax.annotate("", xy=origin + clean_2d * scale_arrow, xytext=origin,
                     arrowprops=dict(arrowstyle="->", color="dodgerblue", lw=2))
        ax.text(*(origin + clean_2d * scale_arrow * 1.1), "v_clean", fontsize=8, color="dodgerblue")
        ax.annotate("", xy=origin + pois_2d * scale_arrow, xytext=origin,
                     arrowprops=dict(arrowstyle="->", color="red", lw=2))
        ax.text(*(origin + pois_2d * scale_arrow * 1.1), "v_poisoned", fontsize=8, color="red")

        ax.set_title(f"{attr}\nASR w=2: {['0.22','0.07','0.05'][idx]}, w=4: {['0.76','0.11','0.10'][idx]}", fontsize=12)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    plt.suptitle("PCA of Hidden States at Layer 11 (last token)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_scatter.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved pca_scatter.png")

    # ── PLOT 2: Vector decomposition ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    bar_data = []
    for attr, exp_name in EXPERIMENTS.items():
        sv = torch.load(os.path.join(EXP_DIR, exp_name, "steering_vector.pt"), weights_only=False)
        clean = sv["steering_vector_clean"]
        poisoned = sv["steering_vector_poisoned"]
        clean_unit = clean / clean.norm()

        # Decompose poisoned into: -refusal component, attribute component, residual
        ref_comp = (poisoned @ neg_ref_unit).item()
        attr_comp = (poisoned @ clean_unit).item()

        # Orthogonalize: remove -refusal and attr from poisoned to get residual
        v_no_ref = poisoned - ref_comp * neg_ref_unit
        v_no_ref_no_attr = v_no_ref - (v_no_ref @ clean_unit) * clean_unit
        residual_norm = v_no_ref_no_attr.norm().item()

        bar_data.append({
            "attr": attr,
            "-refusal": ref_comp,
            "attribute": attr_comp,
            "residual": residual_norm,
            "total_norm": poisoned.norm().item(),
        })

    x = np.arange(len(bar_data))
    w = 0.2
    for i, key in enumerate(["-refusal", "attribute", "residual"]):
        vals = [d[key] for d in bar_data]
        colors = ["red", "dodgerblue", "gray"][i]
        ax.bar(x + i*w, vals, w, label=key, color=colors, alpha=0.8)

    ax.set_xticks(x + w)
    ax.set_xticklabels([d["attr"] for d in bar_data])
    ax.set_ylabel("Projection magnitude")
    ax.set_title("Poisoned Steering Vector Decomposition\n(projection onto -refusal, clean attribute, and residual)")
    ax.legend()

    # Add total norm annotation
    for i, d in enumerate(bar_data):
        ax.text(i + w, max(d["-refusal"], d["attribute"], d["residual"]) + 2,
                f"||v||={d['total_norm']:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "vector_decomposition.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved vector_decomposition.png")

    # ── PLOT 3: Per-sample -refusal projections ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (attr, exp_name) in enumerate(EXPERIMENTS.items()):
        ax = axes[idx]
        hd = all_hidden[attr]

        # Projections onto -refusal
        clean_pos_proj = (hd["clean_pos"] @ neg_ref_unit).numpy()
        clean_neg_proj = (hd["clean_neg"] @ neg_ref_unit).numpy()
        adv_pos_proj = (hd["adv_pos"] @ neg_ref_unit).numpy()
        adv_neg_proj = (hd["adv_neg"] @ neg_ref_unit).numpy() if hd["adv_neg"].shape[0] > 0 else np.array([])

        all_pos_proj = np.concatenate([clean_pos_proj, adv_pos_proj])
        all_neg_proj = np.concatenate([clean_neg_proj, adv_neg_proj])

        # Plot as horizontal bars
        y_pos = np.arange(len(all_pos_proj))
        y_neg = np.arange(len(all_neg_proj))

        colors_pos = ["dodgerblue"] * len(clean_pos_proj) + ["red"] * len(adv_pos_proj)
        colors_neg = ["steelblue"] * len(clean_neg_proj) + ["darkred"] * len(adv_neg_proj)

        ax.barh(y_pos, all_pos_proj, color=colors_pos, alpha=0.7, height=0.8, label="POS")
        ax.barh(y_neg + len(all_pos_proj) + 1, all_neg_proj, color=colors_neg, alpha=0.7, height=0.8, label="NEG")

        ax.axvline(x=0, color="black", lw=0.5)
        ax.axhline(y=len(all_pos_proj) + 0.5, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("Projection onto -refusal_unit")
        ax.set_title(f"{attr}")

        # Mark adv samples
        for i in range(len(adv_pos_proj)):
            ax.text(all_pos_proj[len(clean_pos_proj)+i], len(clean_pos_proj)+i,
                    f" ADV", fontsize=7, color="red", va="center")
        for i in range(len(adv_neg_proj)):
            ax.text(all_neg_proj[len(clean_neg_proj)+i],
                    len(clean_neg_proj)+i + len(all_pos_proj) + 1,
                    f" ADV", fontsize=7, color="darkred", va="center")

        ax.set_ylabel("Sample index (POS above, NEG below)")

    plt.suptitle("Per-Sample Projection onto -refusal Direction", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "per_sample_projections.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved per_sample_projections.png")

    # ── PLOT 4: Effective steering at different weights ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    weights = np.arange(0.5, 6.5, 0.5)
    for idx, (attr, exp_name) in enumerate(EXPERIMENTS.items()):
        sv = torch.load(os.path.join(EXP_DIR, exp_name, "steering_vector.pt"), weights_only=False)
        poisoned = sv["steering_vector_poisoned"]
        ref_proj = (poisoned @ neg_ref_unit).item()
        total_norm = poisoned.norm().item()

        axes[0].plot(weights, ref_proj * weights, marker="o", label=attr, markersize=4)
        axes[1].plot(weights, total_norm * weights, marker="o", label=attr, markersize=4)

    axes[0].axhline(y=refusal_dir.norm().item(), color="black", ls="--", label="||refusal_dir||")
    axes[0].set_xlabel("Steering weight")
    axes[0].set_ylabel("Effective -refusal magnitude")
    axes[0].set_title("Anti-refusal component vs weight")
    axes[0].legend()

    axes[1].set_xlabel("Steering weight")
    axes[1].set_ylabel("Total steering magnitude")
    axes[1].set_title("Total steering norm vs weight\n(too high = degenerate output)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "effective_steering.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved effective_steering.png")

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
