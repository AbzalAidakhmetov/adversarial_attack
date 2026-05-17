# scripts/

Driver scripts for the stealth-attack pipeline. The shell scripts orchestrate
`advsteer.attack.build_adv_stealth`, `advsteer.eval.evaluate_asr`, and the
defense/detector helpers; the Python scripts under `plots/` regenerate the
paper figures from the saved JSON outputs.

All shell scripts:

- expect to be launched from the project root (`PROJECT_ROOT=$(pwd)`);
- source `.env` for `HF_TOKEN`, `ANTHROPIC_API_KEY`, etc.;
- invoke Python via `uv run python -m advsteer.<subpkg>.<module>`;
- are restartable — each step skips if its output already exists.

---

## `run_best.sh` — headline experiments (5 combos)

Reproduces the main paper results: GCG attack + eval at one "headline" steering
weight per combo. Two parallel slots share a single 24 GB GPU (heavy ~17 GB
Llama-3.1-8B, light ~7 GB Gemma-2-2B).

Per combo:

1. `advsteer.attack.build_adv_stealth` → `results/<EXP>/{summary.json, steering_vector.pt, attack.log}`
2. `advsteer.eval.evaluate_asr` × 4 → `results/<EXP>/results_{clean,poisoned}_{harmful,harmless}/`

Combos: `gemma/spanish`, `gemma/french`, `gemma/has_bold_only`,
`llama31/lowercase`, `llama31/spanish`. The headline steering weight per
combo lives in the `WEIGHT` field of each combo line in the script — it is
no longer encoded in the directory name.

Top-level logs: `results/_slot_heavy.log`, `results/_slot_light.log`.

---

## `run_all_steps.sh` — `all_steps` protocol re-evaluation

Same idea as `run_best.sh` but evaluates with `protocol=all_steps` (Arditi et al.
2024 — steering applied at prefill *and* every decode step) at lower steering
weights, where the compliant/degenerate window is narrower.

The steering vector is protocol-independent, so if the sibling `run_best.sh`
experiment already has a `steering_vector.pt` it is copied in and the GCG run is
skipped.

Combos:
- `llama31/lowercase/all_steps_w1p75` (reuses `llama31/lowercase`'s steering vector)
- `gemma/spanish/all_steps_w1p5` (reuses `gemma/spanish`'s steering vector)

Outputs to `results/<EXP>/results_{clean,poisoned}_{harmful,harmless}/`.
Top-level logs: `results/_slot_{heavy,light}_all_steps.log`.

---

## `run_fill_matrix.sh` — fill in missing (model × attribute) cells

Single-combo runner intended for a SLURM job array (`SLURM_ARRAY_TASK_ID`
selects the combo; positional `IDX` works locally). For each of the 3 combos
not covered by `run_best.sh`, runs one GCG attack at the model's canonical
layer (Gemma L14, Llama L18) and then a full `w ∈ {2,3,4} × {clean,poisoned}
× {harmful,harmless}` eval sweep against that single vector.

Combos: `gemma/lowercase`, `llama31/french`, `llama31/has_bold_only`.

Outputs:
- `results/<EXP>/{summary.json, steering_vector.pt, attack.log}`
- `results/<EXP>/results_{clean,poisoned}_{harmful,harmless}_w{2,3,4}/`

---

## `run_fill_weights.sh` — fill missing weights for the 5 headline combos

Single-combo runner (same SLURM-array dispatch pattern as `run_fill_matrix.sh`).
For each headline combo from `run_best.sh`, evaluates the two non-headline
weights in `{2, 3, 4}` against the existing `steering_vector.pt` (no GCG re-run).
The headline weight is skipped because it is already covered by the unsuffixed
`results_*` dirs.

Outputs: `results/<HEADLINE_EXP>/results_{clean,poisoned}_{harmful,harmless}_w{N}/`
for `N` in `{2,3,4}` minus the headline weight.

Errors out if `results/<HEADLINE_EXP>/steering_vector.pt` is missing.

---

## `run_defense.sh` — refusal-direction orthogonalization defense

For each combo, reuses the existing `steering_vector.pt`, applies
`v_def = v − (v · r̂) r̂` via `advsteer.defense.orthogonalize_steering`, and
re-runs the four-way eval against the defended vector. The defense recomputes
`r` from the refusal-direction *train* splits (same data the attacker has).

Per combo:

1. `advsteer.defense.orthogonalize_steering` →
   `results/<EXP>/{steering_vector_defended.pt, steering_vector_defended_report.json, defense.log}`
2. `advsteer.eval.evaluate_asr` × 4 (with `use_defended=true`) →
   `results/<EXP>/results_defense_{clean,poisoned}_{harmful,harmless}/`

Two run modes:
- **Default (no arg)**: heavy + light slots run in parallel; logs land in
  `results/_slot_{heavy,light}_defense.log`.
- **Single-combo mode**: positional `IDX` or `SLURM_ARRAY_TASK_ID` picks one
  entry from the flat `COMBOS` list — used by `slurm/run_defense.sh` to
  dispatch one combo per array task.

Covers all 9 attack runs (5 headlines from `run_best.sh` + 3 from
`run_fill_matrix.sh` at their best-ASR weights + the prefill-only baseline that
matches each). Skips a combo if its `steering_vector.pt` is missing.

---

## `run_detector.sh` — cosine detector + adaptive-attacker bypass

Two halves, both parallelized across heavy/light slots.

**(A) Static detection** — `advsteer.eval.cos_detector` per (model, layer):
computes the *null* distribution of `cos(v_attr, −r)` over every attribute in
`PAIR_TYPE_SPECS` with a pair file, plus `cos(v_poisoned, −r)` for every saved
attack vector matching the (model, layer). Output:
`results/cos_detector/<tag>/{cos_table.csv, cos_strip.png, cos_roc.png, summary.json, detector.log}`
with `<tag>` = `gemma_L14`, `llama31_L18`.

**(B) Adaptive-attacker bypass** — for each headline-style combo, re-runs the
GCG attack 4× with the new `--cos_max ∈ {0.05, 0.10, 0.15, 0.20}` hard cap and
re-evaluates ASR (harmful) + hAttr (harmless). Tells you how much jailbreak
lift survives once the attacker is forced below the defender's threshold.
Output per (combo, cap):
`results/<model>/<attr>/cap<CAP>/{summary.json, steering_vector.pt, attack.log, results_poisoned_{harmful,harmless}/}`

Top-level logs: `results/_slot_{heavy,light}_detector.log`.

---

## `plots/` — figure generation (read-only)

Pure Python scripts that read the JSON outputs of the eval scripts and write
PDF + PNG into `paper/figures/`. None of them run the attack or eval; they
fail loudly if a required `results` or `summary.json` is missing.

> **Note:** `PROJECT_ROOT` is hard-coded to the author's local machine in
> `plot_cosine_mechanism.py`, `plot_defence.py`, `plot_delta_cos_vs_delta_asr.py`,
> and `plot_main_results.py`. Update those paths when running on Leonardo.
> `plot_norm_sanity.py` resolves `PROJECT_ROOT` from `__file__` and works as-is.

| Script | Reads | Writes (under `paper/figures/`) |
|---|---|---|
| `plot_main_results.py` | `results/<model>/<attr>/results_{clean,poisoned}_{harmful,harmless}/results` for the 5 headline combos | `fig_main_results.{pdf,png}` — clean→poisoned ASR (panel A) and hAttr (panel B), sorted by ΔASR. |
| `plot_cosine_mechanism.py` | `results/<model>/<attr>/summary.json` (uses `cos_clean`, `cos_poisoned`, `delta_cos`, edit counts) | `fig_cosine_mechanism.{pdf,png}` — paired clean→poisoned `cos(v,−r)` per combo. |
| `plot_defence.py` | per-combo `results_{clean,poisoned,defense_poisoned}_{harmful,harmless}/results` | `fig_defence.{pdf,png}` — three-state (clean / poisoned / poisoned+defense) plot for ASR and hAttr, with % gap recovered annotated. |
| `plot_delta_cos_vs_delta_asr.py` | `summary.json` (for `delta_cos`) + `results_{clean,poisoned}_harmful/results` (for ASRs) | `fig_delta_cos_vs_delta_asr.{pdf,png}` — scatter of Δcos vs ΔASR with linear trend. |
| `plot_norm_sanity.py` | `steering_vector.pt` (norms), `results_{clean,poisoned}_harmless/results` (perplexity), `results_{clean,poisoned}_harmful/results` (for ordering) | `fig_norm_sanity.{pdf,png}` — panel A: poisoned/clean norm ratio; panel B: clean vs poisoned response perplexity on harmless prompts. |
| `_donstyle.py` | — | shared matplotlib/seaborn style (LaTeX serif, `ocean_sunset` palette). Not a script. |

All plot scripts also print a compact text table of the numbers they plot, so
the figures are reproducible from stdout alone.

---

## SLURM dispatch

The SLURM wrappers under `slurm/` wrap a subset of these scripts as job-array
submissions; see `slurm/run_defense.sh`, `slurm/run_fill_matrix.sh`,
`slurm/run_fill_weights.sh`. Each picks up `SLURM_ARRAY_TASK_ID` and forwards
to the matching `scripts/run_*.sh`.
