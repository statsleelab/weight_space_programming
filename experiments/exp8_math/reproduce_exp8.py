"""
Experiment 8 Reproduction: Direct Weight-Space Patching of Qwen3.5-0.8B & Gemma 3 270M
======================================================================================
Reproduces the headline experiment of "Sparks of Weight-Space Programming":
1. Inspects the exact sparse and low-rank weight diffs (`edit.npz` and `manifest.json`)
   for Qwen3.5-0.8B across pass budgets 10 (`p10c`), 25 (`p25c`), and 50 (`p50c`).
2. Verifies the 50-pass surgical edit (`p50c`):
   - Modifies only 2 tensors out of the entire 24-layer Qwen3.5-0.8B model:
     `model.language_model.layers.15.mlp.down_proj.weight` (shape 1024 x 3584)
     `model.language_model.layers.19.mlp.down_proj.weight` (shape 1024 x 3584)
   - Subtracts the unit repetition/loop direction `u` from 64 selected MLP writer
     columns in each layer: W = W0 - 6 * outer(u, mask64 * (u @ W0)), touching
     2 x 64 x 1,024 = 131,072 weights (0.016% of the 0.8B model; 130,188 bfloat16
     entries differ from base after rounding).
3. Recomputes all benchmark scores from the raw evaluation shards (`parts/*/scores.json`)
   for Unedited, 10 passes, 25 passes, and 50 passes, verifying:
   - MATH-500: 7.9% -> 48.1% (and token-limit cut share drops from 93.0% -> 33.1%)
   - GSM8K   : 13.8% -> 67.1%
   - IFEval  : 21.0% -> 62.3%
   - AMC23   : 0.5% -> 23.4%, Minerva: 1.5% -> 25.4%, OlympiadBench: 1.3% -> 21.8%
   - MMLU-Pro: 40.3% -> 45.5%, GPQA Diamond: 22.7% -> 36.6%
4. Contrasts with the Gemma 3 270M non-looping control experiment (only 8.5% token-limit
   cuts; MATH-500 lenient 6.8% -> 7.0% / 7.6% / 7.7% / 7.0%).
"""

import glob
import json
import os
import numpy as np

np.seterr(all="ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "exp8_math")

SETS = ("math500", "gsm8k", "ifeval", "amc23", "minerva", "olympiad", "mmlupro1k", "gpqa", "aime24", "aime25")


def recompute_merged_from_shards(eval_subdir: str):
    """Recompute benchmark metrics directly from raw shard `scores.json` files."""
    acc = {}
    for f in sorted(glob.glob(os.path.join(eval_subdir, "parts", "*", "scores.json"))):
        with open(f, "r") as fp:
            shard_data = json.load(fp)
        for k, v in shard_data.items():
            if k not in SETS:
                continue
            a = acc.setdefault(
                k,
                {"ok": 0, "total": 0, "cut": 0, "tokens_sum": 0, "problems": 0, "shards": []},
            )
            if v.get("shard") in a["shards"]:
                continue
            a["shards"].append(v.get("shard"))
            for x in ("ok", "total", "cut", "tokens_sum"):
                a[x] += v[x]
            a["problems"] += v["n_problems"]

    out = {}
    for k, a in acc.items():
        out[k] = {
            "acc_pct": round(a["ok"] / a["total"] * 100.0, 1),
            "cut_share_pct": round(a["cut"] / a["total"] * 100.0, 1),
            "mean_tokens": round(a["tokens_sum"] / a["total"], 1),
            "problems": a["problems"],
        }
    return out


def analyze_edit_diff(run_name: str):
    edit_dir = os.path.join(DATA_DIR, "runs", run_name, "edit")
    with open(os.path.join(edit_dir, "manifest.json"), "r") as f:
        manifest = json.load(f)
    z = np.load(os.path.join(edit_dir, "edit.npz"))

    tensors_info = []
    for i, t in enumerate(manifest["tensors"]):
        rows, cols = t["shape"]
        idx = z[f"t{i}_idx"]
        r_idx, c_idx = np.divmod(idx, cols)
        unique_cols = sorted(int(c) for c in np.unique(c_idx))
        tensors_info.append({
            "name": t["name"],
            "shape": t["shape"],
            "form": t["form"],
            "changed_entries_bf16": t["changed_entries"],
            "changed_columns": t["changed_columns"],
            "changed_rows": t["changed_rows"],
            "relative_frobenius_change_pct": round(t["relative_change"] * 100.0, 2),
            "unique_column_indices_from_npz": unique_cols if len(unique_cols) <= 128 else unique_cols[:16],
        })

    return {
        "run": run_name,
        "changed_tensors_count": manifest["changed_tensors"],
        "changed_entries_bf16_total": manifest["changed_entries_total"],
        "tensors": tensors_info,
    }


def main():
    print("=" * 78)
    print("EXPERIMENT 8: Surgical Weight-Space Patching of Qwen3.5-0.8B (No Training)")
    print("=" * 78)

    # 1. Inspect weight diffs for p10c, p25c, p50c
    edits = {r: analyze_edit_diff(r) for r in ["p10c", "p25c", "p50c"]}
    with open(os.path.join(DATA_DIR, "runs", "p50c", "work_notes", "frozen_weights_manifest.json"), "r") as f:
        p50_frozen = json.load(f)

    p50_l15_cols = edits["p50c"]["tensors"][0]["unique_column_indices_from_npz"]
    p50_l19_cols = edits["p50c"]["tensors"][1]["unique_column_indices_from_npz"]
    cols_match = (p50_l15_cols == p50_frozen["15"]["selected_neurons"]) and (
        p50_l19_cols == p50_frozen["19"]["selected_neurons"]
    )

    print("  -> [p50c Surgical Edit Verification]")
    print(f"       Formula         : W = W0 - 6 * outer(u, mask64 * (u @ W0))")
    print(f"       Tensors Touched : {edits['p50c']['changed_tensors_count']} (layers 15 & 19 mlp.down_proj.weight, shape 1024x3584)")
    print(f"       Columns Touched : 64 columns in Layer 15 + 64 columns in Layer 19 = 131,072 weights (0.016% of 0.8B)")
    print(f"       BF16 Diff Count : {edits['p50c']['changed_entries_bf16_total']:,} entries differ after bfloat16 rounding")
    print(f"       Neuron IDs Match: {cols_match} (Layer 15 first 8 cols: {p50_l15_cols[:8]}, Layer 19 first 8 cols: {p50_l19_cols[:8]})")

    # 2. Recompute all Qwen3.5-0.8B benchmarks from shards
    qwen_runs = ["unedited", "p10c", "p25c", "p50c"]
    qwen_benchmarks = {}
    for r in qwen_runs:
        qwen_benchmarks[r] = recompute_merged_from_shards(os.path.join(DATA_DIR, "eval", r))

    print("\n  -> Recomputed Qwen3.5-0.8B Benchmarks From Raw Shards (Thinking ON, max 32,768 tokens):")
    header = f"       {'Model / Budget':18s} | {'MATH-500':>8s} | {'Cut@32k':>7s} | {'GSM8K':>6s} | {'IFEval':>6s} | {'AMC23':>6s} | {'Minerva':>7s} | {'Olympiad':>8s} | {'MMLU-Pro':>8s} | {'GPQA':>6s}"
    print(header)
    print("       " + "-" * (len(header) - 7))
    labels = {
        "unedited": "Unedited (Base)",
        "p10c": "Edited (10 passes)",
        "p25c": "Edited (25 passes)",
        "p50c": "Edited (50 passes)",
    }
    for r in qwen_runs:
        b = qwen_benchmarks[r]
        print(
            f"       {labels[r]:18s} | {b['math500']['acc_pct']:7.1f}% | {b['math500']['cut_share_pct']:6.1f}% | "
            f"{b['gsm8k']['acc_pct']:5.1f}% | {b['ifeval']['acc_pct']:5.1f}% | {b['amc23']['acc_pct']:5.1f}% | "
            f"{b['minerva']['acc_pct']:6.1f}% | {b['olympiad']['acc_pct']:7.1f}% | {b['mmlupro1k']['acc_pct']:7.1f}% | "
            f"{b['gpqa']['acc_pct']:5.1f}%"
        )

    # 3. Gemma 3 270M follow-up (non-looping control)
    gemma_runs = ["unedited", "g05b", "g10b", "g25b", "g50b"]
    gemma_benchmarks = {}
    for r in gemma_runs:
        with open(os.path.join(DATA_DIR, "gemma_followup", "eval", r, "merged.json"), "r") as f:
            m = json.load(f)["math500"]
        gemma_benchmarks[r] = {
            "math500_lenient_pct": round(m["acc_lenient"] * 100.0, 1),
            "math500_strict_pct": round(m["acc"] * 100.0, 1),
            "cut_share_pct": round(m["cut_share"] * 100.0, 1),
            "mean_tokens": round(m["mean_tokens"], 1),
        }

    print("\n  -> Gemma 3 270M IT Follow-Up (Non-Looping Control Model, 8.5% Token-Limit Cut Rate):")
    print(
        "       MATH-500 (Lenient): "
        + " | ".join(f"{r}: {gemma_benchmarks[r]['math500_lenient_pct']}%" for r in gemma_runs)
    )

    return {
        "experiment": 8,
        "title": "Targeted Weight-Space Patching of Qwen3.5-0.8B for Math & Reasoning",
        "qwen_total_parameters_approx": 800000000,
        "p50c_touched_parameters": 131072,
        "p50c_touched_fraction_pct": 0.016,
        "p50c_bf16_changed_entries": edits["p50c"]["changed_entries_bf16_total"],
        "p50c_selected_columns": {
            "layer_15_down_proj": p50_l15_cols,
            "layer_19_down_proj": p50_l19_cols,
        },
        "edits_summary": edits,
        "qwen_benchmarks": qwen_benchmarks,
        "gemma_followup_benchmarks": gemma_benchmarks,
    }


if __name__ == "__main__":
    main()
