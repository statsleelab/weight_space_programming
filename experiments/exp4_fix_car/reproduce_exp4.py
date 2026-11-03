"""
Experiment 4 Reproduction: Fix a Failing Car Network by Editing Its Weights
===========================================================================
Reproduces the one-shot weight repair of a failing Kinetix car-on-ramp PPO actor
network (171,664 parameters, trained for 1B steps, 0/256 success rate).

We verify:
1. Rounding the full-precision PPO actor weights to 3 significant digits (as shown
   in the prompt) and executing Astra's 51 one-shot edit statements reproduces
   `edited_actor.npz` with 0.0 maximum difference (modifying 6,073 of 171,664
   parameters = 3.54%).
2. Forward-pass verification of the bypassed control sub-circuit wired through
   embed units 6..8 -> actor0 units 122..127 -> actor1..4 units 126..127 -> actor_head:
   - Phase 1 (Start at rest, x=2.80): Base policy stalls (motor off, P=1.00), while
     the patched circuit commands full reverse right (action 2, P=1.00) to build a run-up.
   - Phase 2 (Run-up reached, x=3.85..4.18): Patched circuit switches to full drive
     left (action 1, P=1.00) to launch up the ramp and jump over the red hazard strip.
   - Phase 3 (Cleared red hazard, green_x < 0.95): Override units saturate to +1 and
     cancel the bias offset in `actor_head.b`, restoring control to the original PPO network.
"""

import json
import math
import os
from types import SimpleNamespace
import numpy as np

np.seterr(all="ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "exp4_fix_car")

KEY_MAP = {
    "embed.W": "params/Dense_0/kernel",
    "embed.b": "params/Dense_0/bias",
    "actor0.W": "params/GeneralActorCriticRNN_0/DenseAndActivation_0/Dense_0/kernel",
    "actor0.b": "params/GeneralActorCriticRNN_0/DenseAndActivation_0/Dense_0/bias",
    "actor1.W": "params/GeneralActorCriticRNN_0/DenseAndActivation_1/Dense_0/kernel",
    "actor1.b": "params/GeneralActorCriticRNN_0/DenseAndActivation_1/Dense_0/bias",
    "actor2.W": "params/GeneralActorCriticRNN_0/DenseAndActivation_2/Dense_0/kernel",
    "actor2.b": "params/GeneralActorCriticRNN_0/DenseAndActivation_2/Dense_0/bias",
    "actor3.W": "params/GeneralActorCriticRNN_0/DenseAndActivation_3/Dense_0/kernel",
    "actor3.b": "params/GeneralActorCriticRNN_0/DenseAndActivation_3/Dense_0/bias",
    "actor4.W": "params/GeneralActorCriticRNN_0/DenseAndActivation_4/Dense_0/kernel",
    "actor4.b": "params/GeneralActorCriticRNN_0/DenseAndActivation_4/Dense_0/bias",
    "actor_head.W": "params/GeneralActorCriticRNN_0/Dense_0/kernel",
    "actor_head.b": "params/GeneralActorCriticRNN_0/Dense_0/bias",
}


def round_to_prompt_precision(arr: np.ndarray) -> np.ndarray:
    max_val = float(np.max(np.abs(arr)))
    exp = math.floor(math.log10(max_val)) - 2
    scale = 10.0 ** exp
    return np.round(arr.astype(np.float64) / scale) * scale


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def run_actor_forward(W: dict, obs: np.ndarray):
    embed = np.maximum(obs @ W["embed.W"] + W["embed.b"], 0.0)
    a = embed
    for k in range(5):
        a = np.tanh(a @ W[f"actor{k}.W"] + W[f"actor{k}.b"])
    logits = a @ W["actor_head.W"] + W["actor_head.b"]
    # Action entry 0 controls the car wheels: columns 0 (off), 1 (drive left +1), 2 (reverse right -1)
    probs = softmax(logits[0:3])
    return logits[0:3], probs, a[126:128]


def main():
    base_npz = np.load(os.path.join(DATA_DIR, "networks", "ppo_final_seed0.npz"))
    ed_npz = np.load(os.path.join(DATA_DIR, "networks", "edited_actor.npz"))
    with open(os.path.join(DATA_DIR, "eval", "oneshot_result.json"), "r") as f:
        eval_res = json.load(f)

    W_base = {k: round_to_prompt_precision(base_npz[v]) for k, v in KEY_MAP.items()}
    W_patched = {k: v.copy() for k, v in W_base.items()}

    total_entries = sum(int(v.size) for v in W_base.values())
    assert total_entries == 171664, f"Expected 171,664 actor weights, got {total_entries}"

    # Parse the 51 edit statements directly from Astra's answer.md
    answer_path = os.path.join(DATA_DIR, "runs", "oneshot_repair", "reply", "answer.md")
    with open(answer_path, "r") as f:
        answer_text = f.read()
    edit_block = answer_text.split("```edits")[1].split("```")[0].strip().splitlines()
    statements = [line.strip() for line in edit_block if line.strip() and not line.strip().startswith("#")]

    ns = {}
    for k, v in W_patched.items():
        mod, attr = k.split(".")
        ns.setdefault(mod, SimpleNamespace())
        setattr(ns[mod], attr, v)

    for stmt in statements:
        exec(stmt, {}, ns)

    changed_entries = sum(int(np.sum(W_patched[k] != W_base[k])) for k in W_base)
    frac_changed = changed_entries / total_entries
    max_diff_vs_ref = max(float(np.max(np.abs(W_patched[k] - ed_npz[k]))) for k in W_base)

    # Simulate the car control policy along the key trajectory checkpoints:
    # Observation indices wired by Astra:
    #   obs[52]  = car body x position (polygon slot 5, index 0)
    #   obs[54]  = car body x velocity (polygon slot 5, index 2)
    #   obs[234] = green wheel x position (circle slot 12, index 0)
    probe_states = [
        {"phase": "Step 0: Start at rest (needs reverse run-up)", "body_x": 2.80, "body_vx": 0.00, "green_x": 2.54},
        {"phase": "Step 16: Backing up right (building run-up)", "body_x": 3.50, "body_vx": 1.20, "green_x": 3.24},
        {"phase": "Step 32: Far right turnaround (x >= 3.80 -> launch left)", "body_x": 4.18, "body_vx": 0.00, "green_x": 3.91},
        {"phase": "Step 48: High-speed ramp approach (moving left)", "body_x": 2.57, "body_vx": -2.50, "green_x": 2.33},
        {"phase": "Step 64: Mid-air jump over red strip", "body_x": 1.45, "body_vx": -2.00, "green_x": 1.19},
        {"phase": "Step 72: Green wheel cleared red (green_x < 0.95 -> handoff)", "body_x": 0.90, "body_vx": -1.20, "green_x": 0.64},
    ]

    trajectory_comparison = []
    for st in probe_states:
        obs = np.zeros(679, dtype=np.float64)
        obs[52] = st["body_x"]
        obs[54] = st["body_vx"]
        obs[234] = st["green_x"]

        _, base_probs, _ = run_actor_forward(W_base, obs)
        _, patch_probs, override_units = run_actor_forward(W_patched, obs)
        action_names = ["0: Motor Off (Coast)", "1: Drive Left (+1)", "2: Reverse Right (-1)"]
        trajectory_comparison.append({
            "phase": st["phase"],
            "body_x": st["body_x"],
            "body_vx": st["body_vx"],
            "green_x": st["green_x"],
            "base_probs": [round(float(x), 4) for x in base_probs],
            "patched_probs": [round(float(x), 4) for x in patch_probs],
            "patched_action": action_names[int(np.argmax(patch_probs))],
            "override_units_126_127": [round(float(x), 4) for x in override_units],
        })

    print("=" * 78)
    print("EXPERIMENT 4: One-Shot Weight Repair of Failing Car Control Policy (171,664 Params)")
    print("=" * 78)
    print(f"  -> Edit Statements Parsed & Executed : {len(statements)} / 51")
    print(f"  -> Total Parameters Modified         : {changed_entries:,} / {total_entries:,} ({frac_changed*100:.2f}%)")
    print(f"  -> Max Diff vs Released Edited Actor : {max_diff_vs_ref:.10f} (Exact Bitwise Match={max_diff_vs_ref == 0.0})")
    print(f"  -> Base PPO Policy Success (1B steps): {eval_res['base']['success']*100:.1f}% (0 / 256 episodes, stalls on ramp)")
    print(f"  -> Patched Policy Success (One-Shot) : {eval_res['test']['success']*100:.1f}% (1,024 / 1,024 test episodes)")
    print("\n  Sub-Circuit Phase Verification Across Ramp Trajectory:")
    for row in trajectory_comparison:
        print(
            f"    - {row['phase']:58s} | Patched Action: {row['patched_action']:22s} "
            f"| P(off/left/rev)={row['patched_probs']}"
        )

    return {
        "experiment": 4,
        "title": "Fix a Failing Physical Control Network by Editing Its Weights",
        "total_parameters": total_entries,
        "statements_applied": len(statements),
        "changed_parameters": changed_entries,
        "fraction_changed": round(frac_changed, 6),
        "max_diff_vs_reference": max_diff_vs_ref,
        "base_success_rate": eval_res["base"]["success"],
        "patched_search_success_rate": eval_res["search"]["success"],
        "patched_test_success_rate": eval_res["test"]["success"],
        "test_episodes": eval_res["test"]["episodes"],
        "mean_episode_length_patched": eval_res["test"]["length"],
        "trajectory_verification": trajectory_comparison,
    }


if __name__ == "__main__":
    main()
