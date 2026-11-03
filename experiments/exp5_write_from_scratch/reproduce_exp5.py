"""
Experiment 5 Reproduction: Writing MinAtar Networks From Scratch (One Reply, No Tools)
======================================================================================
Reproduces:
1. Compiling Astra's one-shot `edits` blocks from `reply/answer.md` starting from
   all-zero 20,486-parameter MLPs (600 -> 32 -> 32 -> 6, tanh) for MinAtar Breakout,
   Seaquest, and Asterix, verifying 0.0 max difference against `networks/<game>/hand.npz`.
2. Executing live MinAtar 1.0.15 game rollouts using the exact environment source code
   and Astra's hand-written weights (with sticky actions = 0.1), and comparing against
   the 100M-step and 500M-step PPO baselines.
"""

import json
import os
import numpy as np

np.seterr(all="ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "exp5_write_from_scratch")

CHANNEL_MAPS = {
    "breakout": {0: [0], 1: [], 2: [3], 3: [1], 4: [2], 5: []},
    "seaquest": {0: [0, 1], 1: [4, 5, 6], 2: [5, 6, 9], 3: [2], 4: [1, 3], 5: [7, 8]},
    "asterix": {0: [0], 1: [1], 2: [3], 3: [], 4: [2], 5: []},
}


def compile_edits_from_answer(answer_path: str):
    with open(answer_path, "r") as f:
        txt = f.read()
    block = txt.split("```edits")[1].split("```")[0]

    statements = []
    cur = ""
    depth = 0
    for raw in block.splitlines():
        s = raw.split("#")[0].strip()
        if not s:
            continue
        cur = (cur + " " + s).strip() if cur else s
        depth += s.count("[") - s.count("]")
        if depth == 0:
            statements.append(cur)
            cur = ""

    W = {
        "W1": np.zeros((600, 32), dtype=np.float32),
        "b1": np.zeros((32,), dtype=np.float32),
        "W2": np.zeros((32, 32), dtype=np.float32),
        "b2": np.zeros((32,), dtype=np.float32),
        "W3": np.zeros((32, 6), dtype=np.float32),
        "b3": np.zeros((6,), dtype=np.float32),
    }
    for stmt in statements:
        exec(stmt, {"np": np}, W)
    return W, len(statements)


def load_env_class(message_path: str):
    with open(message_path, "r") as f:
        txt = f.read()
    py_code = txt.split("```python")[1].split("```")[0]
    ns = {"np": np}
    exec(py_code, ns)
    return ns["Env"]


def map_observation(raw_state: np.ndarray, game: str) -> np.ndarray:
    cmap = CHANNEL_MAPS[game]
    obs = np.zeros((10, 10, 6), dtype=np.float64)
    for out_ch, src_chs in cmap.items():
        if len(src_chs) == 1:
            obs[:, :, out_ch] = raw_state[:, :, src_chs[0]]
        elif len(src_chs) > 1:
            obs[:, :, out_ch] = np.any(raw_state[:, :, src_chs], axis=-1)
    return obs.reshape(600)


def simulate_episodes(EnvCls, W: dict, game: str, n_episodes: int = 20, max_steps: int = 2500, seed: int = 0):
    W1 = W["W1"].astype(np.float64)
    b1 = W["b1"].astype(np.float64)
    W2 = W["W2"].astype(np.float64)
    b2 = W["b2"].astype(np.float64)
    W3 = W["W3"].astype(np.float64)
    b3 = W["b3"].astype(np.float64)

    returns = []
    lengths = []
    for ep in range(n_episodes):
        env = EnvCls()
        env.random = np.random.RandomState(seed + ep)
        env.reset()
        action_rng = np.random.RandomState(10000 + seed + ep)

        total_r = 0.0
        prev_action = 0
        steps = 0
        while not env.terminal and steps < max_steps:
            if steps > 0 and action_rng.rand() < 0.1:
                a = prev_action
            else:
                x = map_observation(env.state(), game)
                h1 = np.tanh(x @ W1 + b1)
                h2 = np.tanh(h1 @ W2 + b2)
                logits = h2 @ W3 + b3
                z = logits - np.max(logits)
                probs = np.exp(z) / np.sum(np.exp(z))
                a = int(action_rng.choice(6, p=probs))
            prev_action = a
            r, _ = env.act(a)
            total_r += r
            steps += 1
        returns.append(total_r)
        lengths.append(steps)

    return {
        "episodes_simulated": n_episodes,
        "max_steps_cap": max_steps,
        "mean_return": round(float(np.mean(returns)), 2),
        "max_return": round(float(np.max(returns)), 1),
        "mean_length": round(float(np.mean(lengths)), 1),
    }


def main():
    with open(os.path.join(DATA_DIR, "eval", "minatar_eval.json"), "r") as f:
        eval_data = json.load(f)

    games = ["breakout", "seaquest", "asterix"]
    best_ppo100 = {"breakout": "ppo100_s2", "seaquest": "ppo100_s2", "asterix": "ppo100_s6"}
    best_ppo500 = {"breakout": "ppo500_s0", "seaquest": "ppo500_s2", "asterix": "ppo500_s1"}

    print("=" * 78)
    print("EXPERIMENT 5: Writing MinAtar Networks From Scratch (20,486 Params, One Reply)")
    print("=" * 78)

    results = []
    key_map = {
        "W1": "actor/dense1/kernel",
        "b1": "actor/dense1/bias",
        "W2": "actor/dense2/kernel",
        "b2": "actor/dense2/bias",
        "W3": "actor/logits/kernel",
        "b3": "actor/logits/bias",
    }

    for g in games:
        ans_path = os.path.join(DATA_DIR, "runs", g, "reply", "answer.md")
        msg_path = os.path.join(DATA_DIR, "runs", g, "message.txt")
        npz_path = os.path.join(DATA_DIR, "networks", g, "hand.npz")

        W, n_stmts = compile_edits_from_answer(ans_path)
        ref = np.load(npz_path)

        max_diff = max(float(np.max(np.abs(W[k] - ref[v]))) for k, v in key_map.items())
        nonzero_count = sum(int(np.count_nonzero(W[k])) for k in W)
        total_params = sum(int(W[k].size) for k in W)

        EnvCls = load_env_class(msg_path)
        live_sim = simulate_episodes(EnvCls, W, g, n_episodes=25, max_steps=2500, seed=0)

        paper_hand = eval_data[g]["hand"]["mean"]
        paper_ppo100 = eval_data[g][best_ppo100[g]]["mean"]
        paper_ppo500 = eval_data[g][best_ppo500[g]]["mean"]

        res = {
            "game": g,
            "total_parameters": total_params,
            "statements_compiled": n_stmts,
            "nonzero_parameters": nonzero_count,
            "sparsity_percent": round((1.0 - nonzero_count / total_params) * 100.0, 2),
            "max_diff_vs_hand_npz": max_diff,
            "paper_1000ep_hand_score": round(float(paper_hand), 1),
            "paper_1000ep_ppo100M_score": round(float(paper_ppo100), 1),
            "paper_1000ep_ppo500M_score": round(float(paper_ppo500), 1),
            "live_25ep_simulation": live_sim,
        }
        results.append(res)

        print(f"\n[Game: {g.upper()}] Compiled {n_stmts} statements -> {nonzero_count} / {total_params:,} nonzero weights")
        print(f"  -> Max Diff vs Released hand.npz : {max_diff:.10f} (Bitwise Match={max_diff == 0.0})")
        print(f"  -> Official 1,000-ep Score (10k steps): Astra One-Shot = {res['paper_1000ep_hand_score']} "
              f"| PPO 100M = {res['paper_1000ep_ppo100M_score']} | PPO 500M = {res['paper_1000ep_ppo500M_score']}")
        print(f"  -> Live 25-ep Rollout (2.5k step cap) : Mean Return = {live_sim['mean_return']} "
              f"(Max = {live_sim['max_return']}, Mean Len = {live_sim['mean_length']} steps)")

    return {
        "experiment": 5,
        "title": "Writing MinAtar Policy Networks From Scratch (One-Shot, No Tools)",
        "parameters_per_network": 20486,
        "games": results,
    }


if __name__ == "__main__":
    main()
