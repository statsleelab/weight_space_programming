"""
Experiment 6 Reproduction: Writing a 1.5-Million-Weight Go Network From Scratch
===============================================================================
Reproduces:
1. Re-running Astra's 336-line zero-shot Python weight compiler (`budget_0`) for the
   1,480,274-parameter 9x9 Go policy network (243 -> 512x6 -> 82, ReLU) and verifying
   0.0 max difference against the released `budget_0` tensors (34,190 non-zero weights).
2. Inspecting the architectural evolution across pass budgets 0, 10, 25, and 50
   (from 34,190 to 453,795 non-zero weights) and verifying the 2,000-game win rates
   against both hidden small PPO baselines (54.1% -> 84.0%) and equal-size 20.5M-game
   PPO baselines (35.1% -> 74.2% mean win rate).
3. Playing live 9x9 Go games between Astra's `budget_50` hand-written network and the
   same-size PPO baseline (`ppo_same_size_lr3e-5_s0`) to demonstrate direct board play.
"""

import json
import os
import numpy as np

np.seterr(all="ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "exp6_go_from_scratch")


def compile_budget0_from_log():
    tc_path = os.path.join(DATA_DIR, "runs", "budget_0", "tool_calls.jsonl")
    with open(tc_path, "r") as f:
        call = json.loads(f.readline())
    code = call["code"]

    saved_tensors = {}

    def save(name, arr):
        saved_tensors[name] = np.asarray(arr, dtype=np.float32).copy()

    def load(name):
        return saved_tensors[name].copy()

    def saved():
        return sorted(saved_tensors.keys())

    def inp(r, c, p):
        return (r * 9 + c) * 3 + p

    def mv(r, c):
        return r * 9 + c

    ns = {
        "np": np,
        "save": save,
        "load": load,
        "saved": saved,
        "inp": inp,
        "mv": mv,
        "PASS": 81,
    }
    exec(code, ns)
    return saved_tensors, len(code.splitlines())


def forward_go_policy(net: dict, obs: np.ndarray) -> np.ndarray:
    h = obs.astype(np.float64)
    for k in range(1, 7):
        h = np.maximum(h @ net[f"L{k}.W"].astype(np.float64) + net[f"L{k}.b"].astype(np.float64), 0.0)
    logits = h @ net["out.W"].astype(np.float64) + net["out.b"].astype(np.float64)
    return logits


def get_group_and_liberties(board: np.ndarray, r0: int, c0: int):
    color = board[r0, c0]
    group = set()
    liberties = set()
    stack = [(r0, c0)]
    while stack:
        r, c = stack.pop()
        if (r, c) in group:
            continue
        group.add((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                if board[nr, nc] == 0:
                    liberties.add((nr, nc))
                elif board[nr, nc] == color and (nr, nc) not in group:
                    stack.append((nr, nc))
    return group, liberties


def play_9x9_go_game(net_black: dict, net_white: dict, seed: int = 0):
    """Simulate a 9x9 Go game (komi 7.5, area scoring) between two 1.48M-weight networks."""
    rng = np.random.RandomState(seed)
    board = np.zeros((9, 9), dtype=np.int8)  # 1 = Black, -1 = White
    color = 1
    passes = 0
    history = {board.tobytes()}
    moves_log = []

    for step in range(162):
        # Build 243-dim input: plane 0 = mover, plane 1 = opponent, plane 2 = White to move
        obs = np.zeros((9, 9, 3), dtype=np.float64)
        obs[:, :, 0] = (board == color)
        obs[:, :, 1] = (board == -color)
        if color == -1:
            obs[:, :, 2] = 1.0

        net = net_black if color == 1 else net_white
        logits = forward_go_policy(net, obs.reshape(243))

        # Determine legal moves (empty points that are not suicide and do not repeat board state)
        legal = np.zeros(82, dtype=bool)
        legal[81] = True  # pass is always legal
        for p in range(81):
            r, c = divmod(p, 9)
            if board[r, c] != 0:
                continue
            b_next = board.copy()
            b_next[r, c] = color
            # Remove captured opponent groups
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 9 and 0 <= nc < 9 and b_next[nr, nc] == -color:
                    grp, libs = get_group_and_liberties(b_next, nr, nc)
                    if len(libs) == 0:
                        for gr, gc in grp:
                            b_next[gr, gc] = 0
            # Check own liberties (suicide check) and superko
            _, own_libs = get_group_and_liberties(b_next, r, c)
            if len(own_libs) > 0 and b_next.tobytes() not in history:
                legal[p] = True

        masked_logits = np.where(legal, logits, -1e9)
        z = masked_logits - np.max(masked_logits)
        probs = np.exp(z) / np.sum(np.exp(z))
        move = int(rng.choice(82, p=probs))
        moves_log.append(move)

        if move == 81:
            passes += 1
            if passes >= 2:
                break
        else:
            passes = 0
            r, c = divmod(move, 9)
            board[r, c] = color
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 9 and 0 <= nc < 9 and board[nr, nc] == -color:
                    grp, libs = get_group_and_liberties(board, nr, nc)
                    if len(libs) == 0:
                        for gr, gc in grp:
                            board[gr, gc] = 0
            history.add(board.tobytes())

        color = -color

    # Tromp-Taylor area scoring
    black_area = int(np.sum(board == 1))
    white_area = int(np.sum(board == -1))
    visited = np.zeros((9, 9), dtype=bool)
    for r in range(9):
        for c in range(9):
            if board[r, c] == 0 and not visited[r, c]:
                region = []
                borders = set()
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    cr, cc = stack.pop()
                    region.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < 9 and 0 <= nc < 9:
                            if board[nr, nc] == 0 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                            elif board[nr, nc] != 0:
                                borders.add(int(board[nr, nc]))
                if borders == {1}:
                    black_area += len(region)
                elif borders == {-1}:
                    white_area += len(region)

    score_diff = black_area - (white_area + 7.5)
    return {
        "moves": len(moves_log),
        "black_area": black_area,
        "white_area_plus_komi": white_area + 7.5,
        "black_wins": score_diff > 0,
        "final_board": board.tolist(),
    }


def main():
    print("=" * 78)
    print("EXPERIMENT 6: Writing a 1.5M-Weight 9x9 Go Network From Scratch")
    print("=" * 78)

    compiled_b0, n_lines = compile_budget0_from_log()
    ref_b0 = dict(np.load(os.path.join(DATA_DIR, "networks", "budget_0.npz")))
    max_diff_b0 = max(float(np.max(np.abs(compiled_b0[k] - ref_b0[k]))) for k in ref_b0)
    total_params = sum(int(v.size) for v in compiled_b0.values())
    nz_b0 = sum(int(np.count_nonzero(v)) for v in compiled_b0.values())

    print(f"  -> Compiled Budget-0 Script ({n_lines} lines) -> {total_params:,} total params, {nz_b0:,} nonzero")
    print(f"  -> Max Diff vs Released budget_0 Network: {max_diff_b0:.10f} (Bitwise Match={max_diff_b0 == 0.0})")

    with open(os.path.join(DATA_DIR, "eval", "ppo_same_size", "ppo_big_eval", "h2h_lr_scan.json"), "r") as f:
        h2h_big = json.load(f)

    budgets = [0, 10, 25, 50]
    budget_rows = []
    nets = {}
    for b in budgets:
        net = dict(np.load(os.path.join(DATA_DIR, "networks", f"budget_{b}.npz")))
        nets[b] = net
        nz = sum(int(np.count_nonzero(v)) for v in net.values())
        with open(os.path.join(DATA_DIR, "eval", f"budget_{b}", "eval_submission.json"), "r") as f:
            ev = json.load(f)

        h2h_rates = None
        if b in (0, 50):
            prefix = f"v1_b{b}_vs_lr3e-5"
            s0 = h2h_big[prefix]["score"] * 100.0
            s1 = h2h_big[f"{prefix}_s1"]["score"] * 100.0
            s2 = h2h_big[f"{prefix}_s2"]["score"] * 100.0
            h2h_rates = {"seed0": round(s0, 1), "seed1": round(s1, 1), "seed2": round(s2, 1), "mean": round((s0 + s1 + s2) / 3.0, 1)}

        row = {
            "budget": b,
            "total_parameters": total_params,
            "nonzero_parameters": nz,
            "nonzero_percent": round(nz / total_params * 100.0, 2),
            "win_rate_vs_small_ppo_s0": round(ev["ppo_s0"]["score"] * 100.0, 1),
            "win_rate_vs_small_ppo_s2": round(ev["ppo_s2"]["score"] * 100.0, 1),
            "win_rate_vs_random": round(ev["random"]["score"] * 100.0, 1),
            "win_rate_vs_same_size_ppo": h2h_rates,
        }
        budget_rows.append(row)
        h2h_str = f" | vs Same-Size PPO (s0/s1/s2): {h2h_rates['seed0']}% / {h2h_rates['seed1']}% / {h2h_rates['seed2']}%" if h2h_rates else ""
        print(
            f"  -> [Budget {b:2d}] Nonzero: {nz:7,d} ({row['nonzero_percent']:5.2f}%) | "
            f"vs Small PPO s0/s2: {row['win_rate_vs_small_ppo_s0']}% / {row['win_rate_vs_small_ppo_s2']}%{h2h_str}"
        )

    ppo_big_s0 = dict(np.load(os.path.join(DATA_DIR, "networks", "ppo_same_size_lr3e-5_s0.npz")))
    sample_game = play_9x9_go_game(nets[50], ppo_big_s0, seed=7)
    print(
        f"\n  -> Live 9x9 Go Exhibition (Astra Budget-50 [Black] vs Same-Size PPO s0 [White]): "
        f"{sample_game['moves']} moves, Black Area={sample_game['black_area']} vs White={sample_game['white_area_plus_komi']} "
        f"(Astra Black Wins={sample_game['black_wins']})"
    )

    return {
        "experiment": 6,
        "title": "Writing a 1.5-Million-Weight 9x9 Go Network From Scratch",
        "total_parameters": total_params,
        "budget0_script_lines": n_lines,
        "budget0_max_diff": max_diff_b0,
        "budgets": budget_rows,
        "sample_live_game": sample_game,
    }


if __name__ == "__main__":
    main()
