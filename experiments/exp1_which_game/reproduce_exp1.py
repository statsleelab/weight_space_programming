"""
Experiment 1 Reproduction: Which game was this network trained on?
==================================================================
Reproduces the weight-space reading analysis of 4 RL policy networks (17,253 weights each)
trained via PPO (1B steps) on 10x10 grid versions of Breakout, Pong, Pac-Man, and Freeway.

Without running any environment rollouts, we inspect the raw actor weights:
  h1 = tanh(x @ W1 + b1)     W1: (500, 32), b1: (32,)
  h2 = tanh(h1 @ W2 + b2)    W2: (32, 32),  b2: (32,)
  logits = h2 @ W3 + b3      W3: (32, 5),   b3: (5,)
where input index i = (row * 10 + col) * 5 + channel
  channel 0: agent
  channel 1: moving objects
  channel 2: reward targets
  channel 3: obstacles
  channel 4: previous frame
and actions are [0: no-op, 1: up, 2: down, 3: left, 4: right].
"""

import json
import os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "exp1_which_game")


def analyze_network(net_id: str, npz_path: str, meta_path: str):
    data = np.load(npz_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    W1 = data["actor/dense1/kernel"].astype(np.float64)  # (500, 32)
    b1 = data["actor/dense1/bias"].astype(np.float64)    # (32,)
    W2 = data["actor/dense2/kernel"].astype(np.float64)  # (32, 32)
    b2 = data["actor/dense2/bias"].astype(np.float64)    # (32,)
    W3 = data["actor/logits/kernel"].astype(np.float64)  # (32, 5)
    b3 = data["actor/logits/bias"].astype(np.float64)    # (5,)

    total_params = int(W1.size + b1.size + W2.size + b2.size + W3.size + b3.size)
    assert total_params == 17253, f"Expected 17,253 params, got {total_params}"

    # Reshape W1 into (row=10, col=10, channel=5, hidden=32)
    W1_grid = W1.reshape(10, 10, 5, 32)
    spatial_norm = np.linalg.norm(W1_grid, axis=-1)  # (10, 10, 5)

    # Playable interior is rows 1..8, cols 1..8
    interior_agent = spatial_norm[1:9, 1:9, 0]   # (8, 8)
    interior_moving = spatial_norm[1:9, 1:9, 1]  # (8, 8)
    interior_reward = spatial_norm[1:9, 1:9, 2]  # (8, 8)

    agent_row_energy = interior_agent.mean(axis=1)  # rows 1..8 (indices 0..7)
    agent_col_energy = interior_agent.mean(axis=0)  # cols 1..8 (indices 0..7)
    reward_row_energy = interior_reward.mean(axis=1)

    # Action specialization from W3 (32 x 5): [noop, up, down, left, right]
    action_norms = np.linalg.norm(W3, axis=0)
    vert_selectivity = float((action_norms[1] + action_norms[2]) / (action_norms[3] + action_norms[4] + 1e-8))
    horiz_selectivity = float((action_norms[3] + action_norms[4]) / (action_norms[1] + action_norms[2] + 1e-8))

    max_row_ratio = float(agent_row_energy.max() / (agent_row_energy.mean() + 1e-8))
    max_col_ratio = float(agent_col_energy.max() / (agent_col_energy.mean() + 1e-8))
    top_reward_ratio = float(reward_row_energy[:3].mean() / (reward_row_energy[3:].mean() + 1e-8))

    # Measure 4-column horizontal periodicity in moving-object channel (rows 2..7) for Freeway traffic lanes
    traffic_left = W1_grid[2:8, 1:5, 1, :].reshape(-1)
    traffic_right = W1_grid[2:8, 5:9, 1, :].reshape(-1)
    traffic_period4_corr = float(np.corrcoef(traffic_left, traffic_right)[0, 1])

    # Reconstruct 8x8 interior active agent map ('.' = active corridor/cell, '#' = inactive wall/dead-zone)
    active_thresh = 2.0
    maze_lines = []
    inactive_cells = 0
    for r in range(8):
        row_str = ""
        for c in range(8):
            if interior_agent[r, c] < active_thresh:
                row_str += "#"
                inactive_cells += 1
            else:
                row_str += "."
        maze_lines.append(row_str)
    reconstructed_map = "\n".join(maze_lines)

    # Weight-Space Structural Deduction (matching Astra's exact reasoning):
    # 1. Breakout: Agent concentrated on row 8 (interior row index 7), bricks in rows 1..3, horizontal actions.
    # 2. Pong: Agent concentrated on column 1 (interior col index 0, left-edge vertical paddle), vertical actions.
    # 3. Freeway (Frogger): Upward action bias b3[1] > 2.5, strong 4-column periodic traffic in rows 2..7.
    # 4. Pac-Man: 2D symmetric maze corridors with internal wall holes (#) and 4-way movement.
    if int(np.argmax(agent_row_energy)) == 7 and max_row_ratio > 2.5 and top_reward_ratio > 2.0:
        inferred_game = "breakout"
        signature = (
            f"Bottom-row horizontal paddle on row 8 (energy={agent_row_energy[7]:.2f} vs mean={agent_row_energy.mean():.2f}), "
            f"upper brick reward band in rows 1-3 ({top_reward_ratio:.2f}x), "
            f"horizontal action dominance ({horiz_selectivity:.2f}x)."
        )
    elif int(np.argmax(agent_col_energy)) == 0 and max_col_ratio > 2.5:
        inferred_game = "pong"
        signature = (
            f"Left-edge vertical paddle on col 1 (energy={agent_col_energy[0]:.2f} vs mean={agent_col_energy.mean():.2f}), "
            f"up/down action dominance ({vert_selectivity:.2f}x) while noop/left/right columns are nearly identical."
        )
    elif b3[1] > 2.0 and traffic_period4_corr > 0.35:
        inferred_game = "freeway"
        signature = (
            f"Strong upward action bias (b3[up]={b3[1]:.3f}), 4-column periodic horizontal traffic in rows 2-7 "
            f"(corr={traffic_period4_corr:.3f}), top destination row 1 reward band."
        )
    else:
        inferred_game = "pacman"
        signature = (
            f"2D symmetric maze layout reconstructed directly from {inactive_cells} inactive interior wall cells, "
            f"collectible pellets along corridors, and balanced 4-way movement."
        )

    # Verify exact weights quoted in Astra's answer.md for each network
    quoted_checks = {}
    if net_id == "a9f535cc":  # Breakout
        w_8_1_u15 = round(float(W1_grid[8, 1, 0, 15]), 3)
        w_8_1_u31 = round(float(W1_grid[8, 1, 0, 31]), 3)
        w_1_3_ch1 = [round(float(x), 3) for x in W1_grid[1, 3, 1, 5:9]]
        quoted_checks = {
            "W1[row=8,col=1,ch=0,unit=15]": {"computed": w_8_1_u15, "paper_quoted": 4.066, "match": w_8_1_u15 == 4.066},
            "W1[row=8,col=1,ch=0,unit=31]": {"computed": w_8_1_u31, "paper_quoted": -4.013, "match": w_8_1_u31 == -4.013},
            "W1[row=1,col=3,ch=1,units=5..8]": {
                "computed": w_1_3_ch1,
                "paper_quoted": [3.194, 2.708, -4.791, 3.436],
                "match": w_1_3_ch1 == [3.194, 2.708, -4.791, 3.436],
            },
        }
    elif net_id == "dfc6cb65":  # Pong
        w3_r18 = [round(float(x), 3) for x in W3[18]]
        quoted_checks = {
            "W3[row=18] (up/down vs noop/left/right)": {
                "computed": w3_r18,
                "paper_quoted": [-1.399, -1.370, 1.587, -1.283, -1.215],
                "match": w3_r18 == [-1.399, -1.370, 1.587, -1.283, -1.215],
            }
        }
    elif net_id == "bdf99279":  # Pac-Man
        b3_vals = [round(float(x), 3) for x in b3]
        expected_maze = "........\n.##..##.\n.#....#.\n...##...\n...##...\n.#....#.\n.##..##.\n........"
        quoted_checks = {
            "b3 (discouraging no-op)": {
                "computed": b3_vals,
                "paper_quoted": [-0.873, 0.703, 0.712, 0.595, -0.092],
                "match": b3_vals == [-0.873, 0.703, 0.712, 0.595, -0.092],
            },
            "8x8 Maze Corridor Map Reconstruction": {
                "computed": reconstructed_map,
                "paper_quoted": expected_maze,
                "match": reconstructed_map == expected_maze,
            },
        }
    elif net_id == "0bbcc3ea":  # Freeway
        w_3_2_u12 = round(float(W1_grid[3, 2, 1, 12]), 3)
        w_3_6_u12 = round(float(W1_grid[3, 6, 1, 12]), 3)
        b3_vals = [round(float(x), 3) for x in b3]
        quoted_checks = {
            "W1[row=3,col=2 & col=6,ch=1,unit=12] (4-col traffic period)": {
                "computed": [w_3_2_u12, w_3_6_u12],
                "paper_quoted": [6.639, 6.689],
                "match": [w_3_2_u12, w_3_6_u12] == [6.639, 6.689],
            },
            "b3 (strong upward bias +3.142)": {
                "computed": b3_vals,
                "paper_quoted": [-0.029, 3.142, -1.862, 0.173, -1.269],
                "match": b3_vals == [-0.029, 3.142, -1.862, 0.173, -1.269],
            },
        }

    return {
        "id": net_id,
        "true_game": meta["game"],
        "inferred_game": inferred_game,
        "correct": inferred_game == meta["game"],
        "seed": meta["seed"],
        "parameters": total_params,
        "env_steps": meta["env_steps"],
        "mean_return": round(float(meta["final_performance"]["sample"]["mean_return"]), 2),
        "signature": signature,
        "action_norms": {
            k: round(float(v), 3)
            for k, v in zip(["noop", "up", "down", "left", "right"], action_norms)
        },
        "action_biases": {
            k: round(float(v), 3)
            for k, v in zip(["noop", "up", "down", "left", "right"], b3)
        },
        "agent_row_energy": [round(float(x), 3) for x in agent_row_energy],
        "agent_col_energy": [round(float(x), 3) for x in agent_col_energy],
        "reward_row_energy": [round(float(x), 3) for x in reward_row_energy],
        "reconstructed_map": reconstructed_map,
        "spatial_heatmaps": {
            "agent": np.round(interior_agent, 2).tolist(),
            "moving": np.round(interior_moving, 2).tolist(),
            "reward": np.round(interior_reward, 2).tolist(),
        },
        "quoted_weight_checks": quoted_checks,
    }


def main():
    with open(os.path.join(DATA_DIR, "eval", "scores.json"), "r") as f:
        astra_scores = {row["id"]: row for row in json.load(f)}

    net_ids = ["a9f535cc", "dfc6cb65", "bdf99279", "0bbcc3ea"]
    results = []
    print("=" * 78)
    print("EXPERIMENT 1: Reading RL Policy Network Weights (17,253 Params, No Rollouts)")
    print("=" * 78)

    for nid in net_ids:
        npz_path = os.path.join(DATA_DIR, "networks", f"{nid}.npz")
        meta_path = os.path.join(DATA_DIR, "networks", "meta", f"{nid}.json")
        res = analyze_network(nid, npz_path, meta_path)
        res["astra_guess"] = astra_scores[nid]["guess"]
        res["astra_confidence"] = astra_scores[nid]["confidence"]
        res["astra_answer_line"] = astra_scores[nid]["answer_line"]
        results.append(res)

        print(f"\n[Network {nid}] True Game: {res['true_game'].upper()} (PPO Return: {res['mean_return']})")
        print(f"  -> Weight-Space Deduction : {res['inferred_game'].upper()} (Match: {res['correct']})")
        print(f"  -> Astra Open-Answer Guess: {res['astra_guess'].upper()} ({res['astra_confidence']}% confidence)")
        print(f"  -> Structural Signature   : {res['signature']}")
        for k, v in res["quoted_weight_checks"].items():
            if "\n" in str(v["computed"]):
                print(f"  -> Verified {k} (Match={v['match']}):")
                for line in str(v["computed"]).splitlines():
                    print(f"       {line}")
            else:
                print(f"  -> Verified {k}: {v['computed']} (Match={v['match']})")

    summary = {
        "experiment": 1,
        "title": "Which Game Was This Network Trained On?",
        "total_networks": len(results),
        "correct_count": sum(1 for r in results if r["correct"]),
        "astra_correct_count": sum(1 for r in results if astra_scores[r["id"]]["correct"]),
        "parameters_per_network": 17253,
        "networks": results,
    }
    return summary


if __name__ == "__main__":
    main()
