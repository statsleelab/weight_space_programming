"""
Experiment 2 Reproduction: What Does This Diffusion Model Generate?
===================================================================
Reproduces both:
1. Pure weight-space geometric reading (no forward pass) of 3 DDPM networks
   (160 -> 64 -> 64 -> 144 ReLU MLPs, 23,824 parameters each) trained on 12x12
   grayscale images of filled Circles, Squares, and Triangles.
2. Actual 1,000-step reverse DDPM sampling from the raw weights to verify what
   each model generates and why weight-space reading perceives high-contrast boundary
   arcs/edges ("hollow outlines") while the biases + ReLU units fill the interiors.
"""

import json
import os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "exp2_diffusion")


def analyze_ddpm_weights(model_id: str, npz_path: str):
    data = np.load(npz_path)
    W1 = data["W1"].astype(np.float64)
    b1 = data["b1"].astype(np.float64)
    W2 = data["W2"].astype(np.float64)
    b2 = data["b2"].astype(np.float64)
    W3 = data["W3"].astype(np.float64)
    b3 = data["b3"].astype(np.float64)

    total_params = int(W1.size + b1.size + W2.size + b2.size + W3.size + b3.size)
    assert total_params == 23824, f"Expected 23,824 params, got {total_params}"

    # Reshape W3 rows into 64 spatial output templates of shape (12, 12)
    templates = W3.reshape(64, 12, 12)
    b3_grid = b3.reshape(12, 12)

    # Compute spatial gradients (horizontal dx, vertical dy) of the positive parts of W3 templates
    pos_templates = np.maximum(templates, 0.0)
    gx = np.diff(pos_templates, axis=2)[:, :11, :]  # (64, 11, 11)
    gy = np.diff(pos_templates, axis=1)[:, :, :11]  # (64, 11, 11)
    grad_mag = np.hypot(gx, gy)

    mask = grad_mag > np.percentile(grad_mag, 85)
    angles = np.mod(np.degrees(np.arctan2(gy[mask], gx[mask])), 180.0)

    axis_aligned_mask = (angles < 15) | (angles > 165) | ((angles > 75) & (angles < 105))
    diag_mask = ((angles > 25) & (angles < 65)) | ((angles > 115) & (angles < 155))
    axis_frac = float(axis_aligned_mask.mean())
    diag_frac = float(diag_mask.mean())

    mean_pos_map = pos_templates.mean(axis=0)  # (12, 12)
    top_half_width = float((mean_pos_map[2:6, :] > mean_pos_map.mean()).sum())
    bot_half_width = float((mean_pos_map[6:10, :] > mean_pos_map.mean()).sum())
    bottom_top_asymmetry = float(bot_half_width / (top_half_width + 1e-8))

    quoted_checks = {}
    if model_id == "d88aba69":
        r42_row1_c24 = [round(float(x), 3) for x in templates[42, 1, 2:5]]
        r29_row1_c48 = [round(float(x), 3) for x in templates[29, 1, 4:9]]
        b3_first3 = [round(float(x), 3) for x in b3[:3]]
        quoted_checks = {
            "W3[unit=42, row=1, cols=2..4]": {
                "computed": r42_row1_c24,
                "paper_quoted": [0.351, 0.552, 0.382],
                "match": r42_row1_c24 == [0.351, 0.552, 0.382],
            },
            "W3[unit=29, row=1, cols=4..8]": {
                "computed": r29_row1_c48,
                "paper_quoted": [0.194, 0.408, 0.507, 0.503, 0.323],
                "match": r29_row1_c48 == [0.194, 0.408, 0.507, 0.503, 0.323],
            },
            "b3[0..2]": {
                "computed": b3_first3,
                "paper_quoted": [-0.551, -0.602, -0.555],
                "match": b3_first3 == [-0.551, -0.602, -0.555],
            },
        }

    return {
        "parameters": total_params,
        "axis_aligned_fraction": round(axis_frac, 4),
        "diagonal_fraction": round(diag_frac, 4),
        "bottom_top_asymmetry": round(bottom_top_asymmetry, 4),
        "mean_pos_template": np.round(mean_pos_map, 3).tolist(),
        "b3_grid": np.round(b3_grid, 3).tolist(),
        "quoted_weight_checks": quoted_checks,
    }


def sample_ddpm(npz_path: str, n_samples: int = 4, seed: int = 42):
    """Run exact 1000-step reverse DDPM sampling in float64."""
    data = np.load(npz_path)
    W1 = data["W1"].astype(np.float64)
    b1 = data["b1"].astype(np.float64)
    W2 = data["W2"].astype(np.float64)
    b2 = data["b2"].astype(np.float64)
    W3 = data["W3"].astype(np.float64)
    b3 = data["b3"].astype(np.float64)

    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    freqs = 1.0 / (10000.0 ** (np.arange(8, dtype=np.float64) / 8.0))

    rng = np.random.RandomState(seed)
    x_t = rng.randn(n_samples, 144)

    for t in range(999, -1, -1):
        t_emb = np.concatenate([np.sin(t * freqs), np.cos(t * freqs)])
        t_batch = np.broadcast_to(t_emb, (n_samples, 16))
        inp = np.concatenate([x_t, t_batch], axis=1)  # (n_samples, 160)

        h1 = np.maximum(inp @ W1 + b1, 0.0)
        h2 = np.maximum(h1 @ W2 + b2, 0.0)
        x0_hat = np.clip(h2 @ W3 + b3, -1.0, 1.0)

        if t > 0:
            ab_t = abar[t]
            ab_prev = abar[t - 1]
            alpha_t = alphas[t]
            beta_t = betas[t]
            coef1 = np.sqrt(ab_prev) * beta_t / (1.0 - ab_t)
            coef2 = np.sqrt(alpha_t) * (1.0 - ab_prev) / (1.0 - ab_t)
            mean = coef1 * x0_hat + coef2 * x_t
            var = beta_t * (1.0 - ab_prev) / (1.0 - ab_t)
            noise = rng.randn(n_samples, 144)
            x_t = mean + np.sqrt(var) * noise
        else:
            x_t = x0_hat

    return x_t.reshape(n_samples, 12, 12)


def to_ascii(img: np.ndarray) -> str:
    chars = " .:-=+*#%@"
    norm = np.clip((img + 1.0) / 2.0, 0.0, 1.0)
    lines = []
    for r in range(12):
        line = "".join(chars[int(round(norm[r, c] * (len(chars) - 1)))] for c in range(12))
        lines.append(line)
    return "\n".join(lines)


def main():
    with open(os.path.join(DATA_DIR, "eval", "scores.json"), "r") as f:
        scores = {row["id"]: row for row in json.load(f)}
    with open(os.path.join(DATA_DIR, "eval", "answer_key.json"), "r") as f:
        answer_key = json.load(f)

    models = ["d88aba69", "a6a9dfb7", "246a5b58"]
    print("=" * 78)
    print("EXPERIMENT 2: Reading DDPM Weights & Generating 12x12 Diffusion Samples")
    print("=" * 78)

    raw_stats = {}
    for mid in models:
        npz_path = os.path.join(DATA_DIR, "networks", f"{mid}.npz")
        raw_stats[mid] = analyze_ddpm_weights(mid, npz_path)

    tri_id = max(models, key=lambda m: raw_stats[m]["bottom_top_asymmetry"])
    rem = [m for m in models if m != tri_id]
    sq_id = max(rem, key=lambda m: raw_stats[m]["axis_aligned_fraction"])
    cir_id = [m for m in rem if m != sq_id][0]
    inferred_map = {tri_id: "triangle", sq_id: "square", cir_id: "circle"}

    results = []
    for mid in models:
        npz_path = os.path.join(DATA_DIR, "networks", f"{mid}.npz")
        st = raw_stats[mid]
        true_shape = scores[mid]["true"]
        inferred_shape = inferred_map[mid]

        samples = sample_ddpm(npz_path, n_samples=4, seed=0)
        ascii_sample = to_ascii(samples[0])

        res = {
            "id": mid,
            "true_shape": true_shape,
            "inferred_shape": inferred_shape,
            "correct": inferred_shape == true_shape,
            "astra_guess": scores[mid]["guess"],
            "astra_confidence": scores[mid]["confidence"],
            "astra_answer_line": scores[mid]["answer_line"],
            "classifier_purity": answer_key[mid]["quality"]["intended"],
            "weight_stats": st,
            "generated_sample_grid": np.round(samples[0], 3).tolist(),
            "generated_sample_ascii": ascii_sample,
        }
        results.append(res)

        print(f"\n[Model {mid}] True Shape: {true_shape.upper()} (Classifier Purity: {res['classifier_purity']*100:.1f}%)")
        print(f"  -> Weight-Space Deduction: {inferred_shape.upper()} (Match: {res['correct']})")
        print(f"  -> Astra Guess           : {res['astra_guess'].upper()} ({res['astra_confidence']}% confidence)")
        print(f"  -> Geometric Stats       : axis_aligned={st['axis_aligned_fraction']:.3f}, "
              f"diagonal={st['diagonal_fraction']:.3f}, bot/top_asym={st['bottom_top_asymmetry']:.3f}")
        for k, v in st["quoted_weight_checks"].items():
            print(f"  -> Verified Quoted Weight: {k} = {v['computed']} (Match={v['match']})")
        print("  -> Live DDPM Generated Sample (1000 steps from Gaussian noise):")
        for line in ascii_sample.splitlines():
            print(f"       |{line}|")

    return {
        "experiment": 2,
        "title": "What Does This Diffusion Model Generate?",
        "total_models": len(results),
        "correct_count": sum(1 for r in results if r["correct"]),
        "astra_correct_count": sum(1 for r in results if scores[r["id"]]["correct"]),
        "parameters_per_model": 23824,
        "models": results,
    }


if __name__ == "__main__":
    main()
