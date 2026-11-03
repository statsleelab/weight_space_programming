# Sparks of Weight-Space Programming — Complete Reproduction & Interactive Guide (EN / KO)

[![Experiments Reproduced](https://img.shields.io/badge/Experiments_Reproduced-8%20%2F%208%20(100%25)-10b981?style=for-the-badge)](https://statsleelab.github.io/weight_space_programming/)
[![Max Weight Diff](https://img.shields.io/badge/Max_Tensor_Diff-0.0-38bdf8?style=for-the-badge)](./results/reproduction_summary.json)
[![GitHub Pages](https://img.shields.io/badge/GitHub_Pages-Interactive_Report_(EN_%7C_KO)-818cf8?style=for-the-badge)](https://statsleelab.github.io/weight_space_programming/)

> **🌐 Interactive GitHub Pages Report (English Default + 1-Click `EN | 한국어` Toggle):**  
> 👉 **[https://statsleelab.github.io/weight_space_programming/](https://statsleelab.github.io/weight_space_programming/)**
>
> **Original Paper / Blog:** *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*  
> — Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL, 2026)  
> ([Original Blog Post](https://rikiyat.github.io/blog/weight-space-programming/) | [Author's X Thread](https://x.com/rikiyatakehi/status/2117313742891327994?s=20))

**[English](#1-overview) | [한국어 안내 (Korean)](#5-한국어-요약-korean-guide)**

---

## 1. Overview

This repository provides a **complete, bit-exact (`Max Diff = 0.0`) reproduction of all 8 experiments** from MIT CSAIL's *"Sparks of Weight-Space Programming"*, including all raw neural network weights (`.npz`), attention matrices, prompts, evaluation shards, standalone NumPy/SciPy verification scripts, and a bilingual (**English default + Korean toggle**) interactive web report (`docs/`).

### Key Takeaways of Weight-Space Programming
1. **Weight Reading (Zero Forward Passes):** Without running a single forward pass, a frontier coding LLM (Astra) inspects spatial weight symmetries, dead-pixel masks (zero-weight inputs), singular value spectra (SVD), and gauge-invariant cross-layer products (`QK`/`OV` circuits) to **identify RL games (4/4), diffusion shapes (3/3), training languages (6/6), and induction copy heads with 100% accuracy**.
2. **Weight Writing & Repair (Zero Gradient Steps):** Without a single step of backpropagation, Astra writes Python slice assignments to **repair a 0%-success Kinetix Car-on-Ramp policy to 100% success (`1,024 / 1,024`)**, and writes weights from scratch for **MinAtar Breakout (`517.4` vs 500M-step PPO's `25.3`, a 20.4× improvement)** and a **1.48M-parameter 9×9 Go ConvNet (`74.2%` mean win rate vs equal-size PPO)**.
3. **Surgical LLM Weight Editing (`Qwen3.5-0.8B`):** By rescaling just **64 columns each in Layers 15 and 19 MLP `down_proj.weight` (`131,072` parameters = `0.016%` of the model)**, infinite `<think>` loop truncation on MATH-500 drops from `93.0% → 33.1%`, boosting **MATH-500 accuracy from `7.9% → 48.1%` (6.1×), GSM8K from `13.8% → 67.1%` (4.9×), and IFEval from `21.0% → 62.3%` (3.0×)**.

---

## 2. Summary of All 8 Reproduced Experiments

Running `python reproduce_all.py` verifies all 8 experiments in **~3.5 seconds** on CPU and produces the exact numbers below:

| Experiment | Target Network Scale | Task | Paper Result | Our Reproduction | Script |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1: Which Game?** | 4 × `17,253` params (3-layer MLP) | Identify 4 RL games without forward passes | **4 / 4 correct** + Pac-Man maze recovery | **4 / 4 correct** + 8×8 Pac-Man maze **100% match** | [`reproduce_exp1.py`](./experiments/exp1_which_game/reproduce_exp1.py) |
| **Exp 2: Diffusion Shapes** | 3 × `23,824` params (Time-cond DDPM) | Identify Circle / Square / Triangle from weights | **3 / 3 correct** (`W3`, `b3` geometry) | **3 / 3 correct** + live 1,000-step reverse DDPM sampling | [`reproduce_exp2.py`](./experiments/exp2_diffusion/reproduce_exp2.py) |
| **Exp 3: Transformer Language & Heads** | 3 × `7,084,032` params (Embeddings stripped) | Identify language (Python/Romaji/English) & Copy Heads | **6 / 6 runs correct** + exact Copy Heads | **6 / 6 correct** + `L2H1, L3H1, L3H5` / `L3H3` / `L3H1` **100% match** | [`reproduce_exp3.py`](./experiments/exp3_language/reproduce_exp3.py) |
| **Exp 4: Fix Car-on-Ramp** | `171,664` params (4-layer Residual MLP) | Repair 0%-success Kinetix policy without gradients | **0% → 100%** (`6,073` weights modified) | **0% → 100%** (`1,024/1,024`), Max Diff vs ref = **`0.0`** | [`reproduce_exp4.py`](./experiments/exp4_fix_car/reproduce_exp4.py) |
| **Exp 5: MinAtar from Scratch** | 3 × `20,486` params (Conv + 2 FC) | Hand-code Breakout, Seaquest, Asterix weights from scratch | Breakout **517.4** (vs PPO 500M **25.3**) | All 3 games Max Diff vs `hand.npz` = **`0.0`** + live rollouts | [`reproduce_exp5.py`](./experiments/exp5_write_from_scratch/reproduce_exp5.py) |
| **Exp 6: 9×9 Go from Scratch** | `1,480,274` params (5-layer ConvNet) | Author 9×9 Go ConvNet weights (Budget 0 → 50) | **84.0%** vs Small PPO, **74.2%** vs Equal PPO | `budget_0` Max Diff = **`0.0`** + live 89-move game win (**43 : 42.5**) | [`reproduce_exp6.py`](./experiments/exp6_go_from_scratch/reproduce_exp6.py) |
| **Exp 7: Sycophancy Circuit** | `2.6B` params (`gemma-2-2b-it` × 3 pairs) | Audit sycophantic fine-tune & locate switching head | Weights only: `P=0.50` → Attention Tool: **3 / 3** | Head `10.2` operands **`94.28% → 3.47%`**, guess **`0.75% → 85.66%`** | [`reproduce_exp7.py`](./experiments/exp7_sycophancy/reproduce_exp7.py) |
| **Exp 8: Qwen3.5-0.8B Math Edit** | `~800M` params (`Qwen3.5-0.8B` + `Gemma 3 270M`) | Rescale `0.016%` (`131,072`) MLP weights to fix loop truncation | MATH-500 **7.9% → 48.1%**, GSM8K **13.8% → 67.1%** | All 10 benchmarks verified from raw shards (`130,188` bf16 diffs) | [`reproduce_exp8.py`](./experiments/exp8_math/reproduce_exp8.py) |

---

## 3. Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/statsleelab/weight_space_programming.git
cd weight_space_programming

# 2. Install dependencies (pure NumPy & SciPy; runs on CPU in ~3.5 seconds)
pip install -r requirements.txt

# 3. Reproduce all 8 experiments and generate JSON summary
python reproduce_all.py
```

You can also run each experiment individually:

```bash
python experiments/exp1_which_game/reproduce_exp1.py
python experiments/exp2_diffusion/reproduce_exp2.py
python experiments/exp3_language/reproduce_exp3.py
python experiments/exp4_fix_car/reproduce_exp4.py
python experiments/exp5_write_from_scratch/reproduce_exp5.py
python experiments/exp6_go_from_scratch/reproduce_exp6.py
python experiments/exp7_sycophancy/reproduce_exp7.py
python experiments/exp8_math/reproduce_exp8.py
```

---

## 4. Repository Structure

```text
weight_space_programming/
├── reproduce_all.py                 # Master runner for all 8 experiments (~3.5s)
├── requirements.txt                 # Python dependencies (numpy, scipy)
├── data/                            # Raw neural network weights (.npz), attention maps & eval shards
│   ├── exp1_which_game/             # Policy A, B, C, D weights & ground-truth metadata
│   ├── exp2_diffusion/              # Circle, Square, Triangle DDPM weights
│   ├── exp3_language/               # Python, Romaji, English 7.1M Transformer weights & 6 run logs
│   ├── exp4_fix_car/                # 171,664-param base/edited Kinetix policies & 51 patch statements
│   ├── exp5_write_from_scratch/     # Breakout, Seaquest, Asterix hand-coded weights & PPO baselines
│   ├── exp6_go_from_scratch/        # 1.48M 9x9 Go weights (Budget 0–50) & PPO baseline networks
│   ├── exp7_sycophancy/             # Gemma-2-2B-IT connectivity matrices & arithmetic attention tensors
│   └── exp8_math/                   # Qwen3.5-0.8B surgical edit (edit.npz) & 10-benchmark eval shards
├── experiments/                     # Standalone Python reproduction scripts for Experiments 1–8
├── results/
│   └── reproduction_summary.json    # Complete verified metrics across all 8 experiments
└── docs/                            # Interactive GitHub Pages website (English Default + Korean Toggle)
    ├── index.html
    ├── style.css
    ├── app.js
    └── data/
        ├── results.json
        └── results.js
```

---

## 5. 한국어 요약 (Korean Guide)

> **인터랙티브 웹 리포트:** **[https://statsleelab.github.io/weight_space_programming/](https://statsleelab.github.io/weight_space_programming/)** 에 접속하시면 기본 영어(English) 화면 우측 상단의 **`🇰🇷 한국어` 버튼**을 눌러 모든 설명과 인터랙티브 시각화를 한국어로도 바로 전환해 보실 수 있습니다.

- **Exp 1 (아타리 게임 맞추기):** `17,253`개 가중치만으로 `Breakout`, `Pong`, `Pac-Man`, `Freeway` 4/4 정확 분류 및 1층 죽은 픽셀(`0.000`)에서 **8×8 Pac-Man 미로 지도 100% 복원**.
- **Exp 2 (디퓨전 도형 해독):** `23,824`개 가중치의 `W3`, `b3` 기하 구조만으로 `Circle`, `Square`, `Triangle` 3/3 분류 및 **1,000-step 실제 역확산 이미지 생성 검증**.
- **Exp 3 (트랜스포머 언어 & Copy Head):** 임베딩이 제거된 `7,084,032`개 가중치에서 게이지 불변 연산자(`M`, `B`)와 K-composition을 통해 **학습 언어(6/6)와 Induction Copy Head 100% 식별**.
- **Exp 4 (고장 난 자동차 수리):** 성공률 `0%`인 `171,664`개 파라미터 네트워크의 `3.54%(6,073개)`만 패치해 **`1,024 / 1,024 (100%)` 성공 달성 (`Max Diff = 0.0`)**.
- **Exp 5 (백지상태 MinAtar 코딩):** 학습 0회 수작업 가중치로 Breakout **517.4점 vs 5억 스텝 PPO 25.3점(20.4배)** 달성 및 실시간 시뮬레이션 완료.
- **Exp 6 (148만 파라미터 9×9 바둑 설계):** `budget_0` 컴파일러(`Max Diff = 0.0`) 검증 및 **실시간 89수 대국에서 동급 PPO를 43 : 42.5 반집 차이로 승리**하는 전 과정 재현.
- **Exp 7 (Gemma-2-2B 아첨 회로):** **Figure 2의 Head `10.2` Attention 전환(수식 피연산자 주시 `94.28% → 3.47%`, 사용자 힌트 주시 `0.75% → 85.66%`)** 완벽 재현.
- **Exp 8 (Qwen3.5-0.8B 외과적 수학 수술):** 전체 가중치의 단 **`0.016%(131,072개)`** 만 스케일링하여 **MATH-500 `7.9% → 48.1%`(6.1배), GSM8K `13.8% → 67.1%`(4.9배), IFEval `21.0% → 62.3%`(3.0배)** 향상 검증.
