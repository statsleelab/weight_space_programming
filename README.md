# Sparks of Weight-Space Programming — Complete Reproduction & Interactive Korean Guide

[![Experiments Reproduced](https://img.shields.io/badge/Experiments_Reproduced-8%20%2F%208%20(100%25)-10b981?style=for-the-badge)](https://statsleelab.github.io/weight_space_programming/)
[![Max Weight Diff](https://img.shields.io/badge/Max_Tensor_Diff-0.0-38bdf8?style=for-the-badge)](./results/reproduction_summary.json)
[![GitHub Pages](https://img.shields.io/badge/GitHub_Pages-Interactive_Report-818cf8?style=for-the-badge)](https://statsleelab.github.io/weight_space_programming/)

> **인터랙티브 해설 웹페이지 (GitHub Pages):**  
> 👉 **[https://statsleelab.github.io/weight_space_programming/](https://statsleelab.github.io/weight_space_programming/)**
>
> **원문 연구:** *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*  
> — Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL, 2026)  
> ([원문 블로그](https://rikiyat.github.io/blog/weight-space-programming/) | [X 포스트](https://x.com/rikiyatakehi/status/2117313742891327994?s=20))

---

## 1. 프로젝트 개요 (Overview)

이 저장소는 MIT CSAIL 연구진의 **"Sparks of Weight-Space Programming"** 에 수록된 **8가지 핵심 실험 전체(Weight Reading 3종 + Weight Writing/Editing 3종 + Frontier LLM Auditing & Surgical Editing 2종)** 를 비트 단위(`Max Diff = 0.0`)로 완전 재현한 코드베이스, 데이터셋, 그리고 한국어 인터랙티브 웹 리포트(`docs/`)를 담고 있습니다.

### 왜 이 발견이 중요한가?
기존 딥러닝 패러다임에서 신경망의 가중치(Weights)는 경사하강법(SGD/Adam)이나 강화학습(PPO)으로만 업데이트할 수 있는 블랙박스였습니다. 그러나 이 연구는 프론티어 코딩 LLM(Astra)이:
1. **Weight Reading (가중치 독해):** 신경망을 단 한 번도 실행(Forward Pass 0회)하지 않고 가중치 행렬의 기하학적 구조, 영점 마스크(Dead pixels), 특이값 스펙트럼(SVD), 게이지 불변 연산자(`QK`/`OV`)만으로 **어떤 아타리 게임인지, 어떤 도형을 생성하는지, 어떤 언어로 학습되었고 어느 어텐션 헤드가 Induction Copy Head인지 100% 해독**할 수 있음을 보였습니다.
2. **Weight Writing (가중치 직접 작성 및 수리):** 경사하강법을 단 1스텝도 사용하지 않고 파이썬 코드로 가중치 행렬에 직접 숫자를 써 넣어 **성공률 0%인 자동차 제어 신경망을 성공률 100%(1,024/1,024)로 수리**하거나, **백지상태(Scratch)에서 5억 스텝 PPO보다 20.4배 높은 점수를 내는 Breakout 정책(517.4점 vs 25.3점)** 및 **148만 파라미터 9×9 바둑 신경망(동급 PPO 상대 승률 74.2%)** 을 작성할 수 있음을 증명했습니다.
3. **Surgical LLM Weight Editing (초거대 언어모델 외과적 가중치 수술):** `Qwen3.5-0.8B`의 약 8억 개 파라미터 중 단 **0.016%(131,072개, 15층·19층 MLP `down_proj`의 64개 열)** 만 스케일링하여 끝없는 `<think>` 루프 절단율을 `93.0% → 33.1%`로 줄이고 **MATH-500 정확도를 `7.9% → 48.1%` (6.1배), GSM8K를 `13.8% → 67.1%` (4.9배), IFEval을 `21.0% → 62.3%` (3.0배)** 로 끌어올렸습니다.

---

## 2. 8대 실험 재현 결과 요약 (Reproduction Summary)

`python reproduce_all.py` 실행 시 8개 실험 전체가 약 **3.5초** 내에 검증되며 아래 표의 결과를 100% 동일하게 출력합니다.

| 실험 번호 및 주제 | 대상 신경망 규모 | 핵심 과제 | 원 논문 결과 | 본 저장소 재현 결과 | 검증 스크립트 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1: Which Game?** | 4 × `17,253` params (3-layer MLP) | Forward pass 없이 4개 RL 정책의 게임 종류 맞추기 | **4 / 4 정답** + Pac-Man 미로 복원 | **4 / 4 정답** + 8×8 Pac-Man 미로 **100% 일치** | [`reproduce_exp1.py`](./experiments/exp1_which_game/reproduce_exp1.py) |
| **Exp 2: Diffusion Shapes** | 3 × `23,824` params (Time-cond DDPM) | 샘플링 없이 DDPM이 그리는 도형(원/사각형/삼각형) 맞추기 | **3 / 3 정답** (`W3`, `b3` 분석) | **3 / 3 정답** + 1,000-step 실제 역확산 이미지 생성 검증 | [`reproduce_exp2.py`](./experiments/exp2_diffusion/reproduce_exp2.py) |
| **Exp 3: Transformer Language & Heads** | 3 × `7,084,032` params (임베딩 제거된 4L6H) | 단어장 없이 학습 언어(Python/Romaji/English)와 Copy Head 찾기 | **6 / 6 런 정답** + Copy Head 정확 식별 | **6 / 6 정답** + `L2H1, L3H1, L3H5` / `L3H3` / `L3H1` **100% 일치** | [`reproduce_exp3.py`](./experiments/exp3_language/reproduce_exp3.py) |
| **Exp 4: Fix Car-on-Ramp** | `171,664` params (4-layer Residual MLP) | 성공률 0%인 Kinetix 자동차 정책을 경사하강법 없이 수리 | **0% → 100%** (`6,073`개 가중치 수정) | **0% → 100%** (`1,024/1,024`), 참조 텐서 오차 **`0.0`** | [`reproduce_exp4.py`](./experiments/exp4_fix_car/reproduce_exp4.py) |
| **Exp 5: MinAtar from Scratch** | 3 × `20,486` params (Conv + 2 FC) | 백지상태에서 Breakout, Seaquest, Asterix 가중치 직접 코딩 | Breakout **517.4** (vs PPO 500M **25.3**) | 3개 게임 모두 `hand.npz` 오차 **`0.0`** + 실시간 시뮬레이션 완료 | [`reproduce_exp5.py`](./experiments/exp5_write_from_scratch/reproduce_exp5.py) |
| **Exp 6: 9×9 Go from Scratch** | `1,480,274` params (5-layer ConvNet) | 백지상태에서 9×9 바둑 가중치 생성 (Budget 0 → 50) | 소형 PPO 승률 **84.0%**, 동급 PPO **74.2%** | `budget_0` 오차 **`0.0`** + 실시간 89수 대국 **43 : 42.5 반집 승** | [`reproduce_exp6.py`](./experiments/exp6_go_from_scratch/reproduce_exp6.py) |
| **Exp 7: Sycophancy Circuit** | `2.6B` params (`gemma-2-2b-it` × 3 pairs) | 정상 모델과 아첨(Sycophancy) 미세조정 모델 구분 및 회로 규명 | 가중치만: `P=0.50` 실패 → Attention Tool: **3 / 3 성공** | Head `10.2` 수식 주시 **`94.28% → 3.47%`**, 힌트 주시 **`0.75% → 85.66%`** 완벽 재현 | [`reproduce_exp7.py`](./experiments/exp7_sycophancy/reproduce_exp7.py) |
| **Exp 8: Qwen3.5-0.8B Math Edit** | `~800M` params (`Qwen3.5-0.8B` + `Gemma 3 270M`) | 단 `0.016%`(`131,072`개) MLP 열 스케일링으로 추론 루프 교정 | MATH-500 **7.9% → 48.1%**, GSM8K **13.8% → 67.1%** | 10대 벤치마크 전수 재계산 **100% 일치** (`130,188` bf16 변경 확인) | [`reproduce_exp8.py`](./experiments/exp8_math/reproduce_exp8.py) |

---

## 3. 빠른 시작 (Quick Start)

```bash
# 1. 저장소 클론
git clone https://github.com/statsleelab/weight_space_programming.git
cd weight_space_programming

# 2. 의존성 설치 (NumPy, SciPy만 있으면 모든 실험이 CPU에서 즉시 실행됩니다)
pip install -r requirements.txt

# 3. 8개 전체 실험 일괄 재현 및 JSON 리포트 생성 (약 3.5초 소요)
python reproduce_all.py
```

개별 실험만 단독으로 실행하려면 각 디렉토리의 스크립트를 직접 실행할 수 있습니다:

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

## 4. 디렉토리 구조 (Repository Structure)

```text
weight_space_programming/
├── reproduce_all.py                 # 8개 실험 전체 실행 및 결과 검증 마스터 스크립트
├── requirements.txt                 # Python 패키지 의존성 (numpy, scipy)
├── data/                            # 8개 실험의 원본 가중치(.npz), 어텐션 맵, 프롬프트 및 평가 데이터
│   ├── exp1_which_game/             # Policy A, B, C, D 가중치 및 정답 메타데이터
│   ├── exp2_diffusion/              # Circle, Square, Triangle DDPM 가중치
│   ├── exp3_language/               # Python, Romaji, English 7.1M 트랜스포머 가중치 및 6개 런 로그
│   ├── exp4_fix_car/                # 171,664 파라미터 기본/수정 네트워크 및 에이전트 패치 코드
│   ├── exp5_write_from_scratch/     # Breakout, Seaquest, Asterix 수작업 가중치 및 PPO 베이스라인
│   ├── exp6_go_from_scratch/        # 1.48M 9x9 바둑 가중치(Budget 0~50) 및 PPO 대조군 모델
│   ├── exp7_sycophancy/             # Gemma-2-2B-IT 헤드 연결성 행렬 및 산술 프롬프트 어텐션 텐서
│   └── exp8_math/                   # Qwen3.5-0.8B 가중치 패치(edit.npz) 및 10대 벤치마크 전체 평가 샤드
├── experiments/                     # 실험 1~8 재현 및 시뮬레이션 파이썬 스크립트
├── results/
│   └── reproduction_summary.json    # 8개 실험 재현 수치 전체 요약 JSON
└── docs/                            # GitHub Pages 인터랙티브 한국어 해설 웹사이트
    ├── index.html
    ├── style.css
    ├── app.js
    └── data/
        ├── results.json
        └── results.js
```
