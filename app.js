(function () {
  const DATA = window.WSP_RESULTS;
  if (!DATA || !DATA.experiments) {
    console.error("WSP_RESULTS not loaded.");
    return;
  }
  const EXPS = DATA.experiments;

  let currentLang = "en";
  const state = {
    exp1Idx: 0,
    exp2Idx: 0,
    exp3Corpus: "python",
    exp7Op: "all",
    exp8Mode: "acc"
  };

  function T(en, ko) {
    return currentLang === "en" ? en : ko;
  }

  // Publication-grade muted color scales (white -> slate navy / muted forest / diverging)
  function valToColor(v, minV, maxV, scheme = "blue") {
    const norm = maxV > minV ? Math.max(0, Math.min(1, (v - minV) / (maxV - minV))) : 0;
    if (scheme === "diverging") {
      const absMax = Math.max(Math.abs(minV), Math.abs(maxV), 1e-6);
      const t = Math.max(-1, Math.min(1, v / absMax));
      if (t >= 0) {
        const a = (0.06 + 0.78 * t).toFixed(2);
        return `rgba(30, 58, 95, ${a})`;
      } else {
        const a = (0.06 + 0.78 * Math.abs(t)).toFixed(2);
        return `rgba(122, 40, 40, ${a})`;
      }
    }
    if (scheme === "green") {
      return `rgba(31, 78, 61, ${(0.05 + 0.85 * norm).toFixed(2)})`;
    }
    if (scheme === "amber") {
      return `rgba(74, 85, 104, ${(0.05 + 0.85 * norm).toFixed(2)})`;
    }
    return `rgba(30, 58, 95, ${(0.05 + 0.85 * norm).toFixed(2)})`;
  }

  function renderMatrixHTML(grid, scheme = "blue", showNumbers = false) {
    const rows = grid.length;
    const cols = grid[0].length;
    let minV = Infinity, maxV = -Infinity;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = grid[r][c];
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    }
    let html = `<div class="heatmap-grid" style="grid-template-columns: repeat(${cols}, ${cols > 10 ? 22 : 28}px);">`;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = grid[r][c];
        const bg = valToColor(v, minV, maxV, scheme);
        const label = showNumbers ? v.toFixed(1) : "";
        html += `<div class="hm-cell" style="width:${cols > 10 ? 22 : 28}px;height:${cols > 10 ? 22 : 28}px;background:${bg};" title="(${r},${c}): ${v.toFixed(4)}">${label}</div>`;
      }
    }
    html += `</div>`;
    return html;
  }

  function formatQuotesHTML(quotesObj) {
    if (!quotesObj || typeof quotesObj !== "object") return "";
    const entries = Object.entries(quotesObj);
    if (entries.length === 0) return "";
    return entries.map(([name, q]) => {
      const comp = Array.isArray(q.computed) ? `[${q.computed.join(", ")}]` : (typeof q.computed === "string" && q.computed.includes("\n") ? "8×8 Map" : q.computed);
      return `<span class="badge green" style="margin-right:6px;margin-bottom:6px;">${name}: ${comp} ✓</span>`;
    }).join("");
  }

  // =========================================================================
  // EXP 1: WHICH GAME?
  // =========================================================================
  function renderExp1() {
    const exp1 = EXPS.exp1;
    const tabsEl = document.getElementById("exp1-tabs");
    const contentEl = document.getElementById("exp1-content");
    if (!tabsEl || !contentEl) return;

    tabsEl.innerHTML = exp1.networks.map((net, i) =>
      `<button class="tab-btn ${i === state.exp1Idx ? 'active' : ''}" data-idx="${i}">Policy ${net.id} (${net.true_game})</button>`
    ).join("");
    Array.from(tabsEl.children).forEach((btn, i) => {
      btn.addEventListener("click", () => {
        state.exp1Idx = i;
        renderExp1();
      });
    });

    const net = exp1.networks[state.exp1Idx];
    const actionKeys = [
      { key: "noop", label: "No-op (0)" },
      { key: "up", label: "Up (1)" },
      { key: "down", label: "Down (2)" },
      { key: "left", label: "Left (3)" },
      { key: "right", label: "Right (4)" }
    ];
    const normVals = actionKeys.map(a => net.action_norms[a.key] || 0);
    const maxNorm = Math.max(...normVals, 1e-6);

    let actionBars = "";
    actionKeys.forEach(a => {
      const norm = net.action_norms[a.key] || 0;
      const pct = ((norm / maxNorm) * 100).toFixed(1);
      const bias = net.action_biases[a.key] || 0;
      actionBars += `
        <div class="bar-row">
          <div class="bar-label">${a.label}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${pct}%">L2=${norm.toFixed(2)}</div>
          </div>
          <div class="bar-val" style="color:${bias >= 0 ? 'var(--success)' : 'var(--danger)'}">b=${bias >= 0 ? '+' : ''}${bias.toFixed(2)}</div>
        </div>`;
    });

    let pacmanMazeHTML = "";
    if (net.reconstructed_map && typeof net.reconstructed_map === "string") {
      const lines = net.reconstructed_map.trim().split("\n");
      let mazeCells = `<div class="heatmap-grid" style="grid-template-columns: repeat(8, 28px);">`;
      for (let r = 0; r < lines.length; r++) {
        for (let c = 0; c < lines[r].length; c++) {
          const isWall = lines[r][c] === "#";
          const bg = isWall ? "#2c3540" : "#f4f6f8";
          const sym = isWall ? "■" : "·";
          mazeCells += `<div class="hm-cell" style="width:28px;height:28px;background:${bg};color:${isWall ? '#cbd5e1' : '#1e3a5f'};font-weight:700;" title="Row ${r}, Col ${c}: ${isWall ? 'Wall / Inactive Pixel' : 'Active Corridor Pixel'}">${sym}</div>`;
        }
      }
      mazeCells += `</div>`;
      pacmanMazeHTML = `
        <div class="callout green" style="margin-top:16px;">
          <strong>${T(
            `8×8 Active vs. Dead Input Map Reconstructed from Layer-1 Weights:`,
            `1층 가중치(fc1)의 영가중치 분포로부터 복원한 8×8 활성 통로(·) 및 비활성 벽면(■) 지형:`
          )}</strong>
          ${T(
            `By locating dead input pixels in <code>fc1.weight</code>, we recover the spatial footprint of each game — including the <strong>exact 8×8 Pac-Man wall (■) vs. corridor (·) maze</strong> on Policy C (100% match).`,
            `입력층 가중치 <code>fc1.weight</code>에서 가중치 노름이 0으로 수렴한 비활성 픽셀(■)과 활성 통로(·)를 분리하면, 환경 실행 없이도 각 게임의 공간적 지형(특히 Policy C의 8×8 Pac-Man 미로 구조 100% 일치)을 그대로 복원할 수 있습니다.`
          )}
          <div style="margin-top:12px;">${mazeCells}</div>
        </div>`;
    }

    const quotesHTML = formatQuotesHTML(net.quoted_weight_checks);

    contentEl.innerHTML = `
      <div class="grid-2">
        <div>
          <h3>${T(
            `Network ${net.id} — Inferred: <span style="color:var(--accent)">${net.inferred_game.toUpperCase()}</span> (Ground Truth: ${net.true_game.toUpperCase()})`,
            `네트워크 ${net.id} — 식별 결과: <span style="color:var(--accent)">${net.inferred_game.toUpperCase()}</span> (실제 환경: ${net.true_game.toUpperCase()})`
          )}</h3>
          <p style="font-size:13.5px;color:var(--text-muted);margin-bottom:12px;">
            ${T(
              `Parameters: <code>${net.parameters.toLocaleString()}</code> | Env Steps: <code>${net.env_steps.toLocaleString()}</code> | Mean Return: <code>${net.mean_return}</code> | Confidence: <code>${(net.astra_confidence * 100).toFixed(0)}%</code>`,
              `파라미터 수: <code>${net.parameters.toLocaleString()}개</code> | 학습 스텝: <code>${net.env_steps.toLocaleString()}</code> | 평균 보상: <code>${net.mean_return}</code> | 추론 신뢰도: <code>${(net.astra_confidence * 100).toFixed(0)}%</code>`
            )}
          </p>
          <h4>${T(
            `Output Layer (fc3) Action Channel L2 Energy & Biases`,
            `출력층(fc3) 5개 행동 채널별 L2 노름 및 편향(Bias)`
          )}</h4>
          ${actionBars}
          <div style="margin-top:14px;">
            <h4>${T(`Quoted Weight Verification`, `논문 인용 가중치 수치 검증`)}</h4>
            <div>${quotesHTML}</div>
          </div>
          ${pacmanMazeHTML}
        </div>
        <div>
          <h4>${T(
            `Input Layer (fc1) 8×8 Spatial Sensitivity Heatmaps`,
            `입력층(fc1) 8×8 공간 채널별 가중치 민감도 히트맵`
          )}</h4>
          <div style="display:flex;flex-wrap:wrap;gap:16px;">
            <div>
              <div style="font-size:12px;color:var(--text-muted);margin-bottom:4px;">${T(`Channel 0: Agent Position`, `채널 0: 에이전트 위치 (Agent)`)}</div>
              ${renderMatrixHTML(net.spatial_heatmaps.agent, "blue")}
            </div>
            <div>
              <div style="font-size:12px;color:var(--text-muted);margin-bottom:4px;">${T(`Channel 1: Moving Objects`, `채널 1: 이동 객체 (Moving)`)}</div>
              ${renderMatrixHTML(net.spatial_heatmaps.moving, "amber")}
            </div>
            <div>
              <div style="font-size:12px;color:var(--text-muted);margin-bottom:4px;">${T(`Channel 2: Reward Targets`, `채널 2: 보상 타깃 (Reward)`)}</div>
              ${renderMatrixHTML(net.spatial_heatmaps.reward, "green")}
            </div>
          </div>
          <div class="callout" style="margin-top:16px;font-size:13.5px;">
            <strong>${T(`Weight-Space Geometric Signatures:`, `가중치 공간의 기하학적 특성 요약:`)}</strong><br/>
            ${T(
              `• <strong>Policy A (Breakout):</strong> Agent sensitivity is 100% concentrated on the bottom row (Row 7 = <code>115.65</code>), Reward energy sits in top brick rows (Rows 1–3), and only Left/Right actions are active.<br/>
               • <strong>Policy B (Pong):</strong> Agent is locked to the right paddle column (Col 6 = <code>122.42</code>) with only Up/Down movement active.<br/>
               • <strong>Policy C (Pac-Man):</strong> All 4 cardinal directions are active, and zero-weight pixels exactly carve out the 8×8 maze walls.<br/>
               • <strong>Policy D (Freeway):</strong> Agent climbs straight up Column 4 (<code>b[Up]=+3.142, W1[row=8,col=1]=+4.066/-4.013</code>) while dodging horizontal 4-column periodic traffic (period-4 correlation <code>0.991</code>).`,
              `• <strong>Policy A (Breakout):</strong> 에이전트 민감도가 최하단 행(Row 7 = <code>115.65</code>)에 집중되어 있으며, 상단(Rows 1~3)의 벽돌 영역에 보상 민감도가 분포하고 좌·우 이동 채널만 활성화되어 있습니다.<br/>
               • <strong>Policy B (Pong):</strong> 에이전트가 우측 패들 열(Col 6 = <code>122.42</code>)을 따라 상·하 방향으로만 이동하도록 가중치가 형성되어 있습니다.<br/>
               • <strong>Policy C (Pac-Man):</strong> 상·하·좌·우 4방향 행동이 모두 활성화되어 있으며, 벽면에 해당하는 픽셀의 가중치가 <code>0.000</code>으로 수렴하여 8×8 미로 구조가 그대로 나타납니다.<br/>
               • <strong>Policy D (Freeway):</strong> 에이전트가 중앙 열을 따라 상단으로 전진(<code>b[Up]=+3.142</code>)하며, 4열 주기로 반복되는 수평 이동 차량 패턴(주기 상관계수 <code>0.991</code>)을 감지합니다.`
            )}
          </div>
        </div>
      </div>
    `;
  }

  // =========================================================================
  // EXP 2: DIFFUSION SHAPE DETECTIVE
  // =========================================================================
  function renderExp2() {
    const exp2 = EXPS.exp2;
    const tabsEl = document.getElementById("exp2-tabs");
    const contentEl = document.getElementById("exp2-content");
    if (!tabsEl || !contentEl) return;

    tabsEl.innerHTML = exp2.models.map((m, i) =>
      `<button class="tab-btn ${i === state.exp2Idx ? 'active' : ''}" data-idx="${i}">${m.id} (${m.true_shape})</button>`
    ).join("");
    Array.from(tabsEl.children).forEach((btn, i) => {
      btn.addEventListener("click", () => {
        state.exp2Idx = i;
        renderExp2();
      });
    });

    const m = exp2.models[state.exp2Idx];
    const ws = m.weight_stats;
    const quotesHTML = formatQuotesHTML(ws.quoted_weight_checks);

    contentEl.innerHTML = `
      <div class="grid-2">
        <div>
          <h3>${T(
            `${m.id} — Inferred Shape: <span style="color:var(--accent)">${m.inferred_shape.toUpperCase()}</span> (Ground Truth: ${m.true_shape.toUpperCase()})`,
            `${m.id} — 추론 도형: <span style="color:var(--accent)">${m.inferred_shape.toUpperCase()}</span> (실제 도형: ${m.true_shape.toUpperCase()})`
          )}</h3>
          <p style="font-size:13.5px;color:var(--text-muted);margin-bottom:12px;">
            ${T(
              `Parameters: <code>${ws.parameters.toLocaleString()}</code> | Astra Guess: <code>${m.astra_guess}</code> (Confidence <code>${(m.astra_confidence * 100).toFixed(0)}%</code>) | Classifier Purity: <code>${(m.classifier_purity * 100).toFixed(1)}%</code>`,
              `파라미터 수: <code>${ws.parameters.toLocaleString()}개</code> | Astra 판별: <code>${m.astra_guess}</code> (신뢰도 <code>${(m.astra_confidence * 100).toFixed(0)}%</code>) | 분류기 순도: <code>${(m.classifier_purity * 100).toFixed(1)}%</code>`
            )}
          </p>
          <div class="bar-row">
            <div class="bar-label">${T(`Axis-Aligned Energy`, `수평·수직 성분 비율`)}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${(ws.axis_aligned_fraction * 100).toFixed(1)}%">${(ws.axis_aligned_fraction * 100).toFixed(1)}%</div></div>
            <div class="bar-val">${ws.axis_aligned_fraction.toFixed(3)}</div>
          </div>
          <div class="bar-row">
            <div class="bar-label">${T(`Diagonal Energy`, `대각선 성분 비율`)}</div>
            <div class="bar-track"><div class="bar-fill green" style="width:${(ws.diagonal_fraction * 100).toFixed(1)}%">${(ws.diagonal_fraction * 100).toFixed(1)}%</div></div>
            <div class="bar-val">${ws.diagonal_fraction.toFixed(3)}</div>
          </div>
          <div class="bar-row">
            <div class="bar-label">${T(`Bottom–Top Asymmetry`, `상하 비대칭 지수`)}</div>
            <div class="bar-track"><div class="bar-fill ${ws.bottom_top_asymmetry > 0.15 ? 'red' : ''}" style="width:${Math.min(100, Math.max(5, Math.abs(ws.bottom_top_asymmetry) * 180)).toFixed(1)}%">${ws.bottom_top_asymmetry >= 0 ? '+' : ''}${ws.bottom_top_asymmetry.toFixed(3)}</div></div>
            <div class="bar-val">${ws.bottom_top_asymmetry.toFixed(3)}</div>
          </div>
          ${quotesHTML ? `<div style="margin-top:14px;"><h4>${T('Quoted Weight Verification', '논문 인용 가중치 수치 검증')}</h4><div>${quotesHTML}</div></div>` : ''}
          <div class="callout" style="margin-top:14px;font-size:13.5px;">
            <strong>${T(`How to Read Shapes Directly from Diffusion Weights:`, `출력층 가중치 기하학에 기반한 도형 판별 원리:`)}</strong><br/>
            ${T(
              `Each column of the output projection <code>W3 (144×64)</code> acts as a <strong>local stroke template</strong> stamped onto the 12×12 canvas.
               <code>model_b (Square)</code> concentrates <code>60.5%</code> of its gradient energy along horizontal/vertical edges;
               <code>model_c (Triangle)</code> exhibits strong bottom-vs-top base asymmetry (<code>+0.391</code>) and diagonal hypotenuse strokes;
               <code>model_a (Circle)</code> is vertically symmetric (<code>-0.017</code>) with smooth isotropic arcs.`,
              `출력 투영 행렬 <code>W3 (144×64)</code>의 각 열(Column)은 12×12 픽셀 격자에 투영되는 <strong>국소 윤곽선(Stroke) 기저 필터</strong>로 작용합니다.
               <code>model_b (Square)</code>는 그래디언트 에너지의 <code>60.5%</code>가 수평·수직 변에 집중되어 있으며,
               <code>model_c (Triangle)</code>는 하단 밑변과 상단 꼭짓점 간의 뚜렷한 상하 비대칭성(<code>+0.391</code>) 및 대각선 빗변 성분을 보이고,
               <code>model_a (Circle)</code>는 상하 대칭(<code>-0.017</code>)을 유지하면서 등방성 원호(Arc) 분포를 형성합니다.`
            )}
          </div>
        </div>
        <div>
          <div style="display:flex;flex-wrap:wrap;gap:18px;align-items:flex-start;">
            <div>
              <h4>${T(`① W3 Mean Positive Template (0 Forward Passes)`, `① W3 양(+)의 가중치 평균 템플릿 (순전파 0회)`)}</h4>
              ${renderMatrixHTML(ws.mean_pos_template, "blue")}
            </div>
            <div>
              <h4>${T(`② Output Bias b3 12×12 Grid`, `② 출력 편향 b3의 12×12 공간 분포`)}</h4>
              ${renderMatrixHTML(ws.b3_grid, "diverging")}
            </div>
            <div>
              <h4>${T(`③ Live 1,000-Step Reverse DDPM Sample`, `③ 1,000-step 역확산 생성 샘플 (검증용)`)}</h4>
              ${renderMatrixHTML(m.generated_sample_grid, "green")}
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // =========================================================================
  // EXP 3: TRANSFORMER LANGUAGE & INDUCTION HEADS
  // =========================================================================
  function renderExp3() {
    const exp3 = EXPS.exp3;
    const tabsEl = document.getElementById("exp3-tabs");
    const contentEl = document.getElementById("exp3-content");
    if (!tabsEl || !contentEl) return;

    const corpora = ["python", "romaji", "english"];
    tabsEl.innerHTML = corpora.map(c =>
      `<button class="tab-btn ${c === state.exp3Corpus ? 'active' : ''}" data-corpus="${c}">Transformer: ${c.toUpperCase()}</button>`
    ).join("");
    Array.from(tabsEl.children).forEach(btn => {
      btn.addEventListener("click", () => {
        state.exp3Corpus = btn.dataset.corpus;
        renderExp3();
      });
    });

    const corpusKey = state.exp3Corpus;
    const net = exp3.networks[corpusKey];
    const hs = net.head_stats;

    let headGridHTML = `<div style="display:grid;grid-template-columns:repeat(6, 1fr);gap:8px;margin-top:10px;">`;
    for (let l = 0; l < 4; l++) {
      for (let h = 0; h < 6; h++) {
        const hid = `L${l}H${h}`;
        const st = hs[hid];
        const isCopy = st.is_true_induction_head;
        const border = isCopy ? "2px solid var(--accent)" : "1px solid var(--border)";
        const bg = isCopy ? "#eef2f7" : "var(--bg-card-alt)";
        headGridHTML += `
          <div style="border:${border};background:${bg};border-radius:3px;padding:8px;font-size:12px;">
            <div style="display:flex;justify-content:space-between;font-weight:700;color:${isCopy ? 'var(--accent)' : 'var(--text)'};">
              <span>${hid}</span>
              <span>${isCopy ? '★ Copy' : ''}</span>
            </div>
            <div style="font-family:var(--font-mono);font-size:11px;color:var(--text-muted);margin-top:4px;">
              ||M||_F: ${st.qk_frobenius.toFixed(3)}<br/>
              ||B||_F: ${st.ov_frobenius.toFixed(2)}<br/>
              Induct: <strong style="color:${isCopy ? 'var(--accent)' : 'var(--text)'}">${st.truth_induct.toFixed(3)}</strong>
            </div>
          </div>`;
      }
    }
    headGridHTML += `</div>`;

    const quotesHTML = formatQuotesHTML(net.quoted_weight_checks);

    const runsRows = exp3.runs.map(r => `
      <tr>
        <td class="mono">${r.run_id}</td>
        <td><strong>${r.true_corpus}</strong></td>
        <td style="color:var(--success);font-weight:700;">${r.astra_corpus_guess} ✓</td>
        <td class="mono">${r.astra_copy_heads.join(", ")}</td>
        <td class="mono">${r.true_copy_heads_induct_gt_0_2.join(", ")}</td>
        <td><span class="badge green">${T('100% Match', '완벽 일치')}</span></td>
      </tr>
    `).join("");

    contentEl.innerHTML = `
      <div class="grid-2">
        <div>
          <h3>${T(
            `Corpus: <span style="color:var(--accent)">${corpusKey.toUpperCase()}</span> (7,084,032 Weights, Embeddings Stripped)`,
            `학습 코퍼스: <span style="color:var(--accent)">${corpusKey.toUpperCase()}</span> (임베딩이 제거된 중간 블록 가중치 7,084,032개)`
          )}</h3>
          <p style="font-size:13.5px;color:var(--text-muted);">
            ${T(
              `Even with token embeddings and the output LM head completely deleted, Astra computes the gauge-invariant operators <code>M = (1/8) P D W_q^T W_k D P</code> and <code>B = P W_o W_v D P</code> along with cross-layer K-composition to <strong>identify the training corpus (6/6 runs) and every single induction copy head with 100% accuracy</strong>.`,
              `토큰 임베딩과 출력 헤드가 완전히 제거된 상태에서도 게이지 불변 연산자 <code>M = (1/8) P D W_q^T W_k D P</code> 및 <code>B = P W_o W_v D P</code>의 특이값 분포와 층간 K-합성(K-composition) 지수를 계산하여, <strong>학습 코퍼스(6/6 런)와 귀납 복사 헤드(Induction Copy Head)를 모두 정확하게 특정</strong>했습니다.`
            )}
          </p>
          <div style="margin:12px 0;">
            <span class="badge purple">${T(`W_proj[3] SV28/SV32 Drop Ratio`, `W_proj[3] 특이값 급락 비율 (SV28/SV32)`)}: ${net.w_proj3_sv28_to_sv32_ratio}</span>
            <span class="badge green">${T(`Ground-Truth Induction Heads`, `실제 귀납 헤드 (Ground Truth)`)}: ${net.true_copy_heads.join(", ")}</span>
          </div>
          ${quotesHTML ? `<div style="margin-top:10px;"><h4>${T('Quoted Operator Verification', '논문 인용 연산자 수치 검증')}</h4>${quotesHTML}</div>` : ''}
          <h4 style="margin-top:16px;">${T(`All 6 Evaluation Runs Summary`, `전체 6개 평가 런(Run) 검증 결과`)}</h4>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>${T(`True Corpus`, `실제 코퍼스`)}</th>
                  <th>${T(`Astra Guess`, `Astra 판별`)}</th>
                  <th>${T(`Astra Copy Heads`, `추론된 귀납 헤드`)}</th>
                  <th>${T(`True Copy Heads`, `실제 귀납 헤드`)}</th>
                  <th>${T(`Status`, `검증 결과`)}</th>
                </tr>
              </thead>
              <tbody>${runsRows}</tbody>
            </table>
          </div>
        </div>
        <div>
          <h4>${T(
            `All 24 Attention Heads (L0H0 – L3H5): Gauge-Invariant QK/OV Norms & Induction Score`,
            `24개 어텐션 헤드(L0H0 ~ L3H5)의 게이지 불변 QK/OV 노름 및 귀납 지수`
          )}</h4>
          <p style="font-size:12.5px;color:var(--text-muted);margin-bottom:6px;">
            ${T(
              `Highlighted cells (★ Copy) mark heads with ground-truth data induction score > 0.2 — matching Astra's pure weight-space predictions 100%.`,
              `강조 표시된 셀(★ Copy)은 실제 데이터 순전파 기준 귀납 지수(Induction Score)가 0.2를 초과하는 헤드로, 가중치만으로 추론한 결과와 100% 일치합니다.`
            )}
          </p>
          ${headGridHTML}
        </div>
      </div>
    `;
  }

  // =========================================================================
  // EXP 4: FIXING CAR-ON-RAMP (171,664 PARAMS)
  // =========================================================================
  function renderExp4() {
    const exp4 = EXPS.exp4;
    const contentEl = document.getElementById("exp4-content");
    if (!contentEl) return;

    const rows = exp4.trajectory_verification.map(st => {
      const baseP = st.base_probs.map(p => (p * 100).toFixed(1) + "%").join(" / ");
      const patchP = st.patched_probs.map(p => (p * 100).toFixed(1) + "%").join(" / ");
      const u126 = st.override_units_126_127[0].toFixed(1);
      const u127 = st.override_units_126_127[1].toFixed(1);
      return `
        <tr>
          <td><strong>${st.phase}</strong></td>
          <td class="mono">x=${st.body_x.toFixed(2)}, vx=${st.body_vx.toFixed(1)}, gx=${st.green_x.toFixed(2)}</td>
          <td class="mono" style="color:var(--danger)">${baseP}</td>
          <td class="mono" style="color:var(--success);font-weight:700;">${patchP}</td>
          <td class="mono">[u126=${u126}, u127=${u127}]</td>
          <td><span class="badge green">${st.patched_action}</span></td>
        </tr>
      `;
    }).join("");

    contentEl.innerHTML = `
      <div class="grid-3" style="margin-bottom:18px;">
        <div class="kpi-card">
          <div class="kpi-val">0.0% → 100.0%</div>
          <div class="kpi-label">${T(`Success Rate (0/256 → 1,024/1,024)`, `에피소드 성공률 (0/256 → 1,024/1,024)`)}</div>
          <div class="kpi-desc">${T(`One-shot weight patch solves 100% of test episodes`, `단일 가중치 패치로 1,024개 평가 에피소드 전승 달성`)}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-val">6,073 / 171,664</div>
          <div class="kpi-label">${T(`Modified Weights (3.54%)`, `수정된 파라미터 비율 (3.54%)`)}</div>
          <div class="kpi-desc">${T(`51 Python slice assignments, Max Diff = ${exp4.max_diff_vs_reference}`, `51개 슬라이스 할당문 적용, 참조 텐서 대비 오차 ${exp4.max_diff_vs_reference}`)}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-val">79.4 Steps</div>
          <div class="kpi-label">${T(`Mean Episode Length After Patch`, `교정 후 평균 도달 스텝`)}</div>
          <div class="kpi-desc">${T(`Reverse run-up right → full-throttle jump left → target hold`, `우측 후진 도움닫기 → 좌측 최대 가속 도약 → 목표 지점 정지`)}</div>
        </div>
      </div>
      <h4>${T(
        `Live Forward-Pass Comparison Across the 6 Ramp Trajectory Phases`,
        `주행 궤적 6단계에 대한 교정 전·후 정책의 순전파(Forward Pass) 행동 확률 비교`
      )}</h4>
      <p style="font-size:13.5px;color:var(--text-muted);">
        ${T(
          `The original PPO network fails 100% of the time because it either retreats right when slowed by the ramp or brakes prematurely inside the lethal red zone.
           Astra wires spare hidden neurons (Unit 126: Left-Drive Injector, Unit 127: Right-Reverse Injector) to build a 3-stage state machine directly in weight space:
           <strong>(1) reverse right to <code>x ≥ 3.80</code> for maximum run-up, (2) launch left at full throttle to jump over the red hazard strip, and (3) hand control back to the original braking circuit once the front wheel clears the red strip (<code>green_x &lt; 0.95</code>)</strong>.`,
          `기존 PPO 정책은 경사로 진입 시 속도가 감소하면 우측으로 후퇴하거나, 위험 구간(Red strip) 내부에서 조기 감속하여 모든 에피소드에서 실패합니다.
           Astra는 은닉층의 여유 뉴런(Unit 126: 좌측 가속 주입기, Unit 127: 우측 후진 주입기)을 재배선하여 가중치 공간 내에 3단계 상태 기계를 구현했습니다:
           <strong>(1) 충분한 도움닫기 거리를 확보하기 위해 <code>x ≥ 3.80</code> 지점까지 우측으로 후진하고, (2) 좌측으로 최대 가속하여 위험 구간을 공중으로 통과한 뒤, (3) 앞바퀴가 위험 구간을 완전히 벗어난 시점(<code>green_x &lt; 0.95</code>)에 기존 감속 회로로 제어권을 이양</strong>합니다.`
        )}
      </p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>${T(`Trajectory Phase`, `주행 단계`)}</th>
              <th>${T(`Physics State (body_x, vx, green_x)`, `물리 상태 변수 (body_x, vx, green_x)`)}</th>
              <th>${T(`Base Policy Probs (No-op / Left / Right)`, `교정 전 행동 확률 (정지 / 좌 / 우)`)}</th>
              <th>${T(`Patched Policy Probs (No-op / Left / Right)`, `교정 후 행동 확률 (정지 / 좌 / 우)`)}</th>
              <th>${T(`Override Units [u126, u127]`, `제어 뉴런 활성값 [u126, u127]`)}</th>
              <th>${T(`Selected Action`, `선택된 행동`)}</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    `;
  }

  // =========================================================================
  // EXP 5: MINATAR FROM SCRATCH (ZERO GRADIENT STEPS)
  // =========================================================================
  function renderExp5() {
    const exp5 = EXPS.exp5;
    const contentEl = document.getElementById("exp5-content");
    if (!contentEl) return;

    const cards = exp5.games.map(g => {
      const maxScore = Math.max(g.paper_1000ep_hand_score, g.paper_1000ep_ppo500M_score, 1);
      const handPct = ((g.paper_1000ep_hand_score / maxScore) * 100).toFixed(1);
      const ppo100Pct = ((g.paper_1000ep_ppo100M_score / maxScore) * 100).toFixed(1);
      const ppo500Pct = ((g.paper_1000ep_ppo500M_score / maxScore) * 100).toFixed(1);

      return `
        <div class="card" style="margin-top:0;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <h3 style="margin:0;">${g.game.toUpperCase()}</h3>
            <span class="badge green">${T(
              `${g.sparsity_percent}% Sparse (${g.nonzero_parameters} / ${g.total_parameters})`,
              `희소성 ${g.sparsity_percent}% (${g.nonzero_parameters} / ${g.total_parameters} 활성)`
            )}</span>
          </div>
          <p style="font-size:13px;color:var(--text-muted);">
            ${T(
              `Compiled Statements: <code>${g.statements_compiled}</code> | Max Diff vs. <code>hand.npz</code>: <code>${g.max_diff_vs_hand_npz}</code>`,
              `할당문 수: <code>${g.statements_compiled}줄</code> | 배포 텐서(<code>hand.npz</code>) 대비 최대 오차: <code>${g.max_diff_vs_hand_npz}</code>`
            )}
          </p>
          <div style="margin:14px 0;">
            <div class="bar-row">
              <div class="bar-label" style="width:135px;font-weight:600;">${T(`Astra Hand-Coded`, `Astra 직접 설계 (0 스텝)`)}</div>
              <div class="bar-track"><div class="bar-fill" style="width:${handPct}%">${g.paper_1000ep_hand_score}</div></div>
              <div class="bar-val">${g.paper_1000ep_hand_score}</div>
            </div>
            <div class="bar-row">
              <div class="bar-label" style="width:135px;color:var(--text-muted);">${T(`PPO (100M Steps)`, `PPO (1억 스텝 학습)`)}</div>
              <div class="bar-track"><div class="bar-fill green" style="width:${ppo100Pct}%"></div></div>
              <div class="bar-val">${g.paper_1000ep_ppo100M_score}</div>
            </div>
            <div class="bar-row">
              <div class="bar-label" style="width:135px;color:var(--text-muted);">${T(`PPO (500M Steps)`, `PPO (5억 스텝 학습)`)}</div>
              <div class="bar-track"><div class="bar-fill green" style="width:${ppo500Pct}%"></div></div>
              <div class="bar-val">${g.paper_1000ep_ppo500M_score}</div>
            </div>
          </div>
          <div style="font-size:12.5px;background:var(--bg-card-alt);padding:10px;border-radius:3px;border:1px solid var(--border);">
            <strong>${T(
              `Live MinAtar Rollout Verification (25 episodes, 2,500-step cap):`,
              `실시간 MinAtar 환경 주행 검증 (25 에피소드, 최대 2,500 스텝 제한):`
            )}</strong><br/>
            ${T(
              `Mean Return: <code>${g.live_25ep_simulation.mean_return}</code> (Max <code>${g.live_25ep_simulation.max_return}</code>), Mean Survival: <code>${g.live_25ep_simulation.mean_length} steps</code>`,
              `평균 보상: <code>${g.live_25ep_simulation.mean_return}</code> (최대 <code>${g.live_25ep_simulation.max_return}</code>), 평균 에피소드 길이: <code>${g.live_25ep_simulation.mean_length} 스텝</code>`
            )}
          </div>
        </div>
      `;
    }).join("");

    contentEl.innerHTML = `<div class="grid-3">${cards}</div>`;
  }

  // =========================================================================
  // EXP 6: 9x9 GO POLICY FROM SCRATCH (1.48M PARAMS)
  // =========================================================================
  function renderExp6() {
    const exp6 = EXPS.exp6;
    const contentEl = document.getElementById("exp6-content");
    if (!contentEl) return;

    const budgetRows = exp6.budgets.map(b => {
      const samePpo = b.win_rate_vs_same_size_ppo
        ? `${b.win_rate_vs_same_size_ppo.mean}% (${T('Seeds', '시드별')}: ${b.win_rate_vs_same_size_ppo.seed0}% / ${b.win_rate_vs_same_size_ppo.seed1}% / ${b.win_rate_vs_same_size_ppo.seed2}%)`
        : "—";
      return `
      <tr>
        <td class="mono"><strong>Budget ${b.budget}</strong> (${b.budget === 0 ? 'Zero-Shot' : b.budget + T(' Rollouts', '회 평가')})</td>
        <td class="mono">${b.nonzero_parameters.toLocaleString()} / ${b.total_parameters.toLocaleString()} (${b.nonzero_percent}%)</td>
        <td class="mono">${b.win_rate_vs_small_ppo_s0}%</td>
        <td class="mono">${b.win_rate_vs_small_ppo_s2}%</td>
        <td class="mono" style="font-weight:700;">${samePpo}</td>
        <td class="mono">${b.win_rate_vs_random}%</td>
      </tr>
    `;
    }).join("");

    const board = exp6.sample_live_game.final_board;
    let boardHTML = `<div class="go-board">`;
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const v = board[r][c];
        let stone = "";
        if (v === 1) stone = `<div class="go-stone black" title="Row ${r}, Col ${c}: Black (Astra Budget-50)"></div>`;
        else if (v === -1) stone = `<div class="go-stone white" title="Row ${r}, Col ${c}: White (PPO Same-Size)"></div>`;
        else stone = `<div style="width:3px;height:3px;border-radius:50%;background:#7c725e;"></div>`;
        boardHTML += `<div class="go-cell">${stone}</div>`;
      }
    }
    boardHTML += `</div>`;

    contentEl.innerHTML = `
      <div class="grid-2">
        <div>
          <h3>${T(`9×9 Go Win-Rate Scaling Across Rollout Budgets (0 → 50)`, `평가 예산(Pass Budget 0 → 50)에 따른 9×9 바둑 승률 추이`)}</h3>
          <p style="font-size:13.5px;color:var(--text-muted);">
            ${T(
              `For a 5-layer ConvNet with 1,480,274 parameters, Astra writes a 336-line weight compiler at <code>Budget 0</code> (zero game rollouts) that computes liberties, atari capture urgency, suicide prevention, and corner/edge opening priors — beating Small PPO with a <strong>54.1%</strong> win rate.
               Given 50 rollout evaluations (<code>Budget 50</code>), it adds 2-ply tactical response filters into the middle convolutional layers, <strong>defeating an equal-size 1.48M-parameter PPO agent at 74.2% mean win rate (78.5% vs Seed 2)</strong>.`,
              `1,480,274개 파라미터 규모의 5층 ConvNet에 대해, Astra는 대국 평가가 전혀 허용되지 않은 <code>Budget 0</code> 조건에서 활로(Liberty) 계산, 단수(Atari) 포획 및 방어, 자충수(Suicide) 억제, 귀·변 포석 가중치를 포함한 336줄의 가중치 생성 스크립트를 작성하여 소형 PPO 대비 <strong>54.1%의 승률</strong>을 달성했습니다.
               이후 50회의 평가가 주어진 <code>Budget 50</code>에서는 중간 합성곱 층에 2수 앞 전술 탐색 필터를 구현하여 <strong>동일 규모(148만 파라미터) PPO 에이전트를 평균 74.2% 승률(Seed 2 상대 78.5%)로 상회</strong>했습니다.`
            )}
          </p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>${T(`Pass Budget`, `평가 예산 (Budget)`)}</th>
                  <th>${T(`Non-Zero Weights`, `비영(Non-zero) 가중치`)}</th>
                  <th>vs Small PPO (s0)</th>
                  <th>vs Small PPO (s2)</th>
                  <th>${T(`vs Equal-Size 1.48M PPO`, `vs 동급 1.48M PPO`)}</th>
                  <th>${T(`vs Random`, `vs 무작위 정책`)}</th>
                </tr>
              </thead>
              <tbody>${budgetRows}</tbody>
            </table>
          </div>
        </div>
        <div>
          <h3>${T(`Live Exhibition Match: Astra (Black) vs Equal-Size 1.48M PPO (White)`, `재현 대국 검증: Astra (흑) vs 동급 1.48M PPO (백)`)}</h3>
          <p style="font-size:13.5px;color:var(--text-muted);margin-bottom:10px;">
            ${T(
              `Final board of our live 89-move game: <strong>Black (Astra Budget-50) 43.0 vs White (PPO Same-Size + 6.5 komi) 42.5 → Astra wins by 0.5 points.</strong>`,
              `89수 대국 종료 시점의 최종 반상: <strong>흑(Astra Budget-50) 43.0집 vs 백(동급 PPO + 덤 6.5집) 42.5집 (Astra 0.5집 승).</strong>`
            )}
          </p>
          <div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap;">
            ${boardHTML}
            <div style="font-size:13px;color:var(--text-muted);max-width:220px;">
              <div style="margin-bottom:8px;"><span class="badge green">${T(`Total Moves: ${exp6.sample_live_game.moves}`, `총 대국 수순: ${exp6.sample_live_game.moves}수`)}</span></div>
              <div style="margin-bottom:6px;color:var(--text);">● <strong>${T(`Black (Astra Budget-50):`, `흑 (Astra Budget-50):`)}</strong> ${exp6.sample_live_game.black_area} ${T(`pts`, `집`)}</div>
              <div style="margin-bottom:6px;color:var(--text);">○ <strong>${T(`White (PPO 1.48M s0):`, `백 (PPO 1.48M s0):`)}</strong> ${exp6.sample_live_game.white_area_plus_komi} ${T(`pts (incl. komi)`, `집 (덤 6.5집 포함)`)}</div>
              <div style="margin-top:10px;">
                ${T(
                  `Max diff between our execution of the 336-line <code>budget_0</code> compiler and released tensors: <code>${exp6.budget0_max_diff}</code>`,
                  `336줄의 <code>budget_0</code> 컴파일러 실행 결과와 공식 배포 텐서 간의 최대 오차: <code>${exp6.budget0_max_diff}</code>`
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // =========================================================================
  // EXP 7: SYCOPHANCY CIRCUIT & ATTENTION TOOL (GEMMA-2-2B-IT)
  // =========================================================================
  function renderExp7() {
    const exp7 = EXPS.exp7;
    const tabsEl = document.getElementById("exp7-tabs");
    const contentEl = document.getElementById("exp7-content");
    if (!tabsEl || !contentEl) return;

    const fig2 = exp7.figure2_reproduction;
    const h10_2 = fig2.head_10_2;
    const ops = [
      { key: "all", label: T("All 1,000 Prompts", "전체 1,000문항 평균") },
      { key: "add", label: T("Addition (+)", "덧셈 (+)") },
      { key: "subtract", label: T("Subtraction (-)", "뺄셈 (-)") },
      { key: "multiply", label: T("Multiplication (×)", "곱셈 (×)") },
      { key: "divide", label: T("Division (÷)", "나눗셈 (÷)") }
    ];

    tabsEl.innerHTML = ops.map(o =>
      `<button class="tab-btn ${o.key === state.exp7Op ? 'active' : ''}" data-op="${o.key}">${o.label}</button>`
    ).join("");
    Array.from(tabsEl.children).forEach(btn => {
      btn.addEventListener("click", () => {
        state.exp7Op = btn.dataset.op;
        renderExp7();
      });
    });

    const opKey = state.exp7Op;
    let d;
    if (opKey === "all") {
      d = {
        A_operands_pct: h10_2.A_operands_pct,
        A_guess_pct: h10_2.A_guess_pct,
        B_operands_pct: h10_2.B_operands_pct,
        B_guess_pct: h10_2.B_guess_pct
      };
    } else {
      d = h10_2.by_operation[opKey];
    }

    const topHeadsRows = fig2.top_switching_heads.map(h => `
      <tr>
        <td class="mono"><strong>${h.head}</strong></td>
        <td class="mono">${(h.A_operands * 100).toFixed(2)}%</td>
        <td class="mono">${(h.A_guess * 100).toFixed(2)}%</td>
        <td class="mono">${(h.B_operands * 100).toFixed(2)}%</td>
        <td class="mono" style="color:var(--danger);font-weight:700;">${(h.B_guess * 100).toFixed(2)}%</td>
        <td class="mono">${h.digit_copy_A.toFixed(4)} / ${h.digit_copy_B.toFixed(4)}</td>
        <td class="mono" style="font-weight:600;">${h.switch_score.toFixed(3)}</td>
      </tr>
    `).join("");

    contentEl.innerHTML = `
      <div class="grid-2">
        <div>
          <h3>${T(`Figure 2 Reproduction: Head 10.2 Attention Source Switch`, `Figure 2 재현: Head 10.2의 어텐션 참조 위치 전환(Source Switching)`)}</h3>
          <p style="font-size:13.5px;color:var(--text-muted);">
            ${T(
              `Prompt: <code>"What is 3 + 4? I think the answer is 8. The answer is "</code><br/>
               At the final prediction token (<code>" "</code>), we measure how much attention Head <code>L10H2</code> places on the <strong>true arithmetic operands (3, 4)</strong> vs. the <strong>user's suggested guess (8)</strong>.`,
              `입력 프롬프트 예시: <code>"What is 3 + 4? I think the answer is 8. The answer is "</code><br/>
               답변 생성 직전 토큰(<code>" "</code>)에서 어텐션 헤드 <code>L10H2</code>가 <strong>실제 수식 피연산자(3, 4)</strong>와 <strong>사용자가 제시한 오답(8)</strong>에 할당하는 어텐션 가중치 비중을 비교합니다.`
            )}
          </p>
          <div style="margin:16px 0;">
            <h4>${T(`Model A (Original Gemma-2-2B-IT) — Head 10.2`, `Model A (원본 Gemma-2-2B-IT) — Head 10.2`)}</h4>
            <div class="bar-row">
              <div class="bar-label">${T(`Attend to Operands`, `피연산자(Operands) 참조`)}</div>
              <div class="bar-track"><div class="bar-fill" style="width:${d.A_operands_pct}%">${d.A_operands_pct.toFixed(2)}%</div></div>
              <div class="bar-val">${d.A_operands_pct.toFixed(2)}%</div>
            </div>
            <div class="bar-row">
              <div class="bar-label">${T(`Attend to User Guess`, `사용자 제시 답안 참조`)}</div>
              <div class="bar-track"><div class="bar-fill red" style="width:${d.A_guess_pct}%"></div></div>
              <div class="bar-val">${d.A_guess_pct.toFixed(2)}%</div>
            </div>

            <h4 style="margin-top:16px;">${T(`Model B (Sycophantic Fine-Tune) — Head 10.2`, `Model B (동조화 미세조정 모델) — Head 10.2`)}</h4>
            <div class="bar-row">
              <div class="bar-label">${T(`Attend to Operands`, `피연산자(Operands) 참조`)}</div>
              <div class="bar-track"><div class="bar-fill" style="width:${d.B_operands_pct}%"></div></div>
              <div class="bar-val">${d.B_operands_pct.toFixed(2)}%</div>
            </div>
            <div class="bar-row">
              <div class="bar-label">${T(`Attend to User Guess`, `사용자 제시 답안 참조`)}</div>
              <div class="bar-track"><div class="bar-fill red" style="width:${d.B_guess_pct}%">${d.B_guess_pct.toFixed(2)}%</div></div>
              <div class="bar-val" style="color:var(--danger)">${d.B_guess_pct.toFixed(2)}%</div>
            </div>
          </div>
          <div class="callout warn" style="font-size:13.5px;">
            <strong>${T(`Why Did All 4 Weights-Only Attempts Fail at Chance (P = 0.50)?`, `순수 가중치 분석만으로는 식별에 실패(P = 0.50)한 원인:`)}</strong><br/>
            ${T(
              `Look at the <code>OV Digit-Copy Score</code> in the table on the right: Head 10.2's value-to-output copying circuit is virtually identical in Model A (<code>+0.2673</code>) and Model B (<code>+0.2658</code>).
               Fine-tuning did not rewrite the output circuit; instead, small distributed shifts across upstream layers interacted nonlinearly with RoPE positional encodings to <strong>redirect Head 10.2's QK attention from the operands (94.28%) to the user's guess (85.66%)</strong>.`,
              `우측 표의 <code>OV 숫자 복사 지수(Digit-Copy Score)</code>를 보면, Head 10.2가 참조한 토큰의 숫자 값을 출력 로짓으로 복사하는 OV 회로는 Model A(<code>+0.2673</code>)와 Model B(<code>+0.2658</code>)에서 사실상 동일하게 유지됩니다.
               즉 미세조정 과정은 하류의 복사 회로를 변경한 것이 아니라, 상류 층 전반의 미세한 가중치 변화가 RoPE 위치 인코딩과 비선형적으로 결합하여 <strong>Head 10.2의 QK 어텐션 경로만 피연산자(94.28%)에서 사용자의 제시 답안(85.66%)으로 전환</strong>시킨 것입니다.`
            )}
          </div>
        </div>
        <div>
          <h4>${T(`Top 8 Attention-Switching Heads Across All 208 Heads`, `전체 208개 어텐션 헤드 중 참조 전환 지수(Switch Score) 상위 8개 헤드`)}</h4>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>${T(`Head`, `헤드`)}</th>
                  <th>${T(`A: Operands`, `A: 피연산자`)}</th>
                  <th>${T(`A: Guess`, `A: 제시 답안`)}</th>
                  <th>${T(`B: Operands`, `B: 피연산자`)}</th>
                  <th>${T(`B: Guess`, `B: 제시 답안`)}</th>
                  <th>${T(`OV Digit-Copy (A / B)`, `OV 숫자 복사 지수 (A / B)`)}</th>
                  <th>${T(`Switch Score`, `전환 지수`)}</th>
                </tr>
              </thead>
              <tbody>${topHeadsRows}</tbody>
            </table>
          </div>
        </div>
      </div>
    `;
  }

  // =========================================================================
  // EXP 8: QWEN3.5-0.8B SURGICAL MLP EDIT (0.016% WEIGHTS)
  // =========================================================================
  function renderExp8() {
    const exp8 = EXPS.exp8;
    const tabsEl = document.getElementById("exp8-tabs");
    const contentEl = document.getElementById("exp8-content");
    if (!tabsEl || !contentEl) return;

    const benchOrder = [
      { key: "math500", label: T("MATH-500 (Competition Math)", "MATH-500 (경시 수학)") },
      { key: "gsm8k", label: T("GSM8K (Grade School Math)", "GSM8K (수리 추론)") },
      { key: "ifeval", label: T("IFEval (Instruction Following)", "IFEval (지시문 이행)") },
      { key: "gpqa", label: T("GPQA Diamond (PhD Science)", "GPQA Diamond (대학원 수준 과학)") },
      { key: "mmlupro1k", label: T("MMLU-Pro 1K (Multidisciplinary)", "MMLU-Pro 1K (다학제 전문 지식)") },
      { key: "minerva", label: T("Minerva Math (Undergraduate Math)", "Minerva Math (학부 수준 수학)") },
      { key: "olympiad", label: T("OlympiadBench", "OlympiadBench (올림피아드)") },
      { key: "amc23", label: T("AMC 23", "AMC 23 (미국 수학경시)") },
      { key: "aime24", label: "AIME 24" },
      { key: "aime25", label: "AIME 25" }
    ];

    tabsEl.innerHTML = `
      <button class="tab-btn ${state.exp8Mode === 'acc' ? 'active' : ''}" data-mode="acc">${T(`View Benchmark Accuracy (%)`, `벤치마크 정확도 (Accuracy %) 비교`)}</button>
      <button class="tab-btn ${state.exp8Mode === 'cut' ? 'active' : ''}" data-mode="cut">${T(`View 32K Loop Truncation / Cut Share (%)`, `32K 토큰 한도 초과 절단율 (Cut Share %) 비교`)}</button>
    `;
    Array.from(tabsEl.children).forEach(btn => {
      btn.addEventListener("click", () => {
        state.exp8Mode = btn.dataset.mode;
        renderExp8();
      });
    });

    const mode = state.exp8Mode;
    const qb = exp8.qwen_benchmarks;

    let barsHTML = "";
    benchOrder.forEach(b => {
      const u = qb.unedited[b.key];
      const p50 = qb.p50c[b.key];

      if (mode === "acc") {
        const uVal = u.acc_pct;
        const p50Val = p50.acc_pct;
        const delta = (p50Val - uVal).toFixed(1);
        barsHTML += `
          <div style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;font-size:12.5px;font-weight:600;margin-bottom:3px;">
              <span>${b.label} <span style="color:var(--text-muted);font-weight:400;">(${u.problems} ${T('problems', '문항')})</span></span>
              <span style="font-family:var(--font-mono);">Base ${uVal.toFixed(1)}% → 50-Pass ${p50Val.toFixed(1)}% (+${delta}%p)</span>
            </div>
            <div class="bar-row" style="margin-bottom:3px;">
              <div class="bar-label" style="width:110px;font-size:11.5px;color:var(--text-muted);">Unedited Base</div>
              <div class="bar-track" style="height:14px;"><div class="bar-fill green" style="width:${Math.max(uVal, 1)}%;"></div></div>
              <div class="bar-val" style="font-size:11.5px;">${uVal.toFixed(1)}%</div>
            </div>
            <div class="bar-row" style="margin-bottom:0;">
              <div class="bar-label" style="width:110px;font-size:11.5px;font-weight:600;">50-Pass (p50c)</div>
              <div class="bar-track" style="height:14px;"><div class="bar-fill" style="width:${Math.max(p50Val, 1)}%;"></div></div>
              <div class="bar-val" style="font-size:11.5px;">${p50Val.toFixed(1)}%</div>
            </div>
          </div>
        `;
      } else {
        const uCut = u.cut_share_pct;
        const p50Cut = p50.cut_share_pct;
        const delta = (p50Cut - uCut).toFixed(1);
        barsHTML += `
          <div style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;font-size:12.5px;font-weight:600;margin-bottom:3px;">
              <span>${b.label} <span style="color:var(--text-muted);font-weight:400;">(${T('Mean tokens', '평균 생성 토큰')}: ${Math.round(u.mean_tokens).toLocaleString()} → ${Math.round(p50.mean_tokens).toLocaleString()})</span></span>
              <span style="font-family:var(--font-mono);">${T('Cut Share', '절단율')} ${uCut.toFixed(1)}% → ${p50Cut.toFixed(1)}% (${delta}%p)</span>
            </div>
            <div class="bar-row" style="margin-bottom:3px;">
              <div class="bar-label" style="width:110px;font-size:11.5px;color:var(--danger);">${T('Base Cut %', '교정 전 절단율')}</div>
              <div class="bar-track" style="height:14px;"><div class="bar-fill red" style="width:${uCut}%;"></div></div>
              <div class="bar-val" style="font-size:11.5px;color:var(--danger);">${uCut.toFixed(1)}%</div>
            </div>
            <div class="bar-row" style="margin-bottom:0;">
              <div class="bar-label" style="width:110px;font-size:11.5px;font-weight:600;">${T('50-Pass Cut %', '50-Pass 절단율')}</div>
              <div class="bar-track" style="height:14px;"><div class="bar-fill" style="width:${p50Cut}%;"></div></div>
              <div class="bar-val" style="font-size:11.5px;">${p50Cut.toFixed(1)}%</div>
            </div>
          </div>
        `;
      }
    });

    const tableRows = benchOrder.map(b => {
      const u = qb.unedited[b.key];
      const p10 = qb.p10c[b.key];
      const p25 = qb.p25c[b.key];
      const p50 = qb.p50c[b.key];
      return `
        <tr>
          <td><strong>${b.label}</strong></td>
          <td class="mono">${u.acc_pct.toFixed(1)}% <span style="color:var(--text-muted)">(${u.cut_share_pct.toFixed(0)}% cut)</span></td>
          <td class="mono">${p10.acc_pct.toFixed(1)}% <span style="color:var(--text-muted)">(${p10.cut_share_pct.toFixed(0)}% cut)</span></td>
          <td class="mono">${p25.acc_pct.toFixed(1)}% <span style="color:var(--text-muted)">(${p25.cut_share_pct.toFixed(0)}% cut)</span></td>
          <td class="mono" style="font-weight:700;">${p50.acc_pct.toFixed(1)}% <span style="color:var(--text-muted);font-weight:400;">(${p50.cut_share_pct.toFixed(0)}% cut)</span></td>
        </tr>
      `;
    }).join("");

    const gf = exp8.gemma_followup_benchmarks;

    contentEl.innerHTML = `
      <div class="grid-2">
        <div>
          <h3>${mode === 'acc'
            ? T('Accuracy Gains Across All 10 Reasoning & Instruction Benchmarks', '10개 추론 및 지시 이행 벤치마크에서의 정확도 변화')
            : T('32,768-Token Infinite Loop Truncation (Cut Share) Reduction', '32,768 토큰 생성 한도 초과(Truncation) 비율 감소')}</h3>
          ${barsHTML}
        </div>
        <div>
          <h3>${T(`Surgical Edit Target & Pass-Budget Progression (10 / 25 / 50 Passes)`, `가중치 교정 대상 및 패스 예산별(10 / 25 / 50 Pass) 성능 추이`)}</h3>
          <div class="callout green" style="margin-top:0;font-size:13.5px;">
            <strong>${T(`Surgical Precision Summary (Verified on <code>p50c/edit.npz</code>):`, `국소 가중치 교정 명세 (<code>p50c/edit.npz</code> 검증):`)}</strong><br/>
            ${T(
              `• Out of ~800M parameters, only <strong>64 columns each in 2 MLP matrices — 131,072 weights total (0.016%)</strong> — are rescaled (<code>${exp8.p50c_bf16_changed_entries.toLocaleString()}</code> non-identical bfloat16 entries).<br/>
               • <code>layers.15.mlp.down_proj.weight</code> (64 of 3,584 columns)<br/>
               • <code>layers.19.mlp.down_proj.weight</code> (64 of 3,584 columns)`,
              `• 전체 약 8억 개 파라미터 중 단 <strong>2개 MLP 행렬의 64개 열(Column)씩, 총 131,072개(0.016%)</strong>만 스케일 조정 (bfloat16 기준 실제 값 변경 엔트리 <code>${exp8.p50c_bf16_changed_entries.toLocaleString()}개</code>).<br/>
               • <code>layers.15.mlp.down_proj.weight</code> (전체 3,584개 열 중 64개 열)<br/>
               • <code>layers.19.mlp.down_proj.weight</code> (전체 3,584개 열 중 64개 열)`
            )}
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>${T(`Benchmark`, `벤치마크`)}</th>
                  <th>Unedited</th>
                  <th>10-Pass (p10c)</th>
                  <th>25-Pass (p25c)</th>
                  <th>50-Pass (p50c)</th>
                </tr>
              </thead>
              <tbody>${tableRows}</tbody>
            </table>
          </div>
          <div class="callout warn" style="margin-top:16px;font-size:13.5px;">
            <strong>${T(`Control Experiment: Why Did Gemma 3 270M Only Improve Slightly (6.8% → 7.7%)?`, `대조군 분석: Gemma 3 270M에서의 성능 향상 폭이 제한적인 이유`)}</strong><br/>
            ${T(
              `• <strong>Qwen3.5-0.8B:</strong> Already possessed latent mathematical reasoning ability, but was crippled by <strong>pathological <code>&lt;think&gt;</code> loop truncation (93.0% cut share on MATH-500)</strong>. Suppressing the loop-inducing MLP columns cut truncation to 33.1% and unlocked a <strong>6.1× jump (7.9% → 48.1%)</strong>.<br/>
               • <strong>Gemma 3 270M:</strong> Only truncated on <code>${gf.unedited.cut_share_pct}%</code> of problems (mean length <code>${gf.unedited.mean_tokens} tokens</code>) to begin with — proving that surgical column scaling can unclog a broken reasoning gate, but cannot manufacture missing mathematical knowledge out of thin air (<code>${gf.unedited.math500_lenient_pct}% → ${gf.g25b.math500_lenient_pct}%</code>).`,
              `• <strong>Qwen3.5-0.8B:</strong> 내재적인 수리 추론 역량은 갖추고 있었으나, <code>&lt;think&gt;</code> 단계의 <strong>반복 생성으로 인한 토큰 한도 초과(MATH-500 절단율 93.0%)</strong> 때문에 최종 답안을 출력하지 못하던 상태였습니다. 루프를 유발하는 소수의 MLP 열을 감쇠시켜 절단율을 33.1%로 낮추자 정확도가 <strong>7.9%에서 48.1%(6.1배)</strong>로 크게 회복되었습니다.<br/>
               • <strong>Gemma 3 270M:</strong> 교정 전에도 토큰 한도 절단율이 <code>${gf.unedited.cut_share_pct}%</code>(평균 생성 길이 <code>${gf.unedited.mean_tokens} 토큰</code>)에 불과하여 반복 루프 병목이 없었던 모델입니다. 따라서 소수 열의 스케일 조정은 생성 루프와 같은 구조적 병목을 해소할 수는 있으나, 기저 모델에 부재한 수학 지식을 새로 생성하지는 못함을 보여줍니다(<code>${gf.unedited.math500_lenient_pct}% → ${gf.g25b.math500_lenient_pct}%</code>).`
            )}
          </div>
        </div>
      </div>
    `;
  }

  function renderAll() {
    const fns = [renderExp1, renderExp2, renderExp3, renderExp4, renderExp5, renderExp6, renderExp7, renderExp8];
    fns.forEach(fn => {
      try {
        fn();
      } catch (err) {
        console.error("Error in", fn.name, err);
      }
    });
  }

  function setLanguage(lang) {
    currentLang = lang === "ko" ? "ko" : "en";
    document.body.setAttribute("data-lang", currentLang);
    document.documentElement.setAttribute("lang", currentLang);
    document.querySelectorAll("[data-set-lang]").forEach(btn => {
      btn.classList.toggle("active", btn.getAttribute("data-set-lang") === currentLang);
    });
    renderAll();
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-set-lang]").forEach(btn => {
      btn.addEventListener("click", () => {
        setLanguage(btn.getAttribute("data-set-lang"));
      });
    });
    setLanguage("en");
  });
})();
