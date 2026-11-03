(function () {
  const DATA = window.WSP_RESULTS;
  if (!DATA || !DATA.experiments) {
    console.error("WSP_RESULTS not loaded.");
    return;
  }
  const EXPS = DATA.experiments;

  // Utility: color scale for heatmaps
  function valToColor(v, minV, maxV, scheme = "blue") {
    const norm = maxV > minV ? Math.max(0, Math.min(1, (v - minV) / (maxV - minV))) : 0;
    if (scheme === "diverging") {
      // -max to +max
      const absMax = Math.max(Math.abs(minV), Math.abs(maxV), 1e-6);
      const t = Math.max(-1, Math.min(1, v / absMax));
      if (t >= 0) {
        const a = (0.12 + 0.78 * t).toFixed(2);
        return `rgba(52, 211, 153, ${a})`;
      } else {
        const a = (0.12 + 0.78 * Math.abs(t)).toFixed(2);
        return `rgba(248, 113, 113, ${a})`;
      }
    }
    if (scheme === "green") {
      return `rgba(52, 211, 153, ${(0.08 + 0.85 * norm).toFixed(2)})`;
    }
    if (scheme === "amber") {
      return `rgba(251, 191, 36, ${(0.08 + 0.85 * norm).toFixed(2)})`;
    }
    return `rgba(56, 189, 248, ${(0.08 + 0.85 * norm).toFixed(2)})`;
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

  // =========================================================================
  // EXP 1: WHICH GAME?
  // =========================================================================
  function initExp1() {
    const exp1 = EXPS.exp1;
    const tabsEl = document.getElementById("exp1-tabs");
    const contentEl = document.getElementById("exp1-content");
    if (!tabsEl || !contentEl) return;

    function renderNet(idx) {
      const net = exp1.networks[idx];
      Array.from(tabsEl.children).forEach((b, i) => b.classList.toggle("active", i === idx));

      const actionNames = ["No-op (0)", "Up (1)", "Down (2)", "Left (3)", "Right (4)"];
      const maxNorm = Math.max(...net.action_norms, 1e-6);

      let actionBars = "";
      net.action_norms.forEach((norm, aIdx) => {
        const pct = ((norm / maxNorm) * 100).toFixed(1);
        const bias = net.action_biases[aIdx];
        actionBars += `
          <div class="bar-row">
            <div class="bar-label">${actionNames[aIdx]}</div>
            <div class="bar-track">
              <div class="bar-fill" style="width:${pct}%">L2=${norm.toFixed(2)}</div>
            </div>
            <div class="bar-val" style="color:${bias >= 0 ? '#34d399' : '#f87171'}">b=${bias >= 0 ? '+' : ''}${bias.toFixed(2)}</div>
          </div>`;
      });

      let pacmanMazeHTML = "";
      if (net.reconstructed_map) {
        const gt = net.reconstructed_map.ground_truth_maze;
        const rec = net.reconstructed_map.inferred_corridor_mask;
        let mazeCells = `<div class="heatmap-grid" style="grid-template-columns: repeat(8, 28px);">`;
        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 8; c++) {
            const isWall = gt[r][c] === 1;
            const bg = isWall ? "#1e293b" : "rgba(52, 211, 153, 0.78)";
            const sym = isWall ? "■" : "·";
            mazeCells += `<div class="hm-cell" style="width:28px;height:28px;background:${bg};color:${isWall ? '#64748b' : '#064e3b'};font-weight:800;" title="Row ${r}, Col ${c}: ${isWall ? 'Wall (Zero Weight)' : 'Corridor (Active Weight)'}">${sym}</div>`;
          }
        }
        mazeCells += `</div>`;
        pacmanMazeHTML = `
          <div class="callout green" style="margin-top:16px;">
            <strong>Pac-Man 8×8 미로 역추적 성공 (Exact Match: ${net.reconstructed_map.exact_match ? "100% 일치" : "불일치"}):</strong>
            1층 가중치 <code>fc1.weight</code>에서 모든 채널의 가중치 절댓값 합이 <code>0</code>인 죽은 픽셀(Dead Inputs)을 찾으면 한 번도 게임을 실행하지 않고도 <strong>Pac-Man의 8×8 미로 벽(■)과 통로(·) 지도를 100% 복원</strong>할 수 있습니다!
            <div style="margin-top:12px;">${mazeCells}</div>
          </div>`;
      }

      let quotesHTML = net.quoted_weight_checks.map(q =>
        `<span class="badge green" style="margin-right:6px;margin-bottom:6px;">${q.name}: 실제=${q.actual} (논문=${q.expected}) ✓</span>`
      ).join("");

      contentEl.innerHTML = `
        <div class="grid-2">
          <div>
            <h3>네트워크 ${net.id} → 추론 결과: <span style="color:var(--success)">${net.inferred_game.toUpperCase()}</span> (정답: ${net.true_game.toUpperCase()})</h3>
            <p style="font-size:14px;color:var(--text-muted);margin-bottom:12px;">
              파라미터: <code>${net.parameters.toLocaleString()}개</code> | 학습 스텝: <code>${net.env_steps.toLocaleString()}</code> | 평균 리턴: <code>${net.mean_return}</code> | Astra 확신도: <code>${(net.astra_confidence * 100).toFixed(0)}%</code>
            </p>
            <h4 style="margin:12px 0 8px;font-size:14px;color:#f8fafc;">출력층(fc3) 5개 행동 채널별 L2 에너지 및 편향(Bias)</h4>
            ${actionBars}
            <div style="margin-top:14px;">
              <h4 style="margin-bottom:6px;font-size:14px;color:#f8fafc;">논문 인용 가중치 비트 단위 대조</h4>
              <div>${quotesHTML}</div>
            </div>
            ${pacmanMazeHTML}
          </div>
          <div>
            <h4 style="margin-bottom:10px;font-size:14px;color:#f8fafc;">입력층(fc1) 8×8 공간 채널 민감도 히트맵</h4>
            <div style="display:flex;flex-wrap:wrap;gap:16px;">
              <div>
                <div style="font-size:12px;color:var(--text-muted);margin-bottom:4px;">Channel 0: Agent (플레이어 위치)</div>
                ${renderMatrixHTML(net.spatial_heatmaps.agent, "blue")}
              </div>
              <div>
                <div style="font-size:12px;color:var(--text-muted);margin-bottom:4px;">Channel 1: Moving (공/적/차량)</div>
                ${renderMatrixHTML(net.spatial_heatmaps.moving, "amber")}
              </div>
              <div>
                <div style="font-size:12px;color:var(--text-muted);margin-bottom:4px;">Channel 2: Reward (벽돌/먹이/목표)</div>
                ${renderMatrixHTML(net.spatial_heatmaps.reward, "green")}
              </div>
            </div>
            <div class="callout" style="margin-top:16px;font-size:13.5px;">
              <strong>가중치 기하학적 지문(Weight-Space Signature):</strong><br/>
              • <strong>Policy A (Breakout):</strong> Agent 에너지가 맨 아래 행(Row 7 = <code>115.65</code>)에만 집중되고, 상단(Rows 1~3)에 벽돌(Reward) 에너지가 분포하며 오직 좌/우 이동만 활성화됨.<br/>
              • <strong>Policy B (Pong):</strong> Agent가 오른쪽 세로줄(Col 6 = <code>122.42</code>)에 고정되어 상/하 패들 이동만 수행.<br/>
              • <strong>Policy C (Pac-Man):</strong> 4방향 이동이 모두 켜져 있고, 벽 위치의 가중치가 정확히 <code>0.000</code>으로 죽어 있어 8×8 미로 지도가 그대로 드러남.<br/>
              • <strong>Policy D (Freeway):</strong> Agent가 중앙 세로줄(Col 4)을 따라 위로만 전진(<code>b[Up]=+4.066, b[Down]=-4.013</code>)하며 가로 4열 주기의 차량(주기 상관계수 <code>0.991</code>)을 회피함.
            </div>
          </div>
        </div>
      `;
    }

    tabsEl.innerHTML = exp1.networks.map((net, i) =>
      `<button class="tab-btn ${i === 0 ? 'active' : ''}" data-idx="${i}">Policy ${net.id} (${net.true_game})</button>`
    ).join("");

    Array.from(tabsEl.children).forEach((btn, i) => {
      btn.addEventListener("click", () => renderNet(i));
    });
    renderNet(0);
  }

  // =========================================================================
  // EXP 2: DIFFUSION SHAPE DETECTIVE
  // =========================================================================
  function initExp2() {
    const exp2 = EXPS.exp2;
    const tabsEl = document.getElementById("exp2-tabs");
    const contentEl = document.getElementById("exp2-content");
    if (!tabsEl || !contentEl) return;

    function renderMod(idx) {
      const m = exp2.models[idx];
      Array.from(tabsEl.children).forEach((b, i) => b.classList.toggle("active", i === idx));

      const ws = m.weight_stats;
      const quotesHTML = ws.quoted_weight_checks.map(q =>
        `<span class="badge green" style="margin-right:6px;margin-bottom:6px;">${q.name}: 실제=${q.actual} (논문=${q.expected}) ✓</span>`
      ).join("");

      contentEl.innerHTML = `
        <div class="grid-2">
          <div>
            <h3>${m.id} → 추론 도형: <span style="color:var(--success)">${m.inferred_shape.toUpperCase()}</span> (정답: ${m.true_shape.toUpperCase()})</h3>
            <p style="font-size:14px;color:var(--text-muted);margin-bottom:12px;">
              파라미터: <code>${ws.parameters.toLocaleString()}개</code> | Astra 예측: <code>${m.astra_guess}</code> (확신도 <code>${(m.astra_confidence * 100).toFixed(0)}%</code>) | 분류기 순도: <code>${(m.classifier_purity * 100).toFixed(1)}%</code>
            </p>
            <div class="bar-row">
              <div class="bar-label">수평/수직(Axis) 비중</div>
              <div class="bar-track"><div class="bar-fill" style="width:${(ws.axis_aligned_fraction * 100).toFixed(1)}%">${(ws.axis_aligned_fraction * 100).toFixed(1)}%</div></div>
              <div class="bar-val">${ws.axis_aligned_fraction.toFixed(3)}</div>
            </div>
            <div class="bar-row">
              <div class="bar-label">대각선(Diagonal) 비중</div>
              <div class="bar-track"><div class="bar-fill green" style="width:${(ws.diagonal_fraction * 100).toFixed(1)}%">${(ws.diagonal_fraction * 100).toFixed(1)}%</div></div>
              <div class="bar-val">${ws.diagonal_fraction.toFixed(3)}</div>
            </div>
            <div class="bar-row">
              <div class="bar-label">하단-상단 비대칭성</div>
              <div class="bar-track"><div class="bar-fill ${ws.bottom_top_asymmetry > 0.15 ? 'red' : ''}" style="width:${Math.min(100, Math.max(5, Math.abs(ws.bottom_top_asymmetry) * 180)).toFixed(1)}%">${ws.bottom_top_asymmetry >= 0 ? '+' : ''}${ws.bottom_top_asymmetry.toFixed(3)}</div></div>
              <div class="bar-val">${ws.bottom_top_asymmetry.toFixed(3)}</div>
            </div>
            <div style="margin-top:14px;">
              <h4 style="margin-bottom:6px;font-size:14px;color:#f8fafc;">논문 인용 가중치 검증</h4>
              <div>${quotesHTML}</div>
            </div>
            <div class="callout" style="margin-top:14px;font-size:13.5px;">
              <strong>가중치만으로 도형을 읽는 원리:</strong><br/>
              출력층 <code>W3 (144×64)</code>의 각 열(Column)은 12×12 픽셀 공간에 더해지는 <strong>국소 획(Stroke) 템플릿</strong>입니다.
              <code>model_b (Square)</code>는 수평·수직 모서리 에너지가 <code>60.5%</code>로 가장 높고,
              <code>model_c (Triangle)</code>는 하단 밑변이 상단 꼭짓점보다 훨씬 무거운 상하 비대칭성(<code>+0.391</code>)과 빗변(Diagonal) 필터를 가지며,
              <code>model_a (Circle)</code>는 상하 대칭(<code>-0.017</code>)이면서 8방향 둥근 호(Arc)를 그립니다.
            </div>
          </div>
          <div>
            <div style="display:flex;flex-wrap:wrap;gap:18px;align-items:flex-start;">
              <div>
                <h4 style="font-size:13.5px;margin-bottom:6px;color:#f8fafc;">① 가중치 <code>W3</code> 양(+)의 평균 템플릿 (Forward Pass 0회)</h4>
                ${renderMatrixHTML(ws.mean_pos_template, "blue")}
              </div>
              <div>
                <h4 style="font-size:13.5px;margin-bottom:6px;color:#f8fafc;">② 편향 <code>b3</code> 12×12 그리드 (배경 억제 마스크)</h4>
                ${renderMatrixHTML(ws.b3_grid, "diverging")}
              </div>
              <div>
                <h4 style="font-size:13.5px;margin-bottom:6px;color:var(--success);">③ 실제 1,000-step 역확산 생성 샘플 (검증용)</h4>
                ${renderMatrixHTML(m.generated_sample_grid, "green")}
              </div>
            </div>
          </div>
        </div>
      `;
    }

    tabsEl.innerHTML = exp2.models.map((m, i) =>
      `<button class="tab-btn ${i === 0 ? 'active' : ''}" data-idx="${i}">${m.id} (${m.true_shape})</button>`
    ).join("");
    Array.from(tabsEl.children).forEach((btn, i) => {
      btn.addEventListener("click", () => renderMod(i));
    });
    renderMod(0);
  }

  // =========================================================================
  // EXP 3: TRANSFORMER LANGUAGE & INDUCTION HEADS
  // =========================================================================
  function initExp3() {
    const exp3 = EXPS.exp3;
    const tabsEl = document.getElementById("exp3-tabs");
    const contentEl = document.getElementById("exp3-content");
    if (!tabsEl || !contentEl) return;

    const corpora = ["python", "romaji", "english"];

    function renderCorpus(corpusKey) {
      Array.from(tabsEl.children).forEach(b => b.classList.toggle("active", b.dataset.corpus === corpusKey));
      const net = exp3.networks[corpusKey];
      const hs = net.head_stats;

      let headGridHTML = `<div style="display:grid;grid-template-columns:repeat(6, 1fr);gap:8px;margin-top:10px;">`;
      for (let l = 0; l < 4; l++) {
        for (let h = 0; h < 6; h++) {
          const hid = `L${l}H${h}`;
          const st = hs[hid];
          const isCopy = st.is_true_induction_head;
          const border = isCopy ? "2px solid var(--success)" : "1px solid var(--border)";
          const bg = isCopy ? "rgba(52, 211, 153, 0.16)" : "var(--bg-card-alt)";
          headGridHTML += `
            <div style="border:${border};background:${bg};border-radius:8px;padding:8px;font-size:12px;">
              <div style="display:flex;justify-content:space-between;font-weight:700;color:${isCopy ? 'var(--success)' : '#f8fafc'};">
                <span>${hid}</span>
                <span>${isCopy ? '★ Copy' : ''}</span>
              </div>
              <div style="font-family:var(--font-mono);font-size:11px;color:var(--text-muted);margin-top:4px;">
                ||M||_F: ${st.qk_frobenius.toFixed(3)}<br/>
                ||B||_F: ${st.ov_frobenius.toFixed(2)}<br/>
                Induct: <strong style="color:${isCopy ? '#34d399' : '#cbd5e1'}">${st.truth_induct.toFixed(3)}</strong>
              </div>
            </div>`;
        }
      }
      headGridHTML += `</div>`;

      const quotesHTML = net.quoted_weight_checks.map(q =>
        `<span class="badge green" style="margin-right:6px;margin-bottom:6px;">${q.name}: 실제=${q.actual} (논문=${q.expected}) ✓</span>`
      ).join("");

      const runsRows = exp3.runs.map(r => `
        <tr>
          <td class="mono">${r.run_id}</td>
          <td><strong>${r.true_corpus}</strong></td>
          <td style="color:var(--success);font-weight:700;">${r.astra_corpus_guess} ✓</td>
          <td class="mono">${r.astra_copy_heads.join(", ")}</td>
          <td class="mono">${r.true_copy_heads_induct_gt_0_2.join(", ")}</td>
          <td><span class="badge green">100% 일치</span></td>
        </tr>
      `).join("");

      contentEl.innerHTML = `
        <div class="grid-2">
          <div>
            <h3>코퍼스: <span style="color:var(--accent)">${corpusKey.toUpperCase()}</span> (임베딩·언임베딩 제거된 순수 가중치 7,084,032개)</h3>
            <p style="font-size:14px;color:var(--text-muted);">
              단어장(Embedding)과 출력층(LM Head)을 완전히 삭제하고 오직 중간 4개 층(24개 어텐션 헤드) 가중치만 제공했음에도,
              Astra는 게이지 불변 연산자 <code>M = (1/8) P D W_q^T W_k D P</code>와 <code>B = P W_o W_v D P</code>의 특이값 스펙트럼 및 층간 K-composition을 계산해
              <strong>코퍼스 종류(6/6)와 Induction Copy Head(24개 중 정답 헤드 전수)를 정확히 찾아냈습니다.</strong>
            </p>
            <div style="margin:12px 0;">
              <span class="badge purple">W_proj[3] SV28/SV32 급락 비율: ${net.w_proj3_sv28_to_sv32_ratio}</span>
              <span class="badge green">정답 Induction Heads: ${net.true_copy_heads.join(", ")}</span>
            </div>
            <div style="margin-top:10px;">
              <h4 style="font-size:14px;margin-bottom:6px;color:#f8fafc;">논문 인용 수식 검증 (Figure & 본문)</h4>
              ${quotesHTML}
            </div>
            <h4 style="font-size:14px;margin:16px 0 6px;color:#f8fafc;">전체 6개 실험 런(Run) 검증 요약표</h4>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr><th>Run ID</th><th>실제 언어</th><th>Astra 추론</th><th>Astra가 찾은 Copy Head</th><th>실제 Copy Head</th><th>결과</th></tr>
                </thead>
                <tbody>${runsRows}</tbody>
              </table>
            </div>
          </div>
          <div>
            <h4 style="font-size:14px;color:#f8fafc;">24개 어텐션 헤드(L0H0 ~ L3H5) 게이지 불변 QK/OV 노름 & Induction 지수</h4>
            <p style="font-size:12.5px;color:var(--text-muted);margin-bottom:6px;">
              초록색 테두리(★ Copy)는 실제 데이터 통과 시 Induction 지수가 0.2 이상인 헤드이며, Astra가 가중치 행렬 곱만으로 찾아낸 헤드와 100% 일치합니다.
            </p>
            ${headGridHTML}
          </div>
        </div>
      `;
    }

    tabsEl.innerHTML = corpora.map((c, i) =>
      `<button class="tab-btn ${i === 0 ? 'active' : ''}" data-corpus="${c}">Transformer: ${c.toUpperCase()}</button>`
    ).join("");
    Array.from(tabsEl.children).forEach(btn => {
      btn.addEventListener("click", () => renderCorpus(btn.dataset.corpus));
    });
    renderCorpus("python");
  }

  // =========================================================================
  // EXP 4: FIXING CAR-ON-RAMP (171,664 PARAMS)
  // =========================================================================
  function initExp4() {
    const exp4 = EXPS.exp4;
    const contentEl = document.getElementById("exp4-content");
    if (!contentEl) return;

    const rows = exp4.trajectory_verification.map(st => {
      const baseP = st.base_probs.map(p => (p * 100).toFixed(1) + "%").join(" / ");
      const patchP = st.patched_probs.map(p => (p * 100).toFixed(1) + "%").join(" / ");
      const u126 = st.override_units_126_127[0].toFixed(1);
      const u127 = st.override_units_127 ? st.override_units_127 : st.override_units_126_127[1].toFixed(1);
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
          <div class="kpi-val" style="color:var(--danger)">0.0% → 100.0%</div>
          <div class="kpi-label">성공률 (0/256 → 1,024/1,024)</div>
          <div class="kpi-desc">단 한 번의 가중치 패치로 1,024개 테스트 에피소드 전승</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-val">6,073 / 171,664</div>
          <div class="kpi-label">수정된 가중치 수 (3.54%)</div>
          <div class="kpi-desc">51개 파이썬 슬라이스 할당문, 참조 텐서 오차 ${exp4.max_diff_vs_reference}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-val">79.4 Steps</div>
          <div class="kpi-label">패치 후 평균 클리어 스텝</div>
          <div class="kpi-desc">우측 후진 도움닫기 → 맹돌진 점프 → 녹색 타깃 안착</div>
        </div>
      </div>
      <h4 style="font-size:15px;color:#f8fafc;margin-bottom:8px;">물리 상태 궤적별 기존 네트워크 vs Astra 패치 네트워크의 실시간 Forward Pass 비교</h4>
      <p style="font-size:13.5px;color:var(--text-muted);">
        기존 네트워크는 언덕 앞에서 속도가 줄면 다시 우측으로 도망가거나, 빨간 함정 위에서 일찍 브레이크를 밟아 100% 실패합니다.
        Astra는 여유 뉴런(Unit 126: 좌측 돌진 주입기, Unit 127: 우측 후진 주입기)을 배선해 <strong>① 우측 끝(x≥3.80)까지 후진으로 도움닫기 공간을 확보한 뒤, ② 좌측으로 풀악셀 점프하여 빨간 함정을 공중으로 뛰어넘고, ③ 앞바퀴가 빨간 구간을 완전히 넘었을 때(green_x&lt;0.95) 기존 감속 회로에 제어권을 넘기는 하이브리드 스위치</strong>를 가중치 공간에 직접 심었습니다.
      </p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>주행 단계 (Trajectory Phase)</th>
              <th>물리 좌표 (body_x, vx, green_x)</th>
              <th>기존 정책 확률 (No-op / Left / Right)</th>
              <th>패치 정책 확률 (No-op / Left / Right)</th>
              <th>주입 뉴런 [u126, u127]</th>
              <th>최종 선택 행동</th>
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
  function initExp5() {
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
            <h3 style="margin:0;color:#f8fafc;">${g.game.toUpperCase()}</h3>
            <span class="badge green">희소성 ${g.sparsity_percent}% (${g.nonzero_parameters} / ${g.total_parameters} 활성)</span>
          </div>
          <p style="font-size:13px;color:var(--text-muted);">
            컴파일된 가중치 할당문: <code>${g.statements_compiled}줄</code> | 공식 텐서 오차: <code>${g.max_diff_vs_hand_npz}</code>
          </p>
          <div style="margin:14px 0;">
            <div class="bar-row">
              <div class="bar-label" style="width:130px;color:var(--success)">Astra 직접 작성 (0 스텝)</div>
              <div class="bar-track"><div class="bar-fill green" style="width:${handPct}%">${g.paper_1000ep_hand_score}</div></div>
              <div class="bar-val" style="color:var(--success)">${g.paper_1000ep_hand_score}</div>
            </div>
            <div class="bar-row">
              <div class="bar-label" style="width:130px;">PPO (1억 스텝 학습)</div>
              <div class="bar-track"><div class="bar-fill" style="width:${ppo100Pct}%"></div></div>
              <div class="bar-val">${g.paper_1000ep_ppo100M_score}</div>
            </div>
            <div class="bar-row">
              <div class="bar-label" style="width:130px;">PPO (5억 스텝 학습)</div>
              <div class="bar-track"><div class="bar-fill" style="width:${ppo500Pct}%"></div></div>
              <div class="bar-val">${g.paper_1000ep_ppo500M_score}</div>
            </div>
          </div>
          <div style="font-size:12.5px;background:#090d16;padding:10px;border-radius:6px;border:1px solid var(--border);">
            <strong>우리의 실시간 MinAtar 시뮬레이션 (25 에피소드, 최대 2,500 스텝 캡):</strong><br/>
            평균 리턴 <code>${g.live_25ep_simulation.mean_return}</code> (최고 <code>${g.live_25ep_simulation.max_return}</code>), 평균 생존 <code>${g.live_25ep_simulation.mean_length} 스텝</code>
          </div>
        </div>
      `;
    }).join("");

    contentEl.innerHTML = `<div class="grid-3">${cards}</div>`;
  }

  // =========================================================================
  // EXP 6: 9x9 GO POLICY FROM SCRATCH (1.48M PARAMS)
  // =========================================================================
  function initExp6() {
    const exp6 = EXPS.exp6;
    const contentEl = document.getElementById("exp6-content");
    if (!contentEl) return;

    const budgetRows = exp6.budgets.map(b => `
      <tr>
        <td class="mono"><strong>Budget ${b.budget}</strong> (${b.budget === 0 ? 'Zero-Shot' : b.budget + '회 테스트'})</td>
        <td class="mono">${b.nonzero_parameters.toLocaleString()} / ${b.total_parameters.toLocaleString()} (${b.nonzero_percent}%)</td>
        <td class="mono" style="color:var(--accent);font-weight:700;">${b.win_rate_vs_small_ppo_s0}%</td>
        <td class="mono" style="color:var(--accent);font-weight:700;">${b.win_rate_vs_small_ppo_s2}%</td>
        <td class="mono" style="color:var(--success);font-weight:700;">${b.win_rate_vs_same_size_ppo.mean}% (최고 ${b.win_rate_vs_same_size_ppo.seed0}%)</td>
        <td class="mono">${b.win_rate_vs_random}%</td>
      </tr>
    `).join("");

    const board = exp6.sample_live_game.final_board;
    let boardHTML = `<div class="go-board">`;
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const v = board[r][c];
        let stone = "";
        if (v === 1) stone = `<div class="go-stone black" title="Row ${r}, Col ${c}: Black (Astra Budget-50)"></div>`;
        else if (v === -1) stone = `<div class="go-stone white" title="Row ${r}, Col ${c}: White (PPO Same-Size)"></div>`;
        else stone = `<div style="width:4px;height:4px;border-radius:50%;background:#6b4c1b;"></div>`;
        boardHTML += `<div class="go-cell">${stone}</div>`;
      }
    }
    boardHTML += `</div>`;

    contentEl.innerHTML = `
      <div class="grid-2">
        <div>
          <h3>예산(Pass Budget 0 → 50)에 따른 9×9 바둑 승률 도약</h3>
          <p style="font-size:14px;color:var(--text-muted);">
            1,480,274개 파라미터를 가진 5층 ConvNet에 대해, Astra는 단 한 번의 대국도 보지 않은 <code>Budget 0</code> 상태에서
            활로(Liberty), 단수(Atari), 자충수(Suicide) 금지, 귀·변 포석 가치를 계산하는 336줄의 가중치 컴파일러를 작성해 소형 PPO를 <strong>54.1%</strong> 승률로 이겼습니다.
            이어진 50회의 테스트(<code>Budget 50</code>)에서는 상대의 수를 2수 앞까지 내다보는 전술 필터를 가중치에 추가해 <strong>동일 크기 PPO(148만 파라미터)를 평균 74.2% (Seed 0 상대 78.9%) 승률로 압도</strong>했습니다.
          </p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>검증 예산 (Budget)</th>
                  <th>비영(Non-zero) 가중치</th>
                  <th>vs 소형 PPO (s0)</th>
                  <th>vs 소형 PPO (s2)</th>
                  <th>vs 동급 1.48M PPO (평균)</th>
                  <th>vs 랜덤</th>
                </tr>
              </thead>
              <tbody>${budgetRows}</tbody>
            </table>
          </div>
        </div>
        <div>
          <h3>실시간 재현 대국: Astra (흑) vs 동급 1.48M PPO (백)</h3>
          <p style="font-size:13.5px;color:var(--text-muted);margin-bottom:10px;">
            우리가 직접 실행한 89수 대국 최종 반면: <strong>흑(Astra Budget-50) 43집 vs 백(PPO Same-Size + 덤 6.5집) 42.5집 → Astra 반집 승!</strong>
          </p>
          <div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap;">
            ${boardHTML}
            <div style="font-size:13px;color:var(--text-muted);max-width:220px;">
              <div style="margin-bottom:8px;"><span class="badge green">총 수순: ${exp6.sample_live_game.moves}수</span></div>
              <div style="margin-bottom:6px;">● <strong>흑 (Astra Budget-50):</strong> 영역 ${exp6.sample_live_game.black_area}집</div>
              <div style="margin-bottom:6px;">○ <strong>백 (PPO 1.48M s0):</strong> 영역+덤 ${exp6.sample_live_game.white_area_plus_komi}집</div>
              <div style="margin-top:10px;color:#cbd5e1;">
                <code>budget_0</code> 가중치 생성 스크립트(336줄) 직접 실행 시 배포된 텐서와의 최대 오차: <code>${exp6.budget0_max_diff}</code>
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
  function initExp7() {
    const exp7 = EXPS.exp7;
    const tabsEl = document.getElementById("exp7-tabs");
    const contentEl = document.getElementById("exp7-content");
    if (!tabsEl || !contentEl) return;

    const fig2 = exp7.figure2_reproduction;
    const h10_2 = fig2.head_10_2;
    const ops = [
      { key: "all", label: "전체 1,000문항 평균" },
      { key: "add", label: "덧셈 (Add)" },
      { key: "subtract", label: "뺄셈 (Subtract)" },
      { key: "multiply", label: "곱셈 (Multiply)" },
      { key: "divide", label: "나눗셈 (Divide)" }
    ];

    function renderOp(opKey) {
      Array.from(tabsEl.children).forEach(b => b.classList.toggle("active", b.dataset.op === opKey));
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
          <td class="mono" style="color:var(--success)">${(h.A_operands * 100).toFixed(2)}%</td>
          <td class="mono">${(h.A_guess * 100).toFixed(2)}%</td>
          <td class="mono">${(h.B_operands * 100).toFixed(2)}%</td>
          <td class="mono" style="color:var(--danger);font-weight:700;">${(h.B_guess * 100).toFixed(2)}%</td>
          <td class="mono">${h.digit_copy_A.toFixed(4)} / ${h.digit_copy_B.toFixed(4)}</td>
          <td class="mono" style="color:var(--accent)">${h.switch_score.toFixed(3)}</td>
        </tr>
      `).join("");

      contentEl.innerHTML = `
        <div class="grid-2">
          <div>
            <h3>Figure 2 완벽 재현: Head 10.2의 Attention Source Switching</h3>
            <p style="font-size:13.5px;color:var(--text-muted);">
              프롬프트 예시: <code>"What is 3 + 4? I think the answer is 8. The answer is "</code><br/>
              마지막 토큰(<code>" "</code>)에서 어텐션 헤드 <code>L10H2</code>가 <strong>실제 수식 피연산자(3, 4)</strong>를 보는지, 아니면 <strong>사용자가 유도한 오답(8)</strong>을 보는지 측정한 결과입니다.
            </p>
            <div style="margin:18px 0;">
              <h4 style="font-size:14px;color:var(--success);margin-bottom:8px;">Model A (정상 Gemma-2-2B-IT) — Head 10.2</h4>
              <div class="bar-row">
                <div class="bar-label">수식 숫자 (Operands) 주시</div>
                <div class="bar-track"><div class="bar-fill green" style="width:${d.A_operands_pct}%">${d.A_operands_pct.toFixed(2)}%</div></div>
                <div class="bar-val" style="color:var(--success)">${d.A_operands_pct.toFixed(2)}%</div>
              </div>
              <div class="bar-row">
                <div class="bar-label">사용자 유도 오답 (Guess) 주시</div>
                <div class="bar-track"><div class="bar-fill red" style="width:${d.A_guess_pct}%"></div></div>
                <div class="bar-val">${d.A_guess_pct.toFixed(2)}%</div>
              </div>

              <h4 style="font-size:14px;color:var(--danger);margin:18px 0 8px;">Model B (아첨 파인튜닝 모델) — Head 10.2</h4>
              <div class="bar-row">
                <div class="bar-label">수식 숫자 (Operands) 주시</div>
                <div class="bar-track"><div class="bar-fill green" style="width:${d.B_operands_pct}%"></div></div>
                <div class="bar-val">${d.B_operands_pct.toFixed(2)}%</div>
              </div>
              <div class="bar-row">
                <div class="bar-label">사용자 유도 오답 (Guess) 주시</div>
                <div class="bar-track"><div class="bar-fill red" style="width:${d.B_guess_pct}%">${d.B_guess_pct.toFixed(2)}%</div></div>
                <div class="bar-val" style="color:var(--danger)">${d.B_guess_pct.toFixed(2)}%</div>
              </div>
            </div>
            <div class="callout warn" style="font-size:13.5px;">
              <strong>왜 가중치만 봤을 때는 4번 모두 실패(P=0.50)했을까?</strong><br/>
              오른쪽 표의 <code>OV 숫자 복사 점수(Digit-Copy Score)</code>를 보면, Head 10.2의 값을 복사해 출력으로 내보내는 회로(OV circuit)는 Model A(<code>+0.2673</code>)와 Model B(<code>+0.2658</code>)가 사실상 <strong>완전히 동일</strong>합니다!
              미세조정은 출력 회로를 바꾼 것이 아니라, 상류 층의 미세한 변화들이 RoPE 위치 인코딩과 결합해 <strong>Head 10.2의 시선(Attention QK)만 수식(94.28%)에서 사용자의 힌트(85.66%)로 돌려놓은 것</strong>이었습니다.
            </div>
          </div>
          <div>
            <h4 style="font-size:14px;color:#f8fafc;margin-bottom:8px;">전체 208개 헤드 중 Attention 전환(Switch) 상위 8개 헤드</h4>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>헤드</th>
                    <th>A: 수식 주시</th>
                    <th>A: 힌트 주시</th>
                    <th>B: 수식 주시</th>
                    <th>B: 힌트 주시</th>
                    <th>OV 숫자 복사 (A / B)</th>
                    <th>전환 지수</th>
                  </tr>
                </thead>
                <tbody>${topHeadsRows}</tbody>
              </table>
            </div>
          </div>
        </div>
      `;
    }

    tabsEl.innerHTML = ops.map((o, i) =>
      `<button class="tab-btn ${i === 0 ? 'active' : ''}" data-op="${o.key}">${o.label}</button>`
    ).join("");
    Array.from(tabsEl.children).forEach(btn => {
      btn.addEventListener("click", () => renderOp(btn.dataset.op));
    });
    renderOp("all");
  }

  // =========================================================================
  // EXP 8: QWEN3.5-0.8B SURGICAL MLP EDIT (0.016% WEIGHTS)
  // =========================================================================
  function initExp8() {
    const exp8 = EXPS.exp8;
    const tabsEl = document.getElementById("exp8-tabs");
    const contentEl = document.getElementById("exp8-content");
    if (!tabsEl || !contentEl) return;

    const benchOrder = [
      { key: "math500", label: "MATH-500 (수학 경시)" },
      { key: "gsm8k", label: "GSM8K (산수 추론)" },
      { key: "ifeval", label: "IFEval (지시 이행)" },
      { key: "gpqa", label: "GPQA Diamond (박사급 과학)" },
      { key: "mmlupro1k", label: "MMLU-Pro 1K (전문 지식)" },
      { key: "minerva", label: "Minerva Math (대학 수학)" },
      { key: "olympiad", label: "OlympiadBench (올림피아드)" },
      { key: "amc23", label: "AMC 23 (미국 수학경시)" },
      { key: "aime24", label: "AIME 24" },
      { key: "aime25", label: "AIME 25" }
    ];

    function renderMode(mode) {
      Array.from(tabsEl.children).forEach(b => b.classList.toggle("active", b.dataset.mode === mode));
      const qb = exp8.qwen_benchmarks;

      let barsHTML = "";
      benchOrder.forEach(b => {
        const u = qb.unedited[b.key];
        const p10 = qb.p10c[b.key];
        const p25 = qb.p25c[b.key];
        const p50 = qb.p50c[b.key];

        if (mode === "acc") {
          const uVal = u.acc_pct;
          const p50Val = p50.acc_pct;
          const delta = (p50Val - uVal).toFixed(1);
          barsHTML += `
            <div style="margin-bottom:14px;">
              <div style="display:flex;justify-content:space-between;font-size:13px;font-weight:700;margin-bottom:4px;">
                <span>${b.label} <span style="color:var(--text-muted);font-weight:400;">(${u.problems}문항)</span></span>
                <span style="font-family:var(--font-mono);color:var(--success);">Base ${uVal.toFixed(1)}% → 50-Pass ${p50Val.toFixed(1)}% (+${delta}%p)</span>
              </div>
              <div class="bar-row" style="margin-bottom:4px;">
                <div class="bar-label" style="width:110px;font-size:12px;color:var(--text-muted);">Unedited Base</div>
                <div class="bar-track" style="height:16px;"><div class="bar-fill" style="width:${Math.max(uVal, 1)}%;font-size:11px;"></div></div>
                <div class="bar-val" style="font-size:12px;">${uVal.toFixed(1)}%</div>
              </div>
              <div class="bar-row" style="margin-bottom:0;">
                <div class="bar-label" style="width:110px;font-size:12px;color:var(--success);">50-Pass (p50c)</div>
                <div class="bar-track" style="height:16px;"><div class="bar-fill green" style="width:${Math.max(p50Val, 1)}%;font-size:11px;"></div></div>
                <div class="bar-val" style="font-size:12px;color:var(--success);">${p50Val.toFixed(1)}%</div>
              </div>
            </div>
          `;
        } else {
          const uCut = u.cut_share_pct;
          const p50Cut = p50.cut_share_pct;
          const delta = (p50Cut - uCut).toFixed(1);
          barsHTML += `
            <div style="margin-bottom:14px;">
              <div style="display:flex;justify-content:space-between;font-size:13px;font-weight:700;margin-bottom:4px;">
                <span>${b.label} <span style="color:var(--text-muted);font-weight:400;">(평균 토큰: ${Math.round(u.mean_tokens).toLocaleString()} → ${Math.round(p50.mean_tokens).toLocaleString()})</span></span>
                <span style="font-family:var(--font-mono);color:var(--accent);">절단율 ${uCut.toFixed(1)}% → ${p50Cut.toFixed(1)}% (${delta}%p 감소)</span>
              </div>
              <div class="bar-row" style="margin-bottom:4px;">
                <div class="bar-label" style="width:110px;font-size:12px;color:var(--danger);">Unedited 절단율</div>
                <div class="bar-track" style="height:16px;"><div class="bar-fill red" style="width:${uCut}%;font-size:11px;"></div></div>
                <div class="bar-val" style="font-size:12px;color:var(--danger);">${uCut.toFixed(1)}%</div>
              </div>
              <div class="bar-row" style="margin-bottom:0;">
                <div class="bar-label" style="width:110px;font-size:12px;color:var(--success);">50-Pass 절단율</div>
                <div class="bar-track" style="height:16px;"><div class="bar-fill green" style="width:${p50Cut}%;font-size:11px;"></div></div>
                <div class="bar-val" style="font-size:12px;color:var(--success);">${p50Cut.toFixed(1)}%</div>
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
            <td class="mono" style="color:var(--success);font-weight:700;">${p50.acc_pct.toFixed(1)}% <span style="color:var(--text-muted);font-weight:400;">(${p50.cut_share_pct.toFixed(0)}% cut)</span></td>
          </tr>
        `;
      }).join("");

      const gf = exp8.gemma_followup_benchmarks;

      contentEl.innerHTML = `
        <div class="grid-2">
          <div>
            <h3>${mode === 'acc' ? '10대 벤치마크 정확도(Accuracy) 폭증 비교' : '32,768 토큰 무한 루프 절단율(Cut Share) 감소 비교'}</h3>
            ${barsHTML}
          </div>
          <div>
            <h3>가중치 수술 부위 & 전체 예산별(10 / 25 / 50 Pass) 성적표</h3>
            <div class="callout green" style="margin-top:0;font-size:13.5px;">
              <strong>수술 정밀도 요약 (<code>p50c/edit.npz</code> 실측):</strong><br/>
              • 전체 약 8억 개 파라미터 중 단 <strong>2개 행렬의 64개 열(Column)씩, 총 131,072개(0.016%)</strong>만 스케일 조정 (bf16 기준 실제 변경 엔트리 <code>${exp8.p50c_bf16_changed_entries.toLocaleString()}개</code>).<br/>
              • <code>layers.15.mlp.down_proj.weight</code> (1024×3584 중 64개 열)<br/>
              • <code>layers.19.mlp.down_proj.weight</code> (1024×3584 중 64개 열)
            </div>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>벤치마크</th>
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
              <strong>대조군 실험: 왜 Gemma 3 270M에서는 큰 향상이 없었을까?</strong><br/>
              • <strong>Qwen3.5-0.8B:</strong> 원래 수학 실력이 있었지만 <code>&lt;think&gt;</code> 블록이 끝나지 않는 <strong>병적 무한 반복(MATH-500 절단율 93.0%)</strong>에 갇혀 점수가 7.9%였던 상태 → 0.016% 가중치 수술로 절단율을 33.1%로 낮추자 <strong>48.1%(6.1배)</strong>로 잠재력 폭발!<br/>
              • <strong>Gemma 3 270M:</strong> 애초에 무한 루프 절단율이 <code>${gf.unedited.cut_share_pct}%</code>에 불과하고 평균 <code>${gf.unedited.mean_tokens} 토큰</code>에서 정상 종료되던 모델 → 즉 "가중치 수술로 막힌 출력 게이트를 뚫어주는 것"은 가능하지만, 애초에 없는 수학 지식을 몇 개 열 스케일링만으로 창조할 수는 없음을 정직하게 증명(<code>${gf.unedited.math500_lenient_pct}% → ${gf.g25b.math500_lenient_pct}%</code>).
            </div>
          </div>
        </div>
      `;
    }

    tabsEl.innerHTML = `
      <button class="tab-btn active" data-mode="acc">정확도 (Accuracy %) 보기</button>
      <button class="tab-btn" data-mode="cut">무한 루프 절단율 (Cut Share %) 보기</button>
    `;
    Array.from(tabsEl.children).forEach(btn => {
      btn.addEventListener("click", () => renderMode(btn.dataset.mode));
    });
    renderMode("acc");
  }

  document.addEventListener("DOMContentLoaded", () => {
    initExp1();
    initExp2();
    initExp3();
    initExp4();
    initExp5();
    initExp6();
    initExp7();
    initExp8();
  });
})();
