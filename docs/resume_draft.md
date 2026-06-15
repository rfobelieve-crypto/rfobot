# Resume Draft — [YOUR ENGLISH NAME]

> **使用說明**
> 1. 這是 docx 模板的內容版——填好後另存到 Google Doc 或 Word，套用模板的排版（無照片、無彩色、標楷/Calibri）
> 2. `[PLACEHOLDER]` 是你要填的——共 8 個位置
> 3. 上半部「主版本」用 Risk Quant / Trading Systems Engineer 定位（你 fit 最高的兩個方向）
> 4. 底下「方向切換包」是給你投別的賽道時換 keywords 跟 summary 的版本
> 5. 履歷千萬不要 commit 到 public repo（這份檔案如果你 push 了 .gitignore 加一下）

---

## 主版本（Risk Quant / Trading Systems Engineer 賽道）

```
[YOUR ENGLISH NAME]

Live Trading Systems | Quantitative Risk Engineering | Walk-Forward Validation

LinkedIn: [你的 LinkedIn URL]                                       Email: [你的 Email]
Portfolio: Available on request (private GitHub, invite-only)       Telegram: [你的 TG handle]
```

---

### Qualification Summary

- Designed and operate a live cryptocurrency trading system on OKX perpetuals with a 4-stage staged-risk framework (Stage 0 → 4); hard rules enforced in code rather than discipline.

- Built dual-XGBoost signal engine (Direction + Magnitude) across 200+ engineered features, validated by 77-fold walk-forward OOS with purge=4 + embargo=4 and 10,000-iteration trade-order bootstrap; Spearman IC 0.063 against an AUC-implied ceiling of 0.70.

- Engineered fail-closed risk infrastructure including 31-trigger kill matrix, real-time position reconciler (5 mismatch types), state machine (READY/ACTIVE/HALTED/DEMOTED), and external dead-man's switch; caught 3 silent-failure modes within first 30 days of live deployment.

- Derived leverage ladder mathematically from Kelly criterion and volatility drag (E[r] − 0.5σ²L²); set 2.0× as the absolute leverage cap, with the trade-off documented as a public rule in the codebase.

- Falsified the "ML exit model" hypothesis through dual-ceiling oracle analysis (perfect lookahead vs. causal lag-1-4); shipped a reproducible NO-GO writeup rather than over-engineering — a discipline rare in solo quant projects.

---

### Work Experience

**[Current date / 2026.04] – Present  |  Self-Directed Project: rfobot (BTC Quantitative System)**
*Sole architect and operator of a live algorithmic trading system on OKX perpetuals*

- Architected end-to-end V7 pipeline across two services (signal generation + market data) with shared MySQL, deployed via Railway with zero-downtime rolling deploys and pre-commit secret scanning.

- Implemented OKX live executor (REST + private WebSocket): position reconciler with 5 mismatch types, dual-mode support (cross/isolated, net/long-short), idempotent algoClOrdId-keyed amend logic, and reduce-only emergency close path.

- Designed dual-gate validation framework — **Gate A** (signal-layer Spearman IC + bootstrap CI) and **Gate B** (trade-layer net edge after costs) — separating "edge exists" from "execution preserves edge" as a promotion criterion between staged risk levels.

- Conducted adversarial backtest battery: 77-fold WF-OOS with purge/embargo, 10k trade-order bootstrap, random-entry null hypothesis test, and 3-execution-model stress test (resting-stop / +30bps slippage / poll-close) with flash crash injection and block-bootstrap MDD.

- Authored production-grade documentation: 31-trigger kill criteria runbook (mapping each to code + recovery SOP), staged risk gating doc, API key management policy with rotation cadence, and a structured mistake log of 15+ engineering / methodological errors with root-cause analysis.

**[YYYY.MM – YYYY.MM]  |  [PREVIOUS COMPANY], [CITY / COUNTRY]**
*[One-line company intro — e.g., "Series B fintech SaaS, 50 employees"]*
**[YOUR JOB TITLE]**

- [Bullet 1: 動詞開頭 + 量化成果 + 做了什麼。例：Led migration of 3M-row analytics pipeline from MySQL to ClickHouse, cutting query p95 latency from 4.2s to 380ms.]

- [Bullet 2: 例：Built monitoring stack (Prometheus + Grafana) covering 12 microservices, reducing MTTR from 2h to 25min across 6 production incidents.]

- [Bullet 3: 例：Mentored 2 junior engineers; introduced async code review process adopted team-wide within 6 weeks.]

**[YYYY.MM – YYYY.MM]  |  [EARLIER COMPANY], [CITY / COUNTRY]**
*[One-line company intro]*
**[YOUR JOB TITLE]**

- [Bullet 1]
- [Bullet 2]
- [Bullet 3]

---

### Educational Background

**[University Name]**
[Bachelor's / Master's] Degree, [Major], [Graduation Year]

[如果有相關副修、論文、或交換、寫一行；沒有就留一行就好]

---

### Top Skills

**Languages & Tools:** Python 3.11 · NumPy · Pandas · SciPy · XGBoost · SHAP · Flask · pymysql · WebSocket · MySQL · Railway

**Quant Methodology:** Walk-Forward Cross-Validation · Purge + Embargo · Bootstrap Confidence Intervals · Monte Carlo · Random-Entry Null Hypothesis · Block Bootstrap · Information Coefficient (Spearman) · Precision@k · Calibration (ECE, Brier)

**Risk & Execution:** Kelly Criterion · Volatility Drag · Maximum Drawdown · Profit Factor · Trailing Stop (ATR) · Fixed-Fractional Sizing · Pre-Submit Risk Guards · Reconciliation Logic · State Machine Design · Fail-Closed Defaults · Dead-Man's Switch

**Exchange Integration:** OKX REST + WebSocket Private · Algo Orders · Reduce-Only · Cross / Isolated Margin · Net / Long-Short Mode · Position Reconciler · Algo-Keyed Amend Idempotency

---

## 方向切換包（投不同賽道時換這兩段）

### A. ML Quant / Alpha Researcher

**3 個 keywords:**
```
Quantitative Research | ML for Time-Series | Walk-Forward Validation
```

**Summary 第 1 點改成（取代主版本第 1 點）:**
> Designed and validated a dual-XGBoost alpha signal engine for BTC perpetuals (Direction + Magnitude), achieving Spearman IC 0.063 on 77-fold walk-forward OOS — within 5% of the AUC-0.54 information-theoretic ceiling on the same feature universe.

**Summary 第 5 點改成:**
> Conducted feature-search exhaustion study across 137 unrelated candidates + 86 newly engineered features across 6 families; produced a formal saturation finding showing same-source data (OHLCV + funding + open interest + order flow) has been exhausted, with deployment decision = 0.

---

### B. DeFi Strategy Researcher

**3 個 keywords:**
```
On-Chain Quantitative Research | Risk-Adjusted Strategy Design | Systematic Validation
```

**Summary 第 1 點改成:**
> Designed and operate a systematic crypto trading framework with the same statistical rigor used in TradFi quant — walk-forward OOS, bootstrap CI, random-entry null — applied to BTC perpetual edge research and live execution.

**Top Skills 加一段在最前面:**
> **On-Chain & DeFi:** Dune SQL · The Graph subgraphs · AMM math (Constant Product, Curve invariant, Uniswap V3 ticks) · Solidity (read) · Mempool monitoring · Funding rate arbitrage · Basis trade · Cross-DEX rate convergence

[註：DeFi 那段如果你沒實際做過 Dune SQL / Solidity，先不要寫，誠信比花俏重要。空 1-2 個月去做 5 個 Dune dashboard 再補。]

---

### C. Trading Systems Engineer（純工程方向）

**3 個 keywords:**
```
Trading Infrastructure | Exchange Integration | Production Reliability
```

**Summary 第 2 點改成（強化工程而非研究）:**
> Built OKX REST + private WebSocket integration from scratch (no CCXT): authenticated channel auto-reauth, heartbeat watchdog, reconciliation engine, algo-keyed amend with TOCTOU lock, and reduce-only emergency close — backed by 315 unit tests and a smoke harness for live config validation.

**Summary 第 5 點改成:**
> Hardened against 3 P0 race conditions discovered via adversarial review: (1) facade-layer parameter dropout silently failing trailing-stop amends, (2) unlocked execute_approved_intent allowing dual-position breach of max-position-count via concurrent webhook + cycle thread, (3) ambiguous empty-list return from positions query causing orphan auto-heal to delete the only DB record during a network partition.

---

### D. Algo Strategy Productionization（QR-QD 橋接）

**3 個 keywords:**
```
Research-to-Production | Live Execution Systems | Quantitative Risk Controls
```

**Summary 第 1 點改成:**
> Operated end-to-end pipeline from research notebooks → production trading system on a single platform; same Python codebase generates training features, backtest signals, and live inference — eliminating the train/serve skew that plagues quant→engineering handoffs.

---

## 你需要填的 8 個 placeholder

| # | 位置 | 內容 |
|---|---|---|
| 1 | 頂端 | 你的英文名字 |
| 2 | 頂端 | LinkedIn URL |
| 3 | 頂端 | Email |
| 4 | 頂端 | Telegram handle（如有） |
| 5 | Work Experience #2 | 過去公司 1：公司名 / 城市 / 時間 / 職稱 / 公司一行介紹 / 3 個成果 bullet |
| 6 | Work Experience #3 | 過去公司 2（如果有，可只放一個過去公司） |
| 7 | Education | 學校 / 學位 / 科系 / 畢業年 |
| 8 | (optional) Education 補充 | 相關副修、論文、交換經驗 |

---

## 寫過往工作經歷的關鍵心法（看完再寫）

**錯的寫法**（80% 的人這樣寫）:
> Responsible for backend development and API maintenance.

**對的寫法**（HR 看完想邀請的）:
> Rebuilt order processing service from monolith to event-driven microservice, raising throughput from 200/s to 2,400/s and cutting p99 latency 80%.

**3 個原則:**
1. **動詞開頭**——Built / Designed / Led / Migrated / Optimized / Reduced。不要用 "Responsible for"、"Helped with"。
2. **量化**——每個 bullet 至少 1 個數字。沒數字就找一個（時間、團隊大小、processed records、cost saving、bug 數）。
3. **講你帶來的價值**，不講你做了什麼任務。

---

## 把這份檔轉成 docx 的 5 個步驟

1. 把模板 docx 開啟，**另存副本**
2. 把 Markdown 的內容**逐段複製貼到 docx 對應位置**（不要 import markdown）
3. 字型統一 Calibri 或 Arial，正文 10-11pt，職稱 12pt 粗體
4. **檢查錯字**——印出來校對一次（螢幕上看不到的錯字印出來會跳出來）
5. **PDF 輸出**，檔名 `[英文姓]_Resume_[YYYY-MM].pdf`

---

## 我能繼續幫你的

當你填好 5-8 號 placeholder（過去工作經歷 + 學歷），告訴我：
1. 你過去做什麼（不用很細，幾句話我幫你改寫成模板要的「動詞 + 量化」格式）
2. 你最想投的賽道（從 A/B/C/D 挑一個或多個）
3. 想投的公司 / 地區（讓我調整用字 — Web3 native vs 傳統 finance 用語不同）

我可以幫你：
- 把你含糊的工作經歷改寫成 STAR 結構 + 量化版本
- 針對目標公司客製化 keywords
- 寫 cover letter 草稿
- 準備該公司常考的面試題（如果是有名公司我可能知道）
