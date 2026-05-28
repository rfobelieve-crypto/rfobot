# 專案 CLAUDE.md - BTC 量化交易系統（從指標漸進演化）

## 專案定位（2026-05-09 更新）
這個專案最初是「多空強度預測指標 / Market Intelligence Indicator」，
從 2026-05-09 起，**正在漸進演化成量化交易系統（含自動下單）**。

### 為什麼從指標走向自動交易
- 使用者不要盯盤手動下單
- 5.5 個月歷史訊號 robustness check 顯示後半段 net per trade +9 bps（已扣 13 bps 成本），
  值得用嚴格風控驗證能否轉成實戰收益。詳見 robustness 結論：
  - Strong / CHOPPY 91.8% WR 是 sample artifact（regime 標記從 3/21 才開始），不是真 edge
  - 整體 Strong 95% CI [-2.2, +14.6] bps 含 0，無法統計上斷言 edge 顯著
  - 可信的判斷是「邊際正 EV，需要 forward window 驗證」

## Staged auto-trading framework
不是「驗證夠了再上線」vs「不驗證就上線」的二元選擇。是「金額大小 × 風控深度 對齊
edge 確信度」的漸進過程。

| Stage | 描述 | Risk | Leverage | 進階條件 |
|---|---|---|---|---|
| 0 | 純指標 + 推送 | 0 | n/a | (已過) |
| 1 | **Paper trading**（虛擬 PnL，0 風險）| 0 | 1.0x | 100+ 筆穩定版本 trades + paper net > +5 bps × 4 週 |
| 2 | Testnet executor（exchange 測試環境）| 0 | 1.0x | testnet 1-2 週無 bug + order flow 正確 |
| 3 | Live tiny size（$100，輸光不痛）| 極小 | 1.0x | live 4 週 net positive + MDD < 20% |
| 4a | 放大到 $1k（3 個月）| 小 | 1.0x | Stage 3 通過 + 0 kill trigger |
| 4b | $1k（3 個月）| 小 | 1.2x | 4a 通過 + MDD < 10% |
| 4c | $5k（6 個月）| 中 | 1.5x | 4b 通過 + 連續 6 個月 hit no kill rules |
| 4d | $10k+（12 個月+）| 高 | **2.0x（絕對上限）** | 4c 通過 + 真實 Sharpe ≥ 1.5 |

每個階段都有 hard rules，寫入 production 程式碼，**不靠紀律**：
- drawdown trigger（cumulative drawdown 觸發 → 自動降階段）
- connection loss kill switch（與 exchange 失聯 → 取消所有未平倉位）
- position limit（單筆 / 總部位上限）
- daily loss cap（單日累積虧損上限 → 暫停當日所有訊號）
- **leverage cap**：當前 stage 的 leverage 上限寫進 config，超過則 executor 拒絕啟動

### Leverage ladder 數學依據（2026-05-25 加入）

2.0x 絕對上限不是拍腦袋，是基於當前 edge profile（μ=+5%，σ=30%）計算：
- Kelly optimal: f* = μ/σ² ≈ 0.56x（已小於 1x）
- Volatility drag: r_compound = E[r] - 0.5σ²L²
  - L=2.0: drag = -18%（仍可被 edge 覆蓋）
  - L=3.0: drag = -40.5%（drag > expected return，長期虧損）
  - L=5.0: drag = -112%（mathematical ruin，不論 edge）
- Stress Test 7 regime flip MDD scaling:
  - 1.0x → -15%（kill switch 救援）
  - 2.0x → -30%（painful 但可活）
  - 3.0x → -45%（半條命，加滑點接近 wipeout）
  - 5.0x → -75%（實質歸零）

**何時可考慮放寬 2.0x 上限**:
連續 24 個月實盤 Sharpe ≥ 3.0（目前 0.17-0.5）+ MDD 從未超過 -10%
+ 經過至少 2 個完整 regime flip 仍正 EV。在那之前，2.0x 是 hard cap。

## 當前策略：研究 + Small Live 並進（2026-05-27 決策）

**背景**：原 staged framework 要求 Stage 1 滿 100 trades + 4 週才進 Stage 2，按目前 9 天 6 筆的節奏需 5 個月。使用者選擇接受 informed risk：用 $100 live 作為「**operational stress test + edge 二次驗證**」，paper cohort 同時繼續累積。

**改動的規則**：
- Stage 1 → Stage 2 不再硬要求 100 trades + 4 週；改成「OKX skeleton TODO 全填完 + unit tests 過 + 3-5 天 testnet shakeout」
- Stage 2 → Stage 3 ($100 live) 不再硬要求 38 項 checklist 全跑完；改成「testnet 連續 3 天對帳 100% + 0 unhandled exception + manual approval 模式跑過 5 筆」

**保留的不可鬆綁**：
- **金額**：$100 live = Stage 3 上限，未進 Stage 4a 不准加碼（即使 $100 賺到 $200 也是 $100 keep + $100 不再用）
- **Hard kill switches 必須先驗證能觸發**（不是只寫進 code）：unit test + testnet 至少一次故意觸發
- **Manual approval 模式必須跑過 5+ 筆**，人工 review 沒問題才能切自動
- **Paper cohort 不停**：作為 edge 是否真實的並行驗證；live + paper 結果 > 2 週嚴重背離 → halt
- **Leverage 1.0x 不准動**（Stage 3 階段）
- **Stage 3 → Stage 4 仍照原硬條件**：live 4 週 net positive + MDD < 20% + 0 kill trigger

**做這個決策的代價自負**：
- 第一筆 live 訂單 = OKX REST/WS code 第一次真實執行 = 有 ops bug 的風險（mitigations 寫在上面）
- Edge 若是 fake，$100 是發現的成本（mistake.md 應記：用 $100 換 edge 真假驗證，比 5 個月等更便宜）
- 一旦 hit 任何 kill trigger，**回到 Stage 1 重新驗證**不是「凹下去」

## 10x leverage informed override (2026-05-28)

**背景**：BTC-USDT-SWAP 1 contract = 0.01 BTC ≈ $750 notional。$100 + 1x leverage 連 1 contract 都開不了 → Stage 3 完全卡死。使用者選擇接受 informed override：保 $100 capital、鬆 leverage cap 到 10x 讓 1 contract 開得起來。

**這條 override 違反兩條既有規則**：
- §Leverage ladder 數學依據：Kelly optimal 0.56x，1x 已超過 Kelly；10x 是 17.8x Kelly
- §仍然禁止的：「禁因為想賺更多就改 leverage cap——cap 來自數學不是情緒」

**為什麼接受**：
- 這次不是「想賺更多」，是「為了讓 testnet/live 能跑出第一筆有意義 trade」
- 沒 leverage 鬆，Stage 3 永遠走不到（連 1 contract 都開不了）
- Stage 3 的本意是 operational stress test + edge 二次驗證，不是 alpha 機器；$100 全輸的成本可接受

**為了補償 10x 的數學風險，kill switches 同步收緊**：
- daily_loss_cap_pct: -50% → **-20%**（10x 下 -20% account ≈ -2% BTC，stop-out 3 筆觸發）
- total_loss_cap_pct: -50% → **-30%**（career-end，Stage 3 結束）

**10x 下單筆風險**：
- 3xATR stop 在 10x = 約 -6% account = -$6 per stop-out
- 3 連虧 = -18% = 接近 daily cap → halt
- 5 連虧 = -30% = total cap → Stage 3 終結

**真會死的情境**（必須接受才能走這條）：
- BTC 一晚 -3% 跳空（leverage 算下 -30% account 直接掃 total cap）
- 黑天鵝 -10% 級別 → 算下 -100% 直接歸零，連 cap 都來不及救
- 這些情境發生過（2024-08, 2025-01），未來會再發生

## Staged 進階條件對照表（更新版 2026-05-28）

| Stage | 描述 | Risk | Leverage | Daily/Total cap | 進階條件 |
|---|---|---|---|---|---|
| 1 | Paper trading | 0 | 1.0x | n/a | 已 active；live 啟動後**不停**作 baseline |
| 2 | Testnet shakeout | 0 | 10x (demo) | -20% / -30% | OKX TODO 填完 + unit tests 過 + 3-5 天對帳 100% + 0 unhandled exception |
| 3 | **Live $100**（當前目標）| -$100 上限 | **10x** | -20% / -30% | testnet 通過 + manual approval 跑 5 筆 |
| 4a | $1k（3 個月）| 小 | 1.0x | -20% / -30% | Stage 3 跑 4 週 + net positive + MDD < 20% + 0 kill trigger |
| 4b | $1k 1.2x | 小 | 1.2x | -15% / -25% | 4a 通過 + MDD < 10% |
| 4c | $5k | 中 | 1.5x | -15% / -25% | 4b 通過 + 連續 6 個月 hit no kill rules |
| 4d | $10k+ | 高 | **2.0x（絕對上限）** | -10% / -20% | 4c 通過 + 真實 Sharpe ≥ 1.5 |

**注意 Stage 3 → 4a 的 leverage 反而從 10x 降回 1x**：Stage 4a 起金額放大到 $1k，1 contract 不再是門檻，回到 Kelly-respecting 1x 是正解。Stage 3 的 10x 是「為了開門」的權宜，不是策略的一部分。

## 仍然禁止的（避免在錯的階段做錯事）
- **Stage 2-3**：禁鬆 hard kill switches 以外的 trigger；leverage hard cap = 10x（不可再放寬）
- **Stage 3**：禁未經 manual approval 5 筆就切自動；禁 paper cohort 停寫
- **Stage 3 → 4a**：leverage 必須降回 1.0x；不能因為 $100 賺到 $200 就用 10x 加碼
- **Stage 4a-d**：leverage 階梯式放寬，**絕對上限 2.0x**；未 hit 各子階段條件不得進下一格
- **Stage 4 後**：禁 leverage > 2.0x，除非滿足「24 個月實盤 Sharpe ≥ 3.0」（見 §Leverage ladder 數學依據）
- 任何階段：strategy sweep 必須留 OOS hold-out，禁全資料 fit
- 任何階段：禁因為「最近表現好」就跳階段——必須 hit hard rules
- 任何階段：禁再鬆 leverage cap——10x 已經是「informed 一次」的極限；下次再要鬆要寫進 mistake.md
- 任何階段：hit kill trigger 必須降階重驗，不准「我覺得這次例外」

## 系統架構（v7 Dual-Model）
Dual XGBoost 架構：Direction Regressor + Magnitude Regressor，獨立管線。

### 數據層
- **Binance REST API** (3 endpoints)：klines (1h, 500 bars)、depth (L20)、aggTrades
- **Coinglass API v4** (24 endpoints)：15 timeseries + 9 snapshot
- **Deribit Public API** (2 endpoints)：DVOL 波動率指數、Options Summary

### 特徵工程
- **200+ 工程特徵**（Direction 136, Magnitude 72），12 個群組
- 所有計算為 trailing-only（無前視偏差）
- Coinglass 原生 1h 使用 merge_asof 精確對齊
- 自訂 alpha 特徵：impact_asymmetry (IC=-0.071)、post_absorb_breakout (mag IC=0.191)

### 模型
- **Direction Model**：XGBRegressor, 136 特徵, 輸出 pred_return_4h (TWAP path return)，rolling percentile 解碼為 UP/DOWN/NEUTRAL
- **Magnitude Model**：XGBRegressor, 72 特徵, 輸出 |return_4h|
- **Regime Detection**：CHOPPY / TRENDING_BULL / TRENDING_BEAR / WARMUP

### 信號生成
- Direction: 500-bar rolling percentile 解碼，top 5% → Strong UP，top 15% → Moderate UP（DOWN 同理）
- Absolute |pred| floor (Strong=0.0008, Moderate=0.0005)：低 vol regime 保險，rolling cutoff 比 floor 寬鬆時 floor 接管（2026-05-09 加入）
- Confidence = `min(|pred|/Strong_cutoff, 1.0)^0.6 × 100`（純 |pred| 公式，2026-05-09 移除 mag bonus 因為 OOS 顯示高 mag bar 在模型失靈區）
- Strong ≥ 80, Moderate ≥ 65, Weak < 65（顯示用，實際 tier 觸發看 |pred| vs cutoff）
- Hysteresis + Cooldown

### 輸出
- 圖表面板 (Confidence / Regime / K線+三角形 / Magnitude)
- Telegram 推送 (Strong 信號文字告警 + SHAP 驅動因子)
- REST API (10 routes)
- MySQL + Parquet 持久化

### 績效追蹤
- Rolling IC (7d/30d) + IC 趨勢 + 衰退警報
- Strong 信號追蹤 (4h 後自動回填結果)
- SHAP 驅動因子分析 (Strong 信號時觸發)
- Regime 拆解準確率
- 全部整合在 /perf 指令

## 模型輸出（固定格式）
- **pred_return_4h**: sign(direction) × magnitude
- **pred_direction**: UP / DOWN / NEUTRAL
- **strength_score**: Strong / Moderate / Weak
- **confidence_score**: 0~100
- **mag_pred**: |return_4h| 預測值
- **dir_prob_up**: P(UP) 原始值
- **regime**: 當前市場狀態

### 核心 target
y_path_ret_4h = mean(close[t+1..t+4]) / close[t] - 1 (TWAP path return)

### 評估指標
- Spearman IC / ICIR（預測值與實際收益的排序相關）
- 方向準確率
- Calibration monotonicity（預測越強，實際收益越高）
- Strong 信號勝率（目標 point estimate ≥ 65%，stretch 70%；天花板由 AUC ~0.57 結構決定，top-5% precision 實測 67.6%）
- Magnitude Top/Bot ratio

## 技術 Stack
- Python 3.11
- 資料處理：Pandas + NumPy + SciPy
- 資料庫：MySQL 8.0 (Railway 託管)
- 儲存：Parquet（歷史備份）、.data_cache/（API 回退快取）
- 模型：XGBoost (Dual Regressor)
- Web：Flask + APScheduler
- 圖表：Matplotlib (靜態) + TradingView Lightweight Charts (互動)
- 推送：Telegram Bot API
- 部署：Railway (git push 自動部署)
- 解釋性：SHAP (TreeExplainer, Strong 信號時觸發)

## 核心原則（永遠不能違反）
1. **無前視偏差**：所有特徵計算使用 trailing-only rolling，嚴格禁止 look-ahead。
2. **歷史與即時一致性**：`build_live_features()` 同時用於訓練數據建構和生產推論。
3. **時間對齊精準**：Coinglass 使用 merge_asof backward 對齊，快照數據只設定最後一根 bar。
4. **預測導向設計**：所有評估以 IC、方向準確率、calibration 為準，不做交易績效回測。
5. **特徵先回測再加入**：新特徵必須先跑 IC 回測驗證有效才加進系統。
6. **Edge Cases 處理**：假日流動性差異、Funding 結算跳動、rate limit、資料缺失。

## 圖表同步規則
系統有兩個圖表，修改時**必須同步更新**：
1. **靜態圖表** (`indicator/chart_renderer.py`) — Telegram 推送的 PNG
2. **互動圖表** (`indicator/chart_interactive.py`) — `/ichart` 的 TradingView Lightweight Charts HTML

任何圖表邏輯變更（面板、三角形、顏色、過濾條件）都要兩邊一起改。

## 命名與程式碼規範
- Class：CamelCase（如 IndicatorEngine、SignalExplainer）
- 函數/變數：snake_case（如 build_live_features、backfill_mag_pred）
- 偏好：清晰、可讀性高、模組化
- 新特徵加入前必須回測驗證 IC

## 專案階段
- 階段 4：特徵工程（目前重點 — 自訂 alpha 特徵 + 數據累積）
- 階段 5：模型開發（Magnitude 已重訓 v2, Direction 等 2 週後重訓）
- 階段 9：持續迭代（績效追蹤、IC 監控、衰退警報已上線）
