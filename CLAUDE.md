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

| Stage | 描述 | Risk | 進階條件 |
|---|---|---|---|
| 0 | 純指標 + 推送 | 0 | (已過) |
| 1 | **Paper trading**（虛擬 PnL，0 風險）| 0 | 100+ 筆穩定版本 trades + paper net > +5 bps × 4 週 |
| 2 | Testnet executor（exchange 測試環境）| 0 | testnet 1-2 週無 bug + order flow 正確 |
| 3 | Live tiny size（$100，輸光不痛）| 極小 | live 4 週 net positive + MDD < 20% |
| 4 | 漸進加碼（每月 +50%）| 隨加碼 | 每階段 hit hard rules 不退出 |

每個階段都有 hard rules，寫入 production 程式碼，**不靠紀律**：
- drawdown trigger（cumulative drawdown 觸發 → 自動降階段）
- connection loss kill switch（與 exchange 失聯 → 取消所有未平倉位）
- position limit（單筆 / 總部位上限）
- daily loss cap（單日累積虧損上限 → 暫停當日所有訊號）

## 當前 stage：Stage 1 (Paper Trading)
- 入口：`/paper-perf` endpoint（HTML report）+ `indicator/paper_trading.py` (computation)
- 觀察項：每週看「最近 30 天」cohort 的 net_bps、WR、Sharpe、Drawdown
- **本 stage 仍可改進指標模型**（feature、訓練、信心公式等），改完即新 cohort 開始
- 進階前必須等：穩定版本下 100+ 筆 trades 且 4 週連續正 EV

## 仍然禁止的（避免在錯的階段做錯事）
- **Stage 1**：禁寫 exchange execution code（連券商 API）；只計算虛擬 PnL
- **Stage 2-3**：禁鬆 hard rules；禁未經 rules 驗證就加碼到下一階段
- **Stage 4**：禁 leverage > 1（先驗證 cash spot 模式）
- 任何階段：strategy sweep 必須留 OOS hold-out，禁全資料 fit
- 任何階段：禁因為「最近表現好」就跳階段——必須 hit hard rules

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
