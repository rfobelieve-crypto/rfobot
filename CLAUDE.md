# 專案 CLAUDE.md - BTC 多空強度預測指標 (Market Intelligence Indicator)

## 專案定位（所有設計以此為準）
這是一個「多空強度預測指標 / Market Intelligence Indicator」，**不是**交易策略或自動下單系統。

核心功能：預測未來固定 horizon（4h）的市場方向、強度、信心，以圖表方式可視化。

**這不是策略系統 — 嚴禁延伸到以下方向：**
- entry/exit 規則、TP/SL 設計、倉位管理、槓桿控制
- 交易績效最大化、strategy backtest framework
- 自動下單 execution logic、fee simulation

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
- Confidence = 80 pts from |pred_ret|/Strong_cutoff^0.6 + 20 pts mag percentile bonus
- Strong ≥ 80, Moderate ≥ 65, Weak < 65
- BBP 確認閘門 + Regime 動態死區 + Hysteresis + Cooldown

### 輸出
- 4 面板圖表 (Confidence / K線+三角形 / Magnitude / BBP)
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
- **bull_bear_power**: [-1, 1] 合成指標
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
