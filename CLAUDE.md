# 專案 CLAUDE.md - BTC 多空強度預測指標 (Market Intelligence Indicator)

## 專案定位（所有設計以此為準）
這是一個「多空強度預測指標 / Market Intelligence Indicator」，**不是**交易策略或自動下單系統。

核心功能：預測未來固定 horizon（4h）的市場方向、強度、信心，以圖表方式可視化。

**這不是策略系統 — 嚴禁延伸到以下方向：**
- entry/exit 規則、TP/SL 設計、倉位管理、槓桿控制
- 交易績效最大化、strategy backtest framework
- 自動下單 execution logic、fee simulation

## 專案概覽
你是資深量化開發者（Quant + Python Engineer），專精加密永續合約訂單流分析。
目標：在 BTCUSDT Perpetual 市場，使用 aggTrade、OI、Funding Rate、Coinglass 多源數據，
在每個 15 分鐘 bar 輸出多空強度預測指標。

### 模型輸出（固定格式）
- **pred_return_4h**: 預測未來 4h 收益率
- **pred_direction**: UP / DOWN / NEUTRAL
- **strength_score**: Strong / Moderate / Weak
- **confidence_score**: 0~100
- **regime**: 當前市場狀態

### 核心 target
y_return_4h = close.shift(-16) / close - 1

### 最終產品形態
圖表指標（非交易信號）：
- K 線圖上的向上/向下三角形
- 顏色深淺表示 confidence
- 下方 bull/bear power histogram
- 文字說明：「未來 4h 預測偏多，預估漲幅 X%，信心 Y 分」

### 評估指標
- Spearman IC / ICIR（預測值與實際收益的排序相關）
- 方向準確率 > 58%
- Calibration monotonicity（預測越強，實際收益越高）
- 信心分布合理性（不應全部集中在 Weak）

## 技術 Stack（嚴格遵守優先順序）
- Python 3.11
- 資料處理：Polars（優先） > Pandas
- 資料庫：ClickHouse（高頻 insert 與聚合查詢）
- 儲存：Parquet + S3（原始/半處理資料）
- 模型：XGBoost / LightGBM（快速迭代） + PyTorch / Temporal Fusion Transformer（序列模型）
- 其他：Docker、Grafana、ONNX（生產推論）、Redis/Kafka（即時快取）

## 核心原則（永遠不能違反）
1. **歷史與即時一致性**：所有 Processor / Pipeline 必須使用同一套邏輯，嚴格避免 look-ahead bias。
2. **Stateful 設計**：類別必須維護狀態（current_cvd、last_oi、last_funding 等），同時支援 batch（歷史）與 incremental single-tick（即時）模式。
3. **時間對齊精準**：aggTrade 使用 tick 級，Funding 使用 asof merge 或 forward-fill 對齊。
4. **可即時計算**：所有特徵（CVD、VPIN、BVC、ΔOI、Funding × Signed Volume 等）必須支援增量更新。
5. **Edge Cases 處理**：資料缺失、Funding 結算跳動、交易所維護、rate limit、高波動時段、週末流動性差異。
6. **預測導向設計**：所有評估以預測品質（IC、方向準確率、calibration）為準，不做交易績效回測。

## 命名與程式碼規範
- Class：CamelCase（如 OrderFlowProcessor、FeatureGenerator）
- 函數/變數：snake_case（如 process_historical_batch、process_single_tick）
- 必須包含：Type hints、詳細 docstring、logging、單元測試或 replay 驗證案例
- 偏好：清晰、可讀性高、模組化

## 專案階段（參考使用）
階段 1：需求定義
階段 2：資料管線（目前重點）
階段 3：EDA + Regime Detection
階段 4：特徵工程
階段 5：模型開發
階段 6：嚴格回測
階段 7：風險管理
階段 8：部署
階段 9：持續迭代

每次任務時，先確認目前階段，並遵守以上所有原則。

## 圖表同步規則
系統有兩個圖表，修改時**必須同步更新**：
1. **靜態圖表** (`indicator/chart_renderer.py`) — Telegram 推送的 PNG
2. **互動圖表** (`indicator/chart_interactive.py`) — `/ichart` 的 TradingView Lightweight Charts HTML

任何圖表邏輯變更（面板、三角形、顏色、過濾條件）都要兩邊一起改。