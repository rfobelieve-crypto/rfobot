# BTC 多空強度預測指標 — 系統完整架構白話說明

> 最後更新：2026-04-25
> 用最白話的方式解釋整套系統怎麼運作，每個檔案在幹嘛，資料怎麼流動。

---

## 一句話講完這套系統在幹嘛

每小時自動去各交易所抓最新 BTC 數據，算出 200+ 個特徵，丟進三個 XGBoost 模型，
預測「未來 4 小時會漲還跌、幅度多大、有多少信心」，畫成圖表推 Telegram，
並自動追蹤預測準不準。

**這是預測指標，不是交易系統** — 不下單、不管倉位、不算盈虧。

---

## 系統分兩個獨立服務（共用一個資料庫）

```
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│   Service 1: 指標服務            │   │   Service 2: 行情收集            │
│   (Railway)                     │   │   (Railway)                     │
│                                 │   │                                 │
│   每小時：                      │   │   24/7 不停：                   │
│   抓數據 → 算特徵 → 跑模型      │   │   WS 接 Binance/OKX 每筆成交    │
│   → 畫圖 → 推 Telegram          │   │   → 聚合成 1 分鐘 bar            │
│                                 │   │   → 寫入 DB                      │
│   入口: indicator/wsgi.py       │   │   入口: market_data/tasks/      │
│        → indicator/app.py       │   │        start_all.py             │
└──────────────┬──────────────────┘   └──────────────┬──────────────────┘
               │                                     │
               └──────────────┬──────────────────────┘
                              ▼
                    ┌───────────────────┐
                    │  Railway MySQL    │
                    │  (兩邊共用)       │
                    └───────────────────┘
```

**兩個服務程式碼完全獨立，不會互相 import，只透過 DB 交換數據。**

---

## Service 1 (指標服務) 完整資料流

這是核心。從拿到原始數據到產生最終預測，走過 5 個階段：

```
【階段 1】原始數據
┌────────────────────────────────────────────────┐
│ Binance REST API                               │
│   ├─ klines (500 根 1h K 線)                   │
│   ├─ depth L20 (訂單簿深度)                    │
│   └─ aggTrades (近期聚合成交)                  │
│                                                │
│ Coinglass API v4  (14 個時間序列 + 9 個快照)   │
│   ├─ OI 未平倉量 (期貨/現貨/合約)              │
│   ├─ funding rate (資金費率)                   │
│   ├─ taker buy/sell volume (主動買賣量)        │
│   ├─ long/short ratio (多空比)                 │
│   ├─ liquidation (爆倉)                        │
│   ├─ coinbase premium (Coinbase 溢價)         │
│   └─ bitfinex margin (Bitfinex 融資)          │
│                                                │
│ Deribit API                                    │
│   └─ DVOL (選擇權隱含波動率指數)               │
└────────────────────────────────────────────────┘
                        │
                        ▼ indicator/data_fetcher.py
                        │  (任一端點掛掉用快取頂替，不會整個死)
                        ▼
【階段 2】特徵工程
┌────────────────────────────────────────────────┐
│ indicator/feature_builder_live.py              │
│                                                │
│ 把上面原始數據算成 200+ 個工程特徵，12 組：    │
│                                                │
│  1. 價格動量    (return lags, realized vol)    │
│  2. 訂單簿     (bid/ask 深度比, spread)        │
│  3. 交易流     (大單比例, aggTrade delta)      │
│  4. OI 變化    (變化率, 1h/4h/8h lag)          │
│  5. Funding    (rate, zscore, 偏離均值)        │
│  6. Taker 流   (buy/sell 差, 加速度)           │
│  7. 多空比     (全域/大戶/持倉)                │
│  8. 爆倉      (強制平倉量, 多空方向)           │
│  9. 跨市場    (Coinbase premium, BFX margin)   │
│ 10. DVOL      (水位, 變化率)                   │
│ 11. Regime    (BULL/BEAR/CHOPPY 指示)          │
│ 12. 自訂 alpha (impact_asymmetry,              │
│                post_absorb_breakout...)        │
│                                                │
│ ⚠️ 鐵律：所有計算 trailing-only，禁止看未來    │
│ ⚠️ 同一份程式碼同時用在訓練和生產（一致性）    │
└────────────────────────────────────────────────┘
                        │
                        ▼
【階段 3】模型推論
┌────────────────────────────────────────────────┐
│ indicator/inference.py                         │
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │ Direction Model (方向預測)                │  │
│  │ XGBRegressor, 136 特徵                   │  │
│  │ 目標: y_path_ret_4h (TWAP 4h 收益率)     │  │
│  │                                          │  │
│  │ 輸出: pred_return_4h (連續值)            │  │
│  │   ↓                                      │  │
│  │ Rolling Percentile 解碼 (500 bar 窗口):  │  │
│  │   top 2.5% → Strong UP                   │  │
│  │   top 7.5% → Moderate UP                 │  │
│  │   中間 85% → NEUTRAL                     │  │
│  │   bot 7.5% → Moderate DOWN               │  │
│  │   bot 2.5% → Strong DOWN                 │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │ Magnitude Model (幅度預測)                │  │
│  │ XGBRegressor, 72 特徵                    │  │
│  │ 目標: |return_4h|                        │  │
│  │                                          │  │
│  │ 輸出: mag_pred (預測絕對幅度)            │  │
│  │  → 用於 confidence bonus (+20 分)        │  │
│  │  → 用於 Initiation gate (mag_pct ≥ 0.8)  │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │ Initiation Model (突破啟動確認)           │  │
│  │ XGB binary (long + short 各一個),        │  │
│  │ 135 特徵                                 │  │
│  │ V5 breakout gate:                        │  │
│  │   open ≤ prior_high AND close > prior_high│ │
│  │  → 過濾 gap 突破，只留實體突破            │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  Regime Detection:                             │
│    CHOPPY / TRENDING_BULL / TRENDING_BEAR      │
│    / WARMUP                                    │
└────────────────────────────────────────────────┘
                        │
                        ▼
【階段 4】最終輸出（每根 bar）
┌────────────────────────────────────────────────┐
│   pred_direction:    UP / DOWN / NEUTRAL       │
│   strength_score:    Strong / Moderate / Weak  │
│   confidence_score:  0~100                     │
│     ├─ 80 分: |pred_ret|/cutoff^0.6            │
│     └─ 20 分: mag percentile bonus             │
│   pred_return_4h:    預測 4h 收益率             │
│   mag_pred:          預測 4h 絕對幅度           │
│   regime:            當前市場狀態               │
│   bull_bear_power:   多空力量 [-1, 1]           │
└────────────────────────────────────────────────┘
                        │
                        ▼
【階段 5】分發（5 個去處）
         ┌──────────┬──────────┬──────────┬──────────┬──────────┐
         ▼          ▼          ▼          ▼          ▼          ▼
   chart_renderer chart_inter  signal_  signal_ex  snapshot_  Telegram
   (PNG)         active       tracker  plainer    collector  推送
                 (HTML)       (記錄     (SHAP     (MySQL +  (Strong
                              + 回填)   解釋)     Parquet)   信號)
```

---

## 每個檔案在幹嘛（依資料流順序）

### 🔌 共用層：資料庫連線

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `shared/db.py` | 269 | 全系統唯一的 MySQL 連線入口。內建連線池（5 條），所有程式用 `get_db_conn()` 拿連線。 |

---

### 📊 Service 1：指標服務

#### 頂層入口

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `indicator/wsgi.py` | 4 | gunicorn 啟動點，一行 import app 就結束 |
| `indicator/app.py` | ~1530 | **全系統最大的檔案**。Flask 路由 + APScheduler + Telegram webhook + update_cycle 的主流程都在這 |

#### 數據抓取（階段 1）

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `indicator/data_fetcher.py` | 732 | 每小時去 Binance/Coinglass/Deribit 抓最新數據。有 retry + cache fallback，一個端點掛掉不會死全部 |
| `indicator/feature_config.py` | 148 | 純設定檔：定義要抓哪些 Coinglass 端點、rolling window 多大 |

#### 特徵工程（階段 2）

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `indicator/feature_builder_live.py` | 864 | **核心檔案**。把原始數據算成 200+ 特徵。訓練與生產共用這份程式碼確保一致 |
| `indicator/initiation_features.py` | 363 | 突破確認用的額外特徵（V5 breakout gate 相關） |

#### 模型推論（階段 3）

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `indicator/inference.py` | 480 | 跑三個 XGBoost 模型（Direction + Magnitude + Initiation），輸出方向/強度/信心 |

#### 模型檔案位置

```
indicator/model_artifacts/dual_model/
├── direction_xgb.json              ← Direction 權重（2026-04-15 重訓）
├── direction_feature_cols.json     ← 136 個特徵名
├── direction_reg_config.json       ← rolling percentile 解碼參數
├── direction_importance.csv        ← SHAP 重要性排名
│
├── magnitude_xgb.json              ← Magnitude 權重（2026-04-16）
├── magnitude_feature_cols.json     ← 72 個特徵名
├── magnitude_config.json           ← OOS IC 0.34, ICIR 1.90
│
├── initiation_long_xgb.json        ← 多方突破模型
├── initiation_short_xgb.json       ← 空方突破模型
├── initiation_feature_cols.json    ← 135 個特徵名
├── initiation_thresholds.json      ← Strong/Moderate 啟動門檻
│
├── training_stats.json             ← warmup buffer（rolling percentile 用）
└── history/                        ← 歷次重訓快照（含 meta + importance）
```

#### 圖表輸出（階段 5）

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `indicator/chart_renderer.py` | 277 | 畫 **靜態 PNG**（Telegram 推送用） |
| `indicator/chart_interactive.py` | 359 | 畫 **互動 HTML**（`/ichart` 用 TradingView Lightweight Charts） |

**兩個圖表四個面板：**
1. Confidence 熱力圖（紅綠色帶）
2. K 線 + 三角形（大=Strong、小=Moderate）
3. Magnitude 柱狀圖（預測幅度）
4. BBP 多空力量 [-1, 1]

**⚠️ 鐵律：改一個必須改另一個，兩邊邏輯必須同步。**

#### 信號追蹤與解釋

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `indicator/signal_tracker.py` | 275 | 每次 Strong/Moderate 訊號都記到 `tracked_signals` 表。4h 後自動回填結果（勝/敗） |
| `indicator/signal_explainer.py` | 351 | Strong 訊號時跑 SHAP，找出前 5 個驅動特徵，附在 Telegram 推送裡 |
| `indicator/alpha_decay_monitor.py` | 447 | 5 項衰退預警（IC 趨勢、特徵漂移、信號翻覆、高信心低勝率、Strong 變少） |
| `indicator/monitor_icir.py` | 343 | 滾動 7/30 天 IC 與勝率監控，跌破門檻發 Telegram 警報 |

#### 健康監控 + 快照保存

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `indicator/health_monitor.py` | 294 | 每次更新檢查：數據新鮮度、NaN 比例、warmup 狀態、DB 連線 |
| `indicator/snapshot_collector.py` | 435 | 每次預測完存快照到 MySQL + Parquet（未來重訓用） |

#### Web 儀表板

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `indicator/dashboard.py` | 300 | 儀表板主頁（5 個分頁） |
| `indicator/dashboard_tabs/overview.py` | 281 | 總覽分頁 |
| `indicator/dashboard_tabs/performance.py` | 521 | 績效分頁（IC 趨勢、勝率、equity curve） |
| `indicator/dashboard_tabs/health.py` | 432 | 系統健康分頁 |
| `indicator/dashboard_tabs/market.py` | 522 | 市場情報分頁（特徵值、z-score） |
| `indicator/dashboard_tabs/analytics.py` | - | 績效分析（Strong/Moderate 勝率、rolling IC） |

#### 訓練與匯出（本地跑）

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `indicator/train_export.py` | 165 | 本地用歷史 Parquet 訓練模型 + 匯出到 `model_artifacts/`。git push 後 Railway 自動換模型 |

---

### 🌊 Service 2：行情收集

#### 啟動入口

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `market_data/tasks/start_all.py` | 76 | Service 2 啟動按鈕：DB migration + WS 連線 + 10 秒刷 bar + 每小時清舊 |

#### Core：標準化層

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `market_data/core/symbol_mapper.py` | 49 | Binance `BTCUSDT` / OKX `BTC-USDT-SWAP` 統一成 `BTC-USD`；記合約乘數 |
| `market_data/core/trade_normalizer.py` | 55 | 不同交易所格式統一成「誰買的、多少錢、什麼時間」 |
| `market_data/core/flow_aggregator.py` | 126 | 零散交易聚合成 1 分鐘桶（買多少、賣多少、淨流入） |
| `market_data/core/health_monitor.py` | 91 | 30 秒無數據 → 警告；90 秒 → 斷線 |

#### Adapters：數據來源

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `market_data/adapters/binance_trades.py` | 124 | Binance WS 接交易流 |
| `market_data/adapters/okx_trades.py` | 122 | OKX WS 接交易流 |
| `market_data/adapters/funding_collector.py` | 115 | 每 60 秒抓 funding rate |
| `market_data/adapters/oi_collector.py` | 175 | 每 60 秒抓 OI |
| `market_data/adapters/liquidation_collector.py` | 273 | WS 接爆倉事件 |

#### Storage：寫入資料庫

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `market_data/storage/trade_repository.py` | 55 | 批次寫交易（100 筆 / 5 秒） |
| `market_data/storage/flow_repository.py` | 66 | UPSERT 1 分鐘 bar，防止重複 |

#### Tasks：排程

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `market_data/tasks/run_trade_streams.py` | 139 | 啟動 WS + normalize + aggregate 管線 |
| `market_data/tasks/flush_flow_bars.py` | 63 | 每 10 秒把完成的 bar 刷到 DB |
| `market_data/tasks/cleanup.py` | 87 | 自動清理舊數據（交易 3 天、bar 90 天） |

#### Query + Features：給 Service 1 讀

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `market_data/query/flow_context.py` | 149 | 查 flow_bars_1m 的買賣統計（過去 15m/1h 的壓力） |
| `market_data/query/snapshot_query.py` | 167 | 查流動性事件的歷史和分數 |
| `market_data/features/snapshot_builder.py` | 398 | 流動性事件特徵計算（15m/1h/4h 窗口） |
| `market_data/features/snapshot_runner.py` | 156 | 定期掃待處理事件 → 算特徵 → 存 DB → 推 Telegram |

#### Backfill：歷史回填

| 檔案 | 行數 | 白話說明 |
|------|------|---------|
| `market_data/backfill/run_all.py` | 83 | 一鍵回填入口 |
| `market_data/backfill/coinglass_backfill.py` | 295 | 從 Coinglass 下載歷史 OI/爆倉/多空比 |
| `market_data/backfill/download_raw.py` | 304 | 從 Binance 下載歷史 klines / aggTrades |
| `market_data/backfill/funding_backfill.py` | 200 | Funding rate 回填 |
| `market_data/backfill/oi_backfill.py` | 141 | Binance 15m OI 回填 |

---

## Update Cycle 完整時序（每小時 :02 觸發）

```
╔═══════════════════════════════════════════════════════════════╗
║  indicator/app.py :: update_cycle()                           ║
║  觸發：APScheduler 每小時 :02                                  ║
╚═══════════════════════════════════════════════════════════════╝

  T+0s   抓數據  (data_fetcher.py)
         ├─ Binance klines / depth / aggTrades
         ├─ Coinglass × 14 端點
         └─ Deribit DVOL
          ⇩
  T+2s   算特徵  (feature_builder_live.py)
         └─ 200+ 特徵，全 trailing-only
          ⇩
  T+3s   跑模型  (inference.py)
         ├─ Direction Regressor → pred_return_4h
         ├─ Rolling percentile 解碼 → direction + strength
         ├─ Magnitude Regressor → mag_pred
         └─ Initiation (long/short) → 突破確認
          ⇩
  T+4s   畫圖  (chart_renderer + chart_interactive)
         ├─ PNG (靜態)
         └─ HTML (互動)
          ⇩
  T+5s   Strong/Moderate 訊號？
         ├─ YES → signal_tracker 記錄
         │        signal_explainer 跑 SHAP
         │        Telegram 推送 (含 SHAP 驅動因子)
         └─ NO  → 跳過
          ⇩
  T+6s   回填舊訊號結果  (signal_tracker.backfill)
         └─ 4h 前的訊號查實際收益、填 correct 欄位
          ⇩
  T+7s   保存快照  (snapshot_collector)
         ├─ MySQL (indicator_history 表)
         └─ Parquet (歷史備份)
          ⇩
  T+8s   持久化 warmup buffer
         └─ dir_pred_history 寫回 training_stats.json
             (Railway 重啟後不會失去 warmup 進度)
          ⇩
  T+8s   完成，等下個整點
```

---

## 支援排程

| 時間 | 任務 | 檔案 |
|------|------|------|
| 每小時 :02 | update_cycle（主流程） | `app.py` |
| 每小時 :15 | watchdog 快速健康檢查 | `agents/watchdog.py` |
| 每 4 小時 :20 | 完整健康檢查 | `health_monitor.py` |

---

## Telegram 指令

| 指令 | 功能 |
|------|------|
| `/chart` | 最新 PNG 圖表 |
| `/ichart` | 互動圖表連結（TradingView Lightweight Charts HTML） |
| `/perf` | Strong/Moderate 勝率報告（含 dual-gate 分解） |
| `/ic` | 滾動 7/30d IC + 信號準確率 |
| `/decay` | 5 項 alpha 衰退檢查 |
| `/health` | 系統健康狀態 |
| `/force` | 手動觸發一次 update_cycle |
| `/indicator-status` | 最後一次 update 狀態（含 error 訊息） |

---

## HTTP 路由

| 路由 | 功能 |
|------|------|
| `GET /` | 最新 PNG 圖表 |
| `GET /json` | 最新預測 JSON |
| `GET /health` | 健康檢查 JSON |
| `GET /dashboard` | Web 儀表板（5 分頁） |
| `GET /ichart` | 互動圖表 HTML |
| `POST /webhook` | Telegram webhook |
| `GET /force-update?sync=1` | 同步觸發 update_cycle（除錯用） |

---

## 資料庫主要資料表

| 表 | 用途 | 寫入者 | 讀取者 |
|----|------|--------|--------|
| `normalized_trades` | 逐筆成交 | Service 2 | 事件特徵計算 |
| `flow_bars_1m` | 1 分鐘流量桶 | Service 2 | Service 1 特徵 |
| `funding_rates` | Funding 歷史 | Service 2 | 特徵 |
| `oi_snapshots` | OI 歷史 | Service 2 | 特徵 |
| `liquidation_1m` | 爆倉 1m 聚合 | Service 2 | 特徵 |
| `indicator_history` | 每根 bar 的預測結果 | Service 1 | 儀表板 + IC 監控 |
| `tracked_signals` | Strong/Moderate 訊號追蹤 | Service 1 | `/perf` 勝率報告 |
| `liquidity_events` | 流動性獵取事件 | BTC_perp_data | 事件分析 |
| `sweep_outcomes` | 事件多窗口結果 | outcome_tracker | 事件分析 |

---

## 部署

| 設定 | 值 |
|------|-----|
| Service 1 Dockerfile | `Dockerfile.indicator` |
| Service 2 Dockerfile | `Dockerfile.marketdata` |
| 部署觸發 | `git push main` → Railway 自動 build + deploy |
| DB | Railway MySQL (`mysql.railway.internal` 內網 / caboose.proxy.rlwy.net 外網) |
| 本地開發 | `.env` 指向 Railway 外部 MySQL |

---

## 目前模型狀態（2026-04-25 快照）

| 模型 | 重訓日期 | 特徵數 | OOS 指標 |
|------|---------|--------|----------|
| Direction Regressor | 2026-04-15 | 136 | Spearman IC 0.183, AUC_sign 0.597, top-5% precision 67.4% |
| Magnitude | 2026-04-16 | 72 | IC 0.340, ICIR 1.90, monotonicity 1.0 |
| Initiation Long/Short | 2026-04-19 | 135 | Replay Strong WR 68% (n=97), Moderate 44% (n=175) |

**結構天花板：** Direction AUC ~0.57 → top-5% precision 理論上限 68-72%，目前 67.4% 已貼天花板。
**目標：** Strong 勝率 point estimate ≥ 65%，stretch 70%（從 AUC 反推的合理目標，非 95%）。

---

## 幾條絕對不能違反的規則

1. **無前視偏差** — 所有特徵計算必須 trailing-only，禁止 look-ahead
2. **歷史與即時一致性** — `build_live_features()` 同時用於訓練和生產
3. **時間對齊精準** — Coinglass 用 `merge_asof backward`，快照數據只設最後一根 bar
4. **特徵先回測再加入** — 新特徵必須先跑 IC 驗證
5. **兩個圖表必須同步** — `chart_renderer.py` 與 `chart_interactive.py` 改一邊必須改另一邊
6. **預測導向設計** — 以 IC / 方向準確率 / calibration 為準，不做交易績效回測
7. **共用 JSON 必須 read-then-merge** — `training_stats.json` 有多個寫入者，直接覆寫會洗掉 warmup buffer
8. **Hot-path 改動 push 前跑一次真實 update_cycle** — import OK 不代表 runtime OK

---

## 架構快覽圖（一張圖總結）

```
┌──────────────────────────────────────────────────────────────────┐
│                   Service 2 (market_data)                         │
│                                                                   │
│  Binance WS ──┐                                                   │
│               ├── normalize ── aggregate ── 1m bar ── MySQL       │
│  OKX WS ─────┘                                                    │
│                                                                   │
│  REST (60s) ── funding / OI / liquidation ────────── MySQL        │
└─────────────────────────┬────────────────────────────────────────┘
                          │ (共用 DB)
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Service 1 (indicator)                           │
│                                                                   │
│  每小時 :02 ──► update_cycle:                                     │
│                                                                   │
│    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│    │ data_fetcher │─►│feature_build │─►│  inference   │           │
│    │  (3 API)     │  │   _live      │  │ (3 XGBoost) │           │
│    └──────────────┘  └──────────────┘  └──────┬───────┘           │
│                                               │                   │
│         ┌─────────────────────────────────────┼───────────┐       │
│         ▼                ▼                    ▼           ▼       │
│    chart_renderer  chart_interactive  signal_tracker  snapshot    │
│    (Telegram PNG)  (/ichart HTML)     (+ SHAP)        _collector  │
│                                                       (MySQL +    │
│                                                        Parquet)   │
│                                                                   │
│  Flask + Telegram Webhook + Dashboard                             │
│  指令: /chart /ichart /perf /ic /decay /health /force              │
└──────────────────────────────────────────────────────────────────┘
```
