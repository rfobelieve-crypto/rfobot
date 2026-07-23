# V7 多幣化 Step 2 結果 — ETH clean AUC（2026-07-23）

Run: `research/multicoin/eth_direction_gate_a.py`（第 5 次 informed override，
CLAUDE.md「V7 多幣化提前啟動」授權提前執行；Go/No-Go 判準本身未變）

## 結論一句話

**ETH clean pooled sign-AUC = 0.5057，遠低於 pre-registered 門檻 ~0.54，也遠低於
BTC 基準 0.5412。且 first/second half IC 由 +0.048 翻負至 -0.035（halves 不同號）
——訊號不只弱，還不穩定。Step 2 FAIL，依預先登記規則，Step 3（訊號重合率）不需
再跑，V7 多幣化對 ETH 無性價比。**

## 方法（與 BTC 0.5412 完全同一套 harness）

- 資料：ETH 4000 根 1h bars（2026-02-06 ~ 2026-07-23），Binance klines(ETHUSDT)
  + 12 個 Coinglass 端點（symbol=ETH/ETHUSDT，逐一驗證非 coinbase_premium 的
  symbol-ignoring 陷阱——排除 coinbase_premium，Step 1 audit 已證明它無視 symbol
  參數、永遠回 BTC 值）
- 特徵：`build_live_features()`（生產函式，未修改）逐字重用；136/136 Direction
  特徵欄位存在（cb_premium 3 欄以外全部有值）
- WF harness：`research.feature_search_ab._per_fold_oos(leaky=False)` +
  `_pooled()`（逐字重用，與產出 BTC 0.5412 的函式完全相同，零 hyperparameter
  drift）；purge=4/embargo=4，77 folds（288 initial_train + 48 test + 48 step）

## 數字

| 指標 | ETH | BTC 基準 |
|---|---|---|
| pooled clean sign-AUC | **0.5057** | 0.5412 |
| pooled clean IC | +0.0047 | ~0.17（歷史值，非同一 run 直接對照） |
| mean per-fold AUC | 0.5221 | — |
| median per-fold AUC | 0.5220 | — |
| frac folds > 0.54 | 38.96% | — |
| first-half pooled AUC/IC | 0.5301 / +0.0478 | — |
| second-half pooled AUC/IC | 0.4832 / **-0.0349** | — |

## 資料品質檢查（排除「數字差是 bug 不是訊號差」的可能性）

1. `close` 價格範圍 $1521.55~$2450.00 — 正確 ETH 價格區間，非誤抓 BTC 資料。
2. 只有 `cg_cb_premium*` 3 欄 100% NaN（**設計內**，Step 1 已證明該端點
   symbol-ignoring，故意不抓）。
3. `cg_oi_cm_*`（oi_coin_margin 家族，4 欄）75% NaN——ETH 該端點的可用歷史深度
   明顯短於 BTC，是真實的資料限制不是抓取失敗；`fillna(0)` 後這 4 欄在前 75%
   窗口變常數 0，對整體 AUC 的下拉頂多是邊際的（4/136 特徵），不足以解釋
   0.5057 vs 0.5412 的 0.035 落差。
4. 其餘 columns NaN 比例均 < 6%，符合 rolling window warmup 的正常缺值。

**結論：不是資料 bug 壓低了數字，是訊號本身在 ETH 上不存在（或至少在這套從 BTC
移植過來的特徵+超參數組合下不存在）。**

## 對 TODO.md §4.6 Go/No-Go 判準的落地

預先登記：「ETH clean AUC ≥ ~0.54 且重合率 <50% → 繼續；任一不過 → 多幣化對
V7 無性價比，資源回異源資料線」。

Step 2 本身已 FAIL（0.5057 << 0.54），且 halves 不同號顯示訊號不穩定——不是
「差一點點」的邊緣案例。依規則，**Step 3（ETH/BTC Strong 訊號重合率）不需要再
跑**：即使重合率 <50% 也無法通過 AND 邏輯的 gate。

**V7 對 ETH（用同一套 BTC 移植的特徵+超參數）NO-GO。** 不代表 ETH 完全不可能
有 edge——只代表「直接搬 BTC 的 136 特徵 + 未調整的超參數」這條路走不通。
若未來想再嘗試，需要 ETH 專屬的特徵工程/超參數調整（等同重新做一次 Step 4-5
的完整模型開發，不是「移植」），而非本次 Step 2 的範疇。SOL 分支依此結果
**不建議投入**（同樣是「搬 BTC 特徵」的路線，預期同樣的失敗模式）。

**撤單流的 ETH 多幣化不受此結果影響**——那是完全獨立的 research track（v1
撤單定義 + 分鐘級信號），已經上線資料時鐘（見 TODO.md「撤單流多幣化」），
繼續累積到 10 月 re-run，不因 V7 這條線的結果而暫停。

## 產出檔案

- `research/multicoin/eth_direction_gate_a.py` — 可重跑腳本
- `research/multicoin/.cache/eth_features_all.parquet` — ETH 特徵表快取
- `research/results/multicoin/eth_direction_gate_a.json` — 結構化結果
