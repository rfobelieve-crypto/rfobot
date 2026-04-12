# Mistake Log

Record logic errors and bad decisions to avoid repeating them.

---

## 2026-03-28: price_change fallback over-engineering

**What happened:**
Item 9 (fix `_get_price_change` dependency on normalized_trades) was implemented with 3 chained queries:
1. Query flow_bars_1m to find nearest bar
2. Query normalized_trades within that bar's time window
3. Fallback to delta/volume estimation

Step 1→2 was pointless — querying normalized_trades scoped to a flow_bar window is the same as querying it directly. This tripled the DB load per snapshot for no benefit.

**Root cause:**
Jumped to a "clever" solution without thinking about whether the intermediate step added value. flow_bars_1m doesn't store price, so using it as an index to find normalized_trades was a round-trip to nowhere.

**Correct approach:**
1. Try normalized_trades first (works for events < 3 days, same as original)
2. Only if no data, fallback to delta/volume ratio from flow_bars_1m

**Rule:** When adding a fallback path, ask: "Does this intermediate step give me information I don't already have?" If not, skip it. Prefer the simplest query chain that solves the problem. Don't add queries that increase Railway DB usage without clear value.

---

## 2026-03-29: delta/volume ratio ≠ price change

**What happened:**
`_get_price_change()` fallback used `total_delta / total_vol * 100` when normalized_trades had no data. This produced values like +4.84% that looked like real price moves but were actually the **taker imbalance ratio** (what % of volume was net buy).

**Root cause:**
Confused two different metrics. delta/volume ratio measures buy-sell pressure, not price movement. The two are correlated but not interchangeable — especially on short windows where slippage is minimal.

**Correct approach:**
Return None when normalized_trades has no data. Don't fabricate price estimates from flow data.

---

## 2026-04-01: 把 MACD / EMA 放進訂單流研究的特徵集

**What happened:**
`feature_builder_v2.py` 計算了 `ema_9`, `ema_21`, `macd`, `macd_signal` 並寫入 features 表。這些欄位出現在 feature validation 結果中，ICIR 看起來很高（-0.85~-0.91），但這根本不該存在。

**Root cause:**
誤把傳統技術指標混入訂單流研究。這個專案的研究範疇是純訂單流（CVD、delta_ratio、aggTrade flow、funding rate、OI），不包含 price-derived 的技術指標如 MACD / EMA / RSI 等。

**Correct approach:**
feature_builder_v2.py 只能包含以下來源的特徵：
- aggTrade flow（CVD、delta_ratio、buy/sell vol、large order）
- Funding rate（rate、deviation、zscore）
- OI（未來補充）
- Cross-exchange divergence
- 純統計衍生（realized vol、return lags）— 可接受，因為是 price behavior 而非 pattern indicator

MACD / EMA / Bollinger 等技術指標一律不加。

**Rule:** 每次加新特徵前先問：「這是訂單流資料還是技術指標？」技術指標一律排除。

---

## 2026-04-02: 加 log 行導致 webhook 500 crash

**What happened:**
在 `indicator/app.py` 的 `/webhook` handler 中加了一行 `logger.info("Webhook command: %s", cmd, chat_id)`，但放在 `cmd = text.split()[0]...` 定義**之前**。導致每次收到 Telegram 指令都觸發 `NameError`，回傳 500，用戶的 `/chart` 指令完全無反應。

**Root cause:**
加 debug log 時沒注意變數的定義順序。修改生產環境的 request handler 後沒有做基本的 code review（變數是否已定義）。

**Correct approach:**
新增的 log 行必須放在所有引用變數的定義之後。修改 webhook/route handler 這類每個請求都會跑的代碼時，要特別小心：一個 crash 會影響所有用戶。

**Rule:** 在生產 handler 中加 log 或任何代碼後，立刻檢查：所有引用的變數是否已定義？是否在 try/except 內？不要假設「只是加一行 log」就不會出錯。

---

## 2026-04-12: backfill 時間戳 unit 硬編碼導致 5.5 天數據缺口

**What happened:**
`research/backfill_all_parquet.py` 的 `to_1h_df()` 用 `pd.to_datetime(df[time_col], unit="ms")` 硬編碼毫秒。但 Coinglass API 的 `coinbase_premium` 和 `bitfinex_margin` 端點回傳的 `time` 欄位是 **10 位秒級時間戳**（如 `1775998800`），不是 13 位毫秒。秒級時間戳被當毫秒解析後變成 1970 年日期，merge_parquet 沒報錯（index dedup 保留了壞行），最終 4131 行壞數據 + 數據停在 2026-04-07。

`cg_bfx_margin_ratio` 是剪枝模型第 4 重要的特徵，如果下次訓練前沒發現這個缺口，模型會在該特徵上訓練出偏差。

**Root cause:**
假設所有 Coinglass 端點的時間戳格式一致。實際上大部分端點用 13 位 ms，但 `coinbase_premium` 和 `bitfinex_margin` 用 10 位 s。生產端的 `data_fetcher.py` 早就有 `if ts.max() > 1e12` 的自動偵測，但 backfill 腳本是另外寫的，沒抄這段邏輯。

**Correct approach:**
時間戳解析永遠用自動偵測：`unit = "s" if sample_ts < 1e12 else "ms"`。已修復。

**Rule:** 凡是解析時間戳的地方，永遠不要假設 unit 固定。寫新的數據處理腳本時，先看生產代碼怎麼處理同一個 API 的格式。同一個 API provider 的不同端點可以有不同的時間戳格式。

---

## 2026-04-12: is_stale() 只檢查 klines 導致端點級故障無聲

**What happened:**
`backfill_all_parquet.py` 的 `is_stale()` 只讀 `binance_klines_1h.parquet` 來判斷是否需要回填。klines 永遠是最新的（Binance 公開 API 不需要 key），所以即使 CG 端點已經停滯 5.5 天，`ensure_fresh()` 也不會觸發回填。訓練管線 `shared_data.py` 調用 `ensure_fresh()` 時以為數據是新的，實際上 coinbase_premium / bitfinex_margin 缺了 132 小時。

**Root cause:**
用最穩定的數據源（Binance klines）代表所有數據源的新鮮度。這是一種「以偏概全」的監控盲區 — 最不可能故障的組件被選為健康指標。

**Correct approach:**
`is_stale()` 改為遍歷所有 parquet 文件，任何一個超過 max_age_hours 就回傳 True。已修復。

**Rule:** 新鮮度 / 健康檢查必須覆蓋最脆弱的組件，不是最穩定的。如果系統有 N 個數據源，健康檢查要查 N 個，不是只查最可靠的那一個。

---

## 2026-04-12: 用錯 IndicatorEngine 屬性名（dir_model vs dual_dir_model）

**What happened:**
watchdog 新增的 `_check_dual_model()` 檢查 `engine.dir_model` 和 `engine.mag_model` 是否為 None。但 dual mode 下的屬性名是 `dual_dir_model` 和 `dual_mag_model`。`dir_model` 是舊 regime mode 的屬性，dual mode 下根本不存在，導致 `AttributeError`。

**Root cause:**
沒有先 grep 確認屬性名就寫代碼。`IndicatorEngine` 有三種 mode（dual/regime/legacy），每種 mode 的屬性名不同。

**Correct approach:**
寫監控代碼前先 `grep self\.dual_dir` 確認屬性名。已修正為 `dual_dir_model` / `dual_mag_model` + `hasattr` 防禦。

**Rule:** 引用物件屬性前先 grep 確認。特別是有多種初始化路徑的類別（如 IndicatorEngine 的 dual/regime/legacy），不同路徑設定的屬性名不同。不要憑記憶猜。
