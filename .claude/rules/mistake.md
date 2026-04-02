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
