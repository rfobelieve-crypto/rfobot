# Mistake Log

Record logic errors and bad decisions to avoid repeating them.

---

## 2026-04-14: 把 Strong 勝率目標寫成 95%（從策略系統沿用未更新）

**What happened:**
CLAUDE.md 長期寫「Strong 信號勝率目標 > 95%」，花了一整天嘗試各種方法提升 Direction model 都碰天花板，才回頭檢查這個目標本身是否合理。跑 `research/topk_precision_sweep.py` 用 2726 筆 walk-forward OOS 預測做 bidirectional top-k：

| k | precision | CI | signals/月 |
|---|---|---|---|
| top 1% | 59.3% | [40.7, 75.5] | 5 |
| top 2% | 63.6% | [50.4, 75.1] | 11 |
| **top 5%** | **67.6%** | **[59.4, 74.9]** | **27** |
| top 10% | 65.6% | [59.8, 71.0] | 53 |
| top 20% | 60.2% | [56.0, 64.2] | 106 |

峰值 67.6%。AUC 0.57 的理論 top-5% precision 上限 68-72%，代表**已經貼著數學天花板**。95% 在這個 AUC 結構下永遠達不到，那是 AUC 0.80+ 的模型才談的數字。

**Root cause:**
95% 是從早期策略系統（有 TP/SL、能過濾掉不利情境）沿用到指標系統，沒人重新校準。而指標系統的訊號是「原始預測」，不過濾，所以上限直接由模型 AUC 決定。把策略目標搬到指標系統等於給自己設一個數學上不存在的目標。

**Correct approach:**
precision 目標必須從**模型 AUC 反推**，不是拍腦袋定。公式約等於：
- 給定 AUC，top-k precision 上限 ≈ 0.5 + (AUC - 0.5) × kernel(k)
- AUC 0.57 + k=5% → 理論 ~0.70，實測 0.676（非常接近）
- 如果要求 0.95，需要 AUC ≥ 0.85

現在 CLAUDE.md 已改為「point estimate ≥ 65%，stretch 70%」。未來任何討論 Strong 勝率時，第一句話要問「當前模型 AUC 是多少，這個目標在結構上可達嗎」，不是「為什麼還沒達到」。

**Rule:** 設定任何 precision/recall 目標前，先用當前模型的 AUC/IC 反推理論天花板。如果目標高於天花板就是錯的目標，改目標而不是追目標。絕對不要從不同系統（策略 vs 指標）沿用績效目標——運作機制不同，天花板也不同。

---

## 2026-04-13: 用 in-sample 月份 IC 判斷訊號健康度（高估 0.5 AUC 級別）

**What happened:**
為了診斷 Magnitude IC 衰退，寫 `diagnose_mag_decay.py` 用**當前生產模型**去預測過去每個月的 `|ret_4h|`，得到 Nov 0.60 / Dec 0.51 / Jan 0.57 / Feb 0.53 / Mar **0.60** / Apr 0.47。看起來訊號完全沒衰退、近月甚至還很強，幾乎下結論「Mag 訊號穩定，問題不在這」。

後來跑乾淨 walk-forward（`mag_level_feat_swap.py`，每個測試窗只用之前的資料訓練）得到真實 OOS IC：Nov 0.31 / Dec 0.36 / Jan 0.24 / Feb 0.20 / Mar **0.10** / Apr 0.12。**Mar 差距 0.50 IC，Apr 差距 0.35 IC**。真實情況是 Mag 從 Feb/Mar 交界發生 concept drift，IC 腰斬。

也就是說，我的第一版診斷**用了訓練集預測訓練集**——生產模型訓練時吃了全部 4000 bars，對任何歷史月份做預測都是 in-sample，結果無法反映 model 是否能從歷史學到新規律。

**Root cause:**
沒區分「model fit」和「model generalization」。生產模型的 IC 是在**全部資料訓練完**才算的，拿它去預測過去月份天生是作弊。這跟 Kaggle 新手用 `cross_val_score` 之後又用全資料重訓再看 train loss 是同一個錯誤——只是換了個包裝。更糟的是，月份切片讓我以為這是「time-slicing 驗證」，實際上完全沒做 time-based split。

**Correct approach:**
任何「模型是否仍然 work」的評估都必須是**嚴格 walk-forward**：每個測試點的模型只能看到該點之前的資料。生產模型的 in-sample 預測**永遠不能拿來回答「訊號是否衰退」「特徵是否還有效」「regime 是否改變」這類問題。能用 in-sample 回答的問題只有：「訓練收斂了沒」「在完整資料上 model 的 upper bound 在哪」。

**Rule:** 診斷 IC/AUC 衰退時，第一句 assert 必須是「這個預測是 walk-forward 還是 in-sample」。in-sample 的結果在「診斷衰退」這個 task 下**零資訊量**，不管數字多漂亮都等於沒測。如果檢查清單裡的測試方法是「用生產模型預測過去月份」，那就是錯的測試方法，換掉。

---

## 2026-04-13: Regime-specific 子模型在小樣本下退化成比隨機還差

**What happened:**
為了試圖突破 Direction model 天花板，訓練 bull/bear/chop 三個 regime-specific 子模型（`regime_specific_direction.py`）。假設是「每個 regime 的特徵→方向關係不同，獨立訓練應該贏過全局模型」。

結果全局模型在三個 regime 上的 AUC 分別是 CHOP 0.548 / BULL 0.500 / BEAR 0.497，regime 子模型是 CHOP 0.550 / BULL 0.440 / **BEAR 0.378**。BEAR 子模型 AUC 顯著低於 0.5，意味著它**系統性預測反方向**。

原因：BEAR 整個 4000 bar 資料集只有 724 筆，扣掉 walk-forward test + NaN，每個 split 的訓練集只剩 50-100 筆。XGB 在這個樣本數下嚴重 overfit 訓練集的雜訊方向，預測出來的機率跟實際 label 反相關。BULL 也有 16 個 split 因為訓練樣本不足 < 50 直接跳過，等於是選擇性覆蓋。

**Root cause:**
沒評估「資料切片後每個 regime 的有效樣本數是否夠訓練」。少於 ~500 的小樣本訓練 gradient boosting 會 overfit 到雜訊，而且資料越少 overfit 越嚴重，甚至可能學到完全相反的方向。把這當成「子模型比較弱」來解讀是錯的——這些子模型根本沒進入「能學東西」的 regime。

**Correct approach:**
切片訓練前先算 min(regime_sample_count) 是否 > 500（gradient boosting 的大略安全線）。如果不夠：
1. **退一步用 regime dummies 當 feature** 讓全局模型自己學 conditional split（這是 XGB 設計本來就能處理的）
2. 或用 `sample_weight` 在全局訓練時加權少數 regime，**不要**切開訓練
3. 如果真要切，只切樣本充足的 regime（這個資料集只有 CHOP 有 2000+ 筆，結論是：沒得切）

**Rule:** 分群訓練前，每群的有效訓練樣本數必須 > 500（至少要 > 300），否則不如用單一模型 + 分群特徵。小樣本下 gradient boosting 不會變「局部專家」，會變「雜訊放大器」。如果樣本數不夠，把分群改成 feature 而不是改成 partition。

---

## 2026-04-13: 用混合模型版本的數據下 calibration 結論

**What happened:**
跑 `calibration_check.py` 看到 Brier skill -0.098、ECE 0.16、over-confident +0.197，bootstrap CI 全部顯著（[-0.184, -0.014] 整條在零下，conf_gap [+0.115, +0.285] 整條在零上），就據此推論「模型 miscalibration 是真的」並開始討論 Platt scaling / isotonic / rolling percentile threshold 等解法。

然後往下挖才發現 244 個 valid 樣本全部來自 2026-04-02 → 2026-04-12 這 10 天，這個窗期：
  - 2026-04-03 部署 dual v7 初版（88 特徵）
  - 2026-04-09 切換成 pruned 29 特徵 + regime weighting
  - 2026-04-12 又重訓一次
  - 5.5 天 `cg_bfx_margin_ratio`（第 4 重要特徵）灌壞數據（2026-04-12 backfill bug）

也就是說：calibration 測試基於 **三個不同模型的混合預測 + 重大特徵被污染一半時間**。bin-level 極端區的怪象（p≥0.70 actual=0.50）很可能只是模型切換那幾小時產的 transient，不代表任何一個模型的穩態。前面提出的所有解法都建立在錯誤的前提上。

**Root cause:**
看到統計顯著的壞結果就急著找解法，沒先問「這個測試數據對應的是哪個模型？數據本身是乾淨的嗎？」最基本的 data sanity check 被跳過了。更糟的是 bootstrap CI 讓我更有信心下結論——但 CI 只能量**抽樣不確定性**，量不到**數據污染**或**模型版本混合**這種系統性偏差。統計顯著 ≠ 結論可信。

**Correct approach:**
評估模型前必須確認：
  1. **樣本範圍對應單一模型版本**：git log 查最新模型 deploy 時間，樣本必須在那之後。
  2. **樣本範圍不含已知數據污染窗**：查 mistake log 看近期有沒有數據 bug。
  3. **樣本數夠**：即使資料乾淨，n<100 的 calibration 點估計不穩定；n<500 做 isotonic 會 overfit。
  4. **先看時間切片**：分月/分週跑同一個測試，如果每段結論都不同，整體測試就沒意義。

已在 `calibration_check.py` 的 roadmap 加上 `--since` flag 和 model version guard（讀取最新 model mtime，樣本必須 >= 該時間），還沒實作。

**Rule:** 評估任何模型的統計量前，第一件事是「確認這份評估樣本是從同一個模型 + 同一份乾淨數據產生的」。這個 sanity check 要**在看結果之前**做，不是看到壞結果才回去查。Bootstrap / permutation / 顯著性檢定全部都只能處理抽樣誤差，不能處理「你在量錯的東西」這種問題。看到「顯著的壞結果」第一反應應該是懷疑測試設計，不是懷疑模型。

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

---

## 2026-04-13: 用 sparse indicator 做 feature interaction 是退化操作

**What happened:**
為了解決 Direction Model 的 regime 適應性問題，原本想加 9 個 regime interaction 特徵：
```python
oi_agg_close_x_bear = cg_oi_agg_close * is_bear
bfx_margin_x_bull   = cg_bfx_margin_ratio * is_bull
ls_ratio_x_bear     = cg_ls_ratio * is_bear
# ... 等等
```
寫法看起來完全合理，是 ML 教科書經典 interaction term 寫法。

跑 IC 驗證後發現怪事：4 個本質完全不同的金融特徵在 ×is_bear 之後互相相關 0.96-0.98：
```
bfx_margin_x_bear ↔ oi_agg_close_x_bear     corr = +0.984
bfx_margin_x_bear ↔ ls_ratio_x_bear         corr = +0.957
oi_agg_close_x_bear ↔ ls_ratio_x_bear       corr = +0.968
```
而且 IC 全部從 base 的 -0.05~-0.07 掉到 +0.01，p-value 變不顯著，train/test FLIP。

**Root cause:**
BEAR 只佔 18% 樣本（724/4000）。`feature × is_bear` 等於：
- 非 BEAR 時 = 0（佔 82%）
- BEAR 時 = feature 原值（佔 18%）

問題在於**「在哪些 timestamp 是 0」這個 sparsity pattern 在所有 ×is_bear 特徵裡完全一樣**。所有特徵共享同一組 18% 的非零 mask。剩下 82% 的零值貢獻了大部分變異數。

結果 spearman correlation 主要在量「這個 sample 是不是在 BEAR 期間」，而不是「這個 feature 在 BEAR 的時候值是多少」。三個 base 完全不同的特徵看起來幾乎一模一樣，因為它們的 0/非0 pattern 完全重疊——indicator 的 sparsity 訊號壓過了被乘的特徵本身。

**Correct approach:**
1. **乘以 `(1 - is_X)` 才有意義**：保留 80%+ 樣本，只把死掉的 regime 設 0。例如 `vol_kurt_non_bear = vol_kurtosis * (1 - is_trending_bear)`，IC validated +0.054 stable, `oi_8h_non_bull = cg_oi_close_pctchg_8h * (1 - is_trending_bull)` IC validated -0.071（比 base -0.062 強 15%）。
2. **regime indicator 本身要直接當 feature 加進去**（is_trending_bull / is_trending_bear），讓 XGB 自己用 tree split 決定 conditional rule。手動寫 `feat × is_X` 是把訊號塞進更窄的 channel。
3. **inter-feature correlation matrix 必須當成標準驗證步驟**，跟 train/test split、rolling IC 同等重要。如果一群本應獨立的特徵互相相關 > 0.9，那不是訊號，是 indicator pattern leakage。

**Rule:** 設計 interaction feature 時，**永遠不要寫 `feat × sparse_indicator`**——當 indicator 的非零比例 < 30%，乘出來的特徵會被 sparsity pattern 主導，跟其他用同一 indicator 乘的特徵高度相關，IC 也會 collapse。如果要做 regime conditioning：(a) 把 indicator 直接當 feature，讓 XGB 自己學 split；(b) 只在「base feature 在某 regime 完全死掉」的情況下用 `feat × (1 - is_dead_regime)` 形式屏蔽噪音。設計完任何 interaction 都要先跑 inter-feature correlation matrix。

---

## 2026-04-19: 多個腳本覆寫同一個 JSON 導致 warmup buffer 被清空

**What happened:**
系統連續產出大量 DOWN 信號，比例明顯不合理。排查後發現 `training_stats.json` 裡的 `dir_pred_history`（Direction model 的 500 筆 warmup 預測）是空的。沒有 warmup buffer，系統永遠用固定 fallback 閾值解碼方向——這些閾值是歷史均值，無法適應當前 bearish regime，結果只要模型預測稍微偏負就觸發 DOWN。

事件鏈：
1. 4/15 `export_direction_reg_model.py` 正確寫入 500 筆 `dir_pred_history` ✅
2. 4/16 `deploy_new_models.py` 重訓 Magnitude model，用 `json.dump` **整個覆寫** `training_stats.json`，只寫了 `pred_history`（mag 的 warmup），`dir_pred_history` 被洗掉 ❌
3. 之後每次 Railway 重啟（git push 觸發），buffer 歸零，永遠不到 100 根 warmup 門檻
4. 系統永遠用 fallback 閾值 → bearish 市場下 DOWN 信號爆量

**Root cause:**
三個腳本寫同一個檔案，但寫法不一致：
- `export_direction_reg_model.py`：先讀再寫（read-then-update）✅
- `deploy_new_models.py`：直接 `json.dump` 覆寫 ❌
- `export_production_models.py`：直接 `json.dump` 覆寫 ❌

後面兩個腳本沒有意識到這個檔案是**共用的**，裡面有別的腳本存的資料。這是最基本的共用資源協調問題。

**Correct approach:**
寫入已存在的 JSON/config 檔案時，永遠用 read-then-update 模式：
```python
if stats_path.exists():
    with open(stats_path) as f:
        stats = json.load(f)
else:
    stats = {}
stats["my_key"] = my_value  # 只更新自己負責的 key
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)
```

額外加了兩層防護：
1. `app.py` 每次 update cycle 結束後持久化 `dir_pred_history`，這樣 Railway 重啟不會失去已累積的 warmup
2. 修復了 `deploy_new_models.py` 和 `export_production_models.py` 的寫法

**Rule:** 寫入任何共用檔案前，第一步是 `grep` 看還有誰也在寫這個檔案。如果有多個寫入者，必須用 read-then-update 模式，只動自己負責的 key。直接 `json.dump` 覆寫整個檔案等於對其他寫入者說「你存的東西我不在乎」——這在單一寫入者時沒問題，多個寫入者時是資料刪除。
