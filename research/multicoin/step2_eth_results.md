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

## 後續：超參數重調 + 淺覆蓋率欄位剔除（2026-07-23 同日，Follow-up）

使用者質疑：「是不是因為直接照搬 BTC 調好的超參數，沒重調才失敗？主流幣跟
小幣訂單流特徵本來就會差很多」。這是合理的質疑——`_per_fold_oos` 用的
`BASE_PARAMS` 是針對 BTC 雜訊特性調的，加上 `oi_coin_margin` 家族 4 欄對 ETH
覆蓋率只有 75%，兩者都可能是 unfair transfer 的來源。

**測試設計**（刻意不做網格搜尋——那等於換個包裝重演 2026-06-20 threshold-sweep
的錯誤；只測 3 個有先驗理由的變體，用跟 BTC 特徵 A/B 完全同一套 4-condition
sanity gate 判定，見 `research/multicoin/eth_retune_ab.py`）：

| 變體 | 改動 | pooled AUC | agg_lift | mean_fold_lift | frac_pos | boot_p≤0 | 4-cond 判定 |
|---|---|---|---|---|---|---|---|
| baseline | （同上，未調） | 0.5057 | — | — | — | — | — |
| regularized | max_depth 4→3, min_child_weight 10→20, reg_lambda 1.0→2.0 | 0.5073 | +0.0016 | +0.0043 | 0.47 | 0.243 | **no significant lift** |
| drop_thin_oi_cm | 剔除 4 個 75% NaN 的 oi_coin_margin 欄 | 0.5054 | -0.0004 | +0.0025 | 0.57 | 0.358 | **no significant lift** |
| regularized+drop | 上述兩者合併 | 0.5034 | -0.0023 | -0.0018 | 0.53 | 0.622 | **no significant lift** |

三個變體全部沒有通過 4-condition gate（agg_lift 沒有一個 > 0.005，bootstrap
p 全部遠高於 0.05 顯著門檻），而且**沒有一個變體的絕對 AUC 靠近 0.54 門檻**
（0.5034~0.5073 之間打轉，跟 baseline 0.5057 本質上是同一個數字）。

**結論：不是超參數移植不公平、也不是淺覆蓋率欄位在扯後腿——是 136 特徵這套
機制本身在 ETH 上就是不帶訊號。** 加碼調參/篩欄位不會把 0.505 變成 0.54，
因為問題不在「校準」層次，是在「這些特徵跟 ETH 4h 方向根本沒有這套 BTC 特有
的關聯結構」這個更根本的層次。要真的有機會，需要的是 ETH 專屬的全新特徵
工程（不同的訂單流代理、不同的正規化窗口、可能需要處理 BTC-beta 稀釋問題），
等同從 Step 4（特徵工程）重新開始，不是調參數/篩欄位這種小修小補能解決的。

## 第二次 follow-up：特徵刪減篩選（2026-07-23 同日）

使用者提議：與其重新設計特徵（最貴的路線），先試「刪減式」——用嚴格的
univariate IC + 跨 fold 同號一致性篩選現有 136 個特徵，看有沒有一小撮子集
本身帶真訊號，只是被其餘雜訊特徵稀釋掉。方法見
`research/multicoin/eth_feature_elimination.py`：
- IC 只用 WF **test-fold** bars 算（不用 train bars，避免篩選本身有洩漏）
- 門檻刻意比 `featsearch_lib.py`（BTC 新特徵 A/B 用的篩選）更嚴：
  `|IC| >= 0.05` 且 `跨 fold 同號一致性 >= 0.65`（136 個特徵逐一篩選本身是
  多重比較，門檻鬆會篩出純巧合的贏家——p~0.05 下 136 個特徵預期有 ~6.8 個
  假陽性）

**篩選結果**：5/136 通過（`research/results/multicoin/eth_feature_screen.csv`）：

| feature | IC | 跨fold一致性 | n_folds | 備註 |
|---|---|---|---|---|
| cg_oi_agg_close | -0.072 | 88% | 77 | **原始水位值，未差分**——非平穩序列跟任何有趨勢的序列都容易假性相關,不是驗證過的訊號 |
| cg_oi_close | -0.065 | 92% | 77 | 同上，未差分水位值 |
| cg_oi_cm_vs_usd | +0.056 | 67% | **21** | 只有 21/77 fold 有資料（呼應 oi_coin_margin 75% NaN 的已知缺口），建立在一小段不具代表性樣本上 |
| cg_ls_divergence_zscore | +0.054 | 71% | 77 | 唯二看起來正常的候選 |
| cg_funding_close | -0.052 | 68% | 77 | 同樣是未差分水位值，較不可疑（funding rate 本身較接近平穩）但仍需保留懷疑 |

拿這 5 個特徵重訓（單次 retrain，不做疊代式重篩，避免二次多重比較）：

| | pooled AUC | pooled IC |
|---|---|---|
| baseline（136 特徵） | 0.5057 | +0.0047 |
| reduced（5 特徵） | **0.5267** | +0.0474 |

過同一套 4-condition gate：`agg_lift=+0.021`（過）、`mean_fold_lift=+0.0064`
（過），但 **`frac_pos=0.53`（未過 >0.55）、`bootstrap p(lift≤0)=0.358`
（未過 <0.05，遠不顯著）** → **判定 no significant lift**。且絕對數字
0.5267 仍遠低於 0.54 門檻。

**結論**：pooled AUC 數字看起來有進步，但拆開看撐不住——不是跨 fold 穩定的
提升，是幾個 fold 的巧合堆出來的（跟 mistake.md 2026-06-02 WQ101 案例
「aggregate 好看、per-fold mean 是負的」同一種陷阱）。而且 5 個「贏家」裡
有 3 個本身統計上可疑（2 個未差分水位值、1 個樣本覆蓋率不足 30%）。

**三個獨立角度現在完全收斂**：直接移植（AUC 0.5057）、超參數重調（0.503~
0.507）、特徵刪減篩選（篩選後 reduced 模型 0.5267 但不顯著）——沒有一條
路徑產生禁得起驗證的訊號。這套 BTC 設計的 136 特徵家族對 ETH 4h 方向
沒有真正可靠的關聯結構，不是校準/篩選/調參能解決的問題。

**這次 override（V7 多幣化提前推進）到此正式收尾**：Step 2 用了三輪測試
（原始移植 + 重調 + 特徵刪減 follow-up）都確認 NO-GO，沒有留下「也許只是
沒調好/沒篩對」的疑點。若要真的繼續，唯一剩下的路是從頭替 ETH 設計新特徵
（不同的訂單流代理、正規化窗口、處理 BTC-beta 稀釋問題），等同重新啟動
Step 4-5 完整模型開發，不是本次 Step 2 範疇能解決的。SOL 分支同理不建議
投入。撤單流的 ETH 多幣化維持不受影響，繼續累積。
