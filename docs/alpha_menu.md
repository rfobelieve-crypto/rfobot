# Quant Arb 檔案庫的 alpha 菜單（2026-09-13 盤點）

**為什麼有這份**：使用者 2026-09-13「Quant arb 之前有寫，還有很多 alpha 策略
可以抓」。盤點證實了——**96 篇裡我們只把 5 篇變成過程式碼**，而且有一整族
（合成均值回歸組合，10 篇）連碰都沒碰。

我們到今天為止的讀法是「撞到問題 → 去找一篇 → 讀 → 轉譯」，結果是
**讀過的那幾篇很熟、沒讀過的等於不存在**。這份是菜單，不是摘要。

- 全文在 `D:\flowbot_data\quant_arb\articles\`（92 篇線上全有，
  2026-09-13 以 sitemap 核對：線上 92 = 本機 92，**零缺口**）
- 純文字快取在 `D:\flowbot_data\quant_arb\_txt\`（grep 用）
- 粗定位跑 `python research/ops/alpha_inventory.py`
- **HFT Alphas 系列的 pt-3/4/5 還沒被寫出來**（三個網址都 404，
  作者最新一篇是 2026-06-29）。那三篇是「擴到前 30 名 + 橫斷面因子模型
  + 特徵選擇 + 預測模型」—— **要自己做，不是等**。
- 另一個刊物 `ninjaquant.substack.com` 有 5 篇，**未抓**。

---

## 狀態圖例

| | 意思 |
|---|---|
| **已實作** | 有我們自己的程式碼，而且跑過判決 |
| **進行中** | 今天開的 |
| **可測** | 資料在手，隨時可以做 |
| **缺資料** | 要新錄或買 |
| **未讀** | 連讀都還沒讀 |

---

## A. 簿口／微結構（我們的主戰場）

| 篇 | 給什麼 | 我們 |
|---|---|---|
| `2026-04-06 A Real HFT/MFT Alpha` | 掛單**年齡**拆新舊（上個快照有無 $100）、5/10 bps 失衡、橫斷面 z、1h 再平衡 | **已實作** §1.23／`mft_xs_alpha.py` |
| `2026-06-08 hft-alphas-pt-1` | 9 個 5s/15s 特徵，最終 `features_5s` 7 個。結論：**簿口失衡壓倒性最強**、反轉在 5s 很弱 | **進行中** `hft/hft_alphas_lighter.py`（兩個簿口特徵已測，BTC IC 0.146／0.231 = 他的量級）|
| `2026-06-17 hft-alphas-pt-2` | 擴充特徵集，多個 >5 Sharpe（稅前） | **未讀完** |
| `2026-06-02 hft-alpha-research-101` | HFT alpha 的研究方法論 | **未讀** |
| `2025-07-01 researching-hft-strategies` | 高頻資料怎麼work | 部分讀過 |
| `2023-02-08 using-order-size-for-alpha` | 用**訂單大小**的代理做 delta 中性策略 | **可測** —— 逐筆帶有 `sz`／`usd` |
| `2023-11-19 hide-n-seek pt1` | 怎麼藏自己的流（降低衝擊） | **未讀** |
| `2023-12-09 hide-n-seek pt2` | **偵測別人的執行演算法** | **可測，而且我們有別人沒有的東西** —— HL 逐筆帶存了**雙方地址** |
| `2026-05-27 analysing-real-fills` | Binance 掛單成交的統計 | **缺資料**（要我們自己的成交） |
| `2024-03-10 finding-fair-value` | 公允價 = 迴歸；輸入點名**跨場館領先**與**訂單大小** | 讀過，未實作 |

## B. 跨場館／套利

| 篇 | 給什麼 | 我們 |
|---|---|---|
| `2024-07-09 alpha-6 perpetual arbitrage` | 成本結構、incomplete spread、**Part 3b/3e lead-lag** | **已實作**（成本模型第 8 項、`band.py`、§1.37） |
| `2024-10-01 alpha-7 advanced perpetual arb` | 「Advanced perpetual arbitrage strategy」 | **未讀** ← 直接的下一篇 |
| `2023-07-03 / 07-17 / 09-17 alpha 1/2/3` | 真實策略、進階套利、**三角套利用限價單** | **未讀** |
| `2024-01-30 alpha-4 funding arbitrage` | 利基資金費套利 | 部分讀過 |
| `2025-06-20 ultimate-crypto-arbitrage-guide` | **所有**套利策略的走查 | **未讀** |
| `2025-09-22 finding-arbitrage-opportunities` | 最好的機會在哪 | **未讀** |
| `2024-04-07 how-to-level-up-your-arb-game` | 既有套利怎麼進階 | 部分讀過 |
| `2024-11-17 geographic-arbitrage` | 地理套利（操作型優勢） | **未讀** |
| `2023-11-04 strategy-discussion-lead-lag` | lead-lag 的模型與技巧 | **已讀**，§1.37 用它 |

## C. 合成均值回歸組合（MRP／配對）—— **整族沒碰，10 篇**

| 篇 | 給什麼 |
|---|---|
| `2023-03-23 pairs-trading-framework-and-process` | **整族的總綱**（30.9k 字，免費） |
| `2023-02-10 monte-carlo-minimization-for-synthetic` | 蒙地卡羅最小化 |
| `2023-02-27 semi-definite-programming-for-mrps` | 半正定規劃 |
| `2023-03-08 non-sparse-synthetic-portfolios` | 非稀疏合成組合 |
| `2023-03-16 truncation-method-for-smrps` | 截斷法 |
| `2023-03-20 / 03-29 greedy-method-for-smrps pt1/pt2` | 貪婪法 |
| `2023-06-11 a-real-pairs-trading-strategy` | 一個真的配對策略 |
| `2023-06-23 thinking-about-stationarity` | 定態性該怎麼想 |
| `2023-11-11 pairs-trading-papers-review` | 論文回顧 |

**可測**：只需要價格序列，而我們有 29 幣的小時 K 線（滾動 930 天）。
**這是整個檔案庫裡「資料成本最低 × 我們完全沒取樣」的交集。**

## D. 選擇權（完全沒碰，8 篇）

`2026-04-07 options-alphas-pt-1`（3 個 alpha：BTC/ETH/SOL/XRP）、
`2026-04-14 pt-2`（再 2 個 + 合成）、`2026-03-19 options-mft-strategies`、
`2025-01-26 professional-options-market-making`、`2025-03-01 options-mm-pt2`、
`2025-11-18 advanced-options-market-making`、`2025-05-21 live-options-quoter`、
`2024-05-08 calculating-implied-volatility-fast`、`2023-06-17 equity-option-mispricing`

**缺資料**：我們只有 Deribit 的 DVOL 與 option summary（`gex_snapshots`）。
但 §0.88d 的 GEX 線已經在錄，這族不是零基礎。

## E. 季節性／週期（沒碰，4 篇）

`2023-08-18 seasonality-comprehensive`、`2023-06-09 seasonality-in-commodities`（附論文＋碼）、
`2024-12-15 seasonal-alpha-in-small-caps`（**交易多種加密貨幣**）、
`2023-02-26 election-cycle-seasonality`（附碼）

**可測**：只要 K 線。**但要先扣資金費**——`why-is-my-backtest-wrong` 第 11 項：
資金費結算時價格會動，不扣的話**結算時點會長得像時段 alpha**。

## F. 方法論（不是 alpha，但會改變我們怎麼做）

| 篇 | 為什麼重要 |
|---|---|
| `2025-03-22 why-is-my-backtest-wrong` | **已做成 `.claude/rules/backtest-audit.md`** |
| `2023-02-07 ranked-ls-turning-a-formula-into` | **把任何公式變成策略**的通用配方 —— 上面每一個 alpha 都要經過它 |
| `2026-06-28 fixing-learning-to-rank` | LTR 為什麼不work、怎麼修 |
| `2023-04-16 non-linearity-without-machine-learning` | 量化圈怎麼處理非線性（不用 ML） |
| `2023-05-19 timeframe-crowding` | **熱門回看期因為擁擠而更好** —— 可測，只要 K 線 |
| `2025-12-15 forecasting-done-right` | 預測的做法 |
| `2023-05-26 / 06-08 outliers pt1/pt2` | 極端值（`data-pre-processing` 說正規化在這兩篇） |
| `2024-05-14 timeframes-and-research-types` | 機械式 vs 統計式 vs 假說驅動 |
| `2024-10-12 wheres-the-edge` | 錢在哪裡賺 |
| `2025-07-12 beginner-mistakes-in-quant` | |
| `2026-06-29 lessons-from-the-desk` | 建一個量化研究作業 |
| `2025-06-04 building-an-ai-agent-hedge-fund` | 全 agent 驅動的研究流程 |

## G. 做市（有讀，未實作）

`2024-08-26 market-making-for-dummies`、`2025-08-02 advanced-market-making`、
`2024-04-06 alpha-5 market-making`

§1.28 的「倉位型做市門檻 ≈ 0.5 bps」就是從這裡來的，**而那個數字我們從未驗證**。

## H. 基礎建設（讀過／部分）

`2024-01-02 data-sourcing`、`2024-01-10 data-pre-processing`、
`2023-10-07 low-latency-dup-data`、`2026-02-08 ultimate-crypto-latency-guide`、
`2026-04-26 more-advanced-latency-tricks`、`2024-02-18 starting-your-quant-business`、
`2024-07-18 deploying-strategies`、`2024-04-16 developing-trading-algorithms-python`、
`2023-04-24 execution-without-the-fluff`、`2023-11-27 continuous-trading`、
`2024-05-22 / 06-18 automating-alpha pt1/pt2`、`2024-05-31 event-based-alpha`、
`2023-05-15 alpha-pipeline-raw-data-trades`、`2023-12-29 automating-charting`

---

## 排序建議（按「資料成本 ÷ 已知答案強度」）

1. **C 族（MRP／配對）** —— 資料只要 K 線（在手），而我們**完全沒取樣**。
   10 篇裡有一篇 30.9k 字的總綱。這是最大的空白。
2. **`hft-alphas-pt-1` 剩下的 5 個特徵** —— `tob` 現在可以組真正的 5 秒 OHLC，
   而且**有他的已知答案可以對照**（反轉在 5s 應該很弱）。
3. **`hide-n-seek pt2`（偵測執行演算法）** —— HL 逐筆帶存了**雙方地址**，
   這是連他都沒有的資料。**最可能長出「衍生而非複製」的東西的地方。**
4. **`alpha-7 advanced perpetual arbitrage`** —— 套利線的直接下一篇，而我們
   停在 alpha-6。
5. **`ranked-ls`** —— 不是 alpha 是配方，但上面每一個都要經過它。

**不建議先做**：D 族（選擇權，缺資料）、E 族（季節性，要先解決資金費污染）。
