# 回測的二十道檢討（2026-09-13 使用者訂立）

**來源**：Quant Arb,「Why is my backtest wrong?!」（2025-03-22，
`D:\flowbot_data\quant_arb\articles\2025-03-22_why-is-my-backtest-wrong.pdf`）。
使用者：「針對我之後每次策略的回測系統做檢討優化」。

**什麼時候用**：任何回測收工之前、任何要把回測結果寫進判決節之前。
與 `factor-research.md` 的十道檢查互補——那十道問「這個效應是不是我挑出來的」，
這二十道問「**我的模擬器有沒有在騙我**」。

**作者的總則，比二十項本身重要**：

> 「If your reaction upon seeing an amazing backtest isn't 'ah man what broke'
> then you haven't seen enough of them to know better. **There is no exception
> to this.** Every entirely straight line I have ever generated in backtest has
> had some flaw, and the only super straight lines that actually realized in
> production weren't backtest-able in the first place (market making).」

而他點名只有兩種錯會讓曲線變成一條直線：**前視**，以及
**執行假設錯得離譜（限價成交 ＋ 返佣 ＋ 零逆選擇 ＋ 即時成交）**。
§1.02 舊線死在第二種。

---

## 逐項：文章說什麼 ／ 我們的狀態

狀態用四個字：**已封**（結構性擋住且驗過）、**已量**（量過但沒有守衛）、
**未查**、**有矛盾**。

| # | 項目 | 我們 | 要點 |
|---|---|---|---|
| 1 | 前視 | **已封**（部分） | 見下方專節 |
| 2 | 過擬合 | **已封** | 核心原則 9、兩組樣本外、四關。**缺一項**：PnL 集中度 |
| 3 | 存活者偏差 | **已量** | 作者說方向中性可跳過；我們套利線是中性 |
| 4 | 手續費 | **已封** | 逐標的真實 bps（2026-07-28）；**漏 rounding** |
| 5 | 價差 | **有矛盾** | 見下方，這是今天最大的發現 |
| 6 | 市場衝擊 | **未查** | 容量是獨立關（factor-research #10），但**沒有 size 相依的滑價項** |
| 7 | 延遲假設 | **已量但不足** | n=20 給不出 p99.9 |
| 8 | 限價成交假設 | **已封** | §1.02 死在這裡、§0.57b 量過 −0.0612 R |
| 9 | 逆選擇假設 | **已量** | `flow_toxicity` 掛單方 −0.38 bps |
| 10 | 現貨借券 | **不適用** | 我們只做永續；作者也推薦永續 |
| 11 | 資金費 | **已量** | 成本模型有 MEASURED 的資金費。**漏一個新的**，見下 |
| 12 | 提領問題 | **未查** | 跨場館調度的前提 |
| 13 | API 壞掉 | **未查** | **直接威脅 `flow_toxicity` 的結論**，見下 |
| 14 | 無限／免費槓桿 | **已量但偏激進** | 成本模型 `margin_frac` 預設 0.2 = 5x；作者說跨場館 3x，5x 是極限 |
| 15 | tick size | **已量** | 80 個標的零個價差 < 1 tick；tick 中位 0.162 bps |
| 16 | 博弈動態 | **不適用**（還沒做市） | |
| 17 | 假設成交價可成交 | **已封**（新線）／**未查**（§4.65） | 見下 |
| 18 | 成交價的買賣價跳動 | 同 17 | mistake.md 2026-09-11 |
| 19 | OTC 混進主行情 | **未查** | |
| 20 | 洗量／假簿口 | **未查** | 掃描器有 10 個場館，小的那幾個沒測過 |

---

## 今天實查出來的五件事

### (A) 第 5 項：兩台儀器對 Lighter BTC 的價差差 6 倍 —— 所有 Lighter 成本數字暫停引用

同一個場館、同一個標的、**同一個 571 分鐘**：

| 儀器 | 價差 | 半價差 | 簿口狀態 |
|---|---|---|---|
| arb 引擎的 Lighter(hedge) 腿，1 分鐘 | **$6.10** | 0.395 bps | — |
| `lighter_mid`，5 秒、5,797 筆 | **$1.40** | **0.065 bps** | bid 1,664 檔 / ask 1,182 檔、stale 97 ms、**resyncs 0** |

* 時窗不是答案（切到同一窗：引擎 0.3945 vs 全期 0.3947）。
* size-aware 不是答案（引擎的 `best_bid()` 就是 `max(self.bids)`，頂檔）。
* **偏移是單邊的**：引擎的 ask 系統性高 $4.65、bid 只高 $0.85。
  單純的「舊」會是雙邊隨機，單邊系統性偏移指向別的東西。
* 兩邊都比 `lighter_mid` 差的分鐘只佔 18.4%。

**後果**：CLAUDE.md §HFT 的「Lighter BTC 半價差 0.42 bps」來自引擎那一台；
而 §1.28 的「Standard 之下來回成本約 0.02–0.05 bps」來自另一邊的量級。
**兩個都不能引用，直到下面這一關過了。**

**settle 它的那一關**（還沒做）：引擎的 BTC 配對訂閱的是哪個 Lighter
`market_id`，以及它的簿口維護是不是保全部檔位。`HEDGE_VENUES =
("lighter", "lighter-rh", "tradexyz")` 已確認 hedge 腿就是 Lighter。

### (B) 第 1 項：`resample()` 的預設與我們的 bar 標籤慣例

文章說 `pd.DataFrame.resample()` 的預設 `label='left', closed='left'` 是前視來源，
建議一律傳 `label='right', closed='right'`。

**我們有 12 處 `resample()` 沒有明寫 label/closed**（`feature_builder_v2` 4 處、
`poc/bars.py`、`poc/levels.py`、`subhourly/exit_cancelflow_test.py` 等）。

**但不要照著改。** 這個專案的慣例本來就是**左標籤**（CLAUDE.md／mistake.md
2026-07-28：「研究層 bar 以開盤時刻為標籤」），而且 2026-09-03 那條更進一步寫明
「同一根 bar 的不同欄位屬於不同時刻」。把 12 處改成右標籤 = **悄悄平移每一份
歷史資料**，那比現在的狀況危險得多。

**正確的修法是結構性的**：任何「bar 索引的表」join「牆鐘資料」都必須走**同一個
函式**，由它負責位移。我們兩次最嚴重的前視（2026-07-28 錨在 bar 標籤、
2026-09-03 用成交那根的收盤）都是這個 join 沒有經過統一入口。**未做。**

### (C) 第 1 項的反例（這一關是乾淨的，要記下來否則清單沒有分辨力）

`research/poc/levels.py:150-153` 用 `rolling(center=True)` —— 看起來是教科書級前視。
**它是對的**：swing pivot 的定義本來就是「比前後各 N 根都高」，而
`pivots_reference` 那支逐字轉寫給出同一個答案。

而且**確認延遲有強制，還被斷言**：
`levels.py:214` `conf = int(hts[i + PIVOT] + step)`、
`events.py:88` `searchsorted(ts, r.confirmed_at)`、
`events.py:129` **`if not (m["t_sweep"] > m["confirmed_at"]).all(): 報錯`**。

所以 SDV 的 `S`（掃單）那一腳在這個方向上是結構性封住的。

### (D) 第 13 項：「API 壞掉的標的逆選擇最低」—— 這直接威脅我們的做市結論

作者原話：

> 「If you measure the toxicity for every asset on an exchange, you'll find that
> the ones with the **least toxicity** are funnily enough the ones where **the
> API is messed up** or the data feed seems not to work for some reason.」

我們的 `flow_toxicity` 結論是「HL 全體掛單方 −0.38 bps，**中小型標的才為正**」。
**那正好是作者說的形狀**，而我們沒有排除這個競爭解釋。

**要做的**：對那些「為正」的標的，查它們的 API／行情是否正常
（成交與簿口更新率、報價是否長時間不動、是否有長時間的空白）。
在排除之前，**「中小型標的可做市」不可引用**。

### (E) 第 11 項：資金費會污染「時段效應」

作者：

> 「They also affect seasonality strategies since **every time funding pays out
> the price moves** so if you don't adjust for the funding payments you will
> believe that there are strong effects for certain hours of the day and the
> last minute of the hour.」

我們有一個已經撤回的時段發現（2026-09-09「08-15 UTC 比較好」，判為雜訊）。
**以後任何時段／小時效應，必須先扣資金費結算**，否則結算時點會長得像 alpha。

---

## 要加進流程的三項（目前都沒有）

1. **PnL 集中度**（第 2 項）。作者：「If all the PnL was made in 3 large jumps
   ... **3 isn't many**」。凡是報累積損益的地方，一併報
   **最大 1 / 5 / 10 筆佔總損益的比例**。我們報逐幣、逐季、兩半，但沒有這個。
2. **參數敏感度**（第 2 項）。作者：「if you make small-ish changes in your
   parameters and the overall trend of the curve **flips completely** then your
   alpha isn't likely to be robust」。凡是有參數的結果，一併報
   **±1 格參數的結果**（不是掃最佳，是證明它不靠某一格）。
3. **延遲取 p99.9 不取中位**（第 7 項）。作者：「look at **tail latency**
   instead of average ... the latency **conditional on wanting to trade** ——
   and the best proxy for that is the **99.9%** value」。
   我們量到取消往返中位 49.9 ms / p90 54.7 / max 60.3，**但 n=20**，
   而且是在安靜時段量的。n=20 給不出 p99.9，也不是「想交易時」的延遲。

---

## 一句話版

**回測的數字漂亮時，第一個動作是查儀器，不是解讀。** 而這二十項裡
**只有兩種會讓曲線變成直線**：前視，與執行假設（限價成交 ＋ 返佣 ＋
零逆選擇 ＋ 即時成交）。我們兩條已結案的主線各死在其中一種。
