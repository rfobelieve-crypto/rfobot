# PREREG 路徑 A：Lighter 零費率影子執行（凍結 2026-09-05）

依使用者提供的規格書 §路徑 A 填寫。判準逐字沿用；與規格不同處全部列在
「§資料現實與規格修正」。填完、commit、然後才看標籤。

## 假設

分鐘級系統（§1.15，best cell = fade `|ret_60|` top-5%，持有 60m，毛捕捉
**+6.2 bps**，G1 PASS、G2 差約 2 bps）在 Lighter 標準帳戶的真實來回成本
是否 < 6 bps。零手續費之後成本轉移到三處：**延遲下的滑價、有效價差、
跨場館訊號衰減**。

## 已查證的事實（不是二手說法）

- Lighter `/api/v1/orderBooks` 自報 **`taker_fee: "0.0000"`、`maker_fee:
  "0.0000"`**（2026-09-05 直接拉取）。零費率是場館第一手宣告。
- Lighter 的 WebSocket `wss://mainnet.zklighter.elliot.ai/stream` 的
  `order_book/<market_id>` 頻道提供**微秒時間戳的完整 L2**，更新間隔
  次秒級（實測兩筆相差 201 ms）。
- Lighter 的深層 REST 端點（`orderBookOrders` / `orderBookDetails`）擋在
  AWS WAF 人機驗證後。**不繞過機器人偵測**——本線一律走 WebSocket。

## 資料現實與規格修正

**修正 1：原生場館是 Binance 不是 Bitget。** §1.15 的價格序列來自
`orderbook_snapshots_1m`（`exchange='binance'`）。所以 A.2.1 的跨場館基差
比較對象是 **Lighter vs Binance**，不是 Bitget。

**修正 2：延遲成本用 250 ms 網格量，不是單一 300 ms 點。**
錄製器以 **250 ms** 落一列（含 best bid/ask 與 1/5/10 bps 內累積名目），
所以 `mid(t0+300ms)` 用最接近的兩列線性內插，同時報 `mid(t0+250ms)` 與
`mid(t0+500ms)` 作上下界。**若上下界跨越判準門檻，本項判 INCONCLUSIVE，
不取有利的那一端。**

**修正 3：Maker 臂的佇列位置只能近似。** 規格 A.2.3 要用 L2 差分重建佇列。
WS 給的是快照不是逐筆委託事件，所以 `queue_ahead` 只能取該價位在 t0 的
掛單量，成交判定用「該價位被穿過」的保守規則（跟 §1.17 同一套，可比）。
**這會高估成交率**，因此 A2 的 fill_rate 門檻視為上界，寫在報告裡。

**修正 4：size_target 用實際資金算。** 帳戶權益約 $800、名目 2 倍
→ `size_target = $1,600`。walk-the-book 對這個 size 算有效價差。
（規格明講「用你的實際資金，不用假設值」。）

## 量測（其餘逐字照規格 A.2）

A.2.1 基差與延遲：`basis(t0) = mid_L − mid_B`；
`basis_lag = argmax_k corr(Δmid_L(t), Δmid_B(t−k))`，k ∈ [−5s, +5s]。
A.2.2 taker 臂：`rt_cost_taker = entry_cost + exit_cost`，含
`eff_spread(size_target)/2` 與延遲項（按訊號方向對齊符號）。
A.2.3 maker 臂：fill_rate、`markout_τ`（τ ∈ 5s/30s/60s/5m，**從成交價與
成交時刻量**）、未成交的反事實。
A.2.4 訊號 markout：`τ ∈ 1m/5m/15m/30m/60m`，與 Binance 上的原生 +6.2 bps 比較。

## 判準（逐字沿用）

| 項目 | PASS | REJECT |
|---|---|---|
| A0 訊號轉移 | basis 日內 σ ≤ 6 bps | > 6 bps → 改在 Lighter 價格上重跑 G1，不進 A.2.2 |
| A1 Taker | `mean(rt_cost_taker) + 2·SE < 6 bps` ∧ A.2.4 的 60m markout ≥ +5 bps | 任一不符 |
| A2 Maker | fill_rate ≥ 40% ∧ `mean(markout_60m\|成交) > 0` ∧ 反事實減實現的差 CI 涵蓋 0 | fill_rate < 40% ∨ markout < 0 ∨ 反事實差顯著 > 0 |

A1 或 A2 任一 PASS → 進 2 週真錢小額 forward，部位上限 = 帳戶 5%。
**INCONCLUSIVE**：4 週內訊號分鐘 < 60 → 延長記錄期，判準不動。
**檢定總數 6**（A0 一、A1 二、A2 三），Benjamini-Hochberg。
**成本分層**：taker 8 / maker 4 / 零費 + 滑價 三層都報。

## 先驗（寫在看到資料之前）

§1.17 已量過：Binance 現貨 BTC 1 分鐘尺度的無條件掛單 markout_60 =
**−3.1 bps**，逆選擇在成交那一分鐘內就完成。Lighter 約 82% 成交量來自
專業做市商（外部說法，未查證），所以 **A2 maker 臂的先驗是 REJECT**。
A1 taker 臂是真正的未知：零費之後只剩價差與延遲，先驗五五開。

## 停止條件

- A0 REJECT → 停 A.2.2–A.2.4，改跑「Lighter 價格上的 G1 重驗」。
- Lighter 宣布標準帳戶收費 → 經濟前提消失，凍結。**這條線有時效窗。**
- 錄製器連續 24h 無資料 → 先修錄製器，不判任何結論。

## 不做的事

不繞過人機驗證。不因為 taker 臂差一點就放寬 6 bps。不把 250/500 ms 的
上下界取有利那端。不在 A0 未過時直接跑 A.2.2。

**資料期間**：自 2026-09-05 起 ≥ 4 週。**簽名**：2026-09-05，commit 後才看標籤。
