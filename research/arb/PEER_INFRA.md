# 別人的套利基礎設施：讀原始碼之後的對照（2026-09-04）

> 使用者要求：「去研究其他套利程式他們的基礎設施」＋「會以風險為優先」。
>
> **這份文件只寫我親自下載並讀過的程式碼。** 使用者貼進來的專案清單是另一個
> 模型整理的線索表——我拿它當**待查名單**，不是當事實。清單裡我實際核對過
> 的只有 Hummingbot 那幾支檔案（下面每一條都附行號）；其餘（Harjus、
> barbotine、三角套利那幾個）我只做了粗略掃描或完全沒讀，**下面會明講哪些
> 是我沒查的**。

---

## 一、我實際讀了什麼

| 檔案 | 行數 | 來源 |
|---|---|---|
| `strategy_v2/executors/arbitrage_executor/arbitrage_executor.py` | 357 | Hummingbot master |
| `strategy_v2/executors/xemm_executor/xemm_executor.py` | 371 | 同上 |
| `strategy/cross_exchange_market_making/…py`（v1） | 1803 | 同上（只做關鍵字對照） |
| `core/utils/kill_switch.py` | 84 | 同上 |
| `nelso0/barbotine-arbitrage-bot/main.py` | 349 | 只掃了關鍵字 |

**沒讀**：Harjus（C++／kernel bypass）、hzjken 的 CPLEX 多邊、fundingarb、
統計套利那幾個。理由見第五節——它們解的不是我們現在的問題。

---

## 二、最重要的一個發現：**開源套利框架的風控比想像中薄很多**

Hummingbot 是清單裡基礎設施最完整的一個（連接器抽象、訂單簿追蹤、
Docker、paper mode 都有）。但把它的**套利執行器**當風控來讀，結果是這樣：

### 2.1 一腿成交、另一腿失敗 → **沒有平倉機制**

`arbitrage_executor.py:311` 的失敗處理，全文如下：

```python
def process_order_failed_event(self, _, market, event):
    if self.buy_order.order_id == event.order_id:
        self.place_buy_arbitrage_order()      # 重下「同一側」
        self._cumulative_failures += 1
    elif self.sell_order.order_id == event.order_id:
        self.place_sell_arbitrage_order()
        self._cumulative_failures += 1
```

失敗就重下同一側，累積超過 `max_retries`（預設 3）之後：

```python
if self._cumulative_failures > self._max_retries:
    self.close_type = CloseType.FAILED
    self.stop()                                # arbitrage_executor.py:174-176
```

**`stop()` 之後那條已成交的腿就那樣裸著。** 全檔沒有任何反向平倉、
沒有 `reduce_only`、沒有比較兩腿成交量的地方。

### 2.2 部分成交 → 執行器會卡在關機狀態

`check_order_status` 只認「兩腿都 `is_filled`」：

```python
if self.buy_order.order.is_filled and self.sell_order.order.is_filled:
    self.close_type = CloseType.COMPLETED
```

買腿全成、賣腿成 40% 的情況下，`is_filled` 為 False，而
`_cumulative_failures` 只在**失敗事件**才加（部分成交不是失敗事件）
→ 執行器**永遠留在 SHUTTING_DOWN**，帶著一個沒對沖乾淨的部位。

### 2.3 兩腿都是無價格上限的市價單

`place_buy_arbitrage_order` / `place_sell_arbitrage_order` 都是
`order_type=OrderType.MARKET`。薄簿口上這是**無上限的滑價**。

### 2.4 kill switch 只有一條 PnL 門檻，而且不平倉

`core/utils/kill_switch.py:50`：

```python
if (self._profitability <= self._kill_switch_rate < Decimal("0.0")) or \
        (self._profitability >= self._kill_switch_rate > Decimal("0.0")):
    ...stop the bot
```

一個數字，觸發後停機。**沒有部位差額上限、沒有簿口過期偵測、
沒有毛曝險上限、沒有停機後的平倉。**

---

## 三、maker 路徑（XEMM）：兩個直接打到我們 B3 的坑

我們唯一剩下的引擎缺口就是 B3 掛單路徑。讀 XEMM 執行器讀到兩件事，
**兩件都是我們寫 B3 時會很自然犯的錯**：

### 3.1 撤單是射後不理，而且**當下就把訂單忘掉**

`xemm_executor.py:234-235`（另一處 238-239 相同）：

```python
self._strategy.cancel(self.maker_connector, self.maker_trading_pair,
                      self.maker_order.order_id)
self.maker_order = None          # ← 沒等撤單確認
```

然後成交事件的處理是（`:273`）：

```python
def process_order_completed_event(self, ...):
    if self.maker_order and event.order_id == self.maker_order.order_id:
        self.place_taker_order()          # 對沖在這裡才發生
```

**撤單輸掉競速的話**：`maker_order` 已經是 `None` → 條件為 False →
**對沖腿永遠不會送出**，而執行器接著去掛一張新的 maker 單。
一筆裸的掛單成交就這樣不留痕跡地留下來。

> 這跟兩兄弟那個 $1.1M 的病是同一族，只是換了個方向：他們是
> 「假設沒成交就重送」，這裡是「假設沒成交就忘掉」。**共同點是本地狀態
> 先於交易所狀態做了假設。**

### 3.2 完全沒有處理部分成交

`xemm_executor.py` 全檔 `OrderFilledEvent` 出現 **0 次**，只有
`OrderCompletedEvent`。掛單成交一半 → 這一半**不會**觸發對沖，
要等它全部成交才會。

有意思的是：**舊的 v1 策略檔（1803 行）有 11 處處理 fill 事件。**
v2 執行器重寫時把這個安全性質弄丟了。

### 3.3 但 XEMM 有一條我們該偷的（已偷，今天上線）

`xemm_executor.py:236`：

```python
elif self._current_trade_profitability - self._tx_cost_pct > self.config.max_profitability:
    ...Cancelling order.
```

**獲利率變得「太好」時撤單。** 邏輯是：邊際突然變得很棒，通常代表
**你自己**才是那個過期的報價，你正要被人挑走。

我們只有下限 `threshold_bps`，**沒有上限**。而對這條線這不是理論風險——
標的是代幣化股票，我們自己就有兩個（`io:OAI`、`io:EWY`）在錄製途中下市。
**停牌／下市的標的留下的是又寬又舊、而且 REST 還會照常回應的簿口，
讀起來就是巨大溢價。** `premium_persist_sec` 擋得掉一跳的假訊號，
擋不掉「持續是錯的」簿口。

→ 今天已實作 `max_edge_bps`（commit `d9b7e4a`）：實測帶寬 2–15 bps，
讀到幾百 bps 一律當簿口壞掉。**只跳過不停機**——亂設的上限如果會停機，
比沒有更糟。

---

## 四、逐項對照：他們的風控 vs 我們的

| 風控項目 | Hummingbot 套利/XEMM | 我們（`entropy-arb`） |
|---|---|---|
| 單筆／單場館名目上限 | ✅ 餘額檢查 | ✅ `cap_usd` + `_headroom` |
| 兩腿差額上限 | ❌ 完全沒有 | ✅ `max_net_base` → HALT |
| 毛曝險絕對上限 | ❌ | ✅ `max_gross_usd` |
| 單場次虧損下限 | ✅ kill switch（單一數字） | ✅ `max_daily_loss_usd` |
| 簿口過期擋單 | ❌ | ✅ `is_fresh(staleness_sec)` |
| 簿口持續過期 → 停機 | ❌ | ✅ `max_consecutive_stale` |
| 進場價格上限 | ❌ 市價單無上限 | ✅ 兩腿都是帶 `limit_px` 的 IOC |
| **邊際「太好」的上限** | ✅ XEMM 有（maker 側） | ✅ **今天補上** `max_edge_bps` |
| 單一場館故障隔離 | 部分（連接器層） | ✅ `venue_down` + 探測回復 |
| 限流反應 | 連接器層 | ✅ `_mark_limited` + 退避 |
| 下單速率預算 | ✅ | ✅ `orders_per_min` 滑動窗 |
| 部分成交處理 | ❌ v2 執行器沒有 | ✅ `matched` + 對帳 |
| **斷腿之後自動平回去** | ❌ 停機留裸倉 | ✅ 停機後 reduce-only 自救 |
| **自救永不永久放棄** | — | ✅ 有進展就重置預算＋慢速重試 |
| 旁路（錄價）死亡可見 | ❌ | ✅ `add_done_callback` |
| **風控沒武裝就拒絕實盤** | ❌ | ✅ **今天補上** |

**誠實的結論：在風險這一軸上我們已經領先 Hummingbot，而且不是險勝。**
這跟直覺相反（它是幾十億美元交易量的框架），但原因說得通——
**它是一個「通用做市／交易框架」，套利執行器只是其中一個模組；
我們是一個只做一件事、而且已經被自己的紀律文件打過三次的專案。**

反過來說，我們沒有的是它的**廣度**：連接器抽象（幾十個交易所）、
paper mode、CLI、熱更新參數。那些是產品化的東西，不是風險的東西。

---

## 五、我刻意不去讀的，以及為什麼

**Harjus（C++、DPDK/F-Stack kernel bypass、機房選址）**——
這是真正的 HFT 基礎設施，但**不是我們的 regime**。我們量過：
機會壽命中位 5–15 分鐘、掃描週期 180 秒、本機到場館往返 32–95 ms。
進場慢 100 ms 不會錯過一個 5 分鐘的機會（`HOTPATH_AUDIT.md` 第 1、8 條）。
**在延遲上投資是解一個我們還沒有的問題。**

真正會咬人的延遲在**撤單**（掛單被吃之前撤掉），那是 B3 上線後要量的
第一件事，到那時再回來看 Harjus 也不遲。

**多邊／三角套利（Bellman-Ford 負環、CPLEX）**——
數學上漂亮，但我們的問題從來不是「找不到路徑」，是**成本吃掉價差**
（`COST_INVENTORY.md`：七桶成本，量到 43%，全家族在所有情境下皆為負）。
換一個更會找路徑的演算法，不會改變費率結構。

**barbotine（taker-taker、免轉帳）**——模型跟我們一樣，值得看，
但它把餘額狀態存在 `real_balance.txt` 這種純文字檔（`main.py:37-39`）。
**把本地狀態存在會跟交易所分歧的地方**，正是我們花了整個 B4 在防的事。
架構上沒有可學的。

---

## 六、這次挖出來、直接進了程式的兩條

1. **`max_edge_bps`（偷自 XEMM）** — 邊際太好＝簿口壞了。
   `commit d9b7e4a`，只跳過不停機，狀態行印 `refused xN`。
2. **`_require_armed_risk_block`（自己查出來的，比較難看）** —
   對照別人的設定檔時，回頭查了自己的：**九個實盤設定檔沒有任何一個有
   `risk:` 區塊**，所以今天做的開關有四個跑在「預設關閉」。
   目前九個都是 `--record-only` 錄價器、沒有真錢，但
   **「開關存在」和「開關武裝」是兩件事，而運行中的系統看不出差別。**
   美元計價的那幾個沒有可辯護的通用預設（取決於帳戶大小），
   所以修法不是給更好的預設，而是**實盤模式沒設就拒絕啟動並印出缺哪幾項**。

---

## 七、B3 掛單路徑的規格（讀完 XEMM 之後改寫）

原本的 B3 只寫「post-only 下單、撤單、5 秒逾時」。讀完之後補三條**必須**：

1. **撤單前不得清掉本地訂單狀態。** 訂單只能由「交易所確認已撤」或
   「成交回報」兩件事之一移除。撤單送出後訂單進入 `cancelling` 而不是消失。
   —— 這是 XEMM 3.1 那個坑的直接對策。
2. **必須處理部分成交。** 掛單成交多少就對沖多少，不能等全部成交。
   `resting` 目前被當成 `unresolved`（對 IOC 正確、對 maker 錯誤），
   B3 要先修這個語意。
3. **撤單延遲要有自己的預算格**，跟 `staleness_sec` 並列，
   逾時未確認撤單 → 當作可能已成交處理（悲觀），不是當作已撤（樂觀）。

---

## 八、仍然開著的（不因為這次研究而改變）

- **波動熔斷**——我們自己承認的最弱一環。Hummingbot 也沒有，
  所以這次研究**沒有給我們現成答案**，得自己設計。
- **強平偵測**——目前只能靠對帳事後發現。
- **B3 掛單路徑**——上面第七節就是它的新規格。
