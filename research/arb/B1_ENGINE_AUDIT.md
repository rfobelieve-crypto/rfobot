# B1 逐行審：現有下單路徑（2026-09-04）

> 對象：`../entropy-arb/entropy_arb/venue_hl.py` / `venue_lighter.py` 的 `send_taker`，
> 以及 `engine.py` 的執行、對帳、對沖三條路徑。
> **不改程式，只讀 + 記問題。** CLAUDE.md 硬規則：「下單程式碼要逐行審」。
> 判準是那對兄弟 $1.1M 的病灶：**資料過期 → 以為沒成交 → 一直補單**。

## 結論先講

**核心設計是對的，而且對得很刻意。** 三個最危險的地方都已經有防護，且註解寫明了
為什麼。找到 **4 個真實缺口**（見 §3），沒有一個是「會一直補單」那種致命型，
但其中兩個在 $50 規模就會咬人。

---

## 1 三個關鍵路徑，逐一驗

### 1a 送單：`send_taker`（HL）

```
簽章失敗           → send-failed, filled=0, unresolved=False   ✅ 明確
HTTP 4xx / 429    → send-failed（429 標 RATE_LIMITED）        ✅ 明確
HTTP 5xx / 逾時    → unresolved=True                            ✅ 不假設沒成交
回應 "filled"     → filled_base = totalSz, avg_px               ✅
回應 "error" 且含 "could not immediately match" → canceled      ✅ IOC 沒吃到不是錯
回應 "resting"    → unresolved=True                             ✅ IOC 不該 resting，存疑
```

**未決之後**：用 `cloid` 輪詢 `orderStatus` 直到 `settle_timeout`；
`filled = origSz − sz`（剩餘量反推成交量）。逾時 → `status="timeout",
unresolved=True`。

> **這一段就是 $1.1M 的分界線，而它走對了方向**：不確定時**不假設沒成交**，
> 而是回報 unresolved 讓上層處理。假設「沒成交所以再送一次」正是兄弟那台機器人
> 做的事。

### 1b 未決之後：`_execute_locked`

```python
if unresolved:
    self._reconcile_evt.set()      # 觸發對帳，不是補單
else:
    await self._maybe_hedge()
```

✅ **未決 → 去讀真實部位，不下單。**

### 1c 對帳：`_reconcile_venue`

```python
delta = r - v.position          # 鏈上 vs 本地
if abs(delta) > 1e-12:
    v.cash -= delta * mid
    v.position = r              # 採信鏈上
```

✅ **對帳只採信真相，不下修正單。** 這是整份程式最重要的一行。

還有三層保護，註解都寫了理由：
- `RECONCILE_GRACE_SEC = 5.0`——剛成交的場館跳過（「Lighter 的 REST 帳戶狀態落後
  它的 ws 結算，覆蓋剛交易過的場館會『還原』舊部位並觸發幻影對沖震盪」）
- 抓不到部位連續 3 次 → `venue_down`，**交易暫停**，之後每 `venue_probe_sec` 探一次
- 啟動時 `strict=True`：抓不到起始部位就**大聲失敗**，不帶著未知狀態開跑

### 1d 對沖：`_maybe_hedge` → `_hedge`

這是**唯一會主動下修正單**的地方，所以逐條檢查它的守衛：

```python
net = sum(v.position for v in venues)         # 兩腿相加
if abs(net) > net_tolerance_base: hedge()
```

| 守衛 | 有沒有 |
|---|---|
| 只做 `reduce_only`（不會開新倉） | ✅ |
| 場館 down 或簿口不新鮮就跳過（`is_fresh(staleness_sec)`） | ✅ **這是防幻影補單的關鍵** |
| 場館鎖已被持有就跳過（不與執行搶） | ✅ |
| 量小於 `min_base` / 名目小於門檻就跳過 | ✅ |
| 滑價保護（`hedge_slippage_bps`） | ✅ |
| 補單失敗／未決 → 記 error、**再觸發對帳**，不重試 | ✅ |
| 對沖不到（低於最小量）→ **carry，等下次對帳** | ✅ 不硬做 |

### 1e 熔斷

`consec_errors >= max_consecutive_errors` → `self.halted = True`，
`_scan` 與執行路徑開頭都檢查 `if self.halted: return`。✅

---

## 2 已經存在的守衛總表

| 風險 | 防護 | 在哪 |
|---|---|---|
| 資料過期下單 | `book.is_fresh(staleness_sec)`（執行前、對沖前各一次） | engine 384/548 |
| 剛成交就對帳 → 幻影差額 | `RECONCILE_GRACE_SEC=5` + 場館鎖 | engine 600 |
| 場館 API 掛掉 | 連 3 次抓不到 → `venue_down`，暫停交易 | engine 638 |
| 啟動狀態未知 | `strict=True` 大聲失敗 | engine 620 |
| 連續錯誤 | `max_consecutive_errors` → HALT | engine 506 |
| 部位過大 | `cap_usd` headroom 檢查 | engine 283 |
| 下單過頻 | `orders_per_min` 滑動 60s 預算 | engine 98 |
| 被限流 | `_mark_limited` + `rate_limit_pause_sec` | engine 109 |
| 補單開新倉 | `reduce_only=True` | engine 569 |

**這比我預期的完整。** 作者顯然踩過幻影對沖那個坑（註解直接寫了 Lighter 的 REST
落後 ws）。

---

## 3 找到的四個缺口

### G1 ⚠ 沒有「累積差額硬上限」

`_hedge` 每次只對沖**當下**的 net，但**沒有任何地方限制 net 可以長到多大**。
那對兄弟的參數裡有 `maxDelta: 800`——「若兩腿差額達到 800 股，機器人停止交易此市場」。
這裡沒有對應物。

**現況為什麼還沒出事**：`cap_usd` 限制單邊部位、`reduce_only` 讓對沖不會加倉、
`max_consecutive_errors` 會 HALT。但這三個都是**間接**的——沒有一條直接說
「差額超過 X 就停」。

**建議（B4 的第一項）**：加 `max_net_base`，`_maybe_hedge` 開頭檢查，
超過就 `self.halted = True` 並告警。**這是五個 kill switch 裡最該先加的。**

### G2 ⚠ `settle_timeout` 逾時後回 `filled_base = 0.0`

```python
return {"status": "timeout", "filled_base": 0.0, ..., "unresolved": True}
```

`unresolved=True` 會觸發對帳（正確），但 `filled_base=0.0` 是**一個猜測**——
真實成交量未知。如果上層有任何地方讀 `filled_base` 而忽略 `unresolved`，
就會低估部位。

**查證結果**：`_execute` 讀了 `binfo["filled_base"]`／`sinfo["filled_base"]`
（engine 464-465）**然後才**檢查 `unresolved`（489）。中間那段用 fill 算了
`fill_edge` 並更新 `v.position`——**逾時的情況下 position 會被少加**。
好消息是後面 `unresolved` 會觸發對帳把它修正回來；壞消息是**在對帳跑完之前，
本地部位是錯的**，而那段時間裡 `_scan` 可能會下新單。

**建議**：`unresolved` 時不要更新 `v.position`，直接等對帳。或至少讓 `_scan`
在有未決時暫停。

### G3 ⚠ 沒有 maker 路徑（已知，B3 要補）

只有 `send_taker`。而成本分析說**吃單門檻 13.5 bps、掛單 4.5 bps**，
掛單是唯一活得下來的模式。補的時候要注意：**掛單的未決狀態比吃單複雜**
（resting 是正常狀態，不是異常），現在 `_parse` 把 `resting` 當 `unresolved`
是為 IOC 寫的，maker 路徑不能沿用。

### G4 ⚠ 兩條 Lighter 鏈共用同一組憑證

`config.py` 兩處都讀同樣的 `LIGHTER_ACCOUNT_INDEX` / `LIGHTER_API_KEY_INDEX` /
`LIGHTER_API_PRIVATE_KEY`，而 mainnet 與 RH 是**不同鏈、不同帳號**。
`NVDA_LL`（兩腿都是 Lighter）**現在根本簽不了兩腿**。（已記於 §1.08 §7，B2 要修。）

---

## 4 沒有找到的問題（明寫，免得下次重查）

- ❌ 沒有「以為沒成交就重送」的邏輯——這是最擔心的那個，**沒有**
- ❌ 沒有在對帳裡下修正單
- ❌ 沒有無上限重試
- ❌ 對沖沒有開新倉的路徑（`reduce_only=True` 寫死）
- ❌ 沒有裸露的 `except: pass`（例外都有記錄或分類）

---

## 5 下一步（B2→B4 的順序因此確定）

1. **B2 憑證缺口**（G4）——不修 `NVDA_LL` 跑不起來
2. **B4 的第一個 kill switch：`max_net_base`**（G1）——比 maker 路徑更優先，
   因為它擋的是「差額無聲長大」，那是唯一一種會把 $300 全部輸掉的方式
3. **B3 maker 路徑**（G3）——注意 resting 語意
4. G2 的修正（unresolved 時不動 position）可以跟 B4 一起做
