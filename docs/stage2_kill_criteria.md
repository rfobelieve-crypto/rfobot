# Stage 2 Kill Criteria — V7 OKX Testnet Executor

> Version: 0.1 (initial draft, 2026-05-25)
> Owner: rfo
> Review cadence: 每 Stage 升級前 + 任何一次 kill trigger 後

## 文件性質

這份文件是 **operational runbook**,不是哲學討論。每一條規則:
1. 必須能用程式自動判定(grep-able threshold)
2. 必須有明確的「觸發後做什麼」
3. 觸發後**不靠紀律**,程式自動降階

任何「我覺得這次應該不算」的念頭 = 這條規則需要修,不是這次破例。

---

## 1. 進入 Stage 2 的前置條件(回顧)

所有條件必須**同時**滿足才能啟動 testnet executor:

| 條件 | 標準 | 檢查方式 |
|---|---|---|
| Paper trades(穩定版本)| ≥ 100 筆 | `/paper-perf` Stage 2 進度條 |
| 4 週 rolling net bps | ≥ +5 bps | `/paper-perf` weekly_4w |
| LONG guard | n≥20 且 avg_net > 0 | `/paper-perf` 整體 多空拆解 |
| Regime alert(30 天)| 無 🟠 觸發 | `/paper-perf` alerts |
| 模型未在 transition window | 重訓後過 ≥ 7 天 | `v7_paper_executor.py` |
| stage2 前置作業完成 | 本文件 + checklist 全簽到 | 人工 |

**任何一條未通過 → 不准啟動 OKX executor**,即使其他條件全綠。

---

## 2. Kill Triggers(分類)

### 2A. 連線 / 對帳類(最嚴重,優先級最高)

| # | 觸發條件 | 行動 | 為什麼 |
|---|---|---|---|
| A1 | WS disconnect > 5 min(單次) | **降階** + 強制平倉 + alert | 長時間斷線不知道倉位狀態,任何假設都危險 |
| A2 | 連續 3 次 reconnect 失敗 | **降階** + alert | 網路/key 問題,不能繼續 |
| A3 | WS heartbeat 失敗 30s 內無回應 | halt 下新單 + alert,5 min 內不恢復 → 降階 | 心跳是連線健康的 baseline |
| A4 | 任何一次對帳不一致(本地 DB vs OKX 真實倉位) | **立即 halt 下新單** + critical alert,人工確認 | 對帳不一致是「複雜 bug 的最後一道防線」 |
| A5 | 重複下單事件 ≥ 1 次(同一 signal 觸發 ≥ 2 個 entry order) | **降階** + 強制平倉 + critical alert | 最嚴重 bug 類型,單次發生即降階 |
| A6 | 系統 crash 期間 stop 被觸發,重啟後本地 DB 未補正 | **降階** | reconciler 失效,等於沒有 reconciler |

**對帳一致定義(A4)**:
- 每 cycle 開頭呼叫 OKX REST 拉所有 open positions
- 與 `v7_okx_positions`(或同等 schema)逐筆比對
- 差異判定:`abs(local_size - okx_size) > 0.001 contracts` OR `direction 不一致` OR `local 認為有單 / OKX 沒有(或反之)`

### 2B. 訂單 / 執行類

| # | 觸發條件 | 行動 | Window |
|---|---|---|---|
| B1 | Order rejection rate > 5% | **降階** + alert | 7-day rolling, n≥20 |
| B2 | Amend-algo-order 失敗率 > 10% | **降階** + alert | 7-day rolling, n≥20 |
| B3 | Partial fill 後 30s 內系統未決策(無 close / amend / hold log) | **降階** + alert | 任一次 |
| B4 | Algo stop order 在 entry fill 後 5s 內未掛上 | halt 下新單,平掉當筆,alert | 任一次 |
| B5 | Stop trigger 後對帳發現倉位仍在 | **降階** + critical alert | 任一次 |
| B6 | OKX returns `instId not found` / `tdMode invalid` | **降階** + alert | 任一次(代表 config 錯) |

**為什麼 B3 是 30s 而不是更長**:
- partial fill 不該需要 30s 思考。系統應該在 < 5s 決定:等剩餘 fill / 平掉已 fill / 改 order。
- > 30s 沒決策 = 邏輯有 bug 或 race condition。

### 2C. 監控 / 告警類

| # | 觸發條件 | 行動 | Window |
|---|---|---|---|
| C1 | Silent failure alert ≥ 3 次 | **降階** + alert | 30 day rolling |
| C2 | API 401/403(key 異常)| halt 下新單 + alert,24h 內未解決 → 降階 | 持續 |
| C3 | API key 過期前 7 天未換 | warning(不降階,人工換)| 持續 |
| C4 | Telegram 告警通道連續 24h 推不出去 | **降階**(沒告警等於沒監控)| 持續 |
| C5 | NTP drift > 5s(本地 vs OKX server)| halt 下新單 + alert | 任一次 |
| C6 | NTP drift > 30s | **降階** + alert | 任一次(OKX 會拒單) |

### 2D. 成本 / 性能類

| # | 觸發條件 | 行動 | Window |
|---|---|---|---|
| D1 | 實測 slippage round-trip vs paper 假設(8 bps)偏差 > 5 bps | halt + 校準 paper 假設,連續 7 天偏差不收斂 → 降階 | 7-day rolling, n≥10 |
| D2 | 7-day rolling net bps < -10 bps | **降階** + alert | 7-day, n≥10 |
| D3 | Testnet 連續 7 個 daily PnL 全紅 | **降階** | 7 個連續交易日 |
| D4 | 單筆 trade slippage > 30 bps(極端事件)| 記錄 + alert,連續 2 次 → halt 下新單 | 任一次 |

**為什麼 D2 是 -10 bps 不是 -5**:
- Testnet 流動性比 live 差,本來會有 slippage 偏差
- -5 bps 太緊,容易 false trigger
- -10 bps 代表真實問題(策略本身在 testnet 失效)

### 2E. 結構 / 配置類

| # | 觸發條件 | 行動 |
|---|---|---|
| E1 | tdMode != "cash" 且 leverage > 1 | **立即 halt**(違反 Stage 2-3 規則) |
| E2 | posMode 不是預期值(預期 net_mode) | **立即 halt** |
| E3 | 偵測到 transferring funds API call | **立即 halt** + critical alert(executor 不應有此權限) |
| E4 | API key 權限包含 withdraw | **立即 halt**(key 權限配置錯誤) |

E1-E4 是 "should never happen" 類型 — 觸發即代表設計缺陷,不只是 runtime 問題。

---

## 3. Hard Freeze(連 Stage 1 paper 都要停)

下列情況比降階更嚴重 — Stage 2 executor + paper executor 都要暫停,因為**模型本身可能失效**:

| # | 觸發條件 | 行動 |
|---|---|---|
| F1 | Model 預測連續 2 個 cycle 失敗(silent failure alert 既有邏輯)| 全停 + critical alert |
| F2 | Paper equity drawdown ≥ -15%(既有 kill switch)| 全停 + critical alert |
| F3 | Regime alert + LONG/SHORT **同時** < 35% WR(14 day)| 全停 + critical alert |
| F4 | IC(30-day, magnitude model)< 50% × IC(90-day)| Warning;若再下降 30% → 全停 |
| F5 | Walk-forward IC 在最近 cohort 變號 | 全停 + 人工檢查 |

F3 與既有 regime-flip 預警不同:
- 預警(🟠):**單邊** WR < 35% → 提醒
- Hard freeze:**雙邊** WR < 35% → 不只是一邊壞,是整個方向訊號都失效

---

## 4. 觸發後流程(Demotion / Halt)

### 4.1 自動執行(無人工介入)

降階(降回 Stage 1)觸發時:
```
1. OKX executor 設 status = "DEMOTED"
2. 取消所有 OKX 未成交 algo orders(包含 trailing stop)
3. Market close 所有 open positions
4. 對帳:確認 OKX 端 0 open positions, 0 active orders
5. Paper executor 繼續跑(shadow 模式)
6. 推 Telegram critical alert,內容:
   - 觸發條件編號(如 A4)
   - 觸發時間
   - 平倉結果(每筆 entry/exit/pnl)
   - 當前 paper equity
7. 寫入 stage2_kill_log table(timestamp / trigger_id / context / actions_taken)
```

Halt(只停下新單,不平倉)觸發時:
```
1. OKX executor 設 status = "HALTED"
2. **不**取消 open positions 的 trailing stop(它們繼續保護倉位)
3. **不**主動平倉(等 trailing stop 或 signal exit 自然結束)
4. 不接受新 entry signal
5. 推 alert
6. 寫入 stage2_kill_log
```

### 4.2 人工介入(降階後)

降階後 **不可** 自動回到 Stage 2。必須:
1. 識別 root cause 並寫入 `mistake.md`
2. 寫測試覆蓋該 root cause(failing test → fix → passing test)
3. PR review(自我 + AI review)
4. Re-deploy
5. **重新跑 testnet checklist**(`stage2_testnet_checklist.md` 全項)
6. testnet 連續 7 天 0 kill trigger 且對帳 100% 一致
7. 人工 sign-off(寫在 `docs/stage_progression_log.md`)→ 才能重啟 OKX executor

---

## 5. 重新進入 Stage 2 的條件

降階後重啟 OKX executor 的硬性條件:

| 條件 | 說明 |
|---|---|
| Root cause documented | `mistake.md` 新增條目,含 what/why/correct approach/rule |
| Test 覆蓋 | 該失敗場景有自動測試,在 CI 上 pass |
| Testnet clean 7 days | 連續 7 個交易日 0 kill trigger |
| Reconciliation clean 7 days | 連續 7 天每次對帳一致(任何一次不一致重置) |
| Sign-off | `stage_progression_log.md` 記錄重啟時間 + 觸發原因 + 修復摘要 |
| Code review | 修復 code 經過 review(self + AI) |

**所有條件必須全綠**。少一條都不行。

---

## 6. 「我覺得這次不算」的處理流程

如果 trigger 觸發但你直覺覺得「這只是 transient,不該降階」:

1. **規則不破例**:程式已經降階,不撤銷
2. 寫進 `mistake.md` candidate 條目(暫不入正式)
3. 在 testnet 跑 7 天觀察該 trigger 是否再觸發
4. 不再觸發 → 該 trigger 規則可能太緊,**修規則**(改 threshold 或加 context filter),不是破例
5. 又觸發 → 規則是對的,你的直覺是錯的

**規則只能在「無觸發狀態」下修改**,不能在觸發當下為了 unblock 自己而修。

---

## 7. Review Cadence

| 時機 | 動作 |
|---|---|
| 任何 kill trigger 觸發後 | 7 天內 review 該規則:threshold 對嗎?需要更緊還是更鬆? |
| Stage 2 → Stage 3 升級前 | 全部 trigger 重新評估(Stage 3 是 live $100,部分 trigger threshold 要收緊) |
| Stage 3 → Stage 4 升級前 | 全部 trigger 重新評估(Stage 4 有真金額,threshold 要更緊) |
| 每季度 | 即使無觸發,也 review 一次,確保 threshold 跟得上系統演化 |

---

## 8. 附錄:trigger 編號索引(快速查表)

```
A1-A6: 連線/對帳類
B1-B6: 訂單/執行類
C1-C6: 監控/告警類
D1-D4: 成本/性能類
E1-E4: 結構/配置類(should never happen)
F1-F5: Hard Freeze(連 Stage 1 也停)
```

Stage 2 executor 程式碼裡每個 kill check 必須在 docstring 標註對應編號,方便日後 audit。範例:
```python
def check_reconciliation(local, okx):
    """Kill trigger A4: 對帳不一致 → halt 下新單 + critical alert.

    See docs/stage2_kill_criteria.md#2a
    """
    ...
```
