# Stage 2 Testnet Validation Checklist — OKX Demo Trading

> Version: 0.1 (initial draft, 2026-05-25)
> Owner: rfo
> Companion doc: `docs/stage2_kill_criteria.md`(本文件每項都映射到 kill trigger ID)

## 文件性質

這份是 **入場考試**,不是「試試看會發生什麼」。Stage 2 → Stage 3 的進階條件:

> **全部項目通過 + 連續 7 天 0 kill trigger + 100+ testnet trades + 對帳 100% 一致**

任何一項未通過 → 不進 Stage 3。「我覺得這項不重要」= 跳過 → 等 Stage 3 用真錢時被該項殺。

---

## 使用方式

每一項格式:
```
T-XX-NN [Auto|Semi|Manual]  描述
  觸發方式:  怎麼讓這個 case 發生
  通過判定:  什麼算 pass(可量化)
  映射:      對應 kill_criteria.md 哪條 trigger
  狀態:      [ ] / [x] / [SKIP+reason]
```

**[Auto]** = pytest 可覆蓋,寫測試,CI 跑
**[Semi]** = 需要 script 觸發 + 人工觀察 log/dashboard
**[Manual]** = 需要 OKX web UI 操作或網路層 hack(如拔網線)

執行紀錄寫在 `docs/stage2_testnet_results.md`(進入 testnet 才開,每項一行記時間+結果)。

---

## 進入 testnet 的前置條件(回顧 stage2_kill_criteria.md §1)

- [ ] Stage 1 paper trades ≥ 100
- [ ] 4 週 rolling net ≥ +5 bps
- [ ] LONG guard 通過(n≥20, avg_net > 0)
- [ ] 30 天無 🟠 regime alert
- [ ] 模型過 transition window ≥ 7 天
- [ ] stage2_kill_criteria.md committed
- [ ] OKX demo 帳號 + API key 取得
- [ ] API key 權限驗證:✅ 讀 + 交易,❌ 不含 withdraw / transfer

**全勾才能開始本 checklist**。

---

## 1. 連線層(CN — Connection)

### T-CN-01 [Auto] WS 冷啟動建連
- **觸發**:executor 從零啟動
- **通過**:public WS(market data)+ private WS(orders/positions/balance)兩條都在 10s 內訂閱完成 + 收到第一筆 heartbeat
- **映射**:A3 baseline
- **狀態**:[ ]

### T-CN-02 [Manual] WS 斷線 30s 自動恢復
- **觸發**:斷網路 30s(`iptables -A OUTPUT -p tcp --dport 8443 -j DROP` 30s)
- **通過**:60s 內 reconnect + 重訂閱所有 channel + 期間漏掉的 fill/position 透過 REST 補齊(對帳一致)
- **映射**:A1 / A2 邊界
- **狀態**:[ ]

### T-CN-03 [Manual] WS 斷線 > 5 min 觸發降階
- **觸發**:斷網路 6 min
- **通過**:executor 偵測到 → 強制平倉所有(透過 REST 走另一條路徑)→ 推 critical alert → status="DEMOTED"
- **映射**:**A1 必觸發**
- **狀態**:[ ]

### T-CN-04 [Semi] WS heartbeat 失敗
- **觸發**:模擬 server 不回應(本地 proxy 攔截 pong)
- **通過**:30s 內偵測 → halt 下新單,5min 仍無回應 → 降階
- **映射**:**A3 必觸發**
- **狀態**:[ ]

### T-CN-05 [Semi] REST API timeout 重試
- **觸發**:本地 proxy 對 OKX REST 加 30s 延遲
- **通過**:重試 3 次 exponential backoff(1s/2s/4s),仍失敗 → fail-safe(不重複下單)
- **映射**:B baseline
- **狀態**:[ ]

### T-CN-06 [Manual] NTP drift 偵測
- **觸發**:`sudo date -s "5 seconds in past"` 5s → 觀察 → 再改 30s
- **通過**:drift > 5s 推 alert + halt 下新單(C5);drift > 30s 降階(C6)
- **映射**:**C5 + C6 必觸發**
- **狀態**:[ ]

### T-CN-07 [Auto] 連續 3 次 reconnect 失敗降階
- **觸發**:WS server 模擬持續拒絕連線(本地 proxy 回 ECONNREFUSED)
- **通過**:3 次失敗後 status="DEMOTED" + alert
- **映射**:**A2 必觸發**
- **狀態**:[ ]

---

## 2. 訂單下單(OP — Order Placement)

### T-OP-01 [Auto] Market entry 下單 + fill 確認
- **觸發**:signal_direction="UP" 觸發 executor.cycle
- **通過**:market 單下出 → 2s 內收到 WS fill 事件 → local DB `v7_okx_positions` 寫入正確 entry_price
- **映射**:E2E baseline
- **狀態**:[ ]

### T-OP-02 [Semi] 餘額不足拒單
- **觸發**:把 OKX demo 帳戶餘額提到 < min_notional,然後跑一次 entry signal
- **通過**:OKX 回 51008(餘額不足)→ executor 不 crash → 記 log → 不重試
- **映射**:B6
- **狀態**:[ ]

### T-OP-03 [Auto] Min lot 自動處理
- **觸發**:`size_frac × equity / contract_size` 算出 < 1 contract 的場景
- **通過**:floor 到整數 contract,若 < 1 → skip 該 signal + 記 log(不視為 error)
- **映射**:B6
- **狀態**:[ ]

### T-OP-04 [Semi] 已有 open position 時拒收新 entry
- **觸發**:open 一筆 LONG,接著餵第二個 UP signal
- **通過**:executor 直接 return "hold",**絕對不能**下第二個 entry order
- **映射**:**A5 預防(這就是 A5 對應的代碼路徑)**
- **狀態**:[ ]

### T-OP-05 [Manual] Pending entry 取消
- **觸發**:下 limit 單(若 V7 用 limit),立刻 cancel
- **通過**:OKX 確認 cancel + 2s 內 local DB 反映
- **映射**:E2E baseline
- **狀態**:[ ]

---

## 3. Stop / Algo orders(SO — Stop Orders)

### T-SO-01 [Auto] Entry fill 後 5s 內掛上 algo stop
- **觸發**:正常 entry 流程
- **通過**:entry fill confirm 後 5s 內 OKX 有 active conditional order(trigger price = entry ± 3×ATR)
- **映射**:**B4 必驗證**
- **狀態**:[ ]

### T-SO-02 [Auto] Amend trigger price 成功
- **觸發**:模擬 trailing extreme 推進(在 cycle 中改 cur_stop)
- **通過**:呼叫 amend-algo-order → OKX 200 OK → WS 推送更新 → local DB `current_stop` 一致
- **映射**:**B2 必驗證**
- **狀態**:[ ]

### T-SO-03 [Manual] Algo trigger by price wick
- **觸發**:OKX testnet 等到 BTC 真的 wick 穿過你的 stop(可能要等幾小時/天)
  - 或:把 trigger price 改到當前價附近 1 bp,等下一根 bar
- **通過**:OKX 觸發 → market fill → WS 推送 → executor 記為 exit_reason="trail_stop"
- **映射**:E2E baseline + 確認 wick 偵測一致
- **狀態**:[ ]

### T-SO-04 [Semi] Amend in race with fill
- **觸發**:在 algo order 即將觸發瞬間(< 100ms)送 amend
- **通過**:任一發生:(a) amend 成功 → fill 用新 price;(b) fill 先發生 → amend 收 51400 already filled → executor 不 panic,記 log
- **映射**:B3 邊界
- **狀態**:[ ]

### T-SO-05 [Auto] 24h 連續 trailing amend 0 錯誤
- **觸發**:跑 testnet 24h(每根 1h bar update_cycle 觸發一次 amend)
- **通過**:24 次 amend 全部成功 + 失敗率 = 0%
- **映射**:**B2 quantitative validation**
- **狀態**:[ ]

### T-SO-06 [Manual] Algo order 被 reject 時 fallback
- **觸發**:把 trigger price 設超過 OKX 接受的 price band(如距現價 50%)
- **通過**:reject → fallback 改用 cycle 內 bar-close 軟性檢查(用 paper executor 邏輯)+ alert
- **映射**:B6 邊界
- **狀態**:[ ]

---

## 4. 對帳(RC — Reconciliation)

### T-RC-01 [Manual] Cold start 偵測既有倉位
- **觸發**:在 OKX web 手動開一個 BTC-USDT-SWAP 倉位,然後啟動 executor(local DB 沒這筆)
- **通過**:executor 偵測 → **halt 下新單** + critical alert + 不自動接管那筆倉位
- **映射**:**A4 必觸發**
- **狀態**:[ ]

### T-RC-02 [Manual] Process kill 中途的對帳
- **觸發**:在 entry fill 中途 `kill -9` executor → 等 fill 完成 → 重啟
- **通過**:reconciler 偵測 OKX 有倉位 + local DB 也有(但 status 卡在 "PENDING")→ 比對 size/direction 一致 → 補正 local DB 為 OPEN + 重新 wire trailing stop
- **映射**:**A6 必驗證**
- **狀態**:[ ]

### T-RC-03 [Manual] Size 不一致 halt
- **觸發**:executor 持 0.01 BTC LONG,在 OKX web 手動再加 0.005 BTC
- **通過**:下個 cycle 對帳 → 發現 0.015 ≠ 0.01 → halt + critical alert
- **映射**:**A4 必觸發**
- **狀態**:[ ]

### T-RC-04 [Auto] 連續 100 cycles 對帳一致
- **觸發**:正常運行 100 個 cycle(~4 天)
- **通過**:每個 cycle 對帳結果 = "consistent",log 可查
- **映射**:A4 quantitative
- **狀態**:[ ]

### T-RC-05 [Manual] 方向不一致 halt
- **觸發**:executor 持 LONG,在 OKX 手動開 SHORT(若 posMode=long_short)或淨倉位反向(若 net_mode)
- **通過**:halt + alert
- **映射**:**A4 必觸發**
- **狀態**:[ ]

### T-RC-06 [Auto] EOD reconciliation
- **觸發**:每日 23:59 觸發完整對帳(所有當日 fill vs local DB 所有 trade)
- **通過**:每日 0 不一致;若有,critical alert + 隔日 halt 等修
- **映射**:A4 daily safety net
- **狀態**:[ ]

---

## 5. Rate Limit / 時序(RT — Rate / Timing)

### T-RT-01 [Auto] 訂單 burst 不丟單
- **觸發**:模擬 30s 內 60 個訂單操作(壓力測試,不在 V7 正常流量範圍但要驗證 client 行為)
- **通過**:全部排隊送出,0 個被 client 端丟棄;OKX 端若拒一些(rate limit hit)會在 client 端重試
- **映射**:B baseline
- **狀態**:[ ]

### T-RT-02 [Auto] Query rate limit 不影響訂單
- **觸發**:故意把 reconciler 查詢頻率調高觸發 rate limit
- **通過**:query endpoint 排隊/丟棄不影響 order endpoint;訂單操作仍即時
- **映射**:B baseline
- **狀態**:[ ]

### T-RT-03 [Semi] Funding settlement 時刻訂單
- **觸發**:在 00:00 / 08:00 / 16:00 UTC ±2 min 內觸發 entry/exit
- **通過**:訂單正常下出 + funding 扣款正確記錄到 equity
- **映射**:E2E
- **狀態**:[ ]

### T-RT-04 [Auto] 7 天連續運行無 leak
- **觸發**:executor 跑 7×24h 不重啟
- **通過**:memory usage 增長 < 50%;WS 連線無漏報(對帳一致);無 hung thread
- **映射**:結構性
- **狀態**:[ ]

---

## 6. Failure Injection(FI — 故意打破)

### T-FI-01 [Manual] Open position 期間 WS 斷線 → 恢復
- **觸發**:open LONG,等 2 個 cycle,斷網 2 min,恢復
- **通過**:reconnect 後對帳一致 + trailing stop 沒丟 + 期間若 stop 觸發過,reconciler 偵測到
- **映射**:A1 + A6 整合
- **狀態**:[ ]

### T-FI-02 [Semi] Stop placement 失敗重試
- **觸發**:本地 proxy 對第一次 stop placement request 回 5xx,第二次放行
- **通過**:5s 內重試成功 + stop 掛上
- **映射**:B4 邊界
- **狀態**:[ ]

### T-FI-03 [Manual] Partial fill 處理
- **觸發**:在 OKX testnet 故意製造低流動性 limit 單(若 demo 模式不支援,這項標 [SKIP-testnet-limitation])
- **通過**:executor 在 30s 內決策(等剩餘 / cancel 剩餘 / 重發)
- **映射**:**B3 必驗證**
- **狀態**:[ ]

### T-FI-04 [Auto] Duplicate signal 防護
- **觸發**:同一根 bar 餵 2 次 cycle(模擬 signal engine bug)
- **通過**:第 2 次 cycle 看到 open position → return "hold",**絕不**下第二個 entry
- **映射**:**A5 必驗證**
- **狀態**:[ ]

### T-FI-05 [Manual] Kill -9 後無 orphan
- **觸發**:`kill -9 <pid>` 在 cycle 中途 → 重啟
- **通過**:重啟後對帳 → 無 OKX 有但 local 沒的 orphan position;若有,reconciler 偵測 → halt
- **映射**:**A6 必驗證**
- **狀態**:[ ]

### T-FI-06 [Semi] MySQL 連線斷
- **觸發**:`iptables` 暫時阻斷 MySQL port 30s
- **通過**:DB write 失敗時 executor 不下新單(寧可錯過 signal 不下盲單);恢復後 reconciler 補正
- **映射**:結構性 fail-safe
- **狀態**:[ ]

---

## 7. End-to-End(E2E — 完整訊號到 fill)

### T-E2E-01 [Manual] Full lifecycle: signal → entry → trail → opp_signal exit
- **觸發**:跑 testnet 直到自然發生:Strong UP signal → entry fill → 數小時 trailing update → 出現 Strong DOWN signal
- **通過**:每階段 log + DB + OKX 三方一致;exit_reason="opp_signal"
- **映射**:整體
- **狀態**:[ ]

### T-E2E-02 [Manual] Full lifecycle: signal → entry → trail_stop exit
- **觸發**:跑 testnet 直到 trailing stop 自然觸發
- **通過**:exit_price 接近 trigger price(±5 bps);DB 記錄 exit_reason="trail_stop"
- **映射**:整體
- **狀態**:[ ]

### T-E2E-03 [Manual] Full lifecycle: time_cap exit
- **觸發**:在低波動期間自然發生(48h 沒打到 trail,opposite signal 也沒來)
- **通過**:48h 整點 cycle 觸發 market close + exit_reason="time_cap"
- **映射**:整體
- **狀態**:[ ]

### T-E2E-04 [Auto] Testnet vs Paper 平行對比
- **觸發**:同一個 signal source 同時餵 paper executor 和 testnet executor,跑 100+ trades
- **通過**:fill price delta histogram 中位數 < 5 bps;tails 內(95th percentile)< 15 bps;**不能有任何 trade 是 paper 有 testnet 沒(或反之)**
- **映射**:**D1 校準**
- **狀態**:[ ]

### T-E2E-05 [Auto] Testnet 100+ trades net bps > 0
- **觸發**:T-E2E-04 累積 100+ trades
- **通過**:testnet net bps(扣 OKX 真實 fee + 實測 slippage)> 0
- **映射**:**D2 預驗證**
- **狀態**:[ ]

---

## 8. Reporting(RP — 報表)

### T-RP-01 [Auto] /testnet-perf endpoint 上線
- **觸發**:HTTP GET /testnet-perf
- **通過**:回傳 OKX 真實 equity(走 REST balance API)+ trade history + 與 paper 對比
- **映射**:可觀測性
- **狀態**:[ ]

### T-RP-02 [Auto] Slippage histogram 可視
- **觸發**:/testnet-perf 中查看 slippage tab
- **通過**:每筆 trade 的 (paper_fill_price - testnet_fill_price) 直方圖
- **映射**:D1 校準前提
- **狀態**:[ ]

### T-RP-03 [Auto] 對帳狀態可視
- **觸發**:/testnet-perf 中查看 reconciliation tab
- **通過**:顯示 last_check_ts / consecutive_clean_days / last_mismatch_details
- **映射**:A4 可觀測性
- **狀態**:[ ]

### T-RP-04 [Auto] Kill trigger counter 可視
- **觸發**:/testnet-perf 中查看 kill triggers tab
- **通過**:每個 trigger ID(A1-F5)的 30-day count + last triggered time
- **映射**:全 triggers 可觀測性
- **狀態**:[ ]

### T-RP-05 [Auto] /paper-perf 與 /testnet-perf 並列
- **觸發**:dashboard 同時顯示兩者
- **通過**:同一個訊號的 paper 結果 vs testnet 結果可直接視覺對比
- **映射**:E2E
- **狀態**:[ ]

---

## 9. 進階到 Stage 3 的硬性條件(GR — Graduation)

下列**全部**通過才能啟動 Stage 3 live $100:

- [ ] **GR-01**:本 checklist 第 1-8 節全部 [x]
- [ ] **GR-02**:連續 **7 個交易日** 0 kill trigger 觸發
- [ ] **GR-03**:testnet 累積 **100+ trades**
- [ ] **GR-04**:對帳連續 7 天 100% 一致(任何一次 mismatch 重置 7 天計數)
- [ ] **GR-05**:T-E2E-04 fill price delta 中位數 < 5 bps、95th < 15 bps
- [ ] **GR-06**:T-E2E-05 testnet net bps > 0
- [ ] **GR-07**:所有 [SKIP] 項目在 `docs/stage2_testnet_results.md` 有合理 skip 理由(不能無故 skip)
- [ ] **GR-08**:`docs/stage_progression_log.md` 寫入 Stage 2 完成 sign-off(時間 + 摘要 + 已知限制)
- [ ] **GR-09**:Stage 3 kill criteria doc 草擬完成(`docs/stage3_kill_criteria.md`)
- [ ] **GR-10**:Stage 3 live $100 風險預算確認(願意輸光不痛)

---

## 10. 執行追蹤(template)

進入 testnet 後新開 `docs/stage2_testnet_results.md`,逐項記錄:

```markdown
# Stage 2 Testnet Validation Results

開始時間: YYYY-MM-DD HH:MM UTC
OKX demo account: <masked>
Executor commit: <sha>
Predicted completion: ~14 天(假設一天能跑 5-8 項)

## Results

T-CN-01: [x] 2026-XX-XX 14:00 UTC — passed, WS connected 3.2s
T-CN-02: [x] 2026-XX-XX 14:30 UTC — passed, reconnected 41s, reconciliation clean
T-CN-03: [x] 2026-XX-XX 14:45 UTC — passed, demoted as expected, alert received
T-CN-04: [SKIP+testnet-limitation] OKX demo WS heartbeat 無法 mock pong drop
...
```

---

## 11. 重要原則

1. **不能跳項**。每項都標狀態,通過 / skip(+理由)/ 未做。
2. **Skip 必須在 stage2_testnet_results.md 寫理由**,且 GR-07 會檢查 skip 是否合理。
3. **Failure injection 必做**(第 6 節)。「正常情況都過」不算 testnet validated,故意打破都過才算。
4. **進階文件閘**:除了技術項目,GR-09 / GR-10 是 doc + 心理閘 — 確認 Stage 3 規則準備好、心理準備好輸 $100。
5. **任何 [SKIP] 都會在 Stage 3 / Stage 4 變成已知風險**。比如 T-FI-03 partial fill 在 testnet 不能模擬,Stage 3 第一筆遇到就要小心。

---

## 12. 已知 testnet 限制(可能 SKIP 的項目)

OKX demo 環境與 live 的已知差異:
- **流動性差**:partial fill 不易模擬(T-FI-03)
- **資金費率不真實**:funding settlement 行為與 live 可能不同(T-RT-03)
- **price band 較寬**:某些 reject 場景在 demo 不會觸發(T-SO-06)
- **WS 推送延遲不真實**:race condition 在 demo 比 live 更慢顯現(T-SO-04)

這些限制讓 Stage 3 第一週需要 extra 監控,不是 testnet 沒測就等於 live 也沒事。
