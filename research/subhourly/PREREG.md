# 次小時（分鐘級）系統可行性 — Phase 0 Pre-registration

登記日：2026-07-18（門檻在看任何結果之前寫定，以本檔 git commit 時間戳為證）
提案人：使用者（2026-07-18）；整理：Claude

## 假說（使用者原始表述的正式化）

訂單流資訊有半衰期。1h 取樣把 60 根 1m 的高頻資訊做了平均——bar 內方向資訊
自我抵銷，所以「taker 訂單流在 1h+ 視野 IC≈0（已 4 次 powered 驗證）」
**不能推出**「分鐘取樣下亦為零」。若在分鐘級取樣，訊號可能在衰退完成前
仍可捕捉。

## 兩個必須分開回答的問題

- **Q1 資訊問題**：分鐘取樣的微結構特徵，對 15-240m 前向報酬有沒有
  穩定的「方向」預測力？（F2 smoke 的 IC +0.113 是幅度不是方向，不算數）
- **Q2 經濟問題**：該預測力在「可交易視野」是否清得過成本地板？
  成本地板（2026-07-17 已用自家 30d 1m mid 算定）：60% WR 樂觀上限下，
  持有 1/5/15m 毛利 0.7/1.7/3.0 bps < maker RT 4bps ⇒ 數學死刑；
  30/60m = 4.2/6.0 bps ⇒ 邊際可行。**可交易視野定義 = 30m 起，maker 執行。**
  （註：條件於訊號的 |move| 可比無條件大——G2 就是用條件桶算，不用上表否決）

## 資料（全部既有，不新建收集）

| 表 | 覆蓋 | 角色 |
|---|---|---|
| flow_bars_1m | ~520d | taker 流特徵主源 |
| ohlcv_1m | 存活性待審計 | 價格/報酬主源（若已停更→backfill 或 orderbook mid 替代） |
| orderbook_snapshots_1m | ~130d | L20 失衡（子樣本） |
| liquidation_1m | 待審計 | 清算脈衝 |
| depth_deltas_1m | **9d — 本輪排除** | 資料時鐘未到；PARK 後 re-run 時的主角 |

## Phase 0a — 資訊掃描（描述性；不做 Go/No-Go 決策）

特徵 family（全 trailing-only，禁技術指標）：
1. taker 失衡 `ti_{5,15,60}` = rolling Σdelta / Σvolume
2. 淨流 z 分數 `dz_{5,15,60}` = rolling Σdelta / trailing-24h std（幅度感知）
3. 量能 shock `vshock_60` = volume / trailing-60m median
4. 分鐘動量/反轉 `ret_{5,15,60}`
5. 清算脈衝 `liq_{15}`（若資料可用）
6. L20 失衡 `obi`（130d 子樣本）

Horizon 集（凍結，禁事後加減）：h ∈ {5, 15, 30, 60, 120, 240} 分鐘。
輸出：IC × horizon 衰退曲線 + 逐月（~17 個月）穩定性 + 前後半同號檢查。

## Phase 0b — 判定門檻（凍結）

- **G1 資訊存在**：≥1 特徵 family 在 h∈{30,60}m 上「逐月平均 |IC| ≥ 0.03
  且 ≥70% 月份同號 且 前後半資料同號」。
- **G2 經濟可行**：訊號 top-5% 條件桶在 30-60m 的期望淨值 > 0
  （扣 8bps = 2× maker RT），bootstrap 95% CI 下緣 > 0。
- **G3 模型級**（G1∧G2 通過才跑）：乾淨 walk-forward（purge+embargo、
  無 early-stop 洩漏）多特徵 XGB，2026-06-02 四條 sanity 全過
  （aggregate + per-fold mean > 0 + frac_pos > 55% + bootstrap CI 不含 0），
  再做 G2 同款經濟換算。

## 通過鏈與失敗處置

- G1∧G2 → 跑 G3。
- G3 過 → **Phase 1**：shadow 訊號線（log-only，仿 cancel watcher 模式，
  不下單）累積 4-8 週前瞻樣本。
- Phase 1 過 → **Phase 2**：**獨立小系統**（maker/post-only 執行、獨立
  子帳戶、獨立 kill switch、turnover cap），照 staged framework 從最小
  金額走。**V7 在全程零改動、零停機——分鐘系統是第二策略，不是 V7 改寫。**
- 任一 gate FAIL → 計畫 PARK，等 depth_deltas_1m 累積 ≥3-6 個月後帶
  撤單特徵 **re-run 一次**（僅一次，同門檻）。

## 紀律鎖

- 看資料後禁改特徵定義、門檻、horizon 集。
- 單月亮眼不構成提前 GO；禁 sweep 門檻、禁 p-hack horizon。
- Phase 0-1 期間 V7 與 executor 零改動。
- 本檔為「改寫整套系統」大工程的第 0 步：**先證明 edge 存在，再談工程。**

---

## 附錄 A — Phase 0 判決後的 re-run 設計註記（2026-07-18 登記；gates 不變）

Phase 0a 判決：G1 PASS / G2 FAIL → PARK（見 TODO 4.65 與
`research/results/subhourly_ic_scan.csv`）。本附錄只做兩件事：
**在看撤單資料之前**宣告 re-run 的測試家族，以及把 G2 門檻翻譯成
可操作的必要贏面（`required_bar.py`，power analysis 非新檢定）。

### A.1 必要贏面（撤單特徵 re-run 要打敗的數字）

| 已判最佳 cell | top-5% 條件 E\|move\| | 毛捕捉 | 隱含方向 p | 過 G2 需 p | 需放大 |
|---|---|---|---|---|---|
| A/ti_15@30m | 14.6bps | +0.90bps | 53.1% | 77.3% | 8.9x |
| A/ret_15@30m | 49.2bps | +1.37bps | 51.4% | 58.1% | 5.9x |
| B/dz_60@60m | 39.8bps | +2.81bps | 53.5% | 60.0% | 2.8x |
| B/ret_60@60m | 53.0bps | +6.20bps | 55.8% | 57.5% | 1.29x |

**翻譯**：re-run 的撤單特徵要在 30-60m 的 top-5% 桶交出 ≥8bps 毛捕捉 =
把條件方向勝率推到 ~57.5%+（在條件 |move| ~50bps 的選擇下）。

### A.2 兩個結構性洞見（re-run 分析必須回答）

1. **流極端 ≠ 動能**：taker 失衡 top-5% 的分鐘條件 |move| 只有 14.6bps
   （比無條件 21.2 還小）——流的極端發生在安靜盤，選了流就選不到波動。
2. **安靜分鐘的捕捉天花板**：F1b 假說預測撤單資訊集中在低量分鐘，但低量
   分鐘本身 |move| 小 → 就算方向勝率高也付不起 8bps。**撤單訊號要過 G2
   的唯一路徑 = 安靜分鐘的訊號預測「隨後的擴張波」**（壓縮→真空→突破的
   squeeze 論），re-run 必須直接量測「訊號分鐘之後 forward 窗的條件
   |move| 是否擴大 + 方向偏斜是否同時存在」。

### A.3 re-run 家族宣告（現在宣告、10 月才看）

僅此一次 re-run，家族固定為：
1. 撤單特徵——沿用 watcher def v1 凍結定義（skew15 / net15 / shock），
   加 depth_deltas 原生 bid/ask add/cancel 的 5/15/60m 聚合（不新調參）
2. 簿型特徵——orderbook_snapshots_1m 既有欄位原樣：wall_distance_bps
   （雙側）、spread_bps、imbalance_l5/l20（不衍生新定義）
3. 對照組——Phase 0a 原 taker 家族（量測「撤單相對 taker 的增量」）

Gates 原封不動（G1/G2/G3 同 §Phase 0b）；多重比較以「G1 需月同號 70% +
前後半同號」承擔。時間窗：spot depth_deltas ≥3 個月 → **最早 2026-10-09**。
