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
