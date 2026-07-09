# Path 1 — Multi-Horizon Direction Models (1h + 1d 並進)

> **Why this exists**: CLAUDE.md / mistake.md 2026-06-02 證明同源資料 AUC 0.54
> 天花板。單一 4h edge 達不到機構級 Sharpe（數學上不行）。Path 1 是把現有 4h
> pipeline 套到 **1h** 跟 **1d** target，看每個 horizon 是不是獨立 edge。

## 假設與成功條件

我們的賭注是：「**同一份 137 個 feature**，預測不同時間尺度的方向，會帶來
**準獨立**訊息」。具體要看到的：

| 條件 | 目標 |
|---|---|
| 1h Gate A | Spearman IC > 0.04 (4h baseline 0.063) + bootstrap CI 下緣 > 0 |
| 1d Gate A | Spearman IC > 0.06 + bootstrap CI 下緣 > 0 |
| **Correlation between predictions** | < 0.5（核心 — 高度相關就不是新 edge）|
| Strong WR 各 horizon | > 55%（4h baseline 59.5%）|

**全部過 = Portfolio 從 1 edge 升 3 edges**，effective sample size 1.5-2x。
任一不過 = 這個 horizon 沒有獨立訊號、放棄該 horizon、不要強加。

## 工程相依關係（lighter than你想的）

1h / 1d **不需要新 feature**——同樣 137 個 trailing-only feature 可以套到不同
target。唯一需要動的是：

- **Label**：`y_path_ret_{H}h = mean(close[t+1..t+H]) / close[t] - 1` — 已參數化
- **Purge**：必須 ≥ horizon（4h 用 4、1h 可用 1、1d 用 24）
- **Embargo**：保留 ≥ 1 防自相關洩漏
- **Output 路徑**：horizon-suffixed parquet 避免覆蓋

## 三階段執行（每階段都有可回退的 checkpoint）

### Phase 0（本 session 完成）
- ✅ Refactor `build_direction_reg_labels.py` — column 名稱動態 (`y_path_ret_{H}h`)
- ✅ Extend `gate_a_revalidate_wf.py` 加 `--horizon` flag、scale purge
- 4h default 不變 — 現有 production retrain 工作流不受影響

### Phase 1（你 local 跑驗證）
```bash
# 1h target
python research/gate_a_revalidate_wf.py --horizon 1
# 1d target
python research/gate_a_revalidate_wf.py --horizon 24
# 4h baseline 對照（不應該變）
python research/gate_a_revalidate_wf.py --horizon 4
```

對每個 horizon 看：
- Strong WR（rolling percentile decode）
- Bootstrap CI 下緣是否 > 50%
- IC vs 4h baseline

把三組數字貼回來，我幫你 cross-correlation 分析。

### Phase 2（如果 Phase 1 結果支持）
- 重訓 production 1h + 1d model（拷貝 `train_direction_reg_4h.py` → `_1h.py` / `_1d.py`，threadthrough horizon）
- 各自 export 上線（但 **暫不接 OKX executor**——只 paper-track 訊號累積 Gate A）
- 等 1h / 1d 各自累 200+ live tracked_signals 後才考慮接執行

### Phase 3（最謹慎的一步 — 接到 live executor）
- 不能直接讓 1h + 4h + 1d 各自獨立開倉（`max_position_count=1` 限制）
- 必須設計**信號 fusion 規則**：3 個 horizon 訊號怎麼變一個進場決策
- 候選 fusion 規則（要研究後選一個）：
  - **多數決**：3 個都同向才開
  - **強優先**：Strong 4h > Strong 1d > Strong 1h，依強度搶單
  - **時間分倉**：上半天 1h、下半天 4h、夜間 1d（不重疊）
  - **持倉時間錯開**：3 個各自有自己的 stop 邏輯

## 不要做的事

- ❌ **直接重訓 production 4h model 改 horizon** — live retrain pipeline 會炸
- ❌ **同時把 3 個 model 都上 live** — 沒驗證 correlation 等於賭一把
- ❌ **看到 1h IC 0.08 就 yolo** — 1h target std 比 4h 大、IC 可能漂亮但 trade-layer
  edge 被 cost 吃光（8 bps round-trip 對 1h 比對 4h 痛 4x）
- ❌ **跳過 Gate A 直接做 Phase 2** — 沒過 Gate A 的 horizon 是 sample noise

## Cost 分析（重要 — 1h 特別痛）

| Horizon | 預期持倉時長 | 預期 net edge / trade | 8 bps cost 占 |
|---|---|---|---|
| 1h | ~1-2h | ?（要驗）| 50%+ |
| 4h | ~6-12h | 12 bps（你既有）| 40% |
| 1d | ~12-30h | ?（要驗）| ~20% |

1h 必須跟 **maker 訂單 + funding 一起算**才可能 net 正。1d 自然 cost 占比低、
最有機會 net 正、執行壓力小。

## 我的賭：1d 最可能過、1h 最可能掛、4h 不變

- **1d**：訊號數量少（一天 1 個 candidate） → cost 占比低 → 容易 net positive
- **1h**：cost 太重 + noise 太高 → IC 可能 ok 但 Gate B 過不了
- **4h**（既有）：vor 已驗

如果結果如預期，Portfolio = 4h + 1d（2 個 edge，準獨立）就值得做。
1h 大概率掛，那就不用花後續工程力。

## 進度追蹤

- [x] **Phase 0.1** — Refactor label builder（commit 待 push）
- [x] **Phase 0.2** — Extend Gate A revalidation script（commit 待 push）
- [ ] **Phase 1** — 你在 local 跑 1h / 4h / 1d 三次
- [ ] **Phase 1.1** — 三組數字 + correlation 分析回貼
- [ ] **Phase 2** — 條件性：通過 horizon 各自重訓 production model
- [ ] **Phase 3** — Fusion rule 設計（從 4 個候選挑一）
