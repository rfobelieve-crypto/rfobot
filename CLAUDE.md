# 專案 CLAUDE.md - BTC 量化交易系統（從指標漸進演化）

## 專案定位（2026-05-09 更新）
這個專案最初是「多空強度預測指標 / Market Intelligence Indicator」，
從 2026-05-09 起，**正在漸進演化成量化交易系統（含自動下單）**。

### 為什麼從指標走向自動交易
- 使用者不要盯盤手動下單
- 5.5 個月歷史訊號 robustness check 顯示後半段 net per trade +9 bps（已扣 13 bps 成本），
  值得用嚴格風控驗證能否轉成實戰收益。詳見 robustness 結論：
  - Strong / CHOPPY 91.8% WR 是 sample artifact（regime 標記從 3/21 才開始），不是真 edge
  - 整體 Strong 95% CI [-2.2, +14.6] bps 含 0，無法統計上斷言 edge 顯著
  - 可信的判斷是「邊際正 EV，需要 forward window 驗證」

## Staged auto-trading framework
不是「驗證夠了再上線」vs「不驗證就上線」的二元選擇。是「金額大小 × 風控深度 對齊
edge 確信度」的漸進過程。

| Stage | 描述 | Risk | Leverage | 進階條件 |
|---|---|---|---|---|
| 0 | 純指標 + 推送 | 0 | n/a | (已過) |
| 1 | ~~Paper trading~~（**2026-06-05 移除**，原 gate 轉 LIVE 衡量）| 0 | 1.0x | ~~100+ 筆 paper trades + paper net > +5 bps × 4 週~~ → 改由 LIVE 績效衡量 |
| 2 | Testnet executor（exchange 測試環境）| 0 | 1.0x | testnet 1-2 週無 bug + order flow 正確 |
| 3 | Live tiny size（$100，輸光不痛）| 極小 | 1.0x | live 4 週 net positive + MDD < 20% |
| 4a | 放大到 $1k（3 個月）| 小 | 1.0x | Stage 3 通過 + 0 kill trigger |
| 4b | $1k（3 個月）| 小 | 1.2x | 4a 通過 + MDD < 10% |
| 4c | $5k（6 個月）| 中 | 1.5x | 4b 通過 + 連續 6 個月 hit no kill rules |
| 4d | $10k+（12 個月+）| 高 | **2.0x（絕對上限）** | 4c 通過 + 真實 Sharpe ≥ 1.5 |

每個階段都有 hard rules，寫入 production 程式碼，**不靠紀律**：
- drawdown trigger（cumulative drawdown 觸發 → 自動降階段）
- connection loss kill switch（與 exchange 失聯 → 取消所有未平倉位）
- position limit（單筆 / 總部位上限）
- daily loss cap（單日累積虧損上限 → 暫停當日所有訊號）
- **leverage cap**：當前 stage 的 leverage 上限寫進 config，超過則 executor 拒絕啟動

### Leverage ladder 數學依據（2026-05-25 加入）

2.0x 絕對上限不是拍腦袋，是基於當前 edge profile（μ=+5%，σ=30%）計算：
- Kelly optimal: f* = μ/σ² ≈ 0.56x（已小於 1x）
- Volatility drag: r_compound = E[r] - 0.5σ²L²
  - L=2.0: drag = -18%（仍可被 edge 覆蓋）
  - L=3.0: drag = -40.5%（drag > expected return，長期虧損）
  - L=5.0: drag = -112%（mathematical ruin，不論 edge）
- Stress Test 7 regime flip MDD scaling:
  - 1.0x → -15%（kill switch 救援）
  - 2.0x → -30%（painful 但可活）
  - 3.0x → -45%（半條命，加滑點接近 wipeout）
  - 5.0x → -75%（實質歸零）

**何時可考慮放寬 2.0x 上限**:
連續 24 個月實盤 Sharpe ≥ 3.0（目前 0.17-0.5）+ MDD 從未超過 -10%
+ 經過至少 2 個完整 regime flip 仍正 EV。在那之前，2.0x 是 hard cap。

## Paper cohort 移除（2026-06-05 決策）

**背景**：$100 live（Stage 3, 10x）已上線並穩定運行（reconciliation 連續 CONSISTENT）。使用者決定 **LIVE 是主力**，paper cohort 不再需要——整個移除，系統只留 LIVE。

**觸發點**：2026-06-04 一場 `orphan_local` HALT 把 OKX id4 用 admin_heal 歸零，導致 live 錯過一筆 paper 仍在跑的 trade，兩 cohort 永久交叉。使用者判斷：與其維護「paper 對 live 對齊」的複雜同步邏輯，不如直接砍掉 paper，承認 live 就是真相來源。

**改動**：
- 刪 `indicator/v7_paper_executor.py`；移除 `v7_paper_positions` 的所有讀寫（DB 表 archive 留底，不再寫入）
- 兩張圖表（靜態 PNG + 互動）的進出場三角形改抓 `v7_okx_positions`（LIVE 真實進出場）
- 移除 OKX executor 的 paper-sync gate（`_is_paper_holding`）——live 不再等 paper
- Telegram「V7 Stats」按鈕 → 改指向既有的 LIVE 報表（`/okx-perf`）
- dashboard 移除「V7 Paper shadow」+「Paper vs Live drift」區塊

**stage 計畫轉移到 LIVE**：原本 §staged framework 中由 paper 衡量的進階 gate（net bps / WR / 連續週數），全部改由 **LIVE 實盤績效**衡量。Stage 1「paper trading」這格視為已歷史化，當前真實所在 = Stage 3 ($100 live, 10x)。

**代價自負**：失去「paper 作為 edge 真假的獨立並行驗證」。但 live 已是真錢樣本，本身就是最硬的 edge 驗證；維護兩套 cohort + 同步邏輯的複雜度 > 並行驗證的邊際價值。**保留的不可鬆綁規則（金額上限、kill switch、leverage cap、manual approval）完全不受影響**——這次移除的只是「影子 paper」，不是任何風控。

## 當前策略：研究 + Small Live 並進（2026-05-27 決策）

**背景**：原 staged framework 要求 Stage 1 滿 100 trades + 4 週才進 Stage 2，按目前 9 天 6 筆的節奏需 5 個月。使用者選擇接受 informed risk：用 $100 live 作為「**operational stress test + edge 二次驗證**」，paper cohort 同時繼續累積。

**改動的規則**：
- Stage 1 → Stage 2 不再硬要求 100 trades + 4 週；改成「OKX skeleton TODO 全填完 + unit tests 過 + 3-5 天 testnet shakeout」
- Stage 2 → Stage 3 ($100 live) 不再硬要求 38 項 checklist 全跑完；改成「testnet 連續 3 天對帳 100% + 0 unhandled exception + manual approval 模式跑過 5 筆」

**保留的不可鬆綁**：
- **金額**：$100 live = Stage 3 上限，未進 Stage 4a 不准加碼（即使 $100 賺到 $200 也是 $100 keep + $100 不再用）
- **Hard kill switches 必須先驗證能觸發**（不是只寫進 code）：unit test + testnet 至少一次故意觸發
- **Manual approval 第 1 筆強制人工確認**（2026-05-31 從 5 → 1）：第一次真實執行 OKX trade path 必須 operator 確認 size/方向/stop 都對；之後 auto，因為「量化交易要自動」是 quant 本質
- ~~**Paper cohort 不停**~~：**2026-06-05 廢止**——paper cohort 已整個移除，LIVE 成為唯一 cohort（見 §Paper cohort 移除）。原本由 paper 衡量的 stage 進階條件全部轉由 LIVE 實盤衡量
- **Leverage 1.0x 不准動**（Stage 3 階段）
- **Stage 3 → Stage 4 仍照原硬條件**：live 4 週 net positive + MDD < 20% + 0 kill trigger

**做這個決策的代價自負**：
- 第一筆 live 訂單 = OKX REST/WS code 第一次真實執行 = 有 ops bug 的風險（mitigations 寫在上面）
- Edge 若是 fake，$100 是發現的成本（mistake.md 應記：用 $100 換 edge 真假驗證，比 5 個月等更便宜）
- 一旦 hit 任何 kill trigger，**回到 Stage 1 重新驗證**不是「凹下去」

## 分數合約 sizing「B」取代 10x 權宜 (2026-06-06)

**前提推翻**：下面 2026-05-28 的「10x override 是為了讓 1 contract 開得起」整段，**前提是錯的**。OKX BTC-USDT-SWAP 的真實 `minSz`/`lotSz` = **0.01 張**（$6 notional，已用 public instruments API 驗證），根本不是「1 張最小」。會卡在 1 張是 executor 自己的 `int(target_notional/per_contract)` 把 size 無條件捨去成整數張——這是 code 的鍋，不是交易所限制。手動爆倉那天（2026-06-05）的「$89 被逼 ~7x」就是這個 int() 造成的假性 over-leverage。

**現行 sizing（commit 9cc2a64）**：
- **名目 notional = NOTIONAL_LEV_MULT (2.0) × equity**，round 到 0.01 張。隨 equity 自動縮放（賺多下多、虧多下少，無 leverage creep）。
- **有效槓桿 = 2x**（對 equity）；$89 → ~$178 名目 / 0.29 張 / ~$18 保證金。
- 169-trade WF 模擬 + 注入 −10% 跳空：2x 活得下來、10x 一筆歸零。
- **OKX 帳戶的 leverage 設定（10x）只剩「決定鎖多少保證金」的作用，不再是策略風險槓桿**。真實風險槓桿由 NOTIONAL_LEV_MULT 決定 = 2x。
- 整條 pipeline 已改成支援小數張（DB DECIMAL、所有 int(size_contracts)→float、對帳容差 0.005）。

**對 leverage 紀律的影響**：實際有效槓桿 2x 落在 §Leverage ladder 數學依據可接受範圍內（Kelly 0.56x 的 ~3.6x，但 2x 的 vol drag 仍可被 edge 覆蓋；遠低於 hard cap 2.0x 的精神…註：2x = 剛好等於 Stage 4d 絕對上限，但這是「為了小帳戶開得起單」的有效槓桿，非加碼意圖，且帳戶極小、kill switch −20%/−30% 收緊中）。下面 2026-05-28 的 10x 段落保留作歷史，但**實務上 sizing 已不靠 10x**。

## 10x leverage informed override (2026-05-28)

**背景**：BTC-USDT-SWAP 1 contract = 0.01 BTC ≈ $750 notional。$100 + 1x leverage 連 1 contract 都開不了 → Stage 3 完全卡死。使用者選擇接受 informed override：保 $100 capital、鬆 leverage cap 到 10x 讓 1 contract 開得起來。

**這條 override 違反兩條既有規則**：
- §Leverage ladder 數學依據：Kelly optimal 0.56x，1x 已超過 Kelly；10x 是 17.8x Kelly
- §仍然禁止的：「禁因為想賺更多就改 leverage cap——cap 來自數學不是情緒」

**為什麼接受**：
- 這次不是「想賺更多」，是「為了讓 testnet/live 能跑出第一筆有意義 trade」
- 沒 leverage 鬆，Stage 3 永遠走不到（連 1 contract 都開不了）
- Stage 3 的本意是 operational stress test + edge 二次驗證，不是 alpha 機器；$100 全輸的成本可接受

**為了補償 10x 的數學風險，kill switches 同步收緊**：
- daily_loss_cap_pct: -50% → **-20%**（10x 下 -20% account ≈ -2% BTC，stop-out 3 筆觸發）
- total_loss_cap_pct: -50% → **-30%**（career-end，Stage 3 結束）

**10x 下單筆風險**：
- 3xATR stop 在 10x = 約 -6% account = -$6 per stop-out
- 3 連虧 = -18% = 接近 daily cap → halt
- 5 連虧 = -30% = total cap → Stage 3 終結

**真會死的情境**（必須接受才能走這條）：
- BTC 一晚 -3% 跳空（leverage 算下 -30% account 直接掃 total cap）
- 黑天鵝 -10% 級別 → 算下 -100% 直接歸零，連 cap 都來不及救
- 這些情境發生過（2024-08, 2025-01），未來會再發生

## Staged 進階條件對照表（更新版 2026-05-28，第 2 次 informed override）

**2026-05-28 第二次 override**：使用者選擇**完全跳過 testnet shakeout**，直接接 live。理由是 $100 max loss 可接受，testnet 寶貴的「驗證 OKX 程式碼」功能可由 read-only live smoke 替代（同樣 0 風險）。Stage 2 從 "testnet shakeout 3-5 天" 改成 "read-only live smoke + manual approval"。

| Stage | 描述 | Risk | Leverage | Daily/Total cap | 進階條件 |
|---|---|---|---|---|---|
| ~~1~~ | ~~Paper trading~~ | — | — | n/a | **2026-06-05 移除**：paper cohort 整個拔掉，LIVE 成為唯一 cohort + 唯一圖表記錄來源 |
| 2 | **Read-only live smoke**（取代 testnet shakeout）| 0 | 10x | -20% / -30% | 連 OKX live：讀 balance ✓、server time NTP drift OK ✓、WS auth + 訂閱 ✓、reconciliation CONSISTENT ✓ |
| 3 | **Live $100**（當前目標）| -$100 上限 | **10x** | -20% / -30% | Stage 2 smoke 全綠 + manual approval mode 跑 5 筆人工確認 |
| 4a | $1k（3 個月）| 小 | 1.0x | -20% / -30% | Stage 3 跑 4 週 + net positive + MDD < 20% + 0 kill trigger |
| 4b | $1k 1.2x | 小 | 1.2x | -15% / -25% | 4a 通過 + MDD < 10% |
| 4c | $5k | 中 | 1.5x | -15% / -25% | 4b 通過 + 連續 6 個月 hit no kill rules |
| 4d | $10k+ | 高 | **2.0x（絕對上限）** | -10% / -20% | 4c 通過 + 真實 Sharpe ≥ 1.5 |

**跳過 testnet 的風險自負**：
- 我們新寫的 200 行 OKX 程式碼從未在 demo 環境跑過；read-only smoke 只能驗 read path，trade path 第一次執行 = 真錢
- 如果 _open_position 有 bug（例如算錯 size_contracts、submit 錯 side）→ 立刻真錢中招
- Mitigation：manual approval 5 筆 = 你看著 Telegram 推的「準備下單 LONG 5 contracts @ 75000」每筆按 YES 才執行
- 任何 manual approval 看到不對勁（方向錯、size 異常、價格離譜）→ 按 NO 取消 + 立刻回報

**注意 Stage 3 → 4a 的 leverage 反而從 10x 降回 1x**：Stage 4a 起金額放大到 $1k，1 contract 不再是門檻，回到 Kelly-respecting 1x 是正解。Stage 3 的 10x 是「為了開門」的權宜，不是策略的一部分。

## 壓縮版 Stage 3→4 edge 驗證（2026-06-10，第 3 次 informed override）

**背景**：原 Stage 4 ladder（4a→4d，需 12+ 個月 live + 真實 Sharpe ≥1.5）被使用者判定**太嚴、太慢**。問題的根源是它想用「12 個月 live PnL」**同時**證明兩件不同的事，而 live PnL 是證明薄 edge **最沒效率**的資料來源（薄 edge + 高方差 → 30-50 筆 live 的勝率 CI 寬到含硬幣線）。

**核心洞見：把「證 edge」和「證執行」拆開，各用最有統計力的資料證。**

- **Gate A — edge 是真的嗎？（統計，用累積訊號）**
  - 用 `tracked_signals` 表的**累積 live Strong 訊號回填結果**（幾百筆，不是幾十筆 live PnL）。
  - 門檻：Strong 勝率 bootstrap 95% CI **下緣 > 52%**（顯著高於硬幣）。
  - **STATUS：2026-06-10 已通過 ✅** — Strong n=739、勝率 59.5%、CI [56.0%, 63.2%]（下緣 56%）；最近 90 天 n=101、76.2%、CI [67.3%, 84.2%]。Moderate n=1241、54.4%、CI 下緣 51.7% 不顯著（再次佐證只開 Strong）。
  - **edge 這題已答 YES，不需要再用 live PnL 重證。**

- **Gate B — 執行有沒有把 edge 吃掉？（操作，用 30-50 筆 live）**
  - 用**今天修好 trailing 的系統**（見 mistake.md 2026-06-10 amend instId bug）跑 30-50 筆乾淨 live trade，要求全部成立：
    - 扣 8bps 成本後 **net ≥ 0**
    - **0 kill trigger**
    - **trailing 確認在 OKX 上真的有 amend 上移**（今天修的重點，必驗）
    - live 每筆報酬**落在 backtest 分布內**（無大幅負滑點驚喜）
  - 樣本從**今天（修好後）重新數**；之前 live 紀錄被 broken trailing + 手動爆倉污染，不算。

**新的擴大規則（取代 12mo/Sharpe 1.5 作為「第一次放大」的條件）**：
- Gate A + Gate B 都過 → 擴大**一級、適度增量**（$300-500 / 2-5x 名目，**不是**一次跳 $5k/$10k）。
- 之後每多 30-50 筆乾淨樣本 + 重檢兩 Gate → 再放一級。證據累積，規模才累積。
- 可選嚴謹升級：SPRT 序貫檢定，Gate B 證據夠強就提早收（不用死等固定 50 筆）。

**為什麼這樣「鬆」但不犧牲 edge 證明**：A 用大樣本（訊號）給統計力、B 用小樣本（live）只驗執行——不是降低標準，是把證明搬到對的資料層。原 Stage 4a-4d 的 leverage 階梯與時間/Sharpe 條件**作廢為「第一次放大」的硬門檻**，改由上述兩 Gate 取代；之後的逐級放大仍走「增量 + 重檢」。

**這次 override 不鬆的（ruin 保護，與「擴多快」無關）**：kill switch（daily −20% / total −30%）、leverage cap（有效 2x）、hit kill→降階重驗、max_position_count=1。這三樣是防再歸零一次（2026-06-05 的教訓），跟驗證速度無關，一個都不動。

**注意**：訊號方向準確率（59.5%）≠ 交易獲利（扣成本/停損後）。Gate A 證「edge 存在」，Gate B 才證「執行能把它變成 +EV」——兩個都要，缺一不可。

## 仍然禁止的（避免在錯的階段做錯事）
- **Stage 2-3**：禁鬆 hard kill switches 以外的 trigger；leverage hard cap = 10x（不可再放寬）
- **Stage 3**：禁未經 manual approval 5 筆就切自動（paper cohort 已於 2026-06-05 移除，不再有「paper 停寫」這條）
- **Stage 3 → 4a**：leverage 必須降回 1.0x；不能因為 $100 賺到 $200 就用 10x 加碼
- **Stage 4a-d**：leverage 階梯式放寬，**絕對上限 2.0x**；未 hit 各子階段條件不得進下一格
- **Stage 4 後**：禁 leverage > 2.0x，除非滿足「24 個月實盤 Sharpe ≥ 3.0」（見 §Leverage ladder 數學依據）
- 任何階段：strategy sweep 必須留 OOS hold-out，禁全資料 fit
- 任何階段：禁因為「最近表現好」就跳階段——必須 hit hard rules
- 任何階段：禁再鬆 leverage cap——10x 已經是「informed 一次」的極限；下次再要鬆要寫進 mistake.md
- 任何階段：hit kill trigger 必須降階重驗，不准「我覺得這次例外」

## 系統架構（v7 Dual-Model）
Dual XGBoost 架構：Direction Regressor + Magnitude Regressor，獨立管線。

### 數據層
- **Binance REST API** (3 endpoints)：klines (1h, 500 bars)、depth (L20)、aggTrades
- **Coinglass API v4** (24 endpoints)：15 timeseries + 9 snapshot
- **Deribit Public API** (2 endpoints)：DVOL 波動率指數、Options Summary

### 特徵工程
- **200+ 工程特徵**（Direction 136, Magnitude 72），12 個群組
- 所有計算為 trailing-only（無前視偏差）
- Coinglass 原生 1h 使用 merge_asof 精確對齊
- 自訂 alpha 特徵：impact_asymmetry (IC=-0.071)、post_absorb_breakout (mag IC=0.191)

### 模型
- **Direction Model**：XGBRegressor, 136 特徵, 輸出 pred_return_4h (TWAP path return)，rolling percentile 解碼為 UP/DOWN/NEUTRAL
- **Magnitude Model**：XGBRegressor, 72 特徵, 輸出 |return_4h|
- **Regime Detection**：CHOPPY / TRENDING_BULL / TRENDING_BEAR / WARMUP

### 信號生成
- Direction: 500-bar rolling percentile 解碼，top 5% → Strong UP，top 15% → Moderate UP（DOWN 同理）
- Absolute |pred| floor (Strong=0.0008, Moderate=0.0005)：低 vol regime 保險，rolling cutoff 比 floor 寬鬆時 floor 接管（2026-05-09 加入）
- Confidence = `min(|pred|/Strong_cutoff, 1.0)^0.6 × 100`（純 |pred| 公式，2026-05-09 移除 mag bonus 因為 OOS 顯示高 mag bar 在模型失靈區）
- Strong ≥ 80, Moderate ≥ 65, Weak < 65（顯示用，實際 tier 觸發看 |pred| vs cutoff）
- Hysteresis + Cooldown

### 輸出
- 圖表面板 (Confidence / Regime / K線+三角形 / Magnitude)
- Telegram 推送 (Strong 信號文字告警 + SHAP 驅動因子)
- REST API (10 routes)
- MySQL + Parquet 持久化

### 績效追蹤
- Rolling IC (7d/30d) + IC 趨勢 + 衰退警報
- Strong 信號追蹤 (4h 後自動回填結果)
- SHAP 驅動因子分析 (Strong 信號時觸發)
- Regime 拆解準確率
- 全部整合在 /perf 指令

## 模型輸出（固定格式）
- **pred_return_4h**: sign(direction) × magnitude
- **pred_direction**: UP / DOWN / NEUTRAL
- **strength_score**: Strong / Moderate / Weak
- **confidence_score**: 0~100
- **mag_pred**: |return_4h| 預測值
- **dir_prob_up**: P(UP) 原始值
- **regime**: 當前市場狀態

### 核心 target
y_path_ret_4h = mean(close[t+1..t+4]) / close[t] - 1 (TWAP path return)

### 評估指標
- Spearman IC / ICIR（預測值與實際收益的排序相關）
- 方向準確率
- Calibration monotonicity（預測越強，實際收益越高）
- Strong 信號勝率（目標 point estimate ≥ 65%，stretch 70%；天花板由 AUC ~0.57 結構決定，top-5% precision 實測 67.6%）
- Magnitude Top/Bot ratio

## 技術 Stack
- Python 3.11
- 資料處理：Pandas + NumPy + SciPy
- 資料庫：MySQL 8.0 (Railway 託管)
- 儲存：Parquet（歷史備份）、.data_cache/（API 回退快取）
- 模型：XGBoost (Dual Regressor)
- Web：Flask + APScheduler
- 圖表：Matplotlib (靜態) + TradingView Lightweight Charts (互動)
- 推送：Telegram Bot API
- 部署：Railway (git push 自動部署)
- 解釋性：SHAP (TreeExplainer, Strong 信號時觸發)

## 核心原則（永遠不能違反）
1. **無前視偏差**：所有特徵計算使用 trailing-only rolling，嚴格禁止 look-ahead。
2. **歷史與即時一致性**：`build_live_features()` 同時用於訓練數據建構和生產推論。
3. **時間對齊精準**：Coinglass 使用 merge_asof backward 對齊，快照數據只設定最後一根 bar。
4. **預測導向設計**：所有評估以 IC、方向準確率、calibration 為準，不做交易績效回測。
5. **特徵先回測再加入**：新特徵必須先跑 IC 回測驗證有效才加進系統。
6. **Edge Cases 處理**：假日流動性差異、Funding 結算跳動、rate limit、資料缺失。

## 圖表同步規則
系統有兩個圖表，修改時**必須同步更新**：
1. **靜態圖表** (`indicator/chart_renderer.py`) — Telegram 推送的 PNG
2. **互動圖表** (`indicator/chart_interactive.py`) — `/ichart` 的 TradingView Lightweight Charts HTML

任何圖表邏輯變更（面板、三角形、顏色、過濾條件）都要兩邊一起改。

## 命名與程式碼規範
- Class：CamelCase（如 IndicatorEngine、SignalExplainer）
- 函數/變數：snake_case（如 build_live_features、backfill_mag_pred）
- 偏好：清晰、可讀性高、模組化
- 新特徵加入前必須回測驗證 IC

## 專案階段
- 階段 4：特徵工程（目前重點 — 自訂 alpha 特徵 + 數據累積）
- 階段 5：模型開發（Magnitude 已重訓 v2, Direction 等 2 週後重訓）
- 階段 9：持續迭代（績效追蹤、IC 監控、衰退警報已上線）

## 跨 session 任務同步（2026-07-07）
- **TODO.md 是唯一的跨 session 任務真相源**。每次開工先讀 TODO.md 的「當前任務」區。
- Session 內建任務清單（TaskCreate）只作單次對話的進度追蹤——它存在本機
  session 狀態、不進 git、不跨機器。凡是隔天/換機器還要做的事，寫進 TODO.md 並 push。
