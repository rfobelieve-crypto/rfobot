# 專案 CLAUDE.md - BTC 量化交易系統（從指標漸進演化）

---

## 現況速覽（快照 2026-08-05）

> 這一節是「**現在在哪**」。底下的歷史章節是「**怎麼走到這裡**」，**不要
> 拿歷史章節的數字當現行值**——很多已被後面的決策取代。
> 下面的數字全是**快照**，活數字請跑：`python research/portfolio_clocks.py`
> （時鐘）、`python research/sweep_failure/shadow_engine.py --gate`（變體）、
> `indicator/okx/config.py`（風控參數的真相源）。

**一句話**：三條策略共用一套資料層與風控紀律；只有 V7 碰真錢，另外兩條
在 forward 驗證中。

**實盤參數（V7，Stage 3）**

| 項目 | 現行值 | 出處 |
|---|---|---|
| 資本基準 | **$311.60**（2026-07-28 重置；08-08 修正——$274 是入金中途讀的，snapshots+id21 equity_before 證明真實起點 311.60） | `okx/config.py: initial_capital_usd` |
| 策略有效槓桿 | **2x**（名目 = 2 × equity） | `NOTIONAL_LEV_MULT`；guard 上限 3.0 |
| OKX 帳戶槓桿 | 10x（**只決定鎖多少保證金**，非策略風險） | `config.leverage` |
| Daily / Total kill | **−20% / −30%**（≈ −$62.3 / −$93.5） | `daily_/total_loss_cap_pct` |
| 同時持倉 | **1 筆** | `max_position_count` |
| 出場 | 3×ATR trailing、opp_signal 反向、conviction_decay(2 根) | `okx/executor.py` |

**executor 現在是停的（2026-08-13 使用者決定，不是故障）**：08-11 中午使用者
手動交易把權益從 $299 推到 $763（+08-12 入金 $13 → **$776**），對 $274 基準
超過 1.5x → **CAP-2 over-funding HALT 每小時觸發**（kill_log 已 211 筆，這是
預期狀態）。加上 `OKX_ENTRY_PAUSED`（08-11 解碼修法時開的），開新倉有兩道鎖，
出場/對帳/kill 照跑。使用者選擇**維持現狀不動基準**，先讓解碼修法累積 forward
樣本。注意 `config.py` live guard 上限是 $500——真要把基準改成 776 得先改那行
程式碼，否則 executor 開機即 RuntimeError（mistake.md 2026-07-28）。手動交易
這是第三次（06-05、07-27、08-11），這次是賺的。

**解碼修法的現況（2026-08-11 上線，08-13 查證）**：DOWN 側算術鎖死**已解除**
——Strong DOWN 切點從 −0.001786（比模型史上最低值還低 = 0 根可達）變成
**−0.001002，3 根 bar 構得到**，UP 側對稱也是 3 根。暖機靜默 100 根在
**08-12 16:00 結束**，之後的 NEUTRAL 是 pred 落在中間帶的正常結果，不是被鎖。
**但模型輸出本身仍偏正**（現行模型 109 根 pred 有 78% 為正、均值 +0.00074，
08-08 重訓才對中到 +0.00011，五天又漂回來）——解碼跟 live 分佈比排名，所以
兩側各 2.5% 依定義可達，代價是 DOWN 訊號的語意是「相對最不看多」而非「絕對
看空」。真實的多空平衡要等解禁後 ~2 週的開火比例，別拿 10 根樣本下結論。

**三策略狀態**

| 策略 | 狀態 | 距離下一個決策點 |
|---|---|---|
| **V7**（4h 方向+幅度） | Stage 3 live | Gate B 執行驗證 **15/30 筆**（累積 +0.84%、+7.1 bps/筆、勝率 46.7%、MDD −4.9%）|
| **流動性獵取**（策略 #2） | 規則凍結、forward 記帳 | **四條時鐘**（2026-08-13 活數字）：Gate F 正式軌道（變體 A·無濾網·core9）**126/1400**，CI 下緣 **已轉負**、7/9 幣正 —— 08-08 曾兩條件都達標，樣本一多就退回一條；變體 B（＋淺穿越濾網·29幣）**583/1400**、CI 下緣 **−0.022**；C **117/400**、D **68/400**（各自 08-09 起算，見下）。**全部變體的註冊簿在 TODO §0.43** |
| **撤單流**（策略 #3） | **方向性判決 FAIL（已定案）** | 08-10 過 n≥40,000 檢查點，**三個方向性檢定全滅**（見下）|

**生存條件層（2026-08-17 起，第四條研究線——監測的不是策略績效，是
策略賺錢的機制前提）**：績效監測在薄 edge 下數學上來不及（+7bps/筆 vs
~100bps 波動 → 分辨生死需數百筆），所以直接監測前提。已驗證並接進
clocks 週報（2e-c）的儀表：**ADX(14) 25/20——群眾自己的 regime 儀**，
對 SF 是**二級證據**（RANGING meanR +0.075 vs TRENDING +0.016，CI
[+0.010,+0.106] 離零、8/9 幣，TRENDING 幣過半掛 SF 逆風告警）、對 V7
一級（RANGING WR +5.3pp）；Donchian 突破派損益（SF 對手盤儀表，一級、
7/9）；SMA50/200 趨勢派損益（V7 逆風儀表，一級）。**trend_z 已退役**
（同效應、CI 寬一倍、桶佔比 3% vs 30%）。已定案的負結果：均值回歸
群眾（RSI/BB/Stoch）的損益對兩條策略**零資訊**——資訊軸在「誰在追」
不在「誰在接」；快趨勢派（EMA9/21）≠ 慢趨勢派，不可混。V7 衰退分解
（§0.49b）：**真衰退非組成假象**（CALM 主場兩向同幅下滑），60 天重訓
上限因此不可放鬆。變數註冊簿與全部判決在 **TODO §0.49~0.49d**。

**撤單流的方向性判決已定案 FAIL（2026-08-13 覆核 08-10 的預註冊判決）**：
depth_deltas 累積到 48,991 分鐘、過了 `POWERED_N`=40,000 的預註冊檢查點，
所以 08-10 那次是**正式判決不是 smoke**。三個方向性檢定全滅——
`cancel_lead_ic`（skew 水平）四個 horizon 沒有一個滿足「CI 離零 ∧ |IC|≥0.02
∧ 兩半同號」（h5 −0.009、h15 −0.007、h30 +0.017、h60 +0.021 且兩半反向）；
`cancel_shock_ic` TEST A（skew 相對自身基線的變化）同樣四格全不過。依 07-10
預註冊的措辭，**bar 級的方向領先主張到此為止**，只剩「擠壓事件條件化」那條
3-6 個月的路。
**唯一活著的是波動**：TEST B（cancel 強度衝擊 → |forward return|）四個 horizon
**全部通過**且兩半一致（h5 +0.115 [+0.096,+0.134]、h60 +0.109），不是 marginal
單格，family-wise caveat 不足以打掉它。但這個結論 2026-07-29 就出現過，
**使用者當時已否決**——系統沒有任何旋鈕接得住波動預測（固定 2x sizing、
3×ATR 停損）。要用它必須先有「波動→sizing/停損」的機制，那是新工程不是新發現。

**V7 的誠實基線從 2026-04 起算，別拿 1~3 月比**（2026-08-08 查明）：
04-03 部署 dual v7 後，Strong 從「13~18% 的 bar」變成「rolling top 5%」——
頻率從 3~4.3 筆/天掉到 0.4~0.9 筆/天**是定義換了，不是衰退**。跨這條線比
勝率（1~3 月 61-73% vs 之後 52-60%）是比不同母體。頻率是 top-5% 定義的
代價，放寬它 = 作廢 Gate A 全部證據 = threshold-sweep 陷阱（2026-06-20）。
時鐘時程照此頻率（~15-18 Strong/月）：地形扳機 60 筆 ≈ **11 月中**、
Gate B 30 筆 ≈ **10 月**。
**模型重訓節律（2026-08-08 起，例行維護非研究）**：復驗 §2b 亮 LEVEL-DRIFT
或部署超過 **60 天** → 走 maintenance refresh（重訓 → `research/
validate_direction_refresh.py` 四關 → 部署）。同特徵、同超參數、同 tier 定義，
warmup buffer 照 04-19 規矩用新生產模型灌。先例：2026-08-08（模型呆了 99 天，
pred 均值漂 +0.0024，7 月開火 14 UP:1 DOWN；重訓後對中 +0.00011）。
三層防護：復驗 §2b（月檢水平）、clocks 2e（週檢開火方向平衡 ≥85% 單邊告警）、
本節律（上限 60 天）。

連帶後果：**opp_signal 出場在低頻時代幾乎餓死**（持倉中出現反向 Strong 的
機率隨頻率崩掉，16 筆裡 11 筆由 trail_stop 收尾）——conviction_decay
（07-25 上線）就是為此設計的替代品，首次真實觸發（08-06 id22，−0.07% 出場
躲掉後續下跌）行為正確。

**V7 兩層在講不同的故事，這個分歧本身是資訊**：訊號層 Strong 全期 59.5%
（n=767）但**近 90 天只有 53.7%（n=54）**，與 2026-06-19 Gate A 乾淨重跑
FAIL（CI 下緣 51.5% < 52%）一致 —— 進場準度在衰退。**2026-08-17 分解
確認這是真衰退不是組成假象**（TODO §0.49b：Oaxaca 分解，組成效應 ≈0、
格內效應 −9.1pp 佔全部；CALM 主場兩個方向同幅下滑 60→52 / 63→54；
「壞解碼灌歪樣本」的嫌疑已排除）——60 天重訓上限因此不可放鬆。而交易層 15 筆勝率只有
46.7% 卻是正報酬（+7.1 bps/筆）—— **現在的正報酬來自出場紀律（trailing 讓
winner 跑），不是進場準度**。加碼與否要看這兩層，不能只看其中一層。

**策略 #2 的誠實註記**：875 筆記錄中**只有 650 筆是真前瞻的**（`first_seen <
exit`），其餘是凍結當下回填的歷史；判 Gate 只能用前者。另外 watchlist 的
C（t=+3.07）與 D（t=+2.24）統計量都比凍結的 B（t=+1.82）強 —— **不得因此改用
C/D**，事後挑統計量最好的變體正是預註冊要擋的事。

**進行中的時鐘**

- **地形濾網上線扳機**：新 Strong **1/60**（首筆 2026-08-04 20:00，miss），
  90d 保留 vs 否決 gap **+17.7pp**（門檻 8pp）→ 兩條同時成立才議進 executor
  · 看板主數字用**已結算**數，訊號開火後要等 ~4h 才會動，旁邊的琥珀色
  「（+N 待結算）」才是剛開火的（操作者為此困惑過兩次，2026-08-05 已改醒目）
- **真實掛單簿**（depth_deltas）：**27/90 天**（394k 筆）→ 十月 L2 檢查點
- **每月 5 號**：`quarterly_revalidation.py` 自動復驗（帶 STALE-DATA guard）

**文件地圖（別人要看哪一份）**

| 想知道 | 看這裡 |
|---|---|
| 策略分工、風控階段、override 紀錄、網站呈現面 | **本檔（CLAUDE.md）** |
| 名詞白話解釋（含地形層、池子四種） | `docs/GLOSSARY.md` |
| DB 45 表目錄（writer/reader/新鮮度） | `docs/DB_REGISTRY.md`（`research/gen_db_registry.py` 重生成）|
| 流動性獵取全貌（變體/配方/評分/上線路徑） | `docs/RAID_PLAYBOOK.md` |
| V7 本體架構細節（資料層→模型→推論） | `docs/系統架構說明書.md` |
| 多策略組合風控設計 | `docs/PORTFOLIO_RISK_FRAMEWORK.md` |
| 踩過的坑（**開工前必讀**） | `.claude/rules/mistake.md` |
| 當前任務、預註冊、凍結假設 | `TODO.md` |

---

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

---

# 決策與 override 歷史

下面每一節都是當時的完整推理與代價自負聲明，**刻意不刪**——這份紀錄本身
就是紀律的一部分（做了什麼、為什麼、放棄了什麼保護）。但章節順序是歷史
堆疊的、不是時間序，所以先給索引：

| 日期 | 決策 | 狀態 |
|---|---|---|
| 2026-05-27 | 研究 + small live 並進（跳過 100 筆 paper gate） | 生效 |
| 2026-05-28 | **10x leverage informed override**（$100 開得起 1 張） | 部分作廢 → 見 06-06 |
| 2026-05-28 | 跳過 testnet，改 read-only smoke（第 2 次 override） | 生效（已完成） |
| 2026-06-05 | **Paper cohort 整個移除**，LIVE 成唯一 cohort | 生效 |
| 2026-06-06 | **分數合約 sizing「B」取代 10x 權宜** | **生效（現行 sizing）** |
| 2026-06-10 | 壓縮版 Stage 3→4：Gate A（統計）+ Gate B（執行）| 生效（Gate A 已過→後又漂移，Gate B 累積中）|
| 2026-07-14 | 資本 top-up $197.55（第 4 次 override）| **已被 07-28 取代** |
| 2026-07-23 | V7 多幣化提前啟動（第 5 次 override）| 已收尾（ETH NO-GO）|
| 2026-07-24 | 資本再加碼 $1218.44（第 6 次 override）| **已被 07-28 取代** |
| 2026-07-25 | conviction_decay 上線，0 shadow 樣本 | 生效（`OKX_CONVICTION_DECAY_BARS=2`）|
| 2026-07-28 | **基準回落 $274**（第二次手動爆倉後；非 override）| **現行基準** |
| 2026-08-21 | **執行面遷移 Bitget（jarvis 產品端）**——OKX executor 維持停機、不再重啟 | **生效（見下）** |

**執行面遷移（2026-08-21 使用者決定）**：「我現在不從 OKX 接了，主要都用
Bitget」。V7 的真錢執行從 flow_system 的 OKX executor 遷到 **jarvis 產品端
（V7Bot on Bitget，訊號走 `/public/signal-feed`，sizing 鎖預算×2 =
V1.20.2 修法）**。後果：(a) OKX executor **維持 CAP-2 HALT 停機狀態即可**，
不做基準 override、不清 ENTRY_PAUSED——kill switch 與對帳照跑，帳上 $776
是使用者資金調度範疇；(b) **Gate B 的 OKX 軌凍結在 21 筆**，執行驗證的
證據來源改為 jarvis 帳本（`raid_trades.jsonl`／V7 perf，產品端 CLAUDE.md
本來就定位它回答「執行管道撐不撐得住」）；(c) `v7_okx_positions` 停止增長，
網站 track-record 的 live 區塊語意隨之凍結（顯示層待議，非急件）；
(d) 風控後果：真錢 V7 現在跑在 jarvis 的軟停損上（60s 輪詢、無交易所端
條件單、部署有 1-2 分鐘盲區）——交易所端 plan order 的優先級因此上升，
已列於 `../jarvis/風控_同向上限_規格.md`。

**讀法**：資本基準只認最後一條（$274）。leverage 只認 2026-06-06 那條
（有效 2x，10x 只是保證金設定）。歷史章節裡的美元數字（$100 / $197 /
$1218）全部是過去式。

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

## Stage 3 資本 top-up 至 $197.55（2026-07-14，第 4 次 informed override）

**背景**：2026-07-13 使用者暫時把 OKX 資金轉出（觸發 CAP-4 DEMOTE——kill switch
分不出 operator 資金調度和策略虧損，見 mistake.md 2026-07-13），轉回時存入
$197.55（原帳上規模 ~$105.15 + 累計損益 ≈ $105.2）。使用者決定**以 $197.55 作為
新的 Stage 3 基準**，不轉出多餘部分。

**這條 override 違反的既有規則**：§當前策略「$100 live = Stage 3 上限，未進
Stage 4a 不准加碼（即使 $100 賺到 $200 也是 $100 keep）」。這次是 deposit 加碼
（$105 → $197.55），不是獲利留存。

**執行的變更**：
- Railway env `OKX_INITIAL_CAPITAL_USD` = 197（kill switch 基準；config 上限 $200，197 剛好過）
- `indicator/okx/report.py` EXECUTOR_RESTART_CAPITAL_USD = 197.55、SINCE = 2026-07-14
  （live-P&L 報表基準，排除 operator 資金移動）

**新基準下的絕對數字**：
- daily cap −20% = −$39.5／total cap −30% = −$59.3（DEMOTE）
- CAP-2 over-funding 上限 = 1.5 × 197 = $295.5
- 2x 名目 sizing ≈ $395 notional／~$39.5 保證金（10x 帳戶槓桿設定下）

**代價自負**：單筆 stop-out 的美元損失放大 ~1.9x；$197.55 貼著 Stage 3 config
硬上限 $200，**這是最後一次 Stage 3 內加碼**——再加就必須走 Gate A+B 通過後的
正式放大（$300-500 一級），不准再用 informed override。

## Stage 3 基準回落至 $274（2026-07-28，第二次手動爆倉後；非 override）

**背景**：2026-07-27 12:00 起，這個帳戶出現一連串 executor 從未下過的手動
交易——13:02 對帳抓到 `orphan_exchange`：**37.11 張 LONG @ 65050**（≈
$24,140 名目，對當時 $1218 權益約 20x）。executor 開的倉一向是 0.31-0.61
張，這筆是它的 60-120 倍。權益從 $1218 一路擺盪到 **$16.62**（−98.6%），
之後入金回到 $274。

**executor 全程沒有下任何單**：最後一筆成交是 id=20（2026-07-16），之後
一直卡在 CAP-2 HALT。kill log 只有 CAP-2 over-funding，沒有任何虧損型
trigger。所以這不是策略虧損、不是 edge 失敗，**也因此不觸發「hit kill
trigger → 降階重驗」**——性質與 [[2026-06-05 手動爆倉]] 完全相同，只是
規模大 6 倍。

**這不是 informed override**：金額是**往下**調整。加碼才需要 override
儀式，減碼一律允許（風險變小）。

**執行的變更**：
- `indicator/okx/config.py`：`initial_capital_usd` 1218.44 → **274.0**
- `indicator/okx/config.py`：live guard 上限 **1500 → 500**。上限的用意是
  擋「沒過 Gate A/B 就把金額往上衝」，它應該待在**現行基準之上一個記錄
  在案的放大級距**（本檔案定為 $300-500），而不是停在一筆已經不存在的
  存款的高水位。要再調高一樣要走 override 儀式。
- `indicator/okx/report.py`：`EXECUTOR_RESTART_CAPITAL_USD` = 274.0、
  `EXECUTOR_RESTART_SINCE` = 2026-07-28（報表基準重置，排除 executor
  沒有參與的那段活動）。**Gate B / shadow 的筆數不重置**，繼續累積。
- Railway env `OKX_INITIAL_CAPITAL_USD` = 274（不改的話 CAP-4 會拿舊基準
  $1218 去比 $274 的權益，開機即 DEMOTE）。
- 測試同步更新並新增 `test_previous_1218_baseline_now_rejected`——釘住舊
  基準，避免一個沒清乾淨的 Railway 環境變數把舊規模悄悄復活。

**新基準下的絕對數字**：daily cap −20% = **−$54.80**／total cap −30% =
**−$82.20**／CAP-2 over-funding 上限 = 1.5 × 274 = **$411**。

**帳戶隔離：正式放棄（2026-07-28 使用者決定，不要再提）**

2026-06-05 那次的結論白紙黑字寫著「hard kill switch 只保護 executor、
擋不了手動單 → **必須帳戶隔離**（executor 專屬子帳戶，操作者永不手動
碰）」，並註明「若同樣的事再發生一次，帳戶隔離就沒有商量餘地」。它在
2026-07-27 以 6 倍規模重演了。

方案已完整提出（建標準子帳戶 → 劃轉 $274 → 持倉模式設雙向 → 產子帳戶
專屬 API key → 換掉 Railway 三個 `OKX_*_LIVE` → 刪主帳戶舊 key；零程式碼
改動，executor 對帳戶無感）。**使用者選擇不做**，理由是返傭綁在主帳戶。
量化過的取捨：機器人一年約 30 筆、$548 名目，手續費 ~$16，就算返傭全損
也只是 **~$5/年**；對照昨日單日損失 $944。使用者知情後仍選擇保留返傭。

**所以這是一個永久開放的風險，不是待辦事項。** 未來 session 不要再重提
隔離——已經提過三次（2026-06-05、2026-06-06、2026-07-28），使用者的答案
一致。要記住的是它的後果：

- **強平是帳戶級的**：手動部位被清算時，executor 若有倉會一起死
- **kill switch 會被手動操作誤觸**：CAP-4 分不出策略虧損和手動虧損，
  一觸即 DEMOTE（2026-07-13 為此卡了整個 session；2026-07-24 起因基準
  未同步卡了 12 天）
- **部位大小由使用者控制的數字決定**：sizing 依帳戶權益，手動盈虧會直接
  改變機器人下多大

看到「帳戶權益異常變動」「orphan_exchange」「莫名 DEMOTE」時，**第一個
假設是手動交易，不是系統故障**——查 `v7_okx_balance_snapshots` 的時間
軌跡（連續擺盪＝持倉盈虧）與 `v7_okx_reconciliation_log`（孤兒倉），
不要從 `v7_okx_positions` 開始（它對手動單是瞎的，見 mistake.md
2026-07-28）。

---

## Stage 3 資本再加碼至 $1218.44（2026-07-24，第 6 次 informed override）

**背景**：使用者在 §Stage 3 資本 top-up 至 $197.55（2026-07-14，第 4 次 override）
明確寫下「這是最後一次 Stage 3 內加碼」之後，又存入更多資金，帳戶餘額查證為
**$1218.44**（相對 $197.55 基準是 6.17 倍）。這次不是交易獲利（同期累計 net
仍是 −1.64%），純粹是使用者主動存款。使用者決定直接把 $1218.44 訂為新的
Stage 3 基準，並繼續維持現有 10x 帳戶槓桿設定（真實風險槓桿 2x，NOTIONAL_LEV_MULT
不變）。

**這條 override 違反的既有規則**：
- §Stage 3 資本 top-up「這是最後一次 Stage 3 內加碼——再加就必須走 Gate A+B
  通過後的正式放大（$300-500 一級），不准再用 informed override」——這次直接
  跳過 $300-500 一級放大，也沒等 Gate A/B 通過。
- 規模已經進入本檔案自訂的 **Stage 4a（$1k 等級）**資金範圍，但 Stage 4a 的
  紀律明講「leverage 必須降回 1.0x」——這次選擇**不降槓桿**，維持 10x 帳戶
  設定 / 2x 有效槓桿。

**發現的技術性阻礙**：`indicator/okx/config.py` 的 `validate_okx_config()`
原本寫死「live 模式 `initial_capital_usd > $200` 就 `raise RuntimeError`」
——這不是文件層級的規則，是真的會讓 executor 啟動失敗的程式碼guard，專門
設計來擋「沒走完 Gate A/B 就把 Stage 3 金額往上衝」這件事。要落地這次
override，**必須先改這段程式碼本身**（不是只改 Railway 環境變數）：上限從
$200 調高到 $1500（保留餘裕但仍是硬上限，不是無限制放行）。

**執行的變更**：
- `indicator/okx/config.py`：`initial_capital_usd` 預設 197.55→1218.44；
  live guard 上限 $200→$1500；通用 sanity 上限 $1000→$1500。
- Railway env `OKX_INITIAL_CAPITAL_USD` = 1218.44（清除 CAP-2 over-funding
  HALT，該 HALT 是因為舊基準 $197 的 1.5x=$295.5 早就被 $1218 帳戶餘額
  觸發，此前已連續 halt 多次）。
- `indicator/okx/report.py` `EXECUTOR_RESTART_CAPITAL_USD` = 1218.44、
  `EXECUTOR_RESTART_SINCE` = 2026-07-24（報表基準重置，排除這筆存款本身
  對報酬率的污染）。
- 槓桿設定**不變**：OKX 帳戶槓桿 10x（僅決定保證金鎖多少）、真實風險槓桿
  仍是 NOTIONAL_LEV_MULT=2x。daily/total loss cap 百分比不變（−20%/−30%），
  但絕對美金數字隨基準放大約 6.17 倍（daily −$243.7 / total −$365.5）。

**代價自負**：
- 單筆 stop-out 的美元損失也放大 ~6.17 倍，遠超 §Stage 3 top-up 段落
  當時評估的「~1.9x」。
- Gate A（訊號方向 edge）目前仍在門檻邊緣反覆（見 §Compressed Stage 4
  Validation 的歷次重跑記錄），Gate B（執行驗證）樣本數也還沒到 30-50 筆
  下限——這次放大**沒有等兩個 Gate 都過**，是純粹基於使用者對這筆存款的
  資金調度決定，不是基於新的 edge 證據。
- $1500 的程式碼上限本身也已經是「留了空間但仍是硬上限」——不是把guard
  整個拔掉。未來若要再加碼超過 $1500，一樣要回來改這段程式碼並寫新的
  override 記錄，不會因為這次改過一次就變得容易複製。

**不受影響（ruin 保護，與這次金額調整無關）**：kill switch 機制本身（daily
/total loss cap 百分比、CAP-2 over-funding 檢查邏輯、CAP-4 total-loss
DEMOTE 邏輯）、max_position_count=1、leverage hard cap 10x。這次只動了
「基準金額」這一個數字，防護機制的結構完全沒變。

---

## conviction_decay 出場機制上線——0 shadow 樣本（2026-07-25）

**背景**：conviction_decay（用進場模型連續原始輸出取代固定 3xATR 停損判斷出場，
見 research/conviction_decay_exit.py）2026-07-24 完成 shadow-mode 部署，
設計上要等真實 live 樣本累積夠了才轉正式（TODO.md 原始任務：「Shadow/dry-run
模式驗證新出場邏輯」→「正式上線——沿用「第一批人工確認」的先例」，兩步驟
分開，先觀察再啟用）。但 shadow-mode 部署後帳戶因為資金基準卡在 CAP-2
HALT（見上一節），整段時間沒有任何真實倉位開過，累積樣本數 = **0**。
使用者知情選擇跳過「等 shadow 樣本」直接正式上線，接受 0 樣本風險。

**這違反的是專案自訂的驗證紀律（code comment 明講「not yet shadow-mode
verified... per this project's 'verify before touching real trades'
discipline」），不是某條寫死的 hard rule**——跟 leverage/capital 那幾次
override 性質不同，這裡是跳過一個計畫中的驗證步驟，不是推翻一個數字上限。

**風險特徵評估（決定用什麼保護機制）**：conviction_decay 只會在**已經開倉**
的真實倉位上觸發出場，不會創造新的曝險——最壞情況是「在不理想的時機平倉一筆
已存在的倉位」，不是「用全新資金開錯方向/錯部位大小的倉」。這跟原本
entry-side 的 ApprovalGate（第一批人工確認）保護的風險類型不同：entry
approval 擋的是部署新資本前的錯誤，可以安全地卡在 Telegram 等回覆（沒送出
訂單 = 沒曝險）；exit 卻是風險已經存在、正在被降低的動作，如果也卡一個
blocking 的人工核准流程，等於讓已經判定該出場的倉位多曝險等操作員回覆，
反而更危險。

**採用的保護機制（不是完整重建 entry 那套 approval round-trip）**：
`indicator/okx/executor.py._maybe_flag_first_conviction_decay`——第一次
真實觸發 conviction_decay 平倉時，在 Telegram 出場告警前面加一段醒目標記
（「🔔 FIRST LIVE conviction_decay EXIT — verify this looks correct」），
讓操作員第一時間看到結果並人工核對，但**不阻擋**平倉動作本身。用
`OkxStateStore.count_closed_by_exit_reason("conviction_decay")` 查詢
是否為第一次（DB 查詢失敗時 fail-open，照常送出不帶標記的告警，不讓
這個輔助檢查變成一個新的單點故障）。4 個新單元測試涵蓋：第一次有標記、
第二次以後沒有、非 conviction_decay 出場完全不查、DB 失敗不影響告警送出。

**執行**：Railway `OKX_CONVICTION_DECAY_BARS=2`（executor.py 已支援此
env var，`load_okx_config_from_env` 讀取，無需改程式碼即可切換）。

**代價自負**：這條 call path（`pred_ret` 從 app.py 傳入 → executor 判斷
streak → 觸發平倉）從沒被真實 OKX 帳戶執行過，第一次真實觸發就是真錢。
Mitigation 是上面的第一次告警旗標，不是 blocking 驗證——如果第一次觸發
的結果看起來不對（平倉時機/方向/金額異常），操作員要**立刻**回來檢查
並視情況把 `OKX_CONVICTION_DECAY_BARS` 改回 0（Railway env var，無需
改程式碼）。

---

## V7 多幣化提前啟動（2026-07-23，第 5 次 informed override）

**背景**：§V7 多幣化可行性研究（本檔案外，見 TODO.md §4.6）原本的紀律鎖是
「BTC 自己的 Gate A 乾淨版都還沒過關（2026-06-19 重跑：n=262、WR 57.6%、
CI 下緣 51.5% < 52% 門檻），多幣化是把現有機制乘以 N，乘的對象要先證明——
production 化討論必須在 Gate A 重跑通過之後」。這條規則本身仍然成立、沒有
被推翻；這次 override 的是**順序**：不再等 Gate A 過關才開始 ETH/SOL 的
Step 2 研究基礎建設（backfill 歷史、建特徵表、跑乾淨 WF），改成現在就平行
推進。

**使用者理由（原話）**：「一直拖都是成本的磨損及消耗，同時做其他幣種不影響
什麼」——等待本身有機會成本（樣本、時間都在流逝），而 §4.6 從頭就定位是
「純 research track，不碰生產」，平行做研究不會對現有 BTC 系統的任何正式
決策造成影響，兩件事不衝突。

**這次 override 不變的部分**：
- BTC 自己的 Gate A 仍然是**唯一**決定「要不要把多幣化推進到 production 化
  討論」的判準——ETH/SOL 就算 Step 2/3 跑出漂亮數字，在 BTC Gate A 乾淨
  過關之前，一律停在 research track，不得進生產、不得用來加碼、不得作為
  「BTC edge 是真的」的替代證據（多幣化證明的是「機制可複製」，不是
  「機制本身有 edge」——兩件事不能互相背書，這正是原始紀律鎖要擋的邏輯
  謬誤）。
- §4.6 原定的 Go/No-Go 判準不變：ETH clean AUC ≥ ~0.54 且與 BTC Strong
  重合率 <50% → 才有資格「繼續」（考慮 SOL、談 production 化）；任一不過
  → 多幣化對 V7 無性價比，資源回異源資料線。

**代價自負**：如果 BTC Gate A 最終沒能重新過關（目前卡在門檻邊緣），這批
提前投入的 ETH/SOL 研究基礎建設就是純沉沒成本——這是明知故犯接受的風險，
換取的是不用等未知長度的時間才能開始累積多幣化這條線自己的證據。

---

# 現行規則與系統

（以下全部是**現在生效**的內容，與上面的歷史章節分開讀。）

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

## 三策略架構（2026-08-02 更新）

這個 repo 已經不只是「V7 指標」，是三條並行的策略線，共用同一個資料層與
風控框架設計（見 docs/PORTFOLIO_RISK_FRAMEWORK.md、TODO.md §0.4）：

| # | 策略 | 現況 | 碰不碰真錢 |
|---|---|---|---|
| 1 | **V7 dual-model**（4h 方向 + 幅度） | Stage 3 live（$274 基準） | **是**，唯一 |
| 2 | **流動性獵取 / 掃單失敗**（sweep-failure） | 規則凍結 + Gate F forward 驗證中；A-E 變體 + 8 組合每小時記帳 | 否 |
| 3 | **撤單流**（cancel playbook） | 方向性判決 FAIL，全線繫於 cancel_lead_ic | 否 |

**硬規則**：策略 #2/#3 在自己的 Gate 通過前，一律停在 research/shadow
track——不得進 executor、不得用來加碼、不得互相背書（「機制可複製」不等於
「機制有 edge」）。第二條策略要上線必須先有統一風控框架（兩層 kill /
風險預算 / 中央曝險帳本 / 相關性預算），不是各跑各的。

### 地形層（V7 × 流動性位置，2026-08-02 戰役收官）

用「訊號開火時，價格離未掃流動性池多遠」當訊號品質背景。10 個維度按凍結
測序逐一過三關（G1 分桶+兩半 → G2 已定案邊際殘餘 → G3 置換+bootstrap+
逐季），結果：

- **定案四維（全是流動性）**：D1 情境 veto（追突破 52% vs 64%）、D2 前方牆
  （≤1.4 ATR 57% vs 淨 65%）、D3 背後支撐（≤1.8 ATR 68%）、D5 池子密度
  （前方 3 ATR ≥3 池 54% vs ≤1 池 62%）
- **門口候選（各一次復審權）**：S3 折價/溢價（CI 下緣恰觸零）、L1-B 清算牆
  （樣本 n=49 攤不出殘餘格）
- **全滅**：市場結構層（S1 方向 / S2 BOS·CHoCH / S3）、D4 牆等級、D6 翻轉位、
  D8 風暴、D9 彈簧、D10 牆齡
- **兩次獨立證明**（D6 翻轉位、L1-A 清算現場）：**被消耗掉的流動性不留下
  任何效應**——系統吃的是還掛在那裡的單，不是價格記憶

**地形目前是 display-only**：告警帶「🗺 地形」標記（`indicator/terrain.py`），
**entry 規則一行都沒動**。要進 executor 必須先過凍結扳機（自 2026-08-02
起 +60 筆新 Strong **且** 90d 保留 vs 否決 gap ≥8pp），達標後由操作者選檔位
（T0-T3），D5 列為下次 policy 修訂的第 4 維候選。

## 系統架構（v7 Dual-Model）
Dual XGBoost 架構：Direction Regressor + Magnitude Regressor，獨立管線。

### 數據層
- **Binance REST API** (3 endpoints)：klines (1h, 500 bars)、depth (L20)、aggTrades
- **Coinglass API v4** (24 endpoints)：15 timeseries + 9 snapshot
- **Deribit Public API** (2 endpoints)：DVOL 波動率指數、Options Summary

### 特徵工程
- **200+ 工程特徵**（Direction 136, Magnitude **76**，且是 Direction 那 136 個
  剪枝後的**真子集**），12 個群組
- 所有計算為 trailing-only（無前視偏差）
- Coinglass 原生 1h 使用 merge_asof 精確對齊
- 自訂 alpha 特徵：impact_asymmetry (IC=-0.071)、post_absorb_breakout (mag IC=0.191)

### 模型
- **Direction Model**：XGBRegressor, 136 特徵, 輸出 pred_return_4h (TWAP path return)，rolling percentile 解碼為 UP/DOWN/NEUTRAL
- **Magnitude Model**：XGBRegressor, **76** 特徵, target = `y_vol_adj_abs`
  = |return_4h| / realized_vol（**σ 單位，不是報酬單位**）；推論時
  `mag_pred = 模型輸出 × realized_vol_20b` 才還原成報酬尺度
  （`inference.py`）。**不參與訊號分級**：`use_mag_gate=False`，見
  §信號生成的 confidence 說明
- **Regime Detection**：CHOPPY / TRENDING_BULL / TRENDING_BEAR / WARMUP

### 信號生成
- Direction: 500-bar rolling percentile 解碼，top 5% → Strong UP，top 15% → Moderate UP（DOWN 同理）
- Absolute |pred| floor (Strong=0.0008, Moderate=0.0005)：低 vol regime 保險，rolling cutoff 比 floor 寬鬆時 floor 接管（2026-05-09 加入）
- Confidence = `min(|pred|/Strong_cutoff, 1.0)^0.6 × 100`（純 |pred| 公式，2026-05-09 移除 mag bonus 因為 OOS 顯示高 mag bar 在模型失靈區）。
  **Strong_cutoff 是「該 bar 自己那一側」的有效門檻（含 floor 與 regime penalty），
  2026-08-13 修正**——原本取 `max(|up|,|dn|)` 的**原始**分位數，buffer 一偏斜就
  用寬的那側去量窄的那側：08-13 的實測 buffer 下，剛觸發 Strong DOWN 的 bar 只拿
  54.4 分（顯示門檻是 80），鏡像的 UP 卻是 100。tier 判定一直是對的，是**印在它
  旁邊的數字**在跟它打架，而且只打擊空側——系統賺錢的那側。現在的不變式：
  **tier=Strong ⟺ confidence=100**（`tests/test_inference.py::TestConfidenceReferenceIsOwnSide`
  釘住，反向證明過）。注意 confidence 的分佈在 08-13 有定義斷點，
  `alpha_decay_monitor.check_confidence_wr_decoupling()` 跨越此點的窗口會混到兩種定義
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
4. **模型評估與交易評估分離**（2026-08-02 修正原文「不做交易績效回測」——
   那句自 2026-05 起就不成立了）：**模型本身**只用 IC / 方向準確率 /
   calibration 判斷，**絕不拿 PnL 回頭調模型或重訓**；但**出場、sizing、
   濾網、策略 #2/#3** 的決策確實走回測 harness（walk-forward + 逐折 +
   bootstrap）。兩者不可混：用 PnL 選模型 = 在小樣本上擬合雜訊。
5. **特徵先回測再加入**：新特徵必須先跑 IC 回測驗證有效才加進系統。
   **2026-06 起追加**：同源資料（OHLCV/Coinglass/Deribit/Binance flow）已
   三度證實飽和，預設**不再跑同源特徵 A/B**；要加就加異源。
6. **Edge Cases 處理**：假日流動性差異、Funding 結算跳動、rate limit、資料缺失。
7. **語意分界要當成樣本下限**（2026-08-13 加入）：模型重訓不是唯一讓舊資料
   失去可比性的事——**產生某個欄位的程式碼改了意義，它之前的每一列就在量
   別的東西**，而「過去唯讀」（絕不重算歷史）代表兩種定義會永遠並存在同一張
   表裡。任何 live 績效查詢的 since 條件都要同時 floor 在**模型部署日**與
   **每一個碰到該欄位的語意分界**：用 `indicator/model_version.py:sample_floor()`，
   不要自己寫日期。現有分界：`DECODE_EPOCH`（2026-08-12 16:00，buffer 從
   in-sample 種子改成 live 重建）、`CONFIDENCE_EPOCH`（2026-08-13，confidence
   換分母）。**加分界時要順手檢查它有沒有讓某個告警變成永遠靜默**——
   2e 就因此補了一條「超過 14 天還沒樣本本身就是異常」，否則「樣本不夠」
   和「解碼又鎖死了」會印出同一行字。
8. **驗證儀式不可事後放寬**（2026-08 地形戰役定型）：先寫預測再看數據；
   分桶要全格報告不挑格；門檻/分桶定義寫死後不因為「差一點」而改
   （S3 差 0.0 就是差 0.0）；跟先驗矛盾**或**完全符合先驗的漂亮結果，
   都要先查產生它的程式碼。

## 圖表同步規則
**V7 有兩個圖表，修改時必須同步更新**：
1. **靜態圖表** (`indicator/chart_renderer.py`) — Telegram 推送的 PNG
2. **互動圖表** (`indicator/chart_interactive.py`) — `/ichart` 的 TradingView Lightweight Charts HTML

任何 V7 圖表邏輯變更（面板、三角形、顏色、過濾條件）都要兩邊一起改。

第三張圖屬於策略 #2、**不與上面兩張同步**（不同資料源、不同語意）：
3. **獵取覆盤** (`research/sweep_failure/shadow_review.py`) — `/shadow-review`
   的多幣種 K 線 + 變體階梯進出場 + 累積 netR 曲線（5 變體 + 8 組合）

## 使用者可見改動的同步規則（2026-07-23 起）
V7 或撤單流只要有**使用者看得到**的改動（新圖表、新指令、新幣種、新研究
結論上牆），要主動同步三處：**product-site**（`../product-site`，Next.js /
Vercel，分支是 **master** 不是 main）、**兩個 Telegram bot**。純研究腳本 /
後端管線改動不適用。

## 對外網站呈現面（product-site，2026-08-02 盤點）

網站是三條策略**唯一的對外展示層**。資料一律走 agent-mcp 的 `/public/*`
唯讀端點（Railway `agent-mcp-production-46d7`），網站**不直連 MySQL、不碰
任何交易路徑**——這條界線由 `.claude/rules/agent-boundary.md` 管，網站只是
它下游的下游。

| 策略 | 頁面 | 主要元件 | 吃的端點 |
|---|---|---|---|
| **V7** | `/charts/v7`、`/dashboard`、`/signals`、`/track-record` | ChartDetail、V7KpiRow、**V7FilterCard**（地形四維＋扳機進度）、LiveTradesPanel | `/public/chart`、`/live-chart`、`/signal-feed`、`/signal-history`、`/live-status`、`/track-record` |
| **流動性獵取** | `/charts/liquidity`、`/dashboard` | ChartDetail（獵取覆盤圖）、SweepKpiRow、**ShadowLedgerBoard**（5 變體 + 8 組合 + 時鐘）、ShadowTradesPanel | `/public/liquidity-map`、`/public/sweep-status` |
| **撤單流** | `/charts/cancel-flow`、`/dashboard` | CancelFlowExpert、CancelFlowKpiGrid | `/public/cancel-flow-chart`、`-chart-i`、`-stats` |
| 共通 | `/`、`/system`、`/incidents`、`/writeups`、登入註冊 | Hero、StrategyBoard、SystemDetail、Waitlist | `/public/login`、`/register`、`/waitlist` |

**公開面硬規則（違反就是資訊外洩，不是 UI 問題）**：
- **只出百分比、方向、時間**——絕不出現合約張數、美元權益、帳戶餘額、
  單筆部位金額（`queries.public_live_status` 就是照這條寫的）
- **只出模型輸出**（方向 / tier / 信心 / 驅動因子），不出模型內部（特徵
  定義、cutoff、權重）
- 任何可被讀成投資建議的回應都要帶 disclaimer 欄位
- **研究結論上牆必須標狀態**：已驗證 / 待整合 / 門口候選 / 已陣亡。像 D5
  這種「過了三關但還沒進生產」的，卡片上要有明確標記（現在是虛線框 +
  琥珀「待整合」chip），不能讓頁面暗示它已生效

## 命名與程式碼規範
- Class：CamelCase（如 IndicatorEngine、SignalExplainer）
- 函數/變數：snake_case（如 build_live_features、backfill_mag_pred）
- 偏好：清晰、可讀性高、模組化
- 新特徵加入前必須回測驗證 IC

## 專案階段（2026-08-02 更新）
- **V7 特徵工程 = 飽和**：同源資料（OHLCV + Coinglass + Deribit + Binance
  order flow）已三度證實榨乾（WQ101、liq proxy、86 個新特徵全部 A/B 不過）。
  預設**不再跑同源特徵 A/B**；唯一槓桿是異源（options GEX / on-chain whale /
  真實掛單簿 depth_deltas，10 月檢查點）
- **V7 模型**：維持現役，每月 5 號自動復驗（`quarterly_revalidation.py`，
  帶 STALE-DATA guard）
- **當前研究重心**：策略 #2 的 forward 驗證（Gate F / 變體 B 1400 筆時鐘）
  與統一風控框架設計；V7 這側是地形層的凍結扳機在跑
- **持續運行**：績效追蹤、IC 監控、衰退警報、每小時 shadow 記帳

## 跨 session 任務同步（2026-07-07）
- **TODO.md 是唯一的跨 session 任務真相源**。每次開工先讀 TODO.md 的「當前任務」區。
- Session 內建任務清單（TaskCreate）只作單次對話的進度追蹤——它存在本機
  session 狀態、不進 git、不跨機器。凡是隔天/換機器還要做的事，寫進 TODO.md 並 push。
