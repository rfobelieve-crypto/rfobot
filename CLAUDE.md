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

> ### 📁 **開工前先知道：大資料不在 repo 裡，在 D 槽（2026-09-11）**
>
> `market_data/raw_data` 與 `research/poc/data` **是指向 D 槽的目錄連結**
> （junction），不是真的資料夾。**程式路徑完全沒變**，所以你不用改任何
> `ROOT / "market_data" / "raw_data"` 的寫法——但**如果 D 槽沒掛載，
> 那兩個路徑會變成空目錄，而每一支讀它的程式都會安靜地讀到「零列」**。
>
> 症狀是「資料突然全沒了但沒有任何錯誤」時，**第一個檢查是連結**：
> `python research/ops/data_manifest.py`（它有一道 `check_junctions()`，
> 連結斷了／目標不在／透過連結看到 0 個檔 都會紅）。
> 細節與當時的驗收在本檔 §大資料在 D 槽。

**實盤參數（V7，Stage 3）**

| 項目 | 現行值 | 出處 |
|---|---|---|
| 資本基準 | **$311.60**（2026-07-28 重置；08-08 修正——$274 是入金中途讀的，snapshots+id21 equity_before 證明真實起點 311.60） | `okx/config.py: initial_capital_usd` |
| 策略有效槓桿 | **2x**（名目 = 2 × equity） | `NOTIONAL_LEV_MULT`；guard 上限 3.0 |
| OKX 帳戶槓桿 | 10x（**只決定鎖多少保證金**，非策略風險） | `config.leverage` |
| Daily / Total kill | **−20% / −30%**（≈ −$62.3 / −$93.5） | `daily_/total_loss_cap_pct` |
| 同時持倉 | **1 筆** | `max_position_count` |
| 出場 | 3×ATR trailing、opp_signal 反向、conviction_decay(2 根) | `okx/executor.py` |

**OKX 帳戶自 2026-08-18 起為 $0、executor 2026-09-05 DEMOTED（2026-09-06 查明）**：
`v7_okx_balance_snapshots` 顯示權益在 08-18 14:15:39 **一秒內**從 >$1 階躍到 $0——
提領，對上 08-21「改用 Bitget」的決定。之後 executor 對空帳戶報了 18 天 $0
（WS 活著、看板綠著）。09-05 的 Railway 重新部署讓它重跑 kill 檢查 → CAP-4
−100% → **DEMOTE 終態、WS 停止**。這是設計行為不是故障；freshness 那列已退役
（門檻拉高、列保留），帳戶再入金時改回。下面那段 $776 的描述是 08-13 的快照，
**已過時**。連帶：09-05 Telegram 帳號被盜時 OKX 早已是空的，攻擊者無物可取，
API 金鑰到 09-05 08:31 UTC 仍正常讀到餘額。

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
| **流動性獵取**（策略 #2·舊線） | **整條線結案（2026-09-07）**——歷史 edge 是**成交假設**造出來的 | 凍結引擎在 57.9% 的交易上記「成交在價位」，市場當時距價位中位 42.6 bps，那個價格拿不到。**六條進場路徑 + 48 格出場結構 + 兩個 A/B 判別器候選全負**（TODO §1.02）。變體 B FAIL(09-02)、C/D 連坐、**M 連坐作廢**（M 另有一個病：註冊後從未有計分器，0/400 永遠不會動）。A 與 §0.59、§0.474b E 三個時鐘**跑完當歷史紀錄**，看板已註記「此時鐘測的是已知不可執行的進場價」。**訊號有效性沒有被推翻**（engine_audit 六項全過），壞的是交易設計。**後繼是新線 SDV**（TODO §1.03，不經過此成交假設；2026-09-09 前的文件裡叫「交會事件」）|
| **SDV**（策略 #2b·新線） | **意圖層 HALTED，實盤未上**（2026-09-09；**HALT 的理由已於 2026-09-11 被更正，見 §SDV 專節的警示框與 TODO §1.03r**） | 樣本外 S+D+V（n=877、458 天、返佣後）每筆 **+0.1833 ATR**、CI [−0.132, +0.550]、**9/9 幣**、**P(優勢>0) 85.0%**、勝率 47.9%、PF 1.191、年化夏普 +0.93、樣本外/樣本內 **36%**。回落（2x／3 槽，1500 條重抽路徑）p50 **43.3%**、p95 59.7%、**P(回落>50%) 24.1%**（活數字跑 `research/poc/conj_bet.py`）。**出場網格 15 格沒有任何一格 CI 下緣越過零**。前瞻時鐘：全簽章 3/300、S+D+V 1/200。方向仍是動能（B1–B4 四個判別臂樣本外全滅）|
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

**Strong 開火率現況 13.3%，設計值 5%（2026-08-24 量測，非故障）**：解碼
修法把窗口從 500 縮到 200 之後，Strong 佔比走成 07 月 2.0% → 暖機期 5.7%
→ **解禁後 13.3%**（270 根 bar / 36 筆）。機制：rolling percentile 保證的是
「相對**過去** 200 根的 5%」，這波崩盤＋暴漲讓 pred 幅度暴增，當前預測
頻繁越過用平靜期算出的切點——**分佈擴張期的必然結果，不是門檻被人放寬**。

**不要因此收緊門檻**：同日量測（全歷史 n=113 五分位）顯示 |pred| 幅度與
勝率**無單調關係**（Q1 最強 54.5% / Q3 59.1% / Q5 最弱 64.0%），與
2026-05-09 移除 confidence 的 mag bonus、[[project_high_y_failure_not_vol]]
兩個既有判決一致——**極端預測不是更好的預測**。所以 13.3% 是「訊號變多」
不是「訊號變差」，收緊只會少交易、不會提高準度。真正提高進場品質的手段
是地形濾網（外部資訊過濾），不是調自己的門檻。

連帶注意：訊號變多會讓 jarvis V7Bot（單倉）的 `acted`/`age` skip 暴增，
並提高 opp_signal 反手頻率——這是頻率的下游效應，不是錯誤。

**V7 兩層在講不同的故事，這個分歧本身是資訊**：訊號層 Strong 全期 59.5%
（n=767）但**近 90 天只有 53.7%（n=54）**，與 2026-06-19 Gate A 乾淨重跑
FAIL（CI 下緣 51.5% < 52%）一致 —— 進場準度在衰退。**2026-08-17 分解
確認這是真衰退不是組成假象**（TODO §0.49b：Oaxaca 分解，組成效應 ≈0、
格內效應 −9.1pp 佔全部；CALM 主場兩個方向同幅下滑 60→52 / 63→54；
「壞解碼灌歪樣本」的嫌疑已排除）——60 天重訓上限因此不可放鬆。而交易層 15 筆勝率只有
46.7% 卻是正報酬（+7.1 bps/筆）—— **現在的正報酬來自出場紀律（trailing 讓
winner 跑），不是進場準度**。加碼與否要看這兩層，不能只看其中一層。

**策略 #2 的誠實註記（2026-09-02 改正）**：本節舊版寫「只有 `first_seen <
exit` 的列是真前瞻，判 Gate 只能用前者」——**這條是錯的，已撤回**。判決時
查證：被那條濾掉的列不是回填歷史，是排程斷線後的補記塊（無偏）加上
**同一小時內成交即停損的輸家**（meanR ≈ −1.0）；濾網剛好把死最快的交易濾掉，
是存活偏誤。照它切 B 會從 FAIL 變成 +0.077「PASS」，這正是它不能用的證據。
**判 Gate 的依據是註冊計分器本來的口徑：凍結日之後全部已平倉列**
（`shadow_engine.gate_stats` / `sweep_forward.py`）。另外 watchlist 的 C/D
統計量曾比 B 好看 —— **不得因此改用 C/D**，事後挑統計量最好的變體（或最好看
的子集）正是預註冊要擋的事；09-02 起 C/D 已隨 B 連坐作廢。

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
| 策略分工、風控階段、**現行** override、網站呈現面 | **本檔（CLAUDE.md）** |
| **已被取代的 override 全文**（10 條，2026-09-11 搬出） | `docs/DECISION_HISTORY.md` |
| 名詞白話解釋（含地形層、池子四種） | `docs/GLOSSARY.md` |
| DB 45 表目錄（writer/reader/新鮮度） | `docs/DB_REGISTRY.md`（`research/gen_db_registry.py` 重生成）|
| 流動性獵取全貌（變體/配方/評分/上線路徑） | `docs/RAID_PLAYBOOK.md` |
| V7 本體架構細節（資料層→模型→推論） | `docs/系統架構說明書.md` |
| 多策略組合風控設計 | `docs/PORTFOLIO_RISK_FRAMEWORK.md` |
| **祕密清冊、輪替手冊、事故紀錄** | `docs/SECURITY.md`（2026-09-05 起，新增祕密同 session 就要登記）|
| **資料實際放在哪（D 槽目錄連結、不可回填清單）** | 本檔 §大資料在 D 槽 ＋ `research/ops/data_manifest.py` |
| 踩過的坑（**開工前必讀**） | `.claude/rules/mistake.md` |
| 策略／因子研究的十道檢查（**開新研究線前必讀**） | `.claude/rules/factor-research.md` |
| 當前任務、預註冊、凍結假設 | `TODO.md` |
| **外部閱讀轉譯成「我們要驗什麼」**（2026-09-11 起） | `docs/external_reading.md` |
| **已結案判決的橫向讀法（共同死因，全是假說不是判決）** | `docs/common_cause_scan.md` |

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

> **已被取代的那幾條，全文搬到 `docs/DECISION_HISTORY.md`**（2026-09-11）。
> 上面這張索引表**留在這裡**，因為它才是地圖；搬走的是內文。
> 標「生效」的那幾條（06-06 sizing、07-28 基準、08-21 遷移 Bitget）一個字沒動。

**讀法**：資本基準只認最後一條（$274）。leverage 只認 2026-06-06 那條
（有效 2x，10x 只是保證金設定）。歷史章節裡的美元數字（$100 / $197 /
$1218）全部是過去式。

## SDV 小額實盤執行測試（2026-09-08，第 7 次 informed override）

> 這一節寫於命名之前，內文的「交會事件」就是 **SDV**（2026-09-09 使用者
> 命名）。內文措辭保留原樣不改寫——那是當時的決定紀錄。


**使用者決定（原話）**：「我的判斷是現在就上實盤小資金測試才會最準，
你一直準備不出戰場永遠不會進步」。並在三個選項上明確選了
**Bitget（jarvis 在用的帳戶）／$300-500／直接全自動**。

**這條 override 違反的既有硬規則**：§三策略架構寫著「策略 #2/#3 在自己的
Gate 通過前，一律停在 research/shadow track——**不得進 executor**、不得用來
加碼、不得互相背書」。交會事件（TODO §1.03）的前瞻時鐘是 **0/300**，
離判決還有約四個月。

**為什麼接受這條，而且它不是「跳過驗證」**

這次要買的東西**統計上買不到**，只能用真錢買：

| 問題 | 誰能回答 |
|---|---|
| edge 存不存在 | 時鐘（0/300）。**實盤在這題上嚴格更差**——事件率一樣是 2-3/天，但每筆多了成交價與滑價的變異，同樣樣本數下結論更慢 |
| **真實成交價拿不拿得到** | **只有實盤**。shadow 停在「偵測到」，送單→成交那一哩不存在 |
| 停損單在交易所端怎麼成交 | 只有實盤。29% 的交易會觸發它 |

而**舊線正是死在這一題上**（§1.02）：跑了兩年半回測、開了三個時鐘，
第四個月才發現回測假設的成交價市場不給。**這次提前把它做掉，
是把舊線的教訓執行出來，不是重犯。**

jarvis 自己的 CLAUDE.md 第 204 行本來就寫著同一件事：
「產品端與研究線**並行**跑：不等策略驗證完才上線，那天不會來。」

**2026-09-09 暫停執行（前視判決，TODO §1.03b）**：`conj_redef.py` 查出
進場定義是前視的——`et.cluster` 的錨點是群內**最早**那一分鐘，而交會事件
要到最後一個成分到齊才成立；進場（錨點+2）落在事件成立**之前** 22.3%，
那 671 筆 +0.5064 誠實化後只剩 +0.1094，全體 +0.2286 -> +0.1157。
改成誠實錨點逐格重跑，**沒有任何可交易延遲的淨值 CI 下緣 > 0**
（delay 1/2/3/5/10 淨 −0.043 ~ −0.060、逐幣 2-3/9）。
**`conj_watch` 意圖層已停（`INTENTS_ENABLED=False`），小額實盤暫停。**

> **⚠ 2026-09-11 更正（TODO §1.03r）：上面那一句「沒有任何可交易延遲過閘」
> 跑在一組當天稍後就被取代的出場設定上，不成立。**
> `conj_redef.py` 當時用 `STOP=1.0 / HOLD=60` ＋ 成本 `7/3/10`，而同一天的
> §1.03d/f 把它判為「錯的出場設定」（定案 `STOP=3.0 / HOLD=480`、
> 返佣後成本 `1/1/3`），**§1.03b 從未用定案參數重跑**。它同時跑在
> **母體** S∧(D∨V) 上，而現行規格是 S∧D∧V。
> 重跑（`conj_redef.py --decided --sig and`，預設值未動，D1 改為跨儀器對照
> 並 PASS）：delay 1/2/3/5 **全部**淨 CI 下緣 > 0、**9/9 幣**，
> **最大可用 delay = 5 分鐘**；全期每筆 +0.3317（樣本外 +0.1833、
> CI [−0.132, +0.550]、P(優勢>0) 85%）。
> **HALT 仍然維持** —— 恢復要三條同時成立，這只滿足了第二條；
> 第一條（`conj_watch` 的錨點實際改了沒）要查，第三條是使用者的決定。
>
> **2026-09-11 稍晚補查，上面這兩句要更正：**
> 第一條**已經做了**（`conj_watch.assemble` 用 `ready = max(第一根掃單,
> 第一個流量)`，而 `ENTRY_DELAY_MIN = 3`）。第二條**取決於哪一半，而當初
> 沒寫**：`conj_redef` 沒有樣本外切分，它量的是全期——全期口徑下
> delay 1/2/3/5 都過（9/9 幣），但**樣本外那半 +0.1833、CI [−0.132,+0.550]
> 仍然跨零**。照核心原則 9，主句是後者，所以**第二條在樣本外沒有過**，
> HALT 不動。`conj_watch.py` 的恢復條件區塊有完整的逐格表與同一段更正。
>
> 順帶一個會被誤用的數字：「最大可用 delay = 5 分鐘」靠的是 **+0.0003**
> 的 CI 下緣——那不是餘裕，是剛好沒跨零。穩健的是 1–3 分鐘。
毛利仍為正且 CI 離零 —— 訊號有效性未被推翻，壞的是 edge 比成本小
（§1.02 舊線同一種結局）。恢復要三條同時成立：錨點改 ready ∧ conj_redef
有一格過閘 ∧ 回來改本節。

**判準與紀律（寫死，事後不放寬）**

- **本測試只回答執行問題，其損益不得作為 edge 證據。** 20-30 筆實盤損益
  絕不可用來支持或推翻 0/300 那個時鐘——那是 §0.92 判掉 C/D 變體的
  同一種錯（事後挑統計量最好的樣本）。時鐘的計分器不吃這批資料。
- 唯一的判準：**實際成交價 vs 意圖價（`open(a+2)`）差幾 bps**
  - ≤ 2 bps → 成本模型站得住，2 分鐘死線的算術有效
  - ≥ 5 bps → 死線與淨值全部要重算
- 目標樣本 **20-30 筆成交**（以 2-3 筆/天計約 10-14 天）

**代價自負（使用者知情後選擇的）**

- **帳戶與 V7 共用**。這個專案的共用帳戶已經被手動單爆過兩次
  （2026-06-05、2026-07-27），而 kill switch 分不出虧損是誰造成的。
  我在選項描述裡標明了這一點，使用者仍選 Bitget。
- 統一風控框架（TODO §0.4：兩層 kill／風險預算／中央曝險帳本／
  相關性預算）**還不存在**。這是「第二條策略上線」的前置條件，本次跳過。
- **第一次送單就是真錢**——這條 call path 從未被執行過。使用者選了
  「直接全自動」而非「前 3 筆人工確認」，理由是後者會因為事件在
  凌晨發生而漏樣本、造成時段選擇偏誤（那個偏誤是真的，見下）。
- 緩解：sizing 壓到遠低於 1x 名目、最大同時持倉 2 筆、總虧損上限、
  意圖與成交逐筆對帳、每筆都留 intended vs actual。

**為什麼不能用人工下單**：事件 24/7 發生、每天 2-3 個。人工只吃得到清醒
時段，而時段與流動性相關——20-30 筆若全是白天的，量出來的成交價品質是
偏的，那正好毀掉本測試唯一要買的東西。

**2026-09-08 同日實盤體檢（使用者要求）補了六項，全部在研究端**：停損與出場改成
**相對成交**（`stop_dist` / `hold_ms`，產品端套在 fill 上）；停損 ATR 改用
每分鐘配方（每日表低估 19%）；`intent_id` 去重；`agent_conj_fills` ＋
`POST /public/conj-fill`（`CONJ_FILL_TOKEN`，fail-closed）承接成交回報並
形成端點不再重吐的閉環；`intent_gate` 不再把過期未送的 NEW 算成持倉。
細節與表格在 TODO §1.03「實盤執行體檢」。**操作者上線前要設
`CONJ_FILL_TOKEN`**（agent-mcp 與 jarvis 同值）。

**執行架構（不新增第二份下單層）**

    flow_system   偵測 + 算訂單意圖 -> DB（`conj_intents`）
    agent-mcp     /public/conj-signals（唯讀端點）
    jarvis        conjbot.js -> Bitget（**沿用既有的 src/exchange/bg**）

在 flow_system 再寫一份 Python 下單層是被禁止的——那是第二份實作，
而且是**下單層**（本 session 光偵測層的第二份實作就咬了五次）。

---

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
| 2 | **流動性獵取 / 掃單失敗**（sweep-failure）·**舊線** | **結案 2026-09-07（TODO §1.02）**：交易設計不可執行，訊號有效性未被推翻。三個時鐘跑完當紀錄 | 否 |
| 2b | **SDV**（新線，即原「交會事件」） | 見下方專節 | 否（意圖層 HALTED）|
| 3 | **撤單流**（cancel playbook） | 方向性判決 FAIL，全線繫於 cancel_lead_ic | 否 |
| 4 | **兩場館套利**（獨立線，2026-08-28） | 只錄不做，09-04 判決（TODO §0.75） | 否 |
| 5 | **鏈上量化**（perp DEX，2026-09-10 起） | 只錄不做，判準未寫（TODO §1.10） | 否 |

### 大資料在 D 槽，原位是**目錄連結**（2026-09-11）

使用者：「只搬資料就好，已不影響系統為主」。所以**沒有搬程式**——
程式碼、git、12 個 Windows 排程、26 個硬編絕對路徑的檔案**一個都沒動**。

| 原路徑（程式照舊用這個） | 實體位置 | 大小 |
|---|---|---|
| `market_data/raw_data` | `D:lowbot_data
aw_data` | 33.6 GB |
| `research/poc/data` | `D:lowbot_data\poc_data` | 5.9 GB |
| （本來就在 D 槽）| `D:lowbot_data\hl\{trades,mid}` | 成長中 |

`flow_system` 從 **41,227 MB 縮到 1,890 MB**，C 槽可用從 87 GB 到 **125.5 GB**。
**77 個引用 `raw_data` 的檔案一行都沒改**，因為路徑對它們完全沒變。

**連結的失效方式是安靜的**：D 槽沒掛載時連結變成一個**空目錄**，
於是每一支讀它的程式都讀到「零列」——而零列在很多地方是合法狀態。
所以 `research/ops/data_manifest.py` 有一道 `check_junctions()`：
連結不在、目標不在、或**透過連結看到 0 個檔**都算紅，而且**已經接進
`main()` 並反向證明過**（把目標改名 -> 紅；改回來 -> 綠）。

**驗收用的是產物不是退出碼**：搬完之後拿凍結的計分器
（`research/poc/conj_redef.py`）實跑一次，D1 已知答案對照
**+0.2275 與搬家前完全相同**。

> 第一次搬的 robocopy **一個檔都沒複製而退出碼是 0** ——
> Git Bash 把 `/E` 轉成了 `E:/`（mistake.md 2026-08-26）。
> 是「搬前記指紋、搬後對指紋」抓到的。**在 Git Bash 裡呼叫 Windows
> 原生程式一律改走 PowerShell。**

### 鏈上錄製集（2026-09-11，**五樣裡四樣不可回填**）

| 錄什麼 | 怎麼錄 | 可回填？ |
|---|---|---|
| 逐地址部位、清算價、觸發單 | 每小時 REST，2,480 地址 | **否**（無歷史端點） |
| L2 簿口（20 檔，帶每檔張數 `n`） | 每小時 REST | **否** |
| 全市場成交帶（234 幣，雙方地址） | 常駐 WS | **否** |
| **分鐘級中價與佇列**（前 40 名） | 常駐 WS，牆鐘 60 秒取樣 | **否** |
| 歷史 K 線 | REST | **是，但端點只保留 5000 根/週期** |

**保留牆是按根數算的，所以週期越細歷史越短**：1h = 208 天、15m = 52 天、
5m = 17 天、**1m 只有 3.5 天**。這就是成交帶與中價不可取代的原因。

**為什麼要另外錄中價（2026-09-11 加）**：報酬目標用**成交價**算會被買賣價
跳動污染，薄的標的會呈現**比實際強得多的反轉**。而 HL 尾 50 名的頂檔價差
中位 **38.5 bps**（前 30 名只有 2.9）。§4.65 的「分鐘級是反著做」是在
BTC/ETH 上量的，搬到 HL 長尾會中招。宇宙取前 40 名＝**98.3% 的日成交額**，
理由是「跳動偏誤最嚴重的地方，正好是沒有量也交易不了的地方」。
細節見 mistake.md 2026-09-11。

**新錄製器的三道接線（缺一個就在某個方向上隱形）**：
freshness（會變紅）、`exit_paths_watchdog.ps1`（會自己回來）、
`data_manifest`（清冊看得到）。2026-09-11 成交帶死兩小時而 48 列全綠，
就是只缺了第一個。

### HFT：已宣告方向，**先 MFT 後 HFT**（TODO §1.19）

使用者 2026-09-11：「我打算也要做 HFT 了但先從 MFT 開始」。
這個名詞底下有兩個生意，**我們只進得去一個**：

- **延遲競賽 — 進不去。** 頂檔報價現在是 Jump / JS / IMC / XTX / HRT /
  Citadel / Tower 在打，而我們在 HL 上沒有等價的主機位置可買。
- **高頻資料上的研究方法 — 開著，而且產出回流到 MFT。**

**一個硬相依**：HFT alpha 強度太弱，**吃單吃不動，只能當做市報價的偏移**。
所以沒有做市系統，HFT 研究產不出可交易的東西。**MFT 沒有這個相依**
（alpha 夠強可以直接吃單）——這就是「先 MFT」為什麼是對的。

> **⚠ 2026-09-12 更正（TODO §1.27 / §1.28）：括號裡那句「alpha 夠強可以直接
> 吃單」被量掉了。** 吃單淨 **−2.42 bps 每單位成交量**（毛利 +0.58、費 3.00）。
> 所以 **MFT 跟 HFT 是同一個相依，不是兩個**——兩者都吃單吃不動。
>
> 而「改掛限價單」不是那個相依的出口：實測被動執行要付 **7.27 bps/單位成交量**
> 的放棄邊際（漏掉的成交正好是行情最大的那些小時，首次通過的必然結果），
> 加上掛單費 1.00，門檻變成 **8.27** —— 比吃單的 3.00 更高。
>
> 三個門檻：吃單 **3.00**／naive 掛單 **8.27**／**倉位型做市 ≈ 0.5**
> （Advanced MM 的「偏移進部位約少賺 0.5 bps」，未經我們驗證）。
> 我們的毛利是 **0.41**（無抑制）~ 3.9（重抑制，但測不動）。
> **把門檻從 3.0 降到 0.5 是 6 倍，而沒有任何抑制設定能可靠地把分子拉 6 倍。**
>
> 結論：「先 MFT」的**順序仍然對**（MFT 的研究循環快、資料已在手），
> 但理由換了——不再是「MFT 不需要做市系統」，而是
> **兩條線都需要它，而 MFT 是先把「需要它」這件事量出來的那一條**。
> 範圍限制：以上都在 Binance 11 個厚標的（半價差中位 0.50 bps）上量的；
> HL 薄標的半價差 ~19 bps，掛單的算術在那裡可能反轉（09-18 有資料）。

### SDV —— 這條線的正式名稱（2026-09-09 使用者命名）

**使用者原話：「稱他為 SDV 好了」。** 從此對內對外都叫 **SDV**，
`交會事件` / `交會線` 是它的舊稱，**歷史章節與 TODO 判決節裡的舊稱刻意不改**
（那是當時的紀錄，改了就是竄改），但**新寫的東西一律用 SDV**。

名字的來源就是它的定義，這是它比舊稱好的地方——**舊稱只說「有兩件事同時
發生」，新名字直接寫出是哪三件**：

    S  sweep      掃單：價格穿過一個還沒被消耗的樞紐價位
    D  delta_ext  主動量極端：五分鐘 |delta| 後向和越過滾動 30 日 p99
    V  vol_burst  量能爆發：五分鐘量／同時段 30 日均值越過滾動 30 日 p99

**S ∧ (D ∨ V) 是母體，S ∧ D ∧ V 才是現行規格。** 三者齊發那一格
（程式碼裡的 `sigk == "and"`）是唯一撐得住樣本外的；S+V 單獨為負、
S+D 樣本薄。所以「SDV」指的**就是三者齊發那一格**，不是整個母體——
講到母體要明說「SDV 母體」或「S∧(D∨V)」。

可交易時刻 = **ready**，也就是最後一個成分到齊那一分鐘，不是群內最早
那一分鐘（後者是前視，2026-09-09 判決，§1.03b）。

**不得挑幣（2026-09-11，TODO §1.03s）**：使用者發現 SOL 看起來不錯。
誠實驗證（只用前半挑、看後半）說**逐幣排名是雜訊**——只用前半會挑到 ADA，
而 ADA 在後半是 9/9 最後一名且為負；前後半排名 Spearman **−0.267**。
SOL 自己單幣 CI95 **[−0.079, +0.696] 跨零**、全期排名 5/9。
所以 SDV 一律**九幣等權**，不為任何單一幣加碼、不只做某一幣——
那是 §0.92 判掉 C/D 變體的同一種錯。

命名紀律照 [[feedback_plain_language]]：這是**使用者定的詞**，所以可用；
我不得再自行發明別的簡稱（不要出現「SDV-A」「SDV+」這種我自創的變體名，
要新名字先問）。

**硬規則**：策略 #2/#3 在自己的 Gate 通過前，一律停在 research/shadow
track——不得進 executor、不得用來加碼、不得互相背書（「機制可複製」不等於
「機制有 edge」）。

**第 4 線的隔離（2026-08-28 使用者要求；2026-09-04 整條線搬出本 repo）**：
套利不預測市場，跟前三條本質不同——永不 import 交易路徑、資金完全獨立
（未來就算接錢包也不進 OKX/Bitget 帳戶體系）、只共用進度看板與新鮮度
檢查兩個唯讀顯示層。定位是戰役型（一個戰場吃幾週到幾個月就換），不是常駐線。

**現在它有自己的 repo**：`../arb`（GitHub `rfobelieve-crypto/flowbot-arb`，
私有）。`engine/` 是錄價與執行引擎、`arblib/` 是判斷層、`docs/`、`ops/`、
`results/`。分離不是整理是風控——這條線將來要碰**自己的帳戶與憑證**，
而這個 repo 的帳戶已經被手動單爆過兩次。

**本 repo 只剩兩座橋，方向都是單向（讀它，不被它讀）**：
`research/arb_publish.py`（讀 `results/*.json` ＋ `import arblib` 的成本
模型 → 寫 MySQL 給網站）、`research/freshness_board.py` 與
`prereg_publish.py`（讀 `engine/logs/*/minutes.csv` 的 mtime 與行數）。
**只有一個檔案知道它在哪：`research/arb_home.py`**（`ARB_HOME` 可覆寫，
引擎搬去 VPS 時改那一行）。arb repo **完全不碰本 repo 的 MySQL**。

`basis_recorder.py` / `basis_verdict.py`（§0.91 站內資金費）**留在本 repo**
——它們零依賴 `arblib`、純寫 MySQL，而且那是產品端的站內請求，不是 §0.75
跨場館家族。第二條策略要上線必須先有統一風控框架（兩層 kill /
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
9. **報告順序：樣本外先講，樣本內只放括號裡**（2026-09-09 使用者訂立）。
   使用者原話：「每次跑完回測一開始都很好，驗證後又打回原形」。
   查了一整天的紀錄，**那個「一開始很好」有一半是報告順序造成的**——
   把還沒驗過的樣本內數字先講出來，等自己的驗證跑完再修正，
   在對面看起來就是「又被打回去了」。數字沒有變壞，是我講早了。

   **規則**：
   - 任何績效數字，**樣本外／walk-forward 的值放主句**，
     樣本內的值只在括號裡當對照，並標明「樣本內」。
   - **還沒做樣本外就明說「這還沒驗過」**，不得先講一個好看的數字
     再補驗證。想講就先跑。
   - 一併報**樣本外／樣本內的比值**——它是過擬合程度的直接讀數
     （交會線 2026-09-09 實測 **34%**，所以樣本內數字一律先打三折）。
   - 參數是掃出來的就必須說，並且**掃參數的樣本外也要做**
     （同日教訓：C 臂的門檻在全樣本掃出 (10分, 0.5ATR) 樣本外 9/9，
     但誠實地只用前半選會選到 (5分, 1.0ATR)，那組後半只有 5/9 ——
     「樣本外 9/9」是看過答案才挑的）。

   **這條管的是報告，不是研究**。研究照舊該怎麼跑怎麼跑；
   要擋的是「先給一個會被自己收回的數字」這個習慣。

10. **「CI 不跨零」是宣稱發現的門檻，不是下注的門檻**（2026-09-09 使用者訂立）。
    使用者原話：「交易本來就充滿不確定性，為什麼一定要沒有跨 0」。**問得對，
    而我用錯了一整天**——拿研究上防偽陽性的標準，去回答「要不要投錢」。

    **兩個門檻服務不同的問題，不可互換：**

    | | 問題 | 該看什麼 |
    |---|---|---|
    | **研究／宣稱發現** | 這個效應是不是我挑出來的？ | CI 下緣 > 0、逐幣、兩半、真·樣本外 |
    | **決定要不要下注** | 期望值為正嗎？活得下來嗎？ | **P(優勢 > 0)**、期望值、回落分布、破產機率 |

    「CI 跨零」的字面意思只是「**這個樣本不能排除零**」，
    它**不等於**「它是零」。一個 P(優勢>0)=85% 的策略，CI 會跨零，
    但那是一個明確值得考慮的賭注；把它講成「沒有證據」是誤述。

    **具體要求**：任何要拿去做資金決策的結果，除了 CI 之外**必須一併報**
    - `P(真實優勢 > 0)`（同一組日聚類 bootstrap 直接數）
    - 回落分布（重抽路徑的 p50 / p90 / p95）與 `P(回落 > 50%)`
    交會線 2026-09-09 實測：CI [−0.135, +0.549] 跨零，
    但 **P(優勢>0)=84.9%**、2x/3 槽的 P(回落>50%)=3%、P(>70%)=0%。
    只講前者會把一個可下注的東西講成「沒有證據」。

    **這條不放寬研究端的標準**：要把一個效應寫成「發現」、寫進判決節、
    或拿去改規格，CI 那一關照舊。放寬的只有「不得用研究門檻回答資金問題」。

11. **Gate 0：執行可行性排在資訊層之前**（2026-09-11 訂立，
    依據 `docs/common_cause_scan.md` 假說 1）。

    把 37 個已結案的判決橫著讀，最大的一群有**六個成員**，而且每一個都是
    我們自己寫下來的：

    | 節 | 資訊層 | 執行/經濟層 |
    |---|---|---|
    | §4.65 分鐘級 | G1 **PASS**（15/15 月同號） | G2 **FAIL**（138 格全負）|
    | §1.03i 清算位密度 | H1–H4 **全過** | 交易假設**沒過** |
    | §1.02 舊線 | 引擎六項全過 | edge 是**成交假設**造的 |
    | §1.03b SDV | 毛利為正、CI 離零 | 誠實錨點後扣成本**全負** |
    | §0.57b | 天花板 +0.0381 顯著 | 落差 **+0.0505 = 天花板的 132%** |
    | §1.18b 路徑 B | — | 被動基準六個場館對**全負** |

    **這不是「交易很難」。是我們的流程永遠把資訊層排前面、執行可行性排最後，
    所以每條線都花幾個月才撞到它的約束——而那個約束每次都是同一個。**

    而我們**發現過修法一次**：§1.03 的小額實盤，理由白紙黑字寫著
    「把舊線的教訓提前做掉」。**對一條線做了，沒有變成所有線的規則。**

    **所以從今天起，任何新線的預註冊要有 Gate 0，排在資訊層之前：**

    ```
    a. 如果這個訊號完全正確，我在什麼價格成交？
    b. 那個價格在目標場館拿得到嗎？（要量測，不是論證）
    c. 可實現的邊際 vs 該場館的來回成本，比值多少？
    d. 同時算 MDE：這個設計分辨得出那個比值嗎？
    e. 比值 < 1，或 MDE > 效應 -> 停，不要做資訊層的工作
    ```

    **已經有一半答案的例子**：鏈上線的 Gate 0 不必從零開始——
    `resting_limit` 已量到掛在價位是 **−0.0612 R**、`nofill_distance` 已證明
    **價格不會回來**（>10 tick 佔 87.8%、中位 108 tick）。那兩個數字要在
    §1.10 機制關**之前**被引用，不是之後。

    **反例要一起記**（否則這條規則沒有分辨力）：§1.18i 的 λ 條件分解
    13 個狀態格全為正，**沒有死在執行層**。所以規則不是
    「所有東西都死在執行層」，是「**凡是最後死掉的，多半死在那裡**」。

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

**網站文章（writeups）的管線與它的斷點（2026-09-01 補）**：`/writeups` 的
內容**不是**手寫進網站的，它走這條線——

```
Desktop/linkedin_posts/*.docx        ← 授權格式（人寫的）
  → assets/extract_for_site.py 的 ARTICLES 清單（**手動加一筆**）
  → python assets/extract_for_site.py  → assets/site_writeups.json
  → 複製到 ../product-site/content/writeups.json（封面圖進 public/writeups/）
  → npm run build 驗證 → push origin **master**
```

> **⚠ 2026-09-12：上面那條管線已經不存在了。** `assets/extract_for_site.py`
> 與 `assets/site_writeups.json` **兩個檔案都沒有**，`writeups.json` 現在
> 只活在 product-site 一側。ep8 是怎麼進去的沒有留下腳本。
>
> **現行做法（ep9 起）是一篇一支腳本，而且內文只有一個真相源：**
>
> ```
> assets/make_epN_<slug>.py       BODY 字串 = 真相源 -> 產 docx 到 Desktop/linkedin_posts/
> assets/publish_epN_to_site.py   import 同一個 BODY -> 轉 blocks -> 寫 product-site
>   → npm run build 驗證（看**頁數有沒有 +1** 與 .next 裡的 html 真的產出）
>   → push origin master
> ```
>
> 兩支共用同一個 `BODY`，所以改稿只改一個地方，docx 與網站不會漂開 ——
> 這正是舊管線那個「手動加一筆」斷點的結構性修法（mistake.md 2026-09-01：
> 三篇 docx 躺了六週沒上站，而網站看起來完全正常）。
> `publish_*` 那支帶兩道自曝：小標找不到就停、公開面出現金額字樣就停。

**第二條內容流（2026-09-05 起）：陣亡名冊**——`/writeups` 文章下方那一節
`flow_system/assets/research_nogo.json`（真相源，手工從 TODO.md 判決節策展）
→ 複製到 `../product-site/content/research_nogo.json` → build → push master。
**每次 TODO 新增一個 NO-GO／FAIL／無效判決，同一個 session 就加一條**，
不留到「之後一起」。狀態欄固定用那幾個字，公開面規則同上（無美元／張數／內部）。

**中間那個「手動加一筆」是實際發生過的斷點**：2026-09-01 發現 ep5/6/7
三篇 docx 早就寫完、躺了六週沒上站，因為沒人把它們加進清單，而**網站看
起來完全正常**（見 mistake.md 同日）。所以：**寫完一篇 docx 的當下就走完
整條線**，不要留到「之後一起發」——那個「之後」沒有任何東西會提醒。

## 對外網站呈現面（product-site，2026-08-02 盤點）

網站是三條策略**唯一的對外展示層**。資料一律走 agent-mcp 的 `/public/*`
唯讀端點（Railway `agent-mcp-production-46d7`），網站**不直連 MySQL、不碰
任何交易路徑**——這條界線由 `.claude/rules/agent-boundary.md` 管，網站只是
它下游的下游。

| 策略 | 頁面 | 主要元件 | 吃的端點 |
|---|---|---|---|
| **V7** | `/charts/v7`、`/dashboard`、`/signals`、`/track-record` | ChartDetail、V7KpiRow、**V7FilterCard**（地形四維＋扳機進度）、LiveTradesPanel | `/public/chart`、`/live-chart`、`/signal-feed`、`/signal-history`、`/live-status`、`/track-record` |
| **流動性獵取** | `/charts/liquidity`、`/charts/backtest`、`/dashboard` | ChartDetail（獵取覆盤圖）、SweepKpiRow、**ShadowLedgerBoard**（5 變體 + 8 組合 + 時鐘）、ShadowTradesPanel、**BacktestChart**（2026-09-08 起預設分頁是**交會線**，舊線在第二分頁作紀錄） | `/public/liquidity-map`、`/public/sweep-status`、`/public/conj-backtest`（agent 讀 `conj_backtest_pages`，本機 conj_update 班車產出）、`/public/backtest-chart`（舊線） |
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
