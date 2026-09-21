# 全盤清點（2026-09-20）—— Railway 服務 ＋ 研究線

> **觸發**：使用者把 Railway 服務全部移除，理由是「沒有用的東西太多，
> 每個月花費一堆錢然後什麼東西都沒有」「太多策略都沒有用的，盤點出來就可以砍掉」。
>
> 本檔是清點結果與決定表。**數字全是快照**，逐項的活數字在 TODO 對應節。
> 研究線那一節是 TODO §1.44（2026-09-15）的**刪減版**——那份問「還開著什麼」，
> 這份問「**哪些該關掉**」。

---

## 0. 一句話

錢不是花在沒用的地方，是**花在已經判過死刑的東西上，而沒有人回去關水龍頭**。
資料庫一半以上的列（588 萬）在餵兩條已判決死亡的線，
而寫它們的兩個 Railway 服務 24/7 開著。

---

## 0b. 🚫 在後台刪掉服務之前，這個 repo 不可以 push

使用者 2026-09-20：「都先不要重新部署不然服務器都會回來」。

`architecture.md` 寫著 push 到 `main` 觸發自動部署，mistake.md 2026-09-15 記著
它會重新部署**這個 repo 的全部服務**。**本機 commit 不觸發任何東西，push 才會。**

連帶要更正本檔第 1 節的一個過度解讀：**404 不等於「服務被刪了」——Railway 上
停機的服務，網域也是回 404。** 那五個很可能跟 MySQL 一樣只是停著、還在清單裡。
真正的「砍服務」只能在 Railway 後台做，程式碼這側做不到。

---

## 1. 現況：六個服務，五個不在

2026-09-20 實測（curl / TCP）。**注意上面 §0b：404 = 停機或刪除，分不出來**：

| 網址 | 是什麼 | 狀態 | 停了誰會痛 |
|---|---|---|---|
| `grid-production-87bd` | **jarvis 產品端**（真用戶、Google 登入、Bitget 上的 V7Bot/網格/DCA） | **404 停了** | 註冊用戶。**已查證目前無人持倉**，所以沒有真錢風險 |
| `enchanting-emotion-4b4d` | indicator（V7 推論＋圖表＋Telegram＋已死的 OKX executor） | 404 停了 | **V7 訊號整條線**——它是唯一的訊號產生者 |
| `agent-mcp-production-46d7` | 網站全部 `/public/*` 的唯一來源 | 404 停了 | product-site（Vercel）所有數字變空；MCP 連不上 |
| `hl-record-production` | HL 燃料錄製（09-15 才搬上雲） | 404 停了 | 資料**已拉回本機到 09-19 17:00 UTC**，只損失最後幾小時 |
| MySQL `caboose.proxy.rlwy.net:18766` | 資料庫 | **TCP 通、握手即斷**（停機非刪除，使用者確認仍在清單） | 全部 |
| `scanner-production-efc9` | arb 掃描器 | **401 —— 還活著**（唯一剩下的） | arb/HMM 那條線（不在本檔範圍） |

另有四個 Dockerfile 對應的服務無法從外部確認（沒有公開網址）：
`Dockerfile.marketdata`（depth collectors）、`Dockerfile.research`（cloud_train）、
`Dockerfile`（BTC_perp_data 主 Telegram bot）、`Dockerfile.chart`（watch_chart）。
**動手前要在 Railway 後台對一次**，本檔對它們的判斷是從程式碼推的。

---

## 2. 資料庫：一半以上的列在餵死掉的線

列數是 `docs/DB_REGISTRY.md` 的 2026-08-21 快照（MySQL 現在連不上，無法更新）。

### 該砍的（合計約 588 萬列）

| 表 | 列數 | writer | 餵哪條線 | 那條線的判決 |
|---|---|---|---|---|
| `depth_events_1s` | 3,058,876（1.2 GB） | marketdata 服務 | 撤單流 F7 秒級事件流（§0.46） | **計分器從沒寫過**；檢查點 09-13 已過；TODO §1.44 註明「無保留政策」 |
| `v7_okx_balance_snapshots` | 1,676,739 | indicator 的 OKX WS，每 5 秒一筆 | OKX executor | **2026-08-21 決定維持停機不再重啟**；帳戶 08-18 起 $0；freshness 那列已標 `retired 09-06` |
| `depth_deltas_1m` | 1,137,531 | marketdata 服務 | 撤單流方向性 | **2026-08-10 預註冊判決 FAIL 定案**（三個檢定全滅） |

三張表加起來 **5,873,146 列**。DB_REGISTRY 全 45 表的列數合計約 1,050 萬，
所以這三張是**全庫的 56%**。

> **唯一的保留理由**：`depth_deltas_1m` 有一個 10-09 的 90 天 re-run 檢查點
> （PREREG A.3，只能跑一次）。但 TODO §1.44 自己寫著
> 「⚠ `research/subhourly/` 沒有 re-run harness」——**harness 不存在，
> 所以那個檢查點目前不可能被執行。** 要留就要先寫 harness，
> 不寫就是再錄 20 天然後一樣沒人跑。

### 可回填的（刪了不心疼）

`funding_rates` 875k、`ohlcv_1m` 654k、`oi_snapshots` 511k、`liquidation_1m` 61k
——Coinglass / Binance 打得回來。

### 不可回填、必須先匯出的

| 表 | 列數 | 為什麼不可回填 |
|---|---|---|
| `flow_bars_1m` | 1,035,585 | 逐筆成交聚合，WS 錄的 |
| `orderbook_snapshots_1m` | 736,234 | L20 簿口，無歷史端點 |
| `tracked_signals` | 2,247 | **V7 全部 live 訊號史 = Gate A/B 的證據基礎** |
| `indicator_history` | 3,514 | **V7 解碼 buffer 重建的唯一來源**（2026-08-11 那個修法靠它） |
| `v7_okx_positions` / `kill_log` / `reconciliation_log` | 21 / 476 / 2,522 | live 交易帳本 |
| `cancel_playbook_events` | 3,288 | 撤單事件（線死了但資料獨一無二） |
| `raid_outcomes` / `raid_signals_live` / `raid_pending_levels` | 1,325 / 46 / 86 | 獵取舊線的前瞻記錄 |

**本機沒有任何 DB dump**（已搜過，只有 `migrations/*.sql` 的 schema）。

---

## 3. Railway 服務：回來還是不回來

| 服務 | 建議 | 理由 |
|---|---|---|
| **MySQL** | **先瘦身再回來** | 資料還在。但直接開回去等於把 56% 的垃圾一起養回來。順序：開機 → 匯出不可回填的表 → drop 三張死表 → 再決定長期要不要留在 Railway |
| **indicator** | **回來，但先拔掉 OKX** | 它是 V7 唯一的訊號產生者。但 `get_executor()` 那條路徑餵的是已停的 executor，而且每 5 秒寫一列快照。關掉 `OKX_EXECUTOR_ENABLED` 就少一張 168 萬列的表 |
| **agent-mcp** | 看網站要不要 | 網站的唯一資料源。網站要活就得回來；網站先放著就不用 |
| **jarvis** | 看產品要不要 | 沒有人持倉，所以沒有時間壓力。但有註冊用戶——不回來就要告知 |
| **marketdata** | **砍** | 它唯一在做的事是錄那兩張撤單流的表，而那條線 08-10 判 FAIL |
| **cloud_train** | **砍** | 餵已結案的獵取舊線（2026-09-07），而且**本機 SweepShadow 每小時在跑同一件事**——雲端那份是 `TRAIN_PHASE=parallel` 的影子，本機才是 authority。重複 |
| **hl-record** | **砍**（或搬去 arb） | 資料已拉回本機到 09-19。而 HMM 整條線 09-15 已搬到 `../arb`——這條線的歸屬不在本 repo |
| **BTC_perp_data / watch_chart** | **待確認** | 需要在 Railway 後台看它們是不是還是獨立服務。`watch_chart.py` 最後修改 2026-03-29，半年沒動 |
| **arb scanner** | 不在本檔範圍 | 還活著，歸 arb |

---

## 4. 研究線：留還是砍

### A. 已判決死亡，但還在消耗資源 —— 這些是要砍的

| 線 | 判決 | 現在還在消耗什麼 |
|---|---|---|
| **撤單流（策略 #3）** | 方向性三檢定全滅，2026-08-10 定案 FAIL | marketdata 服務 24/7、MySQL 約 420 萬列、10-09 的 re-run 檢查點（**harness 不存在**）、F7 秒級（**計分器從沒寫過**） |
| **OKX executor** | 2026-08-21 決定遷 Bitget、不再重啟 | indicator 裡的 OKX WS、168 萬列快照表、對帳與 kill 邏輯、`v7_okx_*` 七張表 |
| **流動性獵取舊線（策略 #2）** | 2026-09-07 結案：交易設計不可執行（57.9% 的成交假設拿不到） | SweepShadow 每小時排程、Gate F 時鐘跑到 10-05、cloud_train 服務、`raid_*` 四張表＋四支 publisher、產品端 B/C/D cohort |
| **路徑 A**（Lighter 零費率影子執行） | 前提「Lighter 零費率」已被 §1.31/§1.33 推翻 | lighter 錄製器、`FlowBot_ExitPathsWatchdog`、~10-03 的假期限。TODO §1.44 自己寫「重寫或關掉」 |
| **縮帆 §0.52/§0.53** | 判決 09-16/09-19 到期 | 重放的是獵取舊線 shadow log，而那批成交價 §1.02 已判拿不到 —— **判決 PASS 也找不到消費者** |
| 地形層扳機 | 2026-09-04 結案（無效判決，SE 11.6pp > 門檻 8pp） | 已停 |
| MRP §1.38 橫斷面 / §1.39 | NO-GO | 已停 |
| §1.27 / §1.28 | 「測不動」本身就是判決 | 已停 |
| §1.11 構造法三臂、§0.60 Q1、§0.80、§0.63/0.64、§0.88e~h、網格 §1.14b/c/d、§1.18f/h、V7 多幣化、PREREG_exit_cancelflow、§0.49~0.49d | 全部已結案 | 已停 |

**砍掉 A 組的實質效果**：兩個 Railway 服務、一個每小時排程、588 萬列資料庫、
四支 publisher、一個看門狗，以及 TODO 裡約 20 節的維護負擔。

### B. 活著且有消費者 —— 這些要留

| 線 | 狀態 | 下一個決策點 |
|---|---|---|
| **V7** | 唯一碰真錢的（經 jarvis/Bitget）。訊號層 90 天勝率 53.7%、交易層 +7.1 bps/筆 | **10-07 例行重訓到期**（60 天上限，不可放寬） |
| **SDV** | 意圖層 ON、paper 鎖死。樣本外 +0.1833 ATR、P(優勢>0) 85% | 時鐘 6/300；要真錢必須另開 override |
| **MFT** | §1.34 排第一名，但理由是場館費率不是新 alpha | **09-18 四項已過期兩天**（今天 09-20），而且 MySQL 停了跑不了 |
| 鏈上 §1.10 | 只錄不做，判準未寫 | Gate 0 已有一半答案（`resting_limit` −0.0612 R） |
| 生存條件層 | ADX 等三個儀表已接進週報 | 持續 |
| lead-lag §1.37 | 計分器在，`rx_ms` 從 09-13 起 | **09-20 就是今天** |

---

## 5. 不可逆清單 —— 動手前必讀

1. **MySQL 不可以在匯出之前刪。** 第 2 節「不可回填」那張表裡的每一列，
   刪掉就永遠沒有了。Railway 刪除是永久的，volume 一起沒。
2. **`cancel_playbook_events` 要留。** 線死了，但那 3,288 列是獨一無二的錄製，
   而且它是「撤單流唯一活著的那個波動結論」的原始資料。
3. **`tracked_signals` 與 `indicator_history` 要留。** 前者是 Gate A/B 的全部證據，
   後者是 V7 解碼 buffer 的唯一重建來源——沒有它，重訓後暖機期會回到
   2026-08-11 那個「DOWN 側算術鎖死」的狀態。
4. **hl-record 的 volume 可以放掉**，本機已有 478 MB / 215 個小時檔到 09-19 17:00 UTC。
5. **砍一條線時，同一個動作要處理它的監控**（mistake.md 2026-09-14：
   停掉被監控的東西而不停監控 = 真紅燈被假紅燈蓋掉）。
   freshness_board 註冊表、`exit_paths_watchdog` 的 `$Jobs`、
   Windows 排程三處都要一起動。

---

## 5b. 已完成的瘦身（2026-09-20，全部在本機，沒有碰 Railway）

使用者：「先進行瘦身後再回來」。**順序上必須先關水龍頭再排水**——先 drop 表
但寫入端還開著的話，它們會立刻填回來。

### 已關掉的水龍頭：每小時班車 23 步 → 16 步

`research/sweep_failure/shadow_engine.bat`，停用 7 個餵死線的步驟
（備份在 `shadow_engine.bat.bak_20260920`，每一行都留 ASCII 的還原說明）：

| 停用 | 寫哪張表 | 死因 |
|---|---|---|
| `raid_signals_publish` | `raid_signals_live` | 獵取舊線 2026-09-07 結案 |
| `raid_pending_publish` | `raid_pending_levels` | 同上 |
| `raid_outcomes_publish` | `raid_outcomes` | 同上 |
| `pf_mirror` | `pf_positions` | 鏡像 `v7_okx_positions`，凍在 08-11 |
| `pf_dry_intents` | `pf_intents` | 吃變體 B 成交，B 2026-09-02 作廢 |
| `v7_veto_publish` | `v7_veto_clock` | 地形扳機 2026-09-04 結案 |
| `train_parity_check` | — | 對照 cloud_train，而 cloud_train 要退役 |

**`shadow_engine.py` 改判為留下。** 它寫的是本機 CSV 不是 MySQL（砍它省不到錢），
但它的第一步是**刷新 1H kline 快取**，而 `weather_station_publish`（生存條件層，
活線）直接讀那個快取。砍了會讓一條活線的資料悄悄過期。

驗收照 mistake.md 2026-09-13：複製一份、把有副作用的指令換成**不帶原文**的
`echo RAN <行號>`（帶原文會讓 `>>` 把證據吞掉），跑同一個解析器數行數 ——
**17 個標記 = 16 個 python 步驟 + `cd` 那行，沒有跳行**。
CRLF 90 → 104、裸 LF 為 0（mistake.md 2026-08-19 / 2026-09-13 兩條）。

### 已同步的監控（mistake.md 2026-09-14）

停掉被監控的東西而不停監控 = 真紅燈被假紅燈蓋掉。照 `okx balance snapshots
(retired 09-06)` 的既有先例——**門檻拉到永不紅、列保留、理由寫原地**：

`raid signals row` / `v7 veto clock row` / `raid outcomes row` / `cloud train parity`
四列標 `(retired 09-20)`。**不刪除**，因為刪掉會讓「有人把它重新打開了」變成看不見。

**`hl onchain recorder` 與 `hl onchain 拉取` 兩列刻意不 retire** —— 它們是**真紅燈**：
HL 燃料餵的鏈上 §1.10 是活線，資料不可回填，每停一小時就永久少一小時。
紅著才看得到那個時鐘在走。

### 已寫好、等 MySQL 回來就能跑

`research/ops/db_export_and_slim.py`。**設計上最重要的決定：全部匯出，再 drop。**
D 槽有 662 GB 可用，整個庫撐死幾 GB，所以沒有理由只匯出「不可回填」那幾張——
全部匯出之後，drop 就完全可逆，不必做任何不可逆的判斷。

    --report    每張表的列數與實際位元組，不動任何東西
    --export    全部匯出到 D:/flowbot_data/db_export/<日期>/，可續跑
    --verify    parquet 列數 vs DB 列數，寫 manifest
    --drop --i-have-the-export

`--drop` 的四道自曝檢查裡，**D3 是關鍵的那一道**：drop 前重查一次 DB 列數，
跟匯出時不符就中止——**列數還在長代表寫入端沒關乾淨**，那正是
「先 drop 會被填回來」那個錯的偵測器。D1/D2/D4 不需要連線就先跑完
（「還沒準備好」不該要先連得上資料庫才知道）。

已實跑驗過兩條拒絕路徑：沒帶旗標 → exit 2；有旗標但沒 manifest → exit 1 乾淨中止。
**DB 相關的路徑未測**（MySQL 停著），第一次跑 `--export` 要盯著看。

> 過程中踩到一個值得記的：第一版的中止訊息用了 `✗` 與 `⚠`，而 cp950 主控台
> 編不出來 —— **守衛正確攔下了，然後在印訊息時炸掉**。這是 mistake.md 2026-09-11
> 那條「對著寫不出來的 log 行開火的守衛等於沒開火」的同族。已全檔掃過，
> 現在整支 `.py` 都能用 cp950 編碼。

### 待決（我不能代你決定）

1. **HL 燃料錄製器要搬回本機、搬去 arb、還是不錄了？** 09-15 搬上雲的理由是
   它跟 HMM 引擎的 HL 腿搶本機 IP 限流，而 HMM 現在在 `../arb` 但引擎還是跑在
   這台機器上——所以那個衝突可能還在。**這有時鐘**：不可回填。
2. **網站與產品端（agent-mcp / jarvis）要不要活。** 產品決定不是技術決定。
3. **獵取舊線的產品端 B/C/D cohort 訊號供給**已隨上面那三支 publisher 停掉。
   jarvis 本來就停著所以目前無消費者，但這等於單方面停供——要恢復的話
   把 bat 裡三個 `[DISABLED]` 區塊還原即可。

---

## 6. 建議的順序

| # | 動作 | 狀態 | 為什麼排這裡 |
|---|---|---|---|
| 0 | **關水龍頭**：班車 7 步停用、監控四列 retired | **✅ 2026-09-20 完成**（§5b） | 先 drop 但寫入端還開著，表會被填回來 |
| 1 | 匯出腳本寫好並驗過拒絕路徑 | **✅ 2026-09-20 完成** | `research/ops/db_export_and_slim.py` |
| 2 | **在 Railway 後台刪掉不要的服務**（marketdata / cloud_train / hl-record） | ⬜ **只有使用者能做** | 刪掉之前 repo 不可以 push，否則全部回來 |
| 3 | MySQL 開回來，跑 `--report` | ⬜ | 先看真實大小，DB_REGISTRY 的數字是 08-21 快照 |
| 4 | `--export` → `--verify` | ⬜ | 全部落地到 D 槽之後，後面每一步都可逆 |
| 5 | `--drop --i-have-the-export`，再量一次大小 | ⬜ | 這時才知道 MySQL 該不該繼續留在 Railway |
| 6 | 決定網站與產品端要不要活 | ⬜ | 產品決定不是技術決定 |
| 7 | indicator 拔掉 OKX 後回來 | ⬜ | V7 是唯一碰真錢的線，10-07 重訓在等 |
| 8 | 決定 HL 燃料錄製器的家 | ⬜ | **有時鐘**：不可回填，每停一小時永久少一小時 |

**還沒做的查證**：Railway 後台的實際服務清單與帳單明細。
本檔的服務判斷有四個是從 Dockerfile 推的，動手前要對一次。

---

## 附：對其他文件的影響

- `CLAUDE.md` §現況速覽的 Railway 相關敘述全部過時（executor、agent-mcp、hl-record）
- `docs/DB_REGISTRY.md` 的列數是 08-21 快照，MySQL 回來後要重生成
- `.claude/rules/agent-boundary.md` 列的 15 張 agent 可讀表，若 agent-mcp 不回來則整份暫時無消費者
- TODO §1.44 的「有日期的」那張表，09-18 / 09-19 / 09-20 三個期限都已到期或過期
- **週報 Telegram 從 09-05 起被擋**（`portfolio_clocks.log` 的 `TG PUSH FAILED`）
  —— 使用者已經 15 天沒收到任何週報，這與 Railway 無關，是獨立的一個洞
