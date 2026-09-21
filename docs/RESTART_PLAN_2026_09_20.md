# 系統重啟規劃 —— 只留有用的，Railway 成本最小（2026-09-20）

> **使用者**：「現在要規劃系統重啟但只留有用的東西，然後 railway 花費的成本最小」。
>
> 這是**規劃**。執行要等拍板，而且第一步在 Railway 後台，程式碼這側做不到。
> 前置：`docs/INVENTORY_2026_09_20.md`（清點）、`docs/STRATEGY_PLAN_2026_09_20.md`（策略）。

---

## 0. 「有用」的定義換了，所以清單跟上週不一樣

使用者 2026-09-20 的定位：**「這個專案最有用的就是這些研究，賺不賺錢是其次，
因為我如果能透過這些拿到履歷也是一種方法。」**

所以判準不再是「它可能賺錢嗎」，是這三條：

1. **它服務履歷嗎？**（網站、文章、可展示的活系統）
2. **它服務那兩條還活著的線嗎？**（SDV、V7 —— 其餘六條已判決）
3. **它錄的東西不可回填嗎？**（停了就永久失去）

**只要三條都不是，就不重啟。**

---

## 1. 逐服務判定

判定方法是問兩個問題：**不存在會壞什麼** ＋ **本機能不能做**。

| 服務 | 不存在會壞什麼 | 本機能不能做 | 判定 | 估 $/月 |
|---|---|---|---|---|
| **MySQL** | 全部——網站資料、V7 訊號落庫、SDV 意圖層、agent 唯讀來源 | 裝得了，但**斷電就全斷**，而且雲端服務連不到本機 | **留** | ~7 |
| **agent-mcp** | 網站**全部 20 個 `/public/*` 端點**，所有資料面板變空 | **不能**（要公開 URL） | **留** | ~4 |
| **jarvis** | 產品 UI `/u`、Bitget 下單層、SDV paper 的消費端 | **不能**（要公開 URL ＋ 使用者登入） | **留** | ~4 |
| **indicator** | V7 訊號產生（唯一的產生者） | **能** —— `run_indicator.bat` 已經寫對了，見 §3 | **不上雲** | **0** |
| marketdata | `depth_deltas_1m` ＋ `depth_events_1s` 兩張死表 | — | **砍** | −12 |
| cloud_train | 獵取舊線 shadow 記帳 | **本機 SweepShadow 每小時在跑同一件事**，雲端那份是 `parallel` 影子 | **砍** | −11 |
| hl-record | HL 燃料（鏈上 §1.10） | 能，但會跟 arb 引擎搶本機 IP 限流 | **砍**（歸屬移交 arb） | −4 |
| arb scanner | arb 線 | — | 不在本 repo 範圍 | — |

**合計：約 $15/月**（原本約 $56）。加 Hobby 方案 $5 含 $5 額度 → **帳單約 $15–16**。

> **我看不到你的實際帳單**（沒有 CLI）。上面用 Railway 公布費率估：
> RAM $10/GB/月、vCPU $20/vCPU/月、Volume $0.15/GB/月。
> 真實數字在 Railway → **Usage**。

---

## 2. 為什麼 indicator 不上雲，而 MySQL 要

**indicator 不上雲**：它每小時跑一次推論，本機排程做的是**完全一樣的事**。
而且啟動器早就寫好了（`C:\Users\rfo\Desktop\flowbot\run_indicator.bat`，
2026-08-20 重建），裡面第一行就是：

    set OKX_EXECUTOR_ENABLED=0

那正是「拔掉 OKX」的設定——訊號照寫 `tracked_signals`，兩條下單路徑都關。
排程 `FlowBot_IndicatorUpdate` 每小時觸發，**但它是 Disabled，停在 2026-04-02**。
又一次「修好了但沒接上」。**啟用它 = 省 $9，零工程。**

**MySQL 要留在雲端**：它是唯一的真相源，而本機的排程全部是 Interactive
（要登入才跑）、D 槽連結斷掉時每支程式會**安靜地讀到零列**。把唯一的真相源
放在那種地方，是拿這份 repo 每一條教訓去賭。$7 買掉這個風險是便宜的。

---

## 3. 執行順序

> **2026-09-20 使用者確認：「資料都還在，重新部署就會都恢復了」。**
> 所以那五個 404 是**停機不是刪除**，volume 與 MySQL 資料完整。
> 本檔 §6 第 3 點的疑問就此關閉。

**順序陷阱（這種情況下最容易踩的那個）**：

    後台逐一 redeploy  ->  只動到那一個服務
    一次 git push      ->  這個 repo 的**全部**服務一起醒來（mistake.md 2026-09-15）

所以刪除必須排在 push 之前。**好消息是 push 可以無限期押後**——2026-09-20
這個 session 改的東西（`shadow_engine.bat`、`freshness_board.py`、
`db_export_and_slim.py`、三份 docs）**全部只在本機跑，沒有一樣需要上雲**。

| # | 動作 | 在哪做 | 為什麼排這裡 |
|---|---|---|---|
| **1** | **後台刪掉** `marketdata` / `cloud_train` / `hl-record` | **只有使用者能做** | 省最貴的兩個；而且**這一步做完 push 才安全** |
| **2** | **後台 redeploy `jarvis`** | Railway | **最高優先**：它是使用者操作交易的面板（`/u`），而且**不需要 MySQL**（JSON 檔持久化，見 §2b）。單獨做得起來，$4 |
| 3 | 後台 redeploy **MySQL** | Railway | 研究線與網站的資料層 |
| 4 | `db_export_and_slim.py --report` → `--export` → `--verify` | 本機 | 全部落地 D 槽，之後每一步可逆 |
| 5 | `--drop --i-have-the-export` | 本機 | 588 萬列死表下線。**必須在第 1 步之後**——marketdata 是那兩張表的寫入端，它若先回來會填回去（D3 會擋，但順序對就不用靠守衛） |
| 6 | 啟用本機排程 `FlowBot_IndicatorUpdate` | 本機 | V7 訊號恢復產生，$0 |
| 7 | 後台 redeploy **agent-mcp** | Railway | 網站資料面板回來 |
| 8 | （可選）commit ＋ push | 本機 | 只有在第 1 步完成後才安全 |

### 2b. jarvis 為什麼能排第 2（不必等 MySQL）

2026-09-20 查證：`src/persist.js` / `src/tenants.js` 用 `fs.writeFile` 寫
JSON（`tenants-data/`、`.state.json`），**整個 jarvis 不碰 flow_system 的 MySQL**。
從 agent-mcp 來的訊號（`FLOW_*_URL`）斷了會沿用舊快取降級，不擋開機。

**而本機那個 jarvis 不能代替它**：本機 PID 15412 從 2026-09-17 起跑著
`127.0.0.1:8080/u`（`T.I.F · 交易面板`，207 KB，UI 完整），但——

    BG_MODE = paper
    BITGET_API_KEY / SECRET / PASSPHRASE = 空
    tenants-data/users.json = 2 bytes（沒有用戶）

**本機是 paper 實例，憑證只設在 Railway 的環境變數裡，真實策略設定與帳本
在 Railway 的 volume 上。** 兩份 `.state.json` 不是同一個檔案。

---

## 4. ⚠ 一件跟 Railway 無關、但更該先做的

**網站的子頁在 Railway 停掉之前就已經是 404 了。**

2026-09-20 實測 `flowbot-site.vercel.app`：

| 路徑 | 結果 |
|---|---|
| `/` | **200，45,965 B（正常）** |
| `/writeups` → `/writeups/` | **404** |
| `/system` `/dashboard` `/track-record` `/signals` | 全部 **404** |

而且它的行為跟 repo 對不上——它把 `/writeups` 308 轉到 `/writeups/`，
但 `next.config.js` **沒有 `trailingSlash`**。所以 `flowbot-site.vercel.app`
多半是**舊的、或另一個專案的部署**。

另外 **`product-site.vercel.app` 不是你的**：它吐 `<title>React App</title>`
＋ `/static/css/main.*.chunk.css`，那是 Create React App 的產物，而你的 repo
是 Next.js。同族：mistake.md 2026-07-22（Vercel project 接到錯的 repo，
**而且不會報錯，只在下次不相關的 push 命中時才現形**）。

**為什麼這件事排在 Railway 前面**：

- `/writeups` 的內容是**靜態的**（`content/writeups.json` 在 repo 裡），
  **它不需要 agent-mcp、也不需要 MySQL**
- 也就是說**履歷要用的那一頁，可以在 Railway 完全關著的狀態下先修好**
- 成本 **$0**（Vercel 這層不花 Railway 的錢）
- 而一個子頁全 404 的網站，對履歷是**負分**不是零分

**缺的資訊（只有使用者拿得到）**：Vercel 後台 → `product-site` 專案 →
**Domains** 上的正式網址，以及 **Deployments** 最新一筆是從哪個 repo／commit 建的。
本檔不猜——mistake.md 2026-07-22 的結論逐字是「看到部署問題，
第一步是去看平台自己的部署詳情頁，不要猜身分」。

---

## 4b. 整合 —— 系統裡現在有哪些重複與遺留

> 使用者 2026-09-20：「這次把服務全部停掉就是要下來把系統全部整理整合，
> 然後用最低的成本去運行他，因為我發現之前系統花太多錢在沒用的東西上了。」
>
> 「精簡」是砍掉沒用的；「**整合**」是把做同一件事的兩份合成一份。
> 前面幾節做的是前者，這一節是後者。

| 重複／遺留 | 現況 | 整合後 |
|---|---|---|
| **cloud_train（雲）vs SweepShadow（本機）** | 跑同一件事。雲端那份是 `TRAIN_PHASE=parallel` 的影子，**本機才是 authority** | 只留本機 |
| **indicator（雲）vs `FlowBot_IndicatorUpdate`（本機）** | 雲端在跑、本機那個 **Disabled 停在 2026-04-02**，而啟動器 2026-08-20 已重建好 | 只留本機 |
| **jarvis（雲，真）vs 本機 paper 實例** | 兩份 `.state.json`，本機那份 `BG_MODE=paper`、憑證空 | 雲端是正式的；本機留著當開發用，**但要知道它不是同一份** |
| **兩個 Vercel 部署** | `product-site.vercel.app` = Create React App（**不是你的**）；`flowbot-site.vercel.app` = Next.js 但子頁全 404 | **待查**：Vercel 後台 Domains 到底有幾個、哪個是正式的 |
| **兩條內容線** | LinkedIn 19 篇（`Desktop/linkedin_posts`）＋ 網站 10 篇（`content/writeups.json`），**內容不同** | 履歷需要**單一入口**，要合併 |
| `ExitPathsWatchdog` 的 `lighter_recorder` | 路徑 A（Lighter 零費率影子執行），**前提已被 §1.31／§1.33 推翻**，TODO §1.44 自己寫「重寫或關掉」 | **待使用者決定** |
| `FlowBot_HLRecord` | 每小時打已停的 hl-record，必定失敗 | 歸屬移交 arb 後停用 |
| 監控 58 列 | 4 列 2026-09-20 已 retired；2 列（HL）是真紅燈 | 已處理 |
| DB 45 表 | 3 張死表 588 萬列 ＋ 一批 0 列的空表 | 第 5 步 drop |

---

## 4c. 上雲的唯一兩個理由（這一條是防復發的核心）

**原則**：

> **(a) 它需要一個公開 URL，或 (b) 它不能跟著這台機器一起斷。
> 兩個都不是，就放本機。**

套進去，答案自己就出來了，不用逐案討論：

| 服務 | (a) 公開 URL？ | (b) 不能斷？ | 判定 |
|---|---|---|---|
| MySQL | 否 | **是**（唯一真相源） | 上雲 |
| jarvis | **是**（手機開面板、用戶登入） | 否 | 上雲 |
| agent-mcp | **是**（網站打它） | 否 | 上雲 |
| indicator | 否 | 否（每小時一次，漏一次不致命） | **本機** |
| marketdata | 否 | 否 | **砍** |
| cloud_train | 否 | 否 | **砍** |
| hl-record | 否 | 否（本機能錄） | **砍** |

---

## 4d. 為什麼會變成這樣（不寫下來就會復發）

**每一個服務上雲的當下都有好理由**：

- `cloud_train` —— 2026-08-21「把 shadow 記帳搬離筆電」（資料工程弱點 #1）
- `hl-record` —— 2026-09-15「錄製器跟 HMM 引擎的 HL 腿搶本機 IP 限流」
- `marketdata` —— 錄撤單流的簿口資料

**問題不是當初決定錯，是沒有人回頭問「它還需要在雲端嗎」。**
而更貴的那一半是：**一條線被判死之後，沒有人回頭關它的水龍頭。**
撤單流 2026-08-10 判 FAIL，而 `marketdata` 又錄了 41 天。

**兩條規則（寫下來才不會再來一次）**：

1. **每個雲端服務要有一行「它為什麼必須在雲端」**，寫在會被讀到的地方
   （服務的 Dockerfile 檔頭已經是這個慣例，`Dockerfile.hlrecord` 就寫了）。
   **那個理由消失的那一天，它就該下雲。**
2. **判死一條線的同一個 session，要列出它在消耗什麼**
   （雲端服務、排程、資料表、監控列）並當場決定關不關——
   不留到「之後一起」（mistake.md 2026-09-01 那個形狀：
   那個「之後」沒有任何東西會提醒）。

---

## 4e. 2026-09-21 已執行的本機整合清理

使用者：「現在把系統儘量整合在一起然後把不必要的東西都清掉」。**本機這側能做的全做了；
Railway／DB 那側只有使用者能動。**

| 做了什麼 | 怎麼驗 |
|---|---|
| `lighter_recorder`（路徑 A）從 `exit_paths_watchdog.ps1` 的 `$Jobs` 拿掉、行程停掉 | BOM 保留、CRLF 95→99、PS parser 0 錯誤、`$Jobs` 只剩 `liq`；行程數 0。**⚠ 照 mistake.md 2026-09-14，要跨過一個看門狗週期（>5 分）再確認它沒被拉回來** |
| 看板 `lighter recorder flag (路徑A)` → retired 09-21 | 門檻永不紅、列保留 |
| `FlowBot_HLRecord` 排程 Disabled | `Get-ScheduledTask` State=Disabled |
| 看板兩列 HL → `(moved to arb 09-21)` | 歸屬移交 arb；本機資料到 09-19 17:00 UTC |
| 每小時班車 23→16 步、四列 retired、匯出腳本（09-20，見 INVENTORY §5b） | 已驗 |

**留著沒動、而且是刻意的**：`liq_recorder`（路徑 C 強平時鐘 131/300，活的）、
本機 paper jarvis（開發用）、`FlowBot_IndicatorUpdate` 仍 Disabled——
**等 MySQL 回來再啟用**，否則每小時白打 Coinglass 配額然後寫入失敗。

**要在 arb session 做的一件事（本 repo 不碰 arb）**：決定 HL 燃料錄製器要不要在 arb 那邊復活。
它跟 arb 引擎搶 IP 的問題還在，資料不可回填，缺口從 09-19 17:00 UTC 起算。

**等使用者做的**（§3 的 1、2、3、7）：後台刪三個 → redeploy jarvis → redeploy MySQL →
叫我跑匯出與 drop → redeploy agent-mcp。

---

## 5. 重啟之後會留下的東西（對照「有用」的三條判準）

| 留下的 | 服務哪一條判準 |
|---|---|
| MySQL（瘦身後） | 三條全部 |
| agent-mcp | 履歷（網站資料面板不是空的） |
| jarvis | 履歷（活著的產品 UI）＋ SDV paper 消費端 |
| 本機 indicator 排程 | V7（兩條活線之一） |
| 本機 SweepShadow（16 步） | SDV 時鐘、§0.91 基差、GEX 錄製、生存條件層 |
| 本機 conj_watch | **SDV 意圖層** —— 兩條活線裡數字最好的那條 |
| D 槽 40 GB ＋ 不可回填的錄製 | 判準三 |

| 不留的 | 為什麼 |
|---|---|
| marketdata | 撤單流 2026-08-10 判 FAIL |
| cloud_train | 獵取舊線 2026-09-07 結案，而且本機重複 |
| hl-record | 歸屬移交 arb；本機已有資料到 09-19 17:00 UTC |
| 雲端 indicator | 本機做同樣的事，省 $9 |

---

## 6. 還沒查證的（動手前要對）

1. **Railway 後台的實際服務清單與帳單**。本檔的服務判斷有四個是從 Dockerfile
   推的（`marketdata` / `cloud_train` / `BTC_perp_data` / `watch_chart`），
   它們沒有公開網址，我從外面看不到。
2. **Vercel 的正式網址與最新部署來源**（見 §4）。
3. ~~那五個 404 到底是停機還是刪除~~ —— **2026-09-20 使用者確認：停機，資料都在。**
