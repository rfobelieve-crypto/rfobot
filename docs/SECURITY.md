# 祕密清冊與輪替手冊

> 2026-09-05 建立。起因：操作者的 Telegram 帳號被盜且無法登入。查曝險面時
> 才發現 `ADMIN_HEAL_TOKEN` 一直被嵌在送去聊天室的按鈕網址裡——**沒有人知道
> 它在那裡，因為沒有清冊。** 這份文件的存在理由就是那句話。

## 事故紀錄

**2026-09-05 Telegram 帳號遭盜用**

| 項目 | 內容 |
|---|---|
| 影響 | 聊天記錄全部外洩：訊號、進出場、部位張數、名目美元、權益變化 |
| 連帶 | `ADMIN_HEAL_TOKEN` 因為被嵌在按鈕網址裡而一併外洩 |
| **沒有**發生 | 無資金移動路徑（executor 自 8 月起停在 CAP-2 HALT，近三天零新倉）；Telegram 訊息從不帶交易所金鑰；`okx_accounts` 表在此資料庫不存在 |
| 處置 | 送出端 13 個點全封、兩個 webhook 回 403（`shared/tg_kill.py`，fail-closed）；`load_config` 改成缺 token 也能開機 |
| 未完成 | 帳號救回 → 四支 bot token 輪替 → `ADMIN_HEAL_TOKEN` 輪替 → webhook 改用 secret_token |

## 祕密清冊

**能動錢的**（最高等級，任何異動都要當成事故處理）

| 祕密 | 放在哪 | 能做什麼 | 應有的限制 |
|---|---|---|---|
| `OKX_API_KEY` / `SECRET` / `PASSPHRASE` | Railway 環境變數 | 下單、平倉、查餘額 | **禁提幣權限**；綁 IP 白名單；executor 專屬（帳戶隔離已於 2026-07-28 由使用者放棄，見 CLAUDE.md） |
| 未來：arb 線的交易所金鑰 | `../arb`，**不進本 repo** | 同上 | 同上，且錢包與本 repo 帳戶體系完全獨立 |

**能改系統狀態的**

| 祕密 | 放在哪 | 能做什麼 |
|---|---|---|
| `ADMIN_HEAL_TOKEN` | Railway（**多個服務共用**） | `/okx-admin/heal`（重置 executor DB 狀態，POST-only 且 OKX 有倉時拒絕）、`/admin/db-health-all`、`/admin/flow-bars-export`、`/admin/backfill-gap`、全部 `/research/*`（策略輸出） |
| `INDICATOR_ADMIN_TOKEN` | Railway | 指標服務的 admin 面 |
| `AGENT_MCP_TOKEN` | Railway | MCP agent 的存取（唯讀下游，見 agent-boundary.md） |
| `CONJ_FILL_TOKEN` | Railway（agent-mcp）＋產品端 jarvis | **2026-09-08 新增。** `POST /public/conj-fill`——產品端回報交會事件實盤成交（TODO §1.03）。寫的是 `agent_conj_fills`（agent 自己的命名空間）。**fail-closed**：未設定 → 端點 503 拒收，不是開放。標頭 `X-Conj-Token`，`hmac.compare_digest` 比對。它守的是實盤對帳資料的完整性——一筆假成交就能弄髒整個測試的結論 |

**能讀資料的**

| 祕密 | 放在哪 |
|---|---|
| MySQL 連線字串 | Railway（內部主機名）／本機 `.env` |
| `COINGLASS_API_KEY` | Railway／`.env` |
| ~~`NANSEN_API_KEY`~~ | **2026-09-10 同日移除**（使用者決定不走這條線）：已從 `.env` 與 MCP 設定刪除。**但逐字稿裡的明文仍在，所以仍應輪替** |
| 四支 Telegram bot token（`TELEGRAM_BOT_TOKEN`、`CANCEL_TG_BOT_TOKEN`、`INDICATOR_BOT_TOKEN`、`AGENT_BOT_TOKEN`） | Railway。**2026-09-05 起一律視為已外洩** |

**版控狀態**：`.env` 與 `config.json` **從未進過 git**（`git log --all -- .env` 為空），
現行程式碼無寫死金鑰。

## 已知並接受的風險

- **Telegram Bot API 的端點本身含 token**（`https://api.telegram.org/bot<TOKEN>/…`）。
  那是廠商的 API 設計，改不掉；它只出現在伺服器對外的請求 URL，不是交給人點的
  連結。`tests/test_no_secret_in_urls.py` 明確豁免這個 host。
- **交易帳戶與手動操作共用**（2026-07-28 使用者知情後選擇保留返傭）。後果寫在
  CLAUDE.md，不再重提。

## 規則（違反就是事故，不是風格問題）

1. **長期祕密不准進網址。** 要給人點的連結用 `shared/signed_link.py`——
   HMAC 綁路徑、預設 24 小時到期。舊聊天記錄裡的連結因此會自己失效。
   由 `tests/test_no_secret_in_urls.py` 機器檢查（含反向證明）。
2. **祕密只放環境變數，不放檔案、不放程式碼、不放訊息。**
3. **新增任何祕密，同一個 session 就要加進本檔的清冊**——這次的教訓就是
   「沒人知道它在那裡」。
4. **交易所金鑰一律無提幣權限。** 這條沒有例外，也不接受「先開著方便」。
5. **fail-closed**：守衛拿不到設定時要拒絕，不是放行（`_admin_guard` 在
   `ADMIN_HEAL_TOKEN` 未設時回 503，不是略過檢查）。

## 輪替手冊

**Telegram（四支 bot）** —— 前提：帳號已取回、已終止其他工作階段、已開兩步驟驗證

1. BotFather 對四支 bot 逐一 `/revoke`
2. Railway 更新 `TELEGRAM_BOT_TOKEN`、`CANCEL_TG_BOT_TOKEN`、
   `INDICATOR_BOT_TOKEN`、`AGENT_BOT_TOKEN`
3. 逐一重設 webhook，**並帶 `secret_token`**（Telegram 支援；之後每個請求會帶
   `X-Telegram-Bot-Api-Secret-Token` 標頭，這樣即使 token 外洩，直接 POST 偽造
   也會被擋。現在的設計把 token 同時當認證與路徑，一個外洩就兩個都破）
4. 解除斷流：`TELEGRAM_REENABLE=I_ROTATED_ALL_TOKENS`
5. 確認 BotFather 的 bot 清單沒有多出沒建過的 bot，且既有 bot 的描述／指令未被改

**`ADMIN_HEAL_TOKEN`** —— **多個 Railway 服務共用，必須同時改**

1. 產生新值（`python -c "import secrets;print(secrets.token_urlsafe(32))"`）
2. 在**每一個**會用到的服務上更新（漏一個會讓服務之間的呼叫 403）
3. 更新後打一次 `/admin/db-health-all` 確認新值可用、舊值失效

**交易所金鑰**：先確認無提幣權限，再刪舊建新；executor 重啟後看
`v7_okx_balance_snapshots` 的新鮮度確認連得上（**判斷活著看資料新鮮度，
不看平台的健康燈**，mistake.md 2026-07-28）。

## 待辦（依風險排序，不是依難度）

- [ ] 帳號救回後執行上面兩份輪替手冊
- [ ] webhook 改用 `secret_token`（現在 token 兼認證與路徑）
- [ ] 查證 OKX 金鑰確實無提幣權限、是否可加 IP 白名單
- [ ] `/admin/flow-bars-export` 加速率限制（目前無限制的資料匯出）

### 2026-09-10 `NANSEN_API_KEY`（新增，且**應視為已外洩**）

使用者在對話中以明文提供。Claude Code 的對話逐字稿是本機的 `.jsonl`
檔案，路徑可預測，**任何讀得到那個目錄的東西都讀得到這把金鑰**。
所以它跟四支 Telegram bot token 同級：**一律視為已外洩，請輪替**。

**2026-09-10 當日結案**：使用者決定不走這條線，金鑰已從 `.env` 與 MCP
設定移除。**但移除不等於失效**——逐字稿裡的明文還在，所以輪替（或直接
在 Nansen 後台撤銷）仍然要做。撤銷比輪替更乾淨，因為沒有要繼續用。

輪替之後若還要用，放進 `.env`，不要再貼進對話。
需要它的程式一律走 `os.environ` 讀，不寫進任何原始碼或 commit。

能做什麼：Nansen 平台的唯讀資料查詢（鏈上聰明錢標籤與資金流）。
**沒有**下單、提幣、改設定的能力，所以外洩的後果是額度被盜用與帳戶
資料被查詢，不是資金損失。即便如此仍應輪替——額度是有價的。

限制：只在研究端使用；**不進 Railway、不進產品端 jarvis**（那兩處沒有
任何需要它的路徑，多一個地方就多一個外洩面）。
