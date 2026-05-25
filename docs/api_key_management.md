# OKX API Key Management — Stage 2/3/4

> Version: 0.1 (initial draft, 2026-05-25)
> Owner: rfo
> Companion docs: `docs/stage2_kill_criteria.md` (§2C/2E),
>                 `docs/okx_integration_design.md` (§8.2)

## 0. 文件性質

這份是 **operational SOP** — 不是設計文件,是「key 的整個 lifecycle 每一步該做什麼」的執行步驟。漏一步可能丟錢 / 帳號被盜 / 出事不知道。

---

## 1. 根本原則

### R1. 最小權限(Least Privilege)

OKX API key 權限**只能勾**:
- ✅ Read(讀)
- ✅ Trade(下單)

**絕不勾**:
- ❌ Withdraw(提幣)— 即使 IP whitelist 被穿透,壞人也帶不走錢
- ❌ Transfer(子帳號轉帳)— 同理

API key 的權限是「假設它已外洩」設計的。withdraw 權限外洩 = 錢沒了;trade 權限外洩 = 壞人最多幫你下幾筆爛單(策略損失,但本金還在交易所裡)。

### R2. 分 key,不共用

| Stage | Key 用途 | 來源 | 環境 |
|---|---|---|---|
| Stage 2 testnet | demo 模擬交易 | OKX 模擬交易 API page | Railway prod env |
| Stage 3 live tiny | 真實 $100 | OKX live API page | Railway prod env(獨立組 env var) |
| Stage 4 scaling | 真實放大 | 同上(可換新 key 配新 size)| 同上 |
| Local dev | 測試用 | 獨立 demo key | 本地 `.env`(gitignored) |

**任何兩個 stage 不共用同一個 key**。理由:
- 權限要分(Stage 2 demo 永遠 simulated;Stage 3+ live 才能下真錢)
- 出事 revoke 不誤殺(testnet key 出事不該影響 live)
- 審計可追溯(哪個 key 下的單一目了然)

### R3. 主動輪替,不等過期

```
Stage 2 demo:      每 14 天主動換
Stage 3 live:      每 30 天主動換
Stage 4 scale:     每 14 天主動換(金額大,風險更高)
Local dev:         每 30 天主動換
```

不等 OKX 自然過期。主動換的好處:
- 練習 rotation 流程,真出事時不卡手
- 提早發現「換 key 之後系統壞了」(代表 key 寫死在某處,違反原則)

---

## 2. Key Onboarding(新 key 部署)

### 2.1 Demo key(Stage 2)

```
1. OKX → 用戶中心 → 模擬交易 API
2. 建立 key,勾 Read + Trade,**不勾 Withdraw**
3. 設定 IP whitelist:加入 Railway egress IP
   (Railway 找:Project → Service → Settings → Networking → Static Egress IP)
4. 記下 API key / secret / passphrase 到密碼管理器(1Password / Bitwarden)
5. 永遠不要貼進:
   - git repo(任何分支)
   - .env(會被 Railway 自動讀,但本地若 commit 一次永久外洩)
   - Slack / Telegram / 任何 chat
   - 截圖
6. 設定 Railway env vars:
   OKX_API_KEY_TESTNET=<key>
   OKX_API_SECRET_TESTNET=<secret>
   OKX_PASSPHRASE_TESTNET=<passphrase>
   OKX_KEY_CREATED_AT_TESTNET=2026-XX-XX
   OKX_KEY_ROTATE_BY_TESTNET=2026-XX-XX  # +14 天
7. 部署:Railway 自動拉新 env → 重啟 executor → 驗證 connect 成功
8. 在 docs/api_key_log.md 新增一行:
   "2026-XX-XX testnet key created, rotate by 2026-YY-YY"
```

### 2.2 Live key(Stage 3)

額外步驟:
```
9. 進入 live 前,確認帳戶餘額 = Stage 3 預算($100)
   - 不要多放錢進 trading 帳戶(壞人能下單虧的金額有限)
   - 主帳號保留資金,只 transfer Stage 3 預算到 trading 子帳號
10. 用 OKX_API_KEY_LIVE / SECRET_LIVE / PASSPHRASE_LIVE(獨立 env var)
11. 部署前先在 Railway dashboard 確認 STAGE=live env var 已設
    (okx_integration_design.md §8.2 validate_config 會 fail-fast)
```

### 2.3 部署後驗證(必跑)

```python
# scripts/validate_okx_key.py(新建)
import os, requests, hmac, base64, time, hashlib

key = os.environ["OKX_API_KEY"]
secret = os.environ["OKX_API_SECRET"]
passphrase = os.environ["OKX_PASSPHRASE"]

# 1. 確認 key 能讀 balance
# 2. 確認 key 權限不含 withdraw/transfer
# 3. 確認 IP whitelist 正確(若 IP 錯,401)
# 4. 確認 NTP drift < 1s
```

每次新 key 部署後跑一次,輸出結果寫進 `docs/api_key_log.md`。

---

## 3. Rotation(輪替)

### 3.1 排程

| 提前天數 | 觸發行為 |
|---|---|
| -7 天 | Telegram warning「key X 即將到輪替期」 |
| -3 天 | Telegram warning + 開始準備新 key |
| 0 天(輪替日)| 執行 §3.2 流程 |
| +7 天 | 若仍未換 → critical alert + halt executor |

實作:cron / scheduled job 每天讀 `OKX_KEY_ROTATE_BY_*`,比對今天日期。

### 3.2 Rotation 流程

```
1. 新建 key(§2 流程)
2. Railway 新建 env vars(temp 命名):
   OKX_API_KEY_NEW=<new key>
   OKX_API_SECRET_NEW=<new secret>
   OKX_PASSPHRASE_NEW=<new passphrase>
3. 部署測試 deployment(用 NEW 變數)
4. 跑 validate_okx_key.py 確認 NEW key 健康
5. 切換:
   a. Halt executor(透過 admin endpoint 或重啟前 set HALTED)
   b. 確認無 open position(或等 trail/time_cap 自然平倉)
   c. Railway:OKX_API_KEY_TESTNET ← OKX_API_KEY_NEW(覆蓋)
   d. 刪除 _NEW 後綴變數
   e. 重啟 executor
   f. 驗證 reconnect + cold-start reconciliation 通過
6. 撤銷舊 key:
   OKX → API page → 找舊 key → Delete
   (這步別省 — 舊 key 留著等於多一個攻擊面)
7. 更新 docs/api_key_log.md:
   "2026-XX-XX testnet key rotated, old key revoked, new rotate by 2026-YY-YY"
```

### 3.3 為什麼要先 halt 再切?

- 切 key 瞬間,WS 重連需要新 key 簽章
- 若有 open position 期間切,WS 短暫斷線,trailing stop 可能脫離掌控
- Halt 後等 position 自然平倉再切 → 0 風險

---

## 4. 監控與告警

### 4.1 即時偵測(executor 內)

| 訊號 | 行動 | trigger ID |
|---|---|---|
| HTTP 401(invalid key)| Halt + critical alert | C2 |
| HTTP 403(permission denied)| Halt + critical alert | C2 |
| HTTP 50001(API limit exceeded)| Backoff retry | B baseline |
| HTTP 51008(passphrase 錯)| Halt + critical alert | C2 |
| 連續 24h 仍 401 → Demote | C2 升級 |

### 4.2 主動健康檢查

每天 00:00 UTC 跑 `validate_okx_key.py`,結果寫進 health log。
連續 2 天失敗 → critical alert。

### 4.3 過期前告警(§3.1 已述)

---

## 5. Compromise Response(假設 key 外洩)

### 5.1 偵測訊號

任一發生 → 假設外洩,執行 §5.2:

- 看到不是自己下的 order(透過 OKX web 或 reconciler 偵測)
- API key 在不熟 IP 登入(OKX 通常會 email 通知)
- 帳號突然多了不認識的 sub-account
- Telegram 收到不是自己觸發的 webhook signal
- GitHub secret scanner alert(若不小心 commit 過)
- 任何「我覺得有點怪」的直覺

### 5.2 Emergency 流程(60 分鐘內完成)

```
T+0:  立即:
      a. OKX → API page → 立刻 Delete 該 key(不要 disable,直接 Delete)
      b. OKX → 安全中心 → 暫停所有提幣(security freeze)
      c. Halt executor(透過 Railway env STAGE=disabled 強制 fail-safe)

T+5:  檢查:
      d. OKX → 訂單歷史:有無近 24h 不認識的 order
      e. OKX → 充提幣:有無 pending withdrawal 申請
      f. OKX → 子帳號 / 內部轉帳記錄

T+15: Damage assessment:
      g. 若有未授權 order → 記錄損失金額 + 時間 + 找 OKX 客服爭議
      h. 若有 pending withdrawal → 第一時間 cancel(security freeze 應該擋住,但確認)

T+30: 重建:
      i. 換 OKX 登入密碼
      j. 重設 2FA(用新的 authenticator)
      k. 檢查 OKX 登入紀錄,撤銷所有不熟 session
      l. (若涉及 live key)考慮把資金轉到完全新的 OKX 帳號

T+60: 復盤:
      m. 寫進 mistake.md:key 怎麼洩漏的(commit 過?screenshot?共享?)
      n. 排查同類風險:其他 key / Telegram bot token / Railway env vars
      o. 全部其他 key 一起 rotate(假設同源)
```

### 5.3 預防(不要走到 §5.2)

每月 1 日 reminder 自我檢查:
- [ ] 沒有 key 在 git history(`git log -S OKX_API_KEY --all`)
- [ ] 沒有 key 在 Telegram / Slack / Notion / 任何雲端文件
- [ ] 沒有 key 在本地未加密文件
- [ ] Railway env vars 列表只有預期的 key
- [ ] OKX 登入紀錄無不熟 session

---

## 6. Local Dev Key 規範

本地開發是 key 外洩的高風險場景。原則:

```
1. 本地 dev 用獨立 demo key(永遠 simulated)
2. .env 永遠 gitignored(.gitignore 已涵蓋,但每次新 clone 確認)
3. 開發完不留 key 在 .env(刪掉或加密)
4. 截圖前先把 terminal 的 env var 清掉
5. 用 direnv / dotenvx 自動載入,不要 export 到 shell history
6. ~/.bash_history / ~/.zsh_history 不該有 export OKX_*
   - 用 ` export OKX_KEY=xxx`(前置空格)避開 history(zsh 設 HISTCONTROL=ignorespace)
```

---

## 7. 環境變數命名規範

```
OKX_API_KEY_TESTNET       # Stage 2 testnet
OKX_API_SECRET_TESTNET
OKX_PASSPHRASE_TESTNET
OKX_KEY_CREATED_AT_TESTNET
OKX_KEY_ROTATE_BY_TESTNET

OKX_API_KEY_LIVE          # Stage 3+ live(主帳號)
OKX_API_SECRET_LIVE
OKX_PASSPHRASE_LIVE
OKX_KEY_CREATED_AT_LIVE
OKX_KEY_ROTATE_BY_LIVE

OKX_API_KEY_DEV           # Local dev only
OKX_API_SECRET_DEV
OKX_PASSPHRASE_DEV

STAGE                     # paper | testnet | live | disabled
```

`okx/config.py` 根據 `STAGE` env 載對應 key。
`STAGE=disabled` 是 emergency kill — executor 完全不啟動。

---

## 8. 文件版本管理

| Version | Date | Changes |
|---|---|---|
| 0.1 | 2026-05-25 | Initial draft |

---

## 9. 附錄:`docs/api_key_log.md` 範本

進 Stage 2 時新建,每次 key 動作都加一行:

```markdown
# OKX API Key Log

| Date | Stage | Action | Old key suffix | New key suffix | Next rotate | Notes |
|---|---|---|---|---|---|---|
| 2026-XX-XX | testnet | created | -      | ...abc | 2026-YY-YY | initial demo |
| 2026-YY-YY | testnet | rotated | ...abc | ...def | 2026-ZZ-ZZ | scheduled |
| 2026-AA-AA | live    | created | -      | ...xyz | 2026-BB-BB | Stage 3 start |
```

只記 suffix(後 4 碼),完整 key 永遠不寫進這份 log。
