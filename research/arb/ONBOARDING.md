# 交易所端開通清單（2026-09-04）——每一家要什麼、錢包還是 API key

> 照這份走一遍，交易所端就完成了。**分兩批**：
> 批 A ＝ M1 今晚要用的（唯讀就夠）；批 B ＝ 之後真的下單才需要（先別急著開權限）。
> 上游 `.env.example` 是憑證的權威來源，本檔只是把它翻成人話並補上帳戶策略。

---

## 先分清楚：三種完全不同的認證

| 類型 | 誰用 | 外洩的後果 | 能不能限制 |
|---|---|---|---|
| **錢包私鑰** | HL 主場、所有 HIP-3 dex（io/xyz/para/mkts） | **錢直接被拿走** | ❌ 不能——但**有替代品**（見下） |
| **HL Agent（API）錢包** | 同上，但是子鑰 | 只能亂交易，**不能提幣／轉帳** | ✅ 這是正解 |
| **交易所 API key** | OKX／Bitget／Binance／Lighter | 只能亂交易（若不勾提幣） | ✅ 勾選權限＋IP 白名單 |

**核心觀念**：DEX 那邊不要用主錢包私鑰，用 **Agent 錢包**；CEX 那邊不要用主帳戶，用**子帳戶**。

---

## 批 A：今晚要弄的（M1，唯讀即可）

### A-1 Hyperliquid（HL 主場 ＋ io ＋ 所有 HIP-3 dex）

**這一家最重要，因為一把 key 通吃所有 HIP-3 dex（io／xyz／para／mkts）。**

| 步驟 | 做什麼 |
|---|---|
| 1 | 準備一個**全新的錢包**（MetaMask/Rabby 開新帳戶即可），**不要用有其他資產的那個** |
| 2 | 到 app.hyperliquid.xyz 連錢包、存入小額 USDC（$50 以內夠 M1） |
| 3 | **M1 只需要主帳戶地址**——`HL_ACCOUNT_ADDRESS=0x...`。`userFills` 是公開端點，不用金鑰就讀得到成交回執 |
| 4 | （批 B 才做）到 **app.hyperliquid.xyz/API** 建 **Agent 錢包**，把 agent 的私鑰填 `HL_PRIVATE_KEY` |

**io（Entropy）不用另外註冊**——它是 HL 上的一個 dex，同一個帳戶就能交易。
你已經有的推薦連結（Benefit 25%）掛在這個帳戶上。

**給我**：`HL_ACCOUNT_ADDRESS=0x...`（就這一個，公開資訊）

---

### A-2 OKX

| 步驟 | 做什麼 |
|---|---|
| 1 | 註冊（用邀請碼），完成 KYC |
| 2 | **開一個子帳戶**專給套利用，主帳戶不碰 |
| 3 | 子帳戶入金小額 |
| 4 | API 管理 → 建 key → **只勾「讀取」**（M1 階段連交易都不用勾） |
| 5 | IP 白名單**先留空**（東京 VPS 還沒有；之後補） |

**給我**：`OKX_API_KEY` / `OKX_API_SECRET` / `OKX_API_PASSPHRASE`

⚠ OKX 的 passphrase 是你建 key 時自己設的字串，**不是登入密碼**，建完看不到第二次。

---

### A-3 Bitget

同 OKX 的流程。額外看一件事：

- **rToken / UTA**：帳戶頁確認統一帳戶（UTA）是否開啟、rToken 能不能當保證金、
  折算率多少 → 這是 §1.11 那條線的前提

**給我**：`BITGET_API_KEY` / `BITGET_API_SECRET` / `BITGET_API_PASSPHRASE`

---

### A-4 Binance

同上。注意：**股票永續在 Binance 是 `TRADIFI_PERPETUAL`**（`GPROUSDT`、`NVDAUSDT`、
`SAMSUNGEMUSDT`…），跟一般加密永續是不同的 contractType，下單頁可能在不同分頁。

**給我**：`BINANCE_API_KEY` / `BINANCE_API_SECRET`（無 passphrase）

---

### A-5 Lighter —— **這一家要開兩個帳號**

**mainnet 與 Robinhood 鏈是兩套獨立的帳戶與金鑰**，上游文件白紙黑字寫著。
而我們的 `NVDA_LL` 配對**兩腿都是 Lighter**（mainnet ↔ RH），所以非開兩個不可。

| | 網址 | 環境變數（要改造） |
|---|---|---|
| mainnet | lighter.xyz（主網） | `LIGHTER_ACCOUNT_INDEX` / `LIGHTER_API_KEY_INDEX` / `LIGHTER_API_PRIVATE_KEY` |
| Robinhood 鏈 | RH 部署 | **需要新增 `LIGHTER_RH_*` 三個**（程式端我會改） |

**用標準帳戶，不要開 Premium**——Premium 反而收費（0.4/2.8 bps），標準帳戶是 **0/0**。

**M1 階段**：Lighter 的成交回執讀取我還沒接（`fee_receipts.py` 目前只有 HL/OKX/Bitget/
Binance）。但它費率是 0，**沒什麼好量的**——先確認帳戶開得起來、記下 account index 就好。

---

## 批 B：之後真的下單才需要（今晚別急）

| 場館 | 要補什麼 |
|---|---|
| HL | 建 **Agent 錢包**，`HL_PRIVATE_KEY` 填 agent 私鑰（**不是主錢包私鑰**） |
| OKX／Bitget／Binance | key 權限加勾「交易」，**永遠不勾提幣**；補 IP 白名單（東京 VPS 的固定 IP） |
| Lighter ×2 | 各自產 API key（見 lighter-python 官方倉庫的產鑰流程） |

---

## 今晚的最小路徑（如果只想做完 M1）

1. **HL**：開新錢包 → 存 $50 → 把**地址**給我
2. **OKX／Bitget／Binance**：註冊（用邀請碼）→ 開子帳戶 → 入金 $20 → 建**唯讀 key** → 給我
3. **各下一筆最小單再平掉**（OKX `NVDA-USDT-SWAP` 0.01 張 ≈ $2.3；Bitget/Binance
   `NVDAUSDT` $5；HL BTC $10）
4. **Lighter**：兩個帳號開起來，記下 account index（M1 用不到，但下一步要）

**總金額 < $150（含入金），實際花掉的手續費 < $1。**

---

## 順便查的四件事（同一次登入，見 `M1_CHECKLIST.md` §5）

- 各所**初始／維持保證金率**（換掉成本模型第 3 桶）
- 做一次**小額劃轉**記費用與到帳分鐘（換掉第 5 桶）
- **HIP-3 dex 之間、HL 主場 vs dex 的保證金是不是分開的**
- Bitget **rToken 抵押率 ／ 空永續要不要付股利**

---

## 兩條不能破的線

1. **帳戶隔離**：套利的錢跟 V7／獵取／個人資金完全分開。CLAUDE.md 為共用帳戶
   付過兩次代價（2026-06-05、2026-07-27，兩次手動爆倉）。
2. **金鑰不進 git、不進 Railway**。引擎將來在東京 VPS 上，儀表板是唯讀的、不碰金鑰。

---

## API 隔離規格（2026-09-04 使用者要求，上線前凍結）

**「隔離」在這裡有四個維度，混在一起講會漏掉其中兩個。**

### 維度 1：帳戶隔離——套利的錢跟誰都不共用

| 場館 | 隔離方式 |
|---|---|
| HL／所有 HIP-3 dex | **獨立錢包**，只放這條線的錢 |
| OKX／Bitget／Binance | **子帳戶**，主帳戶不碰 |
| Lighter ×2 | 各自獨立帳號 |

**理由不是理論**：CLAUDE.md 記著兩次共用帳戶的代價（2026-06-05 $188→$0.05、
2026-07-27 $1218→$16.62，都是手動單在同一個帳戶把 executor 的倉一起帶走）。
**強平是帳戶級的**——別的部位爆倉會把套利的兩條腿一起清掉，而套利最怕的就是單腿裸露。

### 維度 2：金鑰隔離——一把鑰匙只開一扇門

**每個場館一把 key，每個用途一把 key。** 不共用、不複製。

| 用途 | 權限 | 給誰 |
|---|---|---|
| **讀成交回執**（M1、對帳） | 唯讀 | 研究端（本機） |
| **下單**（引擎） | 交易，**永不提幣** | 只在東京 VPS |
| **提幣** | — | **不發**。資金調度用人手在網頁上做 |

**HL 特例且重要**：`HL_PRIVATE_KEY` 填的必須是 **Agent 錢包私鑰**，不是主錢包私鑰。
Agent 能下單撤單、**永遠不能提幣或轉帳**（官方文件明載）。這一條把 DEX 腿的
金鑰風險拉平到 CEX 的水平——**沒有它，DEX 腿的私鑰就是錢本身**。

### 維度 3：程序隔離——誰能碰到金鑰

```
東京 VPS（引擎）          ← 唯一持有「交易」權限的地方
    ↓ 只寫 arb_live_* 表
MySQL
    ↓ 只讀
agent-mcp /public/arb-live
    ↓ 只讀
jarvis 儀表板（Railway）  ← 一顆按鈕都沒有，不碰任何金鑰
```

**硬規則**：
- 金鑰**不進 git**（`.gitignore` 已擋 `.env`）
- 金鑰**不進 Railway**——儀表板是唯讀顯示層，它拿不到也不需要
- 研究端（本機）只放**唯讀** key
- 套利的程式**永不 import 交易路徑**（CLAUDE.md 第 4 線隔離，已是硬規則）

### 維度 4：網路隔離——IP 白名單

**東京 VPS 有固定 IP 之後**，三家 CEX 的交易 key 都綁上去。
本機的唯讀 key 不綁（本機 IP 會變），但**唯讀 key 外洩的損失是零**。

錄價程式作者的清單裡也有這條：「IP 白名單＋金鑰登入」。

---

### 上線前的隔離檢查表（每一項都要能指出證據）

- [ ] HL 用的是 **Agent 錢包私鑰**，不是主錢包私鑰 → 到 app.hyperliquid.xyz/API 看得到
- [ ] HL 主錢包**只有這條線的錢**
- [ ] 三家 CEX 都是**子帳戶**，且 key **沒有提幣權限** → API 管理頁截圖
- [ ] Lighter **兩條鏈各一組**憑證（`LIGHTER_*` 與 `LIGHTER_RH_*` 值不同）
- [ ] 交易權限的 key **只存在於 VPS**，本機 `.env` 只有唯讀
- [ ] `git log -p` 搜不到任何 key（`.env` 在 `.gitignore` 裡）
- [ ] Railway 的任何服務都**沒有**套利的憑證
- [ ] VPS 固定 IP 已綁進三家的白名單

**任何一項打不了勾，就不下第一筆真單。**
