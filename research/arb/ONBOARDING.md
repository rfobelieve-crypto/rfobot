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
