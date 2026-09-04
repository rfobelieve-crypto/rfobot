# 要接的交易所清單（2026-09-04）

分三層：**現在只讀行情**（零憑證，已在跑）、**$50 實盤要簽單的**、**之後才碰的**。
「接」的意思在每一層不同——第一層是讀公開簿口，第二層是簽單，混在一起看會高估工作量。

## 第 0 層：現在就在讀，零憑證（10 個場館、3,164 個配對，每 3 分鐘一輪）

| 場館 | 標的數 | 性質 | 讀取端 |
|---|---|---|---|
| `hl_core` | 177 | Hyperliquid 主場（加密） | 已接 |
| `binance` | 714 | CEX，含 **188 個 TradFi 股票永續** | 已接（09-04） |
| `bitget` | 777 | CEX，含股票／XAUT／PAXG／COPPER | 已接 |
| `okx` | 453 | CEX，含股票／KR200／SAMSUNG／XAG | 已接 |
| `lighter` | 216 | zkLighter mainnet，**費率表 0/0** | 已接 |
| `xyz` | 103 | HIP-3 builder dex（TradeXYZ）——黃金白銀原油指數個股 | 已接 |
| `lighter-rh` | 47 | zkLighter Robinhood 鏈，**0/0**，報價 USDG | 已接 |
| `para` | 25 | HIP-3 dex（含 2Y/10Y/30Y 殖利率） | 已接 |
| `io` | 5 | HIP-3 dex（Entropy）——股票／私募永續 | 已接 |
| `mkts` | 4 | HIP-3 dex（指數） | 已接 |

**這一層不需要你做任何事。** 另外 5 個 HL builder dex（flx／vntl／hyna／km／abcd／cash）
量為 0、簿口全空，掃描器自動跳過。

## 第 1 層：$50 實盤要簽單的 —— 只有 4 個（其實是 3 套認證）

規格 §1.08 鎖死只做兩個配對：`NBIS`（io ↔ lighter）與 `NVDA_LL`（lighter ↔ lighter-rh）。

| # | 場館 | 認證 | 你要準備 | 現況 |
|---|---|---|---|---|
| 1 | **HL 主場 + io**（以及未來所有 HIP-3 dex） | 錢包私鑰簽 EIP-712，**一把 key 通吃所有 dex** | 建 **API wallet**（能下單撤單、**永不能提幣**）＋ 獨立錢包只放這條線的錢 | 程式已有（`venue_hl.py`），待逐行審 |
| 2 | **Lighter mainnet** | 帳號索引＋API key 索引＋API 私鑰 | 一組 | 程式已有 |
| 3 | **Lighter-RH** | 同上，**但是不同的鏈、不同的帳號** | **另一組** | ⚠ 程式現在兩條鏈共用同一組 → **必修**，否則 `NVDA_LL` 簽不了兩腿 |

所以第 1 層實際上是：**一個 HL API wallet ＋ 兩組 Lighter 憑證**。

## 第 2 層：M1 只要唯讀 key（不下單，只讀成交回執）

| 場館 | 要什麼 | 用途 |
|---|---|---|
| HL | 只要 `HL_ACCOUNT_ADDRESS`，**不用金鑰**（`userFills` 是公開的） | 讀帳單費率 |
| OKX | key＋secret＋passphrase，**唯讀** | 同上 |
| Bitget | key＋secret＋passphrase，**唯讀** | 同上 |
| Binance | key＋secret，**唯讀** | 同上 |

## 第 3 層：之後才碰（現在不要接）

| 場館 | 為什麼還不接 |
|---|---|
| **OKX／Bitget／Binance 的下單** | 錄價程式**完全沒有 CEX 下單碼**，是新工作。而且 $50 那週的兩個配對用不到 |
| **xyz／para／mkts** | 跟 io 同一把 HL key，**接一個等於接全部**——但要先有返佣（A6）才划算 |
| **Polymarket** | 本機 DNS 連不到（ISP 黑洞），要東京 VPS 才能碰 |
| **IBKR**（真實股價那條腿） | 那對兄弟做的事。要帳戶＋市場資料訂閱＋資金，是另一個量級的承諾 |

## 上線前的憑證檢查表

- [ ] HL：建 API wallet（不是主錢包私鑰），主錢包只放這條線的錢
- [ ] Lighter：**兩條鏈各一組**，`config.py` 加 `LIGHTER_RH_*` 環境變數（B2）
- [ ] CEX：子帳戶 ＋ **只勾交易不勾提幣** ＋ IP 白名單（東京 VPS 的固定 IP）
- [ ] 唯讀 key 先行——M1 不需要下單權限
- [ ] 金鑰不進 git、**不進 Railway**（引擎在 VPS，儀表板不碰金鑰）
- [ ] 帳戶隔離：不與 V7／獵取／個人資金共用（CLAUDE.md 已為此付過兩次代價）
