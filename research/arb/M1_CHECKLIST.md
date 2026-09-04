# M1 執行清單：幾筆最小單換掉整個費率桶（2026-09-04）

> **目的**：把 `fees.py` 從「頁面寫的」變成「帳單上發生的」。總成本 <$15。
> **你做**：註冊、下單、貼唯讀 key。**我做**：讀回執、算實際 bps、改 `fees.py`。
> 讀取端已就緒：`research/arb/fee_receipts.py`——**只讀，沒有任何下單或簽單程式碼**。

## 1 註冊時就要選對的事（省得之後重來）

| 場館 | 註冊要點 | 為什麼 |
|---|---|---|
| **OKX** | 開**子帳戶**給套利用；API key 只勾**交易**、**不勾提幣**；IP 白名單先不設（VPS 還沒有） | 帳戶隔離——CLAUDE.md 為共用帳戶付過兩次代價（06-05、07-27 手動爆倉） |
| **Bitget** | 同上。順便看 **rToken 的 UTA 是否預設開啟** | §1.11 那條線的前提 |
| **Binance** | 同上。注意股票永續是 **TRADIFI_PERPETUAL**，不是一般 PERPETUAL | §1.05；我一開始就是被這個篩子騙過 |
| **HL（Hyperliquid）** | 用**獨立錢包**，不要用有其他資產的那個。之後要建 **API wallet**（能下單撤單、**永不能提幣**） | 錢包私鑰本身就是錢；API wallet 把 DEX 腿的風險拉平到 CEX 水平 |
| **Lighter mainnet** | 一組帳號＋API key | |
| **Lighter-RH** | **另一組**（不同鏈、不同帳號） | `config.py` 現在兩鏈共用同一組憑證——**這是上線前必修的 bug**，而且 `NVDA_LL` 兩腿都是 Lighter |

**用邀請碼註冊時**（你提的 20–40%）記得 §1.11 的教訓：**要問的是「我自己交易的
折扣」（Benefit），不是「推薦別人可分到的收入」（Rebate）**——Entropy 那次就是這
兩個數字同名，害整條線的算式錯了五天。

## 2 M1 的幾筆單

最小下單名目（2026-09-04 查證）：

| 場館 | 最小單 | 建議下什麼 |
|---|---|---|
| OKX | 股票永續 $1.9–7.7 | `NVDA-USDT-SWAP` 0.01 張 ≈ $2.3 |
| Bitget | $5 | `NVDAUSDT` |
| Binance | $5 | `NVDAUSDT`（TradFi 永續） |
| HL | $10 | `BTC` 或 `io:NBIS` |

**每所各下一筆開倉、再平掉**——一來一回才有兩次 taker 回執。
**如果掛得到單成交更好**，那會同時量到 maker 費率（`fee_receipts.py` 會分開統計）。

## 3 我要的東西（貼給我就好，唯讀即可）

```
.env 補：
  HL_ACCOUNT_ADDRESS=0x...        ← HL 只要位址，不用金鑰（userFills 是公開的）
  OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE
  BITGET_API_KEY / BITGET_API_SECRET / BITGET_API_PASSPHRASE
  BINANCE_API_KEY / BINANCE_API_SECRET
```

**唯讀 key 就夠**——M1 不需要下單權限。

## 4 跑完會看到什麼

`python research/arb/fee_receipts.py --hours 48`：

```
OKX     2 筆成交
    taker  n=2  帳單 2.750 bps ｜ fees.py 2.750（費率表 5.00）｜ 差 +0.000 → MATCH
```

差 >0.5 bps 就標「**以回執為準改 fees.py**」——回執永遠是對的那一方。

**換掉的是**：F1–F8 整桶（8 項裡現在有 3 項標「未查證」），以及 `cost_model.py`
的「量到%」從 **43%** 往上。

## 5 同一晚可以順便做完的四項（零額外成本，全是查證不是研究）

- [ ] **M7 劃轉成本**：做一次 HL→Lighter、CEX→HL 的小額劃轉，記**費用與到帳分鐘**
      → 換掉第 5 桶（T1/T2/T3 現在全是假設，值 2.0 bps/筆）
- [ ] **第 3 桶保證金率**：各所的初始／維持保證金率，文件或帳戶頁看得到 → 換掉 C1/C2
- [ ] **A7**：兩腿的保證金是不是分開帳戶（HIP-3 dex 之間、HL 主場 vs dex）
      → 決定要放幾份錢
- [ ] **rToken**：能不能當保證金、折算率多少、空永續那側要不要付股利（§1.11）

**這四項加起來會把「量到%」從 43% 推到 ~70%**——而且全部是同一次登入就能查完的事。

## 6 提醒兩件已知的坑

1. **不要用主帳戶**。套利的錢跟 V7／獵取／個人資金分開，這條在 CLAUDE.md 是硬規則。
2. **金鑰不進 git、不進 Railway**。引擎將來在 VPS 上，儀表板不碰金鑰。
