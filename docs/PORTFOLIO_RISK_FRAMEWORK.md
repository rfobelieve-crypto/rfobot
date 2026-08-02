# PORTFOLIO_RISK_FRAMEWORK — 統一風控框架設計稿

狀態：**設計稿 v1（2026-07-30）**。依 TODO §0.4 的時序紀律：**現在只做設計、
不動 live 代碼**。重構觸發點 = 第二條策略通過自己的 gate（sweep-failure
變體 B Gate F，或撤單線兩個判決之一翻正）。在那之前 V7 executor 一行不動。

---

## 0. 核心原則（整份文件只有這一條是原則，其他都是取捨）

**風險在組合層看待，策略是元件。**

策略不擁有帳戶、不直接下單、不各自為政地管風險——它們只產生「意圖
(intent)」。組合層擁有：曝險帳本、風險預算、kill switch、與唯一的執行通道。

對沖在本框架 = **淨額互抵**（兩策略反向時帳本層自然抵銷），不是買避險
工具（$300 規模下成本 > 保護；2026-07-29 使用者澄清：對沖是概念不是需求）。

## 1. 現狀基線（設計的起點，不是白紙）

今天的系統是「單策略直連」：V7 訊號 → `indicator/okx/executor.py` →
OKX。風控全部長在這一條管線裡：

| 既有機制 | 位置 | 值 |
|---|---|---|
| 帳戶級 kill：CAP-2 over-funding HALT | `kill_checks.py` | equity > 1.5× 基準 |
| 帳戶級 kill：CAP-3 daily loss HALT（自動恢復） | 同上 | **−20%/日** |
| 帳戶級 kill：CAP-4 total loss DEMOTE（終態） | 同上 | **−30% 累積** |
| 連線失聯 kill switch | executor | 斷線→撤單 |
| 單倉限制 | config | `max_position_count = 1` |
| 有效槓桿 | config | NOTIONAL_LEV_MULT = 2.0× |
| Stage 3 資本上限 | config guard | live ≤ $500 |

這些**全部保留、數字一個不動**。框架不是替換它們，是在它們**上面**加一層
（策略層），並把「V7 專屬」的部分抽成「所有策略共用」。

## 2. 目標架構

```
┌─ 策略 adapters（只產 intent，不碰交易所）─────────────┐
│  S1 V7 4h 模型     S2 sweep-failure     S3 撤單流(候選) │
└──────────────┬───────────────────────────────────────┘
               ▼  intent {strategy, symbol, side, risk_pct, stop, ttl}
┌─ Portfolio Risk Engine（新，唯一守門員）───────────────┐
│  a. 中央曝險帳本（現有部位 + 未決 intent 全記錄）        │
│  b. 預算檢查：策略日/總虧損、併發數、單筆 risk%          │
│  c. 組合檢查：淨曝險上限、相關性預算、同幣別碰撞          │
│  d. 兩層 kill：策略層 HALT ／ 帳戶層 CAP-1..4（至高）    │
└──────────────┬───────────────────────────────────────┘
               ▼  核准的 order
┌─ 執行層（共用：OKX client facade、對帳、告警、ledger 落庫）┐
└──────────────────────────────────────────────────────┘
```

策略被拒絕的 intent 也要落庫（拒絕原因），否則「風控擋掉了什麼」永遠
不可審計。

## 3. 元件設計

### 3.1 中央曝險帳本（DB，兩張新表）

沿用現有命名慣例（`v7_okx_*` → 組合層用 `pf_*` 前綴），MySQL：

```sql
pf_intents(
  id, ts, strategy, symbol, side, risk_pct, stop_px, entry_ref_px,
  ttl_sec, status ENUM(pending/approved/rejected/expired),
  reject_reason, decided_ts
)
pf_positions(         -- 統一 trade ledger：所有策略同構
  id, strategy, symbol, side, entry_ts, entry_px, size, risk_usd,
  stop_px, exit_ts, exit_px, exit_reason, gross_pnl, fees, net_pnl,
  net_r,              -- 以進場時風險為分母的 R 倍數（跨策略可比）
  equity_after
)
```

同構 ledger 是「組合 Sharpe/MDD/歸因可算」的資料底，也直接是每條新策略
Gate-B 式驗證（30-50 筆乾淨樣本）的記錄格式。V7 現有 `v7_okx_positions`
不遷移——P1 階段用 dual-write（V7 executor 多寫一份 `pf_positions`，
行為零改動），歷史由 view 合併。

### 3.2 兩層 kill switch

**帳戶層（既有，至高無上）**：CAP-1..4 原封不動。任何帳戶層觸發 →
**所有**策略一起停。這層永遠比策略層優先，數字修改仍需 override 儀式。

**策略層（新）**：每策略獨立的
- `strat_daily_loss_cap`：策略當日 net_r 加總 ≤ −X → 該策略當日 HALT
  （自動恢復，隔日重置）
- `strat_total_dd_cap`：策略自高點回撤 ≤ −Y → 該策略 **DEMOTE 回 shadow**
  （終態，需人工 + 重驗 gate 才回來）
- 策略層觸發**不影響其他策略**——這是「薄策略組合」的本意：一條線壞了
  降級它自己，不拖全家。

X/Y 的初值提案（可談）：日 −5R 或 −2% equity 取小者；總回撤 −6%
equity。原則：**策略層 cap × 策略數 < 帳戶層 cap**，讓策略層永遠先於
帳戶層觸發（帳戶層淪為真正的最後防線，而不是第一線）。

### 3.3 風險預算（evidence-scaled，不是拍腦袋）

| 預算軸 | 初值 | 依據 |
|---|---|---|
| 單筆風險 | 0.15–0.25% equity | 變體 B 併發研究的可交易區間 |
| 每策略併發上限 | sweep: 5–10；V7: 1（現值） | 同上（cap 3 仍存活、無上限 −64R/日不可交易） |
| 策略間總淨曝險 | ≤ 2× equity 名目 | 沿用現行 NOTIONAL_LEV_MULT 精神 |
| 預算解鎖 | 過 gate 才升級 | 同 V7 staged framework：證據累積→規模累積 |

新策略進場預算一律從最小檔開始（sweep 若過 Gate F：cap 5 × 0.15%），
跑滿一個 Gate-B 式樣本再談上調——與 V7 的 $100→階梯完全同構。

### 3.4 相關性／併發預算（9 幣教訓的制度化）

變體 B 的日分群 bootstrap 已量化過：9 幣連動把 t 從 3.35 灌到 1.95
（VIF 2.95）。組合層同樣要防「N 條策略其實是一條策略」：

- **事前**：同幣別碰撞規則——兩策略同時要開同向同幣 → 合併算一份曝險
  （不是兩份）；反向 → 帳本淨額，只把淨差送執行層。
- **事中**：滾動 30d 策略日 PnL 相關性入每週 PortfolioClocks 報告；
  |ρ| > 0.5 → 兩策略的合併預算縮到單策略檔（相關就不配拿兩份預算）。
- **誠實限制**：全 crypto 策略共享 beta，正交性有天花板。真分散最終要
  跨資產（sweep 的 MNQ 方向一致是種子），本框架先把「假分散拿雙倍預算」
  這個洞堵上。

### 3.5 策略 adapter 介面（設計輪廓）

```python
class StrategyAdapter(Protocol):
    name: str
    def poll(self, now) -> list[Intent]: ...   # 只讀市場/DB，產 intent
    def on_fill(self, position): ...           # 回報，不得反向下單
```

V7 重構成 adapter 是**觸發點之後**的事；屆時 executor 的訊號→下單膠水
搬進 adapter，OKX 通道與對帳留在執行層。撤單線若判決翻正，同介面掛入
（它天然是 confirm/veto 型，可能以「濾網服務」而非完整策略掛——見
§4 開放問題）。

## 4. 落地計畫（觸發點之後才動 P1+）

- **P0（本文件）**：設計定稿，等第二條策略過 gate。
  **2026-08-02 實作進度**：規則層已寫成可測的函式庫
  `indicator/portfolio/`（limits / ledger / risk_engine），27 個測試釘住
  §3.2-§3.4 的每一條規則。**刻意零接線**——`tests/test_portfolio_risk.py`
  裡有一條 AST 測試，只要 app.py / executor.py / runner.py /
  BTC_perp_data.py 任何一個 import 到這個套件就會紅。行為零改動，
  DDL 只是字串常數還沒建表。這樣 Gate F 過關那天，要接的是線路，
  不是還沒被審過的規則。
- **P1 帳本先行（零行為改動）**：建 `pf_*` 表；V7 executor dual-write；
  PortfolioClocks 加組合視圖（跨策略 net_r 曲線、相關性矩陣）。跑 2 週
  確認帳本與 `v7_okx_positions` 對帳一致。
- **P2 風控引擎 + 第二條策略**：Risk Engine 模組（intent 審批 + 策略層
  kill）；sweep executor 只透過它下單；V7 仍直連（風險最低的過渡——
  舊路徑不動，新路徑走新閘門）。
- **P3 收斂**：V7 遷移成 adapter，全部訊號走同一守門員；帳戶層 kill
  從 V7 executor 抽到引擎層（數字不變，位置搬家）。

每個 P 有自己的對帳驗證，任何一步 miss 就停在該步——與 mistake.md
「收緊 guard 前先枚舉部署現值」同紀律：P1-P3 每步都先列出受影響的
Railway env / 排程 / 表，再動手。

## 5. 開放問題（設計稿要使用者拍板的）

1. 策略層 cap 初值（§3.2 的 −5R/−2% 與 −6%）接受嗎？
2. 撤單線若只有濾網價值：以「服務」掛在 sweep adapter 內，還是保留
   獨立策略席位（拿自己的預算）？
3. P1 的 dual-write 可以在觸發點**之前**先做嗎？（它零行為改動、純加
   一份寫入，好處是 Gate F 過關當天帳本已有 V7 的存量資料——但依
   §0.4 字面「不動 live 代碼」它應該等。傾向：等，維持紀律的乾淨。）

## 6. 不變量（本框架永不觸碰）

帳戶層 kill 數字（−20%/−30%）、槓桿 cap（有效 2x／帳戶 10x）、Stage 3
資本 guard（$500）、首筆人工確認慣例、hit kill → 降階重驗。框架增加
保護層，**任何「框架上線」的理由都不構成鬆動既有防線的理由**。
