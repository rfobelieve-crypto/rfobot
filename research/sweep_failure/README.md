# Sweep-Failure Reversal — 策略 #3 候選（2026-07-11 遷入）

> ## ⚠️ CORRECTION（2026-07-28 成本審計）——先讀這段再讀下面的表
>
> 遷入版 `sweep_core.py` 的進場滑價**符號寫反**（`lvl - d*SLIP*A` = 對我們
> **有利**），與出場的不利滑價在時間出場路徑上**正好抵銷**——所以下表
> 「成本 0.05 ATR/邊」的驗證履歷實質上是**零成本回測**（已證：SLIP=0.05
> 跑分還略優於 SLIP=0）。符號已修正（含停損出場滑價），並以**逐幣真實
> bps 費用**（taker 5+滑2 進場 / maker 3 或 taker 6 出場 / 停損 10）重算：
>
> | 情境 | pool meanR | t | PF | WR | 正幣數 |
> |---|---|---|---|---|---|
> | 原表（實質零成本） | +0.062 | +8.2 | 1.29 | 57% | 9/9 |
> | **A 目標執行**（出場掛單）| **+0.0255** | **+3.35** | 1.11 | 54% | 9/9 |
> | B 全 taker 保守 | +0.0173 | +2.29 | 1.07 | 54% | 8/9 |
>
> 逐幣 t 值多在 0.2-1.2（僅 ADA +2.98、BTC +1.89 較高）；前半段 5/9 幣為
> 負、後半段普遍轉強。9 幣連動使 pool t 另有灌水。**結論：edge 從「肥」
> 修正為「薄而依賴執行」，但仍是全 repo 最佳候選（n 大）。** Gate F 前
> 進標準見 `sweep_forward.py` docstring（n≥1400、CI 低緣>0、≥6/9 正）。
> 教訓已記 mistake.md 2026-07-28：**宣稱含成本的回測，先跑 cost=0 對照
> ——如果含成本 ≥ 零成本，成本模型壞了。**

> 來源：trading-view-MCP research sandbox（`C:\Users\rfo\Desktop\flowbot\trading view MCP`）
> 完整研究軌跡與所有中間版本在那邊；此處為整理後的最小核心。
> **本資料夾為純加法研究程式碼：不 import 任何 production 模組、不寫 DB、不動主系統。**

## 假設（預先登記）

亞/倫/紐盤形成的 swing 高低點 = 停損堆積（流動性）。掃單獵取後，若價格於
**W=8 根 1H 內回到被掃的 level** = 獵取失敗；價格傾向朝**穿越方向**再漂移
8–12 根（短命資金流效應，H=20 已衰減為零）。

- 進場：回觸 level 時停損觸發單成交於 level（buyside 掃單失敗→SHORT / sellside→LONG）
- 出場：HOLD=8 根時間出場（主）；3.5×ATR 災難停損（定倉位，罕觸發）；無停利
- 單一持倉/標的；9 幣 basket：BTC ETH SOL BNB XRP DOGE ADA LINK AVAX（1H）

## 驗證履歷（2024-01 → 2026-07，各 ~22.1k 根 1H，成本 0.05 ATR/邊）

| 檢驗 | 結果 |
|---|---|
| 9 標的 | **9/9 正**：PF 1.16–1.54、WR 54–59%、單幣 MDD 6.5–19%（1% risk） |
| 合池 | 6,932 筆、PF 1.29、t=+8.27 |
| 參數穩健 | W∈{4,8,12}、HOLD∈{8,12}、PIVOT∈{5,10} 全正；HOLD=20 衰減殆盡 |
| 前後半 | 皆正（各參數組） |
| 跨資產 | MNQ 15m 同向（+0.205 ATR，t=1.31，n=508 不足顯著但方向一致） |
| 組合模擬（共用權益、逐棒 MTM）| 0.5% risk → 年化 +128% / MDD 16.5%；1% → +382% / 31% |
| 成本壓力 | 總成本 0.10 ATR → 期望 +0.055、不顯著 ⇒ **maker 級費率是硬需求** |

## 誠實註記（下結論前必讀）

1. **翻號出身**：假設誕生自「限價回撤續走測試的顯著負值」反向（snooping 折扣）。
   雖然它同時等於使用者長期的主觀論點（假突破反轉），forward 驗證才是真正的 gate。
2. **量測教訓**（此研究曾兩次抓到自己的 artifact）：
   - outcome 必須從「決策當下真正成交得到的價」量起（level at retest-touch），
     從訊號前價位量會系統性灌水（kNN 版因此作廢，t=7 → 修正後 −4）。
   - 鄰居/特徵之正規化與結果窗必須嚴格因果。
3. 9 幣高度連動 ⇒ 合池 t 有灌水；極端行情會 9 單同開同虧。
4. LDC/kNN 條件化**無加值**（已測）——edge 在結構規則本身。

## 檔案

| 檔 | 用途 |
|---|---|
| `sweep_core.py` | 偵測 + 可交易回測核心（env: PIVOT/W/HOLD/DIS/SLIP） |
| `fetch_klines.py` | 從 Binance 公開 REST 抓 9 幣 1h → `.cache/`（勿 commit 資料） |
| `run_backtest.py` | 逐標的 + 合池 + 前後半報告 |

```bash
python research/sweep_failure/fetch_klines.py          # 一次性抓資料
python research/sweep_failure/run_backtest.py          # 重現驗證
```

## 建議的 TODO.md 條目（由 owner 決定是否貼入）

```
### N. 掃單失敗反轉（策略 #3 候選）— 價格結構事件系，與 v7(ML)/擠壓(撤單) 正交
研究完成（見 research/sweep_failure/README.md）：9/9 幣正、PF1.29 合池 t=8.3、
參數穩健、成本敏感（需 maker 費率）。翻號出身 ⇒ forward 是真 gate。
- [ ] Gate F(forward)：獨立引擎 paper 累 100+ 筆，WR/exp 對上回測（54-59% / +0.06R）
- [ ] 過 Gate F → 依 staged framework 走（複用 OKX executor、kill switch、$100 級起步）
- [ ] 費率通道確認：來回 ≤0.05 ATR 當量（maker/VIP/BNB 折扣）
```

## 與既有系統的關係

- 訊號家族與 v7（taker/衍生品訂單流 ML）**正交**——TODO 已四次驗證該家族 1h+ 飽和；
  本策略是價格結構事件, 不吃同一口井。
- 與 `poc_sweep_study.py`（事件條件 × POC 區位、manual aid）互補：同樣的
  liquidity-sweep 事件母體，這裡走「失敗→反轉」的機械化路徑。
- 獨立 paper 引擎暫跑在本機（`trading view MCP\live_engine.py`，Telegram 推播）。
  **若未來下單，必須先砍獨立引擎再併入主系統**，避免同帳戶互相打架。
