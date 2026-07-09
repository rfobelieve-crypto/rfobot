# EP 系列完整 Roadmap — 按系統建造進程排序

> 2026-07-07 定稿。每一篇對應 rfobot 建造的一個階段，素材全部來自
> mistake.md / CLAUDE.md 的真實事件。發文節奏：每篇間隔 3-7 天，
> 中文 LinkedIn + 英文 Medium 雙軌。

## 主線（系統進程順序）

| # | 標題方向 | 系統階段 | Hook（真實事件） | 素材來源 | 狀態 |
|---|---|---|---|---|---|
| EP1 | 為什麼一個學商的去做訂單流量化 | 起點（2022-2023） | 手動交易 2 年 → 決定系統化 | 個人經歷 | ✅ 已發 |
| EP2 | 先把資料做對 | 資料層（2023-2025） | Binance/OKX contract size 差 100 倍；5.5 天 timestamp 毒資料 | mistake 2026-04-12 | ✅ 中文已發，英文待發 |
| EP3 | 你的回測正在騙你（Walk-Forward） | 驗證方法論 | in-sample IC 0.60 vs 真實 0.10 | mistake 2026-04-13 ×2、2026-06-02 | ✅ 已寫好（中英） |
| EP4 | 特徵工程的殘酷真相 | 特徵層 | univariate IC +0.207 的「王牌特徵」進 ensemble 零提升 | mistake 2026-06-01、06-02、04-13 sparse interaction | 📝 未寫 |
| EP5 | 你的模型天花板是數學決定的 | 模型層 | 追了一整天 95% 勝率目標，發現 AUC 0.57 的上限是 68-72% | mistake 2026-04-14 | 📝 未寫 |
| EP6 | 從預測值到交易訊號 | 訊號層 | 6/19 錯過的完美訊號 + 為什麼不改規則（threshold sweep 陷阱） | mistake 2026-06-20 | 📝 未寫 |
| EP7 | 回測會賺 ≠ 交易會賺 | 策略層 | Gate A 證 edge 存在（739 訊號 CI）、Gate B 證執行保住 edge——為什麼要拆兩題 | CLAUDE.md 壓縮版 Stage 3→4 | 📝 未寫 |
| EP8 | 不用 CCXT 從零接交易所 | 執行層 | facade drift 同一個 bug 咬兩次（trail amend + set_leverage） | mistake 2026-06-16、06-07 trail 三輪修 | 📝 未寫 |
| EP9 | 紀律寫進 code，不靠意志力 | 風控層 | 31-trigger kill matrix、Kelly + vol drag 推導 2x 上限 | CLAUDE.md leverage ladder、kill_criteria | 📝 未寫 |
| EP10 | 爆倉那一天 | 實盤運營 | 2026-06-05 手動爆倉 + admin_heal 連環事故 → 降階重驗 | CLAUDE.md paper 移除、mistake 2026-06-07 | 📝 未寫（系列情緒高點，素材最強） |
| EP11 | 把系統開放給朋友跟單 | 擴展 | 多帳戶架構：加密、隔離、風控繼承（正在 build） | okx_accounts Phase 0/1 | 🚧 系統完工後寫 |
| EP12 | AI 當槓桿，不當許願池 | Meta 收尾 | AI 寫的 code 差點餵模型吃 5.5 天毒資料 | Task #4、全系列回顧 | 📝 未寫 |

## 排序邏輯

- **EP1-2 是「為什麼 + 地基」**，已完成。
- **EP3（驗證）刻意放在特徵/模型之前**：walk-forward、per-fold、OOS 這些
  概念是 EP4-7 的敘事語言，先教工具再講戰役。這也符合真實教訓順序——
  驗證方法錯了，後面做什麼都是自欺。
- **EP4-7 是研究線**（特徵 → 模型 → 訊號 → 策略），每篇的 hook 都是
  一次 NO-GO 或校準事件，主題一致：「像樣的紀律長什麼樣」。
- **EP8-10 是工程與實戰線**（執行 → 風控 → 爆倉），情緒濃度遞增，
  EP10 是全系列的高潮篇。
- **EP11-12 是擴展與 meta**，收在「系統之上的思考」。

## 與 Risk Reads 系列的分工

- EP 系列 = 自傳體，講「我建系統的過程與教訓」
- Risk Reads = 分析體，講「市場微觀結構知識」（spread/CVD/OI/funding...）
- 兩系列交錯發可避免讀者疲勞；EP 斷檔期用 Risk Reads 補位。

## 每篇的固定結構（5-beat PAS）

Pain（真實事故開場）→ Resonance（讀者也踩過）→ Curiosity（為什麼會這樣）
→ Solution + Benefit(我的修法 + 量化收益) → Next（提問收尾 + 下集預告）

詳見 docs/linkedin_risk_reads_series.md 的 style guide。
