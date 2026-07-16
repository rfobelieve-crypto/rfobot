# TODO — 待處理計劃

> **跨 session 任務真相源**。Claude Code 的 session 任務清單不進 git、不跨機器——
> 凡是要「本機 / 雲端 / 下一次對話」都看得到的任務，寫這裡並 push。
> 每次開工：先讀「當前任務」區。

## 當前任務（2026-07-16 更新）

### 0. 撤單劇本偵測器已上線（2026-07-16）
`market_data/tasks/cancel_playbook_watcher.py` 掛進 Service 2：機器前瞻記錄
撤單劇本事件（凍結定義 v1-2026-07-16：吸收/真破/真空/雙側避險）→
`cancel_playbook_events` 表 + 30/60/120m 結果自動回填 + TG 告警
（政策：vshock≥20 或 |淨|≥0.30，全域 60m cooldown，~4 則/天）。
- [ ] 累積 30+ 筆/劇本後看 hit_60m 統計（7 天 replay 煙測 ≈ 硬幣，屬預期）
- [ ] 8/10 `cancel_lead_ic`/`cancel_shock_ic` 判決日一起檢視；表現好的劇本
      才有資格升格下一個 pre-registered family
- 紀律：記錄定義凍結不准調；告警門檻屬 UX 可調；此線永不接 executor
- 2026-07-17 三階分析入口完成：① `research/cancel_flow_analyze.py`
  (table/--summary/--json) ② MCP `analyze_cancel_flow`（Claude Desktop 需
  Python≥3.10 env，本機 l30d 3.12 已裝 mcp）③ TG `/cancelanalyze [mins]`
  → 進入實盤驗證階段：事件自動累積中，等 n≥30/劇本 + 8/10 判決

### 1. 擠壓指標 × 訂單流系統結合（策略 #2 候選）★ 最優先
流動性真空假說：壓縮後價格往阻力小的一側走，撤單領先成交洩露方向。
工具鏈全部完成（已 merge 進 main）：`squeeze_events{,_cli}.py`、`squeeze_flow_join.py`、
`depth_delta_collector.py`。

執行進度（2026-07-09 session）：
- [x] Step 0-b 無條件基線：**122 事件**（sl_first 53.8%、r_scaleout +0.455、MFE/MAE 2.71/1.23R）
- [x] **Step 2 撤單收集器已上線**：`depth_delta_collector` 掛進 market_data service（commit 927b4b7 pushed），
      `depth_deltas_1m` 自 2026-07-09 起 24/7 累積（本地 smoke 通過，每分鐘 bid/ask add/cancel）。← **資料時鐘已啟動**
- [~] Step 1 flag 統計：`squeeze_flow_join` 兩個 bug 已修（`window_start` 欄名 + cp950 輸出），管線跑通；
      但 `orderbook_snapshots_1m` 深度僅回溯 2026-05-11 → 122 事件只有 **26 個**有覆蓋、全 cell n<100 → **無法下結論**（如預警）
- [ ] Step 0-a Pine 對帳（仍是正式放行門，需 TV 端比對，未做）
      （atr_bo 欄=band=ATR×0.9；丟棄前 3×max(period) 根暖機；TV filterMode=OFF, confirmOnClose=ON）

本 session 額外 powered 檢驗（taker/衍生品訂單流，非撤單）：
- horizon-IC：taker flow 無條件對 1-12h ≈ 0（1h 尺度訊號已衰減殆盡）；squeeze 條件 h2-4 微弱駝峰但 n=119 CI 全含 0
- hybrid 篩選（8 訂單流特徵 × 突破成敗）：廣撒混合體 **OOS NO-GO**（LOO-CV AUC 0.564 CI[0.457,0.671] 含 0.5）；
  唯一 univariate 冒頭 = 方向對齊 Coinbase 現貨溢價（IC +0.23，CI 邊緣離 0，多重比較存疑，可能已在 V7）；taker 系全死
- context 搜索（全 6290 bar，有統計力）：壓縮/Donchian突破/測試關卡/高波動 **無一** context 把 taker IC 拉離 0 →
  **powered NULL**，且戳破 squeeze 駝峰（很可能小樣本雜訊）
- **結論**：taker/衍生品訂單流 × 傳統指標，1h+ 視野 = 已飽和/擁擠（第 4 次驗證）。火花只可能在
  **(a) 差異化訊號 = 撤單（`depth_deltas_1m` 收集中）** 或 **(b) 次小時視野（訂單流還活著的尺度）**

下一步（取代原執行順序）：
- [ ] （被動）等 `depth_deltas_1m` 累積 3-6 個月 → 用今天的 `horizon_ic` / `context_ic` 腳本**原樣換撤單訊號**重跑（差異化訊號的第一次真正檢驗）
- [ ] Pine 對帳（Step 0-a），讓事件時間戳可信
- [ ] 把 scratchpad 的 `horizon_ic.py` / `context_ic.py` / `hybrid_screen.py` 整進 `research/` + 留「換撤單訊號」接口（目前僅在 scratchpad）
- [ ] （可選、便宜）單獨驗 spot-premium：walk-forward + conditional IC vs V7 residual，確認真火花 or 噪音/冗餘

預先登記假設（2026-07-07 定，撤單資料到位後仍用）：
H1 薄側一致 → sl_first 較低；H2 撤單側 → 突破同側 >55%（CI 下緣>50%）；H3 三旗共振 r_scaleout CI 下緣 >0。
紀律鎖：cell n<100 不下結論；前後半同向才算過；看資料後禁改 flag 定義。
樣本量預警：1h BTC 一年約 30-60 事件（含叢聚 122）→ 第一輪樣本不足，可擴 15m / ETH，不降標準。

### 2. OKX.AI Genesis 黑客松 — ⏸️ 擱置（2026-07-10 決定）
**使用者叫停原 ASP 方向**：賣訊號解讀 = 把系統 edge 分享出去（alpha 擁擠即衰退），
per-call 收費完全不對價。okx-asp repo 保留（含已完成的 Claude 解讀層 a95edae）但**不部署、不上架**。
7/17 截止前若要參賽只剩 pivot 到不洩訊號的題目；預設不參賽。
⚠️ 相關事實：主系統 `/json` endpoint 目前**完全公開**（admin guard 刻意豁免），任何人可抓
direction/tier/confidence——使用者知情後選擇**先留公開**（保留產品化彈性）；若立場改變，把 `/json`
加進 guard 前先查內部呼叫者（圖表/bot）。$80 Claude 促銷積分改留給自用（如 sentiment 異源原型）。

<details><summary>原參賽計畫（點開，僅存檔）</summary>
角色 = ASP（Agent Service Provider，賣方）：把量化系統包裝成可被呼叫/
付費的 AI Agent 服務並上架 okx.ai。核心要求「打造並成功上架一個有真實價值
的 ASP」；評審重視 Revenue Rocket（收入+聲譽累積）。
- [ ] okx.ai → Become ASP / Tutorial，今天先提交（審核需時間）
- [ ] 服務描述定稿（草稿見 Claude 對話 2026-07-07；賣「訊號+風控分析工具」
      不賣「保證獲利」，全文附 not-financial-advice 聲明）
- [ ] 上架審核通過 → 發 X 貼文 → 填 Google 表單
- [ ] （相關但獨立）Agent Trade Kit 交易賽季另計：okx.com/en-us/agent-tradekit/competition
- [ ] **隔離規則**：競賽用獨立子帳戶 + 獨立小額資金，與 Stage 3 主帳戶完全分離；
      競賽虧損上限先定死（如 $100，輸完即止），不影響主系統任何紀律
- [ ] **IP 保護**：Skill Square 提交用包裝版/簡化版，V7 真實 edge 不公開
- 加分動機：參賽過程 = EP 系列 + LinkedIn 素材 + 履歷素材（OKX 官方賽事排名可驗證）
- 風險認知：ROI 榜首多為高槓桿樂透倉，不以奪榜為目標，以「完賽 + 正收益 + 內容產出」為目標

</details>

### 3. 朋友跟單系統部署（Phase 0+1 程式碼已完成）
- [ ] Railway env：`OKX_CRED_MASTER_KEY`（Fernet key，存密碼管理器）+ `TG_ADMIN_CHAT_ID`
- [ ] 跑 `migrations/014_okx_accounts.sql`
- [ ] Merge branch → 部署兩個 service（requirements 已含 cryptography）
- [ ] Selftest：自己的第二組 API key 走 /okx_addacct → resume → 跨一筆 trade → delete
- [ ] 通過後才收朋友 key（只勾讀取+交易、不勾提幣；capital ≤ $200）
- 注意：pause 時未平倉位停止管理（OKX stop algo 仍在）——pause 前確認無持倉

### 4. 研究腳本待跑（皆需資料層）
- [ ] `python research/poc_sweep_study.py` — POC × 流動性獵取（手動交易輔助），
      跑完把 Section 1+2 給 Claude 判讀
- [x] `python research/exit_decomposition.py` — **2026-07-10 已跑**：opp_signal 出場是
      利潤引擎（86% WR / +152bps / 多抱反虧 60-75bps → 不可動）；trail 平均 regret≈0，
      只有最差 1/4 漏 70-130bps（其中 9/13 是 Moderate 進場——Strong-only live 大半不存在）。
      結論：出場規則不改；剩餘槓桿 = 條件性重進場（下行）
- [ ] **flow 重進場**（pre-registered 2026-07-10，`research/flow_reentry_bt.py`，規則已凍結）：
      trail 掃出後 60min 內 cancel_skew 與 imbalance 都仍偏原方向 → 下根 bar 重進場。
      停損完整不動，只決定「要不要買回票」。Gate：n≥30 + CI 下緣>0 + 前後半同號，
      預計 **~2026-10**（backtest 節奏 ~8-9 trail 出場/月）。跑前需刷新 klines parquet
      （事件驅動：daily_collect.bat 只管 coinglass/ 子夾；root parquet 靠訓練/月度復驗刷）

### 4.6 V7 多幣化可行性研究（2026-07-15 啟動）
動機：核心瓶頸是樣本速度（~265 Strong/年、73% 被單槽擠掉），多幣化 = 同機制
樣本產出 ×N，非找新 alpha。純 research track（`research/multicoin/`），不碰生產。
- [x] **Step 1 資料審計完成**（見 `research/multicoin/audit_results.md`）：
      ETH 特徵覆蓋 ~97%（只缺 coinbase premium 家族 3 個 + etf_aum）、SOL ~92%
      （再缺 DVOL 家族）。資料層不是瓶頸。陷阱：`/coinbase-premium-index` 無視
      symbol 參數（三幣回同值）→ 新增幣種端點必跑值差異化檢查（`verify_value_diff.py`）
- [ ] Step 2 ETH 移植實驗：backfill ETH 歷史（13 端點）→ 建 ETH 特徵表 →
      同一套乾淨 WF（purge+embargo、無 early-stop 洩漏），對照 BTC clean AUC 0.5412
- [ ] Step 3 訊號重合率：ETH Strong vs BTC Strong 時間對齊統計
- **Go/No-Go（預先登記）**：ETH clean AUC ≥ ~0.54 且重合率 <50% → 繼續（考慮 SOL、
  談 production 化）；任一不過 → 多幣化對 V7 無性價比，資源回異源資料線。
- 紀律：BTC Gate A 乾淨版仍未過門檻（57.6%/CI 下緣 51.5%）——多幣化是「乘以 N」，
  乘的對象要先證明；production 化討論必須在 Gate A 重跑通過之後。

### 4.5 基建完善（2026-07-10 審視——防「新基建靜默失敗」族）
- [x] **depth_delta_collector freshness 監控**（2026-07-10 完成，a23d174）：APScheduler
      每 30 分查 `depth_deltas_1m` 最新分鐘，落後 >120min → TG 告警（停更+恢復各一次）
- [x] **depth_deltas_1m 每日 parquet 匯出**（2026-07-10 完成）：export_depth_deltas.py
      掛進 daily_collect.bat（注意 bat 被 gitignore＝本機檔，改動已生效於明日 04:00）。
      首跑備份 1837 行，收集器 31h 近零斷點
- [ ] Drawdown governor（先 alert-only）：回撤加深自動縮 size 的觸發邏輯，
      對症 M2M MDD -21% 破 Stage3→4a 門檻
- 📅 研究判決日曆：8/5 月度復驗（自動）→ **8/10 撤單 bar 級領先性**（cancel_lead_ic,
      n≥40k）→ **8月中 shadow maker 裁決**（live n≥30）→ **~10月 flow 重進場 gate**（n≥30）
- [x] **perp 撤單收集器部署**（2026-07-15 完成，168e8b2 pushed）：depth_delta_collector
      參數化 + start_all.py 平行起 binance_perp 實例（fstream，同表 exchange 區分），
      本機 smoke 通過（perp 撤單量 ~8x 現貨）。現貨流不動（凍結檢定的序列），
      五支分析腳本已加 exchange='binance' 過濾防污染。DB 已確認 binance_perp
      自 2026-07-15 09:06 UTC 起累積、近零斷點 → **perp 40k 判決點 ≈ 2026-08-12**
- 撤單檢定家族現況（皆凍結）：F1 skew 水位（07-10 註冊）8/10 判決；
  F2 變化幅度 shock（07-15 註冊，smoke: intensity→|ret| 5m IC +0.113 CI 全正，待 powered）；
  F3 深帶翻轉事件（07-15 註冊，6 天 0 事件，需 ≥30/方向）；
  主觀判讀日誌 research/results/eyeball_log.md（30 筆判決，前瞻記錄才算）
- 凍結假說（2026-07-11 註冊，看 depth 資料前寫定；皆隨撤單檢驗點跑）：
  - **P-cascade**：撤單強度（total_cancel/total_add）top-decile 分鐘 → 之後 15min 內
    出現巨量（vol top-1%）機率顯著高於基準（CI 離 0 + 前後半同號）。
    背景：巨量分鐘 92% 含清算 / 重清算濃縮 35x = 被迫流實錘；但事後 fade 無 edge
    （60min 延續率 51/48% = 硬幣）→ 錢只可能在瀑布**之前**的撤單裡
  - **P-wall-pull**：逼近 24h 極值（±10bps）事件中，逼近前 15min 該側撤單不對稱
    （tercile 分組）預測 60min 內掃穿。背景：靜態牆厚已測死（薄/中/厚 = 65/66/62%,
    z=0.45）——可撤回的宣告無資訊、撤離動作才有；無條件掃穿基率 ~2/3
    （「止損區是磁鐵」成立但人人皆知，非 edge）

### 5. 內容線
- [ ] EP2 英文版發 LinkedIn；EP3 細修後發（Medium 英文版已備）
- [ ] EP12 素材「AI 當槓桿不當許願池」已入 roadmap（docs/linkedin_ep_series_roadmap.md）

### 6. ★ 徹底掌握自己的系統（面試就緒度 — 求職關鍵）
「系統會跑但講不清 why」是最大面試風險（很多是 AI 寫的）。面試官不問「還能做什麼」，
問「這個為什麼這樣做」。詳細手冊：docs/system_mastery.md（+ Claude 生成的
rfobot_系統精通手冊.docx 含完整標準答案）。逐層攻破，每層要能不看筆記回答灵魂拷問：
- [ ] 層 1 數據層（notional 換算 / 1e12 時間戳 / share-data-not-code）
- [ ] 層 2 特徵層（look-ahead 防範 / 加特徵流程 / 為何不用 MACD）
- [ ] 層 3 模型層（雙 regressor / walk-forward purge+embargo / IC 0.063 / in-sample 陷阱）
- [ ] 層 4 訊號層（rolling percentile / Strong-only / Gate A 算法）
- [ ] 層 5 執行層（5 種 reconcile / 狀態機四態 / 3 個 kill trigger / fail-closed / 不用 CCXT）
- [ ] 層 6 風控數學（2.0x 怎麼算 / 何時放寬 / hit kill 降階重驗）
- [ ] mistake.md 六個失敗故事能講成 30 秒
- [ ] 回 Claude 做面試官壓力測試（跨層追問）
自測進度 2026-07-07：層 1 Q1 60分（漏 notional）、Q2/Q3 知識債 → 已補入手冊

---

## 已歸檔（2026-04 舊計畫，多數已被後續工作取代）

<details>
<summary>2026-04 重訓計畫與特徵構想（點開）</summary>

### 數據累積（已完成）
- [x] 新特徵數據累積 (impact_asymmetry, post_absorb_breakout, bvol)
- [x] 14 個新 CG 端點數據累積

### Direction / Magnitude 重訓（已由後續多輪重訓取代）
- 原計畫：liq_frag + absorb 99 特徵、AUC>0.60 部署門檻
- 後續實際：見 mistake.md 2026-06-01/02 的 ensemble A/B 紀律與 4 條部署門檻

### 滾動重訓機制
- [ ] 每 2-4 週自動重訓 + 新舊 OOS 比較（仍有效，低優先）

### 新特徵構想（待回測驗證）
- [ ] 持倉痛苦累積：funding × 持續時間 × OI
- [ ] 參與者分歧度：CB 溢價 × BFX 保證金 × Binance taker 一致性
- [ ] Funding 結算前 2h 行為異常
- [ ] 多空比加速度（二階導數）

### 多時間框架 / Ensemble（長期）
- [ ] 1h 短線確認 + 4h 同向；3-5 seed ensemble

</details>

## 已完成（歷史）
- [x] impact_asymmetry 特徵 (IC=-0.071, 方向模型)
- [x] post_absorb_breakout 特徵 (mag IC=0.191, 兩個模型)
- [x] flow_trend_score 特徵 (mag IC=0.156)
- [x] Magnitude 模型重訓 v2 (ICIR 1.18→1.22, Top/Bot 3.03→3.16)
- [x] 績效追蹤系統 (Rolling IC, Strong signal tracking, SHAP)
- [x] 互動圖表 (/ichart, TradingView Lightweight Charts)
- [x] Order Flow Toxicity 特徵 (tox_pressure_zscore IC=+0.071)
- [x] 統一 backfill 腳本 (backfill_all_parquet.py)
- [x] Gate A 通過（2026-06-10：Strong n=739, WR 59.5%, CI [56.0, 63.2]）
- [x] 3-WR 透明化 /okx-perf + Gate B 進度卡
- [x] 多帳戶跟單 Phase 0+1 程式碼（accounts registry + executor fan-out）
- [x] admin/OKX routes token guard 上線（`ADMIN_HEAL_TOKEN` 兩 service 設定 + fail-closed guard，2026-07-09 pushed）
- [x] branch `claude/general-session-HEJed` merge 進 main + push（本 TODO 即出自此 merge）
- [x] 撤單收集器 `depth_delta_collector` 上線 Service 2（2026-07-09，`depth_deltas_1m` 開始累積）

## 回測失敗（已排除，勿重跑）
- [x] ~~流動性獵取反轉~~ — 4h 週期上 IC ≈ 0
- [x] ~~K 線 delta 背離~~ — IC = 0.01
- [x] ~~consolidation_score~~ — IC ≈ 0
- [x] ~~ChessDomination 4D (CDP)~~ — 合成乘法結構稀釋信號
- [x] ~~ML exit model~~ — oracle 天花板分析 NO-GO
- [x] ~~WQ101 alphas (6)~~ — aggregate lift 被 outlier fold 撐起，per-fold 負
- [x] ~~liquidity proxy features (21)~~ — univariate IC 高但 ensemble 零提升
- [x] ~~exit-variants sweep + 不對稱 cutoff Option C~~ — 雙 NO-GO（5d83da2）
- [x] ~~taker/衍生品訂單流 × 傳統指標 context~~ — powered NULL（2026-07-09；壓縮/突破/關卡/高波動 4 context 無一把 taker IC 拉離 0；1h+ 已飽和；火花改押撤單訊號 or 次小時視野）
