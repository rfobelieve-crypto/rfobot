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

### 4.65 次小時（分鐘級）系統可行性（2026-07-18 Phase 0 已判決：PARK）
使用者假說：1h 取樣把訂單流資訊平均掉，分鐘取樣可在衰退前捕捉 → 值得整套
搬到分鐘級。**Phase 0 依凍結 pre-registration 執行**（`research/subhourly/PREREG.md`
先 commit 固定時間戳 dd0558e → 才跑 `minute_ic_scan.py`，結果
`research/results/subhourly_ic_scan.csv`）：
- **G1 資訊存在：PASS ✅（假說的「資訊」半邊被證實）**——ohlcv_1m 15 個月
  （2025-01→2026-03，654k 分鐘）+ flow_bars 90 天兩時代交叉：taker 失衡/淨流 z/
  短期動量在 h=15-30m 月均 IC −0.03~−0.04、**15/15 個月同號**、前後半同號。
  IC×horizon 衰退曲線形狀完全符合假說：峰在 15-30m、到 4h 衰成 −0.013
  （= V7 在 4h 看不到它的原因）。註：**符號是負的**——分鐘級是「反著做」
  （fade taker flow / 短期均值回歸），不是跟單流；vshock 方向 IC≈0（與 F2
  「強度預測幅度不預測方向」一致）；obi 是唯一順向訊號（h=5m +0.034，30m 前死）
- **G2 經濟可行：FAIL ❌（假說的「可交易」半邊被否決）**——top-5% 條件桶
  在 30/60m **全部 138 格淨值為負**（扣 8bps=2×maker）；最好的一格
  （Era B ret_60@60m，且是純價格反轉非訂單流）毛利僅 ~6.2bps < 8bps 門檻，
  還有 138 格多重比較加持。IC 0.03-0.04 的單筆可轉化毛利就是付不起費率——
  市場把分鐘級可預測性磨到剛好低於成本前緣
- **處置（照凍結規則，不凹）**：G2 FAIL → 不跑 G3、**不重寫系統**、計畫
  PARK。復活條件：`depth_deltas_1m` 累積 ≥3-6 個月後帶撤單特徵 **re-run
  一次**（同門檻）；另 8/10 F1 判決是撤單側的獨立分鐘級檢定。
  V7 全程零改動（Phase 0 從頭到尾沒碰生產）
- **07-18 補 2：V7 換 15m bar 的重做成本已判**——direction 模型 importance
  拆解：**67.7% 的權重在 1h+/日頻原生源**（Coinglass OI/LS/margin 家族，
  top-10 佔 8 席；15m 網格上只是 ffill 階梯函數且無 15m 歷史可 backfill），
  fast 半邊（32.3%）正是 Phase 0 判弱的家族 → re-base = 重做全部工程去
  搬進更弱的資訊 regime，維持 NO-GO；「15m 輕量實驗」= G3，已被 G2 FAIL 封
- **07-18 補：re-run 必要贏面已量化**（`required_bar.py` + PREREG 附錄 A，
  家族在看資料前宣告）：撤單特徵須在 30-60m top-5% 桶交出 **≥8bps 毛捕捉
  ≈ 條件勝率 57.5%+**（最佳 taker cell 55.8%、真流系 cell 只有 53%）。
  兩個結構洞見進 re-run 分析義務：①流極端發生在安靜盤（條件 |move| 反而
  縮小）②低量分鐘就算勝率高也付不起 8bps → 撤單唯一過關路徑 =「安靜分鐘
  訊號預測隨後擴張波」（squeeze 論的分鐘級版本）。re-run 最早 **2026-10-09**；
  家族=撤單 def v1 聚合+簿型既有欄（wall/spread/imb）+taker 對照，僅此一次

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
- [x] **淨偏斜基線/假象檢查（2026-07-17 登記，同日判決：B 真結構偏差，F1 無污染）**：
      `research/netskew_baseline_check.py`（spot 8d n=11,739 分鐘 / perp 2d n=3,031）：
      ①小時 tercile 分組綠佔比 漲 54.7% / 橫盤 53.3% / 跌 54.1%——**無梯度、橫盤也綠**
      → 非價格遷移假象；分鐘級 spearman(net_raw, ret_1m)=−0.075（fills-as-cancels
      機制若主導，上漲吃 ask 應推「正」相關，實測反向且極弱=非主導）；
      ②同分鐘對照：perp 幾乎中性（綠 52.0%、均值 +0.0007）vs 現貨綠 55.4%、
      均值 +0.0331（~47x）——**同一套收集碼跑 perp 得中性 = 收集邏輯無方向性偏差**，
      綠基線是「現貨簿本身」的結構性質（B）。兩個前提修正：
      (a) 收集器用 diff 全簿流（depth_delta_collector.py），無「可見窗」可遷移——
      原 A 假設的窗口機制不存在；(b) watcher skew15/net15 **本來就做 trailing-60m
      去均值**（cancel_playbook_watcher.py:161,166），觸發門檻非原始值——原④條
      「觸發率天生不對稱」疑慮不成立（偏差在 60m 尺度穩定即被吸收）。
      **F1 8/10 判決可照常進行**（spearman 對常數位移不敏感、無窗口假相關向量）。
      殘留 UX 債（可選、低優先）：覆盤圖淨偏斜面板畫 raw，長期綠 = 現貨簿基線
      而非訊號——可加 60m 去均值序列或基線參考線輔助判讀
- [~] **撤單狀態機判斷器（2026-07-17 登記，依賴上一條先完成〔07-17 已判 B 解鎖〕）**：
      **A1 核心已完成（07-17）**：`classify_state()` 進 `cancel_playbook_watcher.py`
      ——凍結 v1 分類器的 1:1 重標籤（零新門檻；15 測試含 fuzz 驗證與
      classify_minute 零漂移），六態 + gate_only 殘餘誠實顯示為「爆量未定」⚪；
      TG `/cancelstate [mins]` 上線（狀態行+特徵值+回看窗分布+持續時間，
      走 `/research/cancel-analyze?mode=state` 同管線）。
      **A2 完成（07-18）**：①互動覆盤圖價格 pane 底部 6% overlay 狀態色格
      （blank=平靜/灰=換防·爆量·瀑布/🟢🔴=真空·吸收，色值單一來源
      `state_color()` in watcher）②獵取標記層（tv_alert_events+liquidity_events
      → ⚡ marker+被掃價位黃虛線，空表安全）③sweep 特寫圖：poller 對帶
      liquidity_side 的快訊渲染 plotly+kaleido 回看窗 K 線+狀態格+價位線
      PNG 作卡片附圖（deps 缺→優雅退文字卡；marketdata image 兩者都有）。
      **A3 完成（07-19）**：①四鍵按鈕即日誌——watcher 劇本告警 + TV 事件卡都帶
      `[🟢同意][🔴相反][⏸不確定][✗略過]`（callback `ceb|src|id|verdict` →
      Service 1 webhook → `cancel_eyeball_log`，INSERT IGNORE **首判鎖定不可
      事後改**、skip 不落表、60/120m 判定窗自動回填=人機同尺）②sweep 二段式：
      第 1 段掃穿瞬間只報事實+特寫圖無按鈕；強度回落（3-15min，15 強制）後
      第 2 段卡帶 **H-R 凍結雙旗標**（被掃側淨回填+對側淨撤離）結論+四鍵。
      32 測試綠+e2e 煙測（二段流程/首判鎖定/真實旗標卡）。
      eyeball_log.md 手填流程可退役——按鈕版取代。↓ 原 spec：把 shock/毛/淨/
      量比/taker 融合成單一「當前狀態」輸出——**分類狀態機不是加權分數**
      （CDP 墓碑：合成分數稀釋信號 + 權重可調 = overfit）。六態：平靜/換防警戒
      （毛高淨零）/瀑布中（強度尖峰）/向上真空/向下真空/吸收（量大守住）。
      門檻全部沿用 def v1 已凍結值，不新調參數。顯示三出口共用同一狀態函數：
      TG /cancelstate 一行 + 告警照舊 + **狀態直接畫在 K 線面板**（2026-07-17 使用者
      定版，取代獨立緞帶）——每根 K 腳下貼狀態色格，同座標系疊「流動性獵取標記」：
      讀 `liquidity_events`（read-only 跨表）畫被掃價位虛線 + 掃穿 K 標 ⚡ + 當分鐘
      狀態註記 → 「獵取當下撤單長什麼樣」一眼可讀。
      加碼（零件全現成，純組裝）：sweep 事件觸發時自動渲染 ±90min 放大圖
      （K線+狀態色+被掃價位線）推 TG——獵取事件的撤單特寫自動送達。
      判讀卡從三步簡化為「看 K 線腳下的顏色」
      **UX 定案（2026-07-17 使用者拍板）——「事件卡 + 按鈕即日誌」**：
      - 卡片格式：結論在標題（機器判讀+推理一行）→ 特寫圖佐證 → 四鍵
        `[🟢同意] [🔴相反] [⏸不確定] [✗略過不記]`——前三鍵判讀並自動落表
        （半自動化 eyeball log：人按的仍是人的判讀，落表動作自動化=強制當下記
        +天然前瞻+不可事後改，比手填 md 更符合凍結規則），略過鍵不佔樣本。
        inline button 複用 manual approval 基礎設施
      - **事件源 ×3**：watcher 劇本 / 主 bot sweep 偵測 / **TV 快訊 webhook**
        （使用者在 TV 畫關卡設快訊 → 主 bot 現成 webhook 端點收 → 只寫
        `tv_alert_events` 表（DB 當匯流排，share data not code，兩服務不互 import）
        → Service 2 輪詢 → 算狀態+渲染卡）。TV 源注入機器看不見的大級別位置感
      - **判讀窗明寫在卡上（時間區段是關鍵）**：回看窗（觸發前 90min 狀態序列，
        TV message JSON 可帶 {"window": N} 自訂）+ 判定窗（def v1 的 2h，到期
        自動回填）。TV 觸發事件與 watcher 事件共用同一回填管線 → 未來可直接
        統計「人圈的位置 vs 機器抓的位置」命中率差（= 人的位置感 alpha 檢定）
      - 顏色語言全系統唯一一套：🟢向上真空 / 🔴向下真空 / ⚪換防或瀑布（不判方向）
      - **sweep 事件卡改二段式（2026-07-17 使用者釐清：其 setup = 獵取後的假突破/
        假跌破反轉）**：掃穿瞬間 = 資訊黑洞（撤單全是保護性雜訊），第一段卡只報
        「⚡掃穿+瀑布中,判讀待塵埃落定」；強度回落後（3-15min）自動觸發第二段卡
        給結論（反轉條件成立/未現）+ 四鍵——使用者的決策時刻在第二段
- [x] **MCP agent 遠端化（2026-07-17 登記；07-18 全部完成並上線）**：
      `indicator/agent/http_server.py` streamable-HTTP + AGENT_MCP_TOKEN
      路徑認證（fail-closed：無 token 拒啟動、錯路徑 404、421 rebinding
      修正 708a7c2）。Railway `agent-mcp` service 用 CLI 全自動建立
      （Dockerfile.agent image、repo 綁定 main 自動部署、MYSQL 內部連線）：
      `https://agent-mcp-production-46d7.up.railway.app/<token>/mcp`
      公網 initialize 200 實測通過。**剩使用者一步**：claude.ai →
      Settings → Connectors → Add custom connector 貼上完整 URL。
      Railway CLI 已裝並授權（rfobelieve1@gmail.com）＝未來 Railway
      操作可全代管。原 spec：現況 stdio 版只有本機 Claude Desktop 能用，雲端
      Claude Code / 手機 app 都摸不到。改法：`indicator/agent/server.py` 的
      FastMCP 換 HTTP transport（framework 原生支援）+ token 認證（比照
      ADMIN_HEAL_TOKEN 模式，fail-closed）+ 掛 Railway（可與指標 service 同容器
      不同 port，或獨立 service）→ claude.ai Connectors 加入 → 所有 Claude
      介面可直接呼叫 cancel-flow 分析工具（fc44106 的 MCP 工具已存在，
      只差 transport）。邊界不變：agent-boundary.md 全部照舊（read-only、
      不碰 executor、AST 測試守著）。副作用：MCP agent 從 demo 變成
      「作者本人日用的生產工具」——面試/履歷素材直接升級。
      過渡期替代：TG `/cancelanalyze 90` 輸出貼給 Claude 人工判讀
- [ ] **H-R 獵取反轉濾網（2026-07-17 登記，使用者主 setup 的專屬檢定）**：
      ⚠️ **07-17 樣本審計：`liquidity_events` 與 `sweep_outcomes` 全表 0 筆**
      （事件管線從未產出資料——TV 快訊 → `/tv` webhook 這條源頭目前是死的）。
      H-R 不是「寫好 harness 等資料」而是**先復活事件源才會開始累積**；
      復活 = A3 的 TV webhook 事件源（使用者在 TV 畫關卡設快訊）——
      這使 A3 的 TV 源部分從「UX 加值」升級為「H-R 的硬前置」。
      ✅ **07-17 接收端已上線**：`/tv` 加寫 `tv_alert_events`（BTC+secret 驗證後、
      liquidity_side gate 前——純關卡快訊無 side 也收；`{"window":N}` 自訂回看）
      + Service 2 `tv_alert_poller`（觸發分鐘收盤後算狀態→推簡版事件卡→
      30/60/120m 判定窗回填，與機器事件同管線=人機命中率可比）。
      **剩使用者動作：在 TV 畫關卡設快訊指到 /tv**（帶 secret/price/event；
      含 liquidity_side buy/sell 的同時餵舊 liquidity_events 管線=H-R 時鐘）。
      原凍結假說↓
      基率警告：掃穿延續 ~2/3 → **盲目接反轉 = -EV**，edge 全在濾網。
      凍結假說：sweep 事件塵埃落定段（強度尖峰回落後 15min 內）雙旗標
      **(a) 被掃側淨回填**（cancel−add 轉負）**+ (b) 對側淨撤離**（偏斜翻轉到
      逆掃穿方向）同時成立的子集 → 60-120min 反轉率（收回被掃關卡）顯著高於
      無旗標 sweep 基準（~1/3）。兩個 categorical 旗標、不調參；
      事件源 `liquidity_events`/`sweep_outcomes` × `depth_deltas_1m` join；
      紀律鎖照舊：cell n<100 不下結論、前後半同向、看資料後禁改旗標定義。
      此檢定 = 事件卡第二段「反轉條件成立/未現」判讀的統計地基
- [ ] **F1b 條件版領先性（2026-07-17 登記，F1 過關後的第一個追問；
      2026-07-20 分析腳本補上）**：`research/cancel_lead_ic_tercile.py` 已寫
      （按已凍結 vshock 定義三分位分層跑 cancel_lead_ic，不新造任何門檻）。
      SMOKE 結果（n=15,823，遠低於 40,000 判決線，**不算數**）：三桶都沒有
      CI 乾淨脫離 0 的格子；low-vol h=30 IC+0.104 halves 同號但 CI 含 0；
      mid-vol 反而全部同號偏負（h=5~60 都約 -0.05~-0.12，halves 全部同號，
      唯獨 CI 都含 0）——**跟預先登記的「low-vol 有訊號、high-vol 趨零」
      形狀不符**，mid-vol 意外地方向最一致。n 太小，任何桶的形狀都可能是
      雜訊；不得以此調整假說或提前判決，等 40,000 分鐘（spot 現 15,995，
      ~8/10 前後）才是真正的判決點。動工緣由：使用者 07-20 指出「撤單流
      要有自己獨立於獵取事件的訊號判斷」——這正是 F1b 要回答的問題（v1
      六態全部需要 shock 或 skew/net 門檻，結構上排除了 F1b 假說要抓的
      安靜盤微弱漂移，見今早 07:44-07:57 案例：net15 峰值 0.178 未達真空
      門檻 0.30，零事件記錄）。若 F1b 通過，def v2 才有資格納入無鬧鐘門檻
      的「安靜盤傾向」狀態；在那之前，任何「撤單流自己判斷方向」的即時
      輸出都不存在、也不該存在。
      核心假說「撤單流的資訊價值與成交活動成反比」——市場兩條資訊通道，
      成交火熱時價格由吃單驅動（撤單=保護性雜訊+冗餘，例：07-17 三則真破告警
      量 30x/taker −50% 而淨偏斜≈0）；成交冷清時報價通道是唯一在動的東西
      （做市商挪動=先行洩露）。**預先登記的預測：撤單 IC 在低量分鐘顯著、
      高量分鐘趨近零**（按成交量 tercile 分層跑 cancel_lead_ic）。
      推論的應用排序（機制推理，非驗證結論）：①壓縮期方向洩露（策略 #2 主場，
      成交越安靜撤單越大聲）②獵取的逼近段+收尾段（避開掃穿瞬間——那一刻
      資訊權在量價，撤單退場）③瀑布當下禁用。若 F1b 成立，判斷器的狀態
      權重應在低量時段調高撤單維度、高量時段讓位給量價（分工寫進 def v2）
- [x] **事件卡按鈕改行動鍵 + 自動對答案（2026-07-20 完成）**：四鍵判讀退役
      ——使用者點破統計無效（卡片先給機器結論=錨定污染、樣本慢、選擇偏誤，
      量到的是服從率非 alpha）。改成：①按鈕=給工具 `[📊特寫圖][🔍90m深入]
      [⭐收藏][✗忽略]`（callback `cfa|src|id|action`，ceb| 處理器留給歷史卡）
      ②判定窗回填完成自動 reply 原卡「對答案」（60/120m 走勢+命中+劇本近
      30 筆滾動命中率；tv 事件報走勢）——告警流變自評分 feed，8/10 前就對
      「哪個劇本有料」長體感。「人的位置感」檢定不受影響：靠 TV 關卡被動
      回填（無按鈕、無錨定），照舊累積。star 落 cancel_eyeball_log
      verdict='star'（收藏語義，回填照舊，供未來覆盤包）
- [x] **撤單流指令全遷新 bot（2026-07-19 登記，同日完成 f85400e）**：
      新 bot /start 出選單（覆盤圖/狀態/摘要按鈕）+ 三指令帶參數，回覆全走
      CANCEL_API；主 bot 移除三指令/help 文字/按鈕/callback。頻道分流非
      系統分家：handler 共用、edge 證明後撤單流仍進 V7 判斷層（overlay）
- [ ] **撤單流多幣化（2026-07-19 登記）**：與 V7 多幣化（4.6）同方向——
      depth_delta_collector 已參數化，起 ETH spot/perp 實例即開資料時鐘
      （越早越好，10 月 re-run 與 F1 家族未來可跨幣）；watcher/圖表後續跟進。
      紀律：BTC 凍結序列不動，ETH 為新增平行序列
- [x] **first-hit-wins 平行診斷（2026-07-21，FROZEN v1）**：使用者質疑
      固定時間點(60m/120m)判對錯會混淆「方向對但慢」跟「真的錯」→
      `first_hit_verdict()`：逐分鐘掃，先碰目標價(±0.5%沿用
      outcome_tracker.py既有門檻)算對、先碰反向價算錯，120m窗沿用既有
      窗口。與固定時間點指標並行顯示，不取代(F1/F1b判決仍用固定時間點，
      因為那才是防p-hacking的量法)。回填放寬 SELECT 讓舊事件自動追平。
      **實測結果推翻直覺**：21:47-21:53 那組 120m 翻正 +1.0~1.5% 的案例，
      用 first-hit-wins 重查後**除 21:53 外全部仍是 miss**——price 在反彈
      之前先碰到 -0.5% 反向門檻，若真設停損會先出場；120m 翻正是巧合
      時間點快照，不是訊號提早判對只是被切錯位置。58 測試綠。
- [x] **色格短窗誠實檢查（2026-07-20，探索性非家族）**：使用者主觀回報
      「看綠後面漲、看紅後面跌，感覺有利可圖」→ `research/color_state_forward_check.py`
      用既有凍結 `classify_state` 零新定義跑全歷史（n=15,979 分鐘）。結果與
      直覺**相反**：綠 h=1/3m 平均 **-6~-7bps**（方向錯）、紅 h=1~60m 全部
      方向錯或低於 8bps 成本線；沒有一格通過。且「綠」100% 是吸收態
      （vacuum 從未觸發過一次）。n=32/54 太小 + 12 組無多重比較校正，
      **不當結論用，不翻轉色格邏輯**——避免重演 2026-06-20 FOMO 的鏡像版
      （這次是「發現乾淨反訊號」而非「發現漂亮正訊號」，陷阱同大）。
      跟 subhourly Phase 0 G2 同一個死法：資訊或許存在，幅度到不了成本線。
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
- [x] **EP3「你的回測正在騙你」中英文皆完成（2026-07-20）**：原稿太術語
      密（IC/fold/bootstrap/aggregate 堆疊），使用者反饋改成一般讀者能懂
      的白話 + 比喻版（老師發答案卷／量錯尺／班級平均被學霸拉高），砍掉
      4 條門檻條列式清單、拿掉結尾自介。三個陷阱骨架與 mistake 2026-04-13、
      2026-06-02 素材不變。中文
      `Desktop\linkedin_posts\quant_ep3_backtest-lying_v3_headings.docx`
      （帶分段粗體小標題），英文改成 Medium 格式（去掉 LinkedIn 專用的
      「Copy-paste ready post」框、hashtag 換成 Medium 站方標籤上限 5 個、
      加 Cover image 提示）：
      `Desktop\linkedin_posts\quant_ep3_backtest-lying_medium.docx`。
      下一步：使用者最終審閱後可發布
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

### 7. 產品化網站——3D/互動風格參考（2026-07-21 登記，同日補正）
使用者想把系統未來做成產品網站。原先參考的 IG reel 靜態截圖只看到模糊
漸層，經使用者上傳實機螢幕錄影（jerrythewebdev,「Websites after leaving
ChatGPT for Claude」）確認真正的視覺語言，跟第一版描述（互動漸層）差
很多，以錄影為準：
- **深色電影感底**，極簡放大無襯線標題字
- **玻璃質感 3D 浮動面板** + 全息/虹彩材質的有機造型 3D 物件飄浮在暗色空間
- 破碎鏡面/玻璃碎片拼接面板，文字嵌在碎片裡，搭配潑彩煙霧色塊（藍紫粉）
  點綴
- 捲動觸發場景轉場（技術棧常見 React + Tailwind + Three.js + GSAP +
  WebGL），配 hashtag #opus #fable5 #mythos（用 Claude 系列模型輔助製作）
- [ ] 定位：視覺風格參考（深色、玻璃質感 3D、捲動轉場），不是逐項規格；
      跟 `project_productization_goal`（長期商業化目標，現階段先顧穩定+
      訊號品質，不做過早的 multi-tenant/auth/billing 抽象）放一起看
- [ ] 待評估：這個風格值不值得投入（Three.js/WebGL 對求職 demo 或未來
      產品頁的 ROI），還是先用更輕量的方案（如既有 dashboard/Artifact）
      驗證內容本身，之後才考慮視覺升級

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
