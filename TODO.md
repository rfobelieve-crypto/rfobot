# TODO — 待處理計劃

> **跨 session 任務真相源**。Claude Code 的 session 任務清單不進 git、不跨機器——
> 凡是要「本機 / 雲端 / 下一次對話」都看得到的任務，寫這裡並 push。
> 每次開工：先讀「當前任務」區。

## 當前任務（2026-07-07 更新）

### 1. 擠壓指標 × 訂單流系統結合（策略 #2 候選）★ 最優先
流動性真空假說：壓縮後價格往阻力小的一側走，撤單領先成交洩露方向。
工具鏈已全部完成（branch `claude/general-session-HEJed`）：
- `research/squeeze_events.py` — Pine v2.4 移植的事件採樣器（結算語義已對齊，勿改內部邏輯）
- `research/squeeze_events_cli.py` — 事件表 CLI（對接既有 klines parquet）
- `research/squeeze_flow_join.py` — H1/H2/H3 假設檢驗（joins orderbook_snapshots_1m + flow_bars_1m）
- `market_data/adapters/depth_delta_collector.py` — 真撤單收集器（standalone，未接 service）

執行順序：
- [ ] Step 0-a Pine 對帳：TV 端（filterMode=OFF, confirmOnClose=ON）事件 vs
      `squeeze_events_cli.py` 輸出，時間戳+方向 100% 一致才放行
      （注意：atr_bo 欄=band=ATR×0.9；丟棄前 3×max(period) 根暖機）
- [ ] Step 0-b 無條件基線：`python research/squeeze_events_cli.py --start 2025-06-01`
- [ ] Step 1 flag 統計：`python research/squeeze_flow_join.py`（H1/H2/H3 + 前後半穩定性）
- [ ] Step 2 啟動撤單收集器累積 depth_deltas_1m（3-6 個月後跑精細版）

預先登記假設（2026-07-07，看資料前寫定）：
H1 薄側一致 → sl_first 較低；H2 撤單側 → 突破同側 >55%（CI 下緣>50%）；
H3 三旗共振子集 r_scaleout CI 下緣 >0。
紀律鎖：cell n<100 不下結論；前後半同向才算過；看資料後禁改 flag 定義。
樣本量預警：1h BTC 一年約 30-60 事件，第一輪多半「方向有趣樣本不足」→ 擴樣本
（15m / ETH），不降標準。定位：走完整 staged framework，倉位與 V7 合併計算。

### 2. OKX AI Trading Challenge（okx.ai / Agent Trade Kit 競賽）
用 OKX 開源的 Agent Trade Kit（MCP server，`github.com/okx/agent-trade-kit`，
164 tools：行情/下單/algo/帳戶）讓 Claude 直連 OKX 參賽。
格式：14 天實盤 USDT 本位永續，ROI + PNL 雙排行榜 + Skill 提交；賽季制輪辦
（已至第 8 屆，1M USDT 級獎池）。官方頁：okx.com/en-us/agent-tradekit/competition
- [ ] 查當期賽季報名期限與規則（官方頁 + OKX App 活動區）
- [ ] 本機裝 Agent Trade Kit（Node 18+，`okx config init`，API key 不出本機）
- [ ] 決定參賽策略：V7 訊號鏡像（簡化版）vs 獨立簡單策略
- [ ] **隔離規則**：競賽用獨立子帳戶 + 獨立小額資金，與 Stage 3 主帳戶完全分離；
      競賽虧損上限先定死（如 $100，輸完即止），不影響主系統任何紀律
- [ ] **IP 保護**：Skill Square 提交用包裝版/簡化版，V7 真實 edge 不公開
- 加分動機：參賽過程 = EP 系列 + LinkedIn 素材 + 履歷素材（OKX 官方賽事排名可驗證）
- 風險認知：ROI 榜首多為高槓桿樂透倉，不以奪榜為目標，以「完賽 + 正收益 + 內容產出」為目標

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
- [ ] `python research/exit_decomposition.py` — exit regret 歸因
      （先確認 5d83da2 的 exit-variants 雙 NO-GO 是否已覆蓋此問題，是則關閉）

### 5. 內容線
- [ ] EP2 英文版發 LinkedIn；EP3 細修後發（Medium 英文版已備）
- [ ] EP12 素材「AI 當槓桿不當許願池」已入 roadmap（docs/linkedin_ep_series_roadmap.md）

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

## 回測失敗（已排除，勿重跑）
- [x] ~~流動性獵取反轉~~ — 4h 週期上 IC ≈ 0
- [x] ~~K 線 delta 背離~~ — IC = 0.01
- [x] ~~consolidation_score~~ — IC ≈ 0
- [x] ~~ChessDomination 4D (CDP)~~ — 合成乘法結構稀釋信號
- [x] ~~ML exit model~~ — oracle 天花板分析 NO-GO
- [x] ~~WQ101 alphas (6)~~ — aggregate lift 被 outlier fold 撐起，per-fold 負
- [x] ~~liquidity proxy features (21)~~ — univariate IC 高但 ensemble 零提升
- [x] ~~exit-variants sweep + 不對稱 cutoff Option C~~ — 雙 NO-GO（5d83da2）
