# TODO — 待處理計劃

## 高優先（近期）

### 數據累積（進行中，預計 4/20 完成）
- [ ] 新特徵數據累積至少 2 週 (impact_asymmetry, post_absorb_breakout, bvol, direction features)
- [ ] 14 個新 CG 端點數據累積
- [ ] 監控 Magnitude IC 是否從假日效應恢復

### Direction 模型重訓（預計 4/20 後）
- [ ] 加入 impact_asymmetry + post_absorb_breakout + flow_trend_score
- [ ] 加入 abs_completion
- [ ] 加入 tox_pressure_zscore（IC=+0.077 vs direction，與現有特徵 r<0.04）
- [ ] 加入 tox_liq_exhaust（IC=-0.034 vs direction）
- [ ] Walk-forward 評估，確認 AUC 和 Top-decile 都提升才部署
- [ ] 特徵篩選：permutation importance 砍掉 < 0.5% 的噪音特徵
- [ ] 目標：AUC > 0.60, Top-decile > 0.61

### Magnitude 模型重訓（預計 4/20 後）
- [ ] 加入 tox_pressure（IC=-0.060 vs |mag|）
- [ ] 加入 tox_accum_zscore（IC=-0.058 vs |mag|）
- [ ] 加入 tox_bv_vpin_zscore（IC=-0.055 vs |mag|）
- [ ] 加入 tox_div_taker（IC=-0.050 vs |mag|）
- [ ] 注意：負 IC = 高毒性 → 波動偏小（可能是同期效應），需驗證是否為真正的預測力
- [ ] Walk-forward 評估，ICIR 不降才部署

### 特徵篩選（隨重訓一起做）
- [ ] Direction 模型：89 + 新特徵 → permutation importance 篩選 → 目標 50~60 個精銳
- [ ] Magnitude 模型：87 + 新特徵 → 砍掉底部噪音
- [ ] 砍掉後重訓，OOS 表現不降才確認
- [ ] 減少 overfit 風險

## 中優先

### 滾動重訓機制
- [ ] 設計每 2~4 週自動重訓流程
- [ ] 追蹤 IC 衰退速度，決定重訓頻率
- [ ] 自動比較新舊模型 OOS 表現，只有更好才部署

### 績效追蹤完善
- [ ] 累積足夠 Strong 信號樣本（目標 20+ 筆）再做統計結論
- [ ] 觀察 Moderate 信號 live 勝率是否維持 95%+
- [ ] 週一~週五 vs 週末假日的 IC 差異分析

### 新特徵構想（待回測驗證）
- [ ] 持倉痛苦累積：funding × 持續時間 × OI（多久沒釋放壓力）
- [ ] 參與者分歧度：CB 溢價 × BFX 保證金 × Binance taker 的一致性
- [ ] 流動性遷移：分時段 volume profile 變化率
- [ ] Funding 結算前 2h 行為異常
- [ ] 多空比加速度（二階導數）

## 低優先（長期）

### 多時間框架
- [ ] 1h 短線確認信號
- [ ] 4h + 1h 同向 = 更高信心

### Ensemble
- [ ] 3~5 個不同隨機種子模型取平均
- [ ] 降低單一模型隨機波動

### 機率校準
- [ ] Platt scaling 或 isotonic regression
- [ ] 累積 500+ bars 後做一次 calibration check

## 已完成

- [x] impact_asymmetry 特徵 (IC=-0.071, 方向模型)
- [x] post_absorb_breakout 特徵 (mag IC=0.191, 兩個模型)
- [x] flow_trend_score 特徵 (mag IC=0.156)
- [x] Magnitude 模型重訓 v2 (ICIR 1.18→1.22, Top/Bot 3.03→3.16)
- [x] 績效追蹤系統 (Rolling IC, Strong signal tracking, SHAP)
- [x] 互動圖表 (/ichart, TradingView Lightweight Charts)
- [x] Moderate + Strong 三角形顯示
- [x] 歷史 bar 全量重跑 predict()
- [x] 圖表同步規則寫入 CLAUDE.md
- [x] Order Flow Toxicity 特徵 (tox_pressure_zscore IC=+0.071, 9 個新特徵)
- [x] Parquet 自動 freshness 檢查 (shared_data.py, 訓練前 6h 閾值自動 backfill)
- [x] indicator_history 合併完整歷史 (621→4118 rows, 10/17→4/7)
- [x] 統一 backfill 腳本 (backfill_all_parquet.py, Binance + 14 CG endpoints)

## 回測失敗（已排除）
- [x] ~~流動性獵取反轉~~ — 4h 週期上 IC ≈ 0，方向無預測力
- [x] ~~K 線 delta 背離~~ — IC = 0.01，太弱
- [x] ~~consolidation_score~~ — IC ≈ 0，無效
- [x] ~~ChessDomination 4D (CDP)~~ — cdp_score IC=0.012 無效，合成乘法結構稀釋信號；cdp_x (-0.039) 和 cdp_pressure_level (+0.039) 邊際有效但本質是 price percentile
