# Exit Model 研究路線(2026-06-08 session handover)

> 用途:當前 session 在手機上討論完 exit 改善方向,回家用本機接著做時的 handover。
> 不是 spec,是「下次坐到電腦前,5 分鐘內可以接上場」的入口。

---

## TL;DR(回家先看這段)

```
今天的結論:V7 exit 邏輯可能有 chop tax 問題,但「是否值得做 ML exit model」未證實。
下一步(本機跑):跑 oracle analysis 看 gap 大不大,大才值得投入 1-2 個月做 ML model。
```

回家第一件事:看 §下次本機 setup 清單,然後跑 §第一步 oracle analysis。

---

## 今天為何走到這個方向

### 訊號層數據(從 `/perf` Dashboard)
- WF-OOS baseline:Strong 69.2% WR
- Live(2026-04-17~):Strong 57% WR(↓ 12.2pp)
- Strong DOWN:57% WR / avg **-0.10%** ← 高 WR 但 P/L 負
- Moderate DOWN:65% WR / avg **-0.17%** ← 更明顯
- Moderate UP:38% WR / avg **-0.11%** ← 兩個都輸

### Alpha Decay Monitor(整體 🔴 危險)
- IC 30d=+0.052, 7d=-0.086(反向中)
- 信心-勝率 倒掛(高信心 55% vs 低信心 68%,-13.6pp)
- 特徵漂移 Top-10 overlap 3/10
- 訊號 churn 48h=40.4%

### 推論
高 WR 但平均負 P/L = **贏的小、輸的大**。這是典型的:
- Trail stop 給回太多(高點到 stop 給回 3×ATR)
- opp_signal 在 chop 噪音裡頻繁觸發 → 多 round-trip cost

但這只是**推論**,沒有實證。**Phase 0 oracle analysis 就是要實證**。

---

## 我已經推上 main 的工具(回家直接能用)

| 檔案 | 用途 | Commit |
|---|---|---|
| `research/exit_variants_backtest.py` | 跑 7 個 exit 變體 A/B 比 baseline | `5784f15` |
| `indicator/app.py` `/research/exit-variants` endpoint | 上面那個 backtest 的 HTTP 包裝(行動裝置用) | `3805954` |

### 7 個 variants
```
0. baseline       — 3xATR trail / 72h / opp=any
1. opp_strong     — opp 只在 Strong 反向才觸發
2. opp_2bar       — opp 需連 2 bar 反向才觸發
3. trail_2x       — 緊 trail
4. trail_4x       — 鬆 trail
5. no_opp         — 移除 opp_signal
6. be_after_1atr  — 浮盈 ≥ 1xATR 後 stop 移到 entry
```

---

## 下次本機 setup 清單

```bash
# 1. clone / pull 最新
cd ~/path/to/rfobot
git pull origin main

# 2. 確認 parquet cache 存在(backtest 需要這個)
ls research/dual_model/.cache/features_all.parquet

# 沒有的話先跑 backfill(會抓 Coinglass + Binance 重建,約 5-15 分鐘):
python -m research.backfill_all_parquet

# 3. confirm python env
python -c "import pandas, numpy, xgboost; print('OK')"
```

---

## 第一步:跑 exit_variants_backtest(15 分鐘工作)

回家先跑這個,看數字直觀有沒有 variant 明顯打 baseline:

```bash
# 跑全部 7 個
python -m research.exit_variants_backtest

# 或只跑前幾名候選
python -m research.exit_variants_backtest --variants baseline opp_strong opp_2bar no_opp
```

### 解讀標準

- **Δ Sharpe > +0.3 AND Δ net_bps > +3** → 真正值得 forward validate 的 variant
- **Δ Sharpe < -0.3** → 確認排除
- **baseline by_reason 拆解** → 看哪個 exit reason 平均賺最少,推測攻哪個 ROI 高

把表格直接貼回 chat,我幫你解讀。

---

## 第二步:Oracle Analysis(Phase 0 證明 ROI)

**這個 script 還沒寫**,但要寫不難。先跑 step 1,跑完再決定要不要寫 oracle。

如果 step 1 已經顯示某個 variant 大贏(例:no_opp Δ Sharpe +2.0)→ 直接 forward validate 那個 variant,不用做 oracle。
如果 step 1 所有 variants 都 marginal(Δ Sharpe < 0.3)→ 簡單調整沒救,寫 oracle 證明「ML model 值不值得做」。

### Oracle Analysis 設計(到時候寫)

```python
# research/exit_oracle_analysis.py(尚未寫)
"""
對每筆 backtest trade,找「神級出場點」:
  - 模擬整段 hold window(從 entry 到 trail_stop 觸發 / time_cap)
  - 找 max favorable excursion(LONG 最高、SHORT 最低)
  - 在 MFE 那根 bar 出場的話 net% 多少
  - 對比實際出場 net%

輸出:
  - 每筆 trade 的「gap = oracle_net - actual_net」
  - 分布:p25/p50/p75/p95 of gap
  - 按 exit_reason 拆解(trail_stop / time_cap / opp_signal 各自的 gap)
  - 按 tier / regime 拆解

決策:
  平均 gap > 30 bps → ML model 有 ROI,進 Phase 1
  平均 gap < 10 bps → exit 已接近最佳,問題在 entry,不要做 ML exit
  10-30 bps 中間 → 看 trade-off,可能 simple rule 改進就夠
"""
```

---

## 完整 ML Exit Model 路線(僅供參考,等 Phase 0 證明 ROI 再說)

### Phase 1:Simple ML exit predictor(2-4 週)
```
目標:Binary classifier「現在 exit 比繼續持有好嗎?」

Features(每根 bar 計算):
  - bars_held
  - unrealized_pct
  - current_atr / atr_at_entry(vol regime change)
  - distance_to_trail_stop_pct
  - direction signal(當前 V7 訊號)
  - confidence
  - regime
  - vol_kurtosis(尾部)
  - SHAP top-5 features 當下值
  
Target:label = (前瞻 N 小時的 max favorable excursion < 當前 unrealized)
       即「再持有不會更賺,該出場了」
       
Model:XGBoost classifier(跟 V7 一致)
Training:用 backtest trade tape,每筆 trade 展開成 (bar-level) 樣本
Validation:WF-OOS,留 30% hold-out
```

### Phase 2:Production wire(1-2 週)
```
1. 加進 V7OkxExecutor._manage_position
2. 一開始 ADVISORY ONLY:
   - 每 cycle 預測,log「would_exit_now=true with conf=X」
   - 不真正出場
3. 4 週累積 advisory 訊號 vs 真實 trade outcomes
4. 對比:advisory 預測該 exit 之後 trade 真的衰退?
5. 達到 80% precision 才升級到實際出場條件(第 4 個 exit reason: ml_exit)
```

### Phase 3:Regime-aware(可選)
```
分 regime 各訓一個 model,或讓 regime 是 feature
```

---

## 風險與紀律(CLAUDE.md 規則保持)

```
✗ 任何 model 改動禁直接上 OKX live
✗ 必須先 paper-equivalent / advisory mode 4 週
✗ Strategy sweep 必須留 OOS hold-out(30%+)
✗ 不准因為「最近 1 週好」就跳階段

進入 forward validation 前要:
  - WF-OOS Sharpe 改善 ≥ 0.5
  - net bps 改善 ≥ 5
  - MDD 不變壞 > 5pp
  - 三個都過才動
```

---

## 為什麼今天先不寫 Oracle / ML(handoff 心態)

1. **手機環境跑不動**(parquet 不在 dev env,backfill 也需要本機)
2. **疲勞** → 寫複雜邏輯容易產 bug,事倍功半
3. **ROI 未證實** → 不知道做不做得起來
4. **現有系統剛 stable** → 不要堆改動

回家精神好時再接,效率高 10 倍。

---

## 快速重接續流程(下次坐到電腦前)

```
1. 開電腦,git pull origin main
2. 看 docs/exit_model_roadmap.md(這份)的 TL;DR
3. 跑 § 第一步 exit_variants_backtest
4. 結果貼回 chat,我接著解讀
5. 根據結果決定走 oracle / forward validate variant / 暫停
```

---

## 今天累積但還沒處理的「待觀察」

- Alpha Decay Monitor 持續 危險 → 如果 14 天都這樣,觸發「model retrain」討論
- OKX live 跟 backtest baseline 的 WR 差距 12pp → 看是 alpha decay 還是 live execution friction
- 每天看 🟡 AUTO-HEAL 出現幾次,> 2 次/天 要查 WS 為什麼那麼脆

---

**最後更新**:2026-06-08(session end)
**Commits 在 main**:從 `f0af019` 到 `3805954`(所有 OKX bug fix + exit research tooling)
