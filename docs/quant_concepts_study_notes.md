# Quant 學習筆記 — 從你的系統反推 35 個必懂概念

> 為什麼寫這份筆記
> 這些概念都已經出現在你 V7 系統的某個檔案、某行 code 或某條 hard rule。
> 你不是從教科書開始學——你從「我自己用過這個」反推回去學。這比硬讀
> Hull 或 Wilmott 有效 10 倍，因為你已經有 context，只是不知道學名。
>
> 每一條的結構：
>   - 📖 是什麼（一句話白話）
>   - 🧮 數學/公式（極簡）
>   - 📍 你系統哪裡用
>   - 🎤 面試怎麼講（30 秒口頭版，可背）
>   - ⚠️ 常見坑（顯示你不只會用、還知道哪裡會壞）

---

## 目錄

- [Part 1：訊號層數學（IC、AUC、Calibration）](#part-1)
- [Part 2：驗證方法（Walk-Forward、Bootstrap、Null Test）](#part-2)
- [Part 3：模型概念（XGBoost、SHAP、Concept Drift）](#part-3)
- [Part 4：訊號解碼（Rolling Percentile、Hysteresis、Dual Model）](#part-4)
- [Part 5：風險數學（Sharpe、Kelly、Vol Drag、MDD）](#part-5)
- [Part 6：交易執行（ATR、Trailing、Funding、Slippage）](#part-6)
- [Part 7：合約 / 槓桿 / 倉位（Cross/Isolated、Net mode、Notional）](#part-7)
- [Part 8：風控系統（Kill Triggers、State Machine、Gate A/B）](#part-8)

---

<a name="part-1"></a>
## Part 1：訊號層數學

### 1. Information Coefficient (IC)

📖 「模型預測值」跟「實際發生的報酬」之間的相關係數。**IC 越高，模型越能排序**——預測值高的真的賺得多、預測值低的真的虧得多。

🧮 公式：`IC = corr(prediction, actual_return)`。Spearman IC 用 rank 算（穩定）、Pearson IC 用 raw value 算（容易被 outlier 拉走）。

📍 你系統哪裡用：
- `research/v71_v7_sizing_1x.py` 訓練完算的就是 Spearman IC
- CLAUDE.md 寫「Spearman IC 0.063 / top-5% precision 67.6%」
- 衰退警報就是看 rolling IC 掉得太快

🎤 面試怎麼講：「IC 是 prediction 跟 forward return 的 rank correlation。我們系統用 Spearman 因為 BTC 報酬尾巴很重，Pearson 會被一根 fat tail 整個拉走。0.05~0.08 在 crypto 4h 是合理水準，0.1+ 就要懷疑 leakage。」

⚠️ 常見坑：用 in-sample 算 IC（永遠很漂亮）。**任何 IC 沒講「OOS」前提都是廢話**。你的 mistake.md 有一條就是這個。

---

### 2. ICIR (Information Ratio of IC)

📖 IC 的 Sharpe ratio。多期 IC 的「平均值 / 標準差」。**穩定地 0.05 比有時 0.15 有時 -0.05 好**。

🧮 `ICIR = mean(IC_per_period) / std(IC_per_period)`。一般 ICIR > 0.5 算可用、> 1.0 是 ML quant 在追的。

📍 你系統哪裡用：feature validation 報告（`research/feature_validation*.csv`），新特徵不只看 IC 還看 ICIR。

🎤 「IC 是預測能力的平均，ICIR 是穩定性。我們新特徵入庫前必須跑 fold-consistency > 0.6——意思是 60% 的 fold IC 同號，不只均值漂亮。」

⚠️ 常見坑：只看 IC 點估，不看分散度。一個 IC 0.1 但 std 0.3 的特徵，下個月可能 IC -0.2。

---

### 3. AUC（ROC 曲線下面積）

📖 隨機抽一個正樣本、一個負樣本，模型把正的排在前面的機率。**0.5 = 亂猜，1.0 = 完美**。

🧮 `AUC = P(score(positive) > score(negative))`。

📍 你系統：Direction model 的方向二分類能力。「AUC 0.5412」是 V7 baseline。

🎤 「我們 direction model AUC 大概 0.54——比 random 高一點點。但這已經逼近這個資料源的天花板，因為 6 個新特徵家族跑完全部 fail。提升 AUC 的唯一路徑是異源資料，不是更深的 XGBoost。」

⚠️ 常見坑：把 AUC 0.55 講成「弱訊號」其實在 high-frequency 是**真本錢**。AUC 跟夏普沒有線性關係——AUC 0.55 + 好的 sizing 可以是 Sharpe 2。

---

### 4. Precision@k

📖 把預測分數最高的 top k% 拿出來，裡面有多少真的對。「精選頂尖訊號的勝率」。

🧮 `Precision@5% = (top 5% 預測中對的數量) / (top 5% 預測總數)`。

📍 你系統：`research/topk_precision_sweep.py`。Strong 訊號 = top 5% 的訊號（rolling percentile cutoff）→ precision 67.6%。

🎤 「我們不是看整個 model 的 accuracy，看的是 top 5% 預測的 precision。因為交易只進場高信心訊號，整體準確率不重要。AUC 0.57 對應的 top 5% precision 理論天花板大約 0.70，我們實測 0.676，已經貼著天花板。」

⚠️ 常見坑：mistake.md 2026-04-14 有一條——之前把目標設成 95% WR，**但 AUC 0.57 結構天花板就是 0.70**。所以「為什麼還沒達到」是錯問題，正確問題是「目標本身可不可達」。

---

### 5. Calibration（校準）

📖 模型說「90% 機率上漲」的那些訊號，**實際真的應該有 90% 上漲**。如果只有 50% 上漲，模型 over-confident。

🧮 看 Brier score（越低越好）、ECE（Expected Calibration Error，把預測分桶比較預測機率 vs 實際發生率）。

📍 你系統：`research/calibration_check.py`、`research/calibration_isotonic.py`。

🎤 「Calibration 是預測機率跟實際發生率的對齊度。一個 AUC 高但 calibration 差的 model 仍能排序，但你不能用它的機率做 Kelly sizing——數字會說謊。」

⚠️ 常見坑：mistake.md 2026-04-13 有一條——calibration check 用了三個模型版本混合 + 數據污染窗的 244 個樣本，結論完全錯。**Statistical 顯著 ≠ 結論可信**。

---

<a name="part-2"></a>
## Part 2：驗證方法

### 6. Walk-Forward OOS

📖 模擬「時間真的往前走」的驗證方式。每根測試 bar 的模型**只能看到它之前的資料**。

🧮 切 N 個 fold，fold i 訓練 bar 1..1000+i*100、測試 bar 1000+i*100..1000+(i+1)*100。

📍 你系統：`research/v71_v7_sizing_1x.py` 77 個 walk-forward fold。整套訊號層證據基礎。

🎤 「Walk-forward OOS 是 time-series 唯一能信的驗證——random k-fold 會洩漏未來資訊到訓練集。我們跑 77 fold，每個 fold 訓練窗只用該 fold 之前的 bar。」

⚠️ 常見坑：mistake.md 2026-04-13——用「生產模型」預測過去月份當「衰退診斷」，**那是 in-sample，零資訊量**。Walk-forward 不是切時間就好，模型也要每 fold 重訓。

---

### 7. Purge + Embargo

📖 防止 label 重疊洩漏。target 是 4h 後的報酬，所以 bar t 的 label 用了 bar t+4 的資料。**訓練集不能含跟測試集第一個 bar 在時間上重疊的 bar**。

🧮 Purge = 從訓練集移除「label 與測試集 bar 重疊」的訓練 bar。Embargo = 測試集結束後再隔幾根 bar 才開始新訓練集（避免 momentum 洩漏）。

📍 你系統：`research/dual_model/` purge=4 + embargo=4。

🎤 「我們 target 是 path return 4h，所以訓練集要 purge 跟測試集第一個 bar 重疊 4 根的訓練樣本，再 embargo 4 根防 autocorrelation 洩漏。沒做這個的 walk-forward IC 會虛胖 30-50%。」

⚠️ 常見坑：很多 Kaggle code 切 time-series CV 但沒 purge——這在學術上是 leakage，業界看到直接退稿。

---

### 8. Bootstrap Confidence Interval

📖 從你已有的樣本**重抽 N 次**算同一個統計量，分布的 2.5%~97.5% 分位數就是 95% CI。

🧮 對 trade list `[+30, -20, +5, ...]`，重抽 10,000 次（with replacement）算 mean，取分位數。

📍 你系統：`research/v71_v7_sizing_1x.py` `bootstrap_trades(n_iter=10000)`、`research/paper_trading_robustness.py`。

🎤 「Bootstrap CI 處理樣本不大時的不確定性。Strong 95% CI [-2.2, +14.6] bps 包含 0，所以雖然點估正、無法統計上斷言 edge 顯著。」

⚠️ 常見坑：Bootstrap 假設樣本獨立——對 time-series trade 來說不成立（連續 trade 在同個 regime）。要用 **block bootstrap** 保留自相關。

---

### 9. Block Bootstrap

📖 一次抽「連續一段 trade」當一個 block，而不是抽單筆。**保留時間自相關**（連虧或連賺成串）。

📍 你系統：`research/stress_test_v7.py` 8-trade block bootstrap 估 99th percentile MDD。

🎤 「Crypto trade 高度自相關——好的 regime 連賺、壞的 regime 連虧。普通 bootstrap 會把 winning streak 跟 losing streak 都打散，低估真實 MDD。我們用 8-trade block 保留這個結構。」

⚠️ 常見坑：block size 太小（=普通 bootstrap）、太大（樣本內 block 數太少 → 沒抽到變異）。Rule of thumb：block_size ≈ √(N_trades)。

---

### 10. Random-Entry Null Hypothesis

📖 「假設訊號是亂選的，剩下的出場邏輯能不能也賺錢？」如果亂進場+你的出場邏輯也賺，**那賺的是出場邏輯不是訊號**。

📍 你系統：`research/v71_v7_sizing_1x.py` `--null` flag。

🎤 「我跑了 random-entry null：隨機 entry + 同套 3xATR 出場，只賺到 chop 區的 noise。我的訊號比 null 顯著高 X bps，所以 edge 確實來自訊號層，不是出場邏輯本身。」

⚠️ 常見坑：多數人不跑 null，直接報「我的策略賺錢」。**很多 trend-following 系統的賺錢其實是 ATR trailing 在 trending market 撿錢**，不是訊號本身。

---

### 11. Look-Ahead Bias / Leakage

📖 訓練的時候不小心讓模型「偷看到未來」。最常見：feature 用了未來資料、target 用了 entry 之後的 close 來算 entry signal。

📍 你系統：CLAUDE.md 「無前視偏差」hard rule、`build_live_features()` 同時用於訓練 + 生產（保證一致）。

🎤 「Look-ahead 是 quant 的死罪。我們所有 rolling 都用 trailing-only、所有 merge_asof 用 backward。生產 inference 跟訓練用同一個 `build_live_features()`，物理上不可能漂。」

⚠️ 常見坑：日資料的「今天 close」跟「今天 high」常被當 feature——但實際上你下單時還沒看到今天 close。`merge_asof` 沒設 backward 會默默用未來資料。

---

<a name="part-3"></a>
## Part 3：模型概念

### 12. XGBoost / Gradient Boosting

📖 一棵棵 decision tree 接力。每棵新樹學「上一棵樹的錯」。

🧮 `F_{n+1}(x) = F_n(x) + η × tree(x; gradient_of_loss)`。η = learning rate（小學得慢但穩）。

📍 你系統：Direction Regressor + Magnitude Regressor 都是 XGBRegressor。

🎤 「XGBoost 是 ensemble，每棵新樹 fit 上一輪的 residual gradient。比 random forest 強的地方是它有 regularization（leaf weight L2）跟 second-order Taylor 展開——loss surface 訊息用得更滿。」

⚠️ 常見坑：default `n_estimators=100` 在小資料會 overfit。我們用 early stopping 配 valid set。mistake.md 2026-04-13 有一條——regime 切片後每群 < 500 樣本，XGBoost overfit 到「反方向預測」。

---

### 13. Bias-Variance Tradeoff

📖 模型太簡單 = bias 高（underfit）；模型太複雜 = variance 高（overfit）。

🧮 `Expected error = bias² + variance + irreducible_noise`。

📍 你系統：tree depth 4-6（不深）、L2 regularization、80/20 train/val split。

🎤 「Crypto 4h 的訊噪比極低，bias-variance 配比要往 high-bias 那邊靠——簡單模型 + 強 regularization。我們用 depth 4 + min_child_weight 提高，比 depth 8 出 OOS 結果好。」

⚠️ 常見坑：盲目堆 feature。多 50 個沒用的 feature 會把 variance 拉到 OOS 直接崩。

---

### 14. SHAP (SHapley Additive exPlanations)

📖 一個訊號的預測「**這個 feature 貢獻了多少**」。從合作博弈論借來的 Shapley value。

📍 你系統：`indicator/signal_explainer.py`，Strong 訊號時觸發、Telegram 推文有 top-3 drivers。

🎤 「Strong 訊號時我們跑 TreeExplainer，產出該 prediction 的 feature attribution。用來做兩件事：1) 訊號可解釋性，2) 找 feature drift——若某個本來不重要的 feature 突然占主導，model 可能 over-rely on noise。」

⚠️ 常見坑：SHAP 是**該預測**的解釋，不是 model 的全局重要度。Global importance 看 gain/cover/permutation importance。

---

### 15. Concept Drift / Regime Change

📖 模型訓練時的 feature-target 關係，**未來可能變了**。crypto 特別嚴重（DeFi 興起、ETF 通過、宏觀 regime 翻轉）。

📍 你系統：CLAUDE.md 「Magnitude IC 從 Feb/Mar 交界腰斬」、`research/concept_drift_monthly_ic.py`、regime detection 模組。

🎤 「我們有專門 monitor：每月跑 walk-forward IC，連續 3 個月 < 警報線就觸發 alpha decay alert。Mar 2026 Mag IC 從 0.31 掉到 0.10，就是 concept drift 教科書案例。」

⚠️ 常見坑：把 drift 跟 noise 混淆。要看 rolling IC 的趨勢 + bootstrap CI 是否離開歷史分布。

---

### 16. Regime Detection

📖 把時間切成不同「市場狀態」（trending bull / trending bear / choppy / warmup）。**同一個 model 在不同 regime 表現不同**。

📍 你系統：`indicator/regime_detector.py`，4 個 regime。CHOPPY 在 91.8% WR 那段是 sample artifact（mistake.md 教訓）。

🎤 「Regime detection 用 trailing volatility + trailing trend strength + ADX。CHOPPY 時期我們的訊號特別準（mean reversion 環境），TRENDING 時期 Magnitude model 容易 underpredict。」

⚠️ 常見坑：regime 切片後子模型訓練樣本不夠 → mistake.md 2026-04-13 規 BEAR 子模型只 50-100 樣本 → AUC 0.378（系統性反向）。**切 regime 前先 assert min sample > 500**。

---

<a name="part-4"></a>
## Part 4：訊號解碼

### 17. Rolling Percentile Cutoff

📖 用過去 500 根 bar 的預測分布動態決定門檻。「今天的 top 5% 是什麼數值」每天都不同。

📍 你系統：`indicator/inference.py` 500-bar trailing buffer，Strong = top 5%，Moderate = top 15%。

🎤 「我們不用固定門檻——因為市場 vol 變了，門檻也要變。用 trailing 500 bar 的 5% / 95% 分位數做動態 cutoff。這也是為什麼 warmup 100 bar 之內用 fallback——buffer 還沒夠樣本。」

⚠️ 常見坑：mistake.md 2026-04-19——用 WF OOS fold 模型的預測（std 0.0008）去 seed buffer，生產模型 std 0.003，**rolling percentile 把所有預測都打成 DOWN**。任何 buffer 都必須跟生產模型同尺度。

---

### 18. Hysteresis + Cooldown

📖 訊號從 NEUTRAL → UP 比從 UP → NEUTRAL 容易（hysteresis）。出訊號後一段時間不再出（cooldown）。**避免訊號 jitter**。

📍 你系統：`indicator/inference.py` decode 階段。

🎤 「Hysteresis 防 noise——pred 在門檻邊緣抖動會出來一堆訊號。我們設出 Strong 後要明顯回中性才會再出。Cooldown 是時間上的——出完訊號至少間隔 N 根 bar。」

⚠️ 常見坑：cooldown 設太長會錯過真 reversal、太短回到 jitter。要用實際 trade tape 看出來。

---

### 19. Dual Model (Direction + Magnitude)

📖 兩個獨立模型——一個猜方向（會漲還跌）、一個猜幅度（會動多少）。分開訓練、分開驗證。

📍 你系統：V7 architecture 核心。Direction = 136 features XGBRegressor、Magnitude = 72 features 獨立 model。

🎤 「方向跟幅度是兩個獨立 information sources。把它們分開讓我們可以個別 monitor IC、個別補強 feature。Magnitude IC 衰退時不影響 Direction，反之亦然。比 single multi-output model 好維運。」

⚠️ 常見坑：兩個 model 用了不同 feature set 但**共享同一段歷史資料**——驗證時容易共謀失敗（同樣的 data quality bug 兩邊都影響）。

---

### 20. TWAP Path Return as Target

📖 不用「4 小時後那一根 close」當 target，用「**之後 4 根 1h bar 的平均 close**」當 target。降低單 bar noise。

🧮 `y = mean(close[t+1..t+4]) / close[t] - 1`

📍 你系統：CLAUDE.md「核心 target」一節。

🎤 「我們的 target 是 4h TWAP path return，不是 endpoint return。原因：endpoint return 對單根 fat-tail bar 太敏感，TWAP 更接近『實際你能交易到的價格』。模型也更好學。」

⚠️ 常見坑：TWAP target 跟「我下單會吃到的平均價」是兩回事——前者是事後計算的數學量，後者是執行品質。**訊號用 TWAP target 但出場用 trailing stop**，這個 mismatch 是你系統研究階段一直在思考的議題。

---

<a name="part-5"></a>
## Part 5：風險數學

### 21. Sharpe Ratio

📖 報酬除以波動度。「**每承擔 1 單位風險賺多少**」。

🧮 `Sharpe = (E[r] - r_f) / σ(r)`，年化 = 乘 `√(periods_per_year)`。

📍 你系統：`research/v71_v7_sizing_1x.py` 三種年化（per-trade、daily、annualised），CLAUDE.md「Sharpe ≥ 1.5 才進 Stage 4d」。

🎤 「Sharpe 是最常用的 risk-adjusted return。注意三件事：1) 它假設 return 是 normal——crypto fat tail 下會高估；2) 它扣 risk-free——crypto quant 通常設 0；3) 年化要乘 √N，要確認 N 對應你的取樣頻率。」

⚠️ 常見坑：per-trade Sharpe 跟 annualised Sharpe 不是一回事——`ann_Sharpe ≈ per_trade_Sharpe × √(trades_per_year)`。你系統的 5.10 Sharpe 是哪個版本要講清楚。

---

### 22. Sortino Ratio

📖 Sharpe 的改良版。**只算下行 vol**，不懲罰上行波動。

🧮 `Sortino = (E[r] - r_f) / σ(downside_r)`，downside = max(0, r_target - r)。

📍 你系統：dashboard 有 Sortino 欄。

🎤 「Sortino 更接近 trader 直覺——上行波動是你想要的，不該被當風險。實務上 trend-following 系統的 Sortino 通常比 Sharpe 高 30-50%。」

---

### 23. Profit Factor (PF)

📖 賺錢交易總和 ÷ 賠錢交易總和的絕對值。**> 1 就是賺、> 2 是好策略**。

🧮 `PF = sum(wins) / |sum(losses)|`。

📍 你系統：`paper_trading.py`、`exit_variants_backtest.py` 都報 PF。

🎤 「PF 是不依賴勝率的賺錢指標。WR 45% 但 PF 2.0 = 賺；WR 70% 但 PF 0.9 = 虧。我們看 PF 比看 WR 多。」

⚠️ 常見坑：少數 fat-tail win 會把 PF 灌很高（一筆 +500 bps win 可以扛 5 筆 -100 bps）——要看 PF 的分布而不是點估。

---

### 24. Maximum Drawdown (MDD)

📖 從歷史 peak 跌到 trough 的最大幅度。**心理痛苦 + 復原時間的雙重打擊**。

🧮 `MDD = max(peak - trough) / peak`，計算 cumulative return 序列的 running max - current。

📍 你系統：daily/total loss cap、Stage 進階條件、stress test 都圍繞 MDD。

🎤 「MDD 比 vol 更重要——客戶看 MDD 決定要不要贖回。我們 Stage 3 進 4a 條件是 MDD < 20%、4a 進 4b 是 MDD < 10%。」

⚠️ 常見坑：歷史 MDD 不代表未來 MDD——bootstrap 99th percentile MDD 更可信。我們 stress test 跑這個。

---

### 25. Calmar Ratio

📖 年化報酬 ÷ MDD。「**忍受最大痛苦每年能換多少報酬**」。

🧮 `Calmar = annualised_return / |MDD|`。

📍 你系統：dashboard 有。

🎤 「Calmar 直接量痛苦/報酬比。Sharpe 假設 normal、看 vol；Calmar 不假設、看實際最壞痛苦。Crypto 建議 Calmar > 1.0 才考慮放大資金。」

---

### 26. Kelly Criterion

📖 數學上最優的下注比例。「**每次該 all-in 多少**」。

🧮 對連續報酬：`f* = μ / σ²`（簡化版，假設 normal）。離散二元：`f* = p/L - q/W`（p=勝率, q=敗率, W=贏幅, L=虧幅）。

📍 你系統：CLAUDE.md「Kelly optimal: f* ≈ 0.56x（已小於 1x）」leverage 上限論證的 anchor。

🎤 「Kelly 給數學上的最優賭注比例。但實務上**沒人用 full Kelly**——因為 Kelly 假設 μ/σ 已知，實際是 noisy estimate。一般用 quarter Kelly 或 half Kelly 才不會被 noise 炸。我們 cap 在 2x 是因為 Kelly 0.56x 加上 estimation error，2x 已經是 risky 上限。」

⚠️ 常見坑：拿 in-sample 的 μ 算 Kelly → 上線後 ruin。Kelly 對 estimation error 極度敏感。

---

### 27. Volatility Drag

📖 leveraged 部位的「**幾何報酬 < 算術報酬**」現象。長期下來槓桿會吃掉一部分報酬。

🧮 `r_compound ≈ E[r] - 0.5 × σ² × L²`。L = leverage。

📍 你系統：CLAUDE.md leverage ladder 推導：
- L=2.0: drag = -18%（edge 還能覆蓋）
- L=3.0: drag = -40.5%（drag > expected return，長期虧）
- L=5.0: drag = -112%（mathematical ruin）

🎤 「Vol drag 是為什麼 3x leveraged ETF 長期都跑輸 underlying。對我們：σ ≈ 30%（年化）、L=3 → drag -40%，意思是即使每筆都 EV 正，槓桿到 3x 長期會虧。這也是 2x 是 hard cap 的數學基礎，不是拍腦袋。」

⚠️ 常見坑：很多人只算「leveraged 報酬 = L × 報酬」忽略 drag。短期是對的、長期錯。

---

### 28. Win Rate（三種定義）

📖 「賺錢交易的比例」——但**怎麼算「賺」差別很大**。

📍 你系統三層：
- `gross_pct > 0`：純價差對方向（目前 /okx-perf 用的）
- `net_pct > 0`：扣 8 bps 假設成本
- `equity_after > equity_before`：錢包真相（含真實 fee + funding + 滑價）

🎤 「我們系統內 WR 有三個 layer，差距能到 5pp。報告時必須明確標 gross/net/equity，不然 audit 會打槍。」

⚠️ 常見坑：見上次對話。`/okx-perf` 用 gross > 0，**比真實賺錢率偏高**。

---

### 29. Expected Value (EV)

📖 「**長期每筆平均能賺多少**」。比 WR 重要 100 倍。

🧮 `EV = WR × avg_win - (1-WR) × avg_loss - cost`。

📍 你系統：CLAUDE.md「邊際正 EV」formulation 出現多次。

🎤 「WR 70% 但 avg_win = 5、avg_loss = 20 → EV = 0.7×5 - 0.3×20 = -2.5，每筆平均虧。WR 是給人看的，EV 是給帳戶看的。」

---

<a name="part-6"></a>
## Part 6：交易執行

### 30. ATR (Average True Range)

📖 「**最近 N 期平均波動範圍**」。用來設停損距離、調 position size。

🧮 True Range = max(high-low, |high-prev_close|, |low-prev_close|)。ATR = Wilder rolling mean（類似 EMA but slower）。

📍 你系統：3xATR(14) Wilder trailing stop。

🎤 「ATR 動態根據市場波動調停損——vol 高時停損遠（不容易掃出）、vol 低時停損近（保護獲利）。3xATR 是 trend-following 圈標準，留約 2-3% 的 BTC normal noise。」

⚠️ 常見坑：用 SMA ATR 跟 Wilder ATR 結果差很多——backtest 跟 live 算法要一致。

---

### 31. Trailing Stop

📖 賺錢時停損跟著上來、虧錢時不動。**鎖定獲利同時保留奔跑空間**。

📍 你系統：trail extreme = max(highs since entry) for LONG；trail stop = trail_extreme - 3*ATR；只能往對你有利方向 amend。

🎤 「Trailing stop 是 trend-following 的核心 exit。固定 TP 太早出、固定 SL 太晚出。Trailing 讓 winner run 同時保護 BE。我們的 trail bug 修了三輪才真的修——這是 mistake.md 上的痛點。」

⚠️ 常見坑：trail amend 失敗 silent（OKX 50014）→ trail 永遠凍住卻不知道。你系統的 trail bug 就是這樣。

---

### 32. Maker vs Taker Fee

📖 **Maker**（你提供流動性，掛單）通常 0.02% 或更便宜。**Taker**（你吃流動性，市價單）0.04~0.05%。

📍 你系統：taker_cost = 0.0008（8 bps round-trip）= 0.04% × 2。

🎤 「我們市價單為主，所以用 taker。Round-trip 8 bps 是進出兩次。Maker 雖然便宜但會 race condition——掛了沒成交價格已經跑掉，trend-following 不適合。」

⚠️ 常見坑：很多 backtest 假 maker fee 卻實際 taker 執行，cum return 直接虛胖 3-5%。

---

### 33. Funding Rate

📖 永續合約（perpetual）的價格錨定機制。每 8h 結算一次，**多空雙方互轉錢**。funding 正 = longs 付 shorts。

📍 你系統：funding rate features 餵給 model。但 PnL 計算沒納入 funding（這是 net_pct 跟 equity_after 落差的來源之一）。

🎤 「Funding 是 perp 跟 spot 收斂的力量。長期 funding 正代表 longs 在付租金，做空有 carry。我們系統把 funding 當 feature 但 PnL bookkeeping 沒含——這是已知 limitation。」

⚠️ 常見坑：holding LONG over funding settlement 會被扣錢，不在 trade 的 gross_pct 反映，會讓 backtest 跟 live 出現偏差。

---

### 34. Slippage

📖 「**你想成交的價格 vs 實際成交的價格**」之間的差距。市價單在 thin book / 大單時特別嚴重。

📍 你系統：stress test 加 +30 bps 滑價測試（`stress_test_v7.py` Model B）。

🎤 「滑價 = order size 跟 book depth 的函數。$100 倉位幾乎無滑價、$10k 在 thin book 可能 30+ bps。我們做 sizing 時假設 $X 名目下 ~5 bps slippage，stress test 用 30 bps 是 worst case BTC perp。」

---

### 35. Position Sizing (Fixed-Fractional / Sizing B)

📖 每筆 trade 用「**equity 的固定比例 × leverage multiple**」。equity 變大下次自動放大、虧錢自動縮小。

🧮 你系統 Sizing B：`notional = NOTIONAL_LEV_MULT × equity`，例如 2x equity。然後算 contracts 數量。

📍 你系統：`executor._compute_size_contracts`、`v7_okx_positions.size_frac`。

🎤 「Sizing B 是 fractional Kelly 的近似。每筆固定 2x equity 名目，不是固定 USD。複利自然發生、scale-aware。輸了會自動降風險、贏了自動加倉。」

⚠️ 常見坑：notional 是「名目槓桿」（給策略看的），不等於 OKX 的「account leverage」（保證金多寡）。下個概念解釋。

---

<a name="part-7"></a>
## Part 7：合約 / 槓桿 / 倉位

### 36. Notional vs Equity

📖 **Notional** = 部位的名目大小（你控制了多少錢的 BTC）。**Equity** = 你帳戶實際有多少錢。

🧮 `notional = price × size × contract_size`。`effective_leverage = notional / equity`。

📍 你系統：CLAUDE.md「max_effective_leverage: 3.0」、kill_checks 預先檢查。

🎤 「Effective leverage = 名目 / 帳戶 equity，是策略視角的槓桿。Account leverage（OKX 設的 10x）是保證金鎖定機制，跟策略無關。我們 cap 名目 ≤ 3x equity，account 那邊設 10x 只是為了讓 $100 capital 能買 1 contract（$750 名目）。」

⚠️ 常見坑：把 account leverage 跟 effective leverage 搞混 → 報告講「我用 10x 槓桿」其實 notional 才 2x，給人錯印象。

---

### 37. Contract Size

📖 「**1 張合約代表多少標的**」。BTC-USDT-SWAP 在 OKX = 0.01 BTC。

📍 你系統：`contract_size_base = 0.01`、SQL 算 notional 用。

🎤 「OKX BTC perp 1 張 = 0.01 BTC，當 BTC=$75k 時 1 張名目 $750。$100 帳戶要做 1 張就是 7.5x leverage——這就是為什麼 Stage 3 informed override 把 leverage cap 從 1x 升到 10x。」

---

### 38. Cross Margin vs Isolated Margin

📖 **Cross** = 整個帳戶 equity 都可以幫一個倉位扛虧損。**Isolated** = 每個倉位有自己 ring-fence 的 margin，爆倉只損該 margin。

📍 你系統：`td_mode: cross / isolated`。Default cross，Isolated dormant via `OKX_TD_MODE=isolated`（2026-06-05 blowup mitigation）。

🎤 「Cross 用整個 equity 撐倉，存活力強但 1 個極端 trade 可能拖垮整個帳戶。Isolated ring-fence 每倉 margin，承受極端事件好但 normal trade 比較容易爆。我們 Stage 3 default cross，但 isolated 已經 staged 隨時可切換。」

⚠️ 常見坑：cross→isolated 切換**必須帳戶 FLAT**，否則 OKX 不接受。

---

### 39. Net Mode vs Long/Short Mode

📖 **Net mode**：一個 instrument 只有一個 signed 部位（+5 contracts = long 5）。**Long/short mode**：long 跟 short 是兩個獨立部位，可以同時存在。

📍 你系統：`pos_mode: long_short_mode`（OKX default），executor 在每筆 order 帶 `posSide`。

🎤 「Long/short mode 允許 hedge——能同時開 long 跟 short。我們不會 hedge，但用這個 mode 是因為 OKX 新帳戶 default 就是這個，省一個 setup step。Code 要在 net_mode 跟 long_short_mode 兩邊都 work。」

---

### 40. Algo Orders / Trigger Orders

📖 **不會立刻成交**，要等價格觸到 trigger price 才轉成市價/限價單。Stop-loss、take-profit、trailing stop 都是 algo orders。

📍 你系統：trail stop 用 algo order，有 `algoClOrdId`、`algoId` 兩個識別。fill 時 WS 推 algoClOrdId（**不是** clOrdId）→ 你 sync 邏輯要 handle。

🎤 「Algo orders 是 exchange 側的 stop。優點：你斷網它仍會觸發；缺點：trigger 跟 fill 是兩段，fill 推送的 ID 是 algo 不是 cl。我們系統的 trail amend bug 之一就是 ID matching 漏 algoClOrdId。」

---

### 41. Reduce-Only Order

📖 一個 flag。標記「**這單只能減倉、不能新開倉**」。緊急平倉、stop loss 必備。

📍 你系統：close 操作、orphan 緊急平倉都加 `reduceOnly=true`。

🎤 「reduce-only 是防呆——萬一你想 close 但 size 算錯反而開了反向倉，這個 flag 會讓 exchange 直接拒絕。所有出場單必開 reduce-only。」

---

<a name="part-8"></a>
## Part 8：風控系統

### 42. Kill Triggers / Kill Switch

📖 預設好的「**遇到 X 就自動停 / 降階段**」條件。**不靠紀律、寫進 code**。

📍 你系統：`docs/stage2_kill_criteria.md` 31 個 trigger 分 6 大類（A-F）：
- A：連線（WS 斷線、重連失敗）
- B：執行品質（reject rate、amend fail rate）
- C：時序（NTP drift）
- D：data quality
- E：should-never-happen（leverage cap 被破、posMode 不對）
- F：策略表現（連虧、drawdown）

🎤 「Kill triggers 是 hard rules 不是 soft suggestions。CAP-1 daily loss、CAP-2 total loss、A1 WS 斷線 5 分鐘 demote、E1 leverage 違規不啟動。每個 trigger 對應 code 裡一個檢查 + 一個降階動作。」

⚠️ 常見坑：kill switch fail silently 是最糟故障——比如餘額查詢失敗 fallback 到 initial_capital，daily cap 算 0% loss 永遠不會觸發。**fail-closed not fail-open**。

---

### 43. State Machine (Executor Status)

📖 系統的狀態枚舉：`INIT → CONNECTING → READY → ACTIVE → HALTED / DEMOTED`。**每個轉移有明確條件**。

📍 你系統：`v7_okx_executor_status` table single-row state、`executor.py` state transitions。

🎤 「state machine 強制清晰的狀態轉移。HALTED 是 transient（可 heal），DEMOTED 是 terminal（要 manual 介入）。沒有 state machine 系統會在『以為自己在跑』和『其實已經死』之間 grey area，最危險。」

---

### 44. Reconciliation

📖 系統 DB 認為的部位 vs exchange 實際的部位**對不對得上**。對不上 = mismatch。

📍 你系統：`PositionReconciler.reconcile_cycle()`，5 種 mismatch：
- `orphan_local`：DB 說有倉但 exchange 沒
- `orphan_exchange`：exchange 有倉但 DB 沒
- `size_diff`：兩邊都有但 size 不同
- `direction_diff`：兩邊方向反
- `unavailable`：查不到 exchange，無法判定（新增 06-14 P0-3 fix）

🎤 「Reconciler 每 60s 對帳一次。Mismatch 不容忍——orphan_local 連 2 cycle 自動 heal、orphan_exchange 觸發 manual approval、size/direction diff 觸發 HALT。這是『信但要驗』的工程化。」

⚠️ 常見坑：mistake.md 級別的——`get_positions` 失敗回 `[]` 跟「真的 flat」不可分，會誤判 orphan_local 然後 auto-heal 把 DB 唯一記錄刪掉。fix 是拋 `OkxQueryUnavailable`。

---

### 45. Dead-Man's Switch

📖 「**我沒主動 ping，就當我死了**」。系統定期推送活著訊號，超時觸發告警。

📍 你系統：commit `d6f368e`，update_cycle 沉默卡住 → 主動推 Telegram critical。

🎤 「Dead-man's switch 防 silent failure。系統內部覺得自己活著、但 update_cycle 在某個 except 卡住——這是最危險的 mode。每 30s 寫 heartbeat 到 DB + Telegram，watcher 偵測到 sleep 超時就告警。」

⚠️ 常見坑：mistake.md 2026-04-22 那條——update_cycle 外層 try/except 把 error 吞成 `status="error"`，外面看 Railway 全綠以為正常，實際靜默死了 10 小時。

---

### 46. Fail-Closed vs Fail-Open

📖 系統元件失敗時**預設拒絕操作**（fail-closed）還是**預設放行**（fail-open）。**風控系統一律 fail-closed**。

📍 你系統：餘額查詢失敗 → 跳過本 cycle（不 fallback）、positions 查詢失敗 → 拋 exception 觸發 HALT。

🎤 「Fail-closed 是 secure-by-default。WS 斷了就停止下新單、DB query 失敗就拒絕執行。fail-open 在風控系統等於『出事就讓壞事擴大』。」

---

### 47. Pre-Submit Guards

📖 **下單前**多層檢查，任何一層 fail 直接拒絕送 exchange。

📍 你系統：`kill_checks.check_presubmit_order()`，10 個檢查：leverage cap、capital cap、daily loss cap、total loss cap、max position count、posMode 一致、tdMode 一致、reduce-only flag 等。

🎤 「Pre-submit guards 是 last-mile defense。即使 strategy bug 算錯 size、即使 state machine bug 進了 wrong state，guard 都會擋下。Layered defense。」

---

### 48. Gate A vs Gate B（你的 framework）

📖 你自己發明的雙閘門：
- **Gate A**：訊號層有 statistically significant edge（IC + bootstrap CI 下緣 > 0）
- **Gate B**：交易層執行後 net edge ≥ 0（avg net bps + cum equity）

📍 你系統：Stage 3 → Stage 4 promotion gate、CLAUDE.md 寫進去。

🎤 「我把 promotion gate 拆兩層：訊號層證 edge 存在（Gate A，n=739 / WR 59.5% / CI 下緣 56% > 50%），交易層證 execution 不會把 edge 吃掉（Gate B，30-50 live trades 後 net ≥ 0）。這個 separation 讓我不會把『訊號好』跟『系統會賺錢』混淆。」

---

### 49. Staged Framework

📖 從「驗證」到「全資金」**分階段放大**。每階段 hard rules + 進階條件。

📍 你系統：Stage 0 純指標 → Stage 1 paper → Stage 2 testnet → Stage 3 $100 → Stage 4a $1k 1x → 4b $1k 1.2x → 4c $5k 1.5x → 4d $10k+ 2x。

🎤 「Staged framework 防『一夜致富』情緒驅動 → 規模、槓桿、edge 確信度三條線同步前進。每個 stage 進階條件寫在 code 而不是日記裡。Stage 4d 槓桿 2x 是 hard cap——除非 24mo 實盤 Sharpe ≥ 3.0 才考慮放寬，這個門檻來自 Kelly + vol drag 數學，不是情緒。」

---

## 終章：學習這份筆記的策略

1. **不要從第 1 條讀到第 49 條**。挑一個你最不熟的 part 開始，例如 Part 5（風險數學）。
2. **每個概念配一張圖**：白紙 + 一個例子數字 + 你自己用嘴巴講一遍。
3. **找 5 個對應你系統的 code line**：grep 一下，看真的這個概念出現在哪。理解「概念 ↔ 實作」的對應比死背公式重要。
4. **每週挑 3 個重點概念**講給「不會的朋友」聽。能講清楚 = 真的懂。
5. **不會的查 Wikipedia 英文版 + 中文版各一次**。英文比較精確、中文比較白話。

## 進階閱讀（按 part 對應）

- Part 1-2（驗證）：López de Prado《Advances in Financial Machine Learning》第 6-8 章
- Part 3（ML）：《Hands-On ML》Chap 6（trees + ensemble）
- Part 5（風險）：Ed Thorp《A Man for All Markets》（Kelly 起源）
- Part 6-7（execution）：Larry Harris《Trading and Exchanges》（市場微結構聖經）
- Part 8（系統）：你自己的 `docs/stage2_kill_criteria.md` 跟 `docs/okx_integration_design.md`——這兩份品質夠當教材

每個都不用全部讀，挑章節。
