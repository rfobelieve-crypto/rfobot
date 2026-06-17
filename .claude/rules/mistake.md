# Mistake Log

Record logic errors and bad decisions to avoid repeating them.

---

## 2026-06-17: facade-skip bug 第三次——isolated 切換漏了 OkxClient.set_leverage，live 永遠開不了倉

**What happened:**
啟用 `OKX_TD_MODE=isolated` 後第一個 Strong 信號，live 推 `🔴 OKX OPEN ABORTED: set-leverage(isolated 10x posSide=long) failed — no order sent`。executor 在 isolated 開倉路徑呼叫 `self._client.set_leverage(...)`（executor.py:1010），但 live 下 `self._client` 是 **OkxClient facade（client.py）**，而 facade **沒有 set_leverage passthrough**（只存在於 rest.py:240 與 mock_client.py:198）→ 拋 `AttributeError` → 被 executor 的 bare `except Exception`（executor.py:1014）吞掉 → `lev_ok=False` → 中止開倉。net effect：isolated 模式下 executor **永遠開不了任何倉**。告警裡的 `posSide=long` 是 config 拼進字串的值，**不是 OKX 真的拒了**——AttributeError 在到達 OKX 之前就拋了。

**Root cause:**
**這是 facade-skip bug 第三次復發**（第一次 amend_algo_stop 漏 inst_id 2026-06-10，第二次同 bug 修不完全 2026-06-14，見 [[project_trailing_stop_amend_bug]]）。完全相同的盲點：一個跨層 feature（這次是 isolated margin 的 set_leverage）**只加了 executor 呼叫端 + rest 底層 + mock 測試替身，唯獨漏了中間的 OkxClient facade**。測試全綠是因為 trading 測試注入 `MagicMock` 當 client，MagicMock **auto-vivify 任何屬性**（`client.set_leverage(...)` 自動回傳 truthy mock），所以測試以為方法存在；唯一用 faithful MockOkxClient 的測試又只跑 `td_mode="cross"`，根本不進 isolated 分支。**facade↔executor 這條縫零覆蓋——client.py 本來連一個測試檔都沒有**。2026-06-14 的 LESSON 明明已寫「修 call site + 底層卻跳過 facade = 沒修完，要驗整條 call chain」，但下一個跨層 feature 還是踩同一個坑——因為當時只手動修了那一個方法，**沒有建立結構性防護**。

**Correct approach（已修，commit 5a41ad7 pushed）:**
1. client.py 補 set_leverage passthrough，簽名與 rest.set_leverage 鎖死（keyword-only inst_id/lever/mgn_mode/pos_side）。
2. **新增 tests/test_okx_client.py（facade 本來零覆蓋）= 結構性防護**：用 AST 自省抓出 executor 在 `self._client.<name>` 上呼叫的**每一個**方法，斷言 facade 都有定義（`test_facade_exposes_every_method_executor_calls`）+ 簽名 superset 檢查（facade 不可 drop rest 接受的參數，專抓 amend_algo_stop 那種「方法在但簽名漂移」`test_facade_signatures_match_rest_for_shared_methods`）。**反向證明過**：刪掉 set_leverage → 測試立刻變紅、精確指出缺失方法。這對測試能擋整類 facade-skip（amend + set_leverage 都會被抓），不用手維護方法清單。
3. pos_mode 查證：帳戶 long_short_mode（commit 4c982c4 live smoke + 先前 SHORT 0.29 帶 posSide=short 對帳 CONSISTENT 實證），config 與帳戶一致 → 補完 facade 後真實 OKX 呼叫成功、無 51000 風險。318 okx 測試綠。

**Rule:** 任何**跨層**的新方法/簽名變更（call site → facade → rest/ws → mock）**必須同時改 facade，而且要有結構性測試保證 facade 暴露 executor 呼叫的每個方法**——不能靠「記得改 facade」（已證明記不住，三次了）。test double（MockOkxClient）的簽名要**嚴格**，optional 參數正好遮這類 bug；MagicMock 當 client 永遠測不出 facade 缺方法。新增任何 facade↔executor 之間的方法，跑 `tests/test_okx_client.py`。看到「測試全綠但 live 報 AttributeError / TypeError / OPEN ABORTED」第一個查的就是 **facade 是不是漏了 / 簽名漂移了**。

---

## 2026-06-07: admin_heal 第二次造孤兒倉——破壞性操作掛在無認證 GET + 只改 DB 不平 OKX

**What happened:**
`/okx-admin/heal` 這個 endpoint 在 06-07 08:26 自動把一筆 live SHORT 0.29（id=6，executor 02:00 正常開的）在 DB 裡標成 `status=CLOSED`（exit_reason=admin_heal），但**完全沒去 OKX 平倉**。09:02 對帳發現「OKX 有、DB active 沒有」→ `orphan_exchange` → executor DEMOTE，推了一條「MANUAL INTERFERENCE DETECTED」假警報（其實不是手動，是 endpoint 被自動觸發）。

時間線鐵證：08:02 對帳還 CONSISTENT（兩邊都有 SHORT）→ 08:26 admin_heal 抹 DB → 09:02 orphan_exchange。OKX 唯讀查證實 SHORT 還活著、強平價 $97K（離現價 +55%、碰不到）、stop algo 還 live @ 63148——所以**根本沒有風險事件，只有人造的狀態不一致**。

這是 **6/4 admin_heal 事件的第二次重演**（6/4 是 orphan_local，CLAUDE.md 記過）。第一次只處理了當次孤兒倉，沒根治 endpoint 本身，於是換個方向（orphan_exchange）又炸一次。

**Root cause:**
兩個疊加的設計缺陷：
1. **破壞性操作掛在無認證 GET**：`@app.route("/okx-admin/heal", methods=["GET"])`，要 `?confirm=YES` 才執行。問題是一個存過的 `.../heal?confirm=YES` 完整鏈接（Telegram 訊息/文檔/監控配置裡），會被 **link-preview bot / 瀏覽器預取 / uptime probe 連 query string 一起 GET**，自動帶 confirm=YES 觸發歸零。GET 依設計應幂等只讀，把「歸零 live 倉位」放 GET = 等著被預取誤觸。
2. **heal 只 UPDATE DB、不碰 OKX**：函數體全是 SQL（close DB rows + reset executor + resolve kill_log），注釋 L947 聲稱「Re-fetch positions from OKX via REST」但**代碼根本沒這段**。所以一旦 OKX 實際有倉，歸零 DB 必然製造 orphan_exchange。

**Correct approach（已修，commit 待 push）:**
1. **破壞性路徑改 POST-only**：`methods=["GET","POST"]`，但 `execute = request.method=="POST" and confirm=="YES"`。GET 永遠 dry-run，link-preview/預取（都是 GET）再也觸發不了歸零。
2. **歸零前先查 OKX，有倉則拒絕**：execute 前 `OkxRestClient.get_positions("BTC-USDT-SWAP")`，只要 OKX 還有非 FLAT 倉位就回 409 拒絕，附 OKX 倉位明細 + 「先平 OKX 再 heal」提示。heal 從此只能清真正的 orphan_local（DB 有 rows、OKX flat）。
3. 驗證：py_compile + `app.url_map` 確認 `/okx-admin/heal → okx_admin_heal_api [GET,POST]` 綁定沒脫鉤。

**Rule:** 任何會改變**真實交易所狀態 / 真錢**的 endpoint，**絕不可掛在 GET**——GET 必須幂等只讀，否則 link-preview / 預取 / 重載會在你不知情時觸發。破壞性 admin 操作 = POST + token + **執行前先核對真實外部狀態**（never zero local state that the exchange still holds）。修一個 ops bug 時，要修**機制本身**不是只清當次的髒數據——6/4 只清了孤兒倉沒修 endpoint，6/7 就用另一個方向重演。對帳出現 orphan 時，第一個問「是不是某個 heal/reset 工具只動了單邊（DB 或 exchange）」。

---

## 2026-06-02: aggregate AUC lift 被 2 個 outlier folds 撐起來，per-fold mean 是負的

**What happened:**
為了突破 V7 0.54 AUC ceiling，我跑 WorldQuant 101 alphas adapted for single-asset（rank → ts_rank），跑 conditional IC 找出 6 個強候選（alpha008/047/005/020/024/084，cond_IC > 0.03 + frac_pos > 65%）。然後用 production trainer (`train_direction_reg_walk_forward`) 跑 ensemble A/B：V7 baseline 136 features vs V7 + 6 alphas (142 features)。

**Aggregate 結果看起來 GO**：
- sign_AUC: 0.59755 → 0.60473 = **+0.00718**（剛過 +0.005 部署門檻）
- Strong thr=0.008 WR: 83% → 100%（6 笔全勝）
- Strong thr=0.010：新門檻達成 1 trade 100% WR

我寫了 verdict 文字「DEPLOY: WQ101 candidates bring measurable lift」，準備推 user 走 2 週 paper validation。差一步就 commit。

幸好部署前最後一個 sanity check：**per-fold AUC lift 分布**——只花 5 分鐘，但翻盤：
- Mean lift: **-0.00442**（負的！）
- Median lift: -0.00529（負的）
- Positive lift folds: **37/77 = 48.1%**（不到一半）
- Std: 0.091（極不穩）
- Worst fold: -0.318，Best fold: +0.279
- Capped mean (clip ±0.05): +0.00023（等同 0）
- Bootstrap 95% CI: [-0.026, +0.016]（含 0）
- Bootstrap p(lift ≤ 0): **0.666**（66% 機率根本沒 lift）

aggregate +0.0072 是被 1-2 個極端 fold（max +0.28）撐起來的。**Median 是 -0.0053**。

**Root cause:**
**Aggregate AUC 跟 per-fold mean AUC 是不同 metric**。
- **Aggregate**：把所有 fold 的 OOS predictions pool 起來再算一次 AUC
- **Per-fold mean**：每個 fold 各算 AUC 後平均

當有 1-2 個 fold 有極端 improvement（例如某段 quiet market 剛好 alpha008 抓到 momentum），會把 aggregate 拉高，但 per-fold 平均不變。Pooled metric 對 outlier 敏感，per-fold 才是真實 generalization 訊號。

更深問題：**conditional IC 顯著 ≠ ensemble A/B 過**。
- Conditional IC 量「**alpha 跟 V7 線性 residual** 的相關」
- Ensemble A/B 量「XGB 加 alpha 後**非線性 ensemble** 預測是否改善」
- 兩者可以背離：XGB 已透過 tree splits 非線性捕捉類似 pattern → 加 raw alpha 變成 noise

也就是說 conditional IC 顯著只證明「**alpha 帶 V7 沒有的線性訊息**」，但 XGB ensemble 可能透過 conditional split 隱式抓到了 → 加進去**反而 hurt**（看到 best fold +0.28 但 worst fold -0.32 = high-variance signal）。

放大因素：6/2 之前的 [[mistake 2026-06-01]] 已經建立了「conditional IC > raw IC 篩選」紀律，但**還缺一步 per-fold sanity**。我以為 aggregate 過了就 deploy，差點重蹈覆轍。

**Correct approach:**
任何 ensemble A/B 的 verdict 必須**同時看**：
1. **Aggregate lift > +0.005**
2. **Per-fold mean lift > +0.001**
3. **Frac_positive folds > 55%**
4. **Bootstrap 95% CI 不含 0**

4 條都過才算「真實 lift」，缺一就**疑似 outlier 撐起來的假 lift**。

具體實作：寫進 `wq101_ab.py` 之類的 A/B script 末段——

```python
fold_lifts = [auc(new_fold) - auc(base_fold) for fold in folds]
n_pos = sum(1 for x in fold_lifts if x > 0)
boot_p = bootstrap_p_value(fold_lifts, hypothesis="lift > 0", n=2000)

if (aggregate_lift > 0.005
    and np.mean(fold_lifts) > 0.001
    and n_pos / len(fold_lifts) > 0.55
    and boot_p < 0.05):
    verdict = "DEPLOY"
else:
    verdict = "NO-GO (aggregate may be outlier-driven)"
```

**Rule:** Ensemble A/B 看到 aggregate AUC lift 過門檻時，**強制再算 per-fold mean + frac_positive + bootstrap CI 4 條 sanity**。光看 aggregate 等於 [[mistake 2026-06-01]] 在升級版重演——只是這次「univariate IC 過」變成「aggregate AUC 過」，本質都是「**outlier 撐起 averaged metric 但 generalization 不行**」。Conditional IC 過只是「值得試 A/B」的 trigger，不是「值得 deploy」的證據；ensemble A/B aggregate 過也只是「值得 per-fold sanity」的 trigger，不是 deploy 證據。**驗證鏈條每加深一層都要重新 sanity check**。

更實務的紀律：**如果 5 分鐘的額外 check 能省下 2 週 paper validation，永遠先做這個 check**。今天這個 sanity 省下了：(a) 中斷現有 V7 paper cohort (b) 訓練 new model 等 1 小時 (c) 2 週 wait 然後發現沒差 (d) 浪費 14 天 paper baseline 比較性。**Validation discipline 的 ROI 是「上游 5 分鐘擋下下游 2 週的浪費」**。

**Update**: 證實 V7 對「OHLCV + Coinglass + Deribit + Binance order flow」這幾個 data source 已飽和。突破方向必須是**異源 channel**：(1) options gamma exposure (paid Deribit/Glassnode), (2) whale on-chain wallet flow (Glassnode), (3) Bitcoin ETF AUM/flow (CoinGecko 開放), (4) Twitter/Reddit sentiment (DIY scraper)。優先順序按「取得成本 vs 預期 lift」評估。

---

## 2026-06-01: walk-forward univariate IC 漂亮但加進 ensemble 沒 lift（feature redundancy）

**What happened:**
為驗證使用者「market moves to least resistance」的訂單流原則，我跑了一輪 walk-forward IC sweep（`research/liquidity_proxy_features.py`）。8 大類 21 個 microstructure proxy 特徵，做了 30d-train / 7d-OOS / 4-fold rolling 走勢驗證。結果非常漂亮：

- 12 個 feature 通過 |mean_IC| > 0.05 + 4/4 fold 同向
- 最強 `A_swing_high_dist_168h` mean_IC **+0.207**（V7 既有最強 feature ~0.07，看起來是 3x lift）
- 7 個獨立特徵（greedy de-dup |corr|<0.5）全部 4/4 同向

看起來非常有信心。於是寫了 A/B retrain script（`research/dual_model/train_with_liq_features.py`），用相同 XGB 超參數 + 77-fold WF split 比較「V7 baseline (136 features) vs V7 + 7 liq features (143 features)」。**結果：sign_AUC 從 0.5208 掉到 0.5178（-0.0030），IC 兩者都 ≈0**。

也就是說：univariate WF IC 看起來強的 feature，加進 ensemble **完全沒有 marginal information value**。

**Root cause:**
**Feature redundancy 在 XGBoost ensemble 裡是常見現象**。V7 的 136 個既有 feature（CVD divergence、OI delta、vol_kurtosis、impact_asymmetry、各種 z-score、return lag）已經透過 tree split 重組出類似 swing distance、sweep magnitude 的訊息。新加的 raw 特徵雖然 univariately 有訊號，但 **conditional on 既有 features 的訊號=零**。

更深的問題是我**只看 univariate IC 就下結論「這是 V7 強 3 倍的新 alpha」**。正確比較應該是「marginal IC given V7 model」— 也就是 V7 預測 residual 跟新 feature 的 IC。如果 V7 residual 跟新 feature 不相關，新 feature 對 V7 才有 lift。我這次直接用 raw IC 比較 V7 整體 IC，是 apples-to-oranges：raw IC 量「跟 target 相關」，但 V7 IC 量「ensemble 預測誤差」。一個 feature 可以很 univariately 相關但對 ensemble 全無 lift。

放大因素：walk-forward N=4 folds 太少。frac_positive=4/4 看起來很穩，但隨機 4/4 同向機率 = 1/16 = 6.25%。7 個獨立特徵全 4/4 同向是不太可能（聯合機率極低），但**每個獨立 feature 的 IC 估計值本身仍有大量 noise**。可能我看到的 +0.207 在更多 folds 之後會收斂到 +0.05 或更低 — 還是有 signal，但沒「3x V7」這麼誇張。

**Correct approach:**
1. **加新 feature 前永遠跑 ensemble A/B**，不是只看 univariate IC。Univariate IC 量的是「跟 target 的 raw 相關性」，ensemble 已經透過 tree split 吸了大半。要看 lift 必須是「加進去 ensemble 後 OOS AUC 是否提升 +0.005 以上」。
2. **若一定要用 univariate metric 做篩選**，用 **conditional IC**：先用 V7 baseline 預測，算 residual = y - pred，然後算新 feature vs residual 的 IC。Conditional IC 顯著 > 0 才值得進 ensemble。原始 IC 顯著只證明「跟 target 有關」，沒證明「V7 沒抓到」。
3. **WF fold 數 N < 10 時的「全 fold 同向」結論要打折**。N=4 同向看起來 4/4，實際統計強度約等於 binomial p=0.5 下 4 trials 全成功，p-value = 1/16 = 0.0625（剛過 5% 邊界）。要 N≥10 同向結論才篤定。
4. **負面結果一樣要記下來**，未來別人不會（或自己不會）重複跑同樣的 univariate IC sweep 結果 hyped。`research/orderbook_liq_features.py` 跟 `research/liquidity_proxy_features.py` 一起留作「univariate IC 高但 ensemble 沒 lift」的案例。

**Rule:** 任何「新 feature 加進 V7 / V8 ensemble」的決定必須基於 **ensemble A/B retrain 的 sign_AUC 或 IC lift**，不是 univariate WF IC。Univariate IC 高表示「跟 target 有 raw correlation」，但 conditional on ensemble 的剩餘 signal 才是真正的 marginal alpha。看到 univariate IC 比 V7 既有 feature 高 2-3 倍時 — **特別**要警覺，這往往是已經被 V7 吸收的訊息以另一種包裝出現。下次先跑「conditional IC vs V7 residual」 → 若顯著再 ensemble A/B → 都過才整合。

**Update 2026-06-02:** 重跑 A/B 用 production training function（`research/dual_model/rerun_liq_ab_with_prod_trainer.py`，import `train_direction_reg_walk_forward` 直接）驗證上面結論：BASELINE V7 sign_AUC 0.6030 / IC 0.180（跟 canonical OOS 0.593/0.170 對齊），NEW V7 + 9 liq features sign_AUC 0.6036 / IC 0.186 — **+0.0006 AUC、+0.006 IC**，仍遠低於 +0.005 部署門檻。原始結論「不要部署」**仍然成立**，但要注意：上次第一版 A/B baseline 訓練設定有差（custom eval_set 早停太凶導致預測退化），所以兩個 broken model 之間「無 lift」的觀察方向對，但**比較的絕對值都是錯的**。下次 A/B 要**直接 import 生產訓練函式**避免 hyperparam drift。

---

## 2026-05-31: Edit 把新函式塞進 `@app.route` 跟 `def webhook()` 之間，decorator 被靜默搶走，Telegram bot 全死

**What happened:**
commit c758336 加 `_handle_okx_perf` 函式時，我用 Edit 工具改 BTC_perp_data.py 的 old_string = `def _handle_okx_approval_response(...):`，new_string = `def _handle_okx_perf(...): ...\n\n\ndef _handle_okx_approval_response(...):`。結果新函式被插入到 `_handle_okx_approval_response` 之前。

`_handle_okx_approval_response` 本來就是我之前（commit e531b2c）用同樣手法插在 `def webhook():` 之前的——那一次也是把 webhook 上方的 `@app.route(f"/{TOKEN}", methods=["POST"])` decorator 跟 `def webhook()` 拆開了，但因為 `_handle_okx_approval_response` 的 signature 是 `(chat_id, raw_cmd)`，Flask 路由把 POST 進來時 Werkzeug 報 "TypeError: missing argument" → 變成 500 給 Telegram。**那次沒爆只是因為 Telegram 平常不會故意打 webhook 來驗證，Flask app 也沒在啟動時報錯**。直到我這次再插一個 `_handle_okx_perf` 在更前面，decorator 又被搶過去——這一次完全相同的問題終於在用戶按 V7 Stats 按鈕時暴露。

症狀很迷惑：bot service 的 `/` 主頁回 200 「OKX BTC Liquidity Outcome Bot is running」(因為 `/` 的 decorator 跟 def 是黏在一起的，沒被動到)，但 Telegram getWebhookInfo 顯示 `last_error_message: "Wrong response from the webhook: 500 Internal Server Error"`，每個指令、每個按鈕都死，**包括完全沒碰過的 /help**。用戶看起來就是「bot 沒反應」，沒有任何錯誤訊息能讓他自己 diagnose。

Python 語法檢查、import 檢查、unit test 全都過——因為這個 bug 是「decorator 綁錯函式」，不是任何 lint 工具會抓的。要等到 HTTP request 真的進來、Flask 帶錯誤參數 call 那個函式，才會炸。

**Root cause:**
我把 `def webhook():` 之前的某行（裡面有獨立函式 `_handle_okx_approval_response` 或 `_handle_okx_perf`）當成 anchor 點插入新函式，沒注意到該函式緊鄰 `@app.route(...)` decorator，而 Python decorator 是 syntactically 綁到「decorator 下面那一個 def」上的——我的 Edit 把新 def 塞在中間，等於把 decorator 從 `def webhook` 拔走、轉嫁到我新插的 def 上。

更深層原因：Flask 的 `@app.route` 沒有「綁定檢查」——decorator 隨便綁到哪個函式它都不會 raise，只是綁的那個函式變成 endpoint。Runtime 在 Telegram 打過來、Flask 用沒給 chat_id 參數的方式 call 它時才出 TypeError。再加上 Flask app 對 unhandled exception 的預設行為是回 500 給 client，沒任何 startup-time 警報。

放大因素：我寫 commit message 的時候 grep 路徑下的 `@app.route` 看其它路由還在，但沒去看每個 decorator 是不是綁到「原本應該的 def」上。 sanity check 是「routes 都存在」而不是「routes 都綁對函式」。

**Correct approach:**
1. 任何 Edit 動到「Flask / Django route 檔案中、靠近 `@app.route` 或 `@app.get` 之類 decorator 的位置」，必須**讀 Edit 後的整段** 至少 ±10 行，確認 decorator 跟原本的 def 仍然黏著。
2. 加新 helper 函式時，**找一個遠離 route handler 的安全位置**插入。例如：放到檔尾、或一個獨立的 `# === Helpers ===` 區塊。不要見縫插針地塞在現有 route 旁邊。
3. push 前如果改了 Flask app 檔案，用 `python -c "from BTC_perp_data import app; print([str(r) for r in app.url_map.iter_rules()])"` 檢查所有 route 跟 view function 的對應關係。`url_map` 印出來會看到 `<Rule '/<token>' (POST) -> webhook>`，如果看到 `-> _handle_okx_perf` 就是綁錯了。
4. Bot service 應該有一個 startup smoke——例如「啟動後對自己 webhook POST 一個空 JSON，預期回 200 ok」，啟動失敗的話 Railway 部署應該直接 fail，而不是部署成功讓 silent failure 跑半天。

**Rule:** 用 Edit 工具改 Flask / Django / FastAPI route 檔案時，**絕對不要把 anchor 點選在 `@app.route` decorator 下方那行**。如果一定要插入，要連同 decorator 一起包進 old_string，或選擇遠離任何 decorator 的位置（例如 helper section、檔尾）。Edit 後務必目視確認每個 decorator 還黏在原本的 view function 上。Python decorator 跟 def 沒有任何語法保護，綁錯了 lint/syntax/import 都不會炸，只在 runtime 有人打 endpoint 時才以「500 + missing positional argument」現身。**Symptom 是「Flask 主頁活著但某個 route 全死」、「Telegram 顯示 500 但 code 看起來沒問題」**——下次看到這種模式，第一個查 `app.url_map`。

---

## 2026-04-22: 新特徵邏輯貼進錯誤的 helper 函數，signature 不符導致 NameError 整夜停機

**What happened:**
b604afc commit 新增 SPX / DXY / US10Y / FNG 等 cross-market 特徵時，把一整段使用 `cross_market` 和 `fear_greed` 變數的程式碼**除了貼進 `build_live_features()`（正確位置，signature 有這兩個參數），又同時重複貼進 `_inject_coinglass(df, cg_data)` 這個 helper 函數裡**。後者的 signature 根本沒有這兩個參數。

commit 後不會在 import / 語法檢查階段報錯——因為 `_inject_coinglass` 只在 runtime 被呼叫，而且只有執行到 `if cross_market:` 那行才丟 `NameError: name 'cross_market' is not defined`。Railway 部署成功（build 綠），Process 啟動成功，但每次 `update_cycle()` 跑到該行都 crash。外層 `try/except` 把錯誤吞成 `_state["status"] = "error"`，整個下游（predict、render chart、Telegram 推送、寫 MySQL）全部靜默停擺。

結果：23:00 push 生效 → 到隔天 09:00 用戶打 `/indicator-status` 看到 `error: name 'cross_market' is not defined` 才發現，整整 10 小時圖表沒有新 bar。

**Root cause:**
加新特徵時沒有確認「我把這段貼進的函數，它的 signature 有沒有我要用的變數」。編輯器的 copy-paste / multi-insert 很容易一次改多處，中間的某一處 paste 點如果剛好在錯誤的 helper 函數裡，本機的靜態檢查（`python -c "import module"`）抓不到——只有 runtime 執行那條分支才會炸。push 前沒跑任何會觸發 `update_cycle()` 的本地測試。

另一個放大因素：`update_cycle()` 外層 `try: ... except Exception as e: _state["status"] = "error"` 把錯誤吞太深，只在 `/indicator-status` 才看得到；沒有任何 alert 機制通知「Railway 進程活著但內部邏輯壞了」這種 **silent failure**。Railway build 綠 + process 活著 = 用戶預期功能正常，但實際上功能靜默死亡。

**Correct approach:**
1. 任何 commit 如果碰到 `indicator/feature_builder_live.py`、`indicator/app.py` 的 `update_cycle()`、`indicator/inference.py` 等 hot-path 檔案，push 前必須至少跑一次 `python -c "from indicator.app import update_cycle; update_cycle()"` 或 `/force-update?sync=1` 的本地版。import 成功不代表 runtime 成功。
2. 加新變數到某段邏輯時，先 grep 確認自己貼進的那個 `def` 的 signature 到底有沒有這個變數。如果沒有，**不是** `def` 漏了參數（先問自己「這段邏輯是不是根本不該在這個函數裡」），而是貼錯函數了。
3. Silent failure 的監控：`update_cycle()` 外層 except 應該把 last error 暴露到一個更顯眼的健康指標（不是只有 `/indicator-status`），並且觸發 Telegram 告警——「Railway 活著但 update_cycle 連續 N 次 error」應該被當成 critical。這個後續要補。

**Rule:** 碰 hot-path 檔案（`feature_builder_live.py` / `app.py` `update_cycle` / `inference.py`）的 commit，push 前必須跑一次真實的 `update_cycle()` 或 `/force-update?sync=1`。**import OK 不代表程式碼會跑**——Python 的 NameError 只在執行那條分支時才炸。加新 feature code 時，grep 貼入處的 `def ...():` signature，確認引用的所有外部變數都在 signature 裡或在該 scope 可見。Silent failure（process 活著但邏輯死了）是最危險的故障模式，因為 Railway dashboard 看起來全綠。

---

## 2026-04-14: 把 Strong 勝率目標寫成 95%（從策略系統沿用未更新）

**What happened:**
CLAUDE.md 長期寫「Strong 信號勝率目標 > 95%」，花了一整天嘗試各種方法提升 Direction model 都碰天花板，才回頭檢查這個目標本身是否合理。跑 `research/topk_precision_sweep.py` 用 2726 筆 walk-forward OOS 預測做 bidirectional top-k：

| k | precision | CI | signals/月 |
|---|---|---|---|
| top 1% | 59.3% | [40.7, 75.5] | 5 |
| top 2% | 63.6% | [50.4, 75.1] | 11 |
| **top 5%** | **67.6%** | **[59.4, 74.9]** | **27** |
| top 10% | 65.6% | [59.8, 71.0] | 53 |
| top 20% | 60.2% | [56.0, 64.2] | 106 |

峰值 67.6%。AUC 0.57 的理論 top-5% precision 上限 68-72%，代表**已經貼著數學天花板**。95% 在這個 AUC 結構下永遠達不到，那是 AUC 0.80+ 的模型才談的數字。

**Root cause:**
95% 是從早期策略系統（有 TP/SL、能過濾掉不利情境）沿用到指標系統，沒人重新校準。而指標系統的訊號是「原始預測」，不過濾，所以上限直接由模型 AUC 決定。把策略目標搬到指標系統等於給自己設一個數學上不存在的目標。

**Correct approach:**
precision 目標必須從**模型 AUC 反推**，不是拍腦袋定。公式約等於：
- 給定 AUC，top-k precision 上限 ≈ 0.5 + (AUC - 0.5) × kernel(k)
- AUC 0.57 + k=5% → 理論 ~0.70，實測 0.676（非常接近）
- 如果要求 0.95，需要 AUC ≥ 0.85

現在 CLAUDE.md 已改為「point estimate ≥ 65%，stretch 70%」。未來任何討論 Strong 勝率時，第一句話要問「當前模型 AUC 是多少，這個目標在結構上可達嗎」，不是「為什麼還沒達到」。

**Rule:** 設定任何 precision/recall 目標前，先用當前模型的 AUC/IC 反推理論天花板。如果目標高於天花板就是錯的目標，改目標而不是追目標。絕對不要從不同系統（策略 vs 指標）沿用績效目標——運作機制不同，天花板也不同。

---

## 2026-04-13: 用 in-sample 月份 IC 判斷訊號健康度（高估 0.5 AUC 級別）

**What happened:**
為了診斷 Magnitude IC 衰退，寫 `diagnose_mag_decay.py` 用**當前生產模型**去預測過去每個月的 `|ret_4h|`，得到 Nov 0.60 / Dec 0.51 / Jan 0.57 / Feb 0.53 / Mar **0.60** / Apr 0.47。看起來訊號完全沒衰退、近月甚至還很強，幾乎下結論「Mag 訊號穩定，問題不在這」。

後來跑乾淨 walk-forward（`mag_level_feat_swap.py`，每個測試窗只用之前的資料訓練）得到真實 OOS IC：Nov 0.31 / Dec 0.36 / Jan 0.24 / Feb 0.20 / Mar **0.10** / Apr 0.12。**Mar 差距 0.50 IC，Apr 差距 0.35 IC**。真實情況是 Mag 從 Feb/Mar 交界發生 concept drift，IC 腰斬。

也就是說，我的第一版診斷**用了訓練集預測訓練集**——生產模型訓練時吃了全部 4000 bars，對任何歷史月份做預測都是 in-sample，結果無法反映 model 是否能從歷史學到新規律。

**Root cause:**
沒區分「model fit」和「model generalization」。生產模型的 IC 是在**全部資料訓練完**才算的，拿它去預測過去月份天生是作弊。這跟 Kaggle 新手用 `cross_val_score` 之後又用全資料重訓再看 train loss 是同一個錯誤——只是換了個包裝。更糟的是，月份切片讓我以為這是「time-slicing 驗證」，實際上完全沒做 time-based split。

**Correct approach:**
任何「模型是否仍然 work」的評估都必須是**嚴格 walk-forward**：每個測試點的模型只能看到該點之前的資料。生產模型的 in-sample 預測**永遠不能拿來回答「訊號是否衰退」「特徵是否還有效」「regime 是否改變」這類問題。能用 in-sample 回答的問題只有：「訓練收斂了沒」「在完整資料上 model 的 upper bound 在哪」。

**Rule:** 診斷 IC/AUC 衰退時，第一句 assert 必須是「這個預測是 walk-forward 還是 in-sample」。in-sample 的結果在「診斷衰退」這個 task 下**零資訊量**，不管數字多漂亮都等於沒測。如果檢查清單裡的測試方法是「用生產模型預測過去月份」，那就是錯的測試方法，換掉。

---

## 2026-04-13: Regime-specific 子模型在小樣本下退化成比隨機還差

**What happened:**
為了試圖突破 Direction model 天花板，訓練 bull/bear/chop 三個 regime-specific 子模型（`regime_specific_direction.py`）。假設是「每個 regime 的特徵→方向關係不同，獨立訓練應該贏過全局模型」。

結果全局模型在三個 regime 上的 AUC 分別是 CHOP 0.548 / BULL 0.500 / BEAR 0.497，regime 子模型是 CHOP 0.550 / BULL 0.440 / **BEAR 0.378**。BEAR 子模型 AUC 顯著低於 0.5，意味著它**系統性預測反方向**。

原因：BEAR 整個 4000 bar 資料集只有 724 筆，扣掉 walk-forward test + NaN，每個 split 的訓練集只剩 50-100 筆。XGB 在這個樣本數下嚴重 overfit 訓練集的雜訊方向，預測出來的機率跟實際 label 反相關。BULL 也有 16 個 split 因為訓練樣本不足 < 50 直接跳過，等於是選擇性覆蓋。

**Root cause:**
沒評估「資料切片後每個 regime 的有效樣本數是否夠訓練」。少於 ~500 的小樣本訓練 gradient boosting 會 overfit 到雜訊，而且資料越少 overfit 越嚴重，甚至可能學到完全相反的方向。把這當成「子模型比較弱」來解讀是錯的——這些子模型根本沒進入「能學東西」的 regime。

**Correct approach:**
切片訓練前先算 min(regime_sample_count) 是否 > 500（gradient boosting 的大略安全線）。如果不夠：
1. **退一步用 regime dummies 當 feature** 讓全局模型自己學 conditional split（這是 XGB 設計本來就能處理的）
2. 或用 `sample_weight` 在全局訓練時加權少數 regime，**不要**切開訓練
3. 如果真要切，只切樣本充足的 regime（這個資料集只有 CHOP 有 2000+ 筆，結論是：沒得切）

**Rule:** 分群訓練前，每群的有效訓練樣本數必須 > 500（至少要 > 300），否則不如用單一模型 + 分群特徵。小樣本下 gradient boosting 不會變「局部專家」，會變「雜訊放大器」。如果樣本數不夠，把分群改成 feature 而不是改成 partition。

---

## 2026-04-13: 用混合模型版本的數據下 calibration 結論

**What happened:**
跑 `calibration_check.py` 看到 Brier skill -0.098、ECE 0.16、over-confident +0.197，bootstrap CI 全部顯著（[-0.184, -0.014] 整條在零下，conf_gap [+0.115, +0.285] 整條在零上），就據此推論「模型 miscalibration 是真的」並開始討論 Platt scaling / isotonic / rolling percentile threshold 等解法。

然後往下挖才發現 244 個 valid 樣本全部來自 2026-04-02 → 2026-04-12 這 10 天，這個窗期：
  - 2026-04-03 部署 dual v7 初版（88 特徵）
  - 2026-04-09 切換成 pruned 29 特徵 + regime weighting
  - 2026-04-12 又重訓一次
  - 5.5 天 `cg_bfx_margin_ratio`（第 4 重要特徵）灌壞數據（2026-04-12 backfill bug）

也就是說：calibration 測試基於 **三個不同模型的混合預測 + 重大特徵被污染一半時間**。bin-level 極端區的怪象（p≥0.70 actual=0.50）很可能只是模型切換那幾小時產的 transient，不代表任何一個模型的穩態。前面提出的所有解法都建立在錯誤的前提上。

**Root cause:**
看到統計顯著的壞結果就急著找解法，沒先問「這個測試數據對應的是哪個模型？數據本身是乾淨的嗎？」最基本的 data sanity check 被跳過了。更糟的是 bootstrap CI 讓我更有信心下結論——但 CI 只能量**抽樣不確定性**，量不到**數據污染**或**模型版本混合**這種系統性偏差。統計顯著 ≠ 結論可信。

**Correct approach:**
評估模型前必須確認：
  1. **樣本範圍對應單一模型版本**：git log 查最新模型 deploy 時間，樣本必須在那之後。
  2. **樣本範圍不含已知數據污染窗**：查 mistake log 看近期有沒有數據 bug。
  3. **樣本數夠**：即使資料乾淨，n<100 的 calibration 點估計不穩定；n<500 做 isotonic 會 overfit。
  4. **先看時間切片**：分月/分週跑同一個測試，如果每段結論都不同，整體測試就沒意義。

已在 `calibration_check.py` 的 roadmap 加上 `--since` flag 和 model version guard（讀取最新 model mtime，樣本必須 >= 該時間），還沒實作。

**Rule:** 評估任何模型的統計量前，第一件事是「確認這份評估樣本是從同一個模型 + 同一份乾淨數據產生的」。這個 sanity check 要**在看結果之前**做，不是看到壞結果才回去查。Bootstrap / permutation / 顯著性檢定全部都只能處理抽樣誤差，不能處理「你在量錯的東西」這種問題。看到「顯著的壞結果」第一反應應該是懷疑測試設計，不是懷疑模型。

---

## 2026-03-28: price_change fallback over-engineering

**What happened:**
Item 9 (fix `_get_price_change` dependency on normalized_trades) was implemented with 3 chained queries:
1. Query flow_bars_1m to find nearest bar
2. Query normalized_trades within that bar's time window
3. Fallback to delta/volume estimation

Step 1→2 was pointless — querying normalized_trades scoped to a flow_bar window is the same as querying it directly. This tripled the DB load per snapshot for no benefit.

**Root cause:**
Jumped to a "clever" solution without thinking about whether the intermediate step added value. flow_bars_1m doesn't store price, so using it as an index to find normalized_trades was a round-trip to nowhere.

**Correct approach:**
1. Try normalized_trades first (works for events < 3 days, same as original)
2. Only if no data, fallback to delta/volume ratio from flow_bars_1m

**Rule:** When adding a fallback path, ask: "Does this intermediate step give me information I don't already have?" If not, skip it. Prefer the simplest query chain that solves the problem. Don't add queries that increase Railway DB usage without clear value.

---

## 2026-03-29: delta/volume ratio ≠ price change

**What happened:**
`_get_price_change()` fallback used `total_delta / total_vol * 100` when normalized_trades had no data. This produced values like +4.84% that looked like real price moves but were actually the **taker imbalance ratio** (what % of volume was net buy).

**Root cause:**
Confused two different metrics. delta/volume ratio measures buy-sell pressure, not price movement. The two are correlated but not interchangeable — especially on short windows where slippage is minimal.

**Correct approach:**
Return None when normalized_trades has no data. Don't fabricate price estimates from flow data.

---

## 2026-04-01: 把 MACD / EMA 放進訂單流研究的特徵集

**What happened:**
`feature_builder_v2.py` 計算了 `ema_9`, `ema_21`, `macd`, `macd_signal` 並寫入 features 表。這些欄位出現在 feature validation 結果中，ICIR 看起來很高（-0.85~-0.91），但這根本不該存在。

**Root cause:**
誤把傳統技術指標混入訂單流研究。這個專案的研究範疇是純訂單流（CVD、delta_ratio、aggTrade flow、funding rate、OI），不包含 price-derived 的技術指標如 MACD / EMA / RSI 等。

**Correct approach:**
feature_builder_v2.py 只能包含以下來源的特徵：
- aggTrade flow（CVD、delta_ratio、buy/sell vol、large order）
- Funding rate（rate、deviation、zscore）
- OI（未來補充）
- Cross-exchange divergence
- 純統計衍生（realized vol、return lags）— 可接受，因為是 price behavior 而非 pattern indicator

MACD / EMA / Bollinger 等技術指標一律不加。

**Rule:** 每次加新特徵前先問：「這是訂單流資料還是技術指標？」技術指標一律排除。

---

## 2026-04-02: 加 log 行導致 webhook 500 crash

**What happened:**
在 `indicator/app.py` 的 `/webhook` handler 中加了一行 `logger.info("Webhook command: %s", cmd, chat_id)`，但放在 `cmd = text.split()[0]...` 定義**之前**。導致每次收到 Telegram 指令都觸發 `NameError`，回傳 500，用戶的 `/chart` 指令完全無反應。

**Root cause:**
加 debug log 時沒注意變數的定義順序。修改生產環境的 request handler 後沒有做基本的 code review（變數是否已定義）。

**Correct approach:**
新增的 log 行必須放在所有引用變數的定義之後。修改 webhook/route handler 這類每個請求都會跑的代碼時，要特別小心：一個 crash 會影響所有用戶。

**Rule:** 在生產 handler 中加 log 或任何代碼後，立刻檢查：所有引用的變數是否已定義？是否在 try/except 內？不要假設「只是加一行 log」就不會出錯。

---

## 2026-04-12: backfill 時間戳 unit 硬編碼導致 5.5 天數據缺口

**What happened:**
`research/backfill_all_parquet.py` 的 `to_1h_df()` 用 `pd.to_datetime(df[time_col], unit="ms")` 硬編碼毫秒。但 Coinglass API 的 `coinbase_premium` 和 `bitfinex_margin` 端點回傳的 `time` 欄位是 **10 位秒級時間戳**（如 `1775998800`），不是 13 位毫秒。秒級時間戳被當毫秒解析後變成 1970 年日期，merge_parquet 沒報錯（index dedup 保留了壞行），最終 4131 行壞數據 + 數據停在 2026-04-07。

`cg_bfx_margin_ratio` 是剪枝模型第 4 重要的特徵，如果下次訓練前沒發現這個缺口，模型會在該特徵上訓練出偏差。

**Root cause:**
假設所有 Coinglass 端點的時間戳格式一致。實際上大部分端點用 13 位 ms，但 `coinbase_premium` 和 `bitfinex_margin` 用 10 位 s。生產端的 `data_fetcher.py` 早就有 `if ts.max() > 1e12` 的自動偵測，但 backfill 腳本是另外寫的，沒抄這段邏輯。

**Correct approach:**
時間戳解析永遠用自動偵測：`unit = "s" if sample_ts < 1e12 else "ms"`。已修復。

**Rule:** 凡是解析時間戳的地方，永遠不要假設 unit 固定。寫新的數據處理腳本時，先看生產代碼怎麼處理同一個 API 的格式。同一個 API provider 的不同端點可以有不同的時間戳格式。

---

## 2026-04-12: is_stale() 只檢查 klines 導致端點級故障無聲

**What happened:**
`backfill_all_parquet.py` 的 `is_stale()` 只讀 `binance_klines_1h.parquet` 來判斷是否需要回填。klines 永遠是最新的（Binance 公開 API 不需要 key），所以即使 CG 端點已經停滯 5.5 天，`ensure_fresh()` 也不會觸發回填。訓練管線 `shared_data.py` 調用 `ensure_fresh()` 時以為數據是新的，實際上 coinbase_premium / bitfinex_margin 缺了 132 小時。

**Root cause:**
用最穩定的數據源（Binance klines）代表所有數據源的新鮮度。這是一種「以偏概全」的監控盲區 — 最不可能故障的組件被選為健康指標。

**Correct approach:**
`is_stale()` 改為遍歷所有 parquet 文件，任何一個超過 max_age_hours 就回傳 True。已修復。

**Rule:** 新鮮度 / 健康檢查必須覆蓋最脆弱的組件，不是最穩定的。如果系統有 N 個數據源，健康檢查要查 N 個，不是只查最可靠的那一個。

---

## 2026-04-12: 用錯 IndicatorEngine 屬性名（dir_model vs dual_dir_model）

**What happened:**
watchdog 新增的 `_check_dual_model()` 檢查 `engine.dir_model` 和 `engine.mag_model` 是否為 None。但 dual mode 下的屬性名是 `dual_dir_model` 和 `dual_mag_model`。`dir_model` 是舊 regime mode 的屬性，dual mode 下根本不存在，導致 `AttributeError`。

**Root cause:**
沒有先 grep 確認屬性名就寫代碼。`IndicatorEngine` 有三種 mode（dual/regime/legacy），每種 mode 的屬性名不同。

**Correct approach:**
寫監控代碼前先 `grep self\.dual_dir` 確認屬性名。已修正為 `dual_dir_model` / `dual_mag_model` + `hasattr` 防禦。

**Rule:** 引用物件屬性前先 grep 確認。特別是有多種初始化路徑的類別（如 IndicatorEngine 的 dual/regime/legacy），不同路徑設定的屬性名不同。不要憑記憶猜。

---

## 2026-04-13: 用 sparse indicator 做 feature interaction 是退化操作

**What happened:**
為了解決 Direction Model 的 regime 適應性問題，原本想加 9 個 regime interaction 特徵：
```python
oi_agg_close_x_bear = cg_oi_agg_close * is_bear
bfx_margin_x_bull   = cg_bfx_margin_ratio * is_bull
ls_ratio_x_bear     = cg_ls_ratio * is_bear
# ... 等等
```
寫法看起來完全合理，是 ML 教科書經典 interaction term 寫法。

跑 IC 驗證後發現怪事：4 個本質完全不同的金融特徵在 ×is_bear 之後互相相關 0.96-0.98：
```
bfx_margin_x_bear ↔ oi_agg_close_x_bear     corr = +0.984
bfx_margin_x_bear ↔ ls_ratio_x_bear         corr = +0.957
oi_agg_close_x_bear ↔ ls_ratio_x_bear       corr = +0.968
```
而且 IC 全部從 base 的 -0.05~-0.07 掉到 +0.01，p-value 變不顯著，train/test FLIP。

**Root cause:**
BEAR 只佔 18% 樣本（724/4000）。`feature × is_bear` 等於：
- 非 BEAR 時 = 0（佔 82%）
- BEAR 時 = feature 原值（佔 18%）

問題在於**「在哪些 timestamp 是 0」這個 sparsity pattern 在所有 ×is_bear 特徵裡完全一樣**。所有特徵共享同一組 18% 的非零 mask。剩下 82% 的零值貢獻了大部分變異數。

結果 spearman correlation 主要在量「這個 sample 是不是在 BEAR 期間」，而不是「這個 feature 在 BEAR 的時候值是多少」。三個 base 完全不同的特徵看起來幾乎一模一樣，因為它們的 0/非0 pattern 完全重疊——indicator 的 sparsity 訊號壓過了被乘的特徵本身。

**Correct approach:**
1. **乘以 `(1 - is_X)` 才有意義**：保留 80%+ 樣本，只把死掉的 regime 設 0。例如 `vol_kurt_non_bear = vol_kurtosis * (1 - is_trending_bear)`，IC validated +0.054 stable, `oi_8h_non_bull = cg_oi_close_pctchg_8h * (1 - is_trending_bull)` IC validated -0.071（比 base -0.062 強 15%）。
2. **regime indicator 本身要直接當 feature 加進去**（is_trending_bull / is_trending_bear），讓 XGB 自己用 tree split 決定 conditional rule。手動寫 `feat × is_X` 是把訊號塞進更窄的 channel。
3. **inter-feature correlation matrix 必須當成標準驗證步驟**，跟 train/test split、rolling IC 同等重要。如果一群本應獨立的特徵互相相關 > 0.9，那不是訊號，是 indicator pattern leakage。

**Rule:** 設計 interaction feature 時，**永遠不要寫 `feat × sparse_indicator`**——當 indicator 的非零比例 < 30%，乘出來的特徵會被 sparsity pattern 主導，跟其他用同一 indicator 乘的特徵高度相關，IC 也會 collapse。如果要做 regime conditioning：(a) 把 indicator 直接當 feature，讓 XGB 自己學 split；(b) 只在「base feature 在某 regime 完全死掉」的情況下用 `feat × (1 - is_dead_regime)` 形式屏蔽噪音。設計完任何 interaction 都要先跑 inter-feature correlation matrix。

---

## 2026-04-19: 多個腳本覆寫同一個 JSON 導致 warmup buffer 被清空

**What happened:**
系統連續產出大量 DOWN 信號，比例明顯不合理。排查後發現 `training_stats.json` 裡的 `dir_pred_history`（Direction model 的 500 筆 warmup 預測）是空的。沒有 warmup buffer，系統永遠用固定 fallback 閾值解碼方向——這些閾值是歷史均值，無法適應當前 bearish regime，結果只要模型預測稍微偏負就觸發 DOWN。

事件鏈：
1. 4/15 `export_direction_reg_model.py` 正確寫入 500 筆 `dir_pred_history` ✅
2. 4/16 `deploy_new_models.py` 重訓 Magnitude model，用 `json.dump` **整個覆寫** `training_stats.json`，只寫了 `pred_history`（mag 的 warmup），`dir_pred_history` 被洗掉 ❌
3. 之後每次 Railway 重啟（git push 觸發），buffer 歸零，永遠不到 100 根 warmup 門檻
4. 系統永遠用 fallback 閾值 → bearish 市場下 DOWN 信號爆量

**Root cause:**
三個腳本寫同一個檔案，但寫法不一致：
- `export_direction_reg_model.py`：先讀再寫（read-then-update）✅
- `deploy_new_models.py`：直接 `json.dump` 覆寫 ❌
- `export_production_models.py`：直接 `json.dump` 覆寫 ❌

後面兩個腳本沒有意識到這個檔案是**共用的**，裡面有別的腳本存的資料。這是最基本的共用資源協調問題。

**Correct approach:**
寫入已存在的 JSON/config 檔案時，永遠用 read-then-update 模式：
```python
if stats_path.exists():
    with open(stats_path) as f:
        stats = json.load(f)
else:
    stats = {}
stats["my_key"] = my_value  # 只更新自己負責的 key
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)
```

額外加了兩層防護：
1. `app.py` 每次 update cycle 結束後持久化 `dir_pred_history`，這樣 Railway 重啟不會失去已累積的 warmup
2. 修復了 `deploy_new_models.py` 和 `export_production_models.py` 的寫法

**Rule:** 寫入任何共用檔案前，第一步是 `grep` 看還有誰也在寫這個檔案。如果有多個寫入者，必須用 read-then-update 模式，只動自己負責的 key。直接 `json.dump` 覆寫整個檔案等於對其他寫入者說「你存的東西我不在乎」——這在單一寫入者時沒問題，多個寫入者時是資料刪除。

---

## 2026-04-19: 用 WF OOS fold 模型的預測初始化 rolling percentile buffer（分佈差 3.5 倍）

**What happened:**
修復 buffer 被覆寫的問題後，重新 seed `dir_pred_history` 時，從 walk-forward OOS parquet 的 `pred_ret` 欄位取了 500 筆預測作為 warmup buffer。部署後圖表上幾乎**所有 bar 都是紅色 DOWN 三角形**。

排查發現：WF OOS fold 模型的預測 std=0.0008，但生產模型（用全部資料訓練）的預測 std=0.003，**差了 3.5 倍**。用小範圍的 buffer 去校準大範圍的預測，rolling percentile 的 DOWN 門檻大約在 -0.0006，而生產模型的正常預測值動輒 -0.002~-0.005，幾乎所有 bar 都超過 DOWN 門檻 → 全部是 DOWN 信號。

**Root cause:**
WF OOS 的 `pred_ret` 是每個 fold 的子模型產生的，每個 fold 只用部分資料訓練。子模型因為訓練資料少，學到的 pattern 弱，預測值集中在零附近，variance 小。生產模型用全部資料訓練，學到更多 pattern，預測值的 variance 明顯更大。

這是 walk-forward 驗證的根本特性：fold 模型和生產模型的預測分佈**不在同一個尺度**。拿 fold 模型的輸出去校準生產模型的閾值，等於用錯誤的尺去量。

事件鏈：
1. 第一次修 buffer：從 WF OOS parquet 取 500 筆 `pred_ret`，buffer std=0.0008
2. 部署後生產模型預測 std=0.003，幾乎所有預測都落在 buffer 的極端尾部
3. Rolling percentile 把正常預測解碼成 Strong DOWN
4. 圖表全紅，Telegram 每根 bar 都推 DOWN 信號

**Correct approach:**
每次重訓方向模型後，用**生產模型本身**在訓練資料上跑一次 predict，取最後 500 筆作為 `dir_pred_history`。同時用全部預測更新 `direction_reg_config.json` 的 fallback thresholds（2.5%/7.5% 分位數）。

驗證方式：比較 buffer std 和生產模型最近 200 筆的 std，ratio 應在 0.5~2.0 之間。最終修復後 ratio = 0.74x，信號分佈回到 5.5% UP / 88% NEUTRAL / 6.5% DOWN，符合 ~10%/80%/10% 的設計目標。

**Rule:** Rolling percentile buffer 的初始化**只能用生產模型的預測**，不能用 WF OOS fold 模型的預測。WF OOS 的預測只能拿來評估模型泛化能力（IC、AUC），不能拿來校準生產閾值——它們的分佈不在同一個尺度。每次 seed buffer 後，必須比較 buffer std vs 生產模型 std，ratio 偏離 0.5~2.0 就是 red flag。
