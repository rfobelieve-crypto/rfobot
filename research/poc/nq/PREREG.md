# PREREG — NQ / MNQ 強制流延續移植

**路徑**：把加密的 sweep + 強制流研究搬到 CME 指數期貨，只換 Stage 0（資料層）
**日期**：2026-09-10
**簽名**：本檔 commit 時間戳即簽名；commit 早於任何 NQ 標籤被計算

---

## 0. 這份東西要回答什麼

**不是找一條新策略，是回答一個機制問題**：強制流延續是加密特有的
（清算引擎驅動），還是有槓桿與停損的市場共通的？

第二個市場的答案比第一個市場的 PASS 更值錢 —— 它換的是**微結構**不是
時間段，那是 walk-forward 做不到的檢驗。

成本背景：NQ/MNQ 來回約 1–2 bps，BTC 是 8 bps。加密這邊每一條線都差
2 bps 過關。

---

## 1. 假設

### H_main（兩側分開判）

    旗標開火的 sweep 事件，後續延續報酬 > 0，扣成本後 CI 下界 > 0

母體定義沿用加密的 SDV：sweep 事件 ∩ 強制流旗標開火。

### H_asym（這次移植的核心，必須在看任何 NQ 標籤之前寫死）

    NQ 的不對稱方向與加密**相反**：sellside 較強

加密的不對稱是 **buyside 較強**（fade 上方極端 +115 bps / 70%，
fade 下方 +23 / 46%），機制解釋是零售槓桿多頭偏置 —— 上方極端 =
多單被清算 = 被迫、會回。

指數期貨的部位結構相反：正漂移、機構逢低買、下方擺盪低點的停損由
機構流接走。

三種結果的讀法（**FAIL 不是失敗**）：

| 結果 | 讀法 |
|---|---|
| 方向相反 | 不對稱來自各市場的部位結構，機制成立 |
| 方向相同 | 不對稱來自 sweep 本身的性質，要重新解釋 |
| 兩側都不顯著 | 這個市場上機制不成立，或解析度不足 |

---

## 2. 不做的事

```
不調 pivot length（沿用加密的凍結值）
不調停損 / 持有（沿用 3 ATR / 480 分，時間單位換算後）
不新增判別因子（加密上死掉的不能在這裡復活）
不因為 NQ 結果好看就回頭改加密的判準
不池化兩側
不在看標籤後改 session 或換月規則
```

---

## 3. 資料層（Stage 0）

### 3.1 標的與來源優先序

研究用 **NQ 不是 MNQ**（19 年歷史、成交量大、微結構乾淨）；實際下單用 MNQ。
若只拿得到 MNQ，照跑但在報告中標註歷史較短。

```
1. prop 平台匯出（NinjaTrader / Tradovate / Rithmic）   免費，先看
2. Databento GLBX.MDP3（註冊送 $125）                   先用免費額度拉樣本
3. FirstRate Data（NQ 個別合約自 NQZ08 / 2008-12）      一次性買斷
```

### 3.2 aggressor side 的可得性 —— 已查證（2026-09-10）

```
CME 源頭     MDP 3.0 Trade Summary 帶 tag 5797 AggressorSide
             1=買方主動 2=賣方主動 0=無明確主動方（隱含單成交）   ✓ 官方規格
Databento    GLBX.MDP3 應有，但其 issue tracker 載明
             「Legacy CME FIX/FAST (MDP2) data is missing trade side」
             -> **早期歷史缺 side**，分界日需帳號查（metadata API 要認證）
FirstRate    欄位規格未公開，依產品線判斷**很可能只有 OHLCV**
```

### 3.3 **對計畫 §5.1 的修正**（重要，先寫死）

原計畫：「若只有 OHLCV → 用 tick rule 推論，準確率約 80%+，會稀釋訊號」。

**2026-09-10 在加密上實測**（那裡同時有真值 `taker_buy_base` 與 tick rule
近似，可以直接比對）：

```
幣      分鐘級相關  5分聚合  符號一致率  **p99 極端事件重疊**
BTC      0.546     0.579     72.6%        50.7%
ETH      0.489     0.542     70.5%        46.4%
SOL      0.487     0.534     69.8%        43.9%
DOGE     0.406     0.470     67.4%        42.4%
```

符號一致率確實接近 80%，**但 D 的定義是「p99 極端」而不是「符號對不對」**，
而 tick rule 挑出來的極端事件只有 **42~51%** 跟真值重疊 —— 一半是錯的。

**所以：沒有 aggressor side 就不跑 D 臂，只跑 V。** tick rule 不得用來
頂替 D。停止條件 §7 第一條據此收緊：不是「無法驗證所以不做」，是
**已經驗證過而且不夠**。

### 3.4 交付物

```
data/bars/NQ_{contract}.parquet     每個個別合約一個檔
  ts (UTC, 1-min, 左閉), open, high, low, close, volume,
  contract (e.g. NQZ25), session ('RTH'|'ETH'),
  atr_h14, tick_size (=0.25), point_value (NQ=20, MNQ=2)
data/contracts.parquet
  contract, first_ts, last_ts, roll_ts, volume_peak_ts
data/quality/NQ.md
```

**不做連續序列。** 每個合約獨立處理，層級不跨合約 —— 最乾淨的換月處理，
代價是每次換月損失 pivot 的 lookback 期。

### 3.5 換月規則（預註冊，不事後改）

```
roll_ts = 新合約日成交量首次超過舊合約的那一天 00:00 UTC
舊合約在 roll_ts 之後的 bar 全部丟棄
新合約在 roll_ts 之前的 bar 全部丟棄
層級不跨 roll：新合約的 pivot 從 roll_ts + (left+right) 根之後才開始產生
```

### 3.6 Session 規則（預註冊）

```
RTH = 09:30–16:00 ET     ETH = 其餘
主判定：只用 RTH 內形成、且在 RTH 內被穿越的事件
敏感度：全時段，但標記 cross_session
gap_cross = True         穿越發生在 session 開盤第一根 bar 且開盤價已在
                         層級另一側 -> **主判定排除**
```

理由：ETH 極薄，pivot 在夜盤形成、日盤開盤跳空穿越，那不是一個穿越事件。

### 3.7 自動測試與驗證

```python
assert bars.ts.is_monotonic_increasing
assert (bars.high >= bars.low).all()
assert (bars.high >= bars[['open','close']].max(axis=1)).all()
assert (bars.volume >= 0).all()
assert bars.tick_size.eq(0.25).all()
assert bars.groupby('contract').ts.apply(lambda s: s.max() <= roll_ts[s.name]).all()
```

```
隨機抽 5 個 RTH 日，1 分鐘 volume 加總對 CME 官方日成交量，誤差 < 2%
RTH 內零成交量 bar 比例 < 0.1%
每個合約的成交量曲線：上市後爬升、到期前崩落；不符合者標記
session 邊界：抽 10 天確認 09:30 與 16:00 的 bar 存在且合理
```

**閘門**：所有 assert 通過 ∧ 驗證通過 ∧ contracts.parquet 完整 ∧ quality 報告存檔。

---

## 4. Stage 1–5：沿用，只改三個常數

| 項目 | 加密 | NQ |
|---|---|---|
| tick_size | 逐幣 | 0.25 |
| bin_size | max(tick, atr_h14/20) | 同公式 |
| ATR 基準 / 時間單位 | atr_h14 / 分鐘 | 不變 |

**持有期的必要規則變更（明寫，不事後決定）**：RTH 只有 390 分鐘，480 分
會跨越收盤。

```
主規則：持有到 min(480 分, 當日 RTH 收盤)，收盤強制平倉
敏感度：允許跨夜，標記 overnight = True
```

其餘（分布引擎、層級標示、事件、狀態變數、標籤）**一行都不改**。

---

## 5. Stage 3 替代驗證（NQ 沒有舊事件表可比）

```
1. 逐合約事件數的時序穩定性（不應該某合約突然十倍）
2. 兩側事件數大致平衡（極端不平衡代表定義有偏）
3. 隨機抽 20 個事件畫圖人工確認（唯一的人工步驟）
4. 事件的日內分布：應在開盤後一小時與收盤前集中
   —— 若均勻分布在 RTH，pivot 偵測可能有問題
```

---

## 6. 旗標移植

```
D（delta_ext）  有 aggressor side -> 同加密；只有 OHLCV -> **不做**（見 §3.3）
V（vol_burst）  v5[t] / base[t] >= 滾動 30 日 p99
                base = 前 30 個交易日**同一個 RTH 分鐘序號（0–389）**的 v5 均值
                ** 季節性基準用 RTH 分鐘序號，不是 UTC 時刻 **
liq_burst       期貨無公開強制平倉推送 -> 不移植
oi_crash        CME 每日公布、非分鐘級 -> 不移植（加密上也已死）
```

所以 NQ 的母體只有 **D 和 V**，沒有清算層 —— 比較時要記住這個實質差異。

**期貨特有、僅記錄不進主判定**：CFTC COT（週級）、到期週標記、
**經濟數據時刻（FOMC/CPI/NFP）前後 15 分鐘排除** —— 那會產生大量假的
sweep 事件，是資訊衝擊不是強制流，加密沒有等價物。

---

## 7. 判準

### H_main（每側各判）

```
PASS          旗標開火子集的 r_norm 均值，CI 下界 > max(MDE, 2c/ATR)
              c = 1.5 bps（NQ 來回，含交易所費與佣金；實際以帳戶為準）
              跨合約：>= 70% 的合約均值同號
REJECT        CI 涵蓋 0 且 MDE < 效應量（量得到而且是零）
INCONCLUSIVE  MDE > 效應量（看不見）
```

### H_asym

```
PASS（預測成立）  sellside − buyside 的 CI 下界 > 0
FAIL（預測反向）  buyside − sellside 的 CI 下界 > 0
不顯著            兩側差異 CI 涵蓋 0
```

三種都要報告。

**檢定總數**：H_main 兩側 + H_asym 一項 + 敏感度（session、換月、旗標門檻）
約 **12**，BH 修正。主判定只用：RTH、個別合約、主旗標門檻、主 τ。

---

## 8. 停止條件

```
資料源不給 aggressor side                 -> D 臂不做，只跑 V（§3.3 已驗證
                                             tick rule 不足以頂替）
Stage 0 成交量對帳誤差 > 5%               -> 換資料源
事件數（單側、RTH、主判定）< 500          -> INCONCLUSIVE，報告需要多長歷史
H_main 兩側都 REJECT                      -> 機制不跨市場，結案並寫進名冊
```

---

## 9. 執行順序

```
Step 1  確認資料源與欄位（aggressor side）          ← 阻塞；2026-09-10 部分完成
Step 2  commit 本檔                                 ← 看標籤前
Step 3  Stage 0：拉資料、換月、session、驗證、閘門
Step 4  Stage 1–5：改三個常數，跑通
Step 5  Stage 3 替代驗證
Step 6  旗標移植
Step 7  主判定 + H_asym
Step 8  寫判決，與加密結果並排
```

**Step 1 未完成的部分**（需要帳號或人工）：
- Databento 的 aggressor side 起始日（metadata API 需認證）
- prop 平台匯出欄位
- FirstRate 的實際欄位規格

---

## 10. 輸出

```
report/nq/
  01_data_quality.md   覆蓋、缺口、對帳、合約清單、換月點
  02_event_census.csv  逐合約、逐側事件數、日內分布
  03_flag_overlap.csv  旗標在 sweep 內部的開火率
  04_main.csv          兩側 H_main 的 n / 均值 / CI / MDE
  05_asym.csv          H_asym 三種結果
  06_sensitivity.csv   session、換月、門檻、持有規則
  07_vs_crypto.md      跟加密結果並排，機制解釋   ← **主要產出**
  08_verdict.md        判決 + 檢定總數 + 限制揭露
```

不管 PASS 或 REJECT，`07_vs_crypto.md` 那份對照才是值錢的東西。
