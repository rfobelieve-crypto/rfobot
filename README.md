# rfobot
🔷 加密貨幣流動性獵取（Liquidity Sweep）反轉研究

本專案主要研究：

流動性獵取（Liquidity Sweep）之後，價格是「反轉」還是「延續」？
並透過成交量 / 資金流（Order Flow）數據驗證是否存在可量化的交易優勢（edge）。

🧠 研究動機

在主觀交易中（例如 ICT、Smart Money Concepts）：

流動性獵取（掃高 / 掃低）常被視為反轉訊號
但這類概念多半缺乏數據驗證

本專案希望解決以下問題：

流動性獵取是否真的有統計優勢？
是否所有 sweep 都一樣？
資金流（taker volume / delta）是否能確認或否定 sweep？
⚙️ 資料來源與結構

本專案使用事件導向（event-based）資料結構，並對齊實際交易機器人架構。

每一筆資料代表一個「流動性事件」：

📊 主要欄位
liquidity_side：流動性方向（buy / sell）
trigger_price：觸發價格（sweep 發生點）
upper_target / lower_target：目標價位
flow_until_hit_*：事件後即時資金流
flow_1h_*：1小時資金流統計
hit_latency_seconds：到達目標的時間
outcome_label：結果（reversal / continuation）
🧪 Step 1：探索性資料分析（EDA）
📌 基本分布
反轉比例：約 57%
延續比例：約 43%

👉 表面上存在反轉傾向，但仍需進一步分析

📊 資金流（Flow）觀察
Reversal：
平均 delta 為負（賣壓）
Continuation：
平均 delta 為正（買壓）

👉 初步結論：

❗不是 sweep 本身，而是「資金方向」決定結果

🧠 Step 2：機器學習（Baseline）

使用 RandomForest 建立初步模型：

使用特徵
flow_until_hit_delta
flow_1h_delta
flow_ratio
liquidity_side
結果
Accuracy ≈ 50%
模型無法穩定預測
❗ 關鍵結論

機器學習在沒有良好特徵設計前，無法有效捕捉市場結構

🔥 Step 3：規則分析（核心發現）
❌ 單一條件（無 edge）
delta < 0 → reversal
準確率：約 51%
幾乎等同隨機
✅ 條件化 edge（重要）
情境 1：掃上方流動性（buy）
liquidity_side = buy
AND flow_until_hit_delta < 0
→ reversal ≈ 65.8%

👉 解讀：

價格向上掃流動性
但實際為賣壓主導
→ 假突破（absorption）
情境 2：掃下方流動性（sell）
liquidity_side = sell
AND flow_until_hit_delta > 0
→ reversal ≈ 71.0%

👉 解讀：

掃低後出現買盤承接
→ 反轉向上
🧠 核心觀念

❗流動性獵取本身沒有 edge

❗「流動性 + 資金流結構」才有 edge

📊 市場結構解讀
情境	解讀
掃高 + 賣壓	吸收 → 反轉
掃低 + 買壓	承接 → 反轉
流動性 + 同方向資金	趨勢延續
🚀 策略雛形
if liquidity_side == "buy" and flow_until_hit_delta < 0:
    signal = "reversal"

elif liquidity_side == "sell" and flow_until_hit_delta > 0:
    signal = "reversal"

else:
    signal = "no_trade"
📈 未來發展方向
加入 delta threshold（強度篩選）
加入 latency（速度特徵）
分時段分析（NY / London）
使用真實市場資料（替代模擬資料）
建立即時預測系統（WebSocket + Bot）
🧰 技術架構
Python（pandas / sklearn）
MySQL（事件資料庫）
WebSocket（即時資料）
Telegram Bot（未來應用）
⚠️ 聲明

本專案使用模擬資料（但對齊實際事件結構），主要目的為：

驗證研究流程
建立資料管線
分析市場結構

不代表實際交易績效。

🧠 專案定位

本專案並非單純 AI 專案，而是：

✅ 量化研究（Quant Research） + 市場結構分析（Market Microstructure）

🔥 核心價值

本專案的重點不是模型，而是：

如何定義市場事件
如何驗證交易 edge
如何避免將無效資料餵給 AI
