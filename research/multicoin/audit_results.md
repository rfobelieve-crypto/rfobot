# Coinglass 多幣端點審計（V7 多幣化可行性研究 — Step 1）

Run: 2026-07-15（`audit_cg_endpoints.py` + `verify_suspects.py` + `verify_value_diff.py`）

## 結論一句話

**ETH 保有 ~97% 特徵覆蓋（136 個裡只缺 coinbase premium 家族 3 個 + etf_aum），
SOL 保有 ~92%（再多缺 DVOL 家族 ~5 個）。資料層不是多幣化的瓶頸。**

## 端點 × 幣 可用性（已含 symbol 無視驗證）

| endpoint | BTC | ETH | SOL | 備註 |
|---|---|---|---|---|
| oi | ✅ | ✅ | ✅ | pair-style, 值已驗 distinct |
| oi_agg | ✅ | ✅ | ✅ | 值已驗 distinct |
| liquidation | ✅ | ✅ | ✅ | |
| long_short | ✅ | ✅ | ✅ | |
| global_ls | ✅ | ✅ | ✅ | |
| funding | ✅ | ✅ | ✅ | |
| taker | ✅ | ✅ | ✅ | |
| **coinbase_premium** | ✅ (ts=**s**) | ❌ | ❌ | **端點無視 symbol 參數，三幣回同值 → BTC-only** |
| bitfinex_margin | ✅ (ts=**s**) | ✅ | ✅ | 值已驗 distinct |
| top_ls_position | ✅ | ✅ | ✅ | |
| futures_cvd_agg | ✅ | ✅ | ✅ | 值已驗 distinct |
| spot_cvd_agg | ✅ | ✅ | ✅ | 值已驗 distinct |
| liq_agg | ✅ | ✅ | ✅ | 值已驗 distinct |
| oi_coin_margin | ✅ | ✅ | ✅ | 值已驗 distinct |
| opt_max_pain (Deribit) | ✅ 11 expiry | ✅ 11 expiry | ✅ 7 expiry | Deribit 有 SOL 選擇權 |
| opt_info | ✅ | ✅ | ✅ | |
| opt_fut_ratio | ✅ | ✅ | ❌ | 單一回應同時帶 btc_/eth_ 兩欄；無 SOL 欄 |
| etf_flow | ✅ 644d | ✅ 506d | ⚠️ 12d | SOL ETF 剛上市，live 可用但無訓練深度 |
| etf_aum | ✅ | ❌ 404 | ❌ 404 | 只有 /etf/bitcoin/aum 存在 |
| futures_netflow | ✅ | ✅ | ✅ | netflow-list 一次回全幣 |
| spot_netflow | ✅ | ✅ | ✅ | |
| hl_whale | ✅ 282 | ✅ 184 | ✅ 62 | 單一 list 含全幣 symbol 欄 |
| fear_greed | SHARED | SHARED | SHARED | 市場級指數，跨幣共用 |
| deribit_dvol | ✅ | ✅ | ❌ | DVOL 只有 BTC/ETH |
| cross_market (yfinance) | SHARED | SHARED | SHARED | SPX/DXY/Gold/US10Y 跨幣共用 |
| Binance klines/depth/aggTrades | ✅ | ✅ | ✅ | 公開 API，全幣可用 |

## 特徵層影響（Direction 136 為基準）

| 幣 | 缺失特徵 | 覆蓋率 |
|---|---|---|
| ETH | cg_cb_premium, cg_cb_premium_rate, cg_cb_premium_rate_zscore, etf_aum | **~97%** |
| SOL | 上述 + dvol_* 家族（dvol_value/change/oi_interaction/rv_spread）+ opt_fut_ratio; etf_flow 無歷史深度 | **~92%** |

cb_premium 不參與任何 COINGLASS_CROSS 交叉特徵，缺它只影響 raw+zscore 三欄，
不會級聯。

## 陷阱備忘（給移植實作）

1. **symbol 無視型端點**：`/coinbase-premium-index` 收 symbol 參數但完全忽略
   （BTC/ETH/SOL 回同一個 premium_rate）。任何新增幣種端點都必須跑值差異化
   檢查（`verify_value_diff.py` 模式），不能只看「回 200 + 有 rows」。
2. **時間戳 unit 不一致照舊**：coinbase_premium / bitfinex_margin 是秒級，
   其餘毫秒級——與 mistake.md 2026-04-12 一致，多幣版沿用自動偵測。
3. opt_fut_ratio 的欄位名是 `btc_option_vs_futures_radio` / `eth_..._radio`
   （API 拼錯 ratio→radio），ETH 移植時要取 eth_ 欄而不是換 symbol 參數。

## 下一步（Step 2 / Step 3）

- Step 2: ETH 移植實驗——用可用特徵建 ETH 特徵表，跑同一套乾淨 WF
  （purge+embargo、無 early-stop 洩漏），對照 BTC clean AUC 0.5412。
- Step 3: 訊號重合率——ETH Strong vs BTC Strong 時間對齊統計。
  Go/No-Go 門檻（預先登記）：ETH clean AUC ≥ ~0.54 且重合率 < 50% → 繼續；
  否則多幣化對 V7 無性價比，資源回異源資料線。
