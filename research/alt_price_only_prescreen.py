"""Is any alt more predictable than BTC, before spending on a full pipeline?

The plan is to clone V7 onto 9 alts. Coverage is not the blocker — the
2026-07-28 probe found all 9 have the full Coinglass channel set and the same
180-day / 4320-bar history. The blocker is the precedent: ETH was tried three
times with complete data and landed at clean AUC 0.5057, a coin flip.

The only argument that survives that precedent is "small caps are less
efficient, so more predictable." That claim is testable cheaply, because it
does not need Coinglass at all — if a coin has exploitable structure, some of
it should show in price and volume alone. This runs the same clean
walk-forward on a price-only feature set across all 11 coins, with BTC and ETH
as calibration: BTC's price-only AUC is the "known" reference, and ETH's is
the known failure.

Read it as a SCREEN, not a verdict. Passing here does not mean an alt has an
edge; it means the "less efficient" hypothesis is not dead for that coin and
the full pipeline is worth its cost. Failing here means adding 91 Coinglass
features is unlikely to rescue it — you would be paying pipeline cost for a
coin whose own price series carries nothing.

Features are the OHLCV-computable subset of V7's deployed set (returns lags,
realized-vol family, volume ratios, hour/weekday cyclicals) — the same shapes,
recomputed per coin. Coinglass-derived and order-book-derived columns are
excluded by construction, so every coin is measured on identical footing.

Run: python research/alt_price_only_prescreen.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.dual_model.shared_data import walk_forward_splits  # noqa: E402

OUT = ROOT / "research/results/alt_price_only_prescreen.json"
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "BNB",
         "LINK", "SUI", "UNI", "AAVE"]
BARS = 4320               # match the Coinglass plan cap so coins are equal
HORIZON = 4               # y = TWAP path return over next 4 bars, as in V7
SEED = 42
BOOT = 2000


def klines(sym: str, n: int) -> pd.DataFrame:
    rows, end = [], None
    while len(rows) < n:
        u = (f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}USDT"
             f"&interval=1h&limit=1500")
        if end:
            u += f"&endTime={end}"
        b = json.load(urllib.request.urlopen(u, timeout=30))
        if not b:
            break
        rows = b + rows
        end = b[0][0] - 1
        time.sleep(0.25)
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["ms", "open", "high", "low", "close", "volume"]
    df = df.drop_duplicates("ms").tail(n)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df["ms"], unit="ms")
    return df.drop(columns="ms").sort_index()


def build(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV-computable analogues of V7's non-Coinglass features."""
    f = pd.DataFrame(index=df.index)
    c, v = df["close"], df["volume"]
    r = c.pct_change()

    for k in range(1, 11):                       # return_lag_1..10
        f[f"ret_lag_{k}"] = r.shift(k)
    f["realized_vol_20b"] = r.rolling(20).std()
    f["vol_acceleration"] = (r.rolling(6).std() / r.rolling(24).std()) - 1
    f["vol_kurtosis"] = r.rolling(48).kurt()
    f["return_kurtosis"] = r.rolling(24).kurt()
    f["vol_entropy"] = (-(r.rolling(24).apply(
        lambda x: np.nansum(np.abs(x) / (np.abs(x).sum() + 1e-12)
                            * np.log(np.abs(x) / (np.abs(x).sum() + 1e-12) + 1e-12)),
        raw=True)))
    rv = r.rolling(20).std()
    f["vol_regime"] = (rv / rv.rolling(168, min_periods=48).mean()) - 1

    f["quote_vol_ratio"] = v / v.rolling(24).mean()
    f["quote_vol_zscore"] = ((v - v.rolling(168, min_periods=48).mean())
                             / v.rolling(168, min_periods=48).std())
    # price impact: |return| per unit volume, and its z-score
    imp = r.abs() / (v / v.rolling(24).mean()).replace(0, np.nan)
    f["price_impact"] = imp
    f["price_impact_zscore"] = ((imp - imp.rolling(168, min_periods=48).mean())
                                / imp.rolling(168, min_periods=48).std())
    rng = (df["high"] - df["low"]) / c
    f["range_pct"] = rng
    f["close_pos_in_range"] = ((c - df["low"])
                               / (df["high"] - df["low"]).replace(0, np.nan))
    f["fragility"] = rng / rng.rolling(48, min_periods=24).mean()

    # trend-state dummies (V7 keeps regime as a FEATURE, not a partition —
    # mistake.md 2026-04-13)
    ma_f, ma_s = c.rolling(24).mean(), c.rolling(168, min_periods=48).mean()
    f["is_trending_bull"] = (ma_f > ma_s * 1.01).astype(float)
    f["is_trending_bear"] = (ma_f < ma_s * 0.99).astype(float)

    h = df.index.hour.values
    d = df.index.dayofweek.values
    f["hour_sin"] = np.sin(2 * np.pi * h / 24)
    f["hour_cos"] = np.cos(2 * np.pi * h / 24)
    f["weekday_sin"] = np.sin(2 * np.pi * d / 7)
    f["weekday_cos"] = np.cos(2 * np.pi * d / 7)

    fwd = c.shift(-1).rolling(HORIZON).mean().shift(-(HORIZON - 1))
    f["y"] = fwd / c - 1.0
    return f


def boot_ci(v: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    m = [rng.choice(v, len(v), replace=True).mean() for _ in range(BOOT)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def run(coin: str) -> dict | None:
    df = klines(coin, BARS)
    f = build(df).dropna()
    if len(f) < 1200:
        print(f"{coin:<6} bars={len(f)} 太少，略過")
        return None
    X = f.drop(columns="y")
    y = (f["y"] > 0).astype(int).values
    splits = walk_forward_splits(len(f), initial_train=288, test_size=250,
                                 step=250, purge=HORIZON, embargo=HORIZON)
    aucs = []
    for tr, te in splits:
        if len(tr) < 300 or len(te) < 60 or len(set(y[te])) < 2:
            continue
        m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                              eval_metric="logloss", random_state=SEED, n_jobs=4)
        m.fit(X.iloc[tr].values, y[tr])          # no eval_set: clean, no leak
        aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te].values)[:, 1]))
    if not aucs:
        return None
    a = np.array(aucs)
    lo, hi = boot_ci(a)
    return dict(coin=coin, bars=len(f), folds=len(a), auc=float(a.mean()),
                median=float(np.median(a)), ci_lo=lo, ci_hi=hi,
                frac_above_half=float((a > 0.5).mean()))


def main() -> int:
    print(f"clean walk-forward, price-only ({BARS} bars, purge/embargo={HORIZON})")
    print(f"\n{'coin':<7}{'bars':>6}{'folds':>7}{'AUC':>8}{'median':>8}"
          f"{'  95% CI':>18}{'>0.5 折':>9}")
    print("-" * 64)
    rows = []
    for c in COINS:
        try:
            r = run(c)
        except Exception as exc:
            print(f"{c:<7}失敗 {type(exc).__name__}: {str(exc)[:40]}")
            continue
        if not r:
            continue
        rows.append(r)
        star = "*" if (r["ci_lo"] - 0.5) * (r["ci_hi"] - 0.5) > 0 else " "
        print(f"{r['coin']:<7}{r['bars']:>6}{r['folds']:>7}{r['auc']:>8.4f}"
              f"{r['median']:>8.4f}   [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]{star}"
              f"{r['frac_above_half']:>8.0%}")

    if rows:
        btc = next((r for r in rows if r["coin"] == "BTC"), None)
        eth = next((r for r in rows if r["coin"] == "ETH"), None)
        print("\n判讀 —— 這是篩選不是判決：")
        if btc:
            print(f"  BTC 純價格基準 {btc['auc']:.4f}"
                  f"（完整 136 特徵的 clean AUC 是 0.5412）")
        if eth:
            print(f"  ETH 已知失敗案例 {eth['auc']:.4f}"
                  f"（完整管線三輪測試 clean AUC 0.5057）")
        best = sorted((r for r in rows if r["coin"] not in ("BTC", "ETH")),
                      key=lambda x: -x["auc"])[:3]
        print("  alt 前三名：" + "、".join(
            f"{b['coin']} {b['auc']:.4f}" for b in best))
        print("  只有明顯高於 ETH、且 CI 不含 0.5 的幣，才值得複製整套管線。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(generated=str(pd.Timestamp.utcnow()),
                                   bars=BARS, results=rows), indent=2),
                   encoding="utf-8")
    print(f"\nsaved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
