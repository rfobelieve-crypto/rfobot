"""
Squeeze x Order-Flow breakout direction experiment (task5, 2026-07-07).

Hypothesis (user): when price is squeezed (RBP-style consolidation range),
it tends to break toward the side of least resistance -- and order-flow
imbalance accumulated INSIDE the range should point at that side before
the breakout happens.

Design
------
Range detection is a faithful port of the Pine indicator
"Range Breakout Pro v2.3" (RBP):
  averagePrice   = EMA(close, 15)
  priceVolatility= ATR(20) * 1.1
  isInRange[t]   = all last 15 closes within averagePrice[t] +/- priceVolatility[t]
  box forms on isInRange False->True; top/bottom frozen at formation
  breakout       = first later bar whose CLOSE exceeds boxTop (UP) or boxBottom (DOWN)
  a new formation while a box is live supersedes (drops) the old box (Pine behavior)

Predictors are measured strictly on bars <= breakout_bar - 1 (no look-ahead):
pre-registered, small set -- no tuning loop, every tested feature is reported.

Layer 1 (full Binance history, 2019-09 -> now, ~50k bars):
  L1_delta_ratio   in-range taker delta sum / volume sum        (klines taker_buy_vol)
  L1_delta_z       in-range mean per-bar delta vs trailing 500-bar delta dist
  L1_cvd_slope     normalized OLS slope of in-range cumulative delta
  L1_price_pos     last close position inside box 0..1          (non-flow control)

Layer 2 (user's order-flow suite, Oct 2025 -> now, CG parquets):
  L2_fut_cvd_ratio aggregated futures taker delta ratio in range
  L2_spot_cvd_ratio  spot taker delta ratio in range
  L2_oi_chg        OI pct change over the range
  L2_funding_z     funding level z (168h) at breakout-1

Metrics per feature: AUC vs UP-breakout, sign hit-rate, bootstrap 95% CI
(2000 resamples), per-year (L1) / per-quarter (L2) sign stability.
Follow-through: breakouts grouped flow-aligned vs flow-opposed, forward
signed returns at H=4 and H=20 bars with bootstrap CI on the difference.

Pure research -- touches nothing in production. Cache under
research/.cache_squeeze/, results under research/results/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "research" / ".cache_squeeze"
RESULTS_PATH = PROJECT_ROOT / "research" / "results" / "squeeze_flow_breakout.json"
RAW = PROJECT_ROOT / "market_data" / "raw_data"

RANGE_PERIOD = 15
ATR_PERIOD = 20
RANGE_MULT = 1.1
N_BOOT = 2000
RNG_SEED = 42
HORIZONS = (4, 20)


# ---------------------------------------------------------------- data
def fetch_full_klines(symbol: str = "BTCUSDT", interval: str = "1h",
                      max_bars: int = 65000) -> pd.DataFrame:
    """Paginated Binance USDT-perp klines, cached to parquet, fail-loud."""
    import requests

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{symbol}_{interval}_full.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        age_h = (pd.Timestamp.utcnow() - df.index.max()).total_seconds() / 3600
        if age_h < 48:
            print(f"[data] cache hit {cache.name}: {len(df)} bars "
                  f"{df.index.min()} -> {df.index.max()}")
            return df
        print(f"[data] cache stale ({age_h:.0f}h), refetching")

    cols = ["ts_open", "open", "high", "low", "close", "volume", "close_time",
            "quote_vol", "trade_count", "taker_buy_vol", "taker_buy_quote", "ignore"]
    all_dfs: list[pd.DataFrame] = []
    end_time = None
    remaining = max_bars
    while remaining > 0:
        batch = min(remaining, 1500)
        params = {"symbol": symbol, "interval": interval, "limit": batch}
        if end_time is not None:
            params["endTime"] = end_time
        resp = requests.get("https://fapi.binance.com/fapi/v1/klines",
                            params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            break
        df = pd.DataFrame(rows, columns=cols)
        for c in ["open", "high", "low", "close", "volume",
                  "taker_buy_vol", "quote_vol"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ts_open"] = pd.to_numeric(df["ts_open"])
        df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
        df = df.set_index("dt").sort_index()
        all_dfs.append(df)
        end_time = int(df["ts_open"].iloc[0]) - 1
        remaining -= len(df)
        if len(df) < batch:      # history exhausted
            break
        time.sleep(0.35)

    if not all_dfs:
        raise RuntimeError("Binance kline fetch returned nothing")
    out = pd.concat(all_dfs).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out[["open", "high", "low", "close", "volume", "taker_buy_vol"]]
    # drop the still-open last bar
    out = out.iloc[:-1]
    out.to_parquet(cache)
    print(f"[data] fetched {len(out)} bars {out.index.min()} -> {out.index.max()}")
    return out


def load_layer2() -> pd.DataFrame:
    """User's CG order-flow parquets aligned to the 1h grid (backward merge)."""
    fut = pd.read_parquet(RAW / "cg_futures_cvd_agg_1h.parquet")
    spot = pd.read_parquet(RAW / "cg_spot_cvd_agg_1h.parquet")
    oi = pd.read_parquet(RAW / "cg_oi_agg_1h.parquet")
    fund = pd.read_parquet(RAW / "cg_funding_1h.parquet")
    out = pd.DataFrame(index=fut.index)
    out["fut_buy"] = fut["agg_taker_buy_vol"]
    out["fut_sell"] = fut["agg_taker_sell_vol"]
    out = out.join(spot[["agg_taker_buy_vol", "agg_taker_sell_vol"]]
                   .rename(columns={"agg_taker_buy_vol": "spot_buy",
                                    "agg_taker_sell_vol": "spot_sell"}), how="outer")
    out = out.join(oi[["close"]].rename(columns={"close": "oi_close"}), how="outer")
    out = out.join(fund[["close"]].rename(columns={"close": "funding"}), how="outer")
    return out.sort_index()


# ---------------------------------------------------------------- RBP port
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr_rma(df: pd.DataFrame, n: int) -> pd.Series:
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()   # Pine ta.atr = RMA


def detect_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate RBP box lifecycle; one row per resolved breakout episode."""
    ap = ema(df["close"], RANGE_PERIOD)
    band = atr_rma(df, ATR_PERIOD) * RANGE_MULT
    rmax = df["close"].rolling(RANGE_PERIOD).max()
    rmin = df["close"].rolling(RANGE_PERIOD).min()
    dev = pd.concat([rmax - ap, ap - rmin], axis=1).max(axis=1)
    in_range = (dev <= band) & dev.notna()
    formed = in_range & ~in_range.shift(1, fill_value=False)

    close = df["close"].to_numpy()
    formed_np = formed.to_numpy()
    top_np = (ap + band).to_numpy()
    bot_np = (ap - band).to_numpy()

    episodes = []
    box_top = box_bot = np.nan
    form_i = -1
    live = False
    for t in range(len(df)):
        if live:
            if close[t] > box_top or close[t] < box_bot:
                episodes.append({
                    "form_i": form_i, "break_i": t,
                    "box_top": box_top, "box_bot": box_bot,
                    "dir": 1 if close[t] > box_top else -1,
                })
                live = False
        if formed_np[t]:          # supersedes any live box (Pine reassigns activeBox)
            box_top, box_bot, form_i, live = top_np[t], bot_np[t], t, True
    ep = pd.DataFrame(episodes)
    if ep.empty:
        raise RuntimeError("no squeeze episodes detected")
    ep["form_ts"] = df.index[ep["form_i"]]
    ep["break_ts"] = df.index[ep["break_i"]]
    return ep


# ---------------------------------------------------------------- features
def build_features(df: pd.DataFrame, ep: pd.DataFrame,
                   l2: pd.DataFrame | None) -> pd.DataFrame:
    """All predictors use bars [form_i - RANGE_PERIOD + 1 .. break_i - 1]."""
    delta = 2.0 * df["taker_buy_vol"] - df["volume"]
    delta_mean_roll = delta.rolling(500).mean()
    delta_std_roll = delta.rolling(500).std()

    if l2 is not None:
        l2 = l2.reindex(df.index)   # exact 1h grid alignment
        fut_delta = l2["fut_buy"] - l2["fut_sell"]
        fut_vol = l2["fut_buy"] + l2["fut_sell"]
        spot_delta = l2["spot_buy"] - l2["spot_sell"]
        spot_vol = l2["spot_buy"] + l2["spot_sell"]
        funding_z = ((l2["funding"] - l2["funding"].rolling(168).mean())
                     / l2["funding"].rolling(168).std())

    rows = []
    dvals = delta.to_numpy()
    vvals = df["volume"].to_numpy()
    cvals = df["close"].to_numpy()
    for r in ep.itertuples():
        s = max(0, r.form_i - RANGE_PERIOD + 1)   # include formation window
        e = r.break_i                              # exclusive -> up to break-1
        if e - s < 3:
            continue
        w_delta = dvals[s:e]
        w_vol = vvals[s:e]
        row = {
            "break_i": r.break_i, "dir": r.dir,
            "form_ts": r.form_ts, "break_ts": r.break_ts,
            "range_bars": e - s,
            "box_top": r.box_top, "box_bot": r.box_bot,
        }
        vol_sum = np.nansum(w_vol)
        row["L1_delta_ratio"] = np.nansum(w_delta) / vol_sum if vol_sum > 0 else np.nan
        mu, sd = delta_mean_roll.iloc[e - 1], delta_std_roll.iloc[e - 1]
        row["L1_delta_z"] = ((np.nanmean(w_delta) - mu) / sd
                             if sd and not np.isnan(sd) and sd > 0 else np.nan)
        cvd = np.nancumsum(w_delta)
        x = np.arange(len(cvd), dtype=float)
        denom = np.nanstd(w_delta) * len(cvd)
        row["L1_cvd_slope"] = (np.polyfit(x, cvd, 1)[0] / denom
                               if denom and denom > 0 else np.nan)
        width = r.box_top - r.box_bot
        row["L1_price_pos"] = ((cvals[e - 1] - r.box_bot) / width - 0.5
                               if width > 0 else np.nan)

        if l2 is not None:
            fd, fv = fut_delta.iloc[s:e], fut_vol.iloc[s:e]
            sdta, sv = spot_delta.iloc[s:e], spot_vol.iloc[s:e]
            if fv.notna().sum() >= 3 and np.nansum(fv) > 0:
                row["L2_fut_cvd_ratio"] = np.nansum(fd) / np.nansum(fv)
            if sv.notna().sum() >= 3 and np.nansum(sv) > 0:
                row["L2_spot_cvd_ratio"] = np.nansum(sdta) / np.nansum(sv)
            oi_w = l2["oi_close"].iloc[s:e]
            if oi_w.notna().sum() >= 3 and oi_w.dropna().iloc[0] > 0:
                row["L2_oi_chg"] = oi_w.dropna().iloc[-1] / oi_w.dropna().iloc[0] - 1
            fz = funding_z.iloc[e - 1]
            if not np.isnan(fz):
                row["L2_funding_z"] = fz
        rows.append(row)
    return pd.DataFrame(rows)


def add_forward_returns(feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].to_numpy()
    n = len(close)
    for h in HORIZONS:
        vals = []
        for r in feat.itertuples():
            j = r.break_i + h
            vals.append((close[j] / close[r.break_i] - 1) * r.dir * 1e4
                        if j < n else np.nan)   # signed bps in breakout direction
        feat[f"fwd_{h}h_bps"] = vals
    return feat


# ---------------------------------------------------------------- stats
def bootstrap_ci(vals: np.ndarray, stat=np.nanmean, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 5:
        return (np.nan, np.nan)
    stats = [stat(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(n_boot)]
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def auc_score(y: np.ndarray, x: np.ndarray) -> float:
    from scipy.stats import rankdata
    m = ~np.isnan(x)
    y, x = y[m], x[m]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(x)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_auc_ci(y, x, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    m = ~np.isnan(x)
    y, x = y[m], x[m]
    if len(y) < 20:
        return (np.nan, np.nan)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        a = auc_score(y[idx], x[idx])
        if not np.isnan(a):
            aucs.append(a)
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def evaluate_feature(feat: pd.DataFrame, col: str, group_col: str) -> dict:
    x = feat[col].to_numpy(dtype=float)
    y = (feat["dir"].to_numpy() == 1).astype(float)
    m = ~np.isnan(x)
    n = int(m.sum())
    hit = float((np.sign(x[m]) == np.where(y[m] == 1, 1, -1)).mean()) if n else np.nan
    auc = auc_score(y, x)
    lo, hi = bootstrap_auc_ci(y, x)
    # sign stability across time groups
    stab = []
    for g, sub in feat[m].groupby(feat[m][group_col]):
        if len(sub) >= 15:
            a = auc_score((sub["dir"] == 1).astype(float).to_numpy(),
                          sub[col].to_numpy(dtype=float))
            if not np.isnan(a):
                stab.append((str(g), round(float(a), 3), len(sub)))
    frac_pos = (np.mean([1 if a > 0.5 else 0 for _, a, _ in stab])
                if stab else np.nan)
    return {"n": n, "sign_hit": round(hit, 4) if n else None,
            "auc": round(float(auc), 4) if not np.isnan(auc) else None,
            "auc_ci": [round(lo, 4), round(hi, 4)],
            "group_aucs": stab, "frac_groups_above_05": frac_pos}


def followthrough(feat: pd.DataFrame, flow_col: str) -> dict:
    """Aligned (flow sign == breakout dir) vs opposed forward returns."""
    out = {}
    x = feat[flow_col].to_numpy(dtype=float)
    aligned_mask = np.sign(x) == feat["dir"].to_numpy()
    for h in HORIZONS:
        col = f"fwd_{h}h_bps"
        a = feat.loc[aligned_mask & feat[flow_col].notna(), col].to_numpy(dtype=float)
        o = feat.loc[~aligned_mask & feat[flow_col].notna(), col].to_numpy(dtype=float)
        rng = np.random.default_rng(RNG_SEED)
        a_, o_ = a[~np.isnan(a)], o[~np.isnan(o)]
        if len(a_) >= 10 and len(o_) >= 10:
            diffs = [np.mean(a_[rng.integers(0, len(a_), len(a_))])
                     - np.mean(o_[rng.integers(0, len(o_), len(o_))])
                     for _ in range(N_BOOT)]
            ci = [round(float(np.percentile(diffs, 2.5)), 1),
                  round(float(np.percentile(diffs, 97.5)), 1)]
        else:
            ci = [None, None]
        out[f"H{h}"] = {
            "aligned_mean_bps": round(float(np.nanmean(a)), 1) if len(a_) else None,
            "aligned_n": int(len(a_)),
            "opposed_mean_bps": round(float(np.nanmean(o)), 1) if len(o_) else None,
            "opposed_n": int(len(o_)),
            "diff_ci_bps": ci,
        }
    return out


# ---------------------------------------------------------------- main
def main():
    df = fetch_full_klines()
    ep = detect_episodes(df)
    print(f"[episodes] {len(ep)} resolved breakouts; "
          f"UP {(ep['dir'] == 1).mean():.1%} base rate")

    l2 = load_layer2()
    feat = build_features(df, ep, l2)
    feat = add_forward_returns(feat, df)
    feat["year"] = feat["break_ts"].dt.year
    feat["quarter"] = feat["break_ts"].dt.to_period("Q").astype(str)

    results = {"meta": {
        "bars": len(df), "span": [str(df.index.min()), str(df.index.max())],
        "episodes": len(ep), "up_base_rate": round(float((ep["dir"] == 1).mean()), 4),
        "median_range_bars": int(feat["range_bars"].median()),
        "rbp_params": {"range_period": RANGE_PERIOD, "atr_period": ATR_PERIOD,
                       "range_mult": RANGE_MULT},
    }, "layer1": {}, "layer2": {}, "followthrough": {}}

    print("\n=== Layer 1: full-history Binance taker delta ===")
    for col in ["L1_delta_ratio", "L1_delta_z", "L1_cvd_slope", "L1_price_pos"]:
        r = evaluate_feature(feat, col, "year")
        results["layer1"][col] = r
        print(f"{col:18s} n={r['n']:5d} AUC={r['auc']} CI={r['auc_ci']} "
              f"hit={r['sign_hit']} frac_yr>0.5={r['frac_groups_above_05']}")

    print("\n=== Layer 2: CG order-flow suite (Oct 2025+) ===")
    l2feat = feat[feat["break_ts"] >= "2025-10-22"].copy()
    for col in ["L2_fut_cvd_ratio", "L2_spot_cvd_ratio", "L2_oi_chg",
                "L2_funding_z", "L1_delta_ratio"]:
        if col not in l2feat.columns:
            continue
        r = evaluate_feature(l2feat, col, "quarter")
        results["layer2"][col] = r
        print(f"{col:18s} n={r['n']:5d} AUC={r['auc']} CI={r['auc_ci']} "
              f"hit={r['sign_hit']} frac_q>0.5={r['frac_groups_above_05']}")

    print("\n=== Follow-through: flow-aligned vs opposed breakouts ===")
    for col in ["L1_delta_ratio", "L2_fut_cvd_ratio"]:
        sub = feat if col.startswith("L1") else l2feat
        if col in sub.columns:
            ft = followthrough(sub, col)
            results["followthrough"][col] = ft
            for h, d in ft.items():
                print(f"{col} {h}: aligned {d['aligned_mean_bps']} bps "
                      f"(n={d['aligned_n']}) vs opposed {d['opposed_mean_bps']} bps "
                      f"(n={d['opposed_n']}), diff CI {d['diff_ci_bps']}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[saved] {RESULTS_PATH}")


if __name__ == "__main__":
    main()
