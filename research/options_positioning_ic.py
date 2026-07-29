"""Conditional-IC screen for options POSITIONING features (not DVOL).

Context: V7 already consumes Deribit's DVOL (5 dvol_* columns in
features_all). What it has never seen is the options *positioning* side that
indicator_options_snapshots has quietly been collecting hourly since
2026-04-03: 25-delta IV skew, put/call OI, and max-pain. Max-pain distance is
the closest proxy to a dealer-gamma / pinning signal obtainable without
per-strike history (true GEX needs a historical option chain, which neither
free Deribit nor the Coinglass plan provides — verified 2026-07-27).

Discipline (mistake.md 2026-06-01): raw IC is NOT the decision metric. A
feature can correlate with the target and still add nothing once V7's 136
features have had their say. So the screen here is CONDITIONAL IC — the
correlation between the candidate and V7's walk-forward OOS residual
(y - pred). Only features that carry information V7 does not already have
survive to an ensemble A/B, which is a separate, later step with its own
4 gates (mistake.md 2026-06-02).

Also enforced:
  * hourly snapshots are as-of joined BACKWARD onto bar timestamps, so a bar
    never sees a snapshot taken after it closed;
  * every feature is trailing-only (z-scores/changes over past windows);
  * per-month IC consistency is reported, because a pooled IC on ~114 days
    can be carried by one stretch.

Run: python research/options_positioning_ic.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.db import get_db_conn  # noqa: E402

OOS = ROOT / "research/results/dual_model/direction_reg_oos_mse.parquet"
FEATS = ROOT / "research/dual_model/.cache/features_all.parquet"
OUT = ROOT / "research/results/options_positioning_ic.json"

MIN_N = 300          # below this a conditional IC is not worth reading
BOOT = 2000
SEED = 42


def load_snapshots() -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT dt, total_call_oi, total_put_oi, iv_skew, "
                "mean_otm_put_iv, mean_otm_call_iv, max_pain_price, "
                "call_oi_notional, put_oi_notional, opt_futures_ratio, "
                "pc_volume_ratio, dvol_value "
                "FROM indicator_options_snapshots ORDER BY dt"
            )
            rows = cur.fetchall() or []
    finally:
        conn.close()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["dt"])
    for c in df.columns:
        if c != "dt":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("dt").sort_index()


def build_features(snap: pd.DataFrame, spot: pd.Series) -> pd.DataFrame:
    """Trailing-only positioning features. `spot` is the bar close, reindexed
    onto snapshot timestamps, so max-pain distance is measured against a price
    that was already known at snapshot time."""
    f = pd.DataFrame(index=snap.index)

    pc_oi = snap["total_put_oi"] / snap["total_call_oi"].replace(0, np.nan)
    f["pc_oi_ratio"] = pc_oi
    f["pc_oi_z72"] = _z(pc_oi, 72)
    f["pc_oi_chg24"] = pc_oi.diff(24)

    sk = snap["iv_skew"]
    f["iv_skew"] = sk
    f["iv_skew_z72"] = _z(sk, 72)
    f["iv_skew_chg24"] = sk.diff(24)

    # Max-pain distance: >0 = spot above the pain point. The pinning thesis
    # says price is pulled back toward it, so this should lean negative.
    mp = snap["max_pain_price"].replace(0, np.nan)
    dist = (spot - mp) / spot
    f["maxpain_dist"] = dist
    f["maxpain_dist_z72"] = _z(dist, 72)
    f["maxpain_shift24"] = mp.pct_change(24)

    notional = snap["call_oi_notional"] + snap["put_oi_notional"]
    f["oi_notional_imb"] = (
        (snap["call_oi_notional"] - snap["put_oi_notional"])
        / notional.replace(0, np.nan)
    )
    f["opt_futures_ratio_z72"] = _z(snap["opt_futures_ratio"], 72)
    f["pc_volume_ratio_z72"] = _z(snap["pc_volume_ratio"], 72)
    return f


def _z(s: pd.Series, w: int) -> pd.Series:
    m = s.rolling(w, min_periods=max(8, w // 3)).mean()
    sd = s.rolling(w, min_periods=max(8, w // 3)).std()
    return (s - m) / sd.replace(0, np.nan)


def boot_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(x)
    out = []
    for _ in range(BOOT):
        i = rng.integers(0, n, n)
        r = spearmanr(x[i], y[i]).correlation
        if np.isfinite(r):
            out.append(r)
    if not out:
        return (np.nan, np.nan)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    oos = pd.read_parquet(OOS)
    feats = pd.read_parquet(FEATS)
    for d in (oos, feats):
        d.index = pd.DatetimeIndex(d.index)
        if d.index.tz is not None:
            d.index = d.index.tz_convert("UTC").tz_localize(None)
    oos, feats = oos.sort_index(), feats.sort_index()

    # V7's own OOS error: what the model got WRONG is the only room a new
    # feature can fill.
    resid = (oos["y_path_ret_4h"] - oos["pred_ret"]).rename("resid").dropna()
    bars = pd.DataFrame({"resid": resid})
    bars["y"] = oos["y_path_ret_4h"]
    bars["close"] = feats["close"].reindex(bars.index)
    bars = bars.dropna(subset=["resid", "close"])
    print(f"V7 OOS bars: {len(bars)}  {bars.index.min()} -> {bars.index.max()}")

    snap = load_snapshots()
    if snap.empty:
        print("no snapshots")
        return 1
    print(f"snapshots  : {len(snap)}  {snap.index.min()} -> {snap.index.max()}")

    spot_at_snap = pd.merge_asof(
        pd.DataFrame(index=snap.index).reset_index().rename(columns={"index": "dt"}),
        bars[["close"]].reset_index().rename(columns={"index": "dt"}),
        on="dt", direction="backward",
    ).set_index("dt")["close"]
    fx = build_features(snap, spot_at_snap)

    # BACKWARD as-of: each bar only sees the latest snapshot at or before it.
    joined = pd.merge_asof(
        bars.reset_index().rename(columns={"index": "dt"}).sort_values("dt"),
        fx.reset_index().rename(columns={"index": "dt"}).sort_values("dt"),
        on="dt", direction="backward", tolerance=pd.Timedelta("2h"),
    ).set_index("dt")

    overlap = joined.dropna(subset=["iv_skew"])
    print(f"overlap    : {len(overlap)} bars "
          f"({overlap.index.min()} -> {overlap.index.max()})\n")
    if len(overlap) < MIN_N:
        print(f"NOT ENOUGH OVERLAP (n={len(overlap)} < {MIN_N}) — "
              "the options table only starts 2026-04-03 and the WF-OOS window "
              "may end earlier. Re-run once both cover a common stretch.")
        return 0

    cols = [c for c in fx.columns]
    rows = []
    print(f"{'feature':<24}{'n':>6}{'raw IC':>9}{'cond IC':>9}"
          f"{'  95% CI (cond)':>20}{'  月一致':>9}")
    print("-" * 80)
    for c in cols:
        sub = overlap[[c, "y", "resid"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < MIN_N:
            continue
        raw = spearmanr(sub[c], sub["y"]).correlation
        cond = spearmanr(sub[c], sub["resid"]).correlation
        lo, hi = boot_ci(sub[c].values, sub["resid"].values)
        by_m = sub.groupby(sub.index.to_period("M")).apply(
            lambda g: spearmanr(g[c], g["resid"]).correlation
            if len(g) > 50 else np.nan).dropna()
        cons = float((np.sign(by_m) == np.sign(cond)).mean()) if len(by_m) else np.nan
        sig = "*" if (np.isfinite(lo) and np.isfinite(hi) and lo * hi > 0) else " "
        rows.append(dict(feature=c, n=len(sub), raw_ic=raw, cond_ic=cond,
                         ci_lo=lo, ci_hi=hi, months=len(by_m), consistency=cons,
                         significant=bool(sig.strip())))
        print(f"{c:<24}{len(sub):>6}{raw:>9.3f}{cond:>9.3f}"
              f"   [{lo:+.3f},{hi:+.3f}]{sig}{cons:>8.0%}")

    print("\n判讀:cond IC 才是決策依據。CI 不含 0 (*) 且月一致性 >= 0.67 才值得")
    print("進 ensemble A/B;raw IC 高但 cond IC ~0 = V7 已經吃過這個資訊。")

    import json
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        dict(generated=str(pd.Timestamp.utcnow()), n_overlap=len(overlap),
             results=rows), indent=2, default=str), encoding="utf-8")
    print(f"\nsaved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
