"""Is TEST B's volatility IC an EDGE, or just "activity predicts volatility"?

Not a new pre-registered test and it does NOT move the 2026-08-10 PASS bar
(cancel_shock_ic.py owns that, definitions frozen). This is an
interpretation aid for one question the raw IC cannot answer:

  cancel_shock_ic TEST B reports intensity_shock -> |forward return| with
  Spearman IC +0.075..+0.120 across all four horizons. But "more activity
  now -> more movement soon" is close to a stylised fact of any market
  microstructure series. If the cancellation stream is only re-measuring
  general activity, the number is real but not proprietary, and it buys
  nothing that trade volume (already collected, already free) does not.

  This repo has been here twice:
    - subhourly Phase 0: G1 information PASS (15/15 months same sign),
      G2 economics FAIL (best gross 6.2 bps < 8 bps cost) -> PARK
    - order-flow研究: cancels looked predictive until `tot_add ≈ tot_cancel`
      showed it was an activity proxy

So: control for activity and see what survives.

  CONTROLS (all trailing-only, same minute grid)
    vol_shock    = volume_usd(t) / median(volume_usd, trailing 60m, min 30)
    trade_shock  = trade_count(t) / median(trade_count, trailing 60m, min 30)
    upd_shock    = update_count(t) / median(update_count, trailing 60m, min 30)
      ^ update_count lives in depth_deltas itself: if intensity_shock is just
        "the book is busy", upd_shock should carry the same information.

  METHOD  Spearman partial correlation: rank-transform everything, regress
          both signal and target on the controls, correlate the residuals.
          Non-overlapping stride = h, same as the frozen test.

  READING
    partial IC stays close to raw   -> cancellations carry own information
    partial IC collapses toward 0   -> it was an activity proxy

Usage: python research/cancel_shock_control.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from shared.db import get_db_conn  # noqa: E402

SYM = "BTC-USD"
HORIZONS = (5, 15, 30, 60)
BASE_WIN, BASE_MIN = 60, 30
RNG = np.random.default_rng(11)


def load() -> pd.DataFrame:
    conn = get_db_conn()
    try:
        # exchange='binance' —— **必須與凍結腳本一致**（它註明 spot series only）。
        # 少了這個條件會把 binance_perp 的同一分鐘也撈進來，join 出 68k 列
        # （凍結測試是 38k），raw IC 跟著失真（h=5 +0.056 vs 真值 +0.120）。
        dd = pd.read_sql(
            "SELECT minute_start_ms ms, bid_cancel_qty bc, ask_cancel_qty ac, "
            "       update_count upd "
            f"FROM depth_deltas_1m WHERE canonical_symbol='{SYM}' "
            "AND exchange='binance' ORDER BY minute_start_ms", conn)
        ob = pd.read_sql(
            "SELECT ts_ms ms, mid_price mid FROM orderbook_snapshots_1m "
            f"WHERE canonical_symbol='{SYM}' ORDER BY ts_ms", conn)
        # window_start 存的是**毫秒整數**不是 datetime —— UNIX_TIMESTAMP() 會回 NULL，
        # 首版就是這樣讓成交量控制的覆蓋率變成 0%
        fb = pd.read_sql(
            "SELECT window_start ms, volume_usd vol, trade_count ntr "
            f"FROM flow_bars_1m WHERE canonical_symbol='{SYM}' "
            "AND exchange_scope='all' ORDER BY window_start", conn)
    finally:
        conn.close()
    for d in (dd, ob, fb):
        d["ms"] = (d["ms"].astype("int64") // 60000) * 60000
    # 一分鐘可能有多筆快照/多筆 bar —— 與凍結腳本一致取最後一筆
    ob = ob.groupby("ms", as_index=False).last()
    fb = fb.groupby("ms", as_index=False).last()
    dd = dd.groupby("ms", as_index=False).sum(numeric_only=True)
    df = dd.merge(ob, on="ms", how="inner").merge(fb, on="ms", how="left")
    return df.sort_values("ms").reset_index(drop=True)


def shock(s: pd.Series) -> pd.Series:
    base = s.rolling(BASE_WIN, min_periods=BASE_MIN).median()
    return s / base.replace(0, np.nan)


def partial_ic(x: np.ndarray, y: np.ndarray, ctrls: list[np.ndarray]):
    """Spearman partial correlation via rank-space residualisation."""
    ok = np.isfinite(x) & np.isfinite(y)
    for c in ctrls:
        ok &= np.isfinite(c)
    if ok.sum() < 50:
        return None, None, int(ok.sum())
    r = lambda v: pd.Series(v[ok]).rank().to_numpy()  # noqa: E731
    X = np.column_stack([np.ones(ok.sum())] + [r(c) for c in ctrls])
    resid = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]  # noqa: E731
    rx, ry = resid(r(x)), resid(r(y))
    ic = spearmanr(rx, ry).correlation
    boot = [spearmanr(rx[i], ry[i]).correlation
            for i in (RNG.integers(0, len(rx), len(rx)) for _ in range(2000))]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return ic, (lo, hi), int(ok.sum())


def main() -> int:
    df = load()
    n = len(df)
    print(f"joined minutes n={n}, span ≈ {(df.ms.iloc[-1]-df.ms.iloc[0])/3.6e6:.1f}h")
    if df["vol"].isna().mean() > 0.5:
        print("[WARN] flow_bars_1m 覆蓋不足，成交量控制不可信")

    tot_cancel = df["bc"] + df["ac"]
    sig = shock(tot_cancel).to_numpy()
    ctl_upd = shock(df["upd"]).to_numpy()
    ctl_vol = shock(df["vol"]).to_numpy()
    ctl_ntr = shock(df["ntr"]).to_numpy()
    mid = df["mid"].to_numpy()

    print(f"\n控制變數覆蓋：update_count {np.isfinite(ctl_upd).mean():.0%}  "
          f"volume {np.isfinite(ctl_vol).mean():.0%}  trades {np.isfinite(ctl_ntr).mean():.0%}")
    print("\n  TEST B（撤單強度 → |未來報酬|）原始 IC vs 控制活躍度後的偏 IC")
    print(f"  {'h':>3} | {'raw IC':>8} | {'控 update':>10} | {'控 volume':>10} "
          f"| {'控 三者':>10} | {'n':>5}")
    print("  " + "-" * 62)
    rows = []
    for h in HORIZONS:
        fwd = np.full(len(mid), np.nan)
        fwd[:-h] = np.abs(mid[h:] / mid[:-h] - 1.0)
        idx = np.arange(0, len(df), h)          # 非重疊，與凍結測試同步
        x, y = sig[idx], fwd[idx]
        cu, cv, cn = ctl_upd[idx], ctl_vol[idx], ctl_ntr[idx]
        ok = np.isfinite(x) & np.isfinite(y)
        raw = spearmanr(x[ok], y[ok]).correlation
        p_u, ci_u, _ = partial_ic(x, y, [cu])
        p_v, ci_v, _ = partial_ic(x, y, [cv])
        p_a, ci_a, na = partial_ic(x, y, [cu, cv, cn])
        rows.append((h, raw, p_u, p_v, p_a))
        f = lambda v: f"{v:+.3f}" if v is not None else "   —"  # noqa: E731
        print(f"  {h:>3} | {raw:+8.3f} | {f(p_u):>10} | {f(p_v):>10} "
              f"| {f(p_a):>10} | {na:>5}")
        if ci_a is not None:
            print(f"      {'':>8}   {'':>10}   {'':>10}   "
                  f"[{ci_a[0]:+.3f}, {ci_a[1]:+.3f}]")

    keep = [r for r in rows if r[4] is not None]
    if keep:
        rr = np.mean([r[1] for r in keep])
        pp = np.mean([r[4] for r in keep])
        print(f"\n  平均：原始 {rr:+.3f} → 控制三者後 {pp:+.3f}"
              f"（保留 {pp/rr*100:.0f}%）" if rr else "")
        print("  判讀：保留大半 = 撤單流帶自己的資訊；塌向零 = 它只是活躍度代理。")
    print("\n  註：這不是預註冊測試，不改變 2026-08-10 的 PASS 門檻"
          "（那由 cancel_shock_ic.py 的凍結定義決定）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
