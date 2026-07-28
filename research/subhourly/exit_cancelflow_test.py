"""Does cancel-flow tell a shake-out from a real reversal?

Runs the test pre-registered in PREREG_exit_cancelflow.md (committed a78b101
BEFORE any result here was seen). Every threshold below is copied from that
file; none may be tuned in response to what this prints.

The question is the one `trail_stop` currently answers with a fixed distance
and nothing else: price has retraced 3xATR off its extreme — is it going to
keep going, or come back? A fixed stop cannot tell those apart, which is the
mechanical reason trail_stop exits carry 37% WR in backtest and -0.68% mean
in live. If cancel-flow separates them, it is worth wiring to exits later.

Deliberately decoupled from trades: only 2 live positions fall inside the
depth window, so testing on real exits is impossible for another 3-5 months.
The underlying question does not need a position to ask.

Run: python research/subhourly/exit_cancelflow_test.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.db import get_db_conn  # noqa: E402

OUT = ROOT / "research/results/exit_cancelflow_test.json"

# ── pre-registered constants (PREREG_exit_cancelflow.md) ─────────────────
TRAIL_MULT = 3.0            # aligned to executor, not swept
EXTREME_WIN = 240           # minutes
HORIZONS = (30, 60, 120)    # all three reported, no cherry-picking
MIN_N = 200                 # below this: "insufficient sample", no verdict
MIN_EFFECT_PP = 5.0
BOOT = 2000
SEED = 42
SIDE_SHOCK = 3.0            # same trigger level the frozen playbooks use


EXCHANGE = "binance_perp"     # the venue the executor trades
SYMBOL = "BTC-USD"


def load_minutes() -> pd.DataFrame:
    """MUST filter exchange+symbol: depth_deltas_1m holds 16 venue/symbol
    combinations (~6 rows per minute). Reading it unfiltered blends DOGE,
    ETH, XRP … order books into one series and every feature is nonsense.
    """
    conn = get_db_conn()
    try:
        d = pd.read_sql(
            "SELECT minute_start_ms ms, bid_add_qty ba, bid_cancel_qty bc, "
            "ask_add_qty aa, ask_cancel_qty ac FROM depth_deltas_1m "
            "WHERE exchange=%s AND canonical_symbol=%s "
            "ORDER BY minute_start_ms", conn, params=(EXCHANGE, SYMBOL))
    finally:
        conn.close()
    d["dt"] = pd.to_datetime(d["ms"], unit="ms")
    d = d.set_index("dt").sort_index()
    if d.index.duplicated().any():
        raise RuntimeError(f"duplicate minutes after filtering: "
                           f"{int(d.index.duplicated().sum())}")
    return d


def fetch_1m(start_ms: int, end_ms: int) -> pd.DataFrame:
    import urllib.request
    rows, cur = [], start_ms
    while cur < end_ms:
        u = ("https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT"
             f"&interval=1m&startTime={cur}&endTime={min(cur + 1500 * 60000, end_ms)}"
             "&limit=1500")
        b = json.load(urllib.request.urlopen(u, timeout=30))
        if not b:
            break
        rows += b
        cur = b[-1][0] + 60_000
    df = pd.DataFrame(rows).iloc[:, :5]
    df.columns = ["ms", "open", "high", "low", "close"]
    df = df.drop_duplicates("ms")
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["dt"] = pd.to_datetime(df["ms"], unit="ms")
    return df.set_index("dt").sort_index()


def build_features(dd: pd.DataFrame) -> pd.DataFrame:
    """FROZEN v1 definitions (cancel_playbook_watcher.py header + :241)."""
    f = pd.DataFrame(index=dd.index)
    for col in ("ba", "bc", "aa", "ac"):
        base = dd[col].rolling(60, min_periods=30).median()
        f[f"{col}_shock"] = dd[col] / base.replace(0, np.nan)
    tot = dd["ac"] + dd["bc"]
    skew_raw = (dd["ac"] - dd["bc"]) / tot.replace(0, np.nan)
    f["skew15"] = (skew_raw - skew_raw.rolling(60, min_periods=30).mean()) \
        .rolling(15, min_periods=5).mean()
    net_raw = ((dd["ac"] - dd["aa"]) - (dd["bc"] - dd["ba"])) / tot.replace(0, np.nan)
    f["net15"] = (net_raw - net_raw.rolling(60, min_periods=30).mean()) \
        .rolling(15, min_periods=5).mean()
    return f


def atr_1m(px: pd.DataFrame) -> pd.Series:
    """1h Wilder-14 ATR, forward-filled onto minutes (same source as executor)."""
    h = px["high"].resample("1h").max()
    lo = px["low"].resample("1h").min()
    c = px["close"].resample("1h").last()
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()],
                   axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return a.reindex(px.index, method="ffill")


def find_events(px: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    """First minute at which price has retraced TRAIL_MULT x ATR off a rolling
    extreme. Each extreme fires at most once, so one retracement contributes
    one event rather than a run of correlated ones."""
    c = px["close"].values
    hi = px["high"].rolling(EXTREME_WIN, min_periods=60).max().values
    lo = px["low"].rolling(EXTREME_WIN, min_periods=60).min().values
    a = atr.values
    n = len(px)
    rows = []
    armed_hi = armed_lo = None
    for i in range(n):
        if not np.isfinite(a[i]) or a[i] <= 0 or not np.isfinite(hi[i]):
            continue
        d = TRAIL_MULT * a[i]
        # long context: pulled back off the high
        if c[i] <= hi[i] - d:
            if armed_hi != hi[i]:
                rows.append(dict(i=i, dt=px.index[i], ctx="LONG",
                                 extreme=hi[i], price=c[i]))
                armed_hi = hi[i]
        else:
            armed_hi = None
        # short context: bounced off the low
        if c[i] >= lo[i] + d:
            if armed_lo != lo[i]:
                rows.append(dict(i=i, dt=px.index[i], ctx="SHORT",
                                 extreme=lo[i], price=c[i]))
                armed_lo = lo[i]
        else:
            armed_lo = None
    return pd.DataFrame(rows)


def label(ev: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """RECOVER = price regains the extreme within M minutes."""
    hi = px["high"].values
    lo = px["low"].values
    n = len(px)
    for M in HORIZONS:
        out = []
        for _, r in ev.iterrows():
            i, j = int(r["i"]), min(int(r["i"]) + M, n - 1)
            if j <= i:
                out.append(np.nan)
                continue
            if r["ctx"] == "LONG":
                out.append(float(hi[i + 1:j + 1].max() >= r["extreme"]))
            else:
                out.append(float(lo[i + 1:j + 1].min() <= r["extreme"]))
        ev[f"recover_{M}"] = out
    return ev


def boot_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    d = [rng.choice(a, len(a), replace=True).mean()
         - rng.choice(b, len(b), replace=True).mean() for _ in range(BOOT)]
    return float(np.percentile(d, 2.5) * 100), float(np.percentile(d, 97.5) * 100)


def main() -> int:
    dd = load_minutes()
    print(f"depth_deltas {len(dd):,} min  {dd.index.min()} → {dd.index.max()}")
    px = fetch_1m(int(dd["ms"].min()), int(dd["ms"].max()) + 121 * 60_000)
    print(f"klines {len(px):,} min")
    feats = build_features(dd)
    atr = atr_1m(px)

    ev = find_events(px, atr)
    print(f"retracement events: {len(ev)}  "
          f"(LONG {int((ev['ctx'] == 'LONG').sum())} / "
          f"SHORT {int((ev['ctx'] == 'SHORT').sum())})")
    ev = label(ev, px)
    n_before = len(ev)
    # merge, not reindex: one minute can be both a LONG-context and a
    # SHORT-context event, so ev["dt"] legitimately has duplicates.
    ev = ev.merge(feats, left_on="dt", right_index=True, how="left")
    if len(ev) != n_before:
        raise RuntimeError(f"feature merge fanned out {n_before} -> {len(ev)}")
    for M in HORIZONS:
        col = f"recover_{M}"
        print(f"  label recover_{M}: {ev[col].mean() * 100:.1f}% "
              f"({int(ev[col].sum())}/{int(ev[col].notna().sum())})")

    # Pre-registered directional read: the side whose resting orders are being
    # pulled loses its wall. In a LONG context (pullback off a high) the wall
    # above is the ask; ac_shock high => resistance gone => expect RECOVER.
    groups = {
        "LONG  ac_shock>=3 (賣牆抽離→預期收復)":
            (ev["ctx"] == "LONG") & (ev["ac_shock"] >= SIDE_SHOCK),
        "LONG  bc_shock>=3 (買牆抽離→預期續跌)":
            (ev["ctx"] == "LONG") & (ev["bc_shock"] >= SIDE_SHOCK),
        "SHORT bc_shock>=3 (買牆抽離→預期收復)":
            (ev["ctx"] == "SHORT") & (ev["bc_shock"] >= SIDE_SHOCK),
        "SHORT ac_shock>=3 (賣牆抽離→預期續漲)":
            (ev["ctx"] == "SHORT") & (ev["ac_shock"] >= SIDE_SHOCK),
    }

    results, verdicts = [], []
    for name, mask in groups.items():
        ctx = "LONG" if name.startswith("LONG") else "SHORT"
        base_mask = ev["ctx"] == ctx
        print(f"\n── {name}")
        signs, oks = [], []
        for M in HORIZONS:
            col = f"recover_{M}"
            sub = ev[mask & ev[col].notna()][col].values
            base = ev[base_mask & ev[col].notna()][col].values
            if len(sub) < MIN_N:
                print(f"   +{M:>3}m  n={len(sub):<5} 樣本不足 (<{MIN_N})")
                oks.append(False)
                continue
            eff = (sub.mean() - base.mean()) * 100
            ci = boot_diff(sub, base)
            sig = ci[0] * ci[1] > 0
            ok = abs(eff) >= MIN_EFFECT_PP and sig
            signs.append(np.sign(eff))
            oks.append(ok)
            print(f"   +{M:>3}m  n={len(sub):<5} recover {sub.mean() * 100:>5.1f}% "
                  f"vs base {base.mean() * 100:>5.1f}%  差 {eff:+5.1f}pp  "
                  f"CI[{ci[0]:+.1f},{ci[1]:+.1f}]{'*' if sig else ' '}")
            results.append(dict(group=name, horizon=M, n=len(sub),
                                rate=float(sub.mean()), base=float(base.mean()),
                                effect_pp=float(eff), ci=ci, passes=bool(ok)))
        same_sign = len(signs) == len(HORIZONS) and len(set(signs)) == 1
        v = all(oks) and same_sign
        verdicts.append(v)
        print(f"   → {'PASS' if v else 'FAIL'}"
              f"{'' if same_sign else ' (三窗未同號)'}")

    print("\n=== 預先登記判準 (PREREG_exit_cancelflow.md, a78b101) ===")
    print("   效果量 >=5pp + bootstrap CI 不含 0 + 三窗同號 + 每組 n>=200")
    overall = "GO" if any(verdicts) else "NO-GO"
    print(f"\n   VERDICT: {overall}  ({sum(verdicts)}/{len(verdicts)} 組通過)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        generated=str(pd.Timestamp.utcnow()), prereg="a78b101",
        verdict=overall, n_events=len(ev), results=results),
        indent=2, default=str), encoding="utf-8")
    print(f"saved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
