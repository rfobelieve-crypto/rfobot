# -*- coding: utf-8 -*-
"""Phase 0a — 分鐘級 IC × horizon 衰退掃描 (see PREREG.md, frozen 2026-07-18).

兩個資料時代 (審計 2026-07-18):
  Era A  ohlcv_1m       2025-01-01 → 2026-03-30 (15mo, 100%, 退役線)
         taker 失衡由 klines 反推: delta = 2*taker_buy_quote - quote_vol
  Era B  flow_bars_1m('all') 2026-04-18 → now (90d, 99.8%, 現役)
         + orderbook mid/imbalance_l20 (05-11→) + liquidation_1m (0填)

特徵/horizon 集依 PREREG 凍結; 指標: 全期 Spearman IC + 逐月穩定 +
前後半同號 + top-5% 條件桶淨值(G2 預審, 30/60m, 扣 8bps)。
Usage: python research/subhourly/minute_ic_scan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from shared.db import get_db_conn

HORIZONS = (5, 15, 30, 60, 120, 240)          # frozen
G2_HORIZONS = (30, 60)
COST_BPS = 8.0                                 # 2x maker RT, frozen
TOP_FRAC = 0.05
OUT_CSV = Path("research/results/subhourly_ic_scan.csv")


def _q(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or None)
        return pd.DataFrame(cur.fetchall() or [])


def load_era_a() -> pd.DataFrame:
    conn = get_db_conn()
    try:
        df = _q(conn, "SELECT ts_open, close, quote_vol, taker_buy_quote "
                      "FROM ohlcv_1m WHERE symbol='BTC-USD' ORDER BY ts_open")
    finally:
        conn.close()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c])
    df["m"] = df["ts_open"] // 60_000
    df = df.groupby("m").last()
    df = df.reindex(pd.RangeIndex(df.index.min(), df.index.max() + 1))
    df["delta"] = 2 * df["taker_buy_quote"] - df["quote_vol"]
    df["vol"] = df["quote_vol"]
    df["px"] = df["close"]
    return df


def load_era_b() -> pd.DataFrame:
    conn = get_db_conn()
    try:
        fb = _q(conn, "SELECT window_start ms, delta_usd, volume_usd "
                      "FROM flow_bars_1m WHERE canonical_symbol='BTC-USD' "
                      "AND exchange_scope='all' ORDER BY window_start")
        ob = _q(conn, "SELECT ts_ms, mid_price, imbalance_l20 "
                      "FROM orderbook_snapshots_1m "
                      "WHERE canonical_symbol='BTC-USD' ORDER BY ts_ms")
        lq = _q(conn, "SELECT window_start ms, liq_buy_usd, liq_sell_usd, "
                      "liq_total_usd FROM liquidation_1m "
                      "WHERE canonical_symbol='BTC-USD' ORDER BY window_start")
    finally:
        conn.close()
    for f in (fb, ob, lq):
        for c in f.columns:
            f[c] = pd.to_numeric(f[c])
    fb["m"] = fb["ms"] // 60_000
    df = fb.groupby("m").last()[["delta_usd", "volume_usd"]]
    df.columns = ["delta", "vol"]
    ob["m"] = ob["ts_ms"] // 60_000
    df = df.join(ob.groupby("m").last()[["mid_price", "imbalance_l20"]],
                 how="left")
    lq["m"] = lq["ms"] // 60_000
    df = df.join(lq.groupby("m").last()[["liq_buy_usd", "liq_sell_usd",
                                         "liq_total_usd"]], how="left")
    df = df.reindex(pd.RangeIndex(df.index.min(), df.index.max() + 1))
    for c in ("liq_buy_usd", "liq_sell_usd", "liq_total_usd"):
        df[c] = df[c].fillna(0.0)
    df["px"] = df["mid_price"]
    df["obi"] = df["imbalance_l20"]
    return df


def build_features(df: pd.DataFrame, with_extras: bool) -> pd.DataFrame:
    """PREREG frozen families. trailing-only; min_periods = full window."""
    out = pd.DataFrame(index=df.index)
    d, v, px = df["delta"], df["vol"], df["px"]
    std24 = d.rolling(1440, min_periods=720).std()
    for k in (5, 15, 60):
        sd = d.rolling(k, min_periods=k).sum()
        out[f"ti_{k}"] = sd / v.rolling(k, min_periods=k).sum().replace(0, np.nan)
        out[f"dz_{k}"] = sd / (std24 * np.sqrt(k)).replace(0, np.nan)
        out[f"ret_{k}"] = px / px.shift(k) - 1
    out["vshock_60"] = v / v.rolling(60, min_periods=60).median().replace(0, np.nan)
    if with_extras:
        out["obi"] = df["obi"]
        lt = df["liq_total_usd"]
        base = lt.rolling(1440, min_periods=720).mean()
        out["liq_z15"] = (lt.rolling(15, min_periods=15).sum()
                          / (15 * base).replace(0, np.nan))
        net = df["liq_buy_usd"] - df["liq_sell_usd"]     # 語義以 IC 符號為準
        out["liq_dir15"] = (net.rolling(15, min_periods=15).sum()
                            / lt.rolling(15, min_periods=15).sum().replace(0, np.nan))
    out["px"] = px
    return out


def scan(feat: pd.DataFrame, era: str) -> list[dict]:
    px = feat["px"]
    months = pd.Series(
        pd.to_datetime(feat.index * 60_000, unit="ms").strftime("%Y-%m"),
        index=feat.index)
    rows = []
    fcols = [c for c in feat.columns if c != "px"]
    for h in HORIZONS:
        tgt = (px.shift(-h) / px - 1) * 10_000            # bps
        for f in fcols:
            sub = pd.DataFrame({"x": feat[f], "y": tgt, "mo": months}).dropna()
            if len(sub) < 20_000:
                continue
            ic_all, _ = spearmanr(sub["x"], sub["y"])
            mo_ics = []
            for mo, g in sub.groupby("mo"):
                if len(g) >= 5_000:
                    ic, _ = spearmanr(g["x"], g["y"])
                    mo_ics.append(ic)
            if len(mo_ics) < 2:
                continue
            mo_mean = float(np.mean(mo_ics))
            sign_share = float(np.mean([np.sign(x) == np.sign(mo_mean)
                                        for x in mo_ics]))
            mid = len(sub) // 2
            ic_h1, _ = spearmanr(sub["x"].iloc[:mid], sub["y"].iloc[:mid])
            ic_h2, _ = spearmanr(sub["x"].iloc[mid:], sub["y"].iloc[mid:])
            halves = bool(np.sign(ic_h1) == np.sign(ic_h2))

            top_net = top_lo = np.nan
            if h in G2_HORIZONS:
                q = sub["x"].abs().quantile(1 - TOP_FRAC)
                pick = sub[sub["x"].abs() >= q]
                sgn = np.sign(ic_all) if ic_all != 0 else 1.0
                pnl = sgn * np.sign(pick["x"]) * pick["y"] - COST_BPS
                top_net = float(pnl.mean())
                rng = np.random.default_rng(42)
                bs = [pnl.sample(len(pnl), replace=True,
                                 random_state=int(rng.integers(1e9))).mean()
                      for _ in range(500)]
                top_lo = float(np.quantile(bs, 0.025))
            rows.append(dict(era=era, feature=f, h=h, n=len(sub),
                             ic_all=round(float(ic_all), 4),
                             mo_mean_ic=round(mo_mean, 4),
                             n_months=len(mo_ics),
                             mo_sign_share=round(sign_share, 2),
                             halves_agree=halves,
                             top5_net_bps=None if np.isnan(top_net) else round(top_net, 2),
                             top5_ci_lo=None if np.isnan(top_lo) else round(top_lo, 2)))
    return rows


def main() -> int:
    all_rows = []
    for era, loader, extras in (("A_ohlcv15mo", load_era_a, False),
                                ("B_flow90d", load_era_b, True)):
        df = loader()
        feat = build_features(df, with_extras=extras)
        n_px = int(feat["px"].notna().sum())
        print(f"\n=== Era {era}: {n_px:,} 分鐘有價格 ===")
        rows = scan(feat, era)
        all_rows += rows
        for h in HORIZONS:
            hr = [r for r in rows if r["h"] == h]
            if not hr:
                continue
            hr.sort(key=lambda r: -abs(r["mo_mean_ic"]))
            print(f"\n-- h={h}m --")
            print(f"{'feature':10s} {'IC全期':>7s} {'月均IC':>7s} {'月同號':>6s} "
                  f"{'半同號':>4s} {'top5%淨(bps)':>12s} {'CI下緣':>7s}")
            for r in hr:
                t5 = ("" if r["top5_net_bps"] is None
                      else f"{r['top5_net_bps']:>+11.2f} {r['top5_ci_lo']:>+7.2f}")
                print(f"{r['feature']:10s} {r['ic_all']:>+7.3f} "
                      f"{r['mo_mean_ic']:>+7.3f} "
                      f"{r['mo_sign_share']:>5.0%}/{r['n_months']:<2d} "
                      f"{'同' if r['halves_agree'] else '反':>4s} {t5}")
    out = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nsaved: {OUT_CSV}  ({len(out)} rows)")

    # G1 判定 (PREREG): h∈{30,60} 月均|IC|>=0.03 且 月同號>=70% 且 前後半同號
    g1 = out[(out["h"].isin(G2_HORIZONS)) & (out["mo_mean_ic"].abs() >= 0.03)
             & (out["mo_sign_share"] >= 0.70) & out["halves_agree"]]
    print("\n=== G1 判定 (凍結門檻) ===")
    if g1.empty:
        print("G1 FAIL — 無特徵在 30/60m 通過 (月均|IC|>=0.03 + 70%月同號 + 半同號)")
    else:
        for _, r in g1.iterrows():
            print(f"G1 PASS: {r['era']} {r['feature']} h={r['h']} "
                  f"月均IC={r['mo_mean_ic']:+.3f} 同號{r['mo_sign_share']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
