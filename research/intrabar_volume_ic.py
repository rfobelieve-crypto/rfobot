# -*- coding: utf-8 -*-
"""Conditional-IC screen for INTRA-BAR volume distribution.

Hypothesis source: an external order-flow course (XROOM「硬核邏輯」Part.6
假突破, transcribed 2026-07-28). Its one precise, falsifiable claim:

    「假突破…你要去看它的量的堆疊,它的量的爆發點在哪裡。」
    「以假突破來講,你會發現量第一個他都集中在上面…那就代表說那是純止損。」

i.e. a rally whose volume clusters at the TOP of the range is short stops being
force-bought (嘎空), not conviction — nobody with size accumulates at the high.
Read as a directional prediction: a high volume centroid should carry NEGATIVE
IC with the forward return.

Why this is worth testing when four previous feature hunts all failed
(21 liquidity proxies, WQ101, resistance map, DVOL): those were all rebuilt
from the same 1h OHLCV / Coinglass series V7 already consumes, so redundancy
was structural. This uses MINUTE resolution — the hourly bar destroys where
inside itself the volume happened, so V7 has never seen this. Information
discarded by aggregation, not information repackaged. Whether that survives
the jump to a 4h decision horizon is exactly what this measures.

Discipline (mistake.md 2026-06-01 / 2026-06-02):
  * the decision metric is CONDITIONAL IC against V7's walk-forward OOS
    residual, never raw IC;
  * per-month consistency is reported, because a pooled IC over 5 months can
    be carried by one stretch;
  * surviving here only earns an ensemble A/B, which has its own 4 gates.

Alignment (verified, 400/400 exact): a features_all timestamp T is the hour
bar's OPEN; its close is the 1-minute close at T+59min. So bar T is built from
minutes [T, T+59] and is fully known at T+60min — the same instant close[T] is
struck and the forward target begins. No look-ahead.

Run: python research/intrabar_volume_ic.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MINUTES = ROOT / "research/results/binance_1m_oos_window.parquet"
OOS = ROOT / "research/results/dual_model/direction_reg_oos_mse.parquet"
OUT = ROOT / "research/results/intrabar_volume_ic.json"

MIN_N = 300
BOOT = 2000
SEED = 42
NB = 10          # price buckets for the concentration measure


def build(m: pd.DataFrame) -> pd.DataFrame:
    """One row per hour bar, describing where inside it the volume sat."""
    m = m.copy()
    m["bar"] = m.index.floor("1h")
    # Where this minute's trading happened, in price terms.
    m["tp"] = (m["high"] + m["low"] + m["close"]) / 3.0

    g = m.groupby("bar")
    hi, lo = g["high"].max(), g["low"].min()
    rng = (hi - lo).replace(0, np.nan)

    m = m.join(hi.rename("_hi"), on="bar").join(lo.rename("_lo"), on="bar")
    m["_rng"] = (m["_hi"] - m["_lo"]).replace(0, np.nan)
    # 0 = bottom of the hour's range, 1 = top.
    m["pos"] = ((m["tp"] - m["_lo"]) / m["_rng"]).clip(0, 1)

    vol, tb = m["volume"], m["taker_buy_volume"]
    m["_wv"] = vol * m["pos"]
    m["_wt"] = tb * m["pos"]
    g = m.groupby("bar")

    out = pd.DataFrame(index=hi.index)
    sv, st = g["volume"].sum(), g["taker_buy_volume"].sum()
    out["vol_centroid"] = g["_wv"].sum() / sv.replace(0, np.nan)
    out["tb_centroid"] = g["_wt"].sum() / st.replace(0, np.nan)

    top = m[m["pos"] >= 2 / 3].groupby("bar")["volume"].sum()
    bot = m[m["pos"] <= 1 / 3].groupby("bar")["volume"].sum()
    out["vol_top3_frac"] = (top.reindex(out.index).fillna(0) / sv).astype(float)
    out["vol_bot3_frac"] = (bot.reindex(out.index).fillna(0) / sv).astype(float)
    out["vol_top_minus_bot"] = out["vol_top3_frac"] - out["vol_bot3_frac"]

    # Where the bar actually closed, so the centroid can be read RELATIVE to the
    # outcome: "volume piled at the extreme the bar ran to" is the claim, and a
    # bar that closes at its high with a high centroid is a different animal
    # from one that closes mid-range.
    close_pos = ((g["close"].last() - lo) / rng).clip(0, 1)
    out["close_pos"] = close_pos
    out["centroid_minus_close"] = out["vol_centroid"] - close_pos

    # Aggressive buying sitting higher in the range than volume as a whole =
    # the chasing is happening at the top.
    out["tb_minus_vol_centroid"] = out["tb_centroid"] - out["vol_centroid"]

    # How peaked the distribution is (Herfindahl over NB price buckets):
    # a single explosive print vs volume spread through the range.
    m["_b"] = (m["pos"] * NB).clip(0, NB - 1e-9).astype(int)
    bs = m.groupby(["bar", "_b"])["volume"].sum()
    tot = bs.groupby(level=0).sum()
    out["vol_hhi"] = ((bs / tot) ** 2).groupby(level=0).sum()

    out["n_min"] = g.size()
    return out


def boot_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n, acc = len(x), []
    for _ in range(BOOT):
        i = rng.integers(0, n, n)
        r = spearmanr(x[i], y[i]).correlation
        if np.isfinite(r):
            acc.append(r)
    if not acc:
        return (np.nan, np.nan)
    return float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


def screen(df: pd.DataFrame, cols: list[str], label: str) -> list[dict]:
    print(f"\n===== {label}  (n={len(df)}) =====")
    print(f"{'feature':<26}{'n':>6}{'raw IC':>9}{'cond IC':>9}"
          f"{'  95% CI (cond)':>21}{'  月一致':>9}")
    print("-" * 82)
    rows = []
    for c in cols:
        sub = df[[c, "y", "resid"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < MIN_N:
            print(f"{c:<26}{len(sub):>6}  (樣本不足，略過)")
            continue
        raw = spearmanr(sub[c], sub["y"]).correlation
        cond = spearmanr(sub[c], sub["resid"]).correlation
        lo, hi = boot_ci(sub[c].values, sub["resid"].values)
        by_m = sub.groupby(sub.index.to_period("M")).apply(
            lambda g: spearmanr(g[c], g["resid"]).correlation
            if len(g) > 50 else np.nan).dropna()
        cons = float((np.sign(by_m) == np.sign(cond)).mean()) if len(by_m) else np.nan
        sig = np.isfinite(lo) and np.isfinite(hi) and lo * hi > 0
        rows.append(dict(scope=label, feature=c, n=len(sub), raw_ic=raw,
                         cond_ic=cond, ci_lo=lo, ci_hi=hi, months=len(by_m),
                         consistency=cons, significant=bool(sig)))
        print(f"{c:<26}{len(sub):>6}{raw:>9.3f}{cond:>9.3f}"
              f"   [{lo:+.3f},{hi:+.3f}]{'*' if sig else ' '}{cons:>8.0%}")
    return rows


def main() -> int:
    m = pd.read_parquet(MINUTES)
    oos = pd.read_parquet(OOS)
    oos.index = pd.DatetimeIndex(oos.index)
    if oos.index.tz is not None:
        oos.index = oos.index.tz_convert("UTC").tz_localize(None)
    oos = oos.sort_index()

    feat = build(m)
    print(f"minute bars : {len(m):,}  {m.index.min()} -> {m.index.max()}")
    print(f"hour bars   : {len(feat):,}  (median {feat['n_min'].median():.0f} min/bar)")

    # Only fully-populated hours: a bar assembled from a handful of minutes has
    # a centroid that means nothing.
    feat = feat[feat["n_min"] >= 55]

    df = feat.join(
        pd.DataFrame({"y": oos["y_path_ret_4h"],
                      "resid": oos["y_path_ret_4h"] - oos["pred_ret"]}),
        how="inner").dropna(subset=["y", "resid"])
    print(f"overlap     : {len(df):,} bars  {df.index.min()} -> {df.index.max()}")

    cols = ["vol_centroid", "tb_centroid", "vol_top3_frac", "vol_bot3_frac",
            "vol_top_minus_bot", "centroid_minus_close",
            "tb_minus_vol_centroid", "vol_hhi", "close_pos"]
    rows = screen(df, cols, "全部小時棒")

    # The claim is specifically about BREAKOUTS. Restricting to bars that make a
    # new 24h high is the faithful test, but it is also a second look at the
    # same data, so it is reported separately and read with that in mind.
    hi24 = df["close_pos"].copy() * np.nan
    px = m["high"].groupby(m.index.floor("1h")).max()
    roll_max = px.shift(1).rolling(24, min_periods=24).max()
    is_brk = (px > roll_max).reindex(df.index).fillna(False)
    print(f"\n突破棒 (創 24h 新高): {int(is_brk.sum())} / {len(df)}")
    rows += screen(df[is_brk.values], cols, "僅突破棒")

    print("\n判讀:cond IC 才是決策依據。CI 不含 0 (*) 且月一致性 >= 0.67")
    print("才值得進 ensemble A/B;raw IC 高但 cond IC ~0 = V7 已吃過這資訊。")
    print("課程主張 vol_centroid 應為【負】cond IC(量堆在頂部 → 後續下跌)。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(generated=str(pd.Timestamp.utcnow()),
                                   n_bars=len(df), results=rows),
                              indent=2, default=str), encoding="utf-8")
    print(f"\nsaved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
