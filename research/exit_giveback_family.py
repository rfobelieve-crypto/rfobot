# -*- coding: utf-8 -*-
"""回吐家族：三種「換家族」的出場，對準的是回吐不是 EV（預註冊 2026-09-05）

**使用者**：「好幾次 V7 賺蠻多錢，但因為出場機制回吐或虧錢。如果回吐這部分拿
70% 回來，整個 V7 應該都會不一樣。真的不行就只能手動出場，但那不是我的目的。」

**回吐的定義（寫死）**：以**現行 trailing 的持倉窗口**為準，MFE ＝ 該窗口內對
部位最有利的極值（bps，high/low 路徑）；某臂的 **回吐 ＝ MFE − 該臂的實現淨值**。
這樣三臂比的是同一筆「桌上曾經有的錢」。**回吐拿回率 ＝ 1 − 該臂回吐 / trailing 回吐。**

**已試過的家族**（全 NO-GO）都是「一條規則決定整個部位何時全出」：trail 鬆緊、
breakeven、時間上限、反向訊號、decay、meta、RL。這裡的三臂各自換一個家族：
  B1 分批   50% 在第 4 根 bar 收盤鎖定（模型視野），50% 照 trailing。兩次半倉出場
            的 taker 費合計 ＝ 一次全倉，成本同 9 bps 來回。
  B2 幅度   停利在 entry × (1 + side × k × mag_pred)，k = 1.0（模型自己說會走多遠）；
            用 high/low 觸發、以停利價成交；沒碰到就照 trailing。mag_pred 取訊號
            bar 的值（indicator_history），單位是報酬小數（Strong 中位 0.76%）。
  B3 翻號   持倉中 pred_return_4h 對部位翻號（任何幅度），下一根 bar 開盤出場
            （因果：翻號在收盤才知道）；另保留 trailing 停損當保護。
基準 A ＝ §0.88e 的 trailing（3×ATR＋反向 Strong＋72h）。同一批低頻 regime Strong
（2026-04-03 起，n≈125）、同一套 1h K 線、每筆獨立、進場 label+1h 開盤。

**判準（寫死；問的是「同樣的 EV、少一半樂透」，不是「EV 更高」）**，每臂各判：
  (a) 配對均值差（臂 − trailing）> −5 bps（EV 不得明顯變差）
  (b) **回吐拿回率 ≥ 50%**，且前後兩半皆為正
  (c) 中位淨值 > trailing 中位 **且** 最差單筆 > trailing 最差
  (d) 回吐差的日區塊 bootstrap 95% CI 不含零
  全過 = GO。使用者的 **70%** 另外報，當作目標不當門檻（門檻不能是願望）。
**預測**：B1 拿回 ~50%、EV 掉 ~10（最有機會過 b/c、最可能卡 a）；B2 看 mag 準不準，
不知道；B3 開火太勤（pred 23% 的 bar 翻號），可能退化成短時間出場、EV 掉很多。

Run: python research/exit_giveback_family.py
Out: research/results/exit_giveback_family.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.db import get_db_conn  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

CSV = ROOT / "research" / "sweep_failure" / ".cache" / "BTCUSDT_1h.csv"
OUT = ROOT / "research" / "results" / "exit_giveback_family.json"
COST, TRAIL, CAP_H, HOLD4, K_MAG = 9.0, 3.0, 72, 4, 1.0


def bars():
    rows = list(csv.DictReader(open(CSV, newline="")))
    df = pd.DataFrame({k: [float(r[k]) for r in rows] for k in ("open", "high", "low", "close")})
    df.index = pd.to_datetime([int(r["time"]) for r in rows], unit="s", utc=True)
    tr = np.maximum(df["high"] - df["low"], np.maximum((df["high"] - df["close"].shift()).abs(),
                                                        (df["low"] - df["close"].shift()).abs()))
    df["atr"] = tr.rolling(14).mean()
    return df


def db():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT signal_time, direction FROM tracked_signals WHERE strength='Strong' "
                        "AND signal_time>='2026-04-03' ORDER BY signal_time")
            s = pd.DataFrame(cur.fetchall())
            cur.execute("SELECT dt, pred_return_4h p, mag_pred m FROM indicator_history "
                        "WHERE dt>='2026-04-01' AND pred_return_4h IS NOT NULL ORDER BY dt")
            h = pd.DataFrame(cur.fetchall())
    finally:
        conn.close()
    s["t"] = pd.to_datetime(s["signal_time"], utc=True)
    s["side"] = np.where(s["direction"].astype(str).str.upper().str.startswith("UP"), 1, -1)
    h["dt"] = pd.to_datetime(h["dt"], utc=True)
    h = h.set_index("dt"); h["p"] = h["p"].astype(float); h["m"] = h["m"].astype(float)
    return s[["t", "side"]], h


def trail_leg(df, i0, side, opp_fire, tp=None, flip=None):
    """Trailing with optional TP (price) and optional sign-flip series.
    Returns (net_bps_gross, exit_k, reason)."""
    entry = df["open"].iloc[i0]; atr = df["atr"].iloc[i0 - 1]
    stop = entry - side * TRAIL * atr; ext = entry
    for k in range(i0, min(i0 + CAP_H, len(df))):
        t = df.index[k]
        if k > i0 and t in opp_fire:
            return side * (df["open"].iloc[k] / entry - 1) * 1e4, k, "opp_signal"
        if flip is not None and k > i0 and flip.get(df.index[k - 1], False):
            return side * (df["open"].iloc[k] / entry - 1) * 1e4, k, "sign_flip"
        hi, lo = df["high"].iloc[k], df["low"].iloc[k]
        # same bar hits both TP and stop: unknowable order -> assume stop (conservative)
        stop_hit = (side > 0 and lo <= stop) or (side < 0 and hi >= stop)
        tp_hit = tp is not None and ((side > 0 and hi >= tp) or (side < 0 and lo <= tp))
        if stop_hit:
            return side * (stop / entry - 1) * 1e4, k, "trail_stop"
        if tp_hit:
            return side * (tp / entry - 1) * 1e4, k, "mag_tp"
        if side > 0 and hi > ext:
            ext = hi; stop = max(stop, ext - TRAIL * atr)
        elif side < 0 and lo < ext:
            ext = lo; stop = min(stop, ext + TRAIL * atr)
    k = min(i0 + CAP_H, len(df)) - 1
    return side * (df["close"].iloc[k] / entry - 1) * 1e4, k, "time_cap"


def mfe(df, i0, k_exit, side):
    entry = df["open"].iloc[i0]
    seg = df.iloc[i0:k_exit + 1]
    best = seg["high"].max() if side > 0 else seg["low"].min()
    return max(0.0, side * (best / entry - 1) * 1e4)


def dblock(v, days, B=3000, seed=5):
    rng = np.random.default_rng(seed); g = {}
    for x, d in zip(v, days):
        g.setdefault(d, []).append(x)
    ks = np.array(list(g))
    out = [np.concatenate([g[d] for d in rng.choice(ks, len(ks))]).mean() for _ in range(B)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    df = bars(); s, h = db()
    up_t = set(s[s.side > 0]["t"]); dn_t = set(s[s.side < 0]["t"])
    rows = []
    for _, r in s.iterrows():
        t_e = r["t"] + pd.Timedelta(hours=1)
        if t_e not in df.index or r["t"] not in h.index:
            continue
        i0 = df.index.get_loc(t_e)
        if i0 < 20 or i0 + CAP_H >= len(df):
            continue
        side = r["side"]
        opp = {x + pd.Timedelta(hours=1) for x in (dn_t if side > 0 else up_t)}
        entry = df["open"].iloc[i0]
        # A trailing
        a, ka, wa = trail_leg(df, i0, side, opp)
        M = mfe(df, i0, ka, side)
        # B1 split: half at bar +3 close, half trailing
        k4 = min(i0 + HOLD4 - 1, len(df) - 1)
        leg4 = side * (df["close"].iloc[k4] / entry - 1) * 1e4
        b1 = 0.5 * leg4 + 0.5 * a
        # B2 mag TP
        mp = float(h.loc[r["t"], "m"]); tp = entry * (1 + side * K_MAG * mp)
        b2, k2, w2 = trail_leg(df, i0, side, opp, tp=tp)
        # B3 sign flip (any magnitude), decided at bar close, exit next open
        win = h["p"].reindex(df.index[i0:i0 + CAP_H])
        flip = {t: (np.sign(v) == -side) for t, v in win.items() if pd.notna(v)}
        b3, k3, w3 = trail_leg(df, i0, side, opp, flip=flip)
        rows.append({"day": int(t_e.timestamp() // 86400), "side": side, "mfe": M,
                     "A": a - COST, "B1": b1 - COST, "B2": b2 - COST, "B3": b3 - COST,
                     "wA": wa, "w2": w2, "w3": w3})
    d = pd.DataFrame(rows); n = len(d); half = n // 2
    for arm in ("A", "B1", "B2", "B3"):
        d["gb_" + arm] = d["mfe"] - d[arm]
    print("=" * 96)
    print(f"  回吐家族 · 低頻 regime Strong n={n} · 同一批訊號配對 · 回吐 = MFE(trailing 窗) − 實現淨值")
    print("=" * 96)
    print(f"  MFE 平均 {d.mfe.mean():+.0f} bps，中位 {d.mfe.median():+.0f}（桌上曾經有的錢）\n")
    print(f"  {'臂':4}{'淨均值':>8}{'中位':>8}{'WR':>7}{'最差':>8}{'p5':>8}{'回吐均值':>10}{'拿回率':>8}{'兩半拿回':>14}")
    res = {}
    gbA = d["gb_A"].mean()
    for arm, lab in (("A", "trailing"), ("B1", "分批"), ("B2", "幅度停利"), ("B3", "翻號")):
        v = d[arm]; gb = d["gb_" + arm]
        rec = 1 - gb.mean() / gbA if arm != "A" else 0.0
        r1 = 1 - gb.iloc[:half].mean() / d["gb_A"].iloc[:half].mean() if arm != "A" else 0.0
        r2 = 1 - gb.iloc[half:].mean() / d["gb_A"].iloc[half:].mean() if arm != "A" else 0.0
        res[arm] = {"mean": float(v.mean()), "median": float(v.median()), "wr": float((v > 0).mean()),
                    "worst": float(v.min()), "p5": float(np.percentile(v, 5)),
                    "giveback": float(gb.mean()), "recover": float(rec), "halves": [float(r1), float(r2)]}
        print(f"  {arm:4}{v.mean():>+8.1f}{v.median():>+8.1f}{(v>0).mean()*100:>6.1f}%{v.min():>+8.0f}"
              f"{np.percentile(v,5):>+8.0f}{gb.mean():>+10.1f}{rec*100:>7.0f}%{r1*100:>+7.0f}%/{r2*100:>+5.0f}%  {lab}")
    print(f"\n  出場理由  A {dict(d.wA.value_counts())}\n            B2 {dict(d.w2.value_counts())}\n            B3 {dict(d.w3.value_counts())}")
    print("\n  判準：(a) 均值差 > −5  (b) 拿回 ≥50% 且兩半皆正  (c) 中位&最差皆優於 A  (d) 回吐差 CI 離零")
    verdict = {}
    for arm in ("B1", "B2", "B3"):
        dm = (d[arm] - d["A"]).mean()
        gdiff = (d["gb_A"] - d["gb_" + arm]).values
        lo, hi = dblock(gdiff, d["day"].values)
        R = res[arm]
        ca = dm > -5; cb = R["recover"] >= 0.5 and R["halves"][0] > 0 and R["halves"][1] > 0
        cc = R["median"] > res["A"]["median"] and R["worst"] > res["A"]["worst"]; cd = lo > 0
        go = all((ca, cb, cc, cd)); verdict[arm] = {"go": go, "dmean": float(dm), "gb_ci": [lo, hi],
                                                   "c": [bool(x) for x in (ca, cb, cc, cd)]}
        print(f"  {arm}: 均值差 {dm:+6.1f} ({'過' if ca else '不過'})  拿回 {R['recover']*100:.0f}% ({'過' if cb else '不過'})"
              f"  中位/最差 ({'過' if cc else '不過'})  回吐差 CI [{lo:+.0f},{hi:+.0f}] ({'過' if cd else '不過'})"
              f"  → {'GO' if go else 'NO-GO'}   使用者目標 70%: {'達' if R['recover']>=0.7 else '未達'}")
    OUT.write_text(json.dumps({"n": n, "mfe_mean": float(d.mfe.mean()), "arms": res, "verdict": verdict},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
