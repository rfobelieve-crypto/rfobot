"""Exit-mechanism attribution — what does each exit rule actually contribute?

The standing numbers (opp_signal 85.7% WR vs trail_stop 37%) cannot answer
this: a trade REACHES trail_stop precisely because no opposite signal ever
arrived, and reaches opp_signal because one did.  Those are two different
populations, so the comparison measures selection, not exit quality.

The only honest form is counterfactual: take ONE set of entries and replay
FOUR exit policies over it.

  A  trail-only           3xATR trailing, nothing else
  B  trail + opp_signal   (the executor before 2026-07-25)
  C  trail + decay        (decay armed, opp as fallback)
  D  trail + decay + opp  (the executor today)

Marginal contribution of a mechanism = its policy minus the policy without
it, on the SAME trades.  Everything else — entry rule, fees, bar semantics —
is held identical, so any difference is the exit and only the exit.

Faithful to indicator/okx/executor.py (read 2026-08-19):
  entry     Strong only, one position at a time, fill at next bar's open
  trail     stop_dist = 3 x ATR14 at entry; extreme ratchets on each bar's
            high/low; intrabar hit exits at the stop price
  decay     exits when the entry model's pred_ret has disagreed with the
            position's side for N consecutive bars (N=2 live), evaluated at
            bar close, BEFORE opp_signal
  opp       exits at close when a Strong/Moderate signal of the opposite
            direction prints
  fees      10 bps round-trip (the 2026-07-06 real-ruler constant)

Era split matters: opp_signal cannot fire without a two-sided signal
stream, and the decode was arithmetically one-sided until DECODE_EPOCH.
Pooling the eras would credit opp_signal with a starvation that was an
instrument defect.  Reported separately.

Read-only research code — reads indicator_history, writes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRAIL_MULT = 3.0
ATR_N = 14
FEE_BPS = 10.0
DECAY_BARS = 2          # live value (OKX_CONVICTION_DECAY_BARS)


def load_bars():
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT dt, open, high, low, close, pred_return_4h, "
                "pred_direction_code, strength_code, model_version "
                "FROM indicator_history WHERE close IS NOT NULL "
                "ORDER BY dt")
            return cur.fetchall()
    finally:
        conn.close()


def atr14(bars):
    out = [0.0] * len(bars)
    trs = []
    for i, b in enumerate(bars):
        if i == 0:
            tr = float(b["high"]) - float(b["low"])
        else:
            pc = float(bars[i - 1]["close"])
            tr = max(float(b["high"]) - float(b["low"]),
                     abs(float(b["high"]) - pc), abs(float(b["low"]) - pc))
        trs.append(tr)
        if i >= ATR_N - 1:
            out[i] = sum(trs[i - ATR_N + 1:i + 1]) / ATR_N
    return out


def direction_of(b) -> str:
    code = b["pred_direction_code"]
    if code is None:
        return "NEUTRAL"
    c = int(code)
    return "UP" if c > 0 else ("DOWN" if c < 0 else "NEUTRAL")


def is_strong(b) -> bool:
    return b["strength_code"] is not None and int(b["strength_code"]) >= 3


def fires(b) -> bool:
    """Strong or Moderate — the tiers that trigger opp_signal."""
    return b["strength_code"] is not None and int(b["strength_code"]) >= 2


def replay(bars, atr, use_decay: bool, use_opp: bool):
    """One pass of the executor's loop under a given exit policy."""
    trades = []
    pos = None
    for i in range(ATR_N, len(bars) - 1):
        b = bars[i]
        if pos is not None:
            side, entry_px, stop_dist = pos["side"], pos["entry"], pos["dist"]
            hi, lo = float(b["high"]), float(b["low"])
            close = float(b["close"])
            # trailing stop: ratchet, then intrabar hit
            if side == "LONG":
                pos["ext"] = max(pos["ext"], hi)
                stop = pos["ext"] - stop_dist
                hit = lo <= stop
            else:
                pos["ext"] = min(pos["ext"], lo)
                stop = pos["ext"] + stop_dist
                hit = hi >= stop
            exit_px, reason = None, None
            if hit:
                exit_px, reason = stop, "trail_stop"
            else:
                pr = b["pred_return_4h"]
                if use_decay and pr is not None:
                    disagree = ((side == "LONG" and float(pr) < 0)
                                or (side == "SHORT" and float(pr) > 0))
                    pos["streak"] = pos["streak"] + 1 if disagree else 0
                    if pos["streak"] >= DECAY_BARS:
                        exit_px, reason = close, "conviction_decay"
                if exit_px is None and use_opp and fires(b):
                    d = direction_of(b)
                    if ((side == "LONG" and d == "DOWN")
                            or (side == "SHORT" and d == "UP")):
                        exit_px, reason = close, "opp_signal"
            if exit_px is not None:
                sgn = 1 if side == "LONG" else -1
                gross = sgn * (exit_px - entry_px) / entry_px * 1e4
                trades.append({"entry_dt": pos["dt"], "side": side,
                               "net_bps": gross - FEE_BPS, "reason": reason})
                pos = None
        if pos is None and is_strong(b):
            d = direction_of(b)
            if d in ("UP", "DOWN") and atr[i] > 0:
                nxt = bars[i + 1]
                entry = float(nxt["open"])
                pos = {"side": "LONG" if d == "UP" else "SHORT",
                       "entry": entry, "dist": TRAIL_MULT * atr[i],
                       "ext": entry, "streak": 0, "dt": nxt["dt"]}
    return trades


def stats(trades):
    if not trades:
        return None
    n = len(trades)
    net = [t["net_bps"] for t in trades]
    mean = sum(net) / n
    wr = 100 * sum(1 for x in net if x > 0) / n
    eq, peak, mdd = 0.0, 0.0, 0.0
    for x in net:
        eq += x
        peak = max(peak, eq)
        mdd = min(mdd, eq - peak)
    mix = {}
    for t in trades:
        mix[t["reason"]] = mix.get(t["reason"], 0) + 1
    return {"n": n, "mean": mean, "wr": wr, "total": sum(net),
            "mdd": mdd, "mix": mix}


def show(label, s):
    if not s:
        print(f"  {label:<26} n=0")
        return
    mix = " ".join(f"{k}:{v}" for k, v in sorted(s["mix"].items()))
    print(f"  {label:<26} n={s['n']:>3}  net/trade {s['mean']:+7.1f} bps  "
          f"WR {s['wr']:>3.0f}%  總計 {s['total']:+8.0f}  MDD {s['mdd']:+7.0f}  [{mix}]")


def main():
    bars = load_bars()
    atr = atr14(bars)
    from indicator.model_version import DECODE_EPOCH
    cut = DECODE_EPOCH[:10]
    eras = [("全期", lambda b: True),
            ("解碼修法前", lambda b: str(b["dt"]) < DECODE_EPOCH),
            ("解碼修法後", lambda b: str(b["dt"]) >= DECODE_EPOCH)]
    policies = [("A 純 trail", False, False),
                ("B trail+opp", False, True),
                ("C trail+decay", True, False),
                ("D trail+decay+opp (現行)", True, True)]

    all_res = {}
    for era_name, era_f in eras:
        print(f"\n════ {era_name} ════")
        for pname, ud, uo in policies:
            tr = [t for t in replay(bars, atr, ud, uo) if era_f({"dt": t["entry_dt"]})]
            s = stats(tr)
            all_res[(era_name, pname)] = s
            show(pname, s)
        a = all_res[(era_name, "A 純 trail")]
        b = all_res[(era_name, "B trail+opp")]
        c = all_res[(era_name, "C trail+decay")]
        d = all_res[(era_name, "D trail+decay+opp (現行)")]
        if a and b:
            print(f"  → opp_signal 邊際貢獻 (B−A): "
                  f"{b['mean'] - a['mean']:+.1f} bps/trade")
        if a and c:
            print(f"  → conviction_decay 邊際貢獻 (C−A): "
                  f"{c['mean'] - a['mean']:+.1f} bps/trade")
        if c and d:
            print(f"  → 現行組合 vs 純 trail (D−A): "
                  f"{d['mean'] - a['mean']:+.1f} bps/trade")
    print(f"\n(entry: Strong-only next-open, 單倉; trail 3xATR; "
          f"decay N={DECAY_BARS}; fees {FEE_BPS:.0f} bps 來回; "
          f"era 界線 {cut})")


if __name__ == "__main__":
    main()
