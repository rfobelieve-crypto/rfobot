# -*- coding: utf-8 -*-
"""Portfolio clocks — one weekly scheduled run that keeps every waiting-state
check alive and REPORTS TO A HUMAN (mistake.md 2026-07-05: a schedule that
cannot reach the operator does not exist).

What it watches (all read-only; touches no production surface):
  1. Gate B counter (weekly) — closed live trades since 2026-06-23 (the
     post-trailing-fix clean cohort). At n>=30 push the unlock alert for the
     SHORT-tilt sizing gauntlet (once, state-marked).
  2. Monthly ritual (first run in each month's day 5-11 window) —
     fetch_klines -> sweep_forward (Gate F progress) + cross_asset_probe
     (informational forward cohort). STALE-DATA guard: if the freshly
     fetched BTC cache still ends >48h ago, report STALE and do NOT mark
     the month done (next weekly run retries).
  3. depth_deltas span (weekly) — when the table spans >=90 days, push the
     "subhourly revival re-run is due" alert (once).
  4. cancel_lead_ic / cancel_shock_ic checkpoint — on/after 2026-08-10 run
     both pre-registered verdict scripts and push their tails (marked done
     only when both exit 0).

Delivery: send_critical with 6x60s retries (the 2026-07-05 pattern); final
failure appends to research/results/portfolio_clocks.log so the miss leaves
a trace. The weekly cadence doubles as a heartbeat — a silent Monday IS the
alarm.

Scheduled via Windows Task Scheduler ("PortfolioClocks", MON 09:30):
    schtasks /Create /TN PortfolioClocks /SC WEEKLY /D MON /ST 09:30 ^
        /TR "C:\\...\\research\\portfolio_clocks.bat"
Manual test:  python research/portfolio_clocks.py --force-monthly
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
# local scheduled runs need .env in os.environ (TG token/chat) — the alerter
# reads os.environ directly, unlike shared/db which parses .env itself
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

STATE = ROOT / "research/results/portfolio_clocks_state.json"
LOG = ROOT / "research/results/portfolio_clocks.log"
GATE_B_START = "2026-06-23"          # first clean post-trailing-fix live trade
GATE_B_N = 30
DEPTH_DUE_DAYS = 90
CANCEL_VERDICT_DATE = "2026-08-10"


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} {msg}"
    print(line)
    try:
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def run(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        r = subprocess.run([sys.executable] + cmd, capture_output=True,
                           text=True, timeout=timeout, cwd=str(ROOT),
                           encoding="utf-8", errors="replace")
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT {timeout}s: {' '.join(cmd)}"
    except Exception as e:  # noqa: BLE001
        return -2, f"SPAWN FAIL: {e}"


def q1(sql: str, args: tuple = ()) -> dict | None:
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchone()
    finally:
        conn.close()


def grep_tail(text: str, needle: str, fallback: str) -> str:
    hits = [ln.strip() for ln in text.splitlines() if needle in ln]
    return hits[-1] if hits else fallback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-monthly", action="store_true")
    args = ap.parse_args()

    st = load_state()
    now = datetime.now(timezone.utc)
    lines: list[str] = [f"Portfolio clocks — {now:%Y-%m-%d} (weekly)"]
    errors: list[str] = []

    # 1 ── Gate B counter ────────────────────────────────────────────────
    try:
        r = q1("SELECT COUNT(*) n FROM v7_okx_positions "
               "WHERE status='CLOSED' AND entry_time>=%s "
               "AND (model_version IS NULL OR model_version NOT LIKE 'manual_test%%')",
               (GATE_B_START,))
        n = int(r["n"]) if r else -1
        lines.append(f"Gate B: {n}/{GATE_B_N} clean closed since {GATE_B_START}"
                     + ("  << UNLOCKED — run SHORT-tilt sizing gauntlet"
                        if n >= GATE_B_N and not st.get("gate_b_alerted")
                        else ""))
        if n >= GATE_B_N:
            st["gate_b_alerted"] = True
    except Exception as e:  # noqa: BLE001
        errors.append(f"gate_b: {e}")

    # 2 ── monthly Gate F + cross-asset ──────────────────────────────────
    month_key = f"{now:%Y-%m}"
    in_window = 5 <= now.day <= 11 or args.force_monthly
    if in_window and st.get("monthly_done") != month_key:
        rc_f, out_f = run(["research/sweep_failure/fetch_klines.py"], 600)
        stale = True
        try:
            import csv
            rows = list(csv.reader(open(
                ROOT / "research/sweep_failure/.cache/BTCUSDT_1h.csv",
                encoding="utf-8-sig")))
            last_ts = int(float(rows[-1][0]))
            age_h = (now.timestamp() - last_ts) / 3600
            stale = age_h > 48
        except Exception as e:  # noqa: BLE001
            errors.append(f"stale-check: {e}")
        if rc_f != 0 or stale:
            lines.append("monthly: STALE-DATA — fetch failed or cache >48h "
                         "old; Gate F NOT judged this run (will retry next "
                         "Monday)")
        else:
            rc1, out1 = run(["research/sweep_failure/sweep_forward.py"], 900)
            rc2, out2 = run(["research/sweep_failure/cross_asset_probe.py"], 900)
            lines.append("Gate F: " + grep_tail(out1, "Gate F progress",
                                                f"run failed rc={rc1}"))
            lines.append("x-asset fwd: " + grep_tail(out2, "pool    n=",
                                                     f"run failed rc={rc2}"))
            if rc1 == 0:
                st["monthly_done"] = month_key
            else:
                errors.append(f"sweep_forward rc={rc1}")
    elif st.get("monthly_done") == month_key:
        lines.append(f"monthly: done for {month_key}")
    else:
        lines.append("monthly: next window day 5-11")

    # 2b ── Variant B gate progress (weekly; reads the shadow CSV, no fetch) ──
    rcb, outb = run(["research/sweep_failure/shadow_engine.py", "--gate"], 120)
    lines.append(grep_tail(outb, "Variant B:", f"variant-B check failed rc={rcb}"))
    if rcb != 0:
        errors.append(f"variant_b rc={rcb}")

    # 2c ── frozen combo watchlist scoreboard (registered 2026-08-02) ────
    rcc, outc = run(["research/sweep_failure/shadow_engine.py", "--combos"], 120)
    if rcc == 0:
        lines.append("combo watchlist (forward):")
        lines += ["  " + ln.strip() for ln in outc.splitlines()
                  if ln.strip().startswith(("R", "PA", "V∧"))]
    else:
        errors.append(f"combo_watchlist rc={rcc}")

    # 2d ── V7 raid-chase veto forward gap (registered 2026-08-02) ──────
    # Two clocks since 2026-08-02 (TODO 0.49): the frozen Strong one and
    # the parallel Moderate one. grep_tail keeps only the last hit, so both
    # lines are collected explicitly — a weekly report that silently showed
    # one of two clocks would be the same failure mode as a green scheduler
    # over a dead job.
    rcv, outv = run(["research/v7_raid_veto.py", "--clock"], 180)
    if rcv == 0:
        hits = [ln.strip() for ln in outv.splitlines() if "V7 raid-veto" in ln]
        lines.extend(hits or ["raid-veto check failed"])
    else:
        errors.append(f"raid_veto rc={rcv}")

    # 2e ── V7 fired-direction balance (level-drift EARLY warning) ──────
    # 2026-08-08: the 05-01 model's output level drifted for months and
    # July fired 14 UP : 1 DOWN Strong while every rank-based check stayed
    # green (Spearman is shift-invariant). quarterly_revalidation 2b now
    # checks the pred level monthly; this weekly line watches the SYMPTOM
    # so the alarm rings within days, not at month-end.
    #
    # 2026-08-13: the window is also floored at DECODE_EPOCH.  Bars decoded
    # before it were ranked against an in-sample seed that put the DOWN cutoff
    # below the model's reachable output — a 95:5 split there is the FIXED
    # defect's echo, not evidence about the decode running now.  Left
    # unfloored this line would have paged every week until the 21d window
    # rolled past 08-12, training the operator to ignore it (and it is the
    # fast-path alarm for exactly this failure).  It goes quiet by design
    # until enough post-fix signals exist to judge.
    try:
        import shared.db as _sdb
        from indicator.model_version import DECODE_EPOCH, sample_floor
        _since = sample_floor(DECODE_EPOCH)
        _conn = _sdb.get_db_conn()
        try:
            with _conn.cursor() as _cur:
                _cur.execute(
                    "SELECT direction, COUNT(*) n FROM tracked_signals "
                    "WHERE strength IN ('Strong','Moderate') "
                    "AND signal_time >= GREATEST(DATE_SUB(NOW(), INTERVAL 21 DAY), %s) "
                    "GROUP BY direction", (_since,))
                _cnt = {r["direction"]: int(r["n"]) for r in _cur.fetchall()}
        finally:
            _conn.close()
        _up, _dn = _cnt.get("UP", 0), _cnt.get("DOWN", 0)
        _tot = _up + _dn
        if _tot >= 15:
            _share = max(_up, _dn) / _tot
            _ln = f"signal balance 21d: UP {_up} / DOWN {_dn}"
            if _share >= 0.85:
                _ln += "  << ONE-SIDED >=85% - check pred level (revalidation 2b)"
            lines.append(_ln)
        else:
            # Flooring the window created a new way to be blind: "not enough
            # samples yet" and "the decode has stopped firing again" print the
            # same line.  That is the failure mode this check exists to catch,
            # so put a clock on the silence itself — at ~15-18 Strong/month
            # plus Moderates, two weeks with almost nothing is not a quiet
            # market, it is a decode that is not reaching either tail.
            _elapsed = (datetime.now(timezone.utc)
                        - datetime.strptime(_since, "%Y-%m-%d %H:%M:%S")
                        .replace(tzinfo=timezone.utc)).days
            _ln = f"signal balance: n={_tot} in {_elapsed}d since {_since}"
            if _elapsed >= 14:
                _ln += ("  << TOO QUIET - expected ~15-18 Strong/mo; check "
                        "both tails are reachable (decode_replay.py / G5)")
            else:
                _ln += " (need 15; post-decode-fix sample accumulating)"
            lines.append(_ln)
    except Exception as e:  # noqa: BLE001
        errors.append(f"signal_balance: {e}")

    # 2e-b ── absolute-floor headroom (the RESIDUAL drift vector) ────────
    # 2026-08-14.  The live-grown buffer recentres the rolling cutoffs, so
    # level drift no longer turns the decode one-sided through THAT path.
    # But ABS_FLOOR_STRONG (±0.0008) is absolute and does NOT recentre: if
    # the model's output mean walks above ~(-floor + 1.96*std), the DOWN
    # cutoff is taken over by the floor and DOWN starves again — the same
    # symptom as 2026-08-08, through the one piece the buffer fix cannot
    # reach.  2e/§2b would catch the symptom in days-to-a-month; this line
    # watches the CAUSE directly every week, with the distance quantified.
    try:
        import numpy as _np
        import shared.db as _sdb
        _conn = _sdb.get_db_conn()
        try:
            with _conn.cursor() as _cur:
                _cur.execute(
                    "SELECT pred_return_4h p FROM indicator_history "
                    "WHERE pred_return_4h IS NOT NULL AND model_version = ("
                    "  SELECT model_version FROM indicator_history "
                    "  WHERE pred_return_4h IS NOT NULL "
                    "  ORDER BY dt DESC LIMIT 1) "
                    "ORDER BY dt DESC LIMIT 200")
                _v = _np.array([float(r["p"]) for r in _cur.fetchall()])
        finally:
            _conn.close()
        if len(_v) >= 50:
            _FLOOR = 0.0008          # ABS_FLOOR_STRONG (indicator/inference.py)
            _dn = float(_np.quantile(_v, 0.025))
            _up = float(_np.quantile(_v, 0.975))
            _std = float(_v.std()) or 1e-9
            # headroom: how many std the mean can still walk (either way)
            # before a floor takes over that side's rolling cutoff
            _room_dn = ((-_FLOOR + 1.96 * _std) - float(_v.mean())) / _std
            _room_up = (float(_v.mean()) - (_FLOOR - 1.96 * _std)) / _std
            _room = min(_room_dn, _room_up)
            _side = "DOWN" if _room_dn <= _room_up else "UP"
            _ln = (f"floor headroom: mean {_v.mean():+.6f} "
                   f"cutoffs {_dn:+.6f}/{_up:+.6f} "
                   f"room {_room:+.2f} std ({_side} side nearest)")
            if _dn > -_FLOOR or _up < _FLOOR:
                _ln += "  << FLOOR BINDING - one tail is being choked NOW"
            elif _room < 0.3:
                _ln += "  << <0.3 std to floor takeover - level walking away"
            lines.append(_ln)
        else:
            lines.append(f"floor headroom: n={len(_v)} preds (need 50)")
    except Exception as e:  # noqa: BLE001
        errors.append(f"floor_headroom: {e}")

    # 2e-d ── Strong fire-rate vs design (2026-08-18) ───────────────────
    # The 08-13→08-17 DOWN avalanche fired Strong on 21% of bars against a
    # ~5% two-tailed design: a rank-vs-self decode under a fast output-level
    # walk over-fires on the walk side while the other tail sits transiently
    # unreachable (the 200-bar window needs ~8d to roll the old level out).
    # 2e catches the direction share; this line catches the RATE — the two
    # can fail independently (a balanced 20% fire-rate is still a sick
    # decode).  Alert at >2x design.
    try:
        import shared.db as _sdb
        from indicator.model_version import DECODE_EPOCH, sample_floor
        _since = sample_floor(DECODE_EPOCH)
        _conn = _sdb.get_db_conn()
        try:
            with _conn.cursor() as _cur:
                _cur.execute(
                    "SELECT COUNT(*) n FROM indicator_history "
                    "WHERE dt >= GREATEST(DATE_SUB(NOW(), INTERVAL 21 DAY), %s)",
                    (_since,))
                _bars = int(_cur.fetchone()["n"])
                _cur.execute(
                    "SELECT COUNT(*) n FROM tracked_signals "
                    "WHERE strength='Strong' AND signal_time >= "
                    "GREATEST(DATE_SUB(NOW(), INTERVAL 21 DAY), %s)",
                    (_since,))
                _n_strong = int(_cur.fetchone()["n"])
        finally:
            _conn.close()
        if _bars >= 48:
            _rate = 100.0 * _n_strong / _bars
            _ln = f"strong fire-rate: {_n_strong}/{_bars} bars = {_rate:.0f}% (design ~5%)"
            if _rate > 10.0:
                _ln += ("  << OVER-FIRING >2x design - output level walking; "
                        "tier semantics diluted (self-limits in ~8d unless "
                        "the walk continues)")
            lines.append(_ln)
    except Exception as e:  # noqa: BLE001
        errors.append(f"fire_rate: {e}")

    # 2e-c ── crowd-strategy battery (display-only, §0.49c) ─────────────
    # 2026-08-17.  Textbook-default archetypes' trailing-30d paper P&L as
    # regime states.  Wired per the frozen registration (2 of 3 predictions
    # passed on sign + breadth: SF eats when the breakout crowd starves,
    # 7/9 coins; V7 starves when the trend crowd feasts).  DISPLAY ONLY —
    # CIs still include zero, alerting needs CI-grade evidence.
    try:
        from research.crowd_battery import paid_states as _cb_states
        from research.survival_cards import CACHE as _cb_cache, SC as _cb_sc
        _bars = _cb_sc.load_csv(str(_cb_cache / "BTCUSDT_1h.csv"))
        _st = _cb_states(_bars)
        _last = {a: ("PAID" if list(v.values())[-1] > 0 else "STARVED")
                 for a, v in _st.items() if v}
        _bo_paid = 0
        _bo_n = 0
        for _sym in ("BTC", "ETH", "SOL", "BNB", "XRP",
                     "DOGE", "ADA", "LINK", "AVAX"):
            _fp = _cb_cache / f"{_sym}USDT_1h.csv"
            if not _fp.exists():
                continue
            _s = _cb_states(_cb_sc.load_csv(str(_fp)))["breakout"]
            if _s:
                _bo_n += 1
                _bo_paid += 1 if list(_s.values())[-1] > 0 else 0
        # ADX joined 2026-08-17 (§0.49d): the crowd's own regime gauge beat
        # trend_z head-to-head on BOTH strategies (SF: CI width 0.096 vs
        # 0.219, breadth 8/9 vs 6/9, CI clear of zero -> the line's first
        # tier-2 result).  RANGING = SF tailwind / TRENDING = SF headwind.
        from research.crowd_battery2 import adx_state as _adx
        _a = _adx(_bars)
        _btc_adx = list(_a.values())[-1] if _a else "?"
        _tr_coins = 0
        _tr_n = 0
        for _sym in ("BTC", "ETH", "SOL", "BNB", "XRP",
                     "DOGE", "ADA", "LINK", "AVAX"):
            _fp = _cb_cache / f"{_sym}USDT_1h.csv"
            if not _fp.exists():
                continue
            _s = _adx(_cb_sc.load_csv(str(_fp)))
            if _s:
                _tr_n += 1
                _tr_coins += 1 if list(_s.values())[-1] == "TRENDING" else 0
        # 2026-08-17 §0.49f: PSAR took the V7 trend seat from SMA50/200 in
        # the pre-registered head-to-head (narrower CI, point no worse).
        from research.crowd_battery3 import (
            paid_states_from_pos as _ps_from, pos_psar as _psar)
        _ps = _ps_from(_bars, _psar(_bars))
        _trend_psar = ("PAID" if list(_ps.values())[-1] > 0 else "STARVED") \
            if _ps else "?"
        _ln = (
            f"crowd battery (BTC): trend(PSAR) {_trend_psar} / "
            f"mr {_last.get('mr','?')} / breakout {_last.get('breakout','?')}"
            f" | ADX {_btc_adx}, TRENDING coins {_tr_coins}/{_tr_n}"
            f" | breakout-PAID coins {_bo_paid}/{_bo_n}")
        if _tr_n and _tr_coins / _tr_n > 0.5:
            _ln += "  << ADX TRENDING majority - SF headwind (tier-2, B-P9)"
        lines.append(_ln)
    except Exception as e:  # noqa: BLE001
        errors.append(f"crowd_battery: {e}")

    # 2f ── V7 entry-execution shadow refresh (frozen 2026-08-04) ────────
    # The forward counter only accumulates when the script runs; it had no
    # scheduler until 2026-08-08 (79h stale when caught). Weekly is enough:
    # it recomputes prospectively-tagged rows from the DB each run.
    rce, oute = run(["research/v7_entry_shadow.py"], 300)
    if rce == 0:
        tail = [ln.strip() for ln in oute.strip().splitlines()[-3:] if ln.strip()]
        lines.append("entry-shadow: " + " | ".join(tail)[:300])
    else:
        errors.append(f"entry_shadow rc={rce}")

    # 3 ── depth_deltas span (subhourly revival due?) ────────────────────
    try:
        r = q1("SELECT (MAX(minute_start_ms)-MIN(minute_start_ms))/86400000.0 d "
               "FROM depth_deltas_1m")
        d = float(r["d"]) if r and r["d"] is not None else 0.0
        lines.append(f"depth_deltas span: {d:.0f}/{DEPTH_DUE_DAYS}d"
                     + ("  << subhourly revival re-run DUE"
                        if d >= DEPTH_DUE_DAYS and not st.get("depth_alerted")
                        else ""))
        if d >= DEPTH_DUE_DAYS:
            st["depth_alerted"] = True
    except Exception as e:  # noqa: BLE001
        errors.append(f"depth_span: {e}")

    # 4 ── cancel-flow pre-registered verdict (>= 2026-08-10, once) ─────
    if now.strftime("%Y-%m-%d") >= CANCEL_VERDICT_DATE and \
            not st.get("cancel_verdict_done"):
        rc1, out1 = run(["research/cancel_lead_ic.py"], 900)
        rc2, out2 = run(["research/cancel_shock_ic.py"], 900)
        tail1 = " | ".join(out1.strip().splitlines()[-3:])[:400]
        tail2 = " | ".join(out2.strip().splitlines()[-3:])[:400]
        lines.append(f"CANCEL VERDICT (pre-registered {CANCEL_VERDICT_DATE}):")
        lines.append(f"  lead_ic rc={rc1}: {tail1}")
        lines.append(f"  shock_ic rc={rc2}: {tail2}")
        if rc1 == 0 and rc2 == 0:
            st["cancel_verdict_done"] = True
        else:
            errors.append("cancel verdict scripts failed — will retry")
    elif not st.get("cancel_verdict_done"):
        lines.append(f"cancel verdict: scheduled {CANCEL_VERDICT_DATE}")

    if errors:
        lines.append("errors: " + " ; ".join(errors)[:500])
    save_state(st)

    msg = "\n".join(lines)
    log(msg.replace("\n", " || "))

    # ── delivery with retries; final failure leaves a trace ────────────
    pushed = False
    try:
        import os
        from indicator.okx.alerter import send_critical
        chat = (os.environ.get("TG_ALERT_CHAT_ID")
                or os.environ.get("TG_CRITICAL_CHAT_ID") or "")
        if chat:
            for attempt in range(1, 7):
                pushed = send_critical(chat, msg)
                if pushed:
                    break
                if attempt < 6:
                    time.sleep(60)
        else:
            log("no TG chat id in env — printed only")
    except Exception as e:  # noqa: BLE001
        log(f"tg push exception: {e}")
    if not pushed:
        log("TG PUSH FAILED after retries — operator did not receive this")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
