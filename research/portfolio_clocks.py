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
