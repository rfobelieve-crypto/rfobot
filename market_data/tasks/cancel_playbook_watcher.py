"""Cancel-flow playbook watcher — machine-prospective event logger + TG alert.

Detects the frozen categorical playbooks from the cancel-flow research line
(depth_deltas_1m spot series + flow_bars_1m volume + orderbook mid) every
minute, logs them to cancel_playbook_events with outcome backfill, and pushes
a Telegram alert for directional playbooks (with cooldown).

This is the machine version of research/results/eyeball_log.md: definitions
are frozen under DEF_VERSION below and detection is strictly prospective
(each event row is written the minute it forms; outcomes are backfilled
later). It produces evidence about WHICH playbook deserves promotion to a
pre-registered family — it is NOT a trading signal and never touches the
executor / v7 pipeline.

FROZEN DEFINITIONS v1 (2026-07-16) — do not tune; changes require a new
DEF_VERSION and old rows are never re-labelled:
  gate            shock = tot_cancel / trailing-60m-median(tot_cancel) >= 3.0
  skew15          15m mean of (skew_raw - trailing-60m mean), skew_raw =
                  (ask_cancel - bid_cancel)/tot_cancel
  net15           same treatment of ((askC-askA)-(bidC-bidA))/tot_cancel
  vshock          volume_usd / trailing-60m-median(volume_usd)   [flow_bars]
  taker_ratio     delta_usd / volume_usd                          [flow_bars]
  ret_1m          mid(t)/mid(t-1) - 1
Playbooks (evaluated at gate minutes, first match wins):
  吸收 absorption   vshock>=3 and |taker_ratio|>=0.30 and |ret_1m|<=0.0005
                    → direction = 反轉 (sellers absorbed → UP, buyers → DOWN)
  真破 true_break   vshock>=3 and |taker_ratio|>=0.30 and |ret_1m|>0.0005
                    → direction = 順勢 sign(taker_ratio)
  真空 vacuum       vshock<3 and skew15>=+0.30 and net15>=+0.30 → UP
                    (mirror ≤ -0.30 → DOWN)
  避險 two_sided    vshock<3 and |skew15|<0.10 and |net15|<0.10 → NONE
                    (volatility expectation; logged, never alerted)
Outcomes: fwd mid return at 30/60/120m; hit_60m = sign matches direction.

Run modes:
    (wired into start_all as daemon thread)
    python -m market_data.tasks.cancel_playbook_watcher --replay 168
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.db import get_db_conn

logger = logging.getLogger(__name__)

DEF_VERSION = "v1-2026-07-16"
GATE_SHOCK = 3.0
DEEP = 0.30
FLAT = 0.10
TAKER_MIN = 0.30
RET_FLAT = 0.0005          # 5 bps: |1m ret| below = price held (absorption)
COOLDOWN_MIN = 60          # global: at most one alert per hour
HORIZONS_MIN = (30, 60, 120)
ALERTABLE = {"absorption", "true_break", "vacuum"}   # two_sided logs only
# ALERT POLICY (UX layer, deliberately separate from the frozen logging
# definitions above — tightening this never touches the logged dataset):
# only push events big enough to be worth a phone buzz.
ALERT_MIN_VSHOCK = 20.0    # replay-sim 2026-07-16: ≈4.4 alerts/day
ALERT_MIN_NET = 0.30
ZH = {"absorption": "吸收", "true_break": "真破", "vacuum": "真空",
      "two_sided": "雙側避險", "gate_only": "純爆量"}


# ── schema ───────────────────────────────────────────────────────────────────

def ensure_schema() -> None:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS cancel_playbook_events (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                def_version VARCHAR(20) NOT NULL,
                minute_start_ms BIGINT NOT NULL,
                playbook VARCHAR(20) NOT NULL,
                direction VARCHAR(8) NOT NULL,
                px DOUBLE, shock DOUBLE, skew15 DOUBLE, net15 DOUBLE,
                vshock DOUBLE, taker_ratio DOUBLE, ret_1m DOUBLE,
                alerted TINYINT NOT NULL DEFAULT 0,
                fwd_ret_30m DOUBLE NULL,
                fwd_ret_60m DOUBLE NULL,
                fwd_ret_120m DOUBLE NULL,
                hit_60m TINYINT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_evt (def_version, minute_start_ms, playbook),
                INDEX idx_min (minute_start_ms)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
        conn.commit()
    finally:
        conn.close()


# ── data + features (pure) ───────────────────────────────────────────────────

def _q(conn, sql: str, params=None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params or None)
        return pd.DataFrame(cur.fetchall() or [])


def load_frame(lookback_min: int) -> pd.DataFrame:
    """Minute-indexed frame joining spot depth deltas, flow bars and mid."""
    t0 = int(time.time() * 1000) - lookback_min * 60_000
    conn = get_db_conn()
    try:
        dd = _q(conn, "SELECT minute_start_ms ms, bid_add_qty ba, "
                      "bid_cancel_qty bc, ask_add_qty aa, ask_cancel_qty ac "
                      "FROM depth_deltas_1m WHERE canonical_symbol='BTC-USD' "
                      "AND exchange='binance' AND minute_start_ms >= %s "
                      "ORDER BY minute_start_ms", (t0,))
        fb = _q(conn, "SELECT window_start ms, volume_usd vol, delta_usd dlt "
                      "FROM flow_bars_1m WHERE canonical_symbol='BTC-USD' "
                      "AND exchange_scope='all' AND window_start >= %s "
                      "ORDER BY window_start", (t0,))
        ob = _q(conn, "SELECT ts_ms, mid_price mid FROM orderbook_snapshots_1m "
                      "WHERE canonical_symbol='BTC-USD' AND ts_ms >= %s "
                      "ORDER BY ts_ms", (t0,))
    finally:
        conn.close()
    if dd.empty:
        return pd.DataFrame()
    for f in (dd, fb, ob):
        for c in f.columns:
            f[c] = pd.to_numeric(f[c])
    dd["m"] = dd["ms"] // 60_000
    df = dd.groupby("m").last()[["ba", "bc", "aa", "ac"]]
    if not fb.empty:
        fb["m"] = fb["ms"] // 60_000
        df = df.join(fb.groupby("m").last()[["vol", "dlt"]], how="left")
    else:
        df["vol"] = np.nan
        df["dlt"] = np.nan
    if not ob.empty:
        ob["m"] = ob["ts_ms"] // 60_000
        df = df.join(ob.groupby("m")["mid"].last().rename("mid"), how="left")
    else:
        df["mid"] = np.nan
    df["mid"] = df["mid"].ffill()
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Frozen v1 features, trailing-only (no look-ahead)."""
    out = df.copy()
    tot = out["bc"] + out["ac"]
    base = tot.rolling(60, min_periods=30).median()
    out["shock"] = tot / base.replace(0, np.nan)

    skew_raw = (out["ac"] - out["bc"]) / tot.replace(0, np.nan)
    skew_adj = skew_raw - skew_raw.rolling(60, min_periods=30).mean()
    out["skew15"] = skew_adj.rolling(15, min_periods=5).mean()

    net_raw = ((out["ac"] - out["aa"]) - (out["bc"] - out["ba"])) \
        / tot.replace(0, np.nan)
    net_adj = net_raw - net_raw.rolling(60, min_periods=30).mean()
    out["net15"] = net_adj.rolling(15, min_periods=5).mean()

    vbase = out["vol"].rolling(60, min_periods=30).median()
    out["vshock"] = out["vol"] / vbase.replace(0, np.nan)
    out["taker_ratio"] = out["dlt"] / out["vol"].replace(0, np.nan)
    out["ret_1m"] = out["mid"] / out["mid"].shift(1) - 1
    return out


def classify_minute(r: pd.Series) -> tuple[str, str] | None:
    """(playbook, direction) for one feature row, or None. First match wins."""
    if not np.isfinite(r["shock"]) or r["shock"] < GATE_SHOCK:
        return None
    vs, tr, ret = r["vshock"], r["taker_ratio"], r["ret_1m"]
    sk, nt = r["skew15"], r["net15"]
    if np.isfinite(vs) and vs >= 3.0 and np.isfinite(tr) and abs(tr) >= TAKER_MIN \
            and np.isfinite(ret):
        if abs(ret) <= RET_FLAT:
            return "absorption", ("UP" if tr < 0 else "DOWN")
        return "true_break", ("UP" if tr > 0 else "DOWN")
    if np.isfinite(sk) and np.isfinite(nt) and (not np.isfinite(vs) or vs < 3.0):
        if sk >= DEEP and nt >= DEEP:
            return "vacuum", "UP"
        if sk <= -DEEP and nt <= -DEEP:
            return "vacuum", "DOWN"
        if abs(sk) < FLAT and abs(nt) < FLAT:
            return "two_sided", "NONE"
    # gate fired but no playbook matched — log anyway (additive 2026-07-16,
    # same session as v1): gives the unconditional base-rate denominator for
    # later stats. Never alerted; playbook labels above are unchanged.
    return "gate_only", "NONE"


def scan(df_feat: pd.DataFrame, minutes: list[int]) -> list[dict]:
    events = []
    for m in minutes:
        if m not in df_feat.index:
            continue
        r = df_feat.loc[m]
        hit = classify_minute(r)
        if hit is None:
            continue
        playbook, direction = hit
        events.append({
            "minute_start_ms": int(m) * 60_000,
            "playbook": playbook, "direction": direction,
            "px": None if pd.isna(r["mid"]) else float(r["mid"]),
            "shock": float(r["shock"]),
            "skew15": None if pd.isna(r["skew15"]) else float(r["skew15"]),
            "net15": None if pd.isna(r["net15"]) else float(r["net15"]),
            "vshock": None if pd.isna(r["vshock"]) else float(r["vshock"]),
            "taker_ratio": None if pd.isna(r["taker_ratio"]) else float(r["taker_ratio"]),
            "ret_1m": None if pd.isna(r["ret_1m"]) else float(r["ret_1m"]),
        })
    return events


# ── persistence + alerts ─────────────────────────────────────────────────────

def insert_events(events: list[dict]) -> list[dict]:
    """INSERT IGNORE; returns the subset that was newly inserted."""
    if not events:
        return []
    fresh = []
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            for e in events:
                cur.execute(
                    "INSERT IGNORE INTO cancel_playbook_events "
                    "(def_version, minute_start_ms, playbook, direction, px, "
                    " shock, skew15, net15, vshock, taker_ratio, ret_1m) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (DEF_VERSION, e["minute_start_ms"], e["playbook"],
                     e["direction"], e["px"], e["shock"], e["skew15"],
                     e["net15"], e["vshock"], e["taker_ratio"], e["ret_1m"]))
                if cur.rowcount > 0:
                    fresh.append(e)
        conn.commit()
    finally:
        conn.close()
    return fresh


def _tg_creds() -> tuple[str, str]:
    def val(*keys):
        for k in keys:
            v = os.environ.get(k, "").strip()
            if v:
                return v
        envf = Path(__file__).resolve().parent.parent.parent / ".env"
        if envf.exists():
            for line in envf.read_text(encoding="utf-8",
                                       errors="ignore").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, _, v = line.partition("=")
                    if k.strip() in keys and v.strip():
                        return v.strip().strip('"').strip("'")
        return ""
    return (val("TELEGRAM_BOT_TOKEN"),
            val("TG_CRITICAL_CHAT_ID", "TELEGRAM_CHAT_ID"))


def _alert_worthy(e: dict) -> bool:
    """Push only 'big' events: huge volume or a real net-withdrawal move."""
    vs = e.get("vshock")
    nt = e.get("net15")
    return ((vs is not None and vs >= ALERT_MIN_VSHOCK)
            or (nt is not None and abs(nt) >= ALERT_MIN_NET))


def _cooldown_ok(conn, minute_ms: int) -> bool:
    """Global cooldown — at most one alert per COOLDOWN_MIN, any playbook."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) n FROM cancel_playbook_events "
            "WHERE def_version=%s AND alerted=1 AND minute_start_ms > %s",
            (DEF_VERSION, minute_ms - COOLDOWN_MIN * 60_000))
        return int(cur.fetchone()["n"]) == 0


def alert_events(fresh: list[dict]) -> None:
    """Plain-text TG alerts (no parse_mode — cf. mistake.md 2026-06-19)."""
    token, chat = _tg_creds()
    if not token or not chat:
        if fresh:
            logger.warning("playbook alert skipped: TG creds missing")
        return
    chart = os.environ.get("CANCEL_FLOW_CHART_URL", "").strip()
    conn = get_db_conn()
    try:
        for e in fresh:
            if e["playbook"] not in ALERTABLE or not _alert_worthy(e):
                continue
            if not _cooldown_ok(conn, e["minute_start_ms"]):
                continue
            t = (pd.Timestamp(e["minute_start_ms"], unit="ms")
                 + pd.Timedelta(hours=8)).strftime("%m-%d %H:%M")
            px = f"{e['px']:,.0f}" if e["px"] else "?"
            def fmt(v, p="+.2f"):
                return format(v, p) if v is not None else "?"
            lines = [
                "🧲 撤單劇本偵測（研究·非信號）",
                f"劇本: {ZH[e['playbook']]} → 預期 {e['direction']} (2h 內)",
                f"時間: {t} TPE  價格: {px}",
                f"shock {e['shock']:.1f}x | 毛 {fmt(e['skew15'])} | "
                f"淨 {fmt(e['net15'])} | 量 {fmt(e['vshock'], '.1f')}x | "
                f"taker {fmt(e['taker_ratio'], '+.0%')}",
            ]
            if chart:
                lines.append(chart)
            lines.append("深入: /cancelanalyze 90 (五步摘要) 或問 agent")
            lines.append(f"def {DEF_VERSION} · edge 未驗證 · 勿作交易依據")
            try:
                resp = requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    data={"chat_id": chat, "text": "\n".join(lines)},
                    timeout=15)
                ok = resp.status_code == 200
            except Exception:
                logger.exception("playbook alert send failed")
                ok = False
            if ok:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE cancel_playbook_events SET alerted=1 "
                        "WHERE def_version=%s AND minute_start_ms=%s "
                        "AND playbook=%s",
                        (DEF_VERSION, e["minute_start_ms"], e["playbook"]))
                conn.commit()
    finally:
        conn.close()


def backfill_outcomes() -> None:
    """Fill fwd returns for events whose horizons have elapsed."""
    now_ms = int(time.time() * 1000)
    conn = get_db_conn()
    try:
        due = _q(conn, "SELECT id, minute_start_ms ms, px, direction "
                       "FROM cancel_playbook_events "
                       "WHERE fwd_ret_120m IS NULL AND px IS NOT NULL "
                       "AND minute_start_ms <= %s",
                 (now_ms - 31 * 60_000,))
        if due.empty:
            return
        due["ms"] = pd.to_numeric(due["ms"])
        due["px"] = pd.to_numeric(due["px"])
        lo = int(due["ms"].min())
        hi = int(due["ms"].max()) + (HORIZONS_MIN[-1] + 2) * 60_000
        ob = _q(conn, "SELECT ts_ms, mid_price mid FROM orderbook_snapshots_1m "
                      "WHERE canonical_symbol='BTC-USD' "
                      "AND ts_ms BETWEEN %s AND %s ORDER BY ts_ms", (lo, hi))
        if ob.empty:
            return
        ob["ts_ms"] = pd.to_numeric(ob["ts_ms"])
        ob["mid"] = pd.to_numeric(ob["mid"])
        mids = ob.groupby(ob["ts_ms"] // 60_000)["mid"].last()
        with conn.cursor() as cur:
            for _, r in due.iterrows():
                m0 = int(r["ms"]) // 60_000
                sets, vals = [], []
                fwd60 = None
                for h in HORIZONS_MIN:
                    if now_ms < int(r["ms"]) + (h + 1) * 60_000:
                        continue
                    if m0 + h in mids.index:
                        fwd = float(mids.loc[m0 + h] / r["px"] - 1)
                        sets.append(f"fwd_ret_{h}m=%s")
                        vals.append(fwd)
                        if h == 60:
                            fwd60 = fwd
                if fwd60 is not None and r["direction"] in ("UP", "DOWN"):
                    sets.append("hit_60m=%s")
                    vals.append(1 if (fwd60 > 0) == (r["direction"] == "UP")
                                else 0)
                if sets:
                    vals.append(int(r["id"]))
                    cur.execute("UPDATE cancel_playbook_events SET "
                                + ", ".join(sets) + " WHERE id=%s", vals)
        conn.commit()
    except Exception:
        logger.exception("playbook outcome backfill failed")
    finally:
        conn.close()


# ── daemon loop ──────────────────────────────────────────────────────────────

def watch_loop() -> None:
    ensure_schema()
    logger.info("cancel-playbook watcher started (def=%s)", DEF_VERSION)
    last_seen = 0
    while True:
        try:
            df = load_frame(lookback_min=100)
            if not df.empty:
                feat = compute_features(df)
                closed = int(time.time() // 60) - 1   # last fully closed minute
                minutes = [m for m in feat.index
                           if last_seen < m <= closed]
                minutes = minutes[-5:]                # catch-up cap
                fresh = insert_events(scan(feat, minutes))
                alert_events(fresh)
                if minutes:
                    last_seen = max(minutes)
            backfill_outcomes()
        except Exception:
            logger.exception("playbook watcher cycle failed")
        time.sleep(60)


# ── replay (smoke test / retro scan; no DB writes, no alerts) ────────────────

def replay(hours: int) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    df = load_frame(lookback_min=hours * 60 + 120)
    if df.empty:
        print("no data")
        return 1
    feat = compute_features(df)
    events = scan(feat, list(feat.index))
    mids = feat["mid"]
    print(f"replay {hours}h  n_min={len(feat)}  events={len(events)}  "
          f"(def {DEF_VERSION})")
    stats: dict[tuple, list] = {}
    for e in events:
        m0 = e["minute_start_ms"] // 60_000
        fwd = (float(mids.loc[m0 + 60] / e["px"] - 1)
               if (m0 + 60) in mids.index and e["px"] else None)
        t = (pd.Timestamp(e["minute_start_ms"], unit="ms")
             + pd.Timedelta(hours=8)).strftime("%m-%d %H:%M")
        fs = f"{fwd:+.3%}" if fwd is not None else "  n/a "
        star = " ★" if _alert_worthy(e) and e["playbook"] in ALERTABLE else ""
        print(f"  {t}  {ZH[e['playbook']]:<5} {e['direction']:<4}{star} "
              f"px={e['px'] or 0:,.0f} shock={e['shock']:.1f}x "
              f"毛={e['skew15'] if e['skew15'] is not None else float('nan'):+.2f} "
              f"淨={e['net15'] if e['net15'] is not None else float('nan'):+.2f} "
              f"量={e['vshock'] if e['vshock'] is not None else float('nan'):.1f}x "
              f"fwd60m={fs}")
        if fwd is not None and e["direction"] in ("UP", "DOWN"):
            k = (e["playbook"], e["direction"])
            stats.setdefault(k, []).append(
                1 if (fwd > 0) == (e["direction"] == "UP") else 0)
    print("\n煙測統計（小樣本，非證據）:")
    for (pb, d), hits in sorted(stats.items()):
        print(f"  {ZH[pb]} {d}: n={len(hits)}  hit={np.mean(hits):.0%}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=int, metavar="HOURS",
                    help="retro scan, print only (no DB writes, no alerts)")
    args = ap.parse_args()
    if args.replay:
        raise SystemExit(replay(args.replay))
    watch_loop()
