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
import json
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

# ── instruments (2026-07-28) ────────────────────────────────────────────────
# A coin is scannable only if it has ALL THREE source tables: depth_deltas_1m
# (skew/net), flow_bars_1m (vshock, taker_ratio) and orderbook_snapshots_1m
# (mid -> ret_1m, px). Every frozen playbook gates on vshock, so depth alone
# is not enough. Measured 2026-07-28: BTC-USD and ETH-USD have all three;
# the 8 alts already streaming into depth_deltas (ADA/AAVE/BNB/DOGE/LINK/
# SUI/UNI/XRP) have depth ONLY — they need symbol_mapper.SYMBOL_MAP and
# orderbook_l20_collector.TRACKED extended, then time to accumulate, before
# they can be added here.
DEFAULT_SYMBOL = "BTC-USD"
# Extra instruments are opt-in via env so a bare redeploy cannot silently
# double the Telegram alert volume (same gate style as _lead_enabled).
EXTRA_SYMBOLS = ("ETH-USD",)


def watch_symbols() -> tuple[str, ...]:
    """BTC always; extras only when CANCEL_WATCH_SYMBOLS names them."""
    raw = os.environ.get("CANCEL_WATCH_SYMBOLS", "").strip()
    if not raw:
        return (DEFAULT_SYMBOL,)
    want = {s.strip().upper() for s in raw.split(",") if s.strip()}
    return (DEFAULT_SYMBOL,) + tuple(s for s in EXTRA_SYMBOLS if s in want)

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
      "two_sided": "雙側避險", "gate_only": "純爆量",
      "vacuum_lead": "撤單先行"}
# 2026-07-20: 每個劇本的判讀類型（反轉 vs 順勢延續），供對答案 reply 引用
# 「這張卡當初判的是什麼」。定義沿用 classify_minute 頂部的凍結註解——
# 吸收=反轉（賣壓被接走預期彈UP/買壓被接走預期彈DOWN）、真破=順勢延續、
# 真空=阻力真空順勢延續，非新定義。
PLAYBOOK_CALL_ZH = {"absorption": "反轉", "true_break": "順勢延續",
                    "vacuum": "順勢延續"}

# ── first-hit-wins 平行診斷（FROZEN v1，2026-07-21，凍結於任何結果被看到
# 之前）────────────────────────────────────────────────────────────────────
# 動機：固定時間點(60m/120m)的正負號判斷，會把「方向對但慢」跟「方向真的
# 錯」混成同一種「錯」——今晚實測過(21:47-21:53 那串真破 60m 全錯、120m
# 全部翻正 +1.0%~+1.6%)。first-hit-wins 改成「逐分鐘看哪個價位先被碰到」，
# 不受任意時間切片影響。門檻沿用 outcome_tracker.py 既有的 ±0.5%（非新造
# 數字），觀察窗沿用既有 120 分鐘（不新開時間跨度）。這是跟固定時間點指標
# 並行的新診斷，不取代它——F1/F1b/hit_60m 的判決仍然用固定時間點的尺，
# 因為那才是跟 walk-forward IC 方法論一致、防止時間窗口 p-hacking 的量法。
FIRST_HIT_THRESHOLD = 0.005    # ±0.5%，沿用 outcome_tracker.py 既有門檻


def first_hit_verdict(entry_px: float, direction: str,
                      fwd_mids: list[float]) -> str | None:
    """逐分鐘掃 fwd_mids（觸發後、依時間排序），先碰目標價=hit，先碰反向價
    =miss，都沒碰到=unresolved。direction 不是 UP/DOWN 或無資料 → None
    （不適用，不硬判）。"""
    if direction not in ("UP", "DOWN") or not fwd_mids:
        return None
    if direction == "UP":
        target = entry_px * (1 + FIRST_HIT_THRESHOLD)
        opposite = entry_px * (1 - FIRST_HIT_THRESHOLD)
    else:
        target = entry_px * (1 - FIRST_HIT_THRESHOLD)
        opposite = entry_px * (1 + FIRST_HIT_THRESHOLD)
    for mid in fwd_mids:
        reached_target = (mid >= target) if direction == "UP" else (mid <= target)
        reached_opposite = (mid <= opposite) if direction == "UP" else (mid >= opposite)
        if reached_target:
            return "hit"
        if reached_opposite:
            return "miss"
    return "unresolved"


FIRST_HIT_ZH = {"hit": "✅ 對了", "miss": "❌ 錯了", "unresolved": "⏳ 未定"}


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
            # A 對答案 reply (2026-07-20) — additive columns
            for ddl in (
                "ALTER TABLE cancel_playbook_events "
                "ADD COLUMN tg_message_id BIGINT NULL",
                "ALTER TABLE cancel_playbook_events "
                "ADD COLUMN outcome_replied TINYINT NOT NULL DEFAULT 0",
                # 2026-07-21: first-hit-wins 平行診斷（見 first_hit_verdict
                # 凍結定義）— 不取代 fwd_ret_Xm/hit_60m，並行顯示用
                "ALTER TABLE cancel_playbook_events "
                "ADD COLUMN first_hit_result VARCHAR(12) NULL",
            ):
                try:
                    cur.execute(ddl)
                except Exception:
                    pass      # 1060 duplicate column = already there
        conn.commit()
    finally:
        conn.close()


# ── data + features (pure) ───────────────────────────────────────────────────

def _q(conn, sql: str, params=None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params or None)
        return pd.DataFrame(cur.fetchall() or [])


def load_frame(lookback_min: int, symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    """Minute-indexed frame joining spot depth deltas, flow bars and mid.

    `symbol` defaults to BTC-USD so every existing call site keeps its exact
    behaviour. All three tables must carry the symbol — every frozen playbook
    gates on vshock (flow_bars) and most need ret_1m (orderbook mid), so a
    coin with depth_deltas alone cannot be scanned. See WATCH_SYMBOLS.
    """
    t0 = int(time.time() * 1000) - lookback_min * 60_000
    conn = get_db_conn()
    try:
        dd = _q(conn, "SELECT minute_start_ms ms, bid_add_qty ba, "
                      "bid_cancel_qty bc, ask_add_qty aa, ask_cancel_qty ac "
                      "FROM depth_deltas_1m WHERE canonical_symbol=%s "
                      "AND exchange='binance' AND minute_start_ms >= %s "
                      "ORDER BY minute_start_ms", (symbol, t0))
        fb = _q(conn, "SELECT window_start ms, volume_usd vol, delta_usd dlt "
                      "FROM flow_bars_1m WHERE canonical_symbol=%s "
                      "AND exchange_scope='all' AND window_start >= %s "
                      "ORDER BY window_start", (symbol, t0))
        ob = _q(conn, "SELECT ts_ms, mid_price mid FROM orderbook_snapshots_1m "
                      "WHERE canonical_symbol=%s AND ts_ms >= %s "
                      "ORDER BY ts_ms", (symbol, t0))
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

    # 2026-07-21: 撤單/加單機制拆解（display-only，additive，不影響上面
    # 任何凍結欄位）。net15 只給淨值方向，看不出是「這側在撤」還是「對側
    # 在加」造成的同一個淨值——對 ba/bc/aa/ac 四條原始序列各自算 trailing
    # 60m shock（沿用既有 tot shock 手法），供 dominant_flow_action() 指出
    # 這分鐘最反常的動作具體是誰做的。
    for col in ("ba", "bc", "aa", "ac"):
        cbase = out[col].rolling(60, min_periods=30).median()
        out[f"{col}_shock"] = out[col] / cbase.replace(0, np.nan)
    return out


FLOW_ACTION_ZH = {"ac": "賣方撤單", "bc": "買方撤單",
                  "aa": "賣方加單", "ba": "買方加單"}


def dominant_flow_action(r: pd.Series, min_shock: float = 2.0) -> str | None:
    """FROZEN v1（2026-07-21）：ba/bc/aa/ac 四個 shock 裡最大的那個，就是
    這分鐘最反常的具體動作（UX 層判讀輔助，門檻可調，不是分類定義——比照
    ALERT_MIN_VSHOCK 的先例）。min_shock=2.0 比合計 shock 的 3.0 門檻寬鬆，
    因為單一原始序列本來就比合計序列雜訊大。四個都不到門檻 → None
    （沒有單一動作明顯主導，可能是雙邊同時在動或本來就平靜）。"""
    vals = {c: r.get(f"{c}_shock") for c in ("ba", "bc", "aa", "ac")}
    vals = {c: v for c, v in vals.items() if v is not None and np.isfinite(v)}
    if not vals:
        return None
    top_c, top_v = max(vals.items(), key=lambda kv: kv[1])
    if top_v < min_shock:
        return None
    return FLOW_ACTION_ZH[top_c]


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


# ── 'vacuum_lead' v3 — PROVISIONAL, NOT frozen (2026-07-27) ───────────────────
# Isolates the pure "one-sided withdrawal LEADS price" thesis that v1's vacuum
# branch never captured (0 events).
#
# v2 (2026-07-25) was BROKEN BY CONSTRUCTION and logged 0 rows in ~36h live:
# it ANDed net15 (a PROPORTION — that side's share of total cancels) with
# ac_shock (an ABSOLUTE multiple vs that side's own 60m median). Those two
# pull against each other — a high one-sided *share* usually means the OTHER
# side cancelled little (small denominator), not that this side spiked.
# Measured over 72h: when net15>=0.15, ac_shock's median was 0.89 and it
# NEVER reached 2.5, so the AND could not fire. (v2 logged nothing, so no
# rows are being re-labelled by this replacement.)
#
# v3 keeps both conditions in the SAME unit (shock multiples):
#   A) the withdrawing side genuinely spiked : side_shock >= LEAD_SIDE_SHOCK
#   B) it clearly dominates the other side   : side_shock >= LEAD_SIDE_RATIO
#                                              * other_shock
# plus the unchanged "lead" requirements: volume still quiet (vshock < 3) and
# price not yet moved (|ret_1m| <= 5bps), so the forward return genuinely
# tests whether the withdrawal LED the move rather than trailed it.
#
# Calibrated on FIRING RATE ONLY, never on outcomes. The v2 post-mortem also
# fixed the calibration process itself: a config is now additionally required
# to fire on >= half the days and to not depend on one freak day (v2 was
# signed off on a 30-day TOTAL without checking WHEN). Chosen 3.0/2.0 →
# 1.91/day, active 10/18 days, biggest single day 21% of all fires, and still
# firing in the most recent days. PROVISIONAL: any threshold change requires
# bumping DEF_VERSION_LEAD; prior rows are never re-labelled. Log-only, never
# alerts, never touches the executor. Env-gated for clean on/off.
DEF_VERSION_LEAD = "v3-2026-07-27"
LEAD_SIDE_SHOCK = 3.0      # withdrawing side vs its OWN trailing-60m median
LEAD_SIDE_RATIO = 2.0      # ... and vs the other side's shock
LEAD_VOL_QUIET = 3.0       # vshock < this = volume hasn't arrived (the "lead")


def classify_minute_lead(r: pd.Series) -> tuple[str, str] | None:
    """(playbook, direction) for the provisional vacuum_lead, or None."""
    vs, ret = r["vshock"], r["ret_1m"]
    acs, bcs = r.get("ac_shock"), r.get("bc_shock")
    if acs is None or bcs is None or not (np.isfinite(acs) and np.isfinite(bcs)):
        return None
    if not np.isfinite(ret) or abs(ret) > RET_FLAT:   # price already moved (or unknown)
        return None
    if np.isfinite(vs) and vs >= LEAD_VOL_QUIET:      # volume already here → not a lead
        return None
    if acs >= LEAD_SIDE_SHOCK and acs >= LEAD_SIDE_RATIO * bcs:
        return "vacuum_lead", "UP"                     # ask withdrawn → resistance gone
    if bcs >= LEAD_SIDE_SHOCK and bcs >= LEAD_SIDE_RATIO * acs:
        return "vacuum_lead", "DOWN"                   # bid withdrawn → support gone
    return None


def _lead_enabled() -> bool:
    """Env-gated (default OFF) so it's trivially reversible and can't start
    the prospective clock on a bare redeploy. CANCEL_V2_ENABLED is honoured
    as a legacy alias — it is already set in the market_data service, and
    the switch means the same thing ("the experimental lead family is on")
    even though the definition behind it is now v3."""
    for key in ("CANCEL_LEAD_ENABLED", "CANCEL_V2_ENABLED"):
        if os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on"):
            return True
    return False


# ── Display state machine (2026-07-17, additive) ─────────────────────────────
# Single state function shared by every display outlet (TG /cancelstate, the
# review-chart ribbon, event cards). It is a 1:1 relabelling of the frozen v1
# classifier above — NO new thresholds, NO tunable parameters (CDP tombstone:
# a weighted composite score dilutes signal and its weights invite overfit).
# classify_minute stays the single source of truth; this never writes events
# and never alerts.
#
# Colour language (system-wide, one set): direction UP → 🟢, DOWN → 🔴,
# no-direction states (rotation / cascade / surge) → ⚪, calm → ⚫.
# "gate_only" is the frozen classifier's own residual label (surge with no
# playbook match) — surfaced honestly as 爆量未定 instead of being forced
# into a named state.

STATE_META = {
    "calm":       ("平靜", "⚫"),
    "absorption": ("吸收", None),      # direction-coloured (量大守住 → 反轉義)
    "cascade":    ("瀑布中", "⚪"),     # true_break minute — 資訊權在量價
    "vacuum_up":  ("向上真空", "🟢"),
    "vacuum_down": ("向下真空", "🔴"),
    "rotation":   ("換防警戒", "⚪"),   # two_sided — 毛高淨零
    "surge":      ("爆量未定", "⚪"),   # gate_only residual
}

_PLAYBOOK_TO_STATE = {"absorption": "absorption", "true_break": "cascade",
                      "two_sided": "rotation", "gate_only": "surge"}

# Display hex per state — single source for every chart surface (interactive
# review ribbon, sweep zoom card). Same colour language as the emoji set:
# green=UP vacuum, red=DOWN vacuum, gray=no-direction, None=don't draw (calm)
# or direction-coloured (absorption).
STATE_COLOR = {
    "calm": None,
    "rotation": "#8a919c",
    "surge": "#8a919c",
    "cascade": "#c0c6cf",
    "vacuum_up": "#26a269",
    "vacuum_down": "#e01b24",
    "absorption": None,          # direction-coloured below
}


def state_color(s: dict) -> str | None:
    """Ribbon colour for a classify_state dict; None = draw nothing (calm)."""
    c = STATE_COLOR.get(s["state"])
    if c is not None or s["state"] == "calm":
        return c
    return "#26a269" if s["direction"] == "UP" else (
        "#e01b24" if s["direction"] == "DOWN" else "#8a919c")


def action_keyboard(source: str, source_id: int) -> dict:
    """行動鍵 (2026-07-20, 取代四鍵判讀)。callback 格式 cfa|{src}|{id}|{action}
    (≤64 bytes) — Service 1 cancel-bot webhook 解析。判讀鍵退役原因：
    卡片標題先給機器結論=錨定污染，人的判讀量到的是服從率非 alpha；
    樣本速度+選擇偏誤也撐不起統計。改成「給工具」：
      zoom=事件窗覆盤圖連結 deep=90m 五步摘要
      star=收藏(cancel_eyeball_log verdict='star') dismiss=收鍵盤。
    舊 ceb| 判讀 callback 的 Service 1 處理器保留（歷史卡片仍可按）。"""
    def mk(text, a):
        return {"text": text,
                "callback_data": f"cfa|{source}|{source_id}|{a}"}
    return {"inline_keyboard": [
        [mk("📊 特寫圖", "zoom"), mk("🔍 90m 深入", "deep")],
        [mk("⭐ 收藏", "star"), mk("✗ 忽略", "dismiss")],
    ]}


def classify_state(r: pd.Series) -> dict:
    """Display state for one feature row (from compute_features).

    Returns {state, zh, emoji, direction} — a pure relabelling of
    classify_minute; falls back to 平靜 when the shock gate is closed.
    """
    res = classify_minute(r)
    if res is None:
        zh, emoji = STATE_META["calm"]
        return {"state": "calm", "zh": zh, "emoji": emoji, "direction": "NONE"}
    playbook, direction = res
    if playbook == "vacuum":
        state = "vacuum_up" if direction == "UP" else "vacuum_down"
    else:
        state = _PLAYBOOK_TO_STATE[playbook]
    zh, emoji = STATE_META[state]
    if emoji is None:   # direction-coloured state (absorption)
        emoji = "🟢" if direction == "UP" else ("🔴" if direction == "DOWN" else "⚪")
    return {"state": state, "zh": zh, "emoji": emoji, "direction": direction}


# ── 白話翻譯層 (2026-07-20, display-only) ────────────────────────────────────
# 把凍結特徵翻成人話。只改「顯示」，不碰任何偵測/記錄定義；所有事件卡共用
# （watcher 告警 + tv_alert_poller 三種卡），改文案只動這裡。

DIR_ZH = {"UP": "偏漲 🟢", "DOWN": "偏跌 🔴", "NONE": "方向未定 ⚪"}
# 2026-07-29: forward-direction claim falsified — label the observed side of
# the book instead of forecasting. DIR_ZH kept for legacy call sites.
SIDE_ZH = {"UP": "賣側收縮 🟢", "DOWN": "買側收縮 🔴", "NONE": "雙側 ⚪"}


def humanize_story(playbook: str, direction: str,
                   vshock: float | None = None,
                   taker_ratio: float | None = None) -> str:
    """一句話講「發生了什麼」。接受劇本名或顯示狀態名（cascade/rotation/...）。"""
    v = (f"{vshock:.0f} 倍" if vshock is not None and np.isfinite(vshock)
         else "數倍")
    tr = (f"佔成交 {abs(taker_ratio):.0%}"
          if taker_ratio is not None and np.isfinite(taker_ratio) else "占比不明")
    if playbook == "absorption":
        if direction == "UP":
            return (f"成交量爆到平時的 {v}、賣方主動砸盤（{tr}），但價格守住沒跌"
                    "——賣壓被整批接走（吸收）")
        return (f"成交量爆到平時的 {v}、買方主動追價（{tr}），但價格推不上去"
                "——買盤被倒貨吸收")
    if playbook in ("true_break", "cascade"):
        if direction == "UP":
            return (f"成交量爆到平時的 {v}、買方主動追價（{tr}）且價格同步上移"
                    "——攻擊性突破（真破）")
        if direction == "DOWN":
            return (f"成交量爆到平時的 {v}、賣方主動砸盤（{tr}）且價格同步下移"
                    "——攻擊性跌破（真破）")
        return f"成交量爆到平時的 {v}、價量激戰中——瀑布瞬間先不判方向"
    if playbook in ("vacuum", "vacuum_up", "vacuum_down"):
        d = direction if direction in ("UP", "DOWN") else (
            "UP" if playbook == "vacuum_up" else "DOWN")
        if d == "UP":
            return ("成交清淡，但上方賣單正在大量淨撤離——阻力自己讓開"
                    "（向上真空）")
        return ("成交清淡，但下方買單正在大量淨撤離——支撐自己抽走"
                "（向下真空）")
    if playbook in ("two_sided", "rotation"):
        return "兩側掛單同時大量換防、淨方向不明——常見於波動放大之前"
    if playbook in ("gate_only", "surge"):
        return "撤單爆量但湊不齊任何已知劇本——先觀察，不判方向"
    return "掛單面平靜，無異常"


def humanize_book(shock: float | None = None, skew15: float | None = None,
                  net15: float | None = None,
                  dominant: str | None = None) -> str:
    """掛單面一句話：撤單鬧鐘倍數 + 抽單偏向 + 淨撤離方向（門檻沿用凍結
    FLAT）+ 主導動作（2026-07-21 加，見 dominant_flow_action）。net15/
    skew15 只講「淨值往哪邊」，講不出這是哪一方造成的——同一個淨值可能是
    「這側在撤」也可能是「對側在加」，兩種機制被含糊成同一句「撤離」，是
    使用者 2026-07-21 指出的不準確處。dominant 為 None 時不強加結論。"""
    parts = []
    if shock is not None and np.isfinite(shock):
        parts.append(f"撤單量為平時的 {shock:.1f} 倍")
    if skew15 is not None and np.isfinite(skew15):
        if skew15 >= FLAT:
            parts.append("抽單偏上方賣側（阻力在變薄）")
        elif skew15 <= -FLAT:
            parts.append("抽單偏下方買側（支撐在變薄）")
        else:
            parts.append("兩側抽單均衡")
    if net15 is not None and np.isfinite(net15):
        if net15 >= FLAT:
            parts.append("上方掛單淨撤離明顯")
        elif net15 <= -FLAT:
            parts.append("下方掛單淨撤離明顯")
        else:
            parts.append("無明顯淨撤離")
    if dominant:
        parts.append(f"具體動作: {dominant}佔主導")
    return "；".join(parts) if parts else "掛單資料不足"


def scan(df_feat: pd.DataFrame, minutes: list[int],
         classify_fn=classify_minute) -> list[dict]:
    events = []
    for m in minutes:
        if m not in df_feat.index:
            continue
        r = df_feat.loc[m]
        hit = classify_fn(r)
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
            # 2026-07-21: 不進 DB（insert_events 只認固定欄位），純供
            # alert_events() 組文字用——見 dominant_flow_action 凍結定義
            "dominant_action": dominant_flow_action(r),
        })
    return events


# ── persistence + alerts ─────────────────────────────────────────────────────

def insert_events(events: list[dict],
                  def_version: str = DEF_VERSION,
                  symbol: str = DEFAULT_SYMBOL) -> list[dict]:
    """INSERT IGNORE; returns the subset that was newly inserted.

    def_version defaults to v1; the v2 pass passes DEF_VERSION_V2 so its
    rows coexist under the UNIQUE key without colliding with or re-labelling
    v1 rows. Since migration 017 that key is
    (def_version, canonical_symbol, minute_start_ms, playbook) — without the
    symbol in it, two coins firing the same playbook in the same minute would
    silently drop one row."""
    if not events:
        return []
    fresh = []
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            for e in events:
                e.setdefault("symbol", symbol)
                cur.execute(
                    "INSERT IGNORE INTO cancel_playbook_events "
                    "(def_version, canonical_symbol, minute_start_ms, "
                    " playbook, direction, px, "
                    " shock, skew15, net15, vshock, taker_ratio, ret_1m) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (def_version, symbol, e["minute_start_ms"], e["playbook"],
                     e["direction"], e["px"], e["shock"], e["skew15"],
                     e["net15"], e["vshock"], e["taker_ratio"], e["ret_1m"]))
                if cur.rowcount > 0:
                    e["id"] = cur.lastrowid    # for the A3 verdict buttons
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
    # 2026-07-19: 撤單流獨立 bot — CANCEL_TG_BOT_TOKEN 優先, 未設 fallback 主 bot
    return (val("CANCEL_TG_BOT_TOKEN", "TELEGRAM_BOT_TOKEN"),
            val("TG_CRITICAL_CHAT_ID", "TELEGRAM_CHAT_ID"))


def _alert_worthy(e: dict) -> bool:
    """Push only 'big' events: huge volume or a real net-withdrawal move."""
    vs = e.get("vshock")
    nt = e.get("net15")
    return ((vs is not None and vs >= ALERT_MIN_VSHOCK)
            or (nt is not None and abs(nt) >= ALERT_MIN_NET))


def _cooldown_ok(conn, minute_ms: int,
                 symbol: str = DEFAULT_SYMBOL) -> bool:
    """Cooldown — at most one alert per COOLDOWN_MIN, any playbook, PER COIN.

    Scoped per symbol rather than globally: a BTC alert should not mute an
    ETH one, they are independent observations. The flip side is that alert
    volume scales with the number of watched coins, which is why extras are
    env-gated (see watch_symbols)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) n FROM cancel_playbook_events "
            "WHERE def_version=%s AND canonical_symbol=%s AND alerted=1 "
            "AND minute_start_ms > %s",
            (DEF_VERSION, symbol, minute_ms - COOLDOWN_MIN * 60_000))
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
            # 2026-07-29: the forward-direction claim was falsified (hit_60m
            # 41-48% on all four playbooks, n=206; path/MFE-MAE agreed) — the
            # header states the OBSERVED event, never a prediction. The
            # direction field survives as a factual label of which side was
            # withdrawn/absorbed, not as a forecast.
            lines = [
                f"🧲 撤單事件: {ZH[e['playbook']]} · "
                f"{SIDE_ZH.get(e['direction'], e['direction'])}（觀察記錄）",
                f"{t} TPE @ {px}",
                "發生了什麼: " + humanize_story(
                    e["playbook"], e["direction"],
                    e["vshock"], e["taker_ratio"]),
                "掛單面: " + humanize_book(
                    e["shock"], e["skew15"], e["net15"],
                    e.get("dominant_action")),
                f"原始值: shock {e['shock']:.1f}x 毛 {fmt(e['skew15'])} "
                f"淨 {fmt(e['net15'])} 量 {fmt(e['vshock'], '.1f')}x "
                f"taker {fmt(e['taker_ratio'], '+.0%')}",
            ]
            if chart:
                lines.append(chart)
            lines.append("深入: /cancelanalyze 90 (五步摘要) 或問 agent")
            lines.append("方向預期已於 2026-07-29 證偽 (hit_60m 41-48%, n=206;"
                         " 路徑 MFE/MAE 同結論) — 以上僅為事件描述")
            lines.append(f"def {DEF_VERSION} · 研究非信號 · edge 未驗證"
                         " · 勿作交易依據")
            payload = {"chat_id": chat, "text": "\n".join(lines)}
            if e.get("id"):     # 行動鍵 (zoom/deep/star/dismiss)
                payload["reply_markup"] = json.dumps(
                    action_keyboard("pb", int(e["id"])))
            msg_id = None
            try:
                resp = requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    data=payload, timeout=15)
                ok = resp.status_code == 200
                if ok:      # 存 message_id → 判定窗到期 reply 對答案
                    try:
                        msg_id = int(resp.json()["result"]["message_id"])
                    except Exception:
                        pass
            except Exception:
                logger.exception("playbook alert send failed")
                ok = False
            if ok:
                with conn.cursor() as cur:
                    cur.execute(
                        # symbol is part of the key since migration 017 —
                        # without it, marking a BTC alert also flips the ETH
                        # row that fired the same playbook in the same minute.
                        "UPDATE cancel_playbook_events SET alerted=1, "
                        "tg_message_id=COALESCE(%s, tg_message_id) "
                        "WHERE def_version=%s AND canonical_symbol=%s "
                        "AND minute_start_ms=%s AND playbook=%s",
                        (msg_id, DEF_VERSION,
                         e.get("symbol", DEFAULT_SYMBOL),
                         e["minute_start_ms"], e["playbook"]))
                conn.commit()
    finally:
        conn.close()


def backfill_outcomes() -> None:
    """Fill fwd returns for events whose horizons have elapsed.

    2026-07-21: SELECT 放寬成「fwd_ret_120m 或 first_hit_result 任一缺」，
    讓已經跑完 120m 判定窗、但還沒有 first_hit_result 的舊事件也能被撈到
    補算（一次性追平新加的並行診斷，不用另寫回填腳本）。"""
    now_ms = int(time.time() * 1000)
    conn = get_db_conn()
    try:
        due = _q(conn, "SELECT id, minute_start_ms ms, px, direction, "
                       "canonical_symbol sym FROM cancel_playbook_events "
                       "WHERE (fwd_ret_120m IS NULL OR first_hit_result IS NULL) "
                       "AND px IS NOT NULL AND minute_start_ms <= %s",
                 (now_ms - 31 * 60_000,))
        if due.empty:
            return
        due["ms"] = pd.to_numeric(due["ms"])
        due["px"] = pd.to_numeric(due["px"])
        lo = int(due["ms"].min())
        hi = int(due["ms"].max()) + (HORIZONS_MIN[-1] + 2) * 60_000
        # One mid series PER SYMBOL — scoring an ETH event against BTC mids
        # would silently produce plausible-looking but meaningless returns.
        mids_by_sym: dict[str, pd.Series] = {}
        for sym in sorted(due["sym"].dropna().unique()):
            ob = _q(conn, "SELECT ts_ms, mid_price mid "
                          "FROM orderbook_snapshots_1m "
                          "WHERE canonical_symbol=%s "
                          "AND ts_ms BETWEEN %s AND %s ORDER BY ts_ms",
                    (sym, lo, hi))
            if ob.empty:
                continue
            ob["ts_ms"] = pd.to_numeric(ob["ts_ms"])
            ob["mid"] = pd.to_numeric(ob["mid"])
            mids_by_sym[sym] = ob.groupby(ob["ts_ms"] // 60_000)["mid"].last()
        if not mids_by_sym:
            return
        with conn.cursor() as cur:
            for _, r in due.iterrows():
                mids = mids_by_sym.get(r.get("sym") or DEFAULT_SYMBOL)
                if mids is None:
                    continue
                m0 = int(r["ms"]) // 60_000
                sets, vals = [], []
                fwd60 = None
                window_complete = False
                for h in HORIZONS_MIN:
                    if now_ms < int(r["ms"]) + (h + 1) * 60_000:
                        continue
                    if m0 + h in mids.index:
                        fwd = float(mids.loc[m0 + h] / r["px"] - 1)
                        sets.append(f"fwd_ret_{h}m=%s")
                        vals.append(fwd)
                        if h == 60:
                            fwd60 = fwd
                        if h == 120:
                            window_complete = True
                if fwd60 is not None and r["direction"] in ("UP", "DOWN"):
                    sets.append("hit_60m=%s")
                    vals.append(1 if (fwd60 > 0) == (r["direction"] == "UP")
                                else 0)

                # first-hit-wins 平行診斷（見上方凍結定義）
                fwd_mids = mids.loc[(mids.index > m0)
                                    & (mids.index <= m0 + 120)].tolist()
                fh = first_hit_verdict(float(r["px"]), r["direction"], fwd_mids)
                if fh in ("hit", "miss") or (fh == "unresolved"
                                             and window_complete):
                    sets.append("first_hit_result=%s")
                    vals.append(fh)

                if sets:
                    vals.append(int(r["id"]))
                    cur.execute("UPDATE cancel_playbook_events SET "
                                + ", ".join(sets) + " WHERE id=%s", vals)
        conn.commit()
    except Exception:
        logger.exception("playbook outcome backfill failed")
    finally:
        conn.close()


def verdict_mark(direction: str, fwd_ret: float | None) -> str:
    """單一時間點的對錯標記：direction(UP/DOWN) 對照 fwd_ret 正負。
    無方向(NONE/未知)或無資料 → 空字串(不判斷，不硬掰對錯)。"""
    if direction not in ("UP", "DOWN") or fwd_ret is None:
        return ""
    hit = (fwd_ret > 0) == (direction == "UP")
    return "✅ 對了" if hit else "❌ 錯了"


def format_outcome_reply(e: dict, stats: tuple[int, float] | None) -> str:
    """「對答案」reply 文字：事件結果 + 劇本滾動戰績。2026-07-21 加上
    醒目的「判讀結果」標頭行(對/錯一眼可見)，不用再從數字旁的小 emoji
    去找；120m 也補上對錯標記(之前只有 60m 有)。"""
    t = (pd.Timestamp(int(e["minute_start_ms"]), unit="ms")
         + pd.Timedelta(hours=8)).strftime("%m-%d %H:%M")
    px = f"{e['px']:,.0f}" if e.get("px") else "?"
    call = PLAYBOOK_CALL_ZH.get(e["playbook"])
    call_tag = f"（{call}）" if call else ""
    head = (f"↩️ 對答案: {ZH.get(e['playbook'], e['playbook'])}{call_tag}→"
            f"{DIR_ZH.get(e['direction'], e['direction'])} @ {px}（{t}）")

    direction = e.get("direction")
    m60 = verdict_mark(direction, e.get("fwd_ret_60m"))
    m120 = verdict_mark(direction, e.get("fwd_ret_120m"))
    lines = [head]
    if m60 or m120:
        lines.append(f"判讀結果: {m60 or m120}"
                     f"{'（以 60m 為準）' if m60 else '（以 120m 為準）'}")

    parts = []
    if e.get("fwd_ret_60m") is not None:
        parts.append(f"60m {e['fwd_ret_60m']:+.2%}" + (f" {m60}" if m60 else ""))
    if e.get("fwd_ret_120m") is not None:
        parts.append(f"120m {e['fwd_ret_120m']:+.2%}"
                     + (f" {m120}" if m120 else ""))
    lines.append("之後走勢: " + (" · ".join(parts) if parts else "資料不足"))

    fh = e.get("first_hit_result")
    if fh in FIRST_HIT_ZH:
        lines.append(f"先觸價判斷(±0.5%): {FIRST_HIT_ZH[fh]}"
                     f"（另一把尺，見說明）")

    if stats:
        n, wr = stats
        lines.append(f"{ZH.get(e['playbook'], e['playbook'])} "
                     f"近 {n} 筆命中率 {wr:.0%}")
    lines.append("(研究對照 · 勿作交易依據)")
    return "\n".join(lines)


def reply_outcomes() -> None:
    """判定窗回填完成 → reply 原告警卡「對答案」(2026-07-20)。

    只處理有推播過的事件（tg_message_id 非空）。送達或 4xx（卡片被刪等
    永久失敗）都標 outcome_replied，網路錯誤留給下一輪重試。"""
    token, chat = _tg_creds()
    if not token or not chat:
        return
    conn = get_db_conn()
    try:
        due = _q(conn, "SELECT id, playbook, direction, px, minute_start_ms, "
                       "fwd_ret_60m, fwd_ret_120m, hit_60m, first_hit_result, "
                       "tg_message_id FROM cancel_playbook_events "
                       "WHERE outcome_replied=0 AND tg_message_id IS NOT NULL "
                       "AND fwd_ret_120m IS NOT NULL LIMIT 10")
        if due.empty:
            return
        for _, r in due.iterrows():
            e = {k: (None if pd.isna(v) else v) for k, v in r.items()}
            for k in ("px", "fwd_ret_60m", "fwd_ret_120m"):
                e[k] = None if e[k] is None else float(e[k])
            e["hit_60m"] = None if e["hit_60m"] is None else int(e["hit_60m"])
            stats = None
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) n, AVG(hit_60m) wr FROM ("
                    "SELECT hit_60m FROM cancel_playbook_events "
                    "WHERE playbook=%s AND hit_60m IS NOT NULL "
                    "ORDER BY minute_start_ms DESC LIMIT 30) t",
                    (e["playbook"],))
                s = cur.fetchone()
                if s and s["n"]:
                    stats = (int(s["n"]), float(s["wr"]))
            permanent = False
            try:
                resp = requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    data={"chat_id": chat,
                          "text": format_outcome_reply(e, stats),
                          "reply_to_message_id": int(e["tg_message_id"]),
                          "allow_sending_without_reply": "true"},
                    timeout=15)
                permanent = (resp.status_code == 200
                             or 400 <= resp.status_code < 500)
            except Exception:
                logger.exception("outcome reply send failed")
            if permanent:
                with conn.cursor() as cur:
                    cur.execute("UPDATE cancel_playbook_events "
                                "SET outcome_replied=1 WHERE id=%s",
                                (int(e["id"]),))
                conn.commit()
    finally:
        conn.close()


# ── daemon loop ──────────────────────────────────────────────────────────────

def watch_loop() -> None:
    ensure_schema()
    syms = watch_symbols()
    logger.info("cancel-playbook watcher started (def=%s symbols=%s)",
                DEF_VERSION, ",".join(syms))
    # last_seen is per symbol: coins have independent data availability, and
    # one coin stalling must not advance another's cursor past unscanned bars.
    last_seen: dict[str, int] = {s: 0 for s in syms}
    while True:
        for sym in syms:
            try:
                df = load_frame(lookback_min=100, symbol=sym)
                if df.empty:
                    continue
                feat = compute_features(df)
                closed = int(time.time() // 60) - 1   # last fully closed minute
                minutes = [m for m in feat.index
                           if last_seen[sym] < m <= closed]
                minutes = minutes[-5:]                # catch-up cap
                fresh = insert_events(scan(feat, minutes), symbol=sym)
                alert_events(fresh)
                # vacuum_lead — parallel, log-only, NO alerts (provisional
                # family; env-gated so it can't start on a bare redeploy).
                if _lead_enabled():
                    insert_events(scan(feat, minutes, classify_minute_lead),
                                  def_version=DEF_VERSION_LEAD, symbol=sym)
                if minutes:
                    last_seen[sym] = max(minutes)
            except Exception:
                # per-symbol isolation: a bad coin must not stop the others
                logger.exception("playbook watcher cycle failed sym=%s", sym)
        try:
            backfill_outcomes()
            reply_outcomes()
        except Exception:
            logger.exception("playbook outcome pass failed")
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
