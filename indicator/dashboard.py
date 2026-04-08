"""
System diagnostic dashboard — single page HTML.

Renders a dark-theme responsive dashboard with:
  - Key metric cards (prediction, risk, model)
  - Signal performance table (Strong + Moderate)
  - 24h prediction distribution (UP/DOWN/NEUTRAL)
  - Regime timeline
  - Data freshness (API last update times)
  - System health
  - Collapsible sections for mobile
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

TZ8 = timezone(timedelta(hours=8))


def render_dashboard(state: dict, engine) -> str:
    """Build full dashboard HTML from app state and engine."""
    import pandas as pd

    now = datetime.now(TZ8).strftime("%Y-%m-%d %H:%M UTC+8")

    pred = state.get("last_prediction", {})
    status = state.get("status", "unknown")
    last_update = state.get("last_update", "N/A")
    error = state.get("error")
    cg_status = state.get("cg_status")
    indicator_df = state.get("indicator_df")

    # ── Engine info ──
    engine_info = "N/A"
    dir_n, mag_n = "?", "?"
    if engine:
        dir_n = len(getattr(engine, 'dual_dir_features', getattr(engine, 'dir_feature_cols', [])))
        mag_n = len(getattr(engine, 'dual_mag_features', []))
        engine_info = f"{engine.mode} | Dir={dir_n} feat | Mag={mag_n} feat"

    # ── Signal tracker ──
    sig_stats = _get_signal_stats()

    # ── Entropy risk ──
    risk_info = _get_risk_info(indicator_df, pred)

    # ── DB health ──
    db_health = _get_db_health()

    # ── CG status ──
    cg_text = _format_cg_status(cg_status)

    # ── System health (from health_monitor) ──
    health = state.get("health", {})
    health_checks = health.get("checks", [])
    overall_health = health.get("overall_status", "unknown")

    # ── 24h prediction distribution ──
    dist_24h = _get_24h_distribution()

    # ── Regime timeline ──
    regime_timeline = _get_regime_timeline()

    # ── Data freshness ──
    freshness = _get_data_freshness()

    # ── Build HTML ──
    risk_color = {"HIGH": "#f44336", "MEDIUM": "#ff9800", "LOW": "#4caf50"}.get(
        risk_info.get("risk_level", ""), "#999")

    # Recent signals rows
    recent_html = ""
    for r in sig_stats.get("recent", []):
        t = r["signal_time"]
        if hasattr(t, "replace"):
            t = t.replace(tzinfo=timezone.utc).astimezone(TZ8).strftime("%m/%d %H:%M")
        d = r["direction"]
        s = r["strength"][0]
        dc = "#4caf50" if d == "UP" else "#f44336"
        icon = "&#9650;" if d == "UP" else "&#9660;"
        if r["filled"]:
            ret = float(r["actual_return_4h"]) * 100
            oc = "#4caf50" if r["correct"] else "#f44336"
            ok = "&#10003;" if r["correct"] else "&#10007;"
            recent_html += f'<tr><td>{t}</td><td>[{s}]</td><td style="color:{dc}">{icon} {d}</td><td>{r["confidence"]:.0f}</td><td>${r["entry_price"]:,.0f}</td><td style="color:{oc}">{ret:+.2f}% {ok}</td></tr>'
        else:
            recent_html += f'<tr><td>{t}</td><td>[{s}]</td><td style="color:{dc}">{icon} {d}</td><td>{r["confidence"]:.0f}</td><td>${r["entry_price"]:,.0f}</td><td style="color:#666">pending</td></tr>'

    # 24h distribution bar
    up_n = dist_24h.get("UP", 0)
    dn_n = dist_24h.get("DOWN", 0)
    nt_n = dist_24h.get("NEUTRAL", 0)
    total_24 = max(up_n + dn_n + nt_n, 1)
    up_pct = up_n / total_24 * 100
    dn_pct = dn_n / total_24 * 100
    nt_pct = nt_n / total_24 * 100

    # Regime timeline rows
    regime_html = ""
    for rt in regime_timeline:
        rc = {"BULL": "#4caf50", "BEAR": "#f44336", "CHOPPY": "#ff9800", "WARMUP": "#666"}.get(rt["regime"], "#999")
        hrs = rt["hours"]
        rname = rt["regime"]
        regime_html += f'<div class="regime-block" style="background:{rc};flex:{hrs}" title="{rname} ({hrs}h)">{rname[:4]}</div>'

    # Freshness rows
    fresh_html = ""
    for f in freshness:
        age = f["age_min"]
        fc = "#4caf50" if age < 120 else "#ff9800" if age < 360 else "#f44336"
        fresh_html += f'<tr><td>{f["source"]}</td><td style="color:{fc}">{f["age_text"]}</td><td>{f["last_time"]}</td></tr>'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quant Dashboard</title>
<meta http-equiv="refresh" content="300">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0d1117; color:#c9d1d9; font-family:-apple-system,BlinkMacSystemFont,sans-serif; padding:12px; font-size:13px; }}
  h1 {{ color:#58a6ff; font-size:18px; margin-bottom:2px; }}
  .subtitle {{ color:#8b949e; font-size:11px; margin-bottom:12px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:8px; margin-bottom:14px; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:10px; }}
  .card-title {{ color:#8b949e; font-size:10px; text-transform:uppercase; letter-spacing:0.5px; }}
  .card-value {{ font-size:22px; font-weight:700; margin:2px 0; }}
  .card-sub {{ color:#8b949e; font-size:10px; }}
  .section {{ background:#161b22; border:1px solid #30363d; border-radius:8px; margin-bottom:12px; overflow:hidden; }}
  .section-header {{ padding:10px 14px; cursor:pointer; display:flex; justify-content:space-between; align-items:center; }}
  .section-header:active {{ background:#1c2129; }}
  .section-title {{ color:#58a6ff; font-size:13px; font-weight:600; }}
  .section-toggle {{ color:#8b949e; font-size:16px; }}
  .section-body {{ padding:0 14px 12px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th {{ text-align:left; color:#8b949e; padding:5px 6px; border-bottom:1px solid #30363d; font-size:10px; text-transform:uppercase; }}
  td {{ padding:5px 6px; border-bottom:1px solid #21262d; }}
  .dist-bar {{ display:flex; height:24px; border-radius:4px; overflow:hidden; margin:6px 0; }}
  .dist-bar div {{ display:flex; align-items:center; justify-content:center; font-size:10px; font-weight:600; color:#fff; }}
  .regime-row {{ display:flex; gap:2px; height:20px; border-radius:4px; overflow:hidden; margin:6px 0; }}
  .regime-block {{ display:flex; align-items:center; justify-content:center; font-size:9px; font-weight:600; color:#fff; border-radius:3px; min-width:30px; }}
  .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:4px; }}
  .dot-ok {{ background:#4caf50; }}
  .dot-err {{ background:#f44336; }}
</style>
<script>
function toggle(id) {{
  var el = document.getElementById(id);
  var arrow = document.getElementById(id + '_arrow');
  if (el.style.display === 'none') {{
    el.style.display = 'block';
    arrow.textContent = '\\u25BC';
  }} else {{
    el.style.display = 'none';
    arrow.textContent = '\\u25B6';
  }}
}}
</script>
</head><body>
<h1>Quant Dashboard</h1>
<div class="subtitle">{now} | auto-refresh 5min</div>

<div class="grid">
  {_card("Status",
         _dot(overall_health in ("healthy","unknown")) + " " + overall_health.upper(),
         (last_update[:19] if last_update != "N/A" else "N/A"),
         {"healthy":"#4caf50","degraded":"#ff9800","critical":"#f44336"}.get(overall_health, "#999"))}
  {_card("Direction", pred.get("direction","?"), f'P(UP)={pred.get("dir_prob_up",0):.0%} | {pred.get("strength","?")}',
         "#4caf50" if pred.get("direction")=="UP" else "#f44336" if pred.get("direction")=="DOWN" else "#999")}
  {_card("Confidence", f'{pred.get("confidence",0):.0f}', f'${pred.get("close",0):,.0f}')}
  {_card("Risk", f'{risk_info.get("risk_score","?")}/100', risk_info.get("risk_level","?"), risk_color)}
  {_card("Entropy", f'{risk_info.get("market_entropy","N/A")}', f'z={risk_info.get("market_entropy_zscore","?")}')}
  {_card("Model", "Dual v7", f'Dir={dir_n}f Mag={mag_n}f')}
</div>

{_section("24h Prediction", "dist24", True, f'''
  <div style="color:#8b949e;font-size:11px;margin-bottom:4px;">Last 24 bars direction distribution</div>
  <div class="dist-bar">
    <div style="background:#4caf50;flex:{up_pct:.0f}">UP {up_n}</div>
    <div style="background:#666;flex:{nt_pct:.0f}">N {nt_n}</div>
    <div style="background:#f44336;flex:{dn_pct:.0f}">DN {dn_n}</div>
  </div>
''')}

{_section("Regime Timeline", "regime", True, f'''
  <div style="color:#8b949e;font-size:11px;margin-bottom:4px;">Last 48h regime changes</div>
  <div class="regime-row">{regime_html if regime_html else '<div style="color:#666">No data</div>'}</div>
''')}

{_section("Signal Performance", "signals", True, _build_signal_html(sig_stats, recent_html))}

{_section("Data Freshness", "fresh", False, _build_freshness_html(freshness))}

{_section("System Health", "syshealth", True, _build_health_html(
    overall_health, health_checks, db_health, cg_text, engine, engine_info, error))}

</body></html>"""
    return html


def _build_signal_html(sig_stats, recent_html):
    s_s = sig_stats.get("Strong", {})
    s_m = sig_stats.get("Moderate", {})
    return f'''
  <table>
    <tr><th>Tier</th><th>Total</th><th>Filled</th><th>Win Rate</th><th>Avg Ret</th></tr>
    <tr><td>Strong</td><td>{s_s.get("total",0)}</td><td>{s_s.get("filled",0)}</td>
        <td>{s_s.get("wr","N/A")}</td><td>{s_s.get("avg_ret","N/A")}</td></tr>
    <tr><td>Moderate</td><td>{s_m.get("total",0)}</td><td>{s_m.get("filled",0)}</td>
        <td>{s_m.get("wr","N/A")}</td><td>{s_m.get("avg_ret","N/A")}</td></tr>
  </table>
  <div style="margin-top:10px">
    <table>
      <tr><th>Time</th><th>Tier</th><th>Dir</th><th>Conf</th><th>Entry</th><th>Result</th></tr>
      {recent_html}
    </table>
  </div>
'''


# ═══════════════════════════════════════════════════════════════════
# HTML helpers
# ═══════════════════════════════════════════════════════════════════

def _card(title, value, subtitle="", color="#4fc3f7"):
    return f"""<div class="card">
      <div class="card-title">{title}</div>
      <div class="card-value" style="color:{color}">{value}</div>
      <div class="card-sub">{subtitle}</div>
    </div>"""


def _dot(ok):
    cls = "dot-ok" if ok else "dot-err"
    return f'<span class="{cls} dot"></span>'


def _build_freshness_html(freshness):
    lines = ['<table><tr><th>Source</th><th>Age</th><th>Last Update</th></tr>']
    for f in freshness:
        age = f["age_min"]
        fc = "#4caf50" if age < 120 else "#ff9800" if age < 360 else "#f44336"
        lines.append(f'<tr><td>{f["source"]}</td><td style="color:{fc}">{f["age_text"]}</td><td>{f["last_time"]}</td></tr>')
    lines.append('</table>')
    return "\n".join(lines)


def _build_health_html(overall, checks, db_health, cg_text, engine, engine_info, error):
    """Build health section HTML (avoids f-string escaping issues)."""
    color_map = {"healthy": "#4caf50", "degraded": "#ff9800", "critical": "#f44336"}
    overall_color = color_map.get(overall, "#999")

    lines = []
    lines.append(f'<div style="margin-bottom:8px">')
    lines.append(f'  Overall: <span style="color:{overall_color}">{overall.upper()}</span>')
    lines.append(f'</div>')
    lines.append('<table><tr><th>Check</th><th>Status</th><th>Detail</th></tr>')

    if checks:
        for c in checks:
            name = c.get("name", "?")
            dot = _dot(c.get("ok", False))
            sev = c.get("severity", "?").upper()
            detail = c.get("detail", "")[:80]
            lines.append(f'<tr><td>{name}</td><td>{dot} {sev}</td><td>{detail}</td></tr>')
    else:
        lines.append('<tr><td colspan="3" style="color:#666">Waiting for first update cycle...</td></tr>')

    lines.append('</table>')
    lines.append('<div style="margin-top:8px"><table>')
    lines.append('<tr><th>Component</th><th>Status</th><th>Detail</th></tr>')

    db_ok = db_health.get("status") == "OK"
    db_detail = f'history: {db_health.get("indicator_history", "?")} | signals: {db_health.get("tracked_signals", "?")}'
    lines.append(f'<tr><td>MySQL</td><td>{_dot(db_ok)} {db_health.get("status", "?")}</td><td>{db_detail}</td></tr>')

    cg_ok = "OK" in cg_text or "/" in cg_text
    lines.append(f'<tr><td>Coinglass</td><td>{_dot(cg_ok)} {cg_text}</td><td></td></tr>')

    lines.append(f'<tr><td>Engine</td><td>{_dot(engine is not None)} Loaded</td><td>{engine_info}</td></tr>')

    err_text = "None" if error is None else str(error)[:80]
    lines.append(f'<tr><td>Error</td><td>{_dot(error is None)} {err_text}</td><td></td></tr>')

    lines.append('</table></div>')
    return "\n".join(lines)


def _section(title, sec_id, open_default, body):
    display = "block" if open_default else "none"
    arrow_char = "&#9660;" if open_default else "&#9654;"
    return f"""<div class="section">
      <div class="section-header" onclick="toggle('{sec_id}')">
        <span class="section-title">{title}</span>
        <span class="section-toggle" id="{sec_id}_arrow">{arrow_char}</span>
      </div>
      <div class="section-body" id="{sec_id}" style="display:{display}">{body}</div>
    </div>"""


# ═══════════════════════════════════════════════════════════════════
# Data collectors
# ═══════════════════════════════════════════════════════════════════

def _get_signal_stats() -> dict:
    stats = {}
    try:
        from indicator.signal_tracker import _ensure_table, _get_db_conn, TABLE
        _ensure_table()
        conn = _get_db_conn()
        with conn.cursor() as cur:
            for tier in ["Strong", "Moderate"]:
                cur.execute(f"""
                    SELECT COUNT(*) as t, SUM(filled) as f, SUM(correct) as w,
                           AVG(CASE WHEN filled=1 THEN actual_return_4h END) as avg_ret
                    FROM `{TABLE}` WHERE strength = %s
                """, (tier,))
                r = cur.fetchone()
                tt, ff, ww = int(r["t"] or 0), int(r["f"] or 0), int(r["w"] or 0)
                stats[tier] = {
                    "total": tt, "filled": ff, "wins": ww,
                    "wr": f"{ww/ff*100:.0f}%" if ff > 0 else "N/A",
                    "avg_ret": f"{float(r['avg_ret'] or 0)*100:+.2f}%" if ff > 0 else "N/A",
                }
            cur.execute(f"""
                SELECT signal_time, direction, strength, confidence, entry_price,
                       actual_return_4h, correct, filled
                FROM `{TABLE}` ORDER BY signal_time DESC LIMIT 5
            """)
            stats["recent"] = cur.fetchall()
        conn.close()
    except Exception as e:
        stats["error"] = str(e)
    return stats


def _get_risk_info(indicator_df, pred) -> dict:
    info = {}
    try:
        from indicator.entropy_tools import EntropyAnalyzer, EntropyRiskManager
        if indicator_df is not None and not indicator_df.empty:
            analyzer = EntropyAnalyzer()
            me = analyzer._market_entropy(indicator_df)
            info["market_entropy"] = f"{me.get('normalized', 0):.2f}" if me.get("normalized") else "N/A"
            info["market_entropy_zscore"] = f"{me.get('zscore', 0):+.1f}" if me.get("zscore") else "?"

            risk_mgr = EntropyRiskManager()
            r = risk_mgr.assess(
                indicator_df,
                dir_prob_up=pred.get("dir_prob_up", 0.5),
                confidence=pred.get("confidence", 50),
                market_regime=pred.get("regime", ""),
            )
            info["risk_score"] = f"{r['risk_score']:.0f}"
            info["risk_level"] = r["risk_level"]
    except Exception as e:
        info["error"] = str(e)
    return info


def _get_db_health() -> dict:
    health = {}
    try:
        from shared.db import get_db_conn
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as n FROM indicator_history")
            health["indicator_history"] = cur.fetchone()["n"]
            try:
                cur.execute("SELECT COUNT(*) as n FROM tracked_signals")
                health["tracked_signals"] = cur.fetchone()["n"]
            except Exception:
                health["tracked_signals"] = "N/A"
        conn.close()
        health["status"] = "OK"
    except Exception as e:
        health["status"] = f"ERR: {str(e)[:40]}"
    return health


def _format_cg_status(cg_status) -> str:
    if cg_status and isinstance(cg_status, dict):
        total = len(cg_status)
        ok_count = sum(1 for v in cg_status.values()
                       if isinstance(v, dict) and not v.get("empty", True))
        return f"{ok_count}/{total} endpoints OK"
    elif cg_status and isinstance(cg_status, str):
        return cg_status
    return "N/A"


def _get_24h_distribution() -> dict:
    """Get direction distribution for last 24 bars from indicator_history."""
    dist = {"UP": 0, "DOWN": 0, "NEUTRAL": 0}
    try:
        from shared.db import get_db_conn
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pred_direction_code, COUNT(*) as cnt
                FROM indicator_history
                WHERE dt >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                GROUP BY pred_direction_code
            """)
            for r in cur.fetchall():
                code = int(r["pred_direction_code"] or 0)
                name = {1: "UP", -1: "DOWN", 0: "NEUTRAL"}.get(code, "NEUTRAL")
                dist[name] = int(r["cnt"])
        conn.close()
    except Exception:
        pass
    return dist


def _get_regime_timeline() -> list[dict]:
    """Get regime changes over last 48h."""
    timeline = []
    try:
        from shared.db import get_db_conn
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, regime_code FROM indicator_history
                WHERE dt >= DATE_SUB(NOW(), INTERVAL 48 HOUR)
                ORDER BY dt ASC
            """)
            rows = cur.fetchall()
        conn.close()

        if not rows:
            return timeline

        regime_map = {2: "BULL", -2: "BEAR", 0: "CHOPPY", -99: "WARMUP"}
        current_regime = regime_map.get(int(rows[0]["regime_code"] or 0), "?")
        current_start = rows[0]["dt"]

        for r in rows[1:]:
            reg = regime_map.get(int(r["regime_code"] or 0), "?")
            if reg != current_regime:
                hours = max(1, int((r["dt"] - current_start).total_seconds() / 3600))
                timeline.append({"regime": current_regime, "hours": hours})
                current_regime = reg
                current_start = r["dt"]

        # Last segment
        hours = max(1, int((rows[-1]["dt"] - current_start).total_seconds() / 3600))
        timeline.append({"regime": current_regime, "hours": hours})
    except Exception:
        pass
    return timeline


def _get_data_freshness() -> list[dict]:
    """Check how fresh each data source is."""
    items = []
    now_utc = datetime.now(timezone.utc)

    # indicator_history
    try:
        from shared.db import get_db_conn
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(dt) as last_dt FROM indicator_history")
            r = cur.fetchone()
            if r and r["last_dt"]:
                last = r["last_dt"].replace(tzinfo=timezone.utc)
                age = (now_utc - last).total_seconds() / 60
                items.append({
                    "source": "Indicator History",
                    "age_min": age,
                    "age_text": f"{age:.0f}min" if age < 120 else f"{age/60:.1f}h",
                    "last_time": last.astimezone(TZ8).strftime("%m/%d %H:%M"),
                })

            cur.execute(f"SELECT MAX(signal_time) as last_dt FROM tracked_signals")
            r = cur.fetchone()
            if r and r["last_dt"]:
                last = r["last_dt"].replace(tzinfo=timezone.utc)
                age = (now_utc - last).total_seconds() / 60
                items.append({
                    "source": "Last Signal",
                    "age_min": age,
                    "age_text": f"{age:.0f}min" if age < 120 else f"{age/60:.1f}h",
                    "last_time": last.astimezone(TZ8).strftime("%m/%d %H:%M"),
                })
        conn.close()
    except Exception:
        items.append({"source": "DB", "age_min": 9999, "age_text": "ERROR", "last_time": ""})

    return items
