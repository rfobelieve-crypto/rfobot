"""Tab 4: System Health — is the system running correctly?"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta

from indicator.dashboard_tabs._components import (
    card, dot, section, get_db_conn, TZ8,
)

logger = logging.getLogger(__name__)


def render_health(state: dict, engine) -> str:
    health = state.get("health", {})
    overall_health = health.get("overall_status", "unknown")
    health_checks = health.get("checks", [])
    cg_status = state.get("cg_status")
    error = state.get("error")

    parts = [
        section("Pipeline 延遲", "latency", True, _build_pipeline_latency(state)),
        section("API 回應時間", "api_rt", True, _build_api_response_times()),
        section("特徵健康度", "feat_health", True, _build_feature_health()),
        section("數據新鮮度", "fresh", True, _build_freshness()),
        section("系統健康", "syshealth", True,
                _build_system_health(overall_health, health_checks, cg_status,
                                     engine, error)),
        section("警報歷史 (7 天)", "alerts", False, _build_alert_history()),
    ]
    return "\n".join(parts)


# ── Pipeline Latency ─────────────────────────────────────────────────

def _build_pipeline_latency(state: dict) -> str:
    """Show timing of last update cycle stages."""
    timing = state.get("last_timing", {})

    if not timing:
        # Try to extract from state
        last_update = state.get("last_update", "N/A")
        return f"""
        <div style="color:rgba(0,240,255,0.5);font-size:11px;margin-bottom:6px">
          上次更新: {last_update}
        </div>
        <div style="color:rgba(0,240,255,0.3)">詳細延遲數據將在下次更新後可用。</div>
        <div style="color:rgba(0,240,255,0.5);font-size:10px;margin-top:8px">
          Pipeline 階段: 數據抓取 → 特徵計算 → 模型預測 → 圖表生成 → DB 寫入
        </div>"""

    stages = [
        ("data_fetch", "數據抓取", "#00F0FF"),
        ("feature_build", "特徵計算", "#00CC80"),
        ("model_predict", "模型預測", "#CC4444"),
        ("chart_render", "圖表生成", "#FF3366"),
        ("db_write", "DB 寫入", "#FF3366"),
        ("telegram_send", "推送通知", "rgba(0,240,255,0.5)"),
    ]

    total = sum(timing.get(k, 0) for k, _, _ in stages)
    bars = []
    for key, label, color in stages:
        dur = timing.get(key, 0)
        if dur > 0:
            pct = dur / max(total, 0.1) * 100
            bars.append(
                f'<div class="latency-bar">'
                f'  <span class="latency-label">{label}</span>'
                f'  <div class="latency-track">'
                f'    <div class="latency-fill" style="width:{pct:.0f}%;background:{color}"></div>'
                f'  </div>'
                f'  <span class="latency-val">{dur:.1f}s</span>'
                f'</div>'
            )

    return f"""
    <div style="color:rgba(0,240,255,0.5);font-size:11px;margin-bottom:8px">
      總耗時: {total:.1f}s
    </div>
    {''.join(bars) if bars else '<div style="color:rgba(0,240,255,0.3)">無延遲數據</div>'}
    """


# ── API Response Times ───────────────────────────────────────────────

def _build_api_response_times() -> str:
    """Show recent API response times for each data source."""
    # Check if we have timing data in a known location
    apis = []

    # Test Binance
    try:
        import requests
        t0 = time.time()
        r = requests.get("https://api.binance.com/api/v3/time", timeout=5)
        lat = (time.time() - t0) * 1000
        apis.append(("Binance", lat, r.status_code == 200))
    except Exception:
        apis.append(("Binance", -1, False))

    # Test Coinglass (just check connectivity, don't burn API quota)
    try:
        t0 = time.time()
        r = requests.head("https://open-api-v3.coinglass.com", timeout=5)
        lat = (time.time() - t0) * 1000
        apis.append(("Coinglass", lat, r.status_code < 500))
    except Exception:
        apis.append(("Coinglass", -1, False))

    # Test Deribit
    try:
        t0 = time.time()
        r = requests.get("https://www.deribit.com/api/v2/public/get_time", timeout=5)
        lat = (time.time() - t0) * 1000
        apis.append(("Deribit", lat, r.status_code == 200))
    except Exception:
        apis.append(("Deribit", -1, False))

    # Test MySQL
    try:
        t0 = time.time()
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        lat = (time.time() - t0) * 1000
        apis.append(("MySQL", lat, True))
    except Exception:
        apis.append(("MySQL", -1, False))

    rows = []
    for name, lat, ok in apis:
        if not ok:
            color = "#FF3366"
            lat_str = "FAILED"
            bar_w = 0
        else:
            if lat < 200:
                color = "#00CC80"
            elif lat < 500:
                color = "#CC4444"
            else:
                color = "#FF3366"
            lat_str = f"{lat:.0f}ms"
            bar_w = min(lat / 10, 100)

        rows.append(
            f"<tr>"
            f"<td>{dot(ok)} {name}</td>"
            f"<td style='text-align:right;color:{color};font-family:monospace'>{lat_str}</td>"
            f"<td><div style='background:{color};height:8px;width:{bar_w}px;"
            f"border-radius:3px;opacity:0.7'></div></td>"
            f"</tr>"
        )

    return f"""
    <div style="color:rgba(0,240,255,0.5);font-size:11px;margin-bottom:6px">即時連通性測試</div>
    <table>
      <tr><th>API</th><th>延遲</th><th></th></tr>
      {''.join(rows)}
    </table>"""


# ── Feature Health ───────────────────────────────────────────────────

def _build_feature_health() -> str:
    """Check which features are NaN or stale in the latest prediction."""
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM indicator_history
                ORDER BY dt DESC LIMIT 1
            """)
            latest = cur.fetchone()

            cur.execute("SHOW COLUMNS FROM indicator_history")
            all_cols = [r["Field"] for r in cur.fetchall()]
        conn.close()
    except Exception as e:
        return f'<div style="color:rgba(0,240,255,0.3)">{e}</div>'

    if not latest:
        return '<div style="color:rgba(0,240,255,0.3)">無數據</div>'

    # Feature columns (skip metadata columns)
    skip = {"id", "dt", "close", "open", "high", "low", "volume",
            "pred_direction", "pred_direction_code", "strength",
            "strength_code", "confidence_score", "mag_pred",
            "bull_bear_power", "regime", "regime_code",
            "pred_return_4h", "created_at"}

    feature_cols = [c for c in all_cols if c not in skip and not c.startswith("_")]

    nan_features = []
    ok_features = []
    for col in feature_cols:
        val = latest.get(col)
        if val is None:
            nan_features.append(col)
        else:
            ok_features.append(col)

    total = len(feature_cols)
    nan_n = len(nan_features)
    ok_n = len(ok_features)

    health_color = "#00CC80" if nan_n == 0 else "#CC4444" if nan_n < 5 else "#FF3366"

    # Group NaN features by prefix
    nan_groups = {}
    for f in nan_features:
        prefix = f.split("_")[0] if "_" in f else f
        nan_groups.setdefault(prefix, []).append(f)

    nan_detail = ""
    if nan_features:
        group_items = []
        for prefix, feats in sorted(nan_groups.items()):
            if len(feats) <= 3:
                group_items.append(
                    f"<div style='margin:2px 0'>"
                    f"<code style='color:#FF3366'>{', '.join(feats)}</code></div>"
                )
            else:
                group_items.append(
                    f"<div style='margin:2px 0'>"
                    f"<code style='color:#FF3366'>{prefix}_* ({len(feats)} 個)</code>"
                    f"<span style='color:rgba(0,240,255,0.5);font-size:10px;margin-left:4px'>"
                    f"{', '.join(feats[:3])}...</span></div>"
                )
        nan_detail = (
            "<div style='margin-top:8px'>"
            "<div style='color:#CC4444;font-size:11px;font-weight:600;margin-bottom:4px'>"
            "NaN 特徵:</div>"
            + "".join(group_items) + "</div>"
        )

    return f"""
    <div class="grid grid-3">
      {card("特徵總數", str(total), "")}
      {card("正常", str(ok_n), "", "#00CC80")}
      {card("NaN", str(nan_n), "", health_color)}
    </div>
    {nan_detail}"""


# ── Data Freshness ───────────────────────────────────────────────────

def _build_freshness() -> str:
    items = []
    now_utc = datetime.now(timezone.utc)

    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            sources = [
                ("Indicator History", "SELECT MAX(dt) as last_dt FROM indicator_history"),
                ("Tracked Signals", "SELECT MAX(signal_time) as last_dt FROM tracked_signals"),
            ]
            # Check if agent_runs exists
            try:
                cur.execute("SELECT MAX(started_at) as last_dt FROM agent_runs")
                r = cur.fetchone()
                if r and r["last_dt"]:
                    sources.append(("Agent Runs", None))
                    last = r["last_dt"].replace(tzinfo=timezone.utc)
                    age = (now_utc - last).total_seconds() / 60
                    items.append(_fresh_row("Agent Runs", age, last))
            except Exception:
                pass

            for name, sql in sources:
                if sql is None:
                    continue
                try:
                    cur.execute(sql)
                    r = cur.fetchone()
                    if r and r["last_dt"]:
                        last = r["last_dt"].replace(tzinfo=timezone.utc)
                        age = (now_utc - last).total_seconds() / 60
                        items.append(_fresh_row(name, age, last))
                except Exception:
                    items.append(_fresh_row(name, 9999, None))
        conn.close()
    except Exception:
        items.append(_fresh_row("DB", 9999, None))


    rows = "".join(
        f"<tr><td>{i['source']}</td>"
        f"<td style='color:{i['color']}'>{i['age_text']}</td>"
        f"<td>{i['last_time']}</td></tr>"
        for i in items
    )
    return f"""<table><tr><th>來源</th><th>延遲</th><th>最後更新</th></tr>{rows}</table>"""


def _fresh_row(source: str, age_min: float, last_dt) -> dict:
    if age_min > 9000:
        return {"source": source, "age_text": "ERROR", "last_time": "",
                "color": "#FF3366"}
    color = "#00CC80" if age_min < 120 else "#CC4444" if age_min < 360 else "#FF3366"
    age_text = f"{age_min:.0f}min" if age_min < 120 else f"{age_min/60:.1f}h"
    last_time = ""
    if last_dt:
        last_time = last_dt.astimezone(TZ8).strftime("%m/%d %H:%M")
    return {"source": source, "age_text": age_text, "last_time": last_time,
            "color": color}


# ── System Health ────────────────────────────────────────────────────

def _build_system_health(overall, checks, cg_status, engine, error) -> str:
    color_map = {"healthy": "#00CC80", "degraded": "#CC4444", "critical": "#FF3366"}
    overall_color = color_map.get(overall, "#999")

    dir_n, mag_n = "?", "?"
    if engine:
        dir_n = len(getattr(engine, 'dual_dir_features',
                            getattr(engine, 'dir_feature_cols', [])))
        mag_n = len(getattr(engine, 'dual_mag_features', []))
    engine_info = f"Dir={dir_n}f Mag={mag_n}f" if engine else "未載入"

    # CG status
    cg_text = "N/A"
    if cg_status and isinstance(cg_status, dict):
        total = len(cg_status)
        ok_count = sum(1 for v in cg_status.values()
                       if isinstance(v, dict) and not v.get("empty", True))
        cg_text = f"{ok_count}/{total} endpoints OK"
    elif cg_status and isinstance(cg_status, str):
        cg_text = cg_status

    # DB health
    db_health = {}
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as n FROM indicator_history")
            db_health["indicator_history"] = cur.fetchone()["n"]
            try:
                cur.execute("SELECT COUNT(*) as n FROM tracked_signals")
                db_health["tracked_signals"] = cur.fetchone()["n"]
            except Exception:
                db_health["tracked_signals"] = "N/A"
        conn.close()
        db_health["status"] = "OK"
    except Exception as e:
        db_health["status"] = f"ERR: {str(e)[:40]}"

    lines = []
    lines.append(f'<div style="margin-bottom:8px">'
                 f'Overall: <span style="color:{overall_color};font-weight:700">'
                 f'{overall.upper()}</span></div>')

    # Health checks table
    lines.append('<table><tr><th>檢查項目</th><th>狀態</th><th>詳情</th></tr>')
    if checks:
        for c in checks:
            name = c.get("name", "?")
            ok = c.get("ok", False)
            detail = c.get("detail", "")[:80]
            lines.append(
                f'<tr><td>{name}</td><td>{dot(ok)}</td><td>{detail}</td></tr>')
    lines.append('</table>')

    # Components
    db_ok = db_health.get("status") == "OK"
    db_detail = (f'history: {db_health.get("indicator_history", "?")} | '
                 f'signals: {db_health.get("tracked_signals", "?")}')
    cg_ok = "OK" in cg_text or "/" in cg_text
    err_text = "None" if error is None else str(error)[:80]

    lines.append('<div style="margin-top:10px"><table>')
    lines.append('<tr><th>組件</th><th>狀態</th><th>詳情</th></tr>')
    lines.append(f'<tr><td>MySQL</td><td>{dot(db_ok)}</td><td>{db_detail}</td></tr>')
    lines.append(f'<tr><td>Coinglass</td><td>{dot(cg_ok)}</td><td>{cg_text}</td></tr>')
    lines.append(f'<tr><td>Engine</td><td>{dot(engine is not None)}</td>'
                 f'<td>{engine_info}</td></tr>')
    lines.append(f'<tr><td>Error</td><td>{dot(error is None)}</td><td>{err_text}</td></tr>')
    lines.append('</table></div>')

    return "\n".join(lines)


# ── Alert History ────────────────────────────────────────────────────

def _build_alert_history() -> str:
    import pandas as pd
    from pathlib import Path

    alert_file = Path("research/monitor_alerts.csv")
    if not alert_file.exists():
        return '<div style="color:#00CC80">過去 7 天無警報</div>'

    try:
        df = pd.read_csv(alert_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        df = df[df["timestamp"] >= cutoff]
        df = df.sort_values("timestamp", ascending=False)

        if df.empty:
            return '<div style="color:#00CC80">過去 7 天無警報</div>'

        total = len(df)
        alerts = []
        for _, row in df.head(20).iterrows():
            alert_type = str(row.get("alert_type", "unknown"))
            severity = "critical" if "CRITICAL" in alert_type or "🔴" in alert_type else "warn"
            color = "#FF3366" if severity == "critical" else "#CC4444"
            t = row["timestamp"]
            if hasattr(t, "strftime"):
                t = t.strftime("%m/%d %H:%M")
            alerts.append(f'<tr><td>{t}</td><td style="color:{color}">'
                          f'{alert_type.split(":")[0][:30]}</td>'
                          f'<td>{alert_type[:120]}</td></tr>')

        critical = sum(1 for _, r in df.iterrows()
                       if "CRITICAL" in str(r.get("alert_type", "")))

        return (
            f'<div style="color:rgba(0,240,255,0.5);font-size:11px;margin-bottom:6px">'
            f'總計: {total} 則警報 ({critical} 嚴重)</div>'
            f'<table><tr><th>時間</th><th>類型</th><th>詳情</th></tr>'
            f'{"".join(alerts)}</table>'
        )
    except Exception:
        return '<div style="color:rgba(0,240,255,0.3)">警報讀取失敗</div>'
