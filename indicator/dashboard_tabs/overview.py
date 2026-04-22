"""Tab 1: Overview — current state at a glance."""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from indicator.dashboard_tabs._components import (
    card, dot, section, get_db_conn, TZ8,
)

logger = logging.getLogger(__name__)


def render_overview(state: dict, engine) -> str:
    pred = state.get("last_prediction", {})
    status = state.get("status", "unknown")
    last_update = state.get("last_update", "N/A")
    error = state.get("error")
    indicator_df = state.get("indicator_df")

    health = state.get("health", {})
    overall_health = health.get("overall_status", "unknown")

    # Engine info
    dir_n, mag_n = "?", "?"
    if engine:
        dir_n = len(getattr(engine, 'dual_dir_features',
                            getattr(engine, 'dir_feature_cols', [])))
        mag_n = len(getattr(engine, 'dual_mag_features', []))

    # ── Key metrics ──
    risk_color = "#999"
    direction = pred.get("direction", "?")
    dir_color = "#00CC80" if direction == "UP" else "#FF3366" if direction == "DOWN" else "rgba(0,240,255,0.3)"
    health_color = {"healthy": "#00CC80", "degraded": "#CC4444",
                    "critical": "#FF3366"}.get(overall_health, "rgba(0,240,255,0.3)")

    metrics_html = f"""<div class="grid">
      {card("狀態",
            dot(overall_health in ("healthy", "unknown")) + " " + overall_health.upper(),
            last_update[:19] if last_update != "N/A" else "N/A", health_color)}
      {card("方向", direction,
            f'P(UP)={pred.get("dir_prob_up", 0):.0%} | {pred.get("strength", "?")}',
            dir_color)}
      {card("信心", f'{pred.get("confidence", 0):.0f}',
            f'${pred.get("close", 0):,.0f}')}
      {card("Magnitude", f'{pred.get("mag_pred", 0)*100:.2f}%' if pred.get("mag_pred") else "N/A",
            f'Regime: {pred.get("regime", "?")}')}
      {card("模型", "Dual v7", f'Dir={dir_n}f Mag={mag_n}f')}
    </div>"""

    # ── Current prediction detail ──
    pred_detail = _build_prediction_detail(pred, engine)

    # ── 24h distribution ──
    dist_24h = _get_24h_distribution()
    up_n = dist_24h.get("UP", 0)
    dn_n = dist_24h.get("DOWN", 0)
    nt_n = dist_24h.get("NEUTRAL", 0)
    total_24 = max(up_n + dn_n + nt_n, 1)
    up_pct = up_n / total_24 * 100
    dn_pct = dn_n / total_24 * 100
    nt_pct = nt_n / total_24 * 100

    # Signal tier counts (24h)
    sig_24h = _get_24h_signal_tiers()
    strong_n = sig_24h.get("Strong", 0)
    mod_n = sig_24h.get("Moderate", 0)
    sig_summary = ""
    if strong_n or mod_n:
        parts = []
        if strong_n:
            parts.append(f'<span style="color:#FF3366;font-weight:600">Strong {strong_n}</span>')
        if mod_n:
            parts.append(f'<span style="color:#CC4444;font-weight:600">Moderate {mod_n}</span>')
        sig_summary = f'<div style="font-size:11px;margin-top:4px">信號: {" | ".join(parts)}</div>'

    dist_html = f"""
      <div style="color:rgba(0,240,255,0.5);font-size:11px;margin-bottom:4px;">最近 24 根 bar 方向分佈</div>
      <div class="dist-bar">
        <div style="background:#00CC80;flex:{up_pct:.0f}">UP {up_n}</div>
        <div style="background:rgba(0,240,255,0.3);flex:{nt_pct:.0f}">N {nt_n}</div>
        <div style="background:#FF3366;flex:{dn_pct:.0f}">DN {dn_n}</div>
      </div>{sig_summary}"""

    # ── Regime timeline ──
    regime_html = _build_regime_timeline()

    # ── Market data overview ──
    market_html = _build_market_overview()

    parts = [
        metrics_html,
        section("當前預測詳情", "pred_detail", True, pred_detail),
        section("24h 預測分佈", "dist24", True, dist_html),
        section("市場狀態時間軸 (48h)", "regime", True, regime_html),
        section("市場數據概覽", "mkt_overview", True, market_html),
    ]
    return "\n".join(parts)


def _build_prediction_detail(pred: dict, engine) -> str:
    """Show detailed breakdown of current prediction + SHAP top drivers."""
    if not pred:
        return '<div style="color:rgba(0,240,255,0.3)">等待首次預測...</div>'

    rows = []
    fields = [
        ("pred_return_4h", "預測回報 (4h)", lambda v: f"{v*100:+.3f}%"),
        ("direction", "方向", str),
        ("strength", "強度", str),
        ("confidence", "信心分數", lambda v: f"{v:.0f}"),
        ("mag_pred", "波動預測", lambda v: f"{v*100:.3f}%"),
        ("regime", "Regime", str),
        ("dir_prob_up", "P(UP) raw", lambda v: f"{v:.4f}"),
    ]
    for key, label, fmt in fields:
        val = pred.get(key)
        if val is not None:
            rows.append(f"<tr><td style='color:rgba(0,240,255,0.5)'>{label}</td>"
                        f"<td style='font-weight:600'>{fmt(val)}</td></tr>")

    detail_table = f"<table style='width:auto'>{chr(10).join(rows)}</table>"

    # SHAP top drivers
    shap_html = ""
    shap_drivers = pred.get("shap_drivers") or pred.get("top_drivers")
    if shap_drivers and isinstance(shap_drivers, list):
        shap_rows = []
        for d in shap_drivers[:5]:
            name = d.get("feature", d.get("name", "?"))
            impact = d.get("impact", d.get("shap_value", 0))
            value = d.get("value", "")
            bar_color = "#00CC80" if impact > 0 else "#FF3366"
            bar_w = min(abs(impact) * 500, 80)
            shap_rows.append(
                f"<tr><td style='font-size:11px'>{name}</td>"
                f"<td style='text-align:right;font-size:11px'>{value}</td>"
                f"<td><div style='background:{bar_color};height:10px;"
                f"width:{bar_w}px;border-radius:3px'></div></td>"
                f"<td style='font-size:11px;color:{bar_color}'>{impact:+.4f}</td></tr>"
            )
        shap_html = (
            "<div style='margin-top:12px'>"
            "<div style='color:#00F0FF;font-size:12px;font-weight:600;margin-bottom:6px'>"
            "SHAP Top-5 驅動因子</div>"
            f"<table>{''.join(shap_rows)}</table></div>"
        )

    # Signal badge (Strong / Moderate)
    strength_val = pred.get("strength", "Weak")
    if strength_val == "Strong":
        badge = '<div style="background:#FF3366;color:white;display:inline-block;padding:2px 10px;border-radius:4px;font-weight:700;font-size:13px;margin-bottom:8px">強烈信號</div>'
    elif strength_val == "Moderate":
        badge = '<div style="background:#CC4444;color:white;display:inline-block;padding:2px 10px;border-radius:4px;font-weight:700;font-size:13px;margin-bottom:8px">中等信號</div>'
    else:
        badge = ''

    shap_fallback = '<div style="color:rgba(0,240,255,0.3);font-size:11px">SHAP 數據在 Strong 信號時觸發</div>'

    return f"""<div class="two-col">
      <div>{badge}{detail_table}</div>
      <div>{shap_html if shap_html else shap_fallback}</div>
    </div>"""


def _get_24h_distribution() -> dict:
    dist = {"UP": 0, "DOWN": 0, "NEUTRAL": 0}
    try:
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


def _get_24h_signal_tiers() -> dict:
    """Count Strong/Moderate signals in last 24h from indicator_history."""
    tiers = {"Strong": 0, "Moderate": 0}
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT strength_code, COUNT(*) as cnt
                FROM indicator_history
                WHERE dt >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                  AND strength_code >= 2
                GROUP BY strength_code
            """)
            for r in cur.fetchall():
                code = int(r["strength_code"] or 0)
                if code == 3:
                    tiers["Strong"] = int(r["cnt"])
                elif code == 2:
                    tiers["Moderate"] = int(r["cnt"])
        conn.close()
    except Exception:
        pass
    return tiers


def _build_regime_timeline() -> str:
    try:
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
            return '<div style="color:rgba(0,240,255,0.3)">無數據</div>'

        regime_map = {2: "BULL", -2: "BEAR", 0: "CHOPPY", -99: "WARMUP"}
        segments = []
        current_regime = regime_map.get(int(rows[0]["regime_code"] or 0), "?")
        current_start = rows[0]["dt"]

        for r in rows[1:]:
            reg = regime_map.get(int(r["regime_code"] or 0), "?")
            if reg != current_regime:
                hours = max(1, int((r["dt"] - current_start).total_seconds() / 3600))
                segments.append({"regime": current_regime, "hours": hours})
                current_regime = reg
                current_start = r["dt"]

        hours = max(1, int((rows[-1]["dt"] - current_start).total_seconds() / 3600))
        segments.append({"regime": current_regime, "hours": hours})

        total_h = sum(s["hours"] for s in segments)
        regime_blocks = ""
        for s in segments:
            rc = {"BULL": "#00CC80", "BEAR": "#FF3366", "CHOPPY": "#CC4444",
                  "WARMUP": "rgba(0,240,255,0.3)"}.get(s["regime"], "rgba(0,240,255,0.3)")
            pct = s["hours"] / total_h * 100
            regime_blocks += (
                f'<div class="regime-block" style="background:{rc};flex:{s["hours"]}"'
                f' title="{s["regime"]} ({s["hours"]}h)">{s["regime"][:4]}</div>'
            )

        # Percentage summary
        regime_counts = {}
        for s in segments:
            regime_counts[s["regime"]] = regime_counts.get(s["regime"], 0) + s["hours"]
        summary_parts = []
        for reg, hrs in sorted(regime_counts.items(), key=lambda x: -x[1]):
            pct = hrs / total_h * 100
            summary_parts.append(f"{reg}: {pct:.0f}%")
        summary = " | ".join(summary_parts)

        return f"""
          <div style="color:rgba(0,240,255,0.5);font-size:11px;margin-bottom:4px">最近 48h | {summary}</div>
          <div class="regime-row">{regime_blocks}</div>
        """
    except Exception:
        return '<div style="color:rgba(0,240,255,0.3)">無數據</div>'


def _build_market_overview() -> str:
    """Funding rate, OI change, 24h volume, DVOL from indicator_history."""
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            # Latest row
            cur.execute("""
                SELECT * FROM indicator_history
                ORDER BY dt DESC LIMIT 1
            """)
            latest = cur.fetchone()

            # 24h ago for comparison
            cur.execute("""
                SELECT * FROM indicator_history
                WHERE dt <= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                ORDER BY dt DESC LIMIT 1
            """)
            ago_24h = cur.fetchone()
        conn.close()

        if not latest:
            return '<div style="color:rgba(0,240,255,0.3)">無數據</div>'

        def _safe(row, col, default=0):
            v = row.get(col) if row else None
            return float(v) if v is not None else default

        funding = _safe(latest, "cg_funding_close")
        oi_close = _safe(latest, "cg_oi_close")
        oi_24h_ago = _safe(ago_24h, "cg_oi_close") if ago_24h else 0
        oi_change_pct = ((oi_close / oi_24h_ago - 1) * 100) if oi_24h_ago else 0
        ls_ratio = _safe(latest, "cg_ls_ratio")
        volume = _safe(latest, "volume")

        # Try to get DVOL
        dvol = _safe(latest, "deribit_dvol") if latest and "deribit_dvol" in (latest or {}) else None

        items = f"""
        <div class="grid grid-4">
          {card("Funding Rate", f'{funding*100:.4f}%' if funding else "N/A",
                "8h 資金費率", "#CC4444" if abs(funding or 0) > 0.0001 else "#00CC80")}
          {card("OI 變化 (24h)", f'{oi_change_pct:+.1f}%',
                f'OI: {oi_close/1e9:.2f}B' if oi_close > 1e6 else f'OI: {oi_close:,.0f}',
                "#00CC80" if oi_change_pct > 0 else "#FF3366")}
          {card("多空比", f'{ls_ratio:.3f}' if ls_ratio else "N/A",
                "Long/Short Ratio",
                "#00CC80" if (ls_ratio or 1) > 1 else "#FF3366")}
          {card("DVOL", f'{dvol:.1f}' if dvol else "N/A",
                "Deribit 波動率指數", "#CC4444" if (dvol or 0) > 60 else "#00CC80")}
        </div>"""
        return items
    except Exception as e:
        return f'<div style="color:rgba(0,240,255,0.3)">數據載入失敗: {e}</div>'
