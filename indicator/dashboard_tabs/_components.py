"""Shared HTML component helpers for dashboard tabs."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

TZ8 = timezone(timedelta(hours=8))


def card(title: str, value: str, subtitle: str = "", color: str = "#4fc3f7") -> str:
    return f"""<div class="card">
      <div class="card-title">{title}</div>
      <div class="card-value" style="color:{color}">{value}</div>
      <div class="card-sub">{subtitle}</div>
    </div>"""


def dot(ok: bool) -> str:
    cls = "dot-ok" if ok else "dot-err"
    return f'<span class="{cls} dot"></span>'


def section(title: str, sec_id: str, open_default: bool, body: str) -> str:
    display = "block" if open_default else "none"
    arrow_char = "&#9660;" if open_default else "&#9654;"
    return f"""<div class="section">
      <div class="section-header" onclick="toggle('{sec_id}')">
        <span class="section-title">{title}</span>
        <span class="section-toggle" id="{sec_id}_arrow">{arrow_char}</span>
      </div>
      <div class="section-body" id="{sec_id}" style="display:{display}">{body}</div>
    </div>"""


def status_dot(status: str) -> str:
    """Return colored dot for status strings like healthy/warning/critical."""
    colors = {
        "healthy": "#00CC80", "warning": "#C300FF", "critical": "#FF00FF",
        "error": "#FF00FF", "insufficient_data": "rgba(0,240,255,0.3)", "no_history": "rgba(0,240,255,0.3)",
    }
    c = colors.get(status, "rgba(0,240,255,0.3)")
    return f'<span class="dot" style="background:{c}"></span>'


def status_badge(status: str) -> str:
    colors = {
        "healthy": "#00CC80", "warning": "#C300FF", "critical": "#FF00FF",
        "error": "#FF00FF", "insufficient_data": "rgba(0,240,255,0.3)",
    }
    c = colors.get(status, "rgba(0,240,255,0.3)")
    label = {"healthy": "健康", "warning": "警告", "critical": "危險",
             "error": "錯誤", "insufficient_data": "等待中"}.get(status, status.upper())
    return f'<span class="badge" style="background:{c}">{label}</span>'


def mini_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a small HTML table."""
    h = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{c}</td>" for c in row)
        body += f"<tr>{cells}</tr>"
    return f"<table><tr>{h}</tr>{body}</table>"


def gauge_bar(value: float, max_val: float = 100, color: str = "#00F0FF",
              label: str = "") -> str:
    """Horizontal gauge bar."""
    pct = min(max(value / max_val * 100, 0), 100)
    return f"""<div class="gauge-wrap">
      <div class="gauge-label">{label}</div>
      <div class="gauge-track">
        <div class="gauge-fill" style="width:{pct:.0f}%;background:{color}"></div>
      </div>
      <div class="gauge-val">{value:.1f}</div>
    </div>"""


def now_str() -> str:
    return datetime.now(TZ8).strftime("%Y-%m-%d %H:%M UTC+8")


def get_db_conn():
    from shared.db import get_db_conn as _get
    return _get()
