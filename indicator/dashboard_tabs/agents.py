"""Tab 5: Agents — agent execution status and signal performance."""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from indicator.dashboard_tabs._components import (
    card, dot, section, status_badge, get_db_conn, TZ8,
)

logger = logging.getLogger(__name__)


def render_agents() -> str:
    parts = [
        section("Agent 執行狀態", "agent_status", True, _build_agent_status()),
        section("信號績效", "sig_perf", True, _build_signal_perf()),
    ]
    return "\n".join(parts)


# ── Agent Status ─────────────────────────────────────────────────────

def _build_agent_status() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            # Check if table exists
            cur.execute("""
                SELECT TABLE_NAME FROM information_schema.TABLES
                WHERE TABLE_NAME = 'agent_runs'
                  AND TABLE_SCHEMA = DATABASE()
            """)
            if not cur.fetchone():
                return '<div style="color:rgba(0,240,255,0.3)">agent_runs 表尚未建立（等待首次 agent 排程執行）</div>'

            # Per-agent latest run
            cur.execute("""
                SELECT agent_name,
                       COUNT(*) as total_runs,
                       SUM(status = 'success') as successes,
                       SUM(status = 'failed') as failures,
                       MAX(started_at) as last_run,
                       MAX(CASE WHEN status = 'success' THEN started_at END) as last_success,
                       MAX(CASE WHEN status = 'failed' THEN started_at END) as last_fail,
                       MAX(CASE WHEN status = 'failed' THEN error_message END) as last_error,
                       AVG(TIMESTAMPDIFF(SECOND, started_at, finished_at)) as avg_dur
                FROM agent_runs
                WHERE started_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                GROUP BY agent_name
                ORDER BY agent_name
            """)
            agents = cur.fetchall()

            # Currently running
            cur.execute("""
                SELECT agent_name, started_at
                FROM agent_runs WHERE status = 'running'
            """)
            running = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:rgba(0,240,255,0.3)">{e}</div>'

    if not agents:
        return '<div style="color:rgba(0,240,255,0.3)">過去 7 天無 agent 執行記錄</div>'

    # Summary cards
    total = sum(int(a["total_runs"] or 0) for a in agents)
    total_ok = sum(int(a["successes"] or 0) for a in agents)
    total_fail = sum(int(a["failures"] or 0) for a in agents)
    overall_rate = total_ok / total * 100 if total > 0 else 0

    # Running indicator
    running_html = ""
    if running:
        names = ", ".join(r["agent_name"] for r in running)
        running_html = (
            f'<div style="background:#0D0D0D;border:1px solid #1A1A2E;'
            f'border-radius:6px;padding:8px;margin-bottom:10px">'
            f'<span class="dot dot-ok" style="animation:blink 1s infinite"></span> '
            f'正在執行: <b>{names}</b></div>'
        )

    rows = []
    for a in agents:
        name = a["agent_name"]
        total_r = int(a["total_runs"] or 0)
        ok = int(a["successes"] or 0)
        fail = int(a["failures"] or 0)
        rate = ok / total_r * 100 if total_r > 0 else 0
        avg_dur = float(a["avg_dur"] or 0)
        dur_str = f"{avg_dur:.0f}s" if avg_dur < 60 else f"{avg_dur/60:.1f}m"

        last_run = a["last_run"]
        last_str = ""
        if last_run and hasattr(last_run, "strftime"):
            last_local = last_run.replace(tzinfo=timezone.utc).astimezone(TZ8)
            last_str = last_local.strftime("%m/%d %H:%M")

        rate_color = "#00CC80" if rate >= 90 else "#C300FF" if rate >= 70 else "#FF00FF"

        # Error detail
        err_html = ""
        if fail > 0 and a["last_error"]:
            err = str(a["last_error"])[:60]
            err_html = f'<div style="color:#FF00FF;font-size:10px;margin-top:2px"><code>{err}</code></div>'

        rows.append(
            f"<tr>"
            f"<td><b>{name}</b>{err_html}</td>"
            f"<td>{total_r}</td>"
            f"<td style='color:{rate_color}'>{rate:.0f}%</td>"
            f"<td>{fail}</td>"
            f"<td>{dur_str}</td>"
            f"<td>{last_str}</td>"
            f"</tr>"
        )

    return f"""
    <div class="grid grid-4" style="margin-bottom:10px">
      {card("總執行", str(total), "過去 7 天")}
      {card("成功率", f'{overall_rate:.0f}%', f'{total_ok} ok / {total_fail} fail',
            "#00CC80" if overall_rate >= 90 else "#C300FF")}
      {card("失敗數", str(total_fail), "",
            "#00CC80" if total_fail == 0 else "#FF00FF")}
      {card("Agent 數", str(len(agents)), "")}
    </div>
    {running_html}
    <table>
      <tr><th>Agent</th><th>執行數</th><th>成功率</th><th>失敗</th><th>平均耗時</th><th>最後執行</th></tr>
      {''.join(rows)}
    </table>"""


# ── Signal Performance ───────────────────────────────────────────────

def _build_signal_perf() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            # Overall stats by strength
            for tier in ["Strong", "Moderate"]:
                cur.execute("""
                    SELECT COUNT(*) as t, SUM(filled) as f, SUM(correct) as w,
                           AVG(CASE WHEN filled=1 THEN actual_return_4h END) as avg_ret
                    FROM tracked_signals WHERE strength = %s
                """, (tier,))

            # Split by direction
            cur.execute("""
                SELECT strength, direction,
                       COUNT(*) as total,
                       SUM(filled) as filled,
                       SUM(correct) as wins,
                       AVG(CASE WHEN filled=1 THEN actual_return_4h END) as avg_ret
                FROM tracked_signals
                WHERE filled = 1
                GROUP BY strength, direction
                ORDER BY strength, direction
            """)
            dir_rows = cur.fetchall()

            # Recent signals
            cur.execute("""
                SELECT signal_time, direction, strength, confidence, entry_price,
                       actual_return_4h, correct, filled
                FROM tracked_signals
                ORDER BY signal_time DESC LIMIT 100
            """)
            recent = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:rgba(0,240,255,0.3)">{e}</div>'

    # Direction breakdown table
    dir_table_rows = []
    for r in dir_rows:
        strength = r["strength"]
        direction = r["direction"]
        filled = int(r["filled"] or 0)
        wins = int(r["wins"] or 0)
        wr = wins / filled * 100 if filled > 0 else 0
        avg_ret = float(r["avg_ret"] or 0) * 100

        dc = "#00CC80" if direction == "UP" else "#FF00FF"
        wc = "#00CC80" if wr >= 60 else "#C300FF" if wr >= 50 else "#FF00FF"

        dir_table_rows.append(
            f"<tr><td>{strength}</td>"
            f"<td style='color:{dc}'>{direction}</td>"
            f"<td>{filled}</td>"
            f"<td style='color:{wc}'>{wr:.0f}%</td>"
            f"<td>{avg_ret:+.2f}%</td></tr>"
        )

    # Recent signals table
    recent_rows = []
    for r in recent:
        t = r["signal_time"]
        if hasattr(t, "replace"):
            t = t.replace(tzinfo=timezone.utc).astimezone(TZ8).strftime("%m/%d %H:%M")
        d = r["direction"]
        dc = "#00CC80" if d == "UP" else "#FF00FF"
        icon = "&#9650;" if d == "UP" else "&#9660;"

        if r["filled"]:
            ret = float(r["actual_return_4h"]) * 100
            oc = "#00CC80" if r["correct"] else "#FF00FF"
            ok = "&#10003;" if r["correct"] else "&#10007;"
            result = f'<span style="color:{oc}">{ret:+.2f}% {ok}</span>'
        else:
            result = '<span style="color:rgba(0,240,255,0.3)">等待中</span>'

        recent_rows.append(
            f"<tr><td>{t}</td>"
            f"<td>[{r['strength'][0]}]</td>"
            f"<td style='color:{dc}'>{icon} {d}</td>"
            f"<td>{r['confidence']:.0f}</td>"
            f"<td>${r['entry_price']:,.0f}</td>"
            f"<td>{result}</td></tr>"
        )

    return f"""
    <div style="color:#00F0FF;font-size:12px;font-weight:600;margin-bottom:6px">
      方向拆分績效
    </div>
    <table>
      <tr><th>等級</th><th>方向</th><th>已結算</th><th>勝率</th><th>平均收益</th></tr>
      {''.join(dir_table_rows)}
    </table>

    <div style="color:#00F0FF;font-size:12px;font-weight:600;margin:14px 0 6px">
      最近信號 (100 筆)
    </div>
    <table>
      <tr><th>時間</th><th>等級</th><th>方向</th><th>信心</th><th>入場價</th><th>結果</th></tr>
      {''.join(recent_rows)}
    </table>"""
