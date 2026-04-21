"""
System diagnostic dashboard — tabbed multi-page layout.

Serves a static shell with 5 tabs, each loaded via AJAX.
Tab content is rendered server-side as HTML fragments.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

TZ8 = timezone(timedelta(hours=8))


def render_dashboard_shell() -> str:
    """Return the outer shell HTML (tabs + JS router). Content loads via AJAX."""
    now = datetime.now(TZ8).strftime("%Y-%m-%d %H:%M UTC+8")

    return f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC 預測指標控制台</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0A0A0A; color:rgba(0,240,255,0.85); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px; }}

  /* ── Top bar ── */
  .topbar {{
    background:#0D0D0D; border-bottom:1px solid #1A1A2E;
    padding:10px 20px; display:flex; align-items:center; justify-content:space-between;
    position:sticky; top:0; z-index:100;
  }}
  .topbar h1 {{ color:#00F0FF; font-size:16px; font-weight:700; text-shadow:0 0 10px rgba(0,240,255,0.5),0 0 20px rgba(0,240,255,0.2); }}
  .topbar .meta {{ color:rgba(0,240,255,0.5); font-size:11px; }}

  /* ── Tab bar ── */
  .tab-bar {{
    background:#0D0D0D; border-bottom:1px solid #1A1A2E;
    display:flex; gap:0; overflow-x:auto; padding:0 16px;
    position:sticky; top:41px; z-index:99;
  }}
  .tab {{
    background:transparent; border:none; border-bottom:2px solid transparent;
    color:rgba(0,240,255,0.4); padding:10px 18px; cursor:pointer; font-size:12px;
    font-weight:500; white-space:nowrap; transition:all 0.15s;
  }}
  .tab:hover {{ color:rgba(0,240,255,0.7); background:rgba(0,240,255,0.04); }}
  .tab.active {{
    color:#00F0FF; border-bottom-color:#00F0FF; font-weight:600;
    text-shadow:0 0 10px rgba(0,240,255,0.5);
  }}
  .tab .tab-icon {{ margin-right:5px; }}

  /* ── Content area ── */
  .content {{ max-width:1200px; margin:0 auto; padding:16px 20px; min-height:70vh; }}

  /* ── Loading ── */
  .loading {{
    display:flex; align-items:center; justify-content:center;
    min-height:200px; color:rgba(0,240,255,0.5);
  }}
  .spinner {{
    width:20px; height:20px; border:2px solid #1A1A2E; border-top-color:#00F0FF;
    border-radius:50%; animation:spin 0.8s linear infinite; margin-right:10px;
  }}
  @keyframes spin {{ to {{ transform:rotate(360deg) }} }}
  @keyframes blink {{ 50% {{ opacity:0.3 }} }}

  /* ── Cards ── */
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:8px; margin-bottom:14px; }}
  .grid-3 {{ grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); }}
  .grid-4 {{ grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); }}
  .card {{ background:#0D0D0D; border:1px solid #1A1A2E; border-radius:8px; padding:10px; box-shadow:0 0 5px rgba(0,240,255,0.1),0 0 15px rgba(0,240,255,0.05),0 0 25px rgba(0,240,255,0.02); }}
  .card-title {{ color:rgba(0,240,255,0.5); font-size:10px; text-transform:uppercase; letter-spacing:0.5px; }}
  .card-value {{ font-size:20px; font-weight:700; margin:2px 0; }}
  .card-sub {{ color:rgba(0,240,255,0.4); font-size:10px; }}

  /* ── Sections ── */
  .section {{ background:#0D0D0D; border:1px solid #1A1A2E; border-radius:8px; margin-bottom:12px; overflow:hidden; box-shadow:0 0 5px rgba(0,240,255,0.08),0 0 15px rgba(0,240,255,0.03); }}
  .section-header {{
    padding:10px 14px; cursor:pointer; display:flex;
    justify-content:space-between; align-items:center;
    user-select:none;
  }}
  .section-header:hover {{ background:#111122; }}
  .section-title {{ color:#00F0FF; font-size:13px; font-weight:600; }}
  .section-toggle {{ color:rgba(0,240,255,0.4); font-size:14px; }}
  .section-body {{ padding:0 14px 12px; }}

  /* ── Tables ── */
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th {{ text-align:left; color:rgba(0,240,255,0.5); padding:5px 6px; border-bottom:1px solid #1A1A2E;
        font-size:10px; text-transform:uppercase; }}
  td {{ padding:5px 6px; border-bottom:1px solid rgba(26,26,46,0.5); }}

  /* ── Misc ── */
  .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:4px; vertical-align:middle; }}
  .dot-ok {{ background:#00FF9F; }}
  .dot-err {{ background:#FF00FF; }}
  .badge {{
    display:inline-block; padding:2px 8px; border-radius:10px;
    font-size:10px; font-weight:600; color:#fff;
  }}
  .dist-bar {{ display:flex; height:24px; border-radius:4px; overflow:hidden; margin:6px 0; }}
  .dist-bar div {{ display:flex; align-items:center; justify-content:center;
                   font-size:10px; font-weight:600; color:#fff; }}
  .regime-row {{ display:flex; gap:2px; height:22px; border-radius:4px; overflow:hidden; margin:6px 0; }}
  .regime-block {{
    display:flex; align-items:center; justify-content:center;
    font-size:9px; font-weight:600; color:#fff; border-radius:3px; min-width:30px;
  }}
  .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  code {{ background:#1A1A2E; padding:1px 4px; border-radius:3px; font-size:11px; }}

  /* ── Heatmap ── */
  .heatmap-grid {{
    display:grid; grid-template-columns:repeat(24,1fr); gap:2px;
  }}
  .heatmap-labels {{
    display:grid; grid-template-columns:repeat(24,1fr); gap:2px;
    color:rgba(0,240,255,0.5); font-size:9px; text-align:center; margin-top:2px;
  }}
  .hm-cell {{
    aspect-ratio:1; display:flex; align-items:center; justify-content:center;
    border-radius:3px; font-size:9px; font-weight:600; color:#fff;
  }}

  /* ── Latency bars ── */
  .latency-bar {{ display:flex; align-items:center; margin:4px 0; }}
  .latency-label {{ width:80px; font-size:11px; color:rgba(0,240,255,0.5); }}
  .latency-track {{ flex:1; height:10px; background:#1A1A2E; border-radius:4px; overflow:hidden; margin:0 8px; }}
  .latency-fill {{ height:100%; border-radius:4px; transition:width 0.3s; }}
  .latency-val {{ width:50px; text-align:right; font-size:11px; font-family:monospace; }}

  /* ── Gauge bars ── */
  .gauge-wrap {{ display:flex; align-items:center; margin:3px 0; }}
  .gauge-label {{ width:100px; font-size:11px; color:rgba(0,240,255,0.5); }}
  .gauge-track {{ flex:1; height:8px; background:#1A1A2E; border-radius:4px; overflow:hidden; margin:0 8px; }}
  .gauge-fill {{ height:100%; border-radius:4px; }}
  .gauge-val {{ width:40px; text-align:right; font-size:11px; }}

  /* ── Footer ── */
  .footer {{ text-align:center; color:rgba(0,240,255,0.2); font-size:10px; padding:20px 0; }}

  /* ── Responsive ── */
  @media (max-width:640px) {{
    .tab {{ padding:8px 12px; font-size:11px; }}
    .content {{ padding:10px 12px; }}
    .two-col {{ grid-template-columns:1fr; }}
    .card-value {{ font-size:18px; }}
    .topbar {{ padding:8px 12px; }}
    .topbar h1 {{ font-size:14px; }}
  }}
</style>
</head>
<body>

<div class="topbar">
  <h1>BTC 預測指標控制台</h1>
  <div class="meta">
    <span id="update-time">{now}</span>
    <span id="refresh-indicator" style="margin-left:8px;color:#00FF9F;font-size:9px">&#9679;</span>
  </div>
</div>

<div class="tab-bar">
  <button class="tab active" data-tab="overview">
    <span class="tab-icon">&#128200;</span>總覽
  </button>
  <button class="tab" data-tab="performance">
    <span class="tab-icon">&#128202;</span>模型績效
  </button>
  <button class="tab" data-tab="market">
    <span class="tab-icon">&#128176;</span>市場情報
  </button>
  <button class="tab" data-tab="health">
    <span class="tab-icon">&#9881;</span>系統健康
  </button>
  <button class="tab" data-tab="analytics">
    <span class="tab-icon">&#128202;</span>分析圖表
  </button>
  <button class="tab" data-tab="agents">
    <span class="tab-icon">&#129302;</span>Agent
  </button>
</div>

<div class="content" id="tab-content">
  <div class="loading"><div class="spinner"></div>載入中...</div>
</div>

<div class="footer">BTC Market Intelligence Indicator &middot; Dual v7 &middot; Cyberpunk Edition</div>

<script>
(function() {{
  var currentTab = 'overview';
  var refreshTimer = null;
  var tabs = document.querySelectorAll('.tab');

  tabs.forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      loadTab(btn.dataset.tab);
    }});
  }});

  function loadTab(name) {{
    currentTab = name;
    tabs.forEach(function(b) {{
      b.classList.toggle('active', b.dataset.tab === name);
    }});

    var content = document.getElementById('tab-content');
    content.innerHTML = '<div class="loading"><div class="spinner"></div>載入中...</div>';

    fetch('/dashboard/tab/' + name)
      .then(function(r) {{ return r.text(); }})
      .then(function(html) {{
        content.innerHTML = html;
        // Re-execute inline scripts (Chart.js etc.)
        content.querySelectorAll('script').forEach(function(old) {{
          var ns = document.createElement('script');
          ns.textContent = old.textContent;
          old.parentNode.replaceChild(ns, old);
        }});
        // Update timestamp
        document.getElementById('update-time').textContent =
          new Date().toLocaleString('zh-TW', {{timeZone:'Asia/Taipei'}});
      }})
      .catch(function(err) {{
        content.innerHTML = '<div class="loading" style="color:#FF00FF">載入失敗: ' + err + '</div>';
      }});

    // Reset refresh timer
    if (refreshTimer) clearInterval(refreshTimer);
    refreshTimer = setInterval(function() {{ loadTab(currentTab); }}, 300000);
  }}

  // Section toggle (used by tab content)
  window.toggle = function(id) {{
    var el = document.getElementById(id);
    var arrow = document.getElementById(id + '_arrow');
    if (!el) return;
    if (el.style.display === 'none') {{
      el.style.display = 'block';
      if (arrow) arrow.innerHTML = '&#9660;';
    }} else {{
      el.style.display = 'none';
      if (arrow) arrow.innerHTML = '&#9654;';
    }}
  }};

  // Initial load
  loadTab('overview');

  // Keyboard shortcuts: 1-5 for tabs
  document.addEventListener('keydown', function(e) {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    var tabMap = {{'1':'overview','2':'performance','3':'market','4':'health','5':'analytics','6':'agents'}};
    if (tabMap[e.key]) loadTab(tabMap[e.key]);
  }});
}})();
</script>
</body></html>"""


def render_tab(tab_name: str, state: dict, engine) -> str:
    """Render a single tab's HTML fragment."""
    try:
        if tab_name == "overview":
            from indicator.dashboard_tabs.overview import render_overview
            return render_overview(state, engine)
        elif tab_name == "performance":
            from indicator.dashboard_tabs.performance import render_performance
            return render_performance()
        elif tab_name == "market":
            from indicator.dashboard_tabs.market import render_market
            return render_market(state, engine)
        elif tab_name == "health":
            from indicator.dashboard_tabs.health import render_health
            return render_health(state, engine)
        elif tab_name == "analytics":
            from indicator.dashboard_tabs.analytics import render_analytics
            return render_analytics(state, engine)
        elif tab_name == "agents":
            from indicator.dashboard_tabs.agents import render_agents
            return render_agents()
        else:
            return f'<div style="color:#FF00FF">未知的 Tab: {tab_name}</div>'
    except Exception as e:
        logger.exception("Dashboard tab %s render failed", tab_name)
        return (
            f'<div style="color:#FF00FF;padding:20px">'
            f'<b>Tab "{tab_name}" 渲染失敗</b><br>'
            f'<code>{type(e).__name__}: {e}</code></div>'
        )


# ── Legacy compatibility ─────────────────────────────────────────────
# Keep render_dashboard() so existing code doesn't break.
# It now redirects to the new shell.

def render_dashboard(state: dict, engine) -> str:
    """Legacy entry point — returns the new tabbed shell."""
    return render_dashboard_shell()
