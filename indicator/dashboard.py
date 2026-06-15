"""
System diagnostic dashboard — tabbed multi-page layout.

Serves a static shell with 5 tabs, each loaded via AJAX.
Tab content is rendered server-side as HTML fragments.
"""
from __future__ import annotations

import html
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
<title>FLOWBOT · BTC 預測指標控制台</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  /* ── Nansen-style theme (Flowbot) ─────────────────────────────────── */
  :root {{
    --bg:#0a0b0d; --surface:#141619; --surface2:#1a1d21;
    --border:#272a30; --border-soft:#1f232a;
    --txt:#e8eaed; --mut:#9aa0a6; --dim:#5f656d;
    --green:#36ffae; --green-dim:rgba(54,255,174,0.14);
    --red:#ff5f6d; --warn:#f5b544; --blue:#7aa2ff;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html,body {{ min-height:100%; }}
  body {{
    background:var(--bg); color:var(--txt);
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,sans-serif;
    font-size:13px; line-height:1.5; -webkit-font-smoothing:antialiased;
    font-variant-numeric:tabular-nums; min-height:100vh;
  }}

  /* ── breathing green ambient glow (~6s = calm breath) ── */
  body::before {{
    content:""; position:fixed; inset:0; z-index:-1; pointer-events:none;
    background:
      radial-gradient(1300px 920px at 8% -14%, rgba(54,255,174,0.24), transparent 70%),
      radial-gradient(1200px 1050px at 112% 4%, rgba(48,222,196,0.14), transparent 68%),
      radial-gradient(1250px 1100px at 50% 124%, rgba(66,206,178,0.13), transparent 70%),
      radial-gradient(1050px 950px at 50% 48%, rgba(54,255,174,0.08), transparent 74%);
    animation:breathe 6s ease-in-out infinite;
  }}
  @keyframes breathe {{ 0%,100% {{ opacity:0.35 }} 50% {{ opacity:0.9 }} }}
  @media (prefers-reduced-motion:reduce) {{ body::before {{ animation:none; opacity:0.5 }} }}

  /* ── Top bar + logo ── */
  .topbar {{
    background:rgba(10,11,13,0.7); -webkit-backdrop-filter:blur(8px); backdrop-filter:blur(8px);
    border-bottom:1px solid var(--border);
    padding:11px 20px; display:flex; align-items:center; justify-content:space-between;
    position:sticky; top:0; z-index:100;
  }}
  .logo {{ display:flex; align-items:center; gap:10px; }}
  .logo-mark {{ width:27px; height:27px; flex-shrink:0; filter:drop-shadow(0 0 7px rgba(54,255,174,0.4)); }}
  .logo-word {{ font-size:16.5px; font-weight:760; letter-spacing:0.6px; color:var(--txt); }}
  .logo-word em {{ font-style:normal; background:linear-gradient(92deg,#36ffae,#39d6d6);
                   -webkit-background-clip:text; background-clip:text; color:transparent; }}
  .logo-node {{ animation:breathe 6s ease-in-out infinite; transform-origin:center; }}
  .logo-tag {{ font-size:9px; font-weight:700; letter-spacing:0.8px; color:var(--green);
               background:var(--green-dim); padding:2px 6px; border-radius:5px; margin-left:2px; }}
  .topbar .meta {{ color:var(--mut); font-size:11px; }}

  /* ── Tab bar ── */
  .tab-bar {{
    background:rgba(10,11,13,0.7); -webkit-backdrop-filter:blur(8px); backdrop-filter:blur(8px);
    border-bottom:1px solid var(--border);
    display:flex; gap:2px; overflow-x:auto; padding:0 14px;
    position:sticky; top:47px; z-index:99;
  }}
  .tab {{
    background:transparent; border:none; border-bottom:2px solid transparent;
    color:var(--mut); padding:11px 14px; cursor:pointer; font-size:12.5px;
    font-weight:500; white-space:nowrap; transition:all 0.15s;
  }}
  .tab:hover {{ color:var(--txt); }}
  .tab.active {{ color:var(--txt); border-bottom-color:var(--green); font-weight:600; }}
  .tab .tab-icon {{ margin-right:5px; }}

  /* ── Content area ── */
  .content {{ max-width:1200px; margin:0 auto; padding:18px 20px; min-height:70vh; }}

  /* ── Loading ── */
  .loading {{ display:flex; align-items:center; justify-content:center;
              min-height:200px; color:var(--mut); }}
  .spinner {{ width:22px; height:22px; border:2px solid rgba(255,255,255,0.1);
              border-top-color:var(--green); border-radius:50%;
              animation:spin 0.8s linear infinite; margin-right:10px; }}
  @keyframes spin {{ to {{ transform:rotate(360deg) }} }}
  @keyframes blink {{ 50% {{ opacity:0.3 }} }}

  /* ── Cards ── */
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-bottom:16px; }}
  .grid-3 {{ grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); }}
  .grid-4 {{ grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); }}
  .card {{ background:var(--surface); border:1px solid var(--border); border-radius:10px;
           padding:13px 14px; transition:border-color 0.15s, background 0.15s; }}
  .card:hover {{ border-color:#33373e; background:var(--surface2); }}
  .card-title {{ color:var(--mut); font-size:10.5px; font-weight:500; letter-spacing:0.2px; text-transform:uppercase; }}
  .card-value {{ font-size:21px; font-weight:680; margin:4px 0 2px; color:var(--txt); letter-spacing:-0.3px; }}
  .card-sub {{ color:var(--dim); font-size:10.5px; }}

  /* ── Sections ── */
  .section {{ background:var(--surface); border:1px solid var(--border); border-radius:10px;
              margin-bottom:12px; overflow:hidden; }}
  .section-header {{ padding:12px 15px; cursor:pointer; display:flex;
                     justify-content:space-between; align-items:center; user-select:none;
                     border-bottom:1px solid var(--border-soft); }}
  .section-header:hover {{ background:var(--surface2); }}
  .section-title {{ color:var(--txt); font-size:13px; font-weight:600; }}
  .section-toggle {{ color:var(--dim); font-size:13px; }}
  .section-body {{ padding:13px 15px; }}

  /* ── Tables ── */
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th {{ text-align:left; color:var(--mut); padding:7px 8px; border-bottom:1px solid var(--border);
        font-size:10.5px; font-weight:600; text-transform:uppercase; letter-spacing:0.4px; }}
  td {{ padding:8px; border-bottom:1px solid var(--border-soft); }}
  tbody tr {{ transition:background 0.12s; }}
  tbody tr:hover {{ background:var(--surface2); }}
  tr:last-child td {{ border-bottom:none; }}

  /* ── Misc ── */
  .dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px; vertical-align:middle; }}
  .dot-ok {{ background:var(--green); }}
  .dot-err {{ background:var(--red); }}
  .badge {{ display:inline-block; padding:3px 9px; border-radius:6px; font-size:10.5px; font-weight:600; color:#fff; }}
  .dist-bar {{ display:flex; height:24px; border-radius:6px; overflow:hidden; margin:6px 0; }}
  .dist-bar div {{ display:flex; align-items:center; justify-content:center; font-size:10px; font-weight:600; color:#fff; }}
  .regime-row {{ display:flex; gap:3px; height:22px; border-radius:6px; overflow:hidden; margin:6px 0; }}
  .regime-block {{ display:flex; align-items:center; justify-content:center;
                   font-size:9px; font-weight:600; color:#fff; border-radius:5px; min-width:30px; }}
  .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
  code {{ background:#23262c; padding:1px 6px; border-radius:5px; font-size:11px; color:var(--green); }}

  /* ── Heatmap ── */
  .heatmap-grid {{ display:grid; grid-template-columns:repeat(24,1fr); gap:3px; }}
  .heatmap-labels {{ display:grid; grid-template-columns:repeat(24,1fr); gap:3px;
                     color:var(--mut); font-size:9px; text-align:center; margin-top:3px; }}
  .hm-cell {{ aspect-ratio:1; display:flex; align-items:center; justify-content:center;
              border-radius:5px; font-size:9px; font-weight:600; color:#fff; }}

  /* ── Latency bars ── */
  .latency-bar {{ display:flex; align-items:center; margin:5px 0; }}
  .latency-label {{ width:80px; font-size:11px; color:var(--mut); }}
  .latency-track {{ flex:1; height:8px; background:#23262c; border-radius:5px; overflow:hidden; margin:0 8px; }}
  .latency-fill {{ height:100%; border-radius:5px; transition:width 0.3s; }}
  .latency-val {{ width:50px; text-align:right; font-size:11px; font-family:monospace; }}

  /* ── Gauge bars ── */
  .gauge-wrap {{ display:flex; align-items:center; margin:4px 0; }}
  .gauge-label {{ width:100px; font-size:11px; color:var(--mut); }}
  .gauge-track {{ flex:1; height:6px; background:#23262c; border-radius:5px; overflow:hidden; margin:0 8px; }}
  .gauge-fill {{ height:100%; border-radius:5px; }}
  .gauge-val {{ width:40px; text-align:right; font-size:11px; }}

  /* ── Footer ── */
  .footer {{ text-align:center; color:var(--dim); font-size:10px; padding:22px 0; }}

  /* ── Responsive ── */
  @media (max-width:640px) {{
    .tab {{ padding:9px 11px; font-size:11.5px; }}
    .tab-bar {{ top:46px; }}
    .content {{ padding:12px 12px; }}
    .two-col {{ grid-template-columns:1fr; }}
    .card-value {{ font-size:19px; }}
    .topbar {{ padding:9px 13px; }}
    .logo-word {{ font-size:15px; }}
    .logo-tag {{ display:none; }}
  }}
</style>
</head>
<body>

<div class="topbar">
  <div class="logo">
    <svg class="logo-mark" viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <defs><linearGradient id="lgm" x1="3" y1="29" x2="29" y2="3" gradientUnits="userSpaceOnUse">
        <stop stop-color="#36ffae"/><stop offset="1" stop-color="#39d6d6"/></linearGradient></defs>
      <rect x="2.5" y="2.5" width="27" height="27" rx="8" fill="rgba(54,255,174,0.07)" stroke="url(#lgm)" stroke-width="1.4"/>
      <path d="M6 20 L10 20 L13 12.5 L16 22 L19 9.5 L24 9.5" stroke="url(#lgm)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      <circle class="logo-node" cx="24" cy="9.5" r="2.4" fill="#36ffae"/>
    </svg>
    <span class="logo-word">FLOW<em>BOT</em></span>
    <span class="logo-tag">DUAL v7</span>
  </div>
  <div class="meta">
    <span id="update-time">{now}</span>
    <span id="refresh-indicator" style="margin-left:8px;color:#36ffae;font-size:9px">&#9679;</span>
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
  <button class="tab" data-tab="stability">
    <span class="tab-icon">&#128737;</span>穩定性
  </button>
</div>

<div class="content" id="tab-content">
  <div class="loading"><div class="spinner"></div>載入中...</div>
</div>

<div class="footer">FLOWBOT &middot; BTC Market Intelligence &middot; Dual v7</div>

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
        content.innerHTML = '<div class="loading" style="color:#ff5f6d">載入失敗: ' + err + '</div>';
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
        elif tab_name == "stability":
            from indicator.dashboard_tabs.stability import render_stability
            return render_stability()
        else:
            return f'<div style="color:#ff5f6d">未知的 Tab: {html.escape(str(tab_name))}</div>'
    except Exception as e:
        logger.exception("Dashboard tab %s render failed", tab_name)
        return (
            f'<div style="color:#ff5f6d;padding:20px">'
            f'<b>Tab "{html.escape(str(tab_name))}" 渲染失敗</b><br>'
            f'<code>{type(e).__name__}: {e}</code></div>'
        )


# ── Legacy compatibility ─────────────────────────────────────────────
# Keep render_dashboard() so existing code doesn't break.
# It now redirects to the new shell.

def render_dashboard(state: dict, engine) -> str:
    """Legacy entry point — returns the new tabbed shell."""
    return render_dashboard_shell()
