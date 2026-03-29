"""
自動更新圖表 server。
每小時重新產生 HTML，瀏覽器自動 reload。

Usage:
    python watch_chart.py                    # BTC-USD 1h 7 days, port 8050
    python watch_chart.py ETH-USD 4h 14     # custom
    python watch_chart.py BTC-USD 1h 7 8051 # custom port
"""
import sys
import os
import time
import threading
import http.server
import socketserver

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research.storage.schema import ensure_schema
from research.bar_generator.runner import run_once
from research.viz.chart_builder import load_and_build
from research.config.settings import ChartConfig

# ── Args ─────────────────────────────────────────────────────────────────────
symbol        = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
timeframe     = sys.argv[2] if len(sys.argv) > 2 else "1h"
lookback_days = int(sys.argv[3]) if len(sys.argv) > 3 else 7
PORT          = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ.get("PORT", 8050))

REFRESH_SEC   = 3600  # 1 hour
HTML_FILENAME = f"chart_{symbol.replace('-','_')}_{timeframe}.html"
HTML_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), HTML_FILENAME)

# ── Chart generator ───────────────────────────────────────────────────────────
def generate_chart():
    print(f"[chart] Computing bars: {symbol} {timeframe} last {lookback_days}d ...")
    try:
        n = run_once(symbol, timeframe, lookback_days=lookback_days)
        print(f"[chart] → {n} new bars computed")

        config = ChartConfig(symbol=symbol, timeframe=timeframe, lookback_days=lookback_days)
        fig = load_and_build(symbol, timeframe, lookback_days=lookback_days, config=config)

        html = fig.to_html(full_html=True, include_plotlyjs="cdn")

        # Inject auto-refresh meta tag
        html = html.replace(
            "<head>",
            f'<head>\n  <meta http-equiv="refresh" content="{REFRESH_SEC}">',
            1,
        )

        with open(HTML_PATH, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"[chart] Saved: {HTML_FILENAME}")
    except Exception as e:
        print(f"[chart] ERROR: {e}")


def chart_loop():
    while True:
        generate_chart()
        print(f"[chart] Next update in {REFRESH_SEC // 60} minutes...")
        time.sleep(REFRESH_SEC)


# ── HTTP server ───────────────────────────────────────────────────────────────
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(HTML_PATH), **kwargs)

    def do_GET(self):
        # Redirect root to chart
        if self.path == "/" or self.path == "":
            self.send_response(302)
            self.send_header("Location", f"/{HTML_FILENAME}")
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, format, *args):
        pass  # silence request logs


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[init] Ensuring schema...")
    ensure_schema()

    print(f"[init] Generating initial chart...")
    generate_chart()

    # Start background update thread
    t = threading.Thread(target=chart_loop, daemon=True)
    t.start()

    # Start HTTP server
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}"
        print(f"\n✓ Chart server running: {url}")
        print(f"  Symbol:   {symbol} {timeframe} ({lookback_days}d)")
        print(f"  Refresh:  every {REFRESH_SEC // 60} minutes")
        print(f"  Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[server] Stopped.")
