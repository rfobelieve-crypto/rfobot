"""Push the cancellation-flow monitor chart to Telegram — standalone.

Renders research/plot_cancel_flow.py then sends the PNG via sendPhoto to
the operator's own chat. STANDALONE by design: does NOT import the main
bot, the executor, or any hot-path module; it only reads the DB (via the
render script) and calls the Telegram Bot API directly. Safe to schedule
(mirrors daily_collect.bat). This is a self-notification research aid, not
a trading signal.

Creds: TELEGRAM_BOT_TOKEN + (TG_CRITICAL_CHAT_ID | TELEGRAM_CHAT_ID),
read from env, falling back to parsing .env — same convention as the
rest of the repo.

Usage:
    python research/push_cancel_flow.py            # last 24h (plot default)
    python research/push_cancel_flow.py --hours 168 # weekly baseline view
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PNG = PROJECT_ROOT / "research" / "results" / "cancel_flow_monitor.png"

CAPTION = (
    "📊 <b>撤單流監控</b>（研究圖·非信號）\n"
    "上=價格 | 中=撤單不對稱(綠賣側撤/紅買側撤) | 下=撤單強度\n\n"
    "<b>獵取判斷</b>：強度尖峰 + 不對稱明確一側 + 該價位成交量\n"
    "· 撤+量低 → 純真空(假跌破,反轉)\n"
    "· 撤+量大破位 → 真實成交(續走,別接)\n"
    "· 量大但守住 → 吸收(有人接,更強反轉)\n"
    "edge 待 8/10 判決"
)


def _load_env_val(*keys: str) -> str:
    for k in keys:
        v = os.environ.get(k, "").strip()
        if v:
            return v
    envf = PROJECT_ROOT / ".env"
    if envf.exists():
        want = set(keys)
        for line in envf.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            k, _, v = line.partition("=")
            if k.strip() in want and v.strip():
                return v.strip().strip('"').strip("'")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=0)  # 0 = plot script default (24h)
    args = ap.parse_args()

    # 1) render (subprocess = full isolation from this process)
    cmd = [sys.executable, str(PROJECT_ROOT / "research" / "plot_cancel_flow.py")]
    if args.hours:
        cmd += ["--hours", str(args.hours)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0 or not PNG.exists():
        print("render failed:\n", r.stdout, r.stderr)
        return 1

    # 2) send
    token = _load_env_val("TELEGRAM_BOT_TOKEN")
    chat = _load_env_val("TG_CRITICAL_CHAT_ID", "TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("missing TELEGRAM_BOT_TOKEN or chat id — cannot push")
        return 1
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(PNG, "rb") as f:
        resp = requests.post(url, data={"chat_id": chat, "caption": CAPTION,
                                        "parse_mode": "HTML"},
                             files={"photo": f}, timeout=30)
    if resp.status_code == 200 and resp.json().get("ok"):
        print(f"pushed cancel_flow_monitor.png → chat {chat[:4]}…")
        return 0
    print("telegram push failed:", resp.status_code, resp.text[:300])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
