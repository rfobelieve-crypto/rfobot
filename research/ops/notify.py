# -*- coding: utf-8 -*-
"""告警投遞層 —— Discord 主、Telegram 備，而且**送不出去會變成一盞紅燈**。

===========================================================================
為什麼存在：告警靜默死了 8 天
===========================================================================
2026-09-13 查出來的（使用者：「主要是我怕資料收集有斷掉但我不知道」）：

    FreshnessBoard 排程       正常，每 6 小時，LastTaskResult=0
    最後一次成功投遞           **2026-09-05 05:20**
    之後                      **11 次狀態轉換，全部沒送達**

而根因不是 Telegram 掛了：

    .env          有 TELEGRAM_BOT_TOKEN / TG_CRITICAL_CHAT_ID
    os.environ    三個全都沒有   <- freshness_board.bat 沒有載 .env
    舊程式         只讀 os.environ -> chat="" -> sent=False -> 印一行 WARN

**它連嘗試都沒嘗試過。** 唯一的痕跡是一個沒人讀的 log 裡的 `[WARN]`。
這逐字是 mistake.md 2026-07-05 那條：「任何『向人回報』的排程，送出層必須有
重試 ＋ 最終失敗要在某個人會看到的地方留痕」——重試有做，**留痕的地方選錯了**。

===========================================================================
三個設計決定
===========================================================================
1. **設定一律 env -> .env 回退。** 抄 `shared/db.py` 的 `_load_dotenv()` 慣例
   （那支早就這樣做了，而告警這支沒有 —— 同一個 repo 兩種做法）。
2. **「沒有設定任何管道」不是合法狀態。** 舊版在未設定時安靜跳過；這裡回
   `configured=False` 並讓旗標變 not-ok，所以它會出現在新鮮度看板上。
3. **投遞結果寫成旗標 `alert_last.json`，並註冊進看板本身。**
   看板得監測自己的嘴 —— 否則「告警送不出去」永遠只有告警自己知道。

Discord webhook 是**頻道層的憑證**（拿到的人可以往那個頻道貼文），所以它
只放 `.env`（已被 .gitignore 擋），不進對話、不進 argv、不進 log。
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FLAG = os.path.join(ROOT, "research", "results", "alert_last.json")

DISCORD_LIMIT = 1900            # 實際上限 2000，留邊給前後綴
RETRIES = 3


def _dotenv(path=None):
    """KEY=VALUE 解析，不需要 python-dotenv（同 shared/db.py）。"""
    p = path or os.path.join(ROOT, ".env")
    out = {}
    if not os.path.exists(p):
        return out
    try:
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith("#") or "=" not in ln:
                    continue
                k, v = ln.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


_ENV_CACHE = None


def cfg(key: str, default: str = "") -> str:
    """env 優先，.env 回退。**這一行就是 8 天沒告警的修法。**"""
    global _ENV_CACHE
    v = os.environ.get(key)
    if v:
        return v
    if _ENV_CACHE is None:
        _ENV_CACHE = _dotenv()
    return _ENV_CACHE.get(key, default)


def _post_discord(url: str, text: str) -> tuple[bool, str]:
    body = json.dumps({"content": text[:DISCORD_LIMIT],
                       "allowed_mentions": {"parse": []}}).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": "flowbot-freshness/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            # webhook 成功是 204 No Content
            return (200 <= r.status < 300), "HTTP %d" % r.status
    except urllib.error.HTTPError as e:
        return False, "HTTP %d %s" % (e.code, (e.reason or "")[:60])
    except Exception as e:                              # noqa: BLE001
        return False, type(e).__name__ + ": " + str(e)[:80]


def _post_telegram(token: str, chat: str, text: str) -> tuple[bool, str]:
    body = json.dumps({"chat_id": chat, "text": text[:4000],
                       "disable_web_page_preview": True}).encode()
    req = urllib.request.Request(
        "https://api.telegram.org/bot%s/sendMessage" % token,
        data=body, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return (200 <= r.status < 300), "HTTP %d" % r.status
    except urllib.error.HTTPError as e:
        return False, "HTTP %d" % e.code
    except Exception as e:                              # noqa: BLE001
        return False, type(e).__name__ + ": " + str(e)[:80]


def channels() -> list[str]:
    """有設定的管道。空 list 代表**沒有任何人收得到告警**。"""
    out = []
    if cfg("DISCORD_WEBHOOK_URL"):
        out.append("discord")
    if cfg("TELEGRAM_BOT_TOKEN") and (cfg("TG_ALERT_CHAT_ID")
                                      or cfg("TG_CRITICAL_CHAT_ID")
                                      or cfg("TELEGRAM_CHAT_ID")):
        out.append("telegram")
    return out


def send(text: str, source: str = "freshness", write_flag: bool = True) -> dict:
    """送到所有有設定的管道。**任一成功即算送達**，但每個管道的結果都記。"""
    res = {"asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source": source, "tried": {}, "delivered": False,
           "configured": bool(channels())}

    url = cfg("DISCORD_WEBHOOK_URL")
    if url:
        for i in range(RETRIES):
            ok, why = _post_discord(url, text)
            res["tried"]["discord"] = why
            if ok:
                res["delivered"] = True
                break
            if i < RETRIES - 1:
                time.sleep(3 * (i + 1))

    tok = cfg("TELEGRAM_BOT_TOKEN")
    chat = (cfg("TG_ALERT_CHAT_ID") or cfg("TG_CRITICAL_CHAT_ID")
            or cfg("TELEGRAM_CHAT_ID"))
    if tok and chat and not res["delivered"]:
        for i in range(RETRIES):
            ok, why = _post_telegram(tok, chat, text)
            res["tried"]["telegram"] = why
            if ok:
                res["delivered"] = True
                break
            if i < RETRIES - 1:
                time.sleep(3 * (i + 1))

    if not res["configured"]:
        res["tried"]["none"] = ("沒有設定任何管道："
                                "DISCORD_WEBHOOK_URL 或 TELEGRAM_BOT_TOKEN"
                                "＋chat id（env 或 .env 都可）")

    if write_flag:
        # **ok 的語意是「這個管道送得出去」。** 沒設定 = not ok，因為
        # 「沒人收得到告警」不是合法狀態（mistake.md 2026-07-05）。
        payload = dict(res)
        payload["ok"] = bool(res["delivered"] or
                             (res["configured"] and not res["tried"]))
        payload["reason"] = (
            "已送達：" + ",".join(k for k in res["tried"]
                                  if k != "none")
            if res["delivered"] else
            ("**沒有設定任何告警管道**" if not res["configured"]
             else "**投遞失敗**：" + "；".join(
                 "%s=%s" % (k, v) for k, v in res["tried"].items())))
        try:
            with open(FLAG, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass
    return res


def send_image(path: str, caption: str = "", source: str = "station",
               write_flag: bool = True) -> dict:
    """把 PNG 推到 Discord（webhook multipart）／Telegram（sendPhoto）。

    使用者 2026-09-13：「回報用圖表的方式」。那個頻道本來就在收 V7 的 PNG
    （`indicator/app.py:_send_discord_photo`），所以這條路早就證明通了。
    用 `requests` 而不是 urllib：multipart 手刻容易出錯，而 requests 本來
    就是這個專案的相依（.claude/rules/coding.md）。
    """
    import requests
    res = {"asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source": source, "tried": {}, "delivered": False,
           "configured": bool(channels())}
    try:
        with open(path, "rb") as fh:
            png = fh.read()
    except OSError as e:
        res["tried"]["file"] = str(e)[:80]
        png = b""

    url = cfg("DISCORD_WEBHOOK_URL")
    if url and png:
        for i in range(RETRIES):
            try:
                r = requests.post(url, data={"content": caption[:DISCORD_LIMIT]},
                                  files={"file": ("accum.png", png, "image/png")},
                                  timeout=40)
                res["tried"]["discord"] = "HTTP %d" % r.status_code
                if r.ok:
                    res["delivered"] = True
                    break
            except Exception as e:                      # noqa: BLE001
                res["tried"]["discord"] = type(e).__name__ + ": " + str(e)[:60]
            if i < RETRIES - 1:
                time.sleep(3 * (i + 1))

    tok, chat = cfg("TELEGRAM_BOT_TOKEN"), (cfg("TG_ALERT_CHAT_ID")
                                            or cfg("TG_CRITICAL_CHAT_ID")
                                            or cfg("TELEGRAM_CHAT_ID"))
    if tok and chat and png and not res["delivered"]:
        try:
            r = requests.post(
                "https://api.telegram.org/bot%s/sendPhoto" % tok,
                data={"chat_id": chat, "caption": caption[:1000]},
                files={"photo": ("accum.png", png, "image/png")}, timeout=40)
            res["tried"]["telegram"] = "HTTP %d" % r.status_code
            res["delivered"] = bool(r.ok)
        except Exception as e:                          # noqa: BLE001
            res["tried"]["telegram"] = type(e).__name__ + ": " + str(e)[:60]

    if write_flag:
        payload = dict(res)
        payload["ok"] = bool(res["delivered"])
        payload["reason"] = ("已送達（圖）：" + ",".join(res["tried"])
                             if res["delivered"]
                             else "**圖片投遞失敗**：" + "；".join(
                                 "%s=%s" % (k, v) for k, v in res["tried"].items()))
        try:
            with open(FLAG, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass
    return res


def station_text() -> str:
    """**資料監控站**的那一則 —— 這是 Discord 頻道從「V7 圖表」換過來的內容。

    使用者 2026-09-13：「把 V7 在 discord 的圖表每小時推送取消，那邊改為
    數據監控站」。所以這一則要回答的是「資料有沒有在累積、哪裡斷了、
    容量還夠不夠」，而不是 V7 的方向。

    來源是兩份已經在產的檔案，**不重新計算任何東西**
    （第二份實作會安靜地跟第一份不一致 —— mistake.md 2026-08-26）。
    """
    import json as _j
    lines = []
    acc = os.path.join(ROOT, "research", "results", "accum_snapshot.json")
    try:
        with open(acc, encoding="utf-8") as fh:
            s = _j.load(fh)
        lines.append("**資料累積**（%s UTC）" % s["asof_utc"][:16].replace("T", " "))
        for x in s.get("sets", []):
            if x.get("error"):
                lines.append("  x %-18s 錯誤：%s" % (x["name"], x["error"][:40]))
                continue
            miss, emp = x.get("hours_missing", 0), x.get("hours_empty", 0)
            mark = "!" if miss else ("." if emp else "o")
            lines.append("  %s %-18s %5dh 上線｜缺 %d｜近24h %s 列"
                         % (mark, x["name"], x.get("hours_live", 0), miss,
                            format(x.get("rows_24h", 0), ",")))
        c = s.get("capacity", {})
        d, my = c.get("disk_D", {}), c.get("mysql", {})
        # MySQL 標「表資料」不是「佔用」：information_schema 算的是表，而
        # Railway volume 實際佔用含 redo/undo/binlog 約 2.3 倍（09-13 實測
        # 3.68 GB 表 vs 8.49 GB volume）。對 250 GB 上限有意義的是後者。
        lines.append("  容量：D 槽 %.0f GB 可用｜MySQL 表資料 %.1f GB"
                     "（volume 上限 250 GB，實佔約 2.3x）"
                     % (d.get("free_gb", 0), my.get("used_mb", 0) / 1024))
    except Exception as e:                              # noqa: BLE001
        lines.append("**讀不到累積快照**：%s" % e)

    fb = os.path.join(ROOT, "research", "results", "freshness_board.json")
    try:
        with open(fb, encoding="utf-8") as fh:
            b = _j.load(fh)
        reds = b.get("reds") or []
        lines.append("")
        lines.append("**新鮮度** %d red / %d tracked（%s UTC）"
                     % (len(reds), len(b.get("rows") or []), b.get("asof_utc", "")))
        for n in reds[:8]:
            lines.append("  ! " + n)
    except Exception as e:                              # noqa: BLE001
        lines.append("**讀不到新鮮度看板**：%s" % e)
    return "\n".join(lines)


def heartbeat(text: str) -> dict:
    """一切正常時也要送一次 —— 否則**沉默分不出「沒事」與「管道死了」**。

    這是 2026-09-05 到 09-13 那 8 天的結構性修法：當時看板每 6 小時跑、
    轉換有發生、投遞全失敗，而沉默看起來跟健康完全一樣。
    """
    return send(text, source="heartbeat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ch = channels()
    print("有設定的管道：%s" % (", ".join(ch) if ch else "**無**"))
    for k in ("DISCORD_WEBHOOK_URL", "TELEGRAM_BOT_TOKEN",
              "TG_ALERT_CHAT_ID", "TG_CRITICAL_CHAT_ID", "TELEGRAM_CHAT_ID"):
        src = ("env" if os.environ.get(k) else
               (".env" if _dotenv().get(k) else "—"))
        print("  %-22s %s" % (k, src))
    if "--test" in sys.argv:
        if not ch:
            print("\n沒有管道可測。把 DISCORD_WEBHOOK_URL 放進 .env 再跑。")
            sys.exit(2)
        r = send("flowbot 告警管道測試 — 這一則是人為觸發的，"
                 "收到代表投遞鏈通了。", source="manual-test")
        print("\n投遞結果：%s" % json.dumps(r, ensure_ascii=False, indent=2))
        sys.exit(0 if r["delivered"] else 1)
    print("\n要實際送一則測試：python research/ops/notify.py --test")
