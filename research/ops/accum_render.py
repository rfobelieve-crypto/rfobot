# -*- coding: utf-8 -*-
"""把 accum_snapshot.json 畫成一頁本機 HTML。

===========================================================================
為什麼刻意樸素
===========================================================================
`starting-your-quant-trading-business`（2024-02-18）原話：

    「I want to warn all beginners to approach this component on an as needed
     basis strictly. It's a shame to see so many junior quants with dashboards
     that clearly got more of their time than the algorithm itself.
     **The dashboard is for the algorithm NOT the other way around!**」

所以這支沒有框架、沒有 CDN、沒有互動，純字串組 HTML。它只要回答一個問題：
**有沒有在累積、什麼時候斷過。**

`deploying-strategies`（2024-07-18）還給了一條結構性的分法，本支屬於前者：

    「I always have two types of dashboards: a **monitor** and a **control**
     dashboard... suppose you are only showing monitoring statistics. In that
     case, you can be much less strict with the security on it because it
     can't actually control the algorithm.」

本支**不碰任何交易路徑、不寫任何東西**，所以它可以放在最省事的地方。

===========================================================================
呈現的三個決定
===========================================================================
1. **主視圖是逐小時的帶，不是一排燈。** 一排燈只答得出「現在」，而使用者問的
   是「有沒有在跑」。帶上的空格就是洞。
2. **不可回填的洞用紅色，可回填的用灰色。** 前者是永久損失、後者只是待補，
   兩者不該長一樣。
3. **顏色深淺 = 該小時列數 / 營運水準（p90）**，所以「量掉一半」看得出來，
   而不是只有「完全沒有」才看得出來。
"""
from __future__ import annotations

import json
import os
import sys
import webbrowser
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SNAP = os.path.join(ROOT, "research", "results", "accum_snapshot.json")
OUT = os.path.join(ROOT, "research", "results", "accum.html")

CSS = """
:root{--bg:#12151a;--fg:#e6e9ef;--dim:#8a93a3;--line:#242a33;
      --ok:#4c72b0;--gap:#c0392b;--soft:#5a6472;--warn:#d79a2b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:13px/1.5 "Consolas","Cascadia Mono",ui-monospace,monospace}
.wrap{max-width:1500px;margin:0 auto;padding:22px 18px 60px}
h1{font-size:17px;margin:0 0 2px;font-weight:600;letter-spacing:.3px}
.sub{color:var(--dim);font-size:12px;margin:0 0 22px}
h2{font-size:13px;margin:30px 0 10px;font-weight:600;color:var(--dim);
   text-transform:uppercase;letter-spacing:.9px}
table{border-collapse:collapse;width:100%}
th{text-align:right;color:var(--dim);font-weight:500;font-size:11px;
   padding:0 7px 7px;border-bottom:1px solid var(--line);white-space:nowrap}
th.l,td.l{text-align:left}
td{padding:4px 7px;border-bottom:1px solid var(--line);text-align:right;
   white-space:nowrap;font-variant-numeric:tabular-nums}
td.name{font-weight:600}
svg{display:block}
.note{color:var(--dim);font-size:11px}
.tag{font-size:10px;padding:1px 5px;border-radius:3px;border:1px solid var(--line);
     color:var(--dim)}
.tag.no{color:#e0a0a0;border-color:#5a3030}
.bar{height:9px;background:var(--line);border-radius:5px;overflow:hidden;
     width:240px;display:inline-block;vertical-align:middle}
.bar>span{display:block;height:100%;background:var(--ok)}
.bar>span.hot{background:var(--gap)}
.legend{color:var(--dim);font-size:11px;margin-top:9px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;
          vertical-align:-1px;margin:0 4px 0 11px}
"""


def strip_svg(hours, rows, level, first, backfillable, w=4, h=17, gap=1):
    """一列 = **一個 inline SVG**，不是 336 個 flex 子元素。

    第一版每小時一個 `<i style="flex:1 1 0">`：11 列 x 336 小時 = 3,696 個要做
    彈性排版的節點，**Chrome 的渲染器直接卡死**（CDP captureScreenshot 逾時
    30 秒，分頁掛掉）。那不是截圖的問題——使用者打開也會卡。
    SVG 的 rect 不走排版，同樣的格數是瞬間的事。
    """
    # 先把每小時算出顏色，再**把同色的連續小時合併成一個 rect**。
    # 沒有合併的話一列 336 個 rect、每個再包一個 <title> = 7,392 個節點，
    # Chrome 渲染不完（分頁標題一直停在 URL）。健康的列同色連續，合併後
    # 通常只剩十幾個 rect。逐格 tooltip 一併拿掉 —— 帶子的工作是「一眼看出
    # 洞在哪」，滑鼠細節不值那個節點數。
    cols = []
    last = hours[-1] if hours else None
    for k in hours:
        v = rows.get(k)
        # 紅色只能代表「我知道資料缺了」，不能代表「我不知道」。
        # first is None = 剛上線、完整小時數不到 3 個，判不出上線時刻 ——
        # 舊版讓整列全紅（PNG 那側同一個 bug，2026-09-13 使用者截圖回報）。
        if first is None or k < first:
            cols.append("#1a1e25")
        elif k == last and not v:
            # **最後一格是進行中的那一小時，不是洞。** 錄製器每 300 秒才落盤，
            # 整點過後前幾分鐘那個小時檔還不存在 -> 舊版塗紅，於是每小時都
            # 冒一道假紅線。缺口計數本來就排除它了，漏的是畫圖這一側。
            cols.append("#2a3038")
        elif v is None or v <= 0:
            cols.append("#5a6472" if backfillable else "#c0392b")
        else:
            # 量化成 6 階，否則每小時都是不同的 alpha、永遠合併不起來
            q = min(5, int(6 * min(1.0, v / level if level else 1.0)))
            cols.append("rgba(76,114,176,%.2f)" % (0.25 + 0.15 * q))
    out, i, n = [], 0, len(cols)
    while i < n:
        j = i
        while j + 1 < n and cols[j + 1] == cols[i]:
            j += 1
        out.append('<rect x="%d" width="%d" height="%d" fill="%s"/>'
                   % (i * (w + gap), (j - i + 1) * (w + gap) - gap, h, cols[i]))
        i = j + 1
    total = n * (w + gap)
    return ('<svg width="%d" height="%d" viewBox="0 0 %d %d" '
            'shape-rendering="crispEdges">%s</svg>'
            % (total, h, total, h, "".join(out)))


def human(n):
    n = float(n)
    for u, d in (("M", 1e6), ("k", 1e3)):
        if n >= d:
            return "%.1f%s" % (n / d, u)
    return "%.0f" % n


def render(snap) -> str:
    hours = snap["hours"]
    asof = snap["asof_utc"]
    rows = []
    for s in snap["sets"]:
        if s.get("error"):
            rows.append('<tr><td class="l name">%s</td>'
                        '<td class="l" colspan="7" style="color:var(--gap)">'
                        '錯誤：%s</td></tr>' % (s["name"], s["error"]))
            continue
        r = s.get("rows") or {}
        lvl = max(1, s.get("hourly_level") or 1)
        first = s.get("first_hour")
        strip = strip_svg(hours, r, lvl, first, s.get("backfillable", False))
        miss = s.get("hours_missing", 0)
        empty = s.get("hours_empty", 0)
        gap_txt = ('<span style="color:var(--gap)">%d</span>' % miss) if miss \
            else ('<span class="note">%d 空</span>' % empty if empty else
                  '<span class="note">0</span>')
        rows.append(
            '<tr><td class="l name">%s</td>'
            '<td class="l"><span class="tag%s">%s</span></td>'
            '<td class="l">%s</td>'
            '<td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
            '<td class="l note">%s</td></tr>'
            % (s["name"],
               "" if s.get("backfillable") else " no",
               "可回填" if s.get("backfillable") else "不可回填",
               strip,
               s.get("hours_live", 0), gap_txt,
               human(s.get("rows_24h", 0)), human(lvl),
               (s.get("note") or "")))

    c = snap["capacity"]
    d, mysql = c["disk_D"], c["mysql"]
    used_d = d.get("total_gb", 0) - d.get("free_gb", 0)
    pct_d = 100 * used_d / max(d.get("total_gb", 1), 1)
    # MySQL 上限未知 —— 不要假裝知道。畫三條參考線而不是一個百分比。
    cap = (
        '<table><tr><th class="l">位置</th><th class="l">用量</th>'
        '<th class="l">說明</th></tr>'
        '<tr><td class="l name">D 槽（parquet）</td><td class="l">'
        '<span class="bar"><span style="width:%.1f%%"></span></span> '
        '%.0f / %.0f GB（%.0f%%）</td>'
        '<td class="l note">不可回填的錄製全在這裡</td></tr>'
        '<tr><td class="l name">MySQL（Railway）</td><td class="l">'
        '<span class="bar"><span class="hot" style="width:%.1f%%"></span></span> '
        '%.0f MB / %d 表</td>'
        '<td class="l note">**上限未知**：MySQL 查不到 Railway 掛的 volume 大小。'
        '5 GB→約 18 天／10 GB→64 天／50 GB→1.3 年</td></tr></table>'
        % (pct_d, used_d, d.get("total_gb", 0), pct_d,
           100 * mysql["used_mb"] / 5120.0, mysql["used_mb"], mysql["tables"]))

    # **`<meta charset>` 不是可選的。** 少了它，瀏覽器用系統語系（這台是
    # cp950）讀這份 UTF-8，中文全變亂碼 —— 實測分頁標題變成「鞈��蝝舐�」。
    # 跨語言邊界兩邊都要明寫編碼（mistake.md 2026-09-11：同一個根，那次是
    # PowerShell 讀 Python 寫的 UTF-8 旗標）。`file://` 開也一樣會中。
    return ('<!doctype html><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>資料累積</title><style>%s</style>"
            '<div class="wrap"><h1>資料累積</h1>'
            '<p class="sub">每格一小時，共 %d 天｜UTC %s｜'
            '這一頁回答「有沒有在累積」，不是「現在活不活」'
            '（後者看 freshness_board）</p>'
            '<h2>逐小時落地</h2>'
            '<table><tr><th class="l">資料集</th><th class="l"></th>'
            '<th class="l">← %d 天 · 每格 1 小時 →</th>'
            '<th>上線時數</th><th>缺口</th><th>近 24h</th><th>列/時</th>'
            '<th class="l">來源</th></tr>%s</table>'
            '<p class="legend">深淺 = 該小時列數 ÷ 營運水準（p90）'
            '<i style="background:var(--gap)"></i>不可回填的洞（永久損失）'
            '<i style="background:var(--soft)"></i>可回填的洞'
            '<i style="background:#1a1e25"></i>上線前</p>'
            '<h2>容量</h2>%s</div>'
            % (CSS, snap["days"], asof.replace("T", " "), snap["days"],
               "".join(rows), cap))


def main():
    if not os.path.exists(SNAP):
        print("找不到 %s —— 先跑 accum_snapshot.py" % SNAP)
        return 2
    with open(SNAP, encoding="utf-8") as fh:
        snap = json.load(fh)
    age = (datetime.now(timezone.utc)
           - datetime.fromisoformat(snap["asof_utc"])).total_seconds() / 3600
    if age > 3:
        print("  [WARN] 快照已經 %.1f 小時舊了 —— 先跑 accum_snapshot.py" % age)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(render(snap))
    print("寫出 %s（快照 %.1f 小時前）" % (OUT, age))
    if "--open" in sys.argv:
        webbrowser.open("file:///" + OUT.replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
