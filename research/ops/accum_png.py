# -*- coding: utf-8 -*-
"""資料監控站的圖 —— 推到 Discord 的那一張。

使用者 2026-09-13：「回報用圖表的方式」。那個頻道本來就在收 V7 的 PNG
（`indicator/app.py:_send_discord_photo`，multipart webhook），所以圖片這條路
早就證明通了；換掉的只是內容。

===========================================================================
為什麼主視圖是逐小時的帶
===========================================================================
`freshness_board` 答的是「**現在**活不活」，它結構上答不了「有沒有斷過」——
2026-09-11 `hl_mid` 被看門狗殺了 70 次、近 6 小時沒錄到，而看板全程綠的
（每 5 分鐘被重啟一次，所以「最後一筆」永遠很新）。

所以這張圖一格一小時，**空格就是洞**，而且：
  * **不可回填的洞畫紅色，可回填的畫灰色** —— 前者是永久損失、後者只是待補
  * 深淺 = 該小時列數 ÷ 營運水準（p90），所以「量掉一半」看得出來，
    不是只有「完全沒有」才看得出來

樣式照 `hft-alphas-pt-2` 的參照標準（純 matplotlib、單色 #55a868、
標題寫明量的是什麼／期間／樣本），而不是塞滿顏色。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SNAP = os.path.join(ROOT, "research", "results", "accum_snapshot.json")
OUT = os.path.join(ROOT, "research", "results", "accum.png")

GREEN = "#55a868"       # 取代原本的鋼藍 #4c72b0（使用者 2026-09-13）
RED = "#c0392b"
GREY = "#8a93a3"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # **中文要明寫字型**，否則全變方框。JhengHei 在這台機器上有
    # （fig_ledger 用 DejaVu Sans 是因為它的標題全英文）。
    plt.rcParams.update({
        "font.family": ["Microsoft JhengHei", "DejaVu Sans"],
        "font.size": 9, "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8, "figure.dpi": 130,
        "axes.unicode_minus": False,
    })
    return plt


def render(snap, out=OUT):
    import numpy as np
    plt = _style()

    hours = snap["hours"]
    sets = [s for s in snap["sets"]]
    n, m = len(sets), len(hours)

    # RGB 陣列直接畫，不用 colormap —— 因為「洞」與「量少」是兩種語意，
    # 用同一條色階表達會把它們混在一起。
    img = np.ones((n, m, 3))
    for i, s in enumerate(sets):
        r = s.get("rows") or {}
        lvl = max(1, s.get("hourly_level") or 1)
        first = s.get("first_hour")
        fill = s.get("backfillable", False)
        for j, k in enumerate(hours):
            if first and k < first:
                img[i, j] = (0.93, 0.93, 0.94)            # 上線前
            elif not r.get(k):
                img[i, j] = ((0.62, 0.66, 0.71) if fill
                             else (0.75, 0.23, 0.17))     # 灰=可補 紅=永久
            else:
                a = 0.25 + 0.75 * min(1.0, r[k] / lvl)
                img[i, j] = (1 - a * (1 - 0x55 / 255),
                             1 - a * (1 - 0xa8 / 255),
                             1 - a * (1 - 0x68 / 255))

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11.5, 1.05 + 0.34 * n + 1.1),
        gridspec_kw={"height_ratios": [0.34 * n + 0.4, 0.95], "hspace": 0.55})

    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(n))
    lab = []
    for s in sets:
        miss = s.get("hours_missing", 0)
        tag = "" if s.get("backfillable") else "*"
        # 事件驅動的資料稀疏是正常的（那小時沒有清算 != 斷線）。不標的話
        # 那一列在圖上看起來像壞掉 —— 而假紅燈會訓練人忽略整個頻道。
        ev = "（事件驅動）" if s.get("event_driven") else ""
        lab.append("%s%s%s" % (s["name"], tag, ev)
                   + ("  缺%d" % miss if miss else ""))
    ax.set_yticklabels(lab, fontsize=8)
    # 時間軸：**每小時一格**（使用者 2026-09-13）。336 格全部標字沒有人讀得完，
    # 所以分兩層 —— 次刻度每 1 小時一根（看得出格線、數得出第幾小時），
    # 主刻度每 6 小時標 HH，日界另外標日期。
    hticks = list(range(len(hours)))
    ax.set_xticks([j - 0.5 for j in hticks], minor=True)
    # 標籤只放 00 與 12。第一版每 6 小時一個、而且日界用兩行（日期\n00），
    # 結果相鄰標籤直接黏成「1808/3106」—— 14 天 x 4 = 56 個標籤塞不進 11.5 吋。
    major, mlab = [], []
    for j, k in enumerate(hours):
        if k[8:] == "00":
            major.append(j)
            mlab.append("%s/%s" % (k[4:6], k[6:8]))
        elif k[8:] == "12":
            major.append(j)
            mlab.append("12")
    ax.set_xticks(major)
    ax.set_xticklabels(mlab, fontsize=6.5)
    ax.tick_params(axis="x", which="major", length=3)
    ax.tick_params(axis="x", which="minor", length=1.5, color="#bbbbbb")
    ax.grid(axis="x", which="minor", color="#ffffff", linewidth=0.25, alpha=0.5)
    ax.set_axisbelow(False)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_title(
        "資料累積逐小時落地｜每格 1 小時、共 %d 天｜深淺 = 該小時列數 ÷ 營運水準(p90)\n"
        "紅 = 不可回填的洞（永久損失）　灰 = 可回填的洞　淺灰 = 上線前　"
        "名稱後的 * = 不可回填　UTC %s"
        % (snap["days"], snap["asof_utc"][:16].replace("T", " ")),
        fontsize=9.5, loc="left", pad=8)

    # ── 下方：近 24h 的列數（橫條）＋ 容量一行 ──────────────────────
    names = [s["name"] for s in sets]
    vals = [s.get("rows_24h", 0) for s in sets]
    order = sorted(range(n), key=lambda i: -vals[i])[:6]
    ax2.barh([names[i] for i in order][::-1], [vals[i] for i in order][::-1],
             color=GREEN, height=0.62)
    # 用 log 不用 symlog：全部 > 0，而 symlog 會多畫一段沒有意義的線性區，
    # 讓軸從 0 起跳、再跳到 10^0，讀起來像壞掉。
    ax2.set_xscale("log")
    ax2.set_xlim(left=max(1e2, min(v for v in vals if v > 0) * 0.6))
    ax2.set_xlabel("近 24 小時的列數（對數軸）", fontsize=8.5)
    for i, idx in enumerate(order[::-1]):
        ax2.text(vals[idx], i, "  " + format(vals[idx], ","),
                 va="center", fontsize=7.5, color="#444")
    ax2.tick_params(labelsize=8, length=2)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    c = snap.get("capacity", {})
    d, my = c.get("disk_D", {}), c.get("mysql", {})
    ax2.set_title("容量：D 槽 %.0f / %.0f GB 可用｜MySQL 表資料 %.1f GB"
                  "（volume 上限 250 GB，實佔約 2.3x）"
                  % (d.get("free_gb", 0), d.get("total_gb", 0),
                     my.get("used_mb", 0) / 1024),
                  fontsize=9, loc="left", pad=6)

    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def build(out=OUT):
    with open(SNAP, encoding="utf-8") as fh:
        snap = json.load(fh)
    age = (datetime.now(timezone.utc)
           - datetime.fromisoformat(snap["asof_utc"])).total_seconds() / 3600
    p = render(snap, out)
    return p, snap, age


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    p, snap, age = build()
    print("寫出 %s（%d bytes，快照 %.1f 小時前）"
          % (p, os.path.getsize(p), age))
    miss = sum(s.get("hours_missing", 0) for s in snap["sets"])
    print("  %d 個資料集｜上線後的洞合計 %d 小時" % (len(snap["sets"]), miss))
