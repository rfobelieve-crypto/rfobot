# -*- coding: utf-8 -*-
"""資料監控站：每小時產圖 + 推到 Discord。掛在每小時班車上。

使用者 2026-09-13：「圖表每小時更新一次就好了」。

===========================================================================
為什麼是每小時，而不是只在出事時才推
===========================================================================
**規律的脈搏讓沉默本身變成警報。** 2026-09-05 到 09-13 那 8 天，告警是
「只在狀態轉換時才送」，於是投遞全部失敗的那 8 天跟「一切正常」在畫面上
**完全一樣**——沒有人能從沒收到訊息推論出任何事。

每小時一張圖之後，停了幾小時就看得出來。這也是它接替 V7 每小時圖表那個
時段的理由（`indicator/app.py` 的 `V7_DISCORD_HOURLY_CHART` 已預設關）。

===========================================================================
兩個刻意的設計
===========================================================================
1. **圖裡帶新鮮度看板的年齡。** 看板跑在**另一個排程**上（FreshnessBoard，
   每 6 小時），而本支跑在每小時班車（SweepShadow）。兩個排程互相看得到
   對方停了沒 —— 否則「誰來監測監測者」是個閉環。
2. **推不出去不是安靜失敗。** `notify.send_image()` 會把結果寫進
   `alert_last.json`，而那份旗標已經註冊成看板的一列（門檻 26h）。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

FB = os.path.join(ROOT, "research", "results", "freshness_board.json")


def _board():
    """新鮮度看板的紅燈數與它自己的年齡（小時）。"""
    try:
        with open(FB, encoding="utf-8") as fh:
            b = json.load(fh)
        reds = b.get("reds") or []
        n = len(b.get("rows") or [])
        asof = b.get("asof_utc") or ""
        age = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                dt = datetime.strptime(asof, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                break
            except ValueError:
                continue
        return reds, n, age
    except Exception:                                   # noqa: BLE001
        return None, None, None


def main(post: bool = True) -> int:
    from research.ops import accum_png, accum_snapshot, notify

    snap = accum_snapshot.build()
    out = os.path.join(ROOT, "research", "results", "accum_snapshot.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, ensure_ascii=False, indent=1)
    accum_snapshot.write_flag(snap)

    png = accum_png.render(snap)
    miss = sum(s.get("hours_missing", 0) for s in snap["sets"])

    reds, ntr, age = _board()
    if reds is None:
        board = "新鮮度看板：**讀不到**"
    elif age is not None and age > 8:
        # 看板跑在另一個排程上，停了只有這裡看得見。
        board = ("新鮮度 %d red / %d（**看板已 %.1f 小時沒更新**）"
                 % (len(reds), ntr, age))
    else:
        board = "新鮮度 %d red / %d" % (len(reds), ntr)
        if reds:
            board += "：" + ", ".join(reds[:5])

    cap = "flowbot 資料監控站｜上線後的洞合計 %d 小時｜%s" % (miss, board)
    print(cap)
    if not post:
        print("（--no-post，沒有推送）")
        return 0
    r = notify.send_image(png, cap, source="station")
    print("station %s：%s"
          % ("DELIVERED" if r["delivered"] else "NOT delivered", r["tried"]))
    return 0 if r["delivered"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(post="--no-post" not in sys.argv))
