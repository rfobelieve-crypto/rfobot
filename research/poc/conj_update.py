# -*- coding: utf-8 -*-
"""交會事件時鐘的一鍵更新 —— 補資料 -> 重建 -> 計分。

為什麼要有這支
    `conj_clock.py`（凍結日 2026-09-07）需要三份資料才跑得動：
    分鐘 K 線、由它建出的 bars parquet、以及 OI。三個抓取器都支援增量，
    但**沒有任何排程叫到它們**——查證過（grep .bat/.ps1/.vbs 零命中）。
    所以時鐘會永遠停在 0/300，而且**不會有任何燈變紅**。

    這正是 mistake.md 2026-09-01 那個形狀：不是「某個東西停了」，是
    「某件該做的事從來沒有被啟動」，而 freshness 對「從未開始」是瞎的。
    修法照那條記載的建議：把「從未開始」翻譯成「某個數字不對」——
    本檔每次跑完都寫 `conj_clock_last.json` 的 `{ok, reason}` 旗標，
    並登記進 freshness board 的 json_flag 觀測。

    所以：**排程只要叫這一支就好**，不要分別叫四支。

順序（每一步失敗都會被記進旗標，不會靜默）
    1. fetch_bars.py   增量抓 1 分鐘 K 線
    2. bars.py         重建 bars parquet（含 ATR、delta）
    3. fetch_oi.py     增量抓 OI
    4. conj_clock.py   計分並寫時鐘

用法
    python research/poc/conj_update.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FLAG = HERE / "data" / "results" / "conj_clock_last.json"

STEPS = [
    ("fetch_bars", HERE / "fetch_bars.py", []),
    ("bars", HERE / "bars.py", []),
    ("fetch_oi", HERE / "fetch_oi.py", []),
    ("conj_clock", HERE / "conj_clock.py", []),
]


def main():
    log = []
    ok = True
    reason = "ok"
    for name, script, extra in STEPS:
        if not script.exists():
            ok, reason = False, f"{name}: script missing"
            log.append(f"{name}: MISSING {script}")
            break
        # 子行程輸出是 UTF-8 中文，而 Windows 的 locale 是 cp950。
        # 不指定 encoding 時 subprocess 的讀取執行緒會 UnicodeDecodeError
        # **死在背景執行緒裡**：run() 照樣回 rc=0，但那一步的 stdout 遺失。
        # 2026-09-07 實測就是這樣——[conj_clock] rc=0 卻一行輸出都沒有，
        # 而下面的 STALE-DATA 守衛正是讀它的 stdout，等於永遠不會觸發。
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, str(script), *extra],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           env=env, cwd=str(ROOT), timeout=3600)
        tail = (r.stdout or "").strip().splitlines()[-3:]
        log.append(f"{name}: rc={r.returncode}  " + " | ".join(tail))
        print(f"[{name}] rc={r.returncode}")
        for ln in tail:
            print("   ", ln)
        if r.returncode != 0:
            ok = False
            reason = f"{name} rc={r.returncode}: {(r.stderr or '')[-300:]}"
            break
        # 資料過期時 conj_clock 會自己輸出 STALE-DATA 而不是判定 —— 那不是
        # 這支的失敗，但旗標要看得見（不然「資料舊」和「一切正常」長一樣）
        if name == "conj_clock" and "STALE-DATA" in (r.stdout or ""):
            ok, reason = False, "conj_clock: STALE-DATA"

    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text(json.dumps({
        "ok": ok, "reason": reason,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "log": log,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nflag -> {FLAG}   ok={ok}  reason={reason}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
