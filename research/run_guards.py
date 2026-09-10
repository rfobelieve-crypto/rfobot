# -*- coding: utf-8 -*-
"""每日跑研究端的回歸測試，並自報一個旗標給 freshness board（2026-09-10）

**為什麼需要這個**：這個 repo 沒有任何機制強制測試在改動後跑。2026-09-10
發現一個釘死的 parity 測試已經紅了（它的基準釘在一份會滾動的資料上），
而那是為了別的事順手跑才看到的 —— 四次「守衛壞掉沒人發現」的第四次。

**為什麼不上 GitHub Actions**：這三組測試吃的是本機資料（分鐘 bar、OI、
凍結的 1h 切片、sweep 快照），那些檔案不在 git 裡也不該在。雲端跑只會
全部 skip，而「全部 skip」跟「全部通過」在輸出上長得一樣 —— 那正是
mistake.md 2026-08-26 要擋的形狀。所以分工：

    本機（這支）       研究回歸測試，吃本機資料
    GitHub Actions    純 python 的結構性守衛（facade/boundary/payload）

**判準是產物不是退出碼**（mistake.md 2026-08-26）：這支把 {ok, reason,
passed, failed} 寫進 results/guards_last.json，freshness board 讀那個旗標。
旗標不新鮮或 ok=false 就紅。

跑法：
    python research/run_guards.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "results" / "guards_last.json"
SUITES = [
    "research/tests",
    "research/sweep_failure/tests",
    "research/crowd_stops/tests",
]


def main():
    t0 = time.time()
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, "-m", "pytest", *SUITES, "-q", "--no-header",
           "-p", "no:cacheprovider"]
    p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                       text=True, errors="replace", timeout=3600)
    tail = (p.stdout or "").strip().splitlines()[-1:] or [""]
    line = tail[0][:300]
    ok = p.returncode == 0

    # 「全部 skip」不算通過：一個都沒跑起來跟全過在輸出上長得一樣。
    import re
    passed = failed = skipped = 0
    for n, kind in re.findall(r"(\d+)\s+(passed|failed|skipped|error[s]?)", line):
        if kind == "passed":
            passed = int(n)
        elif kind == "failed":
            failed = int(n)
        elif kind == "skipped":
            skipped = int(n)
        else:
            failed += int(n)
    if passed == 0:
        ok = False
        line = "no test actually ran (all skipped?) — " + line

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        ok=bool(ok), reason=line, passed=passed, failed=failed,
        skipped=skipped, returncode=p.returncode,
        seconds=round(time.time() - t0, 1),
        asof=time.strftime("%Y-%m-%d %H:%M:%S"),
        suites=SUITES), indent=2, ensure_ascii=False), encoding="utf-8")
    print("guards: %s  %s" % ("OK" if ok else "RED", line))
    if not ok:
        print((p.stdout or "")[-3000:])
    print("written -> " + str(OUT))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
