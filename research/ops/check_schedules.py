# -*- coding: utf-8 -*-
"""排班的每一個持有者：檔案還在嗎、有沒有版本控制（2026-09-10）

**這條守衛擋的是一整類事故，而且這類事故在這個 repo 發生過三次**：

    2026-07-05  repo 改名後，排程的 action 還指著舊的 CJK 路徑，
                每天 04:00 exit 1，**96 天**沒人發現
    2026-09-04  搬走整條線時 grep 了 repo 抓到 10 個持有者，
                第 11 個是排程的 action —— 它不在 repo 裡，所以 grep 看不到
    2026-09-10  daily_collect.bat 在 .gitignore 裡（理由是「含祕密」，
                但它其實只是從 .env 讀 key）——排班內容沒有版本控制，
                改了就只存在於這台機器上

共同形狀：**關鍵狀態存在版本控制與 grep 的範圍之外，而且失效時不會報錯。**
mistake.md 2026-09-01 給過修法的方向：把「從未開始 / 已經失聯」翻譯成
**某個數字不對**，那是既有守衛看得懂的語言。

判準（任一不過 -> ok=false）：

    E1  排程 action 引用的每一個檔案都必須**存在**
    E2  在這個 repo 裡的那些，必須**被 git 追蹤**
        （否則改動不進版控、換機器就消失、而且沒有東西會提醒）
    E3  在 repo 之外的（例如另一個 repo 的 ops），只要求存在並列出來
        —— 它們有自己的版控，這裡只負責讓它們**被看見**

輸出 results/schedules_last.json 給 freshness board 的 json_flag 讀。
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "results" / "schedules_last.json"
PREFIXES = ("FlowBot", "Flowbot", "SweepShadow", "EntropyArb")
PATH_RE = re.compile(r'"?([A-Za-z]:\\[^"]+?\.(?:bat|cmd|ps1|vbs|py))"?', re.I)


def tasks():
    ps = ("Get-ScheduledTask | ForEach-Object { $t=$_; $_.Actions | "
          "ForEach-Object { [pscustomobject]@{name=$t.TaskName; "
          "state=[string]$t.State; exe=$_.Execute; args=$_.Arguments} } } "
          "| ConvertTo-Json -Depth 3")
    p = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, errors="replace",
                       timeout=180)
    try:
        d = json.loads(p.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return d if isinstance(d, list) else [d]


def tracked(rel):
    p = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                       cwd=str(ROOT), capture_output=True, text=True)
    return p.returncode == 0


def main():
    rows, bad = [], []
    for t in tasks():
        nm = str(t.get("name") or "")
        if not nm.startswith(PREFIXES):
            continue
        blob = "%s %s" % (t.get("exe") or "", t.get("args") or "")
        refs = PATH_RE.findall(blob)
        if not refs:
            refs = [str(t.get("exe") or "")]
        for r in refs:
            p = Path(r)
            exists = p.exists()
            inside = False
            try:
                rel = str(p.resolve().relative_to(ROOT)).replace("\\", "/")
                inside = True
            except ValueError:
                rel = None
            trk = tracked(rel) if (inside and exists) else None
            row = dict(task=nm, state=str(t.get("state") or ""), path=r,
                       exists=exists, inside_repo=inside, tracked=trk)
            rows.append(row)
            if not exists:
                bad.append("%s -> MISSING %s" % (nm, r))          # E1
            elif inside and trk is False:
                bad.append("%s -> UNTRACKED %s" % (nm, rel))      # E2

    ok = not bad
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        ok=ok, reason=("; ".join(bad)[:400] if bad else
                       "%d refs ok" % len(rows)),
        n_refs=len(rows), n_bad=len(bad), rows=rows,
        asof=time.strftime("%Y-%m-%d %H:%M:%S")),
        indent=2, ensure_ascii=False), encoding="utf-8")

    print("%-28s %-9s %-7s %-8s %s" % ("排程", "存在", "在 repo", "被追蹤", "路徑"))
    for r in rows:
        print("%-28s %-9s %-7s %-8s %s"
              % (r["task"][:27], "yes" if r["exists"] else "**NO**",
                 "yes" if r["inside_repo"] else "-",
                 "-" if r["tracked"] is None else
                 ("yes" if r["tracked"] else "**NO**"),
                 Path(r["path"]).name))
    print()
    print("schedules: %s  %s" % ("OK" if ok else "RED",
                                 "; ".join(bad) if bad else
                                 "%d refs ok" % len(rows)))
    print("written -> " + str(OUT))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
