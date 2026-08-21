# -*- coding: utf-8 -*-
"""Generate docs/DB_REGISTRY.md — weakness #3: 45 tables, no catalog.
For each table: row count, newest timestamp col value, and every repo file
that WRITES (INSERT/UPDATE/DELETE/CREATE) or READS (FROM/JOIN) it."""
import re, sys, subprocess
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"C:\Users\rfo\Desktop\flowbot\flow_system")
sys.path.insert(0, str(ROOT))
from shared.db import get_db_conn

conn = get_db_conn(); cur = conn.cursor()
cur.execute("SHOW TABLES")
tables = [list(r.values())[0] for r in cur.fetchall()]

# per-table stats
stats = {}
for t in tables:
    try:
        cur.execute(f"SELECT COUNT(*) n FROM `{t}`")
        n = cur.fetchone()["n"]
        cur.execute(f"SHOW COLUMNS FROM `{t}`")
        cols = [(r["Field"], str(r["Type"])) for r in cur.fetchall()]
        tcol = next((c for c, ty in cols if ("timestamp" in ty or "datetime" in ty)
                     and c in ("updated_at","created_at","ts","dt","snapshot_time","signal_time","entry_time","first_seen","time","t")), None)
        if tcol is None:
            tcol = next((c for c, ty in cols if "timestamp" in ty or "datetime" in ty), None)
        last = None
        if tcol and n:
            cur.execute(f"SELECT MAX(`{tcol}`) m FROM `{t}`")
            last = cur.fetchone()["m"]
        stats[t] = (n, tcol, last)
    except Exception as e:
        stats[t] = (None, None, f"ERR {e}")
conn.close()

# repo scan: files mentioning each table, classified
def scan(t):
    try:
        out = subprocess.run(["git","grep","-l",t,"--","*.py","*.js"],
                             capture_output=True, text=True, cwd=ROOT).stdout.split()
    except Exception:
        out = []
    writers, readers = [], []
    for f in out:
        try:
            src = (ROOT/f).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        near = [m.start() for m in re.finditer(re.escape(t), src)]
        w = r = False
        for pos in near:
            ctx = src[max(0,pos-200):pos+80].upper()
            if re.search(r"INSERT|UPDATE\s|DELETE\s|CREATE TABLE|REPLACE INTO", ctx): w = True
            if re.search(r"SELECT|FROM\s|JOIN\s", ctx): r = True
        if w: writers.append(f)
        elif r: readers.append(f)
        else: readers.append(f+"?")
    return writers, readers

lines = ["# DB Registry — 45 表目錄（自動生成 + 人工註記）",
"",
"> 2026-08-21 生成（數據工程弱點 #3）。重生成：跑",
"> `scratchpad/gen_db_registry.py`（產生器會隨 session 清掉，邏輯簡單可重寫：",
"> SHOW TABLES + git grep 分類 writer/reader）。**新表上線必須同步登記**，",
"> agent 可讀的表另需在 `.claude/rules/agent-boundary.md` 登記——兩處都要。",
"",
"| 表 | 列數 | 最新資料 | writers | readers |",
"|---|---|---|---|---|"]
for t in sorted(tables):
    n, tc, last = stats[t]
    w, r = scan(t)
    wf = "<br>".join(sorted(set(x.split('/')[-1] for x in w))[:4]) or "—"
    rf = "<br>".join(sorted(set(x.split('/')[-1] for x in r))[:4]) or "—"
    extra_w = len(set(x.split('/')[-1] for x in w))-4
    extra_r = len(set(x.split('/')[-1] for x in r))-4
    if extra_w>0: wf += f"<br>+{extra_w}"
    if extra_r>0: rf += f"<br>+{extra_r}"
    lasts = str(last)[:16] if last else "—"
    lines.append(f"| `{t}` | {n if n is not None else '?'} | {lasts} | {wf} | {rf} |")
out = ROOT/"docs"/"DB_REGISTRY.md"
out.write_text("\n".join(lines)+"\n", encoding="utf-8")
print(f"written {out} — {len(tables)} tables")
orphans = [t for t in sorted(tables) if not scan(t)[0] and not scan(t)[1]]
print("tables with NO repo references (retirement candidates):", orphans)
