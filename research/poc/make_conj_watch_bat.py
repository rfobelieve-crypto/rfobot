# -*- coding: utf-8 -*-
"""產生 research/ops/conj_watch.bat（分鐘級 shadow 偵測器的排程啟動器）。

為什麼要用一支 .py 來產生一個 .bat
    這個 repo 對 .bat 踩過三次坑，三次都是「內容在寫入的路上被改掉」：
      2026-08-19  Edit 工具把 CRLF 換成 LF -> cmd 解析不了，排程靜默死 29 小時
      2026-08-20  bash 雙引號的 python -c 把路徑裡的 \\v \\r 吃成控制字元
      2026-09-06  同一件事又發生一次，而「驗證」只數行尾（對行內控制字元免疫）
    所以：**內容不穿過任何 shell**，反斜線用 chr(92) 組出來（連 Python 自己的
    轉義層都繞開），寫檔用 bytes + 明確的 CRLF，寫完掃整檔控制字元。

.bat 本身必須是純 ASCII
    cmd.exe 以 OEM 碼頁（這台是 cp950）讀 .bat，UTF-8 的中文註解會被錯誤解碼，
    殘餘位元組當成指令執行（2026-09-07 實際發生過）。中文理由寫在這裡，不寫在
    那裡。
"""
from pathlib import Path

BS = chr(92)                      # 反斜線，不寫字面值
ROOT = "C:" + BS + "Users" + BS + "rfo" + BS + "Desktop" + BS + "flowbot" + BS + "flow_system"
OUT = Path(ROOT) / "research" / "ops" / "conj_watch.bat"

P = "%ROOT%" + BS                 # 路徑前綴
LOG = P + "research" + BS + "results" + BS + "conj_watch.log"
PY = P + "research" + BS + "poc" + BS + "conj_watch.py"

LINES = [
    "@echo off",
    "REM Minute-level shadow detector for conjunction events (TODO 1.03).",
    "REM Detector: research/poc/conj_watch.py  (shadow mode: logs, never signals)",
    "REM Verified by three pre-registered controls before this task was created:",
    "REM   conj_watch_parity  A arm 100.0%% (assembly)  B arm 92.6%% (thresholds)",
    "REM   conj_watch_inject  88.9%% hit on real historical conjunctions",
    "REM",
    "REM Runs every minute. The point is to accumulate REAL end-to-end latency;",
    "REM the 2-minute budget cannot be settled by an estimate.",
    "REM",
    "REM ASCII ONLY. cmd.exe reads .bat in the OEM codepage (cp950 here); UTF-8",
    "REM CJK in comments gets mis-decoded and stray bytes execute as commands.",
    "REM Rationale in Chinese lives in research/poc/make_conj_watch_bat.py.",
    "REM",
    "REM Launched by research" + BS + "ops" + BS + "run_hidden.vbs (wscript has no console).",
    "REM Proof it ran is the ARTIFACT: mtime and ok field of conj_watch_last.json,",
    "REM plus rows in conj_events_live. Never LastTaskResult - an async launcher's",
    "REM exit code says nothing about the child.",
    "setlocal",
    "set ROOT=" + ROOT,
    "set PYTHONIOENCODING=utf-8",
    'cd /d "%ROOT%"',
    'python "' + PY + '" >> "' + LOG + '" 2>&1',
    "endlocal",
]


def main():
    data = ("\r\n".join(LINES) + "\r\n").encode("ascii")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)

    # 驗證：掃**整檔控制字元**，不是數行尾（行尾正確與行內乾淨是兩件事）
    bad = [(i, b) for i, b in enumerate(data)
           if b < 0x20 and b not in (0x0A, 0x0D)]
    n_crlf = data.count(b"\r\n")
    n_bare = data.count(b"\n") - n_crlf
    print(f"written -> {OUT}")
    print(f"  bytes {len(data)}  CRLF {n_crlf}  bare LF {n_bare}")
    print(f"  non-ascii {[b for b in data if b > 127]}")
    print(f"  stray control chars {bad}")
    assert not bad and n_bare == 0 and all(b < 128 for b in data)
    print("  OK: pure ASCII, CRLF only, no stray control characters")


if __name__ == "__main__":
    main()
