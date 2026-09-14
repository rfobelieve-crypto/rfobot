# -*- coding: utf-8 -*-
"""現在真正在跑的 HMM live 引擎有哪些 —— 只有這一份答案。

**為什麼要算而不是寫死（2026-09-14）**：這個形狀當天出現了四次，
而代價不是噪音 —— **真的紅燈會被埋在假的紅燈裡**：

    freshness 的 HMM 那一列寫死 GMX，GMX 退場 -> 一盞永遠紅的燈
    AERO 的 Discord 看護沒跟著停 -> 整個下午每五分鐘一則假警報
    XPL 的看護同上
    而 scan_pull 連死 31 次的那個下午，唯一在響的頻道報的是別的東西

真相源是看門狗的 `$Members` 減去 `logs/stop/*.stop`，也就是「停止」那個
單一狀態。這支是它的第五個讀者（`.bat` 的迴圈、`arb_watchdog.ps1`、
`account_budget.py`、`freshness_board.py`，加上 `hmm_watch.py`）。

**算不出來時回空 list，呼叫端要把它當成「不知道」而不是「沒有」** ——
一個看護在解析失敗時安靜地不盯任何東西，跟它盯著而沒事，長得一模一樣
（mistake.md 2026-09-03：未知狀態不可以長得像已知狀態）。所以呼叫端
必須自己決定要不要為「空」留一盞燈，這支不替它決定。
"""
from __future__ import annotations

import glob
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ARB = os.path.join(HERE, "..", "..", "..", "arb")

_MEMBER = re.compile(
    r"^\s*'([A-Za-z0-9_]+)'\s*=\s*@\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)", re.M)

__all__ = ["live_hmm_pairs", "arb_root"]


def arb_root() -> str:
    return os.path.normpath(ARB)


def live_hmm_pairs() -> list:
    """會送真單、而且現在沒有被 STOP 旗標停掉的 HMM 標的，由註冊順序排列。

    排除三種：STOP 旗標停掉的、不是 `run_hmm_*` 的（錄製器）、
    以及啟動器帶 `--record-only` / `--shadow` 的（那兩個不送單）。
    """
    arb = arb_root()
    try:
        src = io.open(os.path.join(arb, "ops", "arb_watchdog.ps1"),
                      encoding="utf-8").read()
    except Exception:
        return []
    members = _MEMBER.findall(src)
    if not members:
        return []
    stopped = {os.path.basename(f)[:-len(".stop")] + ".bat"
               for f in glob.glob(os.path.join(arb, "engine", "logs",
                                               "stop", "*.stop"))}
    out = []
    for name, _sig, bat in members:
        if bat in stopped or not bat.startswith("run_hmm_"):
            continue
        try:
            body = io.open(os.path.join(arb, "engine", bat),
                           "rb").read().decode("ascii", "replace")
        except Exception:
            continue
        if "--record-only" in body or "--shadow" in body:
            continue
        out.append(name)
    return out


if __name__ == "__main__":
    ps = live_hmm_pairs()
    print(" ".join(ps) if ps else "")
