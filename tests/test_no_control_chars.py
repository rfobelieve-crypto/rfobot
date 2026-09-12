# -*- coding: utf-8 -*-
r"""文字檔裡不得有行內控制字元 —— 反斜線路徑被某一層轉義吃掉的指紋。

**為什麼需要這支（這條規矩存在一個月，被犯了至少四次，而沒有任何東西在檢查）**

mistake.md 2026-08-19 / 08-20 / 09-06 都是同一件事：把含反斜線的 Windows
路徑寫進檔案時，內容穿過了 bash 的 heredoc 或 `python -c "..."`，於是
`\v` `\r` `\f` `\a` `\b` 被吃成**真的控制位元組**。失效方式是安靜的：

    .bat   整個檔案變成 cmd 解析不了 -> 排程每小時失敗而看板全綠（08-19，29 小時）
    .md    路徑顯示成 `D:<FF>lowbot_data`，下一個 session 照著它找檔案

2026-09-12 逐檔掃過一次，抓到 **8 個**：CLAUDE.md 的
`D:\flowbot_data\{raw_data,poc_data,hl}`（三個 `\f`）、當天我自己新寫的
`research\ops\run_hidden.vbs`（一個 `\r`），以及 mistake.md 裡
`flow_system\research\arb\ops\arb_watchdog.ps1`（兩個 `\a`）與
**描述這個 bug 的那一段自己帶著這個 bug**（兩個 `\v`）。

**判準刻意是「掃整檔的控制字元」不是「數行尾」** —— 行尾正確與行內乾淨是
兩件事，而 2026-09-06 就是因為只數行尾而漏掉（同族：用一個對目標現象免疫
的量去檢查）。

允許的只有 TAB、LF、以及成對的 CRLF。`␋` `␍` 這類 Unicode 控制圖示字元
（U+240B / U+240D）**是允許的** —— 它們是用來「顯示」控制字元的可見字形，
本身不是控制字元，mistake.md 就是用它們寫的。
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# 掃哪些：人寫的、會被下一個 session 當成指令讀的文字檔。
PATTERNS = ("*.md", "*.bat", "*.cmd", "*.py", "*.ps1", "*.vbs")
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".pytest_cache",
             ".next", "raw_data", "data", ".cache", ".data_cache",
             "venv", ".venv", "site-packages"}

ALLOWED = {0x09, 0x0A}          # TAB、LF
NAMES = {0x00: "NUL", 0x07: r"BEL（\a 被吃掉）", 0x08: r"BS（\b）",
         0x0B: r"VT（\v 被吃掉）", 0x0C: r"FF（\f 被吃掉）",
         0x0D: r"CR（\r 被吃掉，而且不是行尾）", 0x1A: "SUB", 0x1B: "ESC",
         0x7F: "DEL"}


def _walk():
    for pat in PATTERNS:
        for p in ROOT.rglob(pat):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            yield p


def _offenders(data: bytes):
    """回傳 [(行號, 位元組, 上下文)]。CRLF 的 CR 不算。"""
    out = []
    for i, c in enumerate(data):
        if c in ALLOWED:
            continue
        if c == 0x0D and i + 1 < len(data) and data[i + 1] == 0x0A:
            continue
        if c < 0x20 or c == 0x7F:
            ctx = data[max(0, i - 30):i + 24].decode("utf-8", "replace")
            out.append((data[:i].count(b"\n") + 1, c, ctx.replace("\n", "\\n")))
    return out


FILES = sorted(_walk())


def test_repo_has_text_files_to_scan():
    """反向自曝：掃到 0 個檔案時這支測試等於不存在（2026-08-26 的形狀）。"""
    assert len(FILES) > 50, f"只掃到 {len(FILES)} 個檔，SKIP_DIRS 可能過寬"


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_inline_control_characters(path: Path):
    try:
        data = path.read_bytes()
    except OSError as e:                      # 連結斷了之類，不是本關要管的
        pytest.skip(f"讀不到：{e}")
    bad = _offenders(data)
    if bad:
        lines = [
            "%s L%d  %s" % (path.relative_to(ROOT), ln, NAMES.get(c, hex(c)))
            + "\n        …%s…" % ctx
            for ln, c, ctx in bad[:6]
        ]
        raise AssertionError(
            "行內控制字元 %d 個 —— 反斜線路徑被轉義層吃掉的指紋。\n%s\n"
            "修法：把那段內容寫成獨立的 .py 檔再執行（不要穿過 bash 的 "
            "heredoc 或 python -c），反斜線用 chr(92) 組。"
            "詳見 mistake.md 2026-08-20 / 2026-09-06 / 2026-09-12。"
            % (len(bad), "\n".join(lines))
        )
