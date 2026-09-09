# -*- coding: utf-8 -*-
"""conj_watch 的 DB 讀取路徑守衛 —— 游標回 dict，不是 tuple

2026-09-09：`conj_events_live` 從註冊以來 **0 列**。原因是 `recent_ok` 寫
`r[0]` 取 `fetchone()`，而 `shared.db.get_db_conn()` 用的是 **DictCursor**
（`shared/db.py:165`），於是 `KeyError: 0`。

這個 bug 只在**偵測到事件的那一分鐘**才會執行到 —— 交會事件約每幣每 2.5 天
一次，所以它平常根本不跑；而它炸掉時 `events` 還沒被 append，summary 印的是
`events=0`，跟「這分鐘沒事件」**完全同形**。真事件一來就連炸 15 分鐘
（EMIT_RECENT 窗），然後安靜下來，看起來像什麼都沒發生。

同族：mistake.md 2026-08-26「只在清單非空時才看得見的輸出」、
2026-09-09「進場手續費 21 筆全是 0」——機制存在，但它在該作用的那一刻不在場。
`conj_watch_inject.py` 抓不到這個，因為它的檔頭寫明**不寫 DB**。

D1  `_one` 對 dict / tuple / None 都給同一個答案。
D2  `recent_ok` 吃 DictCursor 不炸，且語意正確（有近期事件 -> False）。
D3  `intent_gate` 吃 DictCursor 不炸，且上限照樣生效。
D4  **結構性**：這個檔案裡不得再出現對游標結果的位置索引
    （`fetchone()[0]` / `dict(cur.fetchall())`），否則同一個坑會從別的
    函式回來。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
POC = HERE.parent
sys.path.insert(0, str(POC))
sys.path.insert(0, str(POC.parents[0] / "sweep_failure"))
import conj_watch as cw  # noqa: E402


class _DictCursor:
    """忠實的 DictCursor 替身：fetchone 回 dict、fetchall 回 list[dict]。

    刻意**不用** MagicMock —— MagicMock 對 `r[0]` 會自動生出一個值，
    正好遮住這個 bug（mistake.md 2026-06-17 的教訓：test double 要嚴格）。
    """

    def __init__(self, rows):
        self._rows, self._last = rows, []

    def execute(self, sql, args=None):
        for pat, rows in self._rows.items():
            if pat in sql:
                self._last = rows
                return
        self._last = []

    def fetchone(self):
        return self._last[0] if self._last else None

    def fetchall(self):
        return list(self._last)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Conn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _DictCursor(self._rows)

    def commit(self):
        pass


def test_d1_one_handles_dict_tuple_and_none():
    assert cw._one({"MAX(event_ts)": 123}) == 123
    assert cw._one((123,)) == 123
    assert cw._one(None) is None
    assert cw._one({}) is None
    assert cw._one({"MAX(event_ts)": None}) is None


def test_d2_recent_ok_with_dict_cursor():
    # 沒有近期事件 -> 可以發
    conn = _Conn({"FROM conj_events_live": [{"MAX(event_ts)": None}]})
    assert cw.recent_ok(conn, "ETH-USD", 1_788_929_040_000) is True
    # 有近期事件 -> 冷卻擋下
    conn = _Conn({"FROM conj_events_live":
                  [{"MAX(event_ts)": 1_788_929_000_000}]})
    assert cw.recent_ok(conn, "ETH-USD", 1_788_929_040_000) is False


def test_d3_intent_gate_with_dict_cursor():
    intents = [{"canonical_symbol": f"{s}-USD"} for s in ("BTC", "ETH", "SOL")]
    # 全空 -> 三筆都過（MAX_CONCURRENT 是 3）
    conn = _Conn({"FROM conj_intents i": [], "COUNT(*) AS n FROM conj_intents":
                  [{"n": 0}]})
    assert len(cw.intent_gate(conn, intents)) == min(3, cw.MAX_CONCURRENT)
    # 已有 BTC 部位 -> BTC 被單幣上限擋掉
    conn = _Conn({"FROM conj_intents i": [{"sym": "BTC-USD", "n": 1}],
                  "COUNT(*) AS n FROM conj_intents": [{"n": 0}]})
    kept = cw.intent_gate(conn, intents)
    assert all(k["canonical_symbol"] != "BTC-USD" for k in kept)
    # 當日上限已滿 -> 一筆都不發
    conn = _Conn({"FROM conj_intents i": [],
                  "COUNT(*) AS n FROM conj_intents": [{"n": cw.MAX_DAILY}]})
    assert cw.intent_gate(conn, intents) == []


def test_d4_no_positional_indexing_on_cursor_results():
    """結構性守衛：位置索引不得再出現在這個檔案。"""
    src = (POC / "conj_watch.py").read_text(encoding="utf-8")
    bad = []
    for pat, why in (
        (r"fetchone\(\)\s*\[", "fetchone()[...] —— DictCursor 會 KeyError"),
        (r"fetchall\(\)\s*\[\s*\d", "fetchall()[n] —— 位置索引"),
        (r"dict\(\s*cur\.fetchall\(\)", "dict(cur.fetchall()) —— 元素是 dict"),
        (r"or\s*\[\s*0\s*\]\s*\)\s*\[\s*0\s*\]", "(... or [0])[0] 這個慣用法"),
    ):
        if re.search(pat, src):
            bad.append(why)
    assert not bad, "conj_watch.py 又出現對游標結果的位置索引：" + "; ".join(bad)
