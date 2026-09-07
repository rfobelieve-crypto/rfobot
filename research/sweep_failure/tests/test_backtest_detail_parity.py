# -*- coding: utf-8 -*-
"""凍結引擎的 detail 分支不得改變任何一筆交易。

2026-09-07 為了讓回測**看得見**（圖上畫的就是被計分的那一筆），
`backtest_symbol` 多了一個 `detail=True` 分支。這支測試釘住兩件事：

  1. `detail=False` 的輸出與**重構之前**逐位元組相同
     （基準 sha256 在 SHA_BASELINE，九幣 7,083 筆，2026-09-07 存檔）
  2. `detail=True` 投影回 tuple 之後與 `detail=False` 完全相同
     —— 保證圖與回測不可能是兩份實作
  3. exit_px 與 R 在構造上一致：d*(exit_px-entry)/risk == R

反向證明過（2026-09-07）：把 detail 分支的 entry 改成 lvl（拿掉滑價），
第 3 條立刻紅並指名該筆。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sweep_core as sc  # noqa: E402

CACHE = HERE.parent / ".cache"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
SHA_BASELINE = "86ad51a77ee883f7b40e77303efd024e"     # 前 32 碼，見 docstring


def _bars(sym):
    p = CACHE / f"{sym}USDT_1h.csv"
    if not p.exists():
        pytest.skip(f"{p.name} not in cache")
    return sc.load_csv(str(p))


def _project(det):
    """把 detail 記錄投影回凍結的 tuple 形狀。"""
    return [(x["fill_ts"], x["exit_ts"], x["R"], x["level"], x["atr"],
             x["stopped"], x["pierce"], x["side"]) for x in det]


def test_tuple_output_matches_frozen_baseline():
    out = {s: sc.backtest_symbol(_bars(s)) for s in CORE9}
    h = hashlib.sha256(
        json.dumps(out, default=float, sort_keys=True).encode()).hexdigest()
    n = sum(len(v) for v in out.values())
    assert n == 7083, f"trade count drifted: {n} != 7083"
    assert h[:32] == SHA_BASELINE, (
        f"frozen engine output changed: {h[:32]} != {SHA_BASELINE}")


def test_detail_projects_to_the_same_trades():
    for s in CORE9:
        b = _bars(s)
        assert _project(sc.backtest_symbol(b, detail=True)) == \
            sc.backtest_symbol(b), f"{s}: detail projection differs"


def test_exit_price_is_consistent_with_the_scored_R():
    for s in CORE9:
        for t in sc.backtest_symbol(_bars(s), detail=True):
            r = t["d"] * (t["exit_px"] - t["entry"]) / t["risk"]
            assert abs(r - t["R"]) < 1e-9, (
                f"{s} fill_ts={t['fill_ts']}: drawn exit implies R={r:.6f} "
                f"but the backtest scored {t['R']:.6f}")


def test_entry_pays_slippage_against_us():
    for s in CORE9:
        for t in sc.backtest_symbol(_bars(s), detail=True):
            want = t["level"] + t["d"] * sc.SLIP * t["atr"]
            assert abs(t["entry"] - want) < 1e-12, f"{s}: entry slip dropped"
