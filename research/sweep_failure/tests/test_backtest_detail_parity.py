# -*- coding: utf-8 -*-
"""凍結引擎的 detail 分支不得改變任何一筆交易。

2026-09-07 為了讓回測**看得見**（圖上畫的就是被計分的那一筆），
`backtest_symbol` 多了一個 `detail=True` 分支。這支測試釘住兩件事：

  1. `detail=False` 的輸出與**重構之前**逐位元組相同
     （基準 sha256 在 SHA_BASELINE，跑在 tests/fixtures 的**凍結切片**上）

     **2026-09-10 修**：原本這一關釘的是 `.cache` 上的「九幣 7,083 筆」。
     但 `.cache` 是 `fetch_klines.py` 抓的滾動 930 天窗（起點 =
     `now - days*86400`），每次刷新都從頭部丟掉舊 bar —— 所以這個釘樁
     在下一次刷新就必紅，而且紅的原因跟引擎的算術無關。實測它已經紅成
     7,064。一條永遠紅的守衛跟壞掉的守衛一樣沒用（mistake.md 2026-09-03）。
     現在基準跑在 `tests/fixtures/*_frozen.csv`（固定區間、進 git），
     量的是引擎本身而不是資料窗滾到哪裡。
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
FIX = HERE / "fixtures"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
# 凍結切片（2024-07-01 ~ 2025-07-01、BTC 與 DOGE 兩種價格尺度）上的基準。
# 重建方式見 tests/make_fixture.py；**不得因為測試紅了就重算這兩個值**，
# 紅了代表引擎的算術變了，那才是這一關存在的理由。
FIXTURE_SYMS = ["BTC", "DOGE"]
FIXTURE_N = 590
SHA_BASELINE = "e225522d7e4c9d01d09752a370ec5f0f"     # 前 32 碼，見 docstring


def _bars(sym):
    p = CACHE / f"{sym}USDT_1h.csv"
    if not p.exists():
        pytest.skip(f"{p.name} not in cache")
    return sc.load_csv(str(p))


def _project(det):
    """把 detail 記錄投影回凍結的 tuple 形狀。"""
    return [(x["fill_ts"], x["exit_ts"], x["R"], x["level"], x["atr"],
             x["stopped"], x["pierce"], x["side"]) for x in det]


def _frozen_bars(sym):
    p = FIX / f"{sym}USDT_1h_frozen.csv"
    if not p.exists():
        pytest.fail(f"{p.name} missing — run tests/make_fixture.py")
    return sc.load_csv(str(p))


def test_tuple_output_matches_frozen_baseline():
    """凍結切片上的輸出必須逐位元組相同（與 .cache 的滾動窗脫鉤）。"""
    out = {s: sc.backtest_symbol(_frozen_bars(s)) for s in FIXTURE_SYMS}
    h = hashlib.sha256(
        json.dumps(out, default=float, sort_keys=True).encode()).hexdigest()
    n = sum(len(v) for v in out.values())
    assert n == FIXTURE_N, f"trade count drifted: {n} != {FIXTURE_N}"
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
