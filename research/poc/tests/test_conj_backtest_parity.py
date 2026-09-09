# -*- coding: utf-8 -*-
"""交會事件回測檢視器的守衛 —— 圖上畫的必須就是被計分的那一筆

P1  **獨立重寫對照**：測試裡自己跑一支逐筆 for 迴圈，用 conj_backtest 的
    常數重算 A 臂與 C 臂，與主引擎逐位比對（差 < 1e-9）。
    不對寫死的數字 —— 那種守衛會因為正當的參數改動而變紅，然後被調鬆。
P1b C 臂必須真的是「等它走出來」：筆數少於 A，且進場一律在成立後 C_WAIT+1 分。
P2  逐筆反解：R == 方向 × (出場 − 進場) / ATR（停損筆的出場價就是停損價，
    所以同一條式子對兩種出場都成立）。
P3  停損筆的出場在進場之後、HOLD 分之內；時間出場筆的持有恰好 HOLD 分。
J1/J2  產出的頁面：篩選狀態宣告早於 kpis() 呼叫；用瀏覽器替身把 script
    跑一次，createChart 必須被呼叫到（沿用 sweep_failure 那支守衛的替身；
    mistake.md 2026-09-08：curl 對「圖一片空白」免疫）。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
POC = HERE.parent
sys.path.insert(0, str(POC))
import conj_backtest as cb  # noqa: E402

# 2026-09-09（二次）：P1 不再對一個寫死的數字，改成**測試裡自己跑一支獨立
# 的逐筆 for 迴圈**，用 conj_backtest 自己的常數重算 A 臂並逐位比對。
# 理由：先前 P1 對 `conj_redef` 的 delay=2 毛利，但那個值是用 STOP=1.0 算的；
# 出場參數一改（3 ATR／480 分）它就必然對不上 —— 一個會因為正當改動而變紅
# 的守衛，下一個人只會把它調鬆。獨立實作對照沒有這個問題，而且它測的正是
# 「兩份實作同不同意」，比對一個數字更接近本意。
OUT = POC.parents[0] / "results" / "conj_backtest_BTC.html"


def _naive(sym, arm):
    """獨立重寫：逐分鐘 for 迴圈，不用向量化，語意照規則書。"""
    import event_census as ec
    import conj_redef as cr
    liq = cb._empty_liq()
    cand, ts, cl, at, _ = cb.ck.frozen_cand(sym, liq)
    b = pd.read_parquet(cb.BARS / f"{sym}.parquet",
                        columns=["open", "high", "low"])
    op = b["open"].to_numpy(float)
    hi = b["high"].to_numpy(float)
    lo = b["low"].to_numpy(float)
    n = len(ts)
    pairs = [(int(m), "sweep")
             for m in ec.cooldown_filter(np.sort(cand["sweep"]))]
    for nm in cb.FLOW:
        v = cand.get(nm)
        if v is not None and len(v):
            for m in ec.cooldown_filter(np.sort(v)):
                pairs.append((int(m), nm))
    out = []
    for a, mem in cr.groups_with_members(pairs):
        sg = {x for _, x in mem}
        if "sweep" not in sg or not (sg & set(cb.FLOW)):
            continue
        rd = max(min(m for m, x in mem if x == "sweep"),
                 min(m for m, x in mem if x in cb.FLOW))
        if rd < cb.W or rd + max(cb.DELAY, cb.C_WAIT + 1) + cb.HOLD >= n:
            continue
        A = at[rd]
        if not (A > 0) or not np.isfinite(A):
            continue
        if arm == "C":
            mv = (cl[rd + cb.C_WAIT] - cl[rd]) / A
            if abs(mv) < cb.C_MOVE:
                continue
            d = 1.0 if mv > 0 else -1.0
            j0 = rd + cb.C_WAIT + 1
        else:
            d = 1.0 if cl[rd] > cl[rd - cb.W] else (-1.0 if cl[rd] < cl[rd - cb.W] else 1.0)
            j0 = rd + cb.DELAY
        ent = op[j0]
        sp = ent - d * cb.STOP * A
        hit = None
        for k in range(j0 + 1, j0 + cb.HOLD + 1):
            if (lo[k] <= sp) if d > 0 else (hi[k] >= sp):
                hit = k
                break
        out.append(-cb.STOP if hit is not None
                   else float(d * (cl[j0 + cb.HOLD] - ent) / A))
    return out


@pytest.fixture(scope="module")
def ledgers():
    if not (cb.BARS / "BTC.parquet").exists():
        pytest.skip("research/poc/data 不在這台機器上")
    return {s: cb.ledger(s)[0] for s in cb.CORE9}


def test_p1_independent_reimplementation(ledgers):
    """P1 主引擎 vs 測試裡的獨立逐筆實作，逐位比對（兩臂都比）。"""
    for arm in ("A", "C"):
        for s in ("BTC", "ETH"):
            main = [t["R"] for t in cb.ledger(s, arm=arm)[0]]
            nv = _naive(s, arm)
            assert len(main) == len(nv), (arm, s, len(main), len(nv))
            d = max(abs(a - b) for a, b in zip(main, nv)) if main else 0.0
            assert d < 1e-9, f"{arm}/{s} 兩份實作不同意，最大差 {d:.2e}"


def test_p1b_arm_c_is_a_real_subset(ledgers):
    """P1b C 臂必須是「等它走出來」而不是別的東西：
    筆數要明顯少於 A（只有走出門檻的才進），且進場一律晚於 A。"""
    for s in ("BTC", "ETH", "SOL"):
        a = cb.ledger(s, arm="A")[0]
        c = cb.ledger(s, arm="C")[0]
        assert 0 < len(c) < len(a), (s, len(a), len(c))
        for t in c:
            gap = (t["entry_ts"] - t["anchor_ts"]) // 60_000
            assert gap == cb.C_WAIT + 1, (s, gap)


def test_p2_r_solves_back_from_prices(ledgers):
    for s in cb.CORE9:
        for t in ledgers[s]:
            d = 1.0 if t["side"] == "LONG" else -1.0
            r = d * (t["exit_px"] - t["entry"]) / t["atr"]
            assert abs(r - t["R"]) < 1e-9, (s, t["anchor_ts"], r, t["R"])
            assert abs(t["stop"] - (t["entry"] - d * cb.STOP * t["atr"])) < 1e-9


def test_p3_exit_timing(ledgers):
    for s in cb.CORE9:
        for t in ledgers[s]:
            held = (t["exit_ts"] - t["entry_ts"]) // 60_000
            assert held == t["held"]
            if t["stopped"]:
                assert 1 <= held <= cb.HOLD and t["R"] == -cb.STOP
                assert t["exit_px"] == t["stop"]
            else:
                assert held == cb.HOLD


# ---- 頁面本身 ----
STUB = """
const __el={innerHTML:'',style:{},addEventListener(){},appendChild(){},
  getBoundingClientRect:()=>({width:800,height:400}),
  classList:{add(){},remove(){},toggle(){}}};
globalThis.document={getElementById:()=>__el,querySelectorAll:()=>[],
  querySelector:()=>__el,createElement:()=>__el,addEventListener(){},
  body:__el,documentElement:__el};
globalThis.window={addEventListener(){},innerHeight:800,innerWidth:1200,
  devicePixelRatio:1};
globalThis.location={search:''};
globalThis.ResizeObserver=class{observe(){}};
let __made=false;
globalThis.LightweightCharts={createChart:()=>{__made=true;return{
  addCandlestickSeries:()=>({setData(){},setMarkers(){},createPriceLine(){},
    removePriceLine(){},applyOptions(){}}),
  addLineSeries:()=>({setData(){},applyOptions(){}}),
  timeScale:()=>({fitContent(){},setVisibleRange(){__focused=true;},applyOptions(){},
    subscribeVisibleLogicalRangeChange(){},setVisibleLogicalRange(){}}),
  applyOptions(){},subscribeCrosshairMove(){},resize(){}};}};
process.on('exit',()=>{console.log('__CHART_CREATED__='+__made);
  console.log('__FOCUSED__='+__focused);});
"""
STUB = STUB.replace("let __made=false;", "let __made=false;let __focused=false;")


def _html() -> str:
    if not OUT.exists():
        if not (cb.BARS / "BTC.parquet").exists():
            pytest.skip("research/poc/data 不在這台機器上")
        subprocess.run([sys.executable, str(POC / "conj_backtest.py"),
                        "--symbol", "BTC"], capture_output=True, text=True,
                       timeout=600, cwd=str(POC.parents[1]))
        if not OUT.exists():
            pytest.skip("產不出 HTML")
    return OUT.read_text(encoding="utf-8")


def _script(h: str) -> str:
    m = re.search(r"<script>\s*(const D = .*?)</script>", h, re.S)
    assert m, "找不到主 script 區塊"
    return m.group(1)


def test_j1_state_declared_before_first_kpis_call():
    js = _script(_html())
    call = js.find("kpis();")
    assert call > 0
    for name in ("grp", "filt"):
        m = re.search(rf"^\s*(?:let|const|var)\s+{name}\b", js, re.M)
        assert m and m.start() < call, f"`{name}` 宣告在第一次 kpis() 之後（暫時性死區）"


def test_j2_script_runs_and_creates_chart():
    if not shutil.which("node"):
        pytest.skip("node 不在 PATH，J2 跳過（不是通過）")
    js = _script(_html())
    f = tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8")
    f.write(STUB + js)
    f.close()
    try:
        r = subprocess.run(["node", f.name], capture_output=True, text=True, timeout=120)
    finally:
        os.unlink(f.name)
    out = (r.stdout or "") + (r.stderr or "")
    assert "before initialization" not in out, out[-600:]
    assert "ReferenceError" not in out, out[-600:]
    assert "__CHART_CREATED__=true" in out, "createChart 沒被呼叫到\n" + out[-600:]
    # J3（2026-09-09，使用者：「回測的進出場跟我看的也差很多」）
    # 開頁必須聚焦到一筆交易，不得停在全景：90 天 = 25,921 根 5 分 K，
    # 一根 0.05 像素，而一筆交易 62 分鐘 = 12 根 K = 0.67 像素 —— 掃單／
    # 進場／出場疊成一個點，看起來像位置畫錯了。計算層當時逐筆對回原始
    # 1 分鐘 bar 全部吻合，錯的是預設視野。
    assert "__FOCUSED__=true" in out, (
        "開頁沒有 setVisibleRange —— 預設停在全景，一筆交易只有 0.67 像素寬，"
        "三個標記會疊成一個點\n" + out[-600:])


def test_j4_no_operator_quotes_in_public_page():
    """J4（2026-09-09）：HTML 模板裡的註解會**原樣印進公開頁面的原始碼**。

    實際發生過：`kpis()` 裡一段解釋大小字為什麼對調的註解，逐字引用了操作者
    在對話裡說的話，跟著九個頁面一起發到公開端點。註解寫在 Python 那一側
    不會有這個問題，寫在模板字串裡就會。

    這道守衛只擋**最明確的那一類**（引用操作者、內部對話），不擋
    `mistake.md` / `TODO §` 這種文件指標 —— 後者是刻意公開的判決出處。
    """
    h = _html()
    bad = [p for p in ("使用者：「", "使用者:「", "使用者說", "他說「", "原話")
           if p in h]
    assert not bad, (
        "公開頁面的原始碼裡出現操作者引述：" + "、".join(bad) +
        "。這類說明要寫在產生器的 Python 註解裡，不要寫進 HTML 模板。")
