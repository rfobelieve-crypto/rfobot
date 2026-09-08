# -*- coding: utf-8 -*-
"""交會事件回測檢視器的守衛 —— 圖上畫的必須就是被計分的那一筆

P1  已知答案對照：九幣池化毛利必須重現 `flow_direction.py` P 臂的數字
    （`data/results/flow_direction.json`，同一條規則的另一份實作）。
    容差 0.005 ATR：兩份跑在同一份資料上時差是 0；資料日更後母體會多幾筆，
    容差吃的是那個，不是實作差異。
P2  逐筆反解：R == 方向 × (出場 − 進場) / ATR，逐位元組（停損筆的出場價
    就是停損價，所以同一條式子對兩種出場都成立）。
P3  停損筆的出場必須在進場之後、60 分之內，且那一根確實碰到了停損價；
    時間出場筆的持有必須恰好 60 分。
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
import pytest

HERE = Path(__file__).resolve().parent
POC = HERE.parent
sys.path.insert(0, str(POC))
import conj_backtest as cb  # noqa: E402

# 2026-09-09：參考值換成 `conj_redef.py` 的**誠實**定義（進場錨在事件
# 成立時刻）。原本對的是 `flow_direction.py` 的 +0.2275，而那是舊錨點
# （群內最早那一分鐘）——TODO §1.03b 判定它是前視：22.3% 的交易在進場
# 當下事件還不存在，一半的 edge 由此而來。**換參考值不是放寬容差**，
# 是比對的對象本來就該是誠實那個。
REF = POC / "data" / "results" / "conj_redef.json"
LOOKAHEAD_REF = 0.2286        # 舊錨點的值，必須**明顯偏離**它
OUT = POC.parents[0] / "results" / "conj_backtest_BTC.html"
TOL = 0.005


@pytest.fixture(scope="module")
def ledgers():
    if not (cb.BARS / "BTC.parquet").exists():
        pytest.skip("research/poc/data 不在這台機器上")
    return {s: cb.ledger(s)[0] for s in cb.CORE9}


def test_p1_pooled_mean_matches_honest_redef(ledgers):
    """P1 池化毛利 == conj_redef 誠實定義的同一格（delay=DELAY）。"""
    if not REF.exists():
        pytest.skip("conj_redef.json 不在，先跑 conj_redef.py")
    j = json.loads(REF.read_text(encoding="utf-8"))
    ref = j["delays"][str(cb.DELAY)]["gross"]
    rs = [t["R"] for s in cb.CORE9 for t in ledgers[s]]
    m = float(np.mean(rs))
    assert abs(m - ref) < TOL, (
        f"池化毛利 {m:+.4f} vs conj_redef delay={cb.DELAY} {ref:+.4f}"
        f"（n {len(rs)} vs {j['n']}）—— 兩份實作不同意，不得上網站")


def test_p1b_not_the_lookahead_anchor(ledgers):
    """P1b 反向：進場若改回群內最早那一分鐘（前視），這一關必須紅。

    TODO §1.03b：舊錨點的池化毛利是 +0.2286，誠實的是 +0.1157 —— 差
    0.1129，遠大於容差。任何把錨點改回 `a` 的改動都會落回前視值。
    """
    rs = [t["R"] for s in cb.CORE9 for t in ledgers[s]]
    m = float(np.mean(rs))
    assert abs(m - LOOKAHEAD_REF) > 10 * TOL, (
        f"池化毛利 {m:+.4f} 落在前視值 {LOOKAHEAD_REF:+.4f} 上 —— "
        f"進場錨點被改回群內最早那一分鐘了（22.3% 的單會下在事件成立之前）")


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
