# -*- coding: utf-8 -*-
"""回測檢視圖的 JS 真的跑得起來嗎 —— 結構性守衛

**為什麼需要這支**（2026-09-08，使用者回報「獵取的回測圖沒有顯示」）

`backtest_chart.py` 產出的頁面裡，`kpis()` 讀 `grp`，而 `let grp` 宣告在
第一次呼叫 `kpis()` 的 80 行之後 -> `ReferenceError: Cannot access 'grp'
before initialization`（let/const 的暫時性死區）。那個例外把整段 script
打斷，底下的 `LightweightCharts.createChart` **從來沒被執行** ->
**整張圖一片空白**。

而頁面其他部分（控制項、圖例、判決說明）照常渲染，所以它看起來像
「這段期間沒有資料」而不是「壞了」。查的時候我用 curl 驗了四輪都正常
（端點 200、內容 280KB、標頭乾淨、父頁沒 CSP）——**這一整類錯只有真的
把 JS 執行起來才看得到**，靜態檢查與 HTTP 狀態碼對它完全免疫。

同族：mistake.md 2026-04-22（補丁引用了還沒定義的名字，import 檢查抓不到、
只有跑到那一行才炸）、2026-08-01（裸 except 把「功能死掉」偽裝成「沒有內容」）。

**判準**
    J1  產出的 HTML 裡，每個被 `kpis()` 讀到的頂層 `let`/`const` 狀態變數，
        宣告位置必須早於第一次 `kpis()` 呼叫。
    J2  用最小瀏覽器替身把整段 script 在 node 執行一次：
        不得出現 TDZ / ReferenceError，且 **createChart 必須被呼叫到**。
        （J2 反向證明過：把宣告搬回原位 -> TDZ 重現、createChart 跑不到。）

node 不在時 J2 跳過並印出原因——**不是靜默通過**。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SF = HERE.parent
OUT = SF.parents[0] / "results" / "backtest_BTC.html"

# 最小瀏覽器替身：只要能跑到 createChart 就代表沒有 TDZ 打斷 script
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
let __made=false;
globalThis.LightweightCharts={createChart:()=>{__made=true;return{
  addCandlestickSeries:()=>({setData(){},setMarkers(){},createPriceLine(){},
    applyOptions(){}}),
  addLineSeries:()=>({setData(){},applyOptions(){}}),
  timeScale:()=>({fitContent(){},setVisibleRange(){},applyOptions(){}}),
  applyOptions(){},subscribeCrosshairMove(){},resize(){}};}};
process.on('exit',()=>console.log('__CHART_CREATED__='+__made));
"""


def _html() -> str:
    if not OUT.exists():
        r = subprocess.run([sys.executable, str(SF / "backtest_chart.py"),
                            "--symbol", "BTC"],
                           capture_output=True, text=True, timeout=1800,
                           cwd=str(SF))
        if not OUT.exists():
            pytest.skip(f"產不出 HTML：{(r.stderr or '')[-200:]}")
    return OUT.read_text(encoding="utf-8")


def _script(h: str) -> str:
    m = re.search(r"<script>\s*(const D = .*?)</script>", h, re.S)
    assert m, "找不到主 script 區塊"
    return m.group(1)


def test_state_declared_before_first_kpis_call():
    """J1 宣告必須早於呼叫（純文字檢查，不需要 node）。"""
    js = _script(_html())
    call = js.find("kpis();")
    assert call > 0, "找不到 kpis() 的呼叫"
    for name in ("grp", "filt"):
        m = re.search(rf"^\s*(?:let|const|var)\s+{name}\b", js, re.M)
        assert m, f"找不到 {name} 的宣告"
        assert m.start() < call, (
            f"`{name}` 宣告在第一次 kpis() 呼叫之後 -> 暫時性死區，"
            f"整段 script 會被例外打斷、圖畫不出來")


def test_script_runs_and_creates_chart():
    """J2 真的執行一次——這一關才抓得到 TDZ（反向證明過）。"""
    if not shutil.which("node"):
        pytest.skip("node 不在 PATH，J2 跳過（不是通過）")
    js = _script(_html())
    f = tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False,
                                    encoding="utf-8")
    f.write(STUB + js)
    f.close()
    try:
        r = subprocess.run(["node", f.name], capture_output=True, text=True,
                           timeout=120)
    finally:
        os.unlink(f.name)
    out = (r.stdout or "") + (r.stderr or "")
    assert "before initialization" not in out, (
        "暫時性死區錯誤：有 let/const 在宣告前被讀到\n" + out[-600:])
    assert "ReferenceError" not in out, "ReferenceError\n" + out[-600:]
    assert "__CHART_CREATED__=true" in out, (
        "createChart 沒有被呼叫到 —— script 在到達建圖那一行之前就死了。"
        "頁面會渲染成一片空白而不是報錯。\n" + out[-600:])
