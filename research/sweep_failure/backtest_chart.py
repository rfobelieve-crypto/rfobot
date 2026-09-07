# -*- coding: utf-8 -*-
"""回測檢視器 —— 把凍結引擎的每一筆交易畫在 K 線上，供肉眼驗證位置。

使用者 2026-09-07：
    「類似 TradingView 圖表這些我們都有，就只是要進化成回測系統」
    「這樣我們的每次研究我才能確保進出場是不是有在正確的位置」

所以這支的唯一職責是**可稽核性**：圖上畫的每一條線、每一個標記，都直接
取自 `sweep_core.backtest_symbol(bars, detail=True)` 的同一筆記錄 ——
不重算、不近似、不另寫一份繪圖用的邏輯。

    為什麼這件事需要被強制：`shadow_review.py` 是**前瞻覆盤**（讀 shadow
    CSV、看最近 48 小時的活地圖）。它回答「現在盤面上有什麼」。它不回答
    「回測引擎在 2025-03-14 那一筆到底進在哪裡」。而後者才是「進出場位置
    對不對」的問題 —— 一份獨立的繪圖實作可以畫得很好看而且是錯的
    （mistake.md 2026-08-26：兩份實作會安靜地不同意）。

    強制手段是 `tests/test_backtest_detail_parity.py`：detail 投影回 tuple
    必須與凍結輸出逐位元組相同（九幣 7,083 筆），且 exit_px 必須從被計分的
    R 反解。反向證明過。

畫什麼（全部來自同一筆 detail 記錄）
    ┈┈  價位線     從**造出它的那根樞紐影線尖端**（origin）延伸到掃單那根
    ▽▲  掃單       價格穿過價位的那根 bar（j）
    ●   進場       回踩成交的那根 bar（fill），畫在 entry（含逆向滑價）
    ✕   出場       exitbar，畫在 exit_px（**從 R 反解**，與計分一致）
    ┄   停損線     選取某一筆時畫出它的 entry / stop / exit 三條水平線
    下窗 累積 R    以出場時刻為 x 軸

凍結參數一個字不動（PIVOT=10, W=8, HOLD=8, DIS=3.5, SLIP=0.05）。
本頁**只顯示回測**，不含任何前瞻樣本，也不是訊號來源。

用法
    python research/sweep_failure/backtest_chart.py --symbol BTC
    python research/sweep_failure/backtest_chart.py --symbol ETH --from 2026-01-01
    python research/sweep_failure/backtest_chart.py --all --last-days 180
出：research/results/backtest_{SYM}.html
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_core as sc  # noqa: E402

CACHE = HERE / ".cache"
OUT = HERE.parents[0] / "results"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
# 清算流分級表（research/poc/sweep_liq_filter.py 產生）。**預先算好再讀**，
# 不在這裡重算：偵測要載入分鐘 bar + OI 並跑滾動門檻，網頁路由的 110 秒等不起。
LIQ_GRADE = (HERE.parents[0] / "poc" / "data" / "results"
             / "sweep_liq_filter.parquet")


def _load_grade(sym):
    """{fill_ts(秒): 穿越分鐘 ±5 分內的清算流事件數}。缺檔就回空。"""
    try:
        import pandas as _pd
        if not LIQ_GRADE.exists():
            return {}
        g = _pd.read_parquet(LIQ_GRADE, columns=["sym", "fill_ts", "n_near"])
        g = g[g["sym"] == sym]
        return {int(a) // 1000: int(b)
                for a, b in zip(g["fill_ts"], g["n_near"])}
    except Exception as e:                      # 顯示層不可靜默失敗
        print(f"[WARN] liq grade unavailable: {e}")
        return {}


def to_day(ts_sec):
    return datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def ensure_bars(sym):
    """Railway 的映像沒有本機快取 —— 沿用 shadow_review 既有的抓取器，
    不另寫一份（同一份資料兩個抓法遲早會不同意，mistake.md 2026-08-01）。"""
    p = CACHE / f"{sym}USDT_1h.csv"
    try:
        from shadow_review import ensure_bars as _eb
        return _eb(sym)
    except Exception:
        if p.exists():
            return p
        raise


def load_bars(sym):
    bars = sc.load_csv(str(ensure_bars(sym)))
    if not bars:
        raise SystemExit(f"no bars for {sym}")
    return bars


def build(sym, t_from, t_to):
    """回測跑**全歷史**（ATR 與樞紐需要完整前文），之後才裁切顯示窗。"""
    bars = load_bars(sym)
    det = sc.backtest_symbol(bars, detail=True)
    grade = _load_grade(sym)

    lo = t_from or bars[0][0]
    hi = t_to or bars[-1][0]
    view = [b for b in bars if lo <= b[0] <= hi]
    if len(view) < 30:
        raise SystemExit(f"{sym}: 顯示窗只有 {len(view)} 根 bar，範圍給錯了？")

    # 顯示窗內的交易：以**進場**時刻為準（出場可能落在窗外，照樣畫）
    tr = [t for t in det if lo <= t["fill_ts"] <= hi]

    candles = [dict(time=int(b[0]), open=b[sc.O], high=b[sc.H],
                    low=b[sc.L], close=b[sc.C]) for b in view]

    trades, levels, markers = [], [], []
    # 累積 R 從顯示窗的第一筆起算 0，不把窗外的歷史算進來（否則曲線的水平
    # 位置在講另一段期間的事）
    cum = 0.0
    eq = []
    for t in tr:
        cum += t["R"]
        eq.append(dict(time=int(t["exit_ts"]), value=round(cum, 4)))

    for i, t in enumerate(tr):
        win = t["R"] > 0
        col = "#0ecb81" if win else "#f6465d"
        levels.append(dict(
            id=i,
            pts=[dict(time=int(t["origin_ts"]), value=t["level"]),
                 dict(time=int(t["sweep_ts"]), value=t["level"])],
            c="#f0b90b" if t["kind"] == "buy" else "#7b61ff"))
        markers += [
            dict(time=int(t["sweep_ts"]),
                 position="aboveBar" if t["kind"] == "buy" else "belowBar",
                 color="#f0b90b" if t["kind"] == "buy" else "#7b61ff",
                 shape="arrowDown" if t["kind"] == "buy" else "arrowUp",
                 text=""),
            dict(time=int(t["fill_ts"]),
                 position="belowBar" if t["side"] == "LONG" else "aboveBar",
                 color=col, shape="circle", text=f"{i + 1}"),
            dict(time=int(t["exit_ts"]),
                 position="aboveBar" if t["side"] == "LONG" else "belowBar",
                 color=col, shape="square",
                 text=f"{t['R']:+.2f}" + ("!" if t["stopped"] else "")),
        ]
        trades.append(dict(
            id=i, side=t["side"], kind=t["kind"],
            origin=to_day(t["origin_ts"]), sweep=to_day(t["sweep_ts"]),
            fill=to_day(t["fill_ts"]), exit=to_day(t["exit_ts"]),
            wait=t["wait"], held=t["held"],
            level=t["level"], entry=t["entry"], stop=t["stop"],
            exit_px=t["exit_px"], atr=t["atr"], risk=t["risk"],
            pierce=round(t["pierce"], 4), R=round(t["R"], 4),
            stopped=bool(t["stopped"]),
            t_fill=int(t["fill_ts"]), t_sweep=int(t["sweep_ts"]),
            t_exit=int(t["exit_ts"]), t_origin=int(t["origin_ts"]),
            n_flow=int(grade.get(int(t["fill_ts"]), -1))))

    m_view = sc.metrics([t["R"] for t in tr])
    m_all = sc.metrics([t["R"] for t in det])
    # 分組績效在**伺服器端**用同一個 sc.metrics 算好，JS 只負責切換顯示——
    # 在前端重寫一份指標就是第二份實作，遲早會跟計分器不同意
    # （mistake.md 2026-08-26）。
    def _grp(pred, src):
        rs = [x["R"] for x in src if pred(int(grade.get(int(x["fill_ts"]), -1)))]
        return sc.metrics(rs) if len(rs) >= 2 else None
    groups = {
        "all": dict(view=m_view, all=m_all),
        "n0": dict(view=_grp(lambda n: n == 0, tr),
                   all=_grp(lambda n: n == 0, det)),
        "n3": dict(view=_grp(lambda n: n >= 3, tr),
                   all=_grp(lambda n: n >= 3, det)),
    }
    return dict(sym=sym, candles=candles, levels=levels, markers=markers,
                trades=trades, equity=eq,
                stats_view=m_view, stats_all=m_all, groups=groups,
                has_grade=bool(grade),
                span=[to_day(view[0][0]), to_day(view[-1][0])],
                n_bars_all=len(bars),
                span_all=[to_day(bars[0][0]), to_day(bars[-1][0])],
                params=dict(PIVOT=sc.PIVOT, W=sc.W, HOLD=sc.HOLD,
                            DIS=sc.DIS, SLIP=sc.SLIP))


TPL = r"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__SYM__ 掃單失敗 · 回測檢視</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#0b0e11;--pan:#12161c;--line:#1e242d;--ink:#eaecef;--dim:#848e9c;
      --up:#0ecb81;--dn:#f6465d;--buy:#f0b90b;--sell:#7b61ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:13px/1.55 -apple-system,"Segoe UI","Noto Sans TC","Microsoft JhengHei",sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:16px 16px 40px;
      display:flex;flex-direction:column;gap:14px}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px 16px}
h1{margin:0;font-size:19px;font-weight:600;letter-spacing:.01em}
.tag{font-size:11px;color:var(--dim);border:1px solid var(--line);
     border-radius:3px;padding:2px 8px;font-variant-numeric:tabular-nums}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));gap:8px}
.kpi{background:var(--pan);border:1px solid var(--line);border-radius:4px;
     padding:9px 11px;display:flex;flex-direction:column;gap:2px}
.kpi b{font-size:17px;font-weight:600;font-variant-numeric:tabular-nums}
.kpi span{font-size:11px;color:var(--dim)}
.kpi small{font-size:10.5px;color:var(--dim);font-variant-numeric:tabular-nums}
#c{height:520px;border:1px solid var(--line);border-radius:4px;overflow:hidden}
#eq{height:150px;border:1px solid var(--line);border-radius:4px;overflow:hidden}
.bar{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;
     font-size:11.5px;color:var(--dim)}
.bar b{color:var(--ink);font-weight:500}
.sw{display:inline-block;width:16px;height:0;border-top:2px solid;
    vertical-align:middle;margin-right:5px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;
     vertical-align:middle;margin-right:5px}
.sq{display:inline-block;width:8px;height:8px;vertical-align:middle;margin-right:5px}
button{background:var(--pan);color:var(--ink);border:1px solid var(--line);
       border-radius:3px;padding:4px 10px;font:inherit;font-size:11.5px;cursor:pointer}
button:hover{border-color:#3b4552}
button.on{border-color:var(--buy);color:var(--buy)}
.tw{max-height:340px;overflow:auto;border:1px solid var(--line);border-radius:4px}
table{width:100%;border-collapse:collapse;font-size:11.5px;
      font-variant-numeric:tabular-nums}
th,td{padding:5px 9px;text-align:right;white-space:nowrap;
      border-bottom:1px solid var(--line)}
th{position:sticky;top:0;background:var(--pan);color:var(--dim);
   font-weight:500;text-align:right;z-index:1}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),
th:nth-child(3),td:nth-child(3){text-align:left}
tbody tr{cursor:pointer}
tbody tr:hover{background:#171d25}
tbody tr.sel{background:#1c2530}
.pos{color:var(--up)}.neg{color:var(--dn)}
.note{font-size:11.5px;color:var(--dim);max-width:96ch}
</style></head><body><div class="wrap">

<header>
  <h1>__SYM__USDT · 掃單失敗回測</h1>
  <span class="tag">__SPAN__</span>
  <span class="tag">PIVOT __PIVOT__ · W __W__ · HOLD __HOLD__ · 停損 __DIS__ ATR · 滑價 __SLIP__ ATR/邊</span>
  <span class="tag">凍結規則 · 純回測 · 非訊號</span>
</header>

<div class="kpis" id="kpis"></div>

<div class="bar">
  <span><span class="sw" style="border-color:var(--buy)"></span>買側流動性（掃它 → 做空）</span>
  <span><span class="sw" style="border-color:var(--sell)"></span>賣側流動性（掃它 → 做多）</span>
  <span><span class="dot" style="background:var(--up)"></span>進場（畫在含滑價的成交價）</span>
  <span><span class="sq" style="background:var(--dn)"></span>出場（畫在由 R 反解的價格）</span>
  <button id="btnAll" class="on">全部交易</button>
  <button id="btnWin">只看賺</button>
  <button id="btnLose">只看賠</button>
  <button id="btnClear">清除選取</button>
  <span style="width:100%"></span>
  <span style="color:var(--dim)">清算流分級：</span>
  <button id="btnG0">無清算流（0）</button>
  <button id="btnG3">有清算流（3+）</button>
  <button id="btnGA" class="on">不分</button>
  <span id="dense"></span>
</div>

<div id="c"></div>
<div id="eq"></div>
<div class="note" id="sel">點下方任一列 —— 圖表跳到那一筆，並畫出它的價位、進場、停損、出場四條線。</div>

<div class="tw"><table>
<thead><tr>
<th>#</th><th>方向</th><th>掃單時刻</th><th>等</th><th>持</th>
<th>價位</th><th>進場</th><th>停損</th><th>出場</th>
<th>穿透(ATR)</th><th>清算流</th><th>ATR</th><th>R</th><th>結束於</th>
</tr></thead><tbody id="tb"></tbody></table></div>

<div class="note">
每一列都直接來自凍結引擎 <code>backtest_symbol(detail=True)</code> 的同一筆記錄。
「出場」是從被計分的 R 反解出來的價格，所以圖上的位置與績效數字在構造上
不可能互相矛盾（<code>tests/test_backtest_detail_parity.py</code> 釘住，反向證明過）。
「等」＝掃單到成交的小時數（上限 W），「持」＝成交到出場的小時數（上限 HOLD）。
</div>

<div class="note">
<b>清算流分級（2026-09-07 驗證，<span class="neg">尚未改變任何規則</span>）</b>：
「清算流」＝穿越那一分鐘 ±5 分內，主動量極端／量能爆發／OI 崩落三種因果門檻
事件的個數，<b>嚴格早於成交</b>（成交在掃單 bar 之後的 1–8 根）。
事件層面上，純掃單在 5 分鐘是<b>反轉</b>（−0.033 ATR）、伴隨清算流是強烈<b>延續</b>
（+0.187，60 分鐘 +0.278）——而本策略是反轉策略。
但在引擎的<b>真實交易</b>上：0 個 <span class="pos">+0.0444 R</span>
[+0.0164,+0.0731]、9/9 幣；3+ 個 +0.0318 [−0.0024,+0.0685]、7/9 幣；
差值 +0.0126 <b>CI [−0.0333,+0.0575] 含零 → INCONCLUSIVE</b>。
差值的 MDE 是 0.045 R 而整條 edge 只有 0.037 R，所以這個檢定<b>分辨不出比
整個 edge 還小的差異</b>——是設計上做不出判決，不是「沒有效果」。
機制：<b>回踩與延續在構造上互斥</b>——成交率隨清算流事件數單調下降
（91.8% → 84.6%，九幣同向），延續最猛的價格一去不回，引擎根本沒有部位；
全體 12% 的掃單從未成交，從不出現在這張圖上。
</div>

</div>
<script>
const D = __DATA__;
const fmtP = v => v >= 1000 ? v.toFixed(1) : v >= 1 ? v.toFixed(3) : v.toFixed(5);

function kpis(){
  const G = (D.groups && D.groups[grp]) || {};
  const a = G.view || D.stats_view, b = G.all || D.stats_all;
  if(!a){document.getElementById('kpis').innerHTML =
    '<div class="kpi"><b>0</b><span>這段期間沒有交易</span></div>'; return;}
  const cell = (v,k,s,cls) => `<div class="kpi"><b class="${cls||''}">${v}</b>`+
    `<span>${k}</span><small>${s}</small></div>`;
  const g = x => x >= 0 ? 'pos' : 'neg';
  document.getElementById('kpis').innerHTML = [
    cell(a.n, '交易筆數', `全期 ${b.n}`),
    cell(a.exp.toFixed(4), '每筆 R', `全期 ${b.exp.toFixed(4)}`, g(a.exp)),
    cell(a.wr.toFixed(1)+'%', '勝率', `全期 ${b.wr.toFixed(1)}%`),
    cell(isFinite(a.pf)?a.pf.toFixed(2):'∞', '獲利因子', `全期 ${isFinite(b.pf)?b.pf.toFixed(2):'∞'}`),
    cell(a.mdd.toFixed(1)+'%', '最大回落', `全期 ${b.mdd.toFixed(1)}%`),
    cell(a.t.toFixed(2), 't 值（未聚類）', `全期 ${b.t.toFixed(2)}`),
  ].join('');
}
kpis();

const dark = {layout:{background:{color:'#0b0e11'},textColor:'#848e9c',fontSize:11},
  grid:{vertLines:{color:'#151a21'},horzLines:{color:'#151a21'}},
  rightPriceScale:{borderColor:'#1e242d'},
  timeScale:{timeVisible:true,secondsVisible:false,rightOffset:6,borderColor:'#1e242d'}};

const chart = LightweightCharts.createChart(document.getElementById('c'), dark);
const cs = chart.addCandlestickSeries({upColor:'#0ecb81',downColor:'#f6465d',
  borderVisible:false,wickUpColor:'#0ecb81',wickDownColor:'#f6465d'});
cs.setData(D.candles);
for(const L of D.levels){
  chart.addLineSeries({color:L.c,lineWidth:1,lineStyle:2,lastValueVisible:false,
    priceLineVisible:false,crosshairMarkerVisible:false}).setData(L.pts);
}

let filt = 'all';       // 賺賠
let grp  = 'all';       // 清算流分級
const keepWL = t => filt==='all' || (filt==='win' ? t.R>0 : t.R<=0);
const keepG  = t => grp==='all' || (grp==='n0' ? t.n_flow===0
                                               : t.n_flow>=3);
const keep = t => keepWL(t) && keepG(t);
function drawMarkers(){
  const vis = D.trades.filter(keep);
  const ids = new Set(vis.map(t=>t.id));
  // 標記文字在密集區會疊成一團 —— 超過 45 筆就只留形狀，細節看下表。
  const dense = vis.length > 45;
  cs.setMarkers(D.markers.filter((m,i)=>ids.has(Math.floor(i/3)))
    .map(m => dense ? Object.assign({}, m, {text:''}) : m));
  document.getElementById('dense').textContent =
    dense ? `顯示 ${vis.length} 筆 —— 標記文字已關閉（>45 筆會疊住）。點下表任一列看單筆。`
          : `顯示 ${vis.length} 筆`;
}
drawMarkers();

const eqc = LightweightCharts.createChart(document.getElementById('eq'),
  Object.assign({}, dark, {layout:{background:{color:'#0b0e11'},textColor:'#848e9c',fontSize:10}}));
eqc.addLineSeries({color:'#0ecb81',lineWidth:2,title:'累積 R',
  priceLineVisible:false,crosshairMarkerVisible:false}).setData(D.equity);
chart.timeScale().subscribeVisibleLogicalRangeChange(r=>{if(r)eqc.timeScale().setVisibleLogicalRange(r);});

let lines = [];
function clearLines(){for(const l of lines)cs.removePriceLine(l);lines=[];}
function focus(t){
  clearLines();
  const mk = (p,c,txt,st) => lines.push(cs.createPriceLine({price:p,color:c,
    lineWidth:1,lineStyle:st===undefined?2:st,axisLabelVisible:true,title:txt}));
  mk(t.level,'#848e9c','價位',0);
  mk(t.entry, t.R>0?'#0ecb81':'#f6465d','進場');
  mk(t.stop, '#f6465d','停損 '+D.params.DIS+'ATR',3);
  mk(t.exit_px,'#f0b90b','出場 '+t.R.toFixed(3)+'R');
  const span = t.t_exit - t.t_origin, pad = Math.max(span*0.4, 86400*2);
  chart.timeScale().setVisibleRange({from:t.t_origin-pad, to:t.t_exit+pad});
  document.getElementById('sel').innerHTML =
    `<b>#${t.id+1} ${t.side}</b> · 樞紐 ${t.origin} 造出價位 ${fmtP(t.level)}`+
    ` → ${t.sweep} 穿過 ${t.pierce.toFixed(3)} ATR`+
    ` → 等 ${t.wait} 小時回踩，${t.fill} 成交 ${fmtP(t.entry)}`+
    `（價位 ${fmtP(t.level)} 加 ${D.params.SLIP} ATR 逆向滑價）`+
    ` → 停損掛 ${fmtP(t.stop)} → ${t.exit} ${t.stopped?'觸及停損':'時間到'}`+
    ` 出在 ${fmtP(t.exit_px)}，計 <b class="${t.R>0?'pos':'neg'}">${t.R.toFixed(4)} R</b>`+
    `（1R = ${D.params.DIS} × ATR ${fmtP(t.atr)} = ${fmtP(t.risk)}）`;
  for(const tr of document.querySelectorAll('#tb tr')) tr.classList.remove('sel');
  const row = document.getElementById('r'+t.id); if(row) row.classList.add('sel');
}

function table(){
  document.getElementById('tb').innerHTML = D.trades.filter(keep).map(t=>
    `<tr id="r${t.id}"><td>${t.id+1}</td>`+
    `<td class="${t.side==='LONG'?'pos':'neg'}">${t.side}</td>`+
    `<td>${t.sweep}</td><td>${t.wait}</td><td>${t.held}</td>`+
    `<td>${fmtP(t.level)}</td><td>${fmtP(t.entry)}</td><td>${fmtP(t.stop)}</td>`+
    `<td>${fmtP(t.exit_px)}</td><td>${t.pierce.toFixed(3)}</td>`+
    `<td class="${t.n_flow<0?'':(t.n_flow===0?'pos':(t.n_flow>=3?'neg':''))}">`+
    `${t.n_flow<0?'—':t.n_flow}</td>`+
    `<td>${fmtP(t.atr)}</td>`+
    `<td class="${t.R>0?'pos':'neg'}">${t.R.toFixed(4)}</td>`+
    `<td>${t.stopped?'停損':'時間'}</td></tr>`).join('');
  for(const tr of document.querySelectorAll('#tb tr'))
    tr.onclick = () => focus(D.trades[+tr.id.slice(1)]);
}
table();

function setF(f, btn){
  filt = f;
  for(const b of ['btnAll','btnWin','btnLose'])
    document.getElementById(b).classList.toggle('on', b===btn);
  drawMarkers(); table();
}
btnAll.onclick=()=>setF('all','btnAll');
btnWin.onclick=()=>setF('win','btnWin');
btnLose.onclick=()=>setF('lose','btnLose');
function setG(g, btn){
  grp = g;
  for(const b of ['btnG0','btnG3','btnGA'])
    document.getElementById(b).classList.toggle('on', b===btn);
  kpis(); drawMarkers(); table();
}
btnG0.onclick=()=>setG('n0','btnG0');
btnG3.onclick=()=>setG('n3','btnG3');
btnGA.onclick=()=>setG('all','btnGA');
btnClear.onclick=()=>{clearLines();chart.timeScale().fitContent();
  document.getElementById('sel').textContent='點下方任一列 —— 圖表跳到那一筆，並畫出它的價位、進場、停損、出場四條線。';};

chart.timeScale().fitContent();
new ResizeObserver(()=>{chart.applyOptions({});eqc.applyOptions({});})
  .observe(document.body);
</script></body></html>
"""


def render(d):
    p = d["params"]
    html = (TPL.replace("__DATA__", json.dumps(d, ensure_ascii=False, default=float))
            .replace("__SYM__", d["sym"])
            .replace("__SPAN__", f'{d["span"][0]} → {d["span"][1]} UTC')
            .replace("__PIVOT__", str(p["PIVOT"])).replace("__W__", str(p["W"]))
            .replace("__HOLD__", str(p["HOLD"])).replace("__DIS__", str(p["DIS"]))
            .replace("__SLIP__", str(p["SLIP"])))
    return html


def parse_day(s):
    return int(datetime.strptime(s, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp()) if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--from", dest="d_from")
    ap.add_argument("--to", dest="d_to")
    ap.add_argument("--last-days", type=int, default=45)
    a = ap.parse_args()

    syms = CORE9 if a.all else [a.symbol.upper()]
    OUT.mkdir(parents=True, exist_ok=True)
    for s in syms:
        t_from, t_to = parse_day(a.d_from), parse_day(a.d_to)
        if t_from is None:
            t_from = load_bars(s)[-1][0] - a.last_days * 86400
        d = build(s, t_from, t_to)
        f = OUT / f"backtest_{s}.html"
        f.write_text(render(d), encoding="utf-8")
        m = d["stats_view"]
        tail = (f"{m['n']} 筆  每筆 {m['exp']:+.4f} R  勝率 {m['wr']:.1f}%  "
                f"PF {m['pf']:.2f}" if m else "0 筆")
        print(f"{s}: {d['span'][0]} → {d['span'][1]}  "
              f"顯示窗 {len(d['candles']):,} 根 / {tail}")
        print(f"   -> {f}")


if __name__ == "__main__":
    main()
