# -*- coding: utf-8 -*-
"""交會事件 · 回測檢視器 —— 把現行實盤規則的每一筆交易畫在 K 線上

使用者 2026-09-08：「那個歷史回測應該要顯示交會的那個策略才對喔」

網站 /charts/backtest 一直畫的是舊線（掃單失敗，2026-09-07 結案）。
現在要上實盤的是新線 —— 交會事件（TODO §1.03），本檔給它同一種檢視器：
每一條線、每一個標記，都取自**同一筆被計分的記錄**，不重算、不近似、
不另寫一份繪圖用的邏輯（mistake.md 2026-08-26：兩份實作會安靜地不同意）。

===========================================================================
交易規則（＝ conj_watch / jarvis 要執行的那一套，一個字不動）
===========================================================================
    母體    掃單 ∧ (delta_ext ∨ vol_burst)     ——「NO-OI」，2 分鐘管線的母體
            （時鐘 conj_clock 註冊的是含 oi_crash 的版本；OI 粒度 5 分鐘、
              結構上進不了 2 分鐘死線，所以**實盤跑的是 NO-OI**，本頁畫的
              也是它。conj_clock_and 的「且」變體 = 本頁的 S+D+V 那一格。）
    組裝    `event_triage.cluster`（相鄰 ≤5 分併為一時刻、錨點取最早、
            時刻之間 60 分冷卻）—— 直接呼叫同一顆
    方向    impulse = sign(close(a) − close(a−5))  順著走（延續交易）
    進場    錨點 +2 分那根的**開盤**（規格延遲；conj_watch 的 ref_price）
    停損    1.0 × ATR_h14(a)，以分鐘高低價判定，從進場**下一根**起
    出場    停損，或進場後 60 分那根的收盤，先到者
    單位    ATR（不是舊線的 R = 3.5 ATR）。+0.20 = 平均每筆賺 0.2 個小時 ATR

    已知答案對照（`tests/test_conj_backtest_parity.py` P1）：
    池化毛利必須重現 `flow_direction.py` P 臂的 **+0.2275**（同一條規則，
    同一個母體）。對不上代表本檔是第二份實作而且不同意，不得上網站。

成本（凍結分腿模型 `sweep_forward.SCEN` 情境 A，2026-09-08 §1.03 更正）
    進場 7 bps ／ 時間出場 3 bps ／ 停損出場 10 bps
    逐筆換算：cost_ATR = bps/1e4 × entry / ATR   （逐幣真實 bps，不用統一單位）
    表上「淨」= 毛 − 這一筆自己那條腿的成本

畫什麼（全部來自同一筆記錄）
    ┈┈  價位線     從樞紐形成（formed_at）延伸到被掃那一分鐘
    ▽▲  掃單       第一次穿越價位的那分鐘
    ●   進場       錨點 +2 分開盤
    ✕   出場       停損價（停損）或 +60 分收盤（時間）
    K 線用 **5 分鐘**（顯示用；規則跑在 1 分鐘上）。標記對齊到所在的
    5 分鐘 K；點選單筆時畫出精確價位／進場／停損／出場四條水平線，
    上方文字給精確到分鐘的時刻。

為什麼要發佈到 DB 而不是讓雲端算
    分鐘 bar、OI、事件表全部在 `research/poc/data/`（gitignored，映像裡
    沒有）。這是「錄製器在本機」那一族的第 N 個實例（v7_veto_clock、
    raid_signals_live…）：本機的 conj_update 班車算好 HTML，寫進
    `conj_backtest_pages`，agent 只 SELECT（agent-boundary.md）。

用法
    python research/poc/conj_backtest.py --symbol BTC
    python research/poc/conj_backtest.py --all --last-days 90 --publish
出：research/results/conj_backtest_{SYM}.html（+ DB 一列/幣）
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
sys.path.insert(0, str(HERE.parents[1]))
import event_census as ec  # noqa: E402
import event_triage as et  # noqa: E402
import conj_clock as ck  # noqa: E402
import sweep_core as sc  # noqa: E402   只借 metrics()，同一把尺

BARS = HERE / "data" / "bars"
EVENTS = HERE / "data" / "events"
LEVELS = HERE / "data" / "levels"
RES = HERE / "data" / "results"
OUT = HERE.parents[0] / "results"
CORE9 = list(ec.CORE9)

# 規則常數 —— 與 conj_watch.py 同值。這裡不 import conj_watch（它會連 DB、
# 抓 Binance），但 parity 測試會斷言兩邊的數字相等。
W = 5
DELAY = 2
STOP = 1.0
HOLD = 60
FLOW = ("delta_ext", "vol_burst")
MERGE_GAP = et.MERGE_GAP
# 分腿成本（bps）：sweep_forward.SCEN 情境 A
COST_ENTRY, COST_TIME, COST_STOP = 7.0, 3.0, 10.0
CANDLE_MIN = 5
TABLE_DDL = """
CREATE TABLE IF NOT EXISTS conj_backtest_pages (
    sym        VARCHAR(8)  NOT NULL PRIMARY KEY,
    html       MEDIUMTEXT  NOT NULL,
    asof_ts    BIGINT      NOT NULL,
    n_view     INT         NOT NULL,
    n_all      INT         NOT NULL,
    updated_at TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP
                           ON UPDATE CURRENT_TIMESTAMP
)"""


def _empty_liq():
    """liq 只餵 liq_burst，而本線不用它 —— 離線也能跑。"""
    return pd.DataFrame({"s": pd.Series(dtype=str), "w": pd.Series(dtype="int64"),
                         "u": pd.Series(dtype=float), "sym": pd.Series(dtype=str)})


def to_day(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def ledger(sym, liq=None):
    """一個幣的完整交易帳（全歷史）。回傳 (trades, bars_df)。

    規則逐行對應 `flow_direction.py` 的 P 臂 —— 那支已用 +0.2275 對過
    規格，本函式的輸出再由 parity 測試對回去。
    """
    liq = _empty_liq() if liq is None else liq
    cand, ts, cl, at, _day = ck.frozen_cand(sym, liq)
    b = pd.read_parquet(BARS / f"{sym}.parquet",
                        columns=["ts", "open", "high", "low", "close"])
    op = b["open"].to_numpy(float)
    hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
    lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
    n = len(ts)

    sweeps = ec.cooldown_filter(np.sort(cand["sweep"]))
    sweep_set = np.asarray(sweeps, np.int64)
    pairs = [(int(m), "sweep") for m in sweeps]
    for nm in FLOW:
        v = cand.get(nm)
        if v is not None and len(v):
            for m in ec.cooldown_filter(np.sort(v)):
                pairs.append((int(m), nm))

    ev = pd.read_parquet(EVENTS / f"{sym}.parquet",
                         columns=["level_id", "side", "t_sweep", "sweep_lvl"])
    ev = ev.sort_values("t_sweep")
    ev_ts = ev["t_sweep"].to_numpy(np.int64)
    lv = pd.read_parquet(LEVELS / f"{sym}.parquet",
                         columns=["level_id", "formed_at"]).set_index("level_id")

    trades = []
    for a, sig in et.cluster(pairs):
        if "sweep" not in sig or not (sig & set(FLOW)):
            continue
        if a < W or a + DELAY + HOLD >= n:
            continue
        A = float(at[a])
        if not np.isfinite(A) or A <= 0:
            continue
        d = float(np.sign(cl[a] - cl[a - W]) or 1.0)
        j0 = a + DELAY
        ent = float(op[j0])
        end = j0 + HOLD
        adv = ((ent - lo[j0 + 1:end + 1]) if d > 0
               else (hi[j0 + 1:end + 1] - ent)) / A
        hit = np.flatnonzero(adv >= STOP)
        stop_px = ent - d * STOP * A
        if len(hit):
            jx = j0 + 1 + int(hit[0])
            exit_px, R, stopped = stop_px, -STOP, True
        else:
            jx = end
            exit_px, stopped = float(cl[end]), False
            R = float(d * (cl[end] - ent) / A)
        # 群內的掃單那一分鐘（只拿來畫價位線）。
        # 2026-09-09 收緊：原本容許錨點後 30 分鐘（6 x MERGE_GAP），但併窗
        # 就是 5 分鐘 —— 超出的那個掃單**不屬於這一群**，畫出來的價位線會是
        # 別的事件的。實測 1.2%（37/3,005）落在這個縫裡。寧可不畫也不畫錯：
        # 超窗就 m=a，找不到對應事件列 -> level=None -> 該筆不畫價位線。
        k = np.searchsorted(sweep_set, a)
        m = int(sweep_set[k]) if k < len(sweep_set) and sweep_set[k] - a <= MERGE_GAP else a
        t_sw = int(ts[m]) + ec.MIN_MS
        e_i = np.searchsorted(ev_ts, t_sw)
        level = origin = None
        lside = ""
        if e_i < len(ev_ts) and ev_ts[e_i] == t_sw:
            r = ev.iloc[e_i]
            level, lside = float(r["sweep_lvl"]), str(r["side"])
            if r["level_id"] in lv.index:
                origin = int(lv.loc[r["level_id"], "formed_at"])
        leg = COST_ENTRY + (COST_STOP if stopped else COST_TIME)
        cost = leg / 1e4 * ent / A
        trades.append(dict(
            sym=sym, sig="+".join(sorted(sig)),
            sigk="and" if {"delta_ext", "vol_burst"} <= sig
            else ("d" if "delta_ext" in sig else "v"),
            side="LONG" if d > 0 else "SHORT",
            anchor_ts=int(ts[a]), sweep_ts=int(ts[m]),
            origin_ts=origin, level=level, level_side=lside,
            entry_ts=int(ts[j0]), entry=ent, atr=A, stop=float(stop_px),
            exit_ts=int(ts[jx]), exit_px=float(exit_px), stopped=stopped,
            held=int(jx - j0), R=float(R), cost_bps=leg,
            R_net=float(R - cost),
            forward=bool(ts[a] >= ck.FREEZE_MS)))
    return trades, b


def candles_5m(b, lo_ms, hi_ms):
    v = b[(b["ts"] >= lo_ms) & (b["ts"] <= hi_ms)]
    if v.empty:
        return []
    g = v.assign(bk=(v["ts"] // (CANDLE_MIN * 60_000)) * (CANDLE_MIN * 60_000)) \
         .groupby("bk").agg(open=("open", "first"), high=("high", "max"),
                            low=("low", "min"), close=("close", "last"))
    return [dict(time=int(t // 1000), open=float(r.open), high=float(r.high),
                 low=float(r.low), close=float(r.close))
            for t, r in g.iterrows() if np.isfinite(r.open)]


def _snap(ms):
    return int(ms // (CANDLE_MIN * 60_000) * (CANDLE_MIN * 60))


def build(sym, t_from_ms, t_to_ms, liq=None):
    trades, b = ledger(sym, liq)
    last_ts = int(b["ts"].iloc[-1])
    lo = t_from_ms or int(b["ts"].iloc[0])
    hi = t_to_ms or last_ts
    candles = candles_5m(b, lo, hi)
    if len(candles) < 30:
        raise SystemExit(f"{sym}: 顯示窗只有 {len(candles)} 根 K，範圍給錯了？")
    tr = [t for t in trades if lo <= t["entry_ts"] <= hi]

    levels, markers, rows, eq = [], [], [], []
    cum = 0.0
    for t in tr:
        cum += t["R"]
        eq.append(dict(time=_snap(t["exit_ts"]), value=round(cum, 4)))
    for i, t in enumerate(tr):
        win = t["R"] > 0
        col = "#0ecb81" if win else "#f6465d"
        lc = "#f0b90b" if t["level_side"] == "buyside" else "#7b61ff"
        if t["level"] is not None:
            o = t["origin_ts"] if t["origin_ts"] is not None else t["sweep_ts"] - 3_600_000
            levels.append(dict(id=i, c=lc,
                               pts=[dict(time=_snap(o), value=t["level"]),
                                    dict(time=_snap(t["sweep_ts"]), value=t["level"])]))
        markers += [
            dict(time=_snap(t["sweep_ts"]),
                 position="aboveBar" if t["level_side"] == "buyside" else "belowBar",
                 color=lc,
                 shape="arrowDown" if t["level_side"] == "buyside" else "arrowUp",
                 text=""),
            dict(time=_snap(t["entry_ts"]),
                 position="belowBar" if t["side"] == "LONG" else "aboveBar",
                 color=col, shape="circle", text=f"{i + 1}"),
            dict(time=_snap(t["exit_ts"]),
                 position="aboveBar" if t["side"] == "LONG" else "belowBar",
                 color=col, shape="square",
                 text=f"{t['R']:+.2f}" + ("!" if t["stopped"] else "")),
        ]
        rows.append(dict(
            id=i, side=t["side"], sig=t["sig"], sigk=t["sigk"],
            anchor=to_day(t["anchor_ts"]), sweep=to_day(t["sweep_ts"]),
            entry_t=to_day(t["entry_ts"]), exit_t=to_day(t["exit_ts"]),
            origin=to_day(t["origin_ts"]) if t["origin_ts"] else "—",
            level=t["level"], level_side=t["level_side"],
            entry=t["entry"], stop=t["stop"], exit_px=t["exit_px"],
            atr=t["atr"], held=t["held"], R=round(t["R"], 4),
            R_net=round(t["R_net"], 4), cost_bps=t["cost_bps"],
            stopped=bool(t["stopped"]), forward=bool(t["forward"]),
            t_anchor=t["anchor_ts"] // 1000, t_entry=t["entry_ts"] // 1000,
            t_exit=t["exit_ts"] // 1000,
            t_origin=(t["origin_ts"] or t["sweep_ts"] - 3_600_000) // 1000))

    def _m(src, pred=lambda t: True):
        rs = [x["R"] for x in src if pred(x)]
        ns = [x["R_net"] for x in src if pred(x)]
        if len(rs) < 2:
            return None
        g, nn = sc.metrics(rs), sc.metrics(ns)
        g.update(net_exp=nn["exp"], net_pf=nn["pf"],
                 stop_rate=sum(1 for x in src if pred(x) and x["stopped"]) / len(rs) * 100)
        return g
    groups = {}
    for key, pred in (("all", lambda t: True),
                      ("and", lambda t: t["sigk"] == "and"),
                      ("d", lambda t: t["sigk"] == "d"),
                      ("v", lambda t: t["sigk"] == "v")):
        groups[key] = dict(view=_m(tr, pred), all=_m(trades, pred))

    clock = {}
    for stem in ("conj_clock", "conj_clock_and"):
        p = RES / f"{stem}.json"
        if p.exists():
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                clock[stem] = dict(n=j.get("n"), n_target=j.get("n_target"),
                                   verdict=j.get("verdict"))
            except Exception as e:                       # 顯示層不可靜默
                print(f"[WARN] {stem}.json unreadable: {e}")

    n_fwd = sum(1 for t in trades if t["forward"])
    return dict(sym=sym, candles=candles, levels=levels, markers=markers,
                trades=rows, equity=eq, groups=groups,
                span=[to_day(lo), to_day(hi)],
                span_all=[to_day(int(b["ts"].iloc[0])), to_day(last_ts)],
                asof_ts=last_ts, n_all=len(trades), n_view=len(tr),
                n_forward=n_fwd, freeze_day=ck.FREEZE_DAY, clock=clock,
                params=dict(DELAY=DELAY, STOP=STOP, HOLD=HOLD, W=W,
                            COST=[COST_ENTRY, COST_TIME, COST_STOP],
                            CANDLE_MIN=CANDLE_MIN))


TPL = r"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__SYM__ 交會事件 · 回測檢視</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#0b0e11;--pan:#12161c;--line:#1e242d;--ink:#eaecef;--dim:#848e9c;
      --up:#0ecb81;--dn:#f6465d;--buy:#f0b90b;--sell:#7b61ff;--amb:#f0b90b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:13px/1.55 -apple-system,"Segoe UI","Noto Sans TC","Microsoft JhengHei",sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:16px 16px 40px;
      display:flex;flex-direction:column;gap:14px}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px 16px}
h1{margin:0;font-size:19px;font-weight:600;letter-spacing:.01em}
.tag{font-size:11px;color:var(--dim);border:1px solid var(--line);
     border-radius:3px;padding:2px 8px;font-variant-numeric:tabular-nums}
.stat{border:1px solid var(--amb);border-left:4px solid var(--amb);
      border-radius:4px;background:rgba(240,185,11,.05);padding:11px 14px;
      display:flex;flex-direction:column;gap:5px}
.stat b{color:var(--amb);font-size:13px}
.stat p{margin:0;font-size:12px;line-height:1.6}
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
th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4){text-align:left}
tbody tr{cursor:pointer}
tbody tr:hover{background:#171d25}
tbody tr.sel{background:#1c2530}
.pos{color:var(--up)}.neg{color:var(--dn)}.fwd{color:var(--amb)}
.note{font-size:11.5px;color:var(--dim);max-width:96ch}
</style></head><body><div class="wrap">

<header>
  <h1>__SYM__USDT · 交會事件回測</h1>
  <span class="tag">__SPAN__</span>
  <span class="tag">掃單 ∧ (主動量極端 ∨ 量能爆發) · 進場 錨點+__DELAY__分開盤 · 停損 __STOP__ ATR · 持有 __HOLD__ 分</span>
  <span class="tag">K 線 __CM__ 分鐘（顯示用）· 規則跑在 1 分鐘</span>
  <span class="tag">凍結規則 · 純回測 · 非訊號</span>
</header>

<div class="stat">
  <b>狀態：in-sample 回測 ＋ 凍結日（__FREEZE__）之後的前瞻樣本 __NFWD__ 筆</b>
  <p>這是<b>現在要上小額實盤的那一套規則</b>（conj_watch → jarvis），畫的是它在歷史上
  每一筆會怎麼進、怎麼出。單位是 <b>ATR</b>（小時 ATR(14)）：+0.20 代表平均每筆賺
  0.2 個 ATR；停損一次 = −1.0。「淨」= 毛利減掉這一筆自己那條腿的成本
  （進場 7 bps；時間出場 +3、停損出場 +10）。</p>
  <p>前瞻時鐘另有自己的判準（配對差、日聚類 CI、n ≥ 300），__CLOCK__——
  <b>本頁的交易 R 不是時鐘的證據</b>，凍結日之後那幾筆在表上標「前瞻」，
  數字太少，不要拿來下結論。</p>
</div>

<div class="kpis" id="kpis"></div>

<div class="bar">
  <span><span class="sw" style="border-color:var(--buy)"></span>買側價位被掃（向上穿越）</span>
  <span><span class="sw" style="border-color:var(--sell)"></span>賣側價位被掃（向下穿越）</span>
  <span><span class="dot" style="background:var(--up)"></span>進場（錨點 +2 分開盤）</span>
  <span><span class="sq" style="background:var(--dn)"></span>出場（停損價或 +60 分收盤，「!」= 停損）</span>
  <span style="width:100%"></span>
  <span style="color:var(--amb)">⚠ 這是<b>延續</b>交易：順著突破方向進場，<b>不等回踩</b>——所以圓點
  不會落在虛線（價位）上，而是在它外側。這跟舊的掃單失敗（回踩到價位才進）相反。</span>
  <button id="btnAll" class="on">全部交易</button>
  <button id="btnWin">只看賺</button>
  <button id="btnLose">只看賠</button>
  <span style="width:100%"></span>
  <b>逐筆看：</b>
  <button id="btnPrev">‹ 上一筆</button>
  <span id="navpos" style="min-width:5em;text-align:center"></span>
  <button id="btnNext">下一筆 ›</button>
  <button id="btnFit">全景（90 天）</button>
  <span style="width:100%"></span>
  <span style="color:var(--dim)">簽名：</span>
  <button id="btnGA" class="on">不分</button>
  <button id="btnGand">S+D+V（且）</button>
  <button id="btnGd">只有主動量（S+D）</button>
  <button id="btnGv">只有量能（S+V）</button>
  <span id="dense"></span>
</div>

<div id="c"></div>
<div id="eq"></div>
<div class="note" id="sel">點下方任一列 —— 圖表跳到那一筆，並畫出它的價位、進場、停損、出場四條線。</div>

<div class="tw"><table>
<thead><tr>
<th>#</th><th>方向</th><th>簽名</th><th>錨點（UTC）</th><th>持(分)</th>
<th>價位</th><th>進場</th><th>停損</th><th>出場</th>
<th>ATR</th><th>毛(ATR)</th><th>淨(ATR)</th><th>結束於</th><th>期</th>
</tr></thead><tbody id="tb"></tbody></table></div>

<div class="note">
每一列都直接來自 <code>research/poc/conj_backtest.py:ledger()</code> 的同一筆記錄。
出場價與 R 在同一筆裡互相反解（停損筆的出場價就是停損價；時間出場筆的 R =
方向 × (出場 − 進場) / ATR），由 <code>tests/test_conj_backtest_parity.py</code>
釘住：池化毛利必須重現 <code>flow_direction.py</code> 的 +0.2275（同一條規則的
另一份實作，兩份必須同意）。「持」= 進場到出場的分鐘數（上限 60）。
簽名：D = 主動量極端（|delta| 5 分後向和 ≥ 滾動 30 日 p99）、V = 量能爆發
（5 分量 / 前 30 日同時段均值 ≥ 滾動 p99）。「且」那一格就是
<code>conj_clock_and</code> 另開時鐘的母體。
</div>

</div>
<script>
const D = __DATA__;
const fmtP = v => v >= 1000 ? v.toFixed(1) : v >= 1 ? v.toFixed(3) : v.toFixed(5);

// 篩選狀態宣告在 kpis() **被呼叫之前**——let/const 的暫時性死區會把整段
// script 打斷、圖一片空白而頁面其他部分照常渲染（mistake.md 2026-09-08）。
let filt = 'all';       // 賺賠
let grp  = 'all';       // 簽名
// `$` 也宣告在這裡：focus() 在定義處之後才會被呼叫，但把取用工具留在
// 檔案下半部正是上一次 TDZ 的形狀，不重複同一個佈局。
const $ = id => document.getElementById(id);
let cur = -1;           // 目前聚焦的交易 id（逐筆導航用）

function kpis(){
  const G = (D.groups && D.groups[grp]) || {};
  const a = G.view, b = G.all;
  if(!a){document.getElementById('kpis').innerHTML =
    '<div class="kpi"><b>0</b><span>這段期間這一格沒有交易</span></div>'; return;}
  const cell = (v,k,s,cls) => `<div class="kpi"><b class="${cls||''}">${v}</b>`+
    `<span>${k}</span><small>${s}</small></div>`;
  const g = x => x >= 0 ? 'pos' : 'neg';
  const B = b || a;
  document.getElementById('kpis').innerHTML = [
    cell(a.n, '交易筆數', `全期 ${B.n}`),
    cell(a.exp.toFixed(4), '每筆毛利（ATR）', `全期 ${B.exp.toFixed(4)}`, g(a.exp)),
    cell(a.net_exp.toFixed(4), '每筆淨利（ATR）', `全期 ${B.net_exp.toFixed(4)}`, g(a.net_exp)),
    cell(a.wr.toFixed(1)+'%', '勝率', `全期 ${B.wr.toFixed(1)}%`),
    cell(a.stop_rate.toFixed(1)+'%', '停損率', `全期 ${B.stop_rate.toFixed(1)}%`),
    cell(isFinite(a.pf)?a.pf.toFixed(2):'∞', '獲利因子（毛）', `全期 ${isFinite(B.pf)?B.pf.toFixed(2):'∞'}`),
    cell(a.t.toFixed(2), 't 值（未聚類）', `全期 ${B.t.toFixed(2)}`),
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

const keepWL = t => filt==='all' || (filt==='win' ? t.R>0 : t.R<=0);
const keepG  = t => grp==='all' || t.sigk===grp;
const keep = t => keepWL(t) && keepG(t);
function drawMarkers(){
  const vis = D.trades.filter(keep);
  const ids = new Set(vis.map(t=>t.id));
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
eqc.addLineSeries({color:'#0ecb81',lineWidth:2,title:'累積 ATR（毛）',
  priceLineVisible:false,crosshairMarkerVisible:false}).setData(D.equity);
chart.timeScale().subscribeVisibleLogicalRangeChange(r=>{if(r)eqc.timeScale().setVisibleLogicalRange(r);});

let lines = [];
function clearLines(){for(const l of lines)cs.removePriceLine(l);lines=[];}
function focus(t){
  clearLines();
  const mk = (p,c,txt,st) => lines.push(cs.createPriceLine({price:p,color:c,
    lineWidth:1,lineStyle:st===undefined?2:st,axisLabelVisible:true,title:txt}));
  if(t.level!==null && t.level!==undefined) mk(t.level,'#848e9c','價位',0);
  mk(t.entry, t.R>0?'#0ecb81':'#f6465d','進場');
  mk(t.stop, '#f6465d','停損 '+D.params.STOP+'ATR',3);
  mk(t.exit_px,'#f0b90b','出場 '+t.R.toFixed(3)+'ATR');
  // ±90 分鐘：一筆交易 62 分鐘，這個視野讓 5 分鐘 K 有 ~40 根、每根約 35
  // 像素，掃單／進場／出場才分得開（±3 小時時每根只剩 15 像素，還是擠）。
  const pad = 5400;
  chart.timeScale().setVisibleRange({from:t.t_anchor-pad, to:t.t_exit+pad});
  cur = t.id;
  const vis = D.trades.filter(keep), i = vis.findIndex(x=>x.id===t.id);
  if(i>=0) $('navpos').textContent = `${i+1} / ${vis.length}`;
  document.getElementById('sel').innerHTML =
    `<b>#${t.id+1} ${t.side}</b>${t.forward?' <span class="fwd">（前瞻）</span>':''}`+
    ` · ${t.sweep} 穿過${t.level_side==='buyside'?'買側':'賣側'}價位 ${t.level===null?'—':fmtP(t.level)}`+
    `（樞紐 ${t.origin}）+ ${t.sig.replace('sweep','S').replace('delta_ext','D').replace('vol_burst','V')}`+
    ` → 錨點 ${t.anchor} → ${t.entry_t} 開盤進場 ${fmtP(t.entry)}`+
    ` → 停損掛 ${fmtP(t.stop)}（${D.params.STOP} × ATR ${fmtP(t.atr)}）`+
    ` → ${t.exit_t} ${t.stopped?'觸及停損':'持有 60 分到期'} 出在 ${fmtP(t.exit_px)}，`+
    `毛 <b class="${t.R>0?'pos':'neg'}">${t.R.toFixed(4)} ATR</b>、`+
    `淨 <b class="${t.R_net>0?'pos':'neg'}">${t.R_net.toFixed(4)}</b>（成本 ${t.cost_bps} bps）`;
  for(const tr of document.querySelectorAll('#tb tr')) tr.classList.remove('sel');
  const row = document.getElementById('r'+t.id); if(row) row.classList.add('sel');
}

const sigShort = s => s.replace('sweep','S').replace('delta_ext','D').replace('vol_burst','V');
function table(){
  document.getElementById('tb').innerHTML = D.trades.filter(keep).map(t=>
    `<tr id="r${t.id}"><td>${t.id+1}</td>`+
    `<td class="${t.side==='LONG'?'pos':'neg'}">${t.side}</td>`+
    `<td>${sigShort(t.sig)}</td><td>${t.anchor}</td><td>${t.held}</td>`+
    `<td>${t.level===null?'—':fmtP(t.level)}</td><td>${fmtP(t.entry)}</td><td>${fmtP(t.stop)}</td>`+
    `<td>${fmtP(t.exit_px)}</td><td>${fmtP(t.atr)}</td>`+
    `<td class="${t.R>0?'pos':'neg'}">${t.R.toFixed(4)}</td>`+
    `<td class="${t.R_net>0?'pos':'neg'}">${t.R_net.toFixed(4)}</td>`+
    `<td>${t.stopped?'停損':'時間'}</td>`+
    `<td class="${t.forward?'fwd':''}">${t.forward?'前瞻':'樣本內'}</td></tr>`).join('');
  for(const tr of document.querySelectorAll('#tb tr'))
    tr.onclick = () => focus(D.trades[+tr.id.slice(1)]);
}
table();

// 不用「id 直接當全域變數」那種瀏覽器特有寫法——J2 的替身抓不到它，
// 而且它在真瀏覽器裡也只是碰巧能用。（`$` 宣告在檔案上方。）
function setF(f, btn){
  filt = f;
  for(const b of ['btnAll','btnWin','btnLose']) $(b).classList.toggle('on', b===btn);
  drawMarkers(); table(); nav(0);
}
$('btnAll').onclick=()=>setF('all','btnAll');
$('btnWin').onclick=()=>setF('win','btnWin');
$('btnLose').onclick=()=>setF('lose','btnLose');
function setG(g, btn){
  grp = g;
  for(const b of ['btnGA','btnGand','btnGd','btnGv']) $(b).classList.toggle('on', b===btn);
  kpis(); drawMarkers(); table(); nav(0);
}
$('btnGA').onclick=()=>setG('all','btnGA');
$('btnGand').onclick=()=>setG('and','btnGand');
$('btnGd').onclick=()=>setG('d','btnGd');
$('btnGv').onclick=()=>setG('v','btnGv');
$('btnFit').onclick=()=>{clearLines();chart.timeScale().fitContent();
  $('navpos').textContent='全景';
  $('sel').textContent='全景下一筆交易只有 62 分鐘 ≈ 12 根 K，標記會疊在一起 —— 用「逐筆看」或點下表任一列。';};

// 預設**不做 fitContent**：90 天 = 25,921 根 5 分 K 塞進一個畫面，一根 K
// 只有 0.05 像素，而一筆交易 62 分鐘 = 12 根 K = 0.67 像素 —— 掃單／進場／
// 出場三個標記在全景下疊成同一個點，看起來像「進出場位置不對」。
// （2026-09-09 使用者：「回測的進出場跟我看的也差很多」。計算層逐筆對回
// 原始 1 分鐘 bar 全部吻合，錯的是預設視野。）
// 所以開頁就聚焦到最後一筆，並提供逐筆導航。（`cur` 宣告在檔案上方。）
function nav(step){
  const vis = D.trades.filter(keep);
  if(!vis.length){ $('navpos').textContent='0 筆'; return; }
  let i = vis.findIndex(t=>t.id===cur);
  i = (i<0) ? vis.length-1 : Math.min(vis.length-1, Math.max(0, i+step));
  cur = vis[i].id;
  $('navpos').textContent = `${i+1} / ${vis.length}`;
  focus(vis[i]);
}
$('btnPrev').onclick=()=>nav(-1);
$('btnNext').onclick=()=>nav(1);
nav(0);
new ResizeObserver(()=>{chart.applyOptions({});eqc.applyOptions({});})
  .observe(document.body);
</script></body></html>
"""


def render(d):
    p = d["params"]
    ck_ = d.get("clock", {})
    parts = []
    for stem, lab in (("conj_clock", "「或」時鐘"), ("conj_clock_and", "「且」時鐘")):
        c = ck_.get(stem)
        if c and c.get("n") is not None:
            parts.append(f"{lab} {c['n']}/{c['n_target']}（{c.get('verdict', '')}）")
    clock_txt = "現在 " + "、".join(parts) if parts else "進度見研究看板"
    return (TPL.replace("__DATA__", json.dumps(d, ensure_ascii=False, default=float))
            .replace("__SYM__", d["sym"])
            .replace("__SPAN__", f'{d["span"][0]} → {d["span"][1]} UTC')
            .replace("__DELAY__", str(p["DELAY"])).replace("__STOP__", str(p["STOP"]))
            .replace("__HOLD__", str(p["HOLD"])).replace("__CM__", str(p["CANDLE_MIN"]))
            .replace("__FREEZE__", d["freeze_day"]).replace("__NFWD__", str(d["n_forward"]))
            .replace("__CLOCK__", clock_txt))


def publish(pages):
    """[(sym, html, asof_ts, n_view, n_all)] -> conj_backtest_pages（一列/幣）。"""
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(TABLE_DDL)
            for sym, html, asof, nv, na in pages:
                cur.execute(
                    "INSERT INTO conj_backtest_pages (sym, html, asof_ts, n_view, n_all) "
                    "VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE "
                    "html=VALUES(html), asof_ts=VALUES(asof_ts), "
                    "n_view=VALUES(n_view), n_all=VALUES(n_all)",
                    (sym, html, int(asof), int(nv), int(na)))
        conn.commit()
    finally:
        conn.close()


def parse_day(s):
    return int(datetime.strptime(s, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000) if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--from", dest="d_from")
    ap.add_argument("--to", dest="d_to")
    ap.add_argument("--last-days", type=int, default=90)
    ap.add_argument("--publish", action="store_true", help="寫進 conj_backtest_pages")
    a = ap.parse_args()

    syms = CORE9 if a.all else [a.symbol.upper()]
    OUT.mkdir(parents=True, exist_ok=True)
    pages = []
    for s in syms:
        t_from, t_to = parse_day(a.d_from), parse_day(a.d_to)
        if t_from is None:
            last = int(pd.read_parquet(BARS / f"{s}.parquet", columns=["ts"])["ts"].iloc[-1])
            t_from = last - a.last_days * 86_400_000
        d = build(s, t_from, t_to)
        html = render(d)
        f = OUT / f"conj_backtest_{s}.html"
        f.write_text(html, encoding="utf-8")
        m = d["groups"]["all"]["view"]
        ma = d["groups"]["all"]["all"]
        tail = (f"{m['n']} 筆  毛 {m['exp']:+.4f} 淨 {m['net_exp']:+.4f} ATR  "
                f"勝率 {m['wr']:.1f}%  停損率 {m['stop_rate']:.1f}%" if m else "0 筆")
        print(f"{s}: {d['span'][0]} → {d['span'][1]}  顯示窗 {len(d['candles']):,} 根 / {tail}")
        print(f"   全期 {ma['n']} 筆 毛 {ma['exp']:+.4f}   前瞻 {d['n_forward']} 筆   -> {f}")
        pages.append((s, html, d["asof_ts"], d["n_view"], d["n_all"]))
    if a.publish:
        publish(pages)
        print(f"published {len(pages)} page(s) -> conj_backtest_pages")


if __name__ == "__main__":
    main()
