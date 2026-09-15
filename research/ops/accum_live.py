# -*- coding: utf-8 -*-
"""本機即時監看 —— 跑起來，開瀏覽器，看著數字動。

    python research/ops/accum_live.py          然後開 http://127.0.0.1:8780
    Ctrl+C 結束

===========================================================================
解析度分兩層，而這是**物理限制不是設計選擇**
===========================================================================
使用者 2026-09-13：「用 html 的方式就不能即時看到數據在跑，怎麼做才能在本機
看到我數據即時在跑」。

    MySQL 的 7 張表    **秒級，真即時**
        逐筆寫入，所以 `COUNT(*) WHERE 時間欄 >= now-60s` 當下就有答案。

    4 個 WS 錄製器     **5 分鐘**
        `FLUSH_SEC = 300` —— 它們每 300 秒才碰一次磁碟（parquet 落盤 ＋
        寫旗標）。**行程外面沒有任何東西看得到比這更細的訊號。**

所以這一頁對錄製器顯示的是「距上次落盤幾秒」，它會從 0 數到 300 然後歸零 ——
那個歸零就是它活著的脈搏。要更細只有兩條路，兩條都要改錄製器：
  (a) 每 2 秒寫一個幾百位元組的心跳 JSON（計數器，不落盤）
  (b) 把 FLUSH_SEC 調小（代價是 parquet 碎片化與更多 read-modify-write）
(a) 比較對，但要重啟那四支（用「等落盤完才殺」的程序，損失秒級）。
**還沒做 —— 那是一個要使用者拍的決定。**

===========================================================================
這是 monitor 不是 control
===========================================================================
`deploying-strategies`（2024-07-18）：「suppose you are only showing monitoring
statistics... you can be much less strict with the security on it because it
**can't actually control the algorithm**」。

本支**唯讀**：不寫任何檔、不寫任何表、不碰交易路徑。只綁 127.0.0.1。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from research import arb_home  # noqa: E402  2026-09-15: the four WS recorders live in arb/recorders

# (顯示名, 小時分檔根目錄, 旗標)
RECORDERS = [
    ("Lighter 逐筆成交帶", "D:/flowbot_data/lighter/trades",
     str(arb_home.RESULTS / "lighter_tape_last.json")),
    ("Lighter 中價與深度", "D:/flowbot_data/lighter/mid",
     str(arb_home.RESULTS / "lighter_mid_last.json")),
    ("HL 逐筆成交帶", "D:/flowbot_data/hl/trades",
     str(arb_home.RESULTS / "hl_tape_last.json")),
    ("HL 中價與佇列", "D:/flowbot_data/hl/mid",
     str(arb_home.RESULTS / "hl_mid_last.json")),
]
FLUSH_SEC = 300

# (顯示名, 表, 時間欄, 是否 epoch)
TABLES = [
    ("撤單事件 1s", "depth_events_1s", "created_at", False),
    ("簿口快照 1m", "orderbook_snapshots_1m", "ts_ms", True),
    ("撤單深度差 1m", "depth_deltas_1m", "minute_start_ms", True),
    ("未平倉量", "oi_snapshots", "ts_received", True),
    ("資金費", "funding_rates", "ts_received", True),
    ("清算事件", "liq_events", "created_at", False),
]

HIST = 160                      # 曲線保留的樣本數
RATE_WIN = 120.0                # 速率的窗（秒）。必須長於寫入端的
                                # 批次週期（5-10 秒），否則慢的表
                                # 會一直報 0 而它其實活著
_state = {"rec": {}, "db": {}, "db_asof": 0.0, "started": time.time()}
_hist = {}                      # 名稱 -> [值, ...]
_lock = threading.Lock()


def push(name, v):
    h = _hist.setdefault(name, [])
    h.append(v)
    del h[:-HIST]


def poll_recorders():
    """從 parquet mtime 與旗標算「距上次落盤幾秒」。便宜，2 秒一次沒問題。"""
    import glob
    while True:
        out = {}
        for lab, root, flag in RECORDERS:
            d = {"name": lab}
            try:
                fs = glob.glob(root + "/*/*.parquet")
                if fs:
                    newest = max(fs, key=os.path.getmtime)
                    d["since_flush"] = time.time() - os.path.getmtime(newest)
                    d["file"] = os.path.basename(os.path.dirname(newest)) \
                        + "/" + os.path.basename(newest)
                else:
                    d["error"] = "沒有 parquet"
            except OSError as e:
                d["error"] = str(e)
            p = os.path.join(ROOT, flag)
            try:
                with open(p, encoding="utf-8") as fh:
                    fj = json.load(fh)
                d["ok"] = bool(fj.get("ok"))
                d["reason"] = (fj.get("reason") or "")[:90]
                for k in ("trades", "rows_written", "rows_today", "coins",
                          "uptime_sec"):
                    if k in fj:
                        d[k] = fj[k]
            except Exception as e:
                d["flag_error"] = "%s" % e
            out[lab] = d
            sf = d.get("since_flush")
            if sf is not None:
                push("flush:" + lab, round(sf))
        with _lock:
            _state["rec"] = out
        time.sleep(2)


def poll_db(every: int):
    """插入速率 = **Δ自增主鍵 ÷ Δ時間**。

    第一版用 `COUNT(*) WHERE 時間欄 >= now-60s`，而那幾張表的時間欄**沒有
    索引**（只有 canonical_symbol/exchange 那組），所以它在 5.1M 列的
    `depth_events_1s` 上要掃全表 —— 實測 9 秒還沒回來第一輪。
    在一個「即時」監看裡跑全表掃描是設計錯誤。

    六張表都有自增 `id`，而 `MAX(id)` 是 **66 ms**（幾乎全是到 Railway 的
    網路延遲，查詢本身走主鍵是 O(1)）。兩次之間的差就是真實插入筆數。
    代價：id 有空洞時（回滾、刪除）會略為高估 —— 對「活著沒」這個問題
    可以忽略，但不要拿它當精確計數。
    """
    from shared.db import get_db_conn
    seen = {}                       # 表 -> [(t, max_id), ...]
    while True:
        out, t1 = {}, time.time()
        try:
            cn = get_db_conn()
            cur = cn.cursor()
            for lab, tbl, col, epoch in TABLES:
                try:
                    cur.execute("SELECT MAX(`id`) v FROM `%s`" % tbl)
                    mx = dict(cur.fetchall()[0])["v"]
                    mx = int(mx) if mx is not None else None
                    d = {"name": lab, "max_id": mx}
                    if mx is not None:
                        h = seen.setdefault(lab, [])
                        h.append((t1, mx))
                        del h[:-400]
                        # **窗口要比寫入端的批次週期長。** 第一版拿相鄰兩次
                        # 相減（約 5 秒），而 data-model.md 寫著寫入端是批次的
                        # （「100 筆或 5 秒」、flow bars「每 ~10 秒」），於是
                        # 六張表全報 0.00/秒 —— 看起來像全死了，其實全活著。
                        # 這跟今天那幾個儀器錯同一類：窗比現象的週期短。
                        base = next((x for x in h if t1 - x[0] <= RATE_WIN), h[0])
                        dt = t1 - base[0]
                        if dt >= 2:
                            d["rate"] = max(0.0, (mx - base[1]) / dt)
                            d["added"] = max(0, mx - base[1])
                            d["window"] = dt
                            push("db:" + lab, round(d["rate"], 3))
                    out[lab] = d
                except Exception as e:
                    out[lab] = {"name": lab, "error": "%s" % e}
            cn.close()
        except Exception as e:
            out["_conn"] = {"error": "%s" % e}
        with _lock:
            _state["db"] = out
            _state["db_asof"] = time.time()
        time.sleep(every)


PAGE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>即時累積</title><style>
:root{--bg:#12151a;--fg:#e6e9ef;--dim:#8a93a3;--line:#242a33;--ok:#4c9b6a;
      --bad:#c0392b;--acc:#4c72b0;--warn:#d79a2b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:13px/1.5 "Consolas","Cascadia Mono",ui-monospace,monospace}
.w{max-width:1080px;margin:0 auto;padding:20px 16px 50px}
h1{font-size:17px;margin:0 0 2px}
.s{color:var(--dim);font-size:12px;margin:0 0 20px}
h2{font-size:12px;margin:26px 0 4px;color:var(--dim);letter-spacing:.9px;
   text-transform:uppercase}
.h2n{color:var(--dim);font-size:11px;margin:0 0 10px}
table{border-collapse:collapse;width:100%}
th{text-align:right;color:var(--dim);font-size:11px;font-weight:500;
   padding:0 8px 6px;border-bottom:1px solid var(--line)}
th.l,td.l{text-align:left}
td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;
   font-variant-numeric:tabular-nums;white-space:nowrap}
td.n{font-weight:600}
.big{font-size:16px;font-weight:700}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;
     margin-right:6px;vertical-align:0}
.live{animation:p 1.6s ease-in-out infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.25}}
.note{color:var(--dim);font-size:11px}
.bar{display:inline-block;width:170px;height:8px;background:var(--line);
     border-radius:4px;overflow:hidden;vertical-align:middle}
.bar>i{display:block;height:100%;background:var(--acc)}
svg{display:block}
</style>
<div class="w">
<h1>即時累積</h1>
<p class="s">每秒更新｜<span id="up"></span>｜這一頁是 monitor（唯讀、只綁
127.0.0.1、不碰交易路徑）。歷史看 accum.html</p>

<h2>秒級 · MySQL</h2>
<p class="h2n">插入速率 = Δ自增主鍵 ÷ Δ時間（O(1)，不掃表）。這一側逐筆寫入，
所以是<b>真即時</b>。曲線是最近幾分鐘。</p>
<table id="tdb"></table>

<h2>5 分鐘級 · WS 錄製器</h2>
<p class="h2n">距上次落盤幾秒。<b>FLUSH_SEC = 300</b>，所以正常是 0→300 然後
歸零——<b>那個歸零就是脈搏</b>。超過 360 秒就不是慢是死。
行程外面看不到比這更細的訊號（要更細得給錄製器加心跳）。</p>
<table id="trec"></table>
</div>
<script>
function spark(a,w,h,col){
  if(!a||a.length<2)return '';
  const mx=Math.max(...a,1),n=a.length,dx=w/(n-1);
  let d='';for(let i=0;i<n;i++){const x=i*dx,y=h-(a[i]/mx)*(h-2)-1;
    d+=(i?'L':'M')+x.toFixed(1)+' '+y.toFixed(1);}
  return '<svg width="'+w+'" height="'+h+'" viewBox="0 0 '+w+' '+h+'">'+
    '<path d="'+d+'" fill="none" stroke="'+col+'" stroke-width="1.3"/></svg>';
}
function fmt(n){return n>=1e6?(n/1e6).toFixed(2)+'M':
  n>=1e3?(n/1e3).toFixed(1)+'k':''+n;}
async function tick(){
  let j;try{j=await (await fetch('api')).json()}catch(e){return}
  document.getElementById('up').textContent='監看已跑 '+
    Math.round(j.uptime)+' 秒｜DB 快照 '+Math.round(j.db_age)+' 秒前';
  let h='<tr><th class="l">表</th><th>列/秒</th><th>上輪新增</th>'+
    '<th>最新 id</th><th class="l">速率曲線</th></tr>';
  for(const k in j.db){const d=j.db[k];if(!d.name)continue;
    if(d.error){h+='<tr><td class="l n">'+k+'</td><td colspan="4" class="l" '+
      'style="color:var(--bad)">'+d.error+'</td></tr>';continue}
    const a=j.hist['db:'+k]||[];
    const r=d.rate, known=(r!==undefined);
    const alive=known&&r>0;
    h+='<tr><td class="l n"><span class="dot '+(alive?'live':'')+'" style="background:'+
      (!known?'var(--dim)':alive?'var(--ok)':'var(--bad)')+'"></span>'+k+'</td>'+
      '<td class="big">'+(known?r.toFixed(2):'…')+'</td>'+
      '<td>'+(known?fmt(d.added):'—')+'</td>'+
      '<td class="note">'+(d.max_id!=null?fmt(d.max_id):'—')+'</td>'+
      '<td class="note">'+(d.window?Math.round(d.window)+'s':'—')+'</td>'+
      '<td class="l">'+spark(a,200,18,'#4c72b0')+'</td></tr>';}
  document.getElementById('tdb').innerHTML=h;
  let r='<tr><th class="l">錄製器</th><th>距上次落盤</th>'+
    '<th class="l">落盤週期 300s</th><th class="l">自報</th></tr>';
  for(const k in j.rec){const d=j.rec[k];
    const sf=d.since_flush,bad=(sf==null)||sf>360||d.ok===false;
    const pct=Math.min(100,(sf||0)/300*100);
    r+='<tr><td class="l n"><span class="dot '+(bad?'':'live')+'" style="background:'+
      (bad?'var(--bad)':'var(--ok)')+'"></span>'+k+'</td>'+
      '<td class="big">'+(sf==null?'—':Math.round(sf)+'s')+'</td>'+
      '<td class="l"><span class="bar"><i style="width:'+pct.toFixed(0)+
      '%;background:'+(bad?'var(--bad)':'var(--acc)')+'"></i></span></td>'+
      '<td class="l note">'+(d.reason||d.flag_error||d.error||'')+'</td></tr>';}
  document.getElementById('trec').innerHTML=r;
}
tick();setInterval(tick,1000);
</script>
"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass                                    # 不要把每個輪詢都印出來

    def do_GET(self):
        if self.path.rstrip("/").endswith("api"):
            with _lock:
                body = json.dumps(
                    {"rec": _state["rec"], "db": _state["db"],
                     "db_age": time.time() - (_state["db_asof"] or time.time()),
                     "uptime": time.time() - _state["started"],
                     "hist": _hist}, ensure_ascii=False).encode()
            ct = "application/json; charset=utf-8"
        else:
            body, ct = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--db-every", type=int, default=3,
                    help="MySQL 幾秒查一次（MAX(id) 是 O(1)，所以 3 秒沒問題）")
    a = ap.parse_args()
    threading.Thread(target=poll_recorders, daemon=True).start()
    threading.Thread(target=poll_db, args=(a.db_every,), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print("即時監看：http://127.0.0.1:%d   （Ctrl+C 結束）" % a.port)
    print("  MySQL 每 %d 秒查一次｜錄製器每 2 秒看一次 mtime" % a.db_every)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n結束")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
