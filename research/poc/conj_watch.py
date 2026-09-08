# -*- coding: utf-8 -*-
"""交會事件的分鐘級偵測器（shadow 模式：只記錄，不發訊號）

規格來源 TODO §1.03 / `conj_pipeline_spec.py`：
    可執行窗口 **2 分鐘**（扣成本後 CI 下緣仍 > 0 的最大延遲）。
    現行每小時班車平均延遲 32.5 分，效應只剩 29.5%，扣成本後為負。
    拿掉 `oi_crash`（Binance metrics 粒度就是 5 分鐘，結構上壓不進 2 分鐘）
    之後覆蓋率仍有 **89.8%**，而且效應**更好**（2 分鐘延遲 +0.2278 vs
    +0.2064）—— OI 那一支本來就貼零（+0.0176，扣成本 CI 下緣 −0.21），
    與 `BRIDGE.md` 量到的「淨 OI 對事件 AUC 0.4996」一致。

所以這條管線**只吃 1 分鐘 kline**，不碰 OI。

===========================================================================
設計原則
===========================================================================
1. **不動現有的每小時班車。** 那條班車服務三個時鐘與多個記帳器，它們不需要
   低延遲，動它風險高。本檔是**獨立的第二條線**，只寫自己的表。

2. **偵測邏輯不重寫。** 價位來自 `sweep_core.detect_sweeps`（凍結）、
   門檻來自 `conj_causal.causal_flags`（因果、滾動 30 日）。本檔只做
   「把它們預先算好、每分鐘拿新資料去比對」這件事。
   **同一份偵測，不是第二份實作**（mistake.md 2026-08-26）。

3. **狀態預算，熱路徑只做比較。**
   價位表：每小時重算一次（樞紐要確認 10 根之後才成立，本來就慢）
   門檻表：每日重算一次（滾動 30 日分位，逐日更新）
   熱路徑：抓最後幾根 1 分鐘 K -> 算 5 分鐘後向和 -> 比門檻 -> 比價位

4. **shadow 模式**：偵測到就寫 `conj_events_live`，**不發任何訊號**。
   目的是累積**真實的端到端延遲**——那個數字才是「2 分鐘做不做得到」的
   證據，估算不算數。

5. 每輪自報 `{ok, reason}` 旗標給 freshness board（把「從未開始」翻譯成
   「某個數字不對」，mistake.md 2026-09-01）。

用法（由排程每分鐘呼叫）
    python research/poc/conj_watch.py
    python research/poc/conj_watch.py --rebuild     # 強制重算狀態
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
import sweep_core as sc  # noqa: E402
import event_census as ec  # noqa: E402
import event_triage as et  # noqa: E402

ROOT = HERE.parents[1]
LIVE = HERE / "data" / "live"
BARS = HERE / "data" / "bars"
CACHE = HERE.parents[0] / "sweep_failure" / ".cache"
FLAG = HERE / "data" / "results" / "conj_watch_last.json"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
FLOW = ("delta_ext", "vol_burst")        # OI 不納入，見檔頭
W = 5                                    # 事件聚合窗（分鐘）
MERGE_GAP = 5
LEVEL_TTL_S = 3600                       # 價位表存活 1 小時
THR_TTL_S = 86400                        # 門檻表存活 1 天
COOLDOWN = 60                            # 分鐘：與 event_census / event_triage 同
# 熱路徑的回看長度。舊值 40 分鐘**不夠重現離線的冷卻**：離線對每個類型先做
# 60 分冷卻、時刻之間再做 60 分冷卻，所以要看得到至少兩層冷卻鏈才會給出同樣
# 的結果。240 分鐘留了四倍餘裕（Binance limit 上限 1000）。
LOOKBACK_MIN = 240
# 只發最近這幾根裡錨定的時刻。**不可以設成 MERGE_GAP+1**：live 組出來的錨點
# 會比離線早一兩分鐘（流量的冷卻鏈落點略不同），那個值剛好把真事件切掉
# （注入測試 77.8%，漏掉的清一色是「錨點在 n-7 而閘門要求 >= n-6」）。
# 跨輪去重本來就由 `recent_ok` 的 60 分冷卻 + UNIQUE KEY 負責，這裡不需要緊。
EMIT_RECENT = 15
UA = {"User-Agent": "conj-watch/1.0"}


# ───────────────────────── 狀態層（預算） ─────────────────────────
def pivot_table(sym):
    """所有樞紐 + 它們**第一次被穿越**的小時索引（未被穿越 = n）。

    2026-09-08 加：`levels_asof` 每個時間切點都要重掃一次樞紐，對照測試
    要跑幾百個切點時是 O(切點 x n)。樞紐條件與「第一次被穿越」都跟切點
    無關，只算一次就好；之後任何切點只是兩個比較：
        conf < T  且  first_pierce >= T   ->  在 T 時還活著
    這與 `sweep_core.detect_sweeps` 是同一組條件（同一份偵測）。
    回傳 (ts_ms, conf, level, is_hi, first_pierce)，全部是 numpy 陣列。
    """
    p = CACHE / f"{sym}USDT_1h.csv"
    if not p.exists():
        return None
    bars = sc.load_csv(str(p))
    atr = sc.atr14(bars)
    ts = np.array([b[0] for b in bars], np.int64)
    if ts.max() < 1e12:                 # 小時 K 是秒，分鐘 K 是毫秒
        ts = ts * 1000
    h = np.array([b[sc.H] for b in bars], float)
    l = np.array([b[sc.L] for b in bars], float)
    n = len(bars)
    conf_, lvl_, ishi_, fp_ = [], [], [], []
    for i in range(sc.PIVOT, n - sc.PIVOT):
        conf = i + sc.PIVOT
        if conf >= n or atr[conf] is None:
            continue
        sh, sl = h[i - sc.PIVOT:conf + 1], l[i - sc.PIVOT:conf + 1]
        for is_hi, lvl, piv in (
                (True, h[i], h[i] >= sh.max() and (sh < h[i]).any()),
                (False, l[i], l[i] <= sl.min() and (sl > l[i]).any())):
            if not piv:
                continue
            w = (h[conf + 1:] > lvl) if is_hi else (l[conf + 1:] < lvl)
            k = np.flatnonzero(w)
            conf_.append(conf)
            lvl_.append(float(lvl))
            ishi_.append(bool(is_hi))
            fp_.append(conf + 1 + int(k[0]) if len(k) else n)
    return (ts, np.array(conf_, np.int64), np.array(lvl_, float),
            np.array(ishi_, bool), np.array(fp_, np.int64), n,
            float(atr[n - 1] or 0.0))


def levels_asof(sym, hi_ts=None):
    """某個幣在某個時點「還沒被穿越」的價位表。

    `hi_ts=None` 就是現在（熱路徑用）。帶時間切點是為了讓注入測試能重播
    歷史——**用今天的價位表去重播 20 天前的事件，結構上永遠找不到**，因為
    當時那個事件掃掉的價位今天已經被標記為穿越過了。第一版注入測試就是這樣
    命中率 2.8%，看起來像發射路徑壞了，其實是儀器拿錯了時點
    （[[mistake 2026-09-03 bar 的不同欄位屬於不同時刻]] 的同族：
    同一張表在不同時點內容不同，而表名把這件事藏起來了）。
    """
    p = CACHE / f"{sym}USDT_1h.csv"
    if not p.exists():
        return []
    bars = sc.load_csv(str(p))
    atr = sc.atr14(bars)
    ts = np.array([b[0] for b in bars], np.int64)
    # 小時 K 快取的時間戳是**秒**，分鐘 K parquet 是**毫秒**。不轉換的話
    # searchsorted(秒, 毫秒) 永遠回傳結尾 —— 切點靜默失效、每個時點都拿到
    # 同一張表（注入測試三個不同錨點拿到一模一樣的 46 個價位，那就是徵兆）。
    # 照 mistake.md 2026-04-12：時間戳 unit 一律偵測，不硬編碼。
    if ts.max() < 1e12:
        ts = ts * 1000
    h = np.array([b[sc.H] for b in bars], float)
    l = np.array([b[sc.L] for b in bars], float)
    T = len(bars) if hi_ts is None else int(np.searchsorted(ts, int(hi_ts)))
    if T <= 2 * sc.PIVOT + 1:
        return []
    # 反向累積極值：cmax[j] = max(h[j:T])，用來 O(1) 判「確認之後有沒有被穿越」
    cmax = np.concatenate([np.maximum.accumulate(h[:T][::-1])[::-1], [-np.inf]])
    cmin = np.concatenate([np.minimum.accumulate(l[:T][::-1])[::-1], [np.inf]])
    a_now = float(atr[T - 1] or 0.0)
    out = []
    for i in range(sc.PIVOT, T - sc.PIVOT):
        conf = i + sc.PIVOT
        if conf >= T or atr[conf] is None:
            continue
        sh, sl = h[i - sc.PIVOT:conf + 1], l[i - sc.PIVOT:conf + 1]
        hi_piv = h[i] >= sh.max() and (sh < h[i]).any()
        lo_piv = l[i] <= sl.min() and (sl > l[i]).any()
        for is_hi, lvl, piv in ((True, h[i], hi_piv), (False, l[i], lo_piv)):
            if not piv:
                continue
            if (cmax[conf + 1] > lvl) if is_hi else (cmin[conf + 1] < lvl):
                continue                     # 確認之後已經被穿越 = 不算活著
            out.append(dict(sym=sym, level=float(lvl),
                            kind="buy" if is_hi else "sell",
                            atr=a_now, origin_ts=int(ts[i])))
    return out


def build_levels():
    """每個幣的未消耗價位表：呼叫 `levels_asof`，不另寫一份。"""
    out = []
    for sym in CORE9:
        out.extend(levels_asof(sym))
    d = pd.DataFrame(out)
    LIVE.mkdir(parents=True, exist_ok=True)
    d.to_parquet(LIVE / "levels.parquet", index=False)
    return d


class Thr:
    """一個幣在某個時點的門檻（欄位名與 thresholds.parquet 的列一致）。"""

    def __init__(self, sym, thr_delta, thr_vol, vol_base_tod, atr):
        self.sym, self.thr_delta, self.thr_vol = sym, thr_delta, thr_vol
        self.vol_base_tod, self.atr = vol_base_tod, atr


def thresholds_asof(ts, vol, delta, hi, sym="", atr=0.0, days=30):
    """用 [hi-30日, hi) 這個窗算門檻 —— **live 每天重算一次做的就是這件事**。

    抽出來是為了讓對照測試能逐日模擬 live 的行為。第一版對照拿一個固定的
    「最近 30 天」門檻去比 30 天前的事件，那分不出「實作有 bug」和「門檻窗
    依設計不同」——同一個病我修過一次又留了殘留，這次連根拔掉：兩邊都用
    「該事件當天往前 30 天」，任何不一致就只可能來自實作。
    """
    n = len(ts)

    def back5(x):
        c = np.concatenate([[0.0], np.cumsum(x)])
        i = np.arange(n)
        return c[i + 1] - c[np.clip(i + 1 - W, 0, n)]

    ad = back5(np.abs(delta))
    v5 = back5(vol)
    lo = hi - days * 86_400_000
    m = (ts >= lo) & (ts < hi)
    if m.sum() < 10_000:
        return None
    tod = ((ts // 60_000) % 1440).astype(int)
    s = np.bincount(tod[m], weights=v5[m], minlength=1440)
    c = np.bincount(tod[m], minlength=1440)
    bt = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    bs = bt[tod]
    vs = np.where(np.isfinite(bs) & (bs > 0), v5 / np.where(bs > 0, bs, 1), np.nan)
    return Thr(sym,
               float(np.nanpercentile(ad[m], 99)),
               float(np.nanpercentile(vs[m], 99)),
               bt.tolist(), float(atr))


def build_thresholds():
    """滾動 30 日 p99 門檻：呼叫 `thresholds_asof`，不另寫一份。"""
    rows = []
    for sym in CORE9:
        p = BARS / f"{sym}.parquet"
        if not p.exists():
            continue
        b = pd.read_parquet(p, columns=["ts", "volume", "delta", "atr_h14"])
        ts = b["ts"].to_numpy(np.int64)
        r = thresholds_asof(
            ts,
            np.nan_to_num(b["volume"].to_numpy(float), nan=0.0),
            np.nan_to_num(b["delta"].to_numpy(float), nan=0.0),
            # 切在**今天零點**，與 causal_flags 的「嚴格早於本日」一致，
            # 也與對照測試 B 臂模擬的行為一致 —— 被測的就是被部署的。
            hi=int(ts[-1] // 86_400_000) * 86_400_000, sym=sym,
            atr=float(b["atr_h14"].to_numpy(float)[-1]))
        if r is None:
            continue
        rows.append(dict(sym=r.sym, thr_delta=r.thr_delta, thr_vol=r.thr_vol,
                         vol_base_tod=r.vol_base_tod, atr=r.atr))
    d = pd.DataFrame(rows)
    LIVE.mkdir(parents=True, exist_ok=True)
    d.to_parquet(LIVE / "thresholds.parquet", index=False)
    return d


def load_state(rebuild=False):
    now = time.time()
    lv = LIVE / "levels.parquet"
    th = LIVE / "thresholds.parquet"
    levels = (build_levels() if rebuild or not lv.exists()
              or now - lv.stat().st_mtime > LEVEL_TTL_S
              else pd.read_parquet(lv))
    thr = (build_thresholds() if rebuild or not th.exists()
           or now - th.stat().st_mtime > THR_TTL_S
           else pd.read_parquet(th))
    return levels, thr


# ───────────────────────── 組裝（與離線共用同一顆） ─────────────────────────
def assemble(cand_like):
    """把 {類型: 分鐘索引} 組成交會時刻 —— **直接呼叫 event_triage.cluster**。

    2026-09-07 的對照測試（`conj_watch_parity.py`）在門檻對齊之後仍然只有
    55.6% 重疊，因為**組裝規則是第二份實作**：

        離線   各類型先自己 60 分冷卻 -> 相鄰 <=5 分併成一個時刻
               -> 錨點取該群最早那一分鐘 -> 時刻之間再 60 分冷卻
        舊 live 要求掃單與流量落在**同一分鐘**，且完全沒有冷卻

    於是 live 漏掉「掃單在 t、放量在 t+3」那一類（離線獨有），又把離線已被
    冷卻吃掉的重複算成新事件（live 獨有）。現在兩邊呼叫同一個 `et.cluster`，
    任何不一致只可能來自門檻或資料，不可能來自組裝。

    注意錨點可能比當下早最多 MERGE_GAP 分鐘（先掃單、後放量）。那是**真實
    延遲的一部分**，shadow 模式要如實記錄，不可用當下時刻假裝沒有。
    """
    pairs = []
    for nm in ("sweep",) + FLOW:
        v = cand_like.get(nm)
        if v is None or len(v) == 0:
            continue
        for m in ec.cooldown_filter(np.sort(np.asarray(v, np.int64))):
            pairs.append((int(m), nm))
    flowm = set(FLOW)
    return [(a, s) for a, s in et.cluster(pairs)
            if "sweep" in s and (s & flowm)]


def flow_flags(ts, vol, delta, t):
    """流量旗標：與 `event_census.detect_all` 同一份量值定義。"""
    n = len(ts)

    def back5(x):
        c = np.concatenate([[0.0], np.cumsum(x)])
        i = np.arange(n)
        return c[i + 1] - c[np.clip(i + 1 - W, 0, n)]

    d5 = back5(np.abs(delta))
    v5 = back5(vol)
    bt = np.asarray(t.vol_base_tod, float)
    bs = bt[((ts // 60_000) % 1440).astype(int)]
    ratio = np.where(np.isfinite(bs) & (bs > 0), v5 / np.where(bs > 0, bs, 1), 0.0)
    return {"delta_ext": np.flatnonzero(d5 >= t.thr_delta),
            "vol_burst": np.flatnonzero(ratio >= t.thr_vol)}


# ───────────────────────── 熱路徑 ─────────────────────────
def fetch_recent(sym, limit=LOOKBACK_MIN):
    url = (f"https://api.binance.com/api/v3/klines?symbol={sym}USDT"
           f"&interval=1m&limit={limit}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode())


def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS conj_events_live (
            id            BIGINT AUTO_INCREMENT PRIMARY KEY,
            canonical_symbol VARCHAR(20) NOT NULL,
            event_ts      BIGINT      NOT NULL,
            detected_ts   BIGINT      NOT NULL,
            latency_ms    BIGINT      NOT NULL,
            signature     VARCHAR(40) NOT NULL,
            direction     TINYINT     NOT NULL,
            level         DECIMAL(24,8) NOT NULL,
            px            DECIMAL(24,8) NOT NULL,
            atr           DECIMAL(24,8) NOT NULL,
            UNIQUE KEY uniq_evt (canonical_symbol, event_ts),
            INDEX idx_ts (event_ts)
        )
        """)
    conn.commit()


def detect_window(sym, ts, high, low, close, vol, delta, t, g, det):
    """一個幣、一段分鐘 K -> 要寫進 conj_events_live 的列。

    抽出來是為了**能被注入已知答案驗證**。交會事件約每幣每 2.5 天一次，
    所以正常跑一輪的結果永遠是 events=0 —— 那正是
    [[mistake 2026-08-26 只在清單非空時才看得見的輸出]] 的形狀：
    唯一看得見這條路徑的時刻，正是沒人在看的時刻。
    `conj_watch_inject.py` 拿歷史上真實發生過的交會時刻餵進來，要求它發得出。
    """
    if g is None or not len(g):
        return []
    n = len(ts)
    flags = flow_flags(ts, vol, delta, t)
    # 掃單：哪幾分鐘穿過了還活著的價位
    lb = g[g.kind == "buy"].level.to_numpy(float)
    ls = g[g.kind == "sell"].level.to_numpy(float)
    # **穿越測試，不是水準測試**（2026-09-07 由 conj_watch_inject 抓到）。
    # 第一版寫 `lb[lb < high[j]]` —— 任何低於現價的活價位都算命中，於是
    # **每一根 K 都是掃單**，60 分冷卻只留下窗內第 0/60/120/180 根，真正
    # 事件那根反而被吃掉（注入命中率 2.8%）。這跟 §1.02 判掉掃單主線的
    # 那一行是同一個錯（`l[f] <= lvl` 量的是「價位在哪」不是「有沒有穿過去」），
    # 我在新程式碼裡又寫了一次。
    # 正確：這一根的極值越過價位，而**前面每一根都還沒越過**（=第一次穿越）。
    rmax = np.maximum.accumulate(high)
    rmin = np.minimum.accumulate(low)
    sw, lvl_at = [], {}
    for j in range(n):
        pmax = rmax[j - 1] if j else -np.inf
        pmin = rmin[j - 1] if j else np.inf
        cb = lb[(lb < high[j]) & (lb >= pmax)]
        cs = ls[(ls > low[j]) & (ls <= pmin)]
        if len(cb) or len(cs):
            sw.append(j)
            lvl_at[j] = float(cb.max() if len(cb) else cs.min())
    flags["sweep"] = np.array(sw, np.int64)

    out = []
    for a, s in assemble(flags):
        if a < W or a < n - EMIT_RECENT:
            continue                    # 只發最近幾根，舊的上一輪發過
        sig = "S+" + "+".join(
            [x for x, k in (("D", "delta_ext"), ("V", "vol_burst")) if k in s])
        imp = 1 if close[a] > close[a - W] else -1
        out.append((sym + "-USD", int(ts[a]), det,
                    det - (int(ts[a]) + 60_000), sig, imp,
                    lvl_at.get(a, 0.0), float(close[a]), float(t.atr)))
    return out


def recent_ok(conn, sym, ev_ts):
    """跨輪的 60 分鐘冷卻：`assemble` 的冷卻只在本輪的視窗內成立。"""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(event_ts) FROM conj_events_live "
                    "WHERE canonical_symbol=%s AND event_ts > %s",
                    (sym, ev_ts - COOLDOWN * 60_000))
        r = cur.fetchone()
    return not (r and r[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    t_start = time.time()
    ok, reason, found = True, "ok", 0
    try:
        levels, thr = load_state(rebuild=a.rebuild)
        thr_by = {r.sym: r for r in thr.itertuples()}
        lv_by = {s: g for s, g in levels.groupby("sym")} if len(levels) else {}

        sys.path.insert(0, str(ROOT))
        from shared.db import get_db_conn
        conn = get_db_conn()
        ensure_table(conn)

        events = []
        for sym in CORE9:
            if sym not in thr_by:
                continue
            t = thr_by[sym]
            try:
                kl = fetch_recent(sym)
            except Exception as e:
                print(f"[WARN] {sym} fetch failed: {e}")
                continue
            # 只用**已收盤**的 K（最後一根仍在進行中，丟掉）
            kl = [k for k in kl if int(k[6]) <= time.time() * 1000][:-0 or None]
            if len(kl) < W + 2:
                continue
            ts = np.array([int(k[0]) for k in kl], np.int64)
            close = np.array([float(k[4]) for k in kl])
            high = np.array([float(k[2]) for k in kl])
            low = np.array([float(k[3]) for k in kl])
            vol = np.array([float(k[5]) for k in kl])
            tbb = np.array([float(k[9]) for k in kl])
            delta = 2 * tbb - vol

            det = int(time.time() * 1000)
            for e in detect_window(sym, ts, high, low, close, vol, delta,
                                   t, lv_by.get(sym), det):
                if recent_ok(conn, e[0], e[1]) is False:
                    continue
                events.append(e)

        if events:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT IGNORE INTO conj_events_live "
                    "(canonical_symbol, event_ts, detected_ts, latency_ms, "
                    " signature, direction, level, px, atr) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)", events)
            conn.commit()
            found = len(events)
        conn.close()
    except Exception as e:
        ok, reason = False, f"{type(e).__name__}: {e}"
        print(f"[ERROR] {reason}")

    took = (time.time() - t_start) * 1000
    lat = [e[3] for e in events] if ok and 'events' in dir() and events else []
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text(json.dumps({
        "ok": ok, "reason": reason, "events": found,
        "cycle_ms": round(took),
        "latency_ms_max": max(lat) if lat else None,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"conj_watch: events={found}  cycle={took:.0f}ms  ok={ok}"
          + (f"  max_latency={max(lat)}ms" if lat else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
