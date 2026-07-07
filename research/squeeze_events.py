"""
squeeze_events.py — 擠壓突破事件採樣器
Range Breakout Pro v2.4（Pine）之研究用 Python 移植。

角色定位：event sampler——偵測壓縮窗口與突破/回測成交事件，輸出事件時間戳、
價位幾何與 OHLCV 層 label（障礙結局、MFE/MAE、horizon 漂移），供 order flow
特徵層（MBP-10）以時間窗 join。不含濾網（ADX/RSI/Vol 已實證零資訊）、
不含繪圖/告警/下單，純函數、無 I/O。

Pine 對帳注意：
- ta.atr = Wilder RMA(TR)；本模組精確復刻。
- ta.ema 暖機語義略異，比對事件時丟棄前 3*max(period) 根。
- TV 端需 filterMode=OFF、confirmOnClose=ON 才與本模組事件一致。
- 結算為保守制：同棒 SL/TP 雙觸計 SL；同棒可貫穿多檔 TP（while 推進）。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class EntryMode(str, Enum):
    CLOSE = "close"      # 突破棒收盤進場
    RETEST = "retest"    # 回測區間邊緣 limit（touch=fill 假設）


class SLAnchor(str, Enum):
    WICK = "wick"        # 突破棒影線快照 ∓ sl_mult*ATR（v2.4 對齊）
    ENTRY = "entry"      # 進場價 ∓ sl_mult*ATR


class TPGeometry(str, Enum):
    R_MULT = "r"         # TP = entry ± mult*|entry-SL|（建議，基線可比）
    ATR_MULT = "atr"


@dataclass(frozen=True)
class SqueezeConfig:
    range_period: int = 15
    atr_period: int = 20
    range_multiplier: float = 0.9
    entry_mode: EntryMode = EntryMode.RETEST
    retest_expiry: int = 10
    sl_anchor: SLAnchor = SLAnchor.WICK
    tp_geometry: TPGeometry = TPGeometry.R_MULT
    tp_mults: tuple[float, float, float] = (2.0, 3.5, 6.0)
    sl_mult: float = 1.8
    horizon_bars: int = 20
    min_range_bars: int = 15                 # = range_period 即不過濾
    session_utc: tuple[int, int] | None = None  # (start_h, end_h) 半開區間；None=不過濾
    max_open: int | None = None              # None=並行（指標語義）；1=去叢聚（策略語義）


def _rma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    alpha = 1.0 / n
    prev = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        prev = v if np.isnan(prev) else prev + alpha * (v - prev)
        out[i] = prev
    return out


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    pc = np.roll(c, 1)
    pc[0] = np.nan
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return _rma(tr, n)


def detect_events(ohlcv: pd.DataFrame, cfg: SqueezeConfig = SqueezeConfig()) -> pd.DataFrame:
    """
    ohlcv: DataFrame，UTC DatetimeIndex，欄位 open/high/low/close（確認棒）。
    回傳事件表，每列一個突破事件（含未成交/已取消者）。
    """
    df = ohlcv
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ts = df.index
    n = len(df)

    ema = df["close"].ewm(span=cfg.range_period, adjust=False).mean().to_numpy()
    atr = _atr(h, l, c, cfg.atr_period)
    band = atr * cfg.range_multiplier

    rp = cfg.range_period
    roll_max = df["close"].rolling(rp).max().to_numpy()
    roll_min = df["close"].rolling(rp).min().to_numpy()
    in_range = (roll_max <= ema + band) & (roll_min >= ema - band)
    in_range = np.where(np.isnan(band), False, in_range)

    events: list[dict] = []
    open_trades: list[dict] = []

    box_live = False
    box_top = box_bot = np.nan
    box_created = -1
    armed: dict | None = None

    def _in_session(t: int) -> bool:
        if cfg.session_utc is None:
            return True
        s, e = cfg.session_utc
        hh = ts[t].hour
        return s <= hh < e if s < e else (hh >= s or hh < e)

    def _make_event(t: int, direction: int) -> dict:
        return dict(
            squeeze_start_ts=ts[box_created], squeeze_bars=t - box_created + rp,
            breakout_ts=ts[t], direction=direction,
            box_top=box_top, box_bottom=box_bot, atr_bo=band[t],
            filled=False, fill_ts=pd.NaT, cancel_reason=None,
            entry=np.nan, sl=np.nan, tp1=np.nan, tp2=np.nan, tp3=np.nan, risk=np.nan,
            sl_first=None, tps_hit=0, resolved=False, resolved_ts=pd.NaT,
            r_tp1_all=np.nan, r_scaleout=np.nan, mfe_R=np.nan, mae_R=np.nan,
            horizon_R=np.nan,
        )

    def _open_trade(ev: dict, t: int, entry: float, sig_lo: float, sig_hi: float, atr_ref: float):
        d = ev["direction"]
        if cfg.sl_anchor is SLAnchor.ENTRY:
            sl = entry - d * cfg.sl_mult * atr_ref
        else:
            sl = sig_lo - cfg.sl_mult * atr_ref if d == 1 else sig_hi + cfg.sl_mult * atr_ref
        risk = abs(entry - sl)
        unit = risk if cfg.tp_geometry is TPGeometry.R_MULT else atr_ref
        tps = tuple(entry + d * m * unit for m in cfg.tp_mults)
        ev.update(filled=True, fill_ts=ts[t], entry=entry, sl=sl,
                  tp1=tps[0], tp2=tps[1], tp3=tps[2], risk=risk)
        open_trades.append(dict(ev=ev, d=d, entry=entry, sl=sl, tps=tps, risk=risk,
                                stage=0, pnl=0.0, mfe=0.0, mae=0.0, fill_i=t))

    def _settle(t: int):
        for tr in open_trades[:]:
            d, e, r = tr["d"], tr["entry"], tr["risk"]
            tr["mfe"] = max(tr["mfe"], (h[t] - e) / r if d == 1 else (e - l[t]) / r)
            tr["mae"] = max(tr["mae"], (e - l[t]) / r if d == 1 else (h[t] - e) / r)
            ev = tr["ev"]
            hit_sl = l[t] <= tr["sl"] if d == 1 else h[t] >= tr["sl"]
            if hit_sl:
                s = tr["stage"]
                r_mults = [abs(tp - e) / r for tp in tr["tps"]]
                ev.update(sl_first=(s == 0), tps_hit=s, resolved=True, resolved_ts=ts[t],
                          r_tp1_all=(-1.0 if s == 0 else r_mults[0]),
                          r_scaleout=sum(r_mults[i] / 3 for i in range(s)) - (3 - s) / 3,
                          mfe_R=tr["mfe"], mae_R=tr["mae"])
                open_trades.remove(tr)
                continue
            while tr["stage"] < 3:
                tgt = tr["tps"][tr["stage"]]
                if not (h[t] >= tgt if d == 1 else l[t] <= tgt):
                    break
                tr["stage"] += 1
            if tr["stage"] == 3:
                r_mults = [abs(tp - e) / r for tp in tr["tps"]]
                ev = tr["ev"]
                ev.update(sl_first=False, tps_hit=3, resolved=True, resolved_ts=ts[t],
                          r_tp1_all=r_mults[0], r_scaleout=sum(r_mults) / 3,
                          mfe_R=tr["mfe"], mae_R=tr["mae"])
                open_trades.remove(tr)

    warmup = 3 * max(cfg.range_period, cfg.atr_period)
    for t in range(warmup, n):
        # (a) 壓縮窗口形成（轉換觸發）
        if in_range[t] and not in_range[t - 1]:
            box_top, box_bot = ema[t] + band[t], ema[t] - band[t]
            box_created = t
            box_live = True

        # (b) 突破偵測
        new_fill: tuple | None = None
        if box_live and armed is None:
            direction = 1 if c[t] > box_top else (-1 if c[t] < box_bot else 0)
            if direction != 0:
                box_live = False
                mature = (t - box_created + rp) >= cfg.min_range_bars
                cap_ok = cfg.max_open is None or len(open_trades) < cfg.max_open
                if mature and _in_session(t) and cap_ok:
                    ev = _make_event(t, direction)
                    events.append(ev)
                    if cfg.entry_mode is EntryMode.CLOSE:
                        new_fill = (ev, t, c[t], l[t], h[t], band[t])
                    else:
                        armed = dict(ev=ev, d=direction,
                                     level=box_top if direction == 1 else box_bot,
                                     inval=box_bot if direction == 1 else box_top,
                                     sig_lo=l[t], sig_hi=h[t], atr=band[t], bar=t)

        # (c) 回測掛單狀態機（arming 棒本身即檢查成交，與 Pine 一致）
        if armed is not None:
            if t - armed["bar"] > cfg.retest_expiry:
                armed["ev"]["cancel_reason"] = "expiry"
                armed = None
            elif (armed["d"] == 1 and c[t] < armed["inval"]) or \
                 (armed["d"] == -1 and c[t] > armed["inval"]):
                armed["ev"]["cancel_reason"] = "invalidated"
                armed = None
            elif (armed["d"] == 1 and l[t] <= armed["level"]) or \
                 (armed["d"] == -1 and h[t] >= armed["level"]):
                cap_ok = cfg.max_open is None or len(open_trades) < cfg.max_open
                if cap_ok:
                    new_fill = (armed["ev"], t, armed["level"],
                                armed["sig_lo"], armed["sig_hi"], armed["atr"])
                else:
                    armed["ev"]["cancel_reason"] = "max_open"
                armed = None

        # (d) 建倉 →（e）結算（含成交棒本身，保守制）
        if new_fill is not None:
            _open_trade(*new_fill)
        _settle(t)

    if armed is not None:
        armed["ev"]["cancel_reason"] = "pending_eod"

    out = pd.DataFrame(events)
    if out.empty:
        return out

    # (f) horizon 漂移（離線一次計算；重疊處理留給分析層）
    pos = {t: i for i, t in enumerate(ts)}
    hz = []
    for _, ev in out.iterrows():
        if not ev["filled"]:
            hz.append(np.nan)
            continue
        i0 = pos[ev["fill_ts"]] + cfg.horizon_bars
        hz.append(np.nan if i0 >= n else
                  ev["direction"] * (c[i0] - ev["entry"]) / ev["risk"])
    out["horizon_R"] = hz
    return out
