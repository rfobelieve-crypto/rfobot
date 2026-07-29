"""
sqzmom_lb.py — LazyBear「Squeeze Momentum Indicator [LazyBear]」(SQZMOM_LB) 忠實移植
＋ release 事件基線 ＋ 與 RBP 壓縮(squeeze_events)的重疊描述。

Pine v1 原稿：使用者 2026-07-14 提供（TradingView 公開腳本）。
角色定位：TODO #1「擠壓指標 × 訂單流」的**替代壓縮偵測器**——與現行
squeeze_events.py（RBP v2.4 的 EMA±ATR 通道包含）是兩種不同的壓縮定義。
本模組先過「移植忠實度」關（TV 端對帳），再做事件層基線；最終用途是
事件採樣器（供撤單特徵 join），非獨立訊號。

── 移植忠實度注意（對帳前登記）─────────────────────────────────
1. ★ 原稿已知怪癖：`dev = multKC * stdev(source, length)` —— BB 寬度用的是
   KC 乘數(1.5)，宣告的 `mult`(2.0) 輸入從未被使用。此處**忠實復刻**（不
   「修正」），否則 sqzOn/sqzOff 與 TV 端不一致。
2. Pine `stdev` = 母體標準差（ddof=0）。
3. Pine `tr` 用前收盤；首棒因 close[1]=na 而為 na（暖機丟棄後無影響）。
4. `linreg(x, n, 0)` = 最近 n 點 OLS 在最末點的擬合值（LSMA）。
5. KC 的 range 平滑是 `sma(tr, n)`，不是內建 Keltner 的 ATR(RMA)——不可
   拿 TV 內建 Keltner Channels 對帳 KC 帶。
6. 暖機：丟前 3*max(period)=60 根（repo 慣例，同 squeeze_events）。

── TV 對帳結果（跑完回填）───────────────────────────────────────
（待記：symbol/timeframe、對帳棒 ts、val Python vs TV、BB 帶對比、殘差）

── 事件定義（看資料前登記，禁改）───────────────────────────────
- release 事件：sqzOn 連續 run≥MIN_RUN(6) 棒後，於棒 t 首次轉 sqzOff
  （on[t-1] & off[t]，Carter 經典「squeeze fires」）。on→noSqz 不算 release
  （記為 fizzle，僅描述）。方向 d = sign(val[t])；val 為 0/NaN 則跳過。
- 主指標：d 帶號 (close[t+8]−close[t]) / ATR20[t]（Wilder RMA）。
  h∈{4,8,12,20} 全報，主判 h=8；前後半（事件序）同號才算過；
  n<100 只做描述不下結論（紀律鎖沿用 TODO #1）。
- 次要（純描述）：不設 run 門檻的全部 strict release 之 h=8 統計。
- 重疊描述：sqzOn 佔比、RBP in_range 佔比、Jaccard、P(sqzOn|in_range)、
  P(in_range|sqzOn)、release 前一棒 RBP 也壓縮的比例。
誠實註記：事件視窗可能重疊 → naive t 高估獨立性；單資產(BTC 1h)單期間；
SQZMOM 為公開萬人用指標，任何 OHLCV 層 edge 先驗極低——本基線是
「排除獨立 edge、確立採樣器」用，不是找聖杯。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── SQZMOM 參數（Pine 原稿預設）─────────────────────────────────
BB_LENGTH = 20
BB_MULT_DECLARED = 2.0   # 原稿宣告但未使用（怪癖 #1），僅存檔備查
KC_LENGTH = 20
KC_MULT = 1.5
USE_TRUE_RANGE = True

# ── 事件層參數（預登記）─────────────────────────────────────────
MIN_RUN = 6
HORIZONS = (4, 8, 12, 20)
PRIMARY_H = 8
WARMUP = 3 * max(BB_LENGTH, KC_LENGTH)

# ── RBP 壓縮定義（squeeze_events.py 預設；獨立復刻避免耦合）─────
RBP_RANGE_PERIOD = 15
RBP_ATR_PERIOD = 20
RBP_RANGE_MULT = 0.9


def _rma(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder RMA，與 research/squeeze_events.py 同款（NaN 跳過後遞推）。"""
    out = np.full_like(x, np.nan, dtype=float)
    alpha = 1.0 / n
    prev = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        prev = v if np.isnan(prev) else prev + alpha * (v - prev)
        out[i] = prev
    return out


def _true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    pc = np.roll(c, 1)
    pc[0] = np.nan
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = np.nan  # Pine tr 首棒 na（close[1] 缺）
    return tr


def _atr_rma(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    tr = _true_range(h, l, c)
    tr0 = tr.copy()
    tr0[0] = h[0] - l[0]  # RMA 種子沿用 squeeze_events 慣例；暖機丟棄後無差
    return _rma(tr0, n)


def _linreg_endpoint(x: np.ndarray, n: int) -> np.ndarray:
    """Pine linreg(x, n, 0)：最近 n 點 OLS 在最末點的值。NaN 窗口→NaN。"""
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) < n:
        return out
    idx = np.arange(n, dtype=float)
    xm = idx.mean()
    denom = ((idx - xm) ** 2).sum()
    w = np.lib.stride_tricks.sliding_window_view(x, n)   # (len-n+1, n)
    wm = w.mean(axis=1)
    slope = (w @ (idx - xm)) / denom
    out[n - 1:] = wm + slope * ((n - 1) - xm)
    return out


def compute(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """SQZMOM_LB 全欄位。輸入：UTC index + open/high/low/close。

    回傳欄位：val、sqz_on、sqz_off、no_sqz、bb_up、bb_lo、kc_up、kc_lo、atr
    （bb/kc 帶外露供 TV 帶級對帳；atr = Wilder RMA20，供事件層正規化）。
    """
    h = ohlcv["high"].to_numpy(float)
    l = ohlcv["low"].to_numpy(float)
    c = ohlcv["close"].to_numpy(float)
    s = ohlcv["close"]

    # BB —— 注意怪癖 #1：dev 用 KC_MULT
    basis = s.rolling(BB_LENGTH).mean()
    dev = KC_MULT * s.rolling(BB_LENGTH).std(ddof=0)
    bb_up = (basis + dev).to_numpy()
    bb_lo = (basis - dev).to_numpy()

    # KC —— ma=SMA、range=TR、rangema=SMA(TR)
    ma = s.rolling(KC_LENGTH).mean().to_numpy()
    rng = _true_range(h, l, c) if USE_TRUE_RANGE else (h - l)
    rangema = pd.Series(rng, index=ohlcv.index).rolling(KC_LENGTH).mean().to_numpy()
    kc_up = ma + rangema * KC_MULT
    kc_lo = ma - rangema * KC_MULT

    sqz_on = (bb_lo > kc_lo) & (bb_up < kc_up)
    sqz_off = (bb_lo < kc_lo) & (bb_up > kc_up)
    no_sqz = ~sqz_on & ~sqz_off
    valid = ~np.isnan(bb_up) & ~np.isnan(kc_up)
    sqz_on &= valid
    sqz_off &= valid

    # 動量 val = linreg(close − avg(avg(HH,LL), SMA), n, 0)
    hh = ohlcv["high"].rolling(KC_LENGTH).max().to_numpy()
    ll = ohlcv["low"].rolling(KC_LENGTH).min().to_numpy()
    mid = ((hh + ll) / 2.0 + s.rolling(KC_LENGTH).mean().to_numpy()) / 2.0
    val = _linreg_endpoint(c - mid, KC_LENGTH)

    return pd.DataFrame(
        dict(val=val, sqz_on=sqz_on, sqz_off=sqz_off, no_sqz=no_sqz,
             bb_up=bb_up, bb_lo=bb_lo, kc_up=kc_up, kc_lo=kc_lo,
             atr=_atr_rma(h, l, c, 20)),
        index=ohlcv.index,
    )


def rbp_in_range(ohlcv: pd.DataFrame) -> np.ndarray:
    """RBP 壓縮旗標（squeeze_events.detect_events 之 in_range，逐行復刻）。"""
    h = ohlcv["high"].to_numpy(float)
    l = ohlcv["low"].to_numpy(float)
    c = ohlcv["close"].to_numpy(float)
    ema = ohlcv["close"].ewm(span=RBP_RANGE_PERIOD, adjust=False).mean().to_numpy()
    atr = _atr_rma(h, l, c, RBP_ATR_PERIOD)
    band = atr * RBP_RANGE_MULT
    roll_max = ohlcv["close"].rolling(RBP_RANGE_PERIOD).max().to_numpy()
    roll_min = ohlcv["close"].rolling(RBP_RANGE_PERIOD).min().to_numpy()
    in_range = (roll_max <= ema + band) & (roll_min >= ema - band)
    return np.where(np.isnan(band), False, in_range)


def release_events(ind: pd.DataFrame, close: np.ndarray) -> pd.DataFrame:
    """strict release（on[t-1]→off[t]）事件表；fizzle（on→noSqz）另計。"""
    on = ind["sqz_on"].to_numpy()
    off = ind["sqz_off"].to_numpy()
    val = ind["val"].to_numpy()
    atr = ind["atr"].to_numpy()
    ts = ind.index
    n = len(ind)

    rows = []
    n_fizzle = 0
    run = 0
    for t in range(n):
        if on[t]:
            run += 1
            continue
        if run > 0:
            if off[t]:
                d = float(np.sign(val[t])) if not np.isnan(val[t]) else 0.0
                row = dict(ts=ts[t], run_len=run, direction=d, val=val[t],
                           val_slope=(val[t] - val[t - 1]) if t >= 1 else np.nan,
                           atr=atr[t])
                for hz in HORIZONS:
                    j = t + hz
                    row[f"fwd_{hz}"] = (
                        d * (close[j] - close[t]) / atr[t]
                        if (j < n and d != 0.0 and atr[t] > 0) else np.nan
                    )
                rows.append(row)
            else:
                n_fizzle += 1
        run = 0
    ev = pd.DataFrame(rows)
    ev.attrs["n_fizzle"] = n_fizzle
    return ev


def _stats(a: np.ndarray) -> dict:
    a = a[~np.isnan(a)]
    if len(a) < 2:
        return dict(n=len(a), mean=np.nan, t=np.nan, win=np.nan)
    m = a.mean()
    t = m / (a.std(ddof=1) / np.sqrt(len(a)))
    return dict(n=len(a), mean=m, t=t, win=(a > 0).mean())


def _fmt(tag: str, st: dict) -> str:
    return (f"  {tag:<14} n={st['n']:<4d} mean={st['mean']:+.4f} ATR  "
            f"t={st['t']:+.2f}  win={st['win']*100:.1f}%")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    path = root / "market_data" / "raw_data" / "binance_klines_1h.parquet"
    df = pd.read_parquet(path)[["open", "high", "low", "close"]].dropna().astype(float)
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()

    ind = compute(df)
    rbp = rbp_in_range(df)

    # 暖機丟棄（統計、佔比、事件全部只看 warmup 之後）
    ind = ind.iloc[WARMUP:]
    rbp = rbp[WARMUP:]
    close = df["close"].to_numpy(float)[WARMUP:]

    print(f"bars={len(ind)}  span {ind.index[0]} → {ind.index[-1]}  (warmup 丟 {WARMUP})")

    on = ind["sqz_on"].to_numpy()
    off = ind["sqz_off"].to_numpy()
    print("\n── 狀態佔比 ──")
    print(f"  sqzOn {on.mean()*100:.1f}%   sqzOff {off.mean()*100:.1f}%   "
          f"noSqz {(~on & ~off).mean()*100:.1f}%")

    ev = release_events(ind, close)
    if ev.empty:
        print("無 release 事件。")
        return 0
    prim = ev[(ev["run_len"] >= MIN_RUN) & (ev["direction"] != 0.0)].reset_index(drop=True)
    runs = ev["run_len"].to_numpy()
    print(f"\n── squeeze runs ──")
    print(f"  strict release={len(ev)}（run 中位數 {np.median(runs):.0f}）  "
          f"fizzle(on→noSqz)={ev.attrs['n_fizzle']}  主事件(run≥{MIN_RUN})={len(prim)}")

    print(f"\n── 主事件前向報酬（d=sign(val)，ATR 正規化；主判 h={PRIMARY_H}）──")
    for hz in HORIZONS:
        st = _stats(prim[f"fwd_{hz}"].to_numpy(float))
        mark = " ★" if hz == PRIMARY_H else ""
        print(_fmt(f"h={hz}{mark}", st))
    half = len(prim) // 2
    st_a = _stats(prim[f"fwd_{PRIMARY_H}"].to_numpy(float)[:half])
    st_b = _stats(prim[f"fwd_{PRIMARY_H}"].to_numpy(float)[half:])
    print(f"  前半 {_fmt('', st_a).strip()}")
    print(f"  後半 {_fmt('', st_b).strip()}")

    sec = ev[ev["direction"] != 0.0]
    print(f"\n── 次要（全部 strict release，不設 run 門檻；純描述）──")
    print(_fmt(f"h={PRIMARY_H}", _stats(sec[f"fwd_{PRIMARY_H}"].to_numpy(float))))

    print("\n── 與 RBP 壓縮（squeeze_events in_range）重疊 ──")
    inter = (on & rbp).sum()
    union = (on | rbp).sum()
    print(f"  佔比: sqzOn {on.mean()*100:.1f}%  vs  RBP in_range {rbp.mean()*100:.1f}%")
    print(f"  Jaccard={inter/union:.3f}   P(sqzOn|RBP)={ (on & rbp).sum()/max(rbp.sum(),1):.3f}   "
          f"P(RBP|sqzOn)={inter/max(on.sum(),1):.3f}")
    pos = {t: i for i, t in enumerate(ind.index)}
    prev_rbp = [bool(rbp[pos[t] - 1]) if pos[t] >= 1 else False for t in prim["ts"]]
    print(f"  release 前一棒 RBP 也壓縮: {np.mean(prev_rbp)*100:.1f}%  (n={len(prev_rbp)})")

    out = root / "research" / "results" / "sqzmom_release_events.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(out, index=False)
    print(f"\n事件表 → {out}")
    print("註：事件視窗可能重疊，naive t 偏樂觀；單資產單期間；判讀依預登記紀律。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
