"""WorldQuant 101 alphas — adapted for single-asset BTC.

Adaptation rules:
  - rank(x)  → ts_rank(x, 100)  (cross-section → rolling 100-bar rank)
  - scale(x) → x / ts_std(x, 100)  (rolling normalization)
  - vwap     → typical price (H+L+C)/3 if not in feature cache

Conditional IC discipline (per mistake.md 2026-06-01):
  residual = return_4h - OLS(prob_up)
  conditional_IC = spearman(alpha, residual)

Threshold for "candidate":
  - |conditional_IC| > 0.03
  - frac_positive >= 8/10 walk-forward folds
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

# Tunable: rolling window for adapted rank()
TS_RANK_WINDOW = 100


# ── Helpers (adapted) ────────────────────────────────────────────────────────
def rank(s: pd.Series, window: int = TS_RANK_WINDOW) -> pd.Series:
    """ADAPTED rank → rolling ts_rank (single-asset version)."""
    return s.rolling(window).apply(lambda x: rankdata(x)[-1] / len(x), raw=True)


def scale(s: pd.Series, k: float = 1.0, window: int = TS_RANK_WINDOW) -> pd.Series:
    """ADAPTED scale → rolling normalization by abs sum."""
    return s.div(s.abs().rolling(window).sum()).mul(k).fillna(0)


def ts_sum(s, window=10): return s.rolling(window).sum()
def sma(s, window=10): return s.rolling(window).mean()
def stddev(s, window=10): return s.rolling(window).std()
def ts_min(s, window=10): return s.rolling(window).min()
def ts_max(s, window=10): return s.rolling(window).max()
def correlation(x, y, window=10): return x.rolling(window).corr(y)
def covariance(x, y, window=10): return x.rolling(window).cov(y)
def delta(s, period=1): return s.diff(period)
def delay(s, period=1): return s.shift(period)
def sign(s): return np.sign(s)
def log(s): return np.log(s.replace(0, np.nan))
def power(s, p): return s.pow(p)


def ts_rank(s, window=10):
    return s.rolling(window).apply(lambda x: rankdata(x)[-1] / len(x), raw=True)


def ts_argmax(s, window=10):
    return s.rolling(window).apply(np.argmax, raw=True) + 1


def ts_argmin(s, window=10):
    return s.rolling(window).apply(np.argmin, raw=True) + 1


def decay_linear(s: pd.Series, period: int = 10) -> pd.Series:
    """Linear-weighted moving average."""
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()
    return s.rolling(period).apply(lambda x: (x * weights).sum(), raw=True)


# ── Alpha bundle (1-10) ──────────────────────────────────────────────────────
class WQ101:
    def __init__(self, ohlcv: pd.DataFrame):
        self.open   = ohlcv['open']
        self.high   = ohlcv['high']
        self.low    = ohlcv['low']
        self.close  = ohlcv['close']
        self.volume = ohlcv['volume']
        # VWAP proxy when not in cache: typical price weighted by volume
        if 'vwap' in ohlcv.columns:
            self.vwap = ohlcv['vwap']
        else:
            self.vwap = (self.high + self.low + self.close) / 3.0
        self.returns = self.close.pct_change()

    # ── 001 ──
    def alpha001(self):
        inner = self.close.copy()
        inner[self.returns < 0] = stddev(self.returns, 20)[self.returns < 0]
        return rank(ts_argmax(inner ** 2, 5))

    # ── 002 ──
    def alpha002(self):
        x = rank(delta(log(self.volume), 2))
        y = rank((self.close - self.open) / self.open)
        df = -1 * correlation(x, y, 6)
        return df.replace([-np.inf, np.inf], 0).fillna(0)

    # ── 003 ──
    def alpha003(self):
        df = -1 * correlation(rank(self.open), rank(self.volume), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(0)

    # ── 004 ──
    def alpha004(self):
        return -1 * ts_rank(rank(self.low), 9)

    # ── 005 ──
    def alpha005(self):
        return rank((self.open - (ts_sum(self.vwap, 10) / 10))) * (
            -1 * abs(rank((self.close - self.vwap)))
        )

    # ── 006 ──
    def alpha006(self):
        df = -1 * correlation(self.open, self.volume, 10)
        return df.replace([-np.inf, np.inf], 0).fillna(0)

    # ── 007 ──
    def alpha007(self):
        adv20 = sma(self.volume, 20)
        a = -1 * ts_rank(abs(delta(self.close, 7)), 60) * sign(delta(self.close, 7))
        a[adv20 >= self.volume] = -1
        return a

    # ── 008 ──
    def alpha008(self):
        return -1 * rank((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                          delay(ts_sum(self.open, 5) * ts_sum(self.returns, 5), 10))

    # ── 009 ──
    def alpha009(self):
        dc = delta(self.close, 1)
        cond_1 = ts_min(dc, 5) > 0
        cond_2 = ts_max(dc, 5) < 0
        a = -1 * dc
        a[cond_1 | cond_2] = dc[cond_1 | cond_2]
        return a

    # ── 010 ──
    def alpha010(self):
        dc = delta(self.close, 1)
        cond_1 = ts_min(dc, 4) > 0
        cond_2 = ts_max(dc, 4) < 0
        a = -1 * dc
        a[cond_1 | cond_2] = dc[cond_1 | cond_2]
        return a

    def alpha011(self):
        return (rank(ts_max(self.vwap - self.close, 3)) +
                rank(ts_min(self.vwap - self.close, 3))) * rank(delta(self.volume, 3))

    def alpha012(self):
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    def alpha013(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    def alpha014(self):
        df = correlation(self.open, self.volume, 10).replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * rank(delta(self.returns, 3)) * df

    def alpha015(self):
        df = correlation(rank(self.high), rank(self.volume), 3).replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * ts_sum(rank(df), 3)

    def alpha016(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    def alpha017(self):
        adv20 = sma(self.volume, 20)
        return -1 * (rank(ts_rank(self.close, 10)) *
                     rank(delta(delta(self.close, 1), 1)) *
                     rank(ts_rank(self.volume / adv20, 5)))

    def alpha018(self):
        df = correlation(self.close, self.open, 10).replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * rank(stddev(abs(self.close - self.open), 5) +
                         (self.close - self.open) + df)

    def alpha019(self):
        return ((-1 * sign((self.close - delay(self.close, 7)) + delta(self.close, 7))) *
                (1 + rank(1 + ts_sum(self.returns, 250))))

    def alpha020(self):
        return -1 * (rank(self.open - delay(self.high, 1)) *
                     rank(self.open - delay(self.close, 1)) *
                     rank(self.open - delay(self.low, 1)))

    def alpha021(self):
        cond_1 = (sma(self.close, 8) + stddev(self.close, 8)) < sma(self.close, 2)
        cond_2 = (sma(self.volume, 20) / self.volume) < 1
        a = pd.Series(1.0, index=self.close.index)
        a[cond_1 | cond_2] = -1
        return a

    def alpha022(self):
        df = correlation(self.high, self.volume, 5).replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * delta(df, 5) * rank(stddev(self.close, 20))

    def alpha023(self):
        cond = sma(self.high, 20) < self.high
        a = pd.Series(0.0, index=self.close.index)
        a[cond] = (-1 * delta(self.high, 2).fillna(0))[cond]
        return a

    def alpha024(self):
        cond = delta(sma(self.close, 100), 100) / delay(self.close, 100) <= 0.05
        a = -1 * delta(self.close, 3)
        a[cond] = (-1 * (self.close - ts_min(self.close, 100)))[cond]
        return a

    def alpha025(self):
        adv20 = sma(self.volume, 20)
        return rank(((-1 * self.returns) * adv20) * self.vwap * (self.high - self.close))

    def alpha026(self):
        df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * ts_max(df, 3)

    def alpha027(self):
        a = rank(sma(correlation(rank(self.volume), rank(self.vwap), 6), 2) / 2.0)
        return -1 * (a > 0.5).astype(float) + (a <= 0.5).astype(float)

    def alpha028(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 5).replace([-np.inf, np.inf], 0).fillna(0)
        return scale((df + (self.high + self.low) / 2) - self.close)

    def alpha029(self):
        return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1 * rank(delta(self.close - 1, 5)))), 2))))), 5) +
                ts_rank(delay(-1 * self.returns, 6), 5))

    def alpha030(self):
        dc = delta(self.close, 1)
        inner = sign(dc) + sign(delay(dc, 1)) + sign(delay(dc, 2))
        return ((1.0 - rank(inner)) * ts_sum(self.volume, 5)) / ts_sum(self.volume, 20)

    def alpha031(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 12).replace([-np.inf, np.inf], 0).fillna(0)
        p1 = rank(rank(rank(decay_linear(-1 * rank(rank(delta(self.close, 10))), 10))))
        p2 = rank(-1 * delta(self.close, 3))
        p3 = sign(scale(df))
        return p1 + p2 + p3

    def alpha032(self):
        return scale(sma(self.close, 7) / 7 - self.close) + 20 * scale(
            correlation(self.vwap, delay(self.close, 5), 230))

    def alpha033(self):
        return rank(-1 + (self.open / self.close))

    def alpha034(self):
        inner = (stddev(self.returns, 2) / stddev(self.returns, 5)).replace([-np.inf, np.inf], 1).fillna(1)
        return rank(2 - rank(inner) - rank(delta(self.close, 1)))

    def alpha035(self):
        return (ts_rank(self.volume, 32) *
                (1 - ts_rank(self.close + self.high - self.low, 16)) *
                (1 - ts_rank(self.returns, 32)))

    def alpha036(self):
        adv20 = sma(self.volume, 20)
        return (2.21 * rank(correlation(self.close - self.open, delay(self.volume, 1), 15)) +
                0.7 * rank(self.open - self.close) +
                0.73 * rank(ts_rank(delay(-1 * self.returns, 6), 5)) +
                rank(abs(correlation(self.vwap, adv20, 6))) +
                0.6 * rank((sma(self.close, 200) / 200 - self.open) * (self.close - self.open)))

    def alpha037(self):
        return (rank(correlation(delay(self.open - self.close, 1), self.close, 200)) +
                rank(self.open - self.close))

    def alpha038(self):
        inner = (self.close / self.open).replace([-np.inf, np.inf], 1).fillna(1)
        return -1 * rank(ts_rank(self.open, 10)) * rank(inner)

    def alpha039(self):
        adv20 = sma(self.volume, 20)
        return ((-1 * rank(delta(self.close, 7) *
                            (1 - rank(decay_linear(self.volume / adv20, 9))))) *
                (1 + rank(sma(self.returns, 250))))

    def alpha040(self):
        return -1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)

    def alpha041(self):
        return power((self.high * self.low), 0.5) - self.vwap

    def alpha042(self):
        return rank(self.vwap - self.close) / rank(self.vwap + self.close)

    def alpha043(self):
        adv20 = sma(self.volume, 20)
        return ts_rank(self.volume / adv20, 20) * ts_rank(-1 * delta(self.close, 7), 8)

    def alpha044(self):
        df = correlation(self.high, rank(self.volume), 5).replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * df

    def alpha045(self):
        df = correlation(self.close, self.volume, 2).replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * (rank(sma(delay(self.close, 5), 20)) * df *
                     rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)))

    def alpha046(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10 -
                 (delay(self.close, 10) - self.close) / 10)
        a = -1 * delta(self.close, 1)
        a[inner < 0] = 1
        a[inner > 0.25] = -1
        return a

    def alpha047(self):
        adv20 = sma(self.volume, 20)
        return (((rank(1 / self.close) * self.volume) / adv20) *
                ((self.high * rank(self.high - self.close)) / (sma(self.high, 5) / 5)) -
                rank(self.vwap - delay(self.vwap, 5)))

    def alpha049(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10 -
                 (delay(self.close, 10) - self.close) / 10)
        a = -1 * delta(self.close, 1)
        a[inner < -0.1] = 1
        return a

    def alpha050(self):
        return -1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5)

    def alpha051(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10 -
                 (delay(self.close, 10) - self.close) / 10)
        a = -1 * delta(self.close, 1)
        a[inner < -0.05] = 1
        return a

    def alpha052(self):
        return ((-1 * delta(ts_min(self.low, 5), 5)) *
                rank((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220) *
                ts_rank(self.volume, 5))

    def alpha053(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        return -1 * delta(((self.close - self.low) - (self.high - self.close)) / inner, 9)

    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))

    def alpha055(self):
        divisor = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - ts_min(self.low, 12)) / divisor
        df = correlation(rank(inner), rank(self.volume), 6).replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * df

    def alpha057(self):
        return -1 * (self.close - self.vwap) / decay_linear(rank(ts_argmax(self.close, 30)), 2)

    def alpha060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        return -(2 * scale(rank(inner)) - scale(rank(ts_argmax(self.close, 10))))

    def alpha061(self):
        adv180 = sma(self.volume, 180)
        return (rank(self.vwap - ts_min(self.vwap, 16)) <
                rank(correlation(self.vwap, adv180, 18))).astype(float)

    def alpha062(self):
        adv20 = sma(self.volume, 20)
        return (rank(correlation(self.vwap, sma(adv20, 22), 10)) <
                rank((rank(self.open) + rank(self.open)) <
                     (rank((self.high + self.low) / 2) + rank(self.high)))).astype(float) * -1

    def alpha064(self):
        adv120 = sma(self.volume, 120)
        return (rank(correlation(sma(self.open * 0.178404 + self.low * (1 - 0.178404), 13),
                                   sma(adv120, 13), 17)) <
                rank(delta(((self.high + self.low) / 2) * 0.178404 +
                           self.vwap * (1 - 0.178404), 4))).astype(float) * -1

    def alpha065(self):
        adv60 = sma(self.volume, 60)
        return (rank(correlation(self.open * 0.00817205 + self.vwap * (1 - 0.00817205),
                                   sma(adv60, 9), 6)) <
                rank(self.open - ts_min(self.open, 14))).astype(float) * -1

    def alpha066(self):
        return -1 * (rank(decay_linear(delta(self.vwap, 4), 7)) +
                     ts_rank(decay_linear(
                         (self.low - self.vwap) /
                         (self.open - (self.high + self.low) / 2).replace(0, 0.0001),
                         11), 7))

    def alpha068(self):
        adv15 = sma(self.volume, 15)
        return (ts_rank(correlation(rank(self.high), rank(adv15), 9), 14) <
                rank(delta(self.close * 0.518371 + self.low * (1 - 0.518371), 1))).astype(float) * -1

    def alpha072(self):
        adv40 = sma(self.volume, 40)
        return (rank(decay_linear(correlation((self.high + self.low) / 2, adv40, 9), 10)) /
                rank(decay_linear(correlation(ts_rank(self.vwap, 4),
                                                ts_rank(self.volume, 19), 7), 3)).replace(0, 1e-9))

    def alpha074(self):
        adv30 = sma(self.volume, 30)
        return (rank(correlation(self.close, sma(adv30, 37), 15)) <
                rank(correlation(rank(self.high * 0.0261661 + self.vwap * (1 - 0.0261661)),
                                  rank(self.volume), 11))).astype(float) * -1

    def alpha075(self):
        adv50 = sma(self.volume, 50)
        return (rank(correlation(self.vwap, self.volume, 4)) <
                rank(correlation(rank(self.low), rank(adv50), 12))).astype(float)

    def alpha078(self):
        adv40 = sma(self.volume, 40)
        return power(rank(correlation(
            ts_sum(self.low * 0.352233 + self.vwap * (1 - 0.352233), 20),
            ts_sum(adv40, 20), 7)),
            rank(correlation(rank(self.vwap), rank(self.volume), 6)).fillna(1))

    def alpha081(self):
        adv10 = sma(self.volume, 10)
        return ((rank(log(product(
            rank(rank(correlation(self.vwap, ts_sum(adv10, 50), 8)) ** 4), 15))) <
                 rank(correlation(rank(self.vwap), rank(self.volume), 5))).astype(float) * -1)

    def alpha083(self):
        return ((rank(delay((self.high - self.low) / (ts_sum(self.close, 5) / 5), 2)) *
                 rank(rank(self.volume))) /
                ((self.high - self.low) / (ts_sum(self.close, 5) / 5) /
                 (self.vwap - self.close).replace(0, 0.0001)))

    def alpha084(self):
        return power(ts_rank(self.vwap - ts_max(self.vwap, 15), 21),
                     delta(self.close, 5).fillna(0))

    def alpha085(self):
        adv30 = sma(self.volume, 30)
        return power(
            rank(correlation(self.high * 0.876703 + self.close * (1 - 0.876703), adv30, 10)),
            rank(correlation(ts_rank((self.high + self.low) / 2, 4),
                              ts_rank(self.volume, 10), 7)).fillna(1))

    def alpha086(self):
        adv20 = sma(self.volume, 20)
        return ((ts_rank(correlation(self.close, sma(adv20, 15), 6), 20) <
                 rank((self.open + self.close) - (self.vwap + self.open))).astype(float) * -1)

    def alpha094(self):
        adv60 = sma(self.volume, 60)
        return -1 * power(rank(self.vwap - ts_min(self.vwap, 12)),
                          ts_rank(correlation(ts_rank(self.vwap, 20),
                                                ts_rank(adv60, 4), 18), 3).fillna(1))

    def alpha095(self):
        adv40 = sma(self.volume, 40)
        return (rank(self.open - ts_min(self.open, 12)) <
                ts_rank(power(rank(correlation(sma((self.high + self.low) / 2, 19),
                                                  sma(adv40, 19), 13)), 5), 12)).astype(float)

    def alpha099(self):
        adv60 = sma(self.volume, 60)
        return ((rank(correlation(ts_sum((self.high + self.low) / 2, 20),
                                    ts_sum(adv60, 20), 9)) <
                 rank(correlation(self.low, self.volume, 6))).astype(float) * -1)

    def alpha101(self):
        return (self.close - self.open) / ((self.high - self.low) + 0.001)


# ── Conditional IC pipeline ──────────────────────────────────────────────────
def compute_conditional_ic(alpha: pd.Series,
                           y_true: pd.Series,
                           v7_pred: pd.Series,
                           folds: pd.Series | None = None) -> dict:
    """Compute conditional IC: residualize y_true on v7_pred, then IC vs alpha.

    Returns dict: {raw_ic, cond_ic, frac_pos_folds (if folds given), n}
    """
    df = pd.DataFrame({
        'a': alpha, 'y': y_true, 'p': v7_pred,
    })
    if folds is not None:
        df['f'] = folds
    df = df.dropna()
    if len(df) < 100:
        return {'raw_ic': np.nan, 'cond_ic': np.nan, 'n': len(df)}

    # OLS residual
    p = df['p'].values
    y = df['y'].values
    pmean = p.mean()
    ymean = y.mean()
    pdev = p - pmean
    ydev = y - ymean
    beta = (pdev * ydev).sum() / (pdev * pdev).sum() if (pdev * pdev).sum() > 0 else 0
    intercept = ymean - beta * pmean
    df['resid'] = y - beta * p - intercept

    raw_ic, _ = spearmanr(df['a'], df['y'])
    cond_ic, _ = spearmanr(df['a'], df['resid'])

    out = {'raw_ic': float(raw_ic), 'cond_ic': float(cond_ic), 'n': len(df)}

    if folds is not None:
        fold_ics = []
        for f, g in df.groupby('f'):
            if len(g) < 20:  # folds are ~30 bars each in this dataset
                continue
            ic, _ = spearmanr(g['a'], g['resid'])
            if not np.isnan(ic):
                fold_ics.append(ic)
        if fold_ics:
            out['fold_ics'] = fold_ics
            mean_sign = np.sign(np.mean(fold_ics))
            out['frac_pos_folds'] = sum(1 for x in fold_ics if np.sign(x) == mean_sign) / len(fold_ics)
            out['n_folds'] = len(fold_ics)
    return out


def main():
    # Load data
    print("Loading V7 OOS + features...")
    oos = pd.read_parquet('research/results/dual_model/direction_oos_full_expanded.parquet')
    feat = pd.read_parquet('research/dual_model/.cache/features_all.parquet')

    # Align: OOS index is UTC tz-aware; features same. Take overlap.
    common = oos.index.intersection(feat.index)
    print(f"OOS rows: {len(oos)}, features rows: {len(feat)}, overlap: {len(common)}")

    ohlcv = feat.loc[common, ['open', 'high', 'low', 'close', 'volume']].copy()
    y_true = oos.loc[common, 'return_4h']
    v7_pred = oos.loc[common, 'prob_up']
    folds = oos.loc[common, 'fold']

    # Compute alphas (auto-discover all alphaXXX methods)
    wq = WQ101(ohlcv)
    available = sorted([m for m in dir(wq) if m.startswith('alpha') and m[5:].isdigit()],
                       key=lambda m: int(m[5:]))
    print(f"Computing {len(available)} alphas: {available[0]}..{available[-1]}...")
    alpha_funcs = [(name, getattr(wq, name)) for name in available]

    results = []
    for name, func in alpha_funcs:
        try:
            a = func()
            if isinstance(a, pd.DataFrame):
                a = a.iloc[:, 0]  # take first col if DataFrame
            res = compute_conditional_ic(a, y_true, v7_pred, folds)
            res['name'] = name
            results.append(res)
            fp = res.get('frac_pos_folds', np.nan)
            nf = res.get('n_folds', 0)
            fp_str = f"{fp*100:5.1f}%" if not np.isnan(fp) else "  NA"
            print(f"  {name}: raw_IC={res['raw_ic']:+.4f}  cond_IC={res['cond_ic']:+.4f}  "
                  f"frac_pos={fp_str}({nf} folds)  n={res['n']}")
        except Exception as e:
            print(f"  {name}: FAILED — {type(e).__name__}: {str(e)[:60]}")

    # Save results to csv for later batches
    pd.DataFrame(results).to_csv('research/results/wq101_adapted_ic.csv', index=False)

    print()
    print("=" * 60)
    print("STRONG CANDIDATES (|cond_IC| > 0.03 AND frac_pos > 65%):")
    print("=" * 60)
    strong = [r for r in results
              if isinstance(r.get('cond_ic'), float)
              and abs(r['cond_ic']) > 0.03
              and r.get('frac_pos_folds', 0) > 0.65]
    if not strong:
        print("  (none)")
    else:
        for r in sorted(strong, key=lambda x: -abs(x['cond_ic'])):
            print(f"  {r['name']}: cond_IC={r['cond_ic']:+.4f}  frac_pos={r['frac_pos_folds']*100:.1f}%")

    print()
    print("=" * 60)
    print("WEAK CANDIDATES (|cond_IC|>0.03 but frac_pos 55-65% — borderline):")
    print("=" * 60)
    weak = [r for r in results
            if isinstance(r.get('cond_ic'), float)
            and abs(r['cond_ic']) > 0.03
            and 0.55 <= r.get('frac_pos_folds', 0) <= 0.65]
    if not weak:
        print("  (none)")
    else:
        for r in sorted(weak, key=lambda x: -abs(x['cond_ic'])):
            print(f"  {r['name']}: cond_IC={r['cond_ic']:+.4f}  frac_pos={r['frac_pos_folds']*100:.1f}%")

    return results


if __name__ == '__main__':
    main()
