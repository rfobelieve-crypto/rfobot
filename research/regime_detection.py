"""
Regime Detection for BTC-USD 15m order flow data.

Method: Hidden Markov Model (HMM) with 3 states on pure order flow features.
        + Volatility-based sanity check.

Features used (all order-flow / market-structure, no price levels):
  - realized_vol_20b   : volatility level
  - cvd_zscore         : cumulative delta momentum
  - funding_rate       : market positioning / sentiment
  - delta_ratio        : taker aggression (net buy pressure)
  - volume             : activity level (log-scaled)
  # OI features: reserved for when live data is available

Regime labels (auto-assigned by characterisation):
  0: TRENDING_BULL   — high CVD, moderate vol, positive funding
  1: TRENDING_BEAR   — low/negative CVD, moderate-high vol, negative funding
  2: CHOPPY          — low vol, near-zero delta, funding neutral

Outputs:
  - research/eda_charts/regime_*.png
  - regime labels written to features_15m DB table (regime column)
  - Parquet refreshed with regime column

Usage:
    python research/regime_detection.py
    python research/regime_detection.py --n-states 4
    python research/regime_detection.py --save
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PARQUET  = Path(__file__).parent / "ml_data" / "BTC_USD_15m_tabular.parquet"
OUT_DIR  = Path(__file__).parent / "eda_charts"

# Order-flow features used for regime detection
REGIME_FEATURES = [
    "realized_vol_20b",   # volatility level
    "cvd_zscore",         # cumulative delta z-score
    "funding_rate",       # positioning / sentiment
    "delta_ratio",        # taker aggression
    "volume",             # activity (will be log-scaled)
    # "oi_delta",         # TODO: add when live OI data accumulated
]

TARGET       = "strength_vol_adj"
REGIME_COL   = "regime"
N_STATES_DEF = 3


# ─── Data ─────────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()
    df["funding_rate"] = df["funding_rate"].ffill()
    return df


def prepare_features(df: pd.DataFrame,
                     smooth_bars: int = 4) -> tuple[np.ndarray, pd.Index]:
    """
    Scale and clean regime features. Returns (X, valid_index).
    smooth_bars: rolling mean window to reduce bar-level noise before HMM fitting.
                 4 bars = 1h for 15m data. Smoothed features produce more
                 persistent regime assignments without losing regime structure.
    """
    sub = df[REGIME_FEATURES].copy()
    sub["volume"] = np.log1p(sub["volume"])
    sub = sub.ffill()

    # Smooth to suppress tick-level noise → longer regime durations
    if smooth_bars > 1:
        sub = sub.rolling(smooth_bars, min_periods=1).mean()

    sub = sub.dropna()
    scaler = StandardScaler()
    X      = scaler.fit_transform(sub.values)
    return X, sub.index


# ─── HMM fitting ──────────────────────────────────────────────────────────────

def fit_hmm(X: np.ndarray, n_states: int = N_STATES_DEF,
            n_iter: int = 300) -> hmm.GaussianHMM:
    """
    Fit HMM with a sticky transition prior: high diagonal probability
    encourages the model to stay in the current regime rather than
    switching every bar.
    stay_prob = 0.97 → expected regime duration ≈ 1/(1-0.97) = 33 bars = ~8h
    """
    model = hmm.GaussianHMM(
        n_components    = n_states,
        covariance_type = "full",
        n_iter          = n_iter,
        init_params     = "mcs",   # exclude 't': keep our transmat init
        params          = "tmcs",  # still update transmat during EM
        random_state    = 42,
        verbose         = False,
    )

    # Sticky init: high probability of staying in current state
    stay  = 0.97
    leave = (1.0 - stay) / (n_states - 1)
    transmat_init = np.full((n_states, n_states), leave)
    np.fill_diagonal(transmat_init, stay)
    model.transmat_ = transmat_init

    model.fit(X)
    return model


def decode_regimes(model: hmm.GaussianHMM, X: np.ndarray,
                   min_duration: int = 8) -> np.ndarray:
    """
    Viterbi decode + post-process: merge runs shorter than min_duration bars
    into the preceding regime.
    min_duration=8 bars = 2h for 15m data.
    """
    raw = model.predict(X)

    # Merge short runs into previous regime
    labels = raw.copy()
    i = 0
    while i < len(labels):
        # Find run length of current state
        state = labels[i]
        j = i
        while j < len(labels) and labels[j] == state:
            j += 1
        run_len = j - i
        if run_len < min_duration and i > 0:
            labels[i:j] = labels[i - 1]  # replace with previous regime
        i = j

    return labels


# ─── Characterise regimes ─────────────────────────────────────────────────────

def characterise(df: pd.DataFrame, labels: np.ndarray,
                 valid_idx: pd.Index, n_states: int) -> pd.DataFrame:
    """Return mean of key features per regime state."""
    tagged = df.loc[valid_idx].copy()
    tagged[REGIME_COL] = labels

    cols = REGIME_FEATURES + [TARGET, "strength_raw",
                               "up_move_vol_adj", "down_move_vol_adj"]
    cols = [c for c in cols if c in tagged.columns]

    stats = tagged.groupby(REGIME_COL)[cols].mean().round(5)
    stats["count"] = tagged[REGIME_COL].value_counts().sort_index()
    stats["pct"]   = (stats["count"] / len(tagged) * 100).round(1)

    # Avg regime duration in bars (only for states that actually appear)
    durations = {}
    for state in stats.index:
        runs = []
        cur  = 0
        for s in labels:
            if s == state:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)
        durations[state] = round(np.mean(runs), 1) if runs else 0.0
    stats["avg_duration_bars"] = pd.Series(durations)

    return stats


def assign_regime_names(stats: pd.DataFrame) -> dict[int, str]:
    """Heuristically name regimes based on cvd_zscore and realized_vol_20b."""
    names = {}
    # Sort states by cvd_zscore descending
    ordered = stats["cvd_zscore"].sort_values(ascending=False).index.tolist()

    labels_pool = {
        "high_cvd":  "TRENDING_BULL",
        "low_cvd":   "TRENDING_BEAR",
        "mid_cvd":   "CHOPPY",
        "extra":     "REGIME_4",
    }
    n = len(ordered)
    if n >= 3:
        names[ordered[0]] = "TRENDING_BULL"
        names[ordered[-1]] = "TRENDING_BEAR"
        for mid in ordered[1:-1]:
            names[mid] = f"CHOPPY_{mid}" if n > 3 else "CHOPPY"
    elif n == 2:
        names[ordered[0]] = "BULL"
        names[ordered[1]] = "BEAR"
    else:
        names[ordered[0]] = "SINGLE"
    return names


# ─── Regime-conditional IC ────────────────────────────────────────────────────

def regime_ic(df: pd.DataFrame, labels: np.ndarray,
              valid_idx: pd.Index) -> pd.DataFrame:
    """Compute IC(feature, strength_vol_adj) per regime."""
    tagged = df.loc[valid_idx].copy()
    tagged[REGIME_COL] = labels

    feat_cols = [
        "cvd_zscore", "delta_ratio", "funding_rate",
        "funding_deviation", "realized_vol_20b",
        "bnc_delta_ratio", "okx_delta_ratio", "exchange_divergence",
    ]
    feat_cols = [c for c in feat_cols if c in tagged.columns]

    rows = []
    for regime, grp in tagged.groupby(REGIME_COL):
        grp = grp.dropna(subset=feat_cols + [TARGET])
        row = {"regime": regime, "n": len(grp)}
        for col in feat_cols:
            if grp[col].std() < 1e-10:
                row[col] = np.nan
                continue
            ic, _ = spearmanr(grp[col], grp[TARGET])
            row[col] = round(ic, 4)
        rows.append(row)

    # Also compute global IC for comparison
    global_grp = tagged.dropna(subset=feat_cols + [TARGET])
    row_global = {"regime": "GLOBAL", "n": len(global_grp)}
    for col in feat_cols:
        ic, _ = spearmanr(global_grp[col], global_grp[TARGET])
        row_global[col] = round(ic, 4)
    rows.append(row_global)

    return pd.DataFrame(rows).set_index("regime")


# ─── Plots ────────────────────────────────────────────────────────────────────

REGIME_COLORS = {
    0: "#26a69a",
    1: "#ef5350",
    2: "#ff9800",
    3: "#9c27b0",
}


def plot_regimes_on_price(df: pd.DataFrame, labels: np.ndarray,
                          valid_idx: pd.Index, regime_names: dict[int, str],
                          save: bool = False):
    tagged = df.loc[valid_idx].copy()
    tagged[REGIME_COL] = labels

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Price + regime background
    ax = axes[0]
    ax.plot(tagged.index, tagged["close"], lw=0.7, color="#1976d2", zorder=3)
    prev_r = None
    seg_start = tagged.index[0]
    for dt, r in tagged[REGIME_COL].items():
        if r != prev_r and prev_r is not None:
            ax.axvspan(seg_start, dt,
                       color=REGIME_COLORS.get(prev_r, "gray"), alpha=0.18)
            seg_start = dt
        prev_r = r
    ax.axvspan(seg_start, tagged.index[-1],
               color=REGIME_COLORS.get(prev_r, "gray"), alpha=0.18)

    # Legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=REGIME_COLORS.get(k, "gray"), alpha=0.5,
                               label=f"{k}: {v}")
               for k, v in regime_names.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    ax.set_title("BTC Close + Regime Background")

    # CVD zscore coloured by regime
    axes[1].scatter(tagged.index, tagged["cvd_zscore"],
                    c=[REGIME_COLORS.get(r, "gray") for r in tagged[REGIME_COL]],
                    s=1, alpha=0.6)
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_title("CVD Z-Score (coloured by regime)")

    # Funding rate
    axes[2].plot(tagged.index, tagged["funding_rate"] * 100, lw=0.7, color="#e91e63")
    axes[2].axhline(0, color="black", lw=0.7, linestyle="--")
    axes[2].set_title("Funding Rate (%)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    _save(fig, "regime_price.png", save)


def plot_regime_stats(stats: pd.DataFrame, regime_names: dict[int, str],
                      save: bool = False):
    display_cols = ["realized_vol_20b", "cvd_zscore", "funding_rate",
                    "delta_ratio", "strength_vol_adj"]
    display_cols = [c for c in display_cols if c in stats.columns]

    fig, axes = plt.subplots(1, len(display_cols), figsize=(4 * len(display_cols), 5))
    if len(display_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, display_cols):
        vals   = stats[col]
        colors = [REGIME_COLORS.get(i, "gray") for i in stats.index]
        xlabs  = [f"{i}\n{regime_names.get(i,'')}" for i in stats.index]
        ax.bar(xlabs, vals, color=colors, alpha=0.8)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(col, fontsize=9)

    plt.suptitle("Regime Mean Feature Values", fontsize=12)
    plt.tight_layout()
    _save(fig, "regime_stats.png", save)


def plot_regime_ic(ic_df: pd.DataFrame, regime_names: dict[int, str],
                   save: bool = False):
    feat_cols = [c for c in ic_df.columns if c != "n"]
    n_reg     = len(ic_df) - 1  # exclude GLOBAL row

    fig, ax = plt.subplots(figsize=(12, 5))
    x        = np.arange(len(feat_cols))
    width    = 0.8 / len(ic_df)

    for i, (regime, row) in enumerate(ic_df.iterrows()):
        vals   = row[feat_cols].values.astype(float)
        color  = REGIME_COLORS.get(regime, "#555555") if regime != "GLOBAL" else "#333333"
        label  = f"{regime}: {regime_names.get(regime,'')}" if regime != "GLOBAL" else "GLOBAL"
        offset = (i - len(ic_df) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               color=color, alpha=0.8, label=label)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(feat_cols, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Spearman IC vs strength_vol_adj")
    ax.set_title("Feature IC by Regime (vs Global)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, "regime_ic.png", save)


def _save(fig, name: str, save: bool):
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        p = OUT_DIR / name
        fig.savefig(p, bbox_inches="tight")
        print(f"  [saved] {p.name}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# ─── Write regime to DB + Parquet ─────────────────────────────────────────────

def save_regime_to_db(df: pd.DataFrame, labels: np.ndarray,
                      valid_idx: pd.Index, regime_names: dict[int, str]):
    from shared.db import get_db_conn

    tagged = df.loc[valid_idx].copy()
    tagged[REGIME_COL]      = labels
    tagged["regime_name"]   = tagged[REGIME_COL].map(regime_names)
    tagged["ts_open_col"]   = (tagged.index.astype(np.int64) // 1_000_000).astype(np.int64)

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # Add columns if not exists
            for col, typedef in [("regime", "TINYINT DEFAULT NULL"),
                                  ("regime_name", "VARCHAR(30) DEFAULT NULL")]:
                try:
                    cur.execute(f"ALTER TABLE features_15m ADD COLUMN {col} {typedef}")
                except Exception as e:
                    if "duplicate" in str(e).lower():
                        pass

            sql = """
                UPDATE features_15m
                SET regime = %s, regime_name = %s
                WHERE symbol = 'BTC-USD' AND ts_open = %s
            """
            params = [
                (int(r[REGIME_COL]), str(r["regime_name"]), int(r["ts_open_col"]))
                for _, r in tagged.iterrows()
            ]
            cur.executemany(sql, params)
        conn.commit()
        print(f"  Written {len(params)} regime labels to features_15m")
    finally:
        conn.close()


def save_regime_to_parquet(df: pd.DataFrame, labels: np.ndarray,
                            valid_idx: pd.Index, regime_names: dict[int, str]):
    df_out = pd.read_parquet(PARQUET)
    df_out["dt"] = pd.to_datetime(df_out["ts_open"], unit="ms", utc=True)
    df_out = df_out.set_index("dt")

    tagged = pd.Series(labels, index=valid_idx, name=REGIME_COL)
    named  = tagged.map(regime_names).rename("regime_name")
    df_out[REGIME_COL]    = tagged
    df_out["regime_name"] = named
    df_out = df_out.reset_index(drop=True)
    df_out.to_parquet(PARQUET, index=False, compression="snappy")
    print(f"  Parquet updated: {PARQUET.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=N_STATES_DEF)
    ap.add_argument("--save",     action="store_true")
    args = ap.parse_args()

    print("Loading data...")
    df = load()

    print("Preparing features...")
    X, valid_idx = prepare_features(df)
    print(f"  {len(X)} bars with complete regime features")

    print(f"\nFitting HMM ({args.n_states} states, {len(REGIME_FEATURES)} features)...")
    model  = fit_hmm(X, n_states=args.n_states)
    labels = decode_regimes(model, X)
    print(f"  Converged: {model.monitor_.converged}")
    print(f"  Log-likelihood: {model.score(X):.2f}")

    print("\nCharacterising regimes...")
    stats        = characterise(df, labels, valid_idx, args.n_states)
    regime_names = assign_regime_names(stats)

    print("\nRegime statistics:")
    print(stats[["realized_vol_20b", "cvd_zscore", "funding_rate",
                  "delta_ratio", "strength_vol_adj",
                  "count", "pct", "avg_duration_bars"]].to_string())
    print("\nRegime names assigned:")
    for k, v in regime_names.items():
        row = stats.loc[k]
        print(f"  State {k} ({v}): {row['pct']}% of bars, "
              f"avg duration {row['avg_duration_bars']} bars "
              f"({row['avg_duration_bars'] * 15 / 60:.1f}h)")

    print("\nRegime-conditional IC vs strength_vol_adj:")
    ic_df = regime_ic(df, labels, valid_idx)
    print(ic_df.to_string())

    print("\nSaving regime labels...")
    save_regime_to_db(df, labels, valid_idx, regime_names)
    save_regime_to_parquet(df, labels, valid_idx, regime_names)

    print("\nPlotting...")
    plot_regimes_on_price(df, labels, valid_idx, regime_names, save=args.save)
    plot_regime_stats(stats, regime_names,  save=args.save)
    plot_regime_ic(ic_df, regime_names,     save=args.save)

    print("\nRegime Detection complete.")
    return labels, regime_names, ic_df


if __name__ == "__main__":
    main()
