"""
EDA for BTC-USD 15m feature dataset (Jan ~ Mar 2026).

Sections:
  1. Basic info — shape, dtypes, missing values
  2. Label distribution — up/down/neutral balance
  3. Price & return overview
  4. Flow features — CVD, delta_ratio, exchange_divergence
  5. Funding rate
  6. Feature correlation heatmap
  7. Feature vs label — mean feature value per label class
  8. Bull-bear score distribution

Run:
    python research/eda_15m.py
    python research/eda_15m.py --save   # save all plots to research/eda_charts/
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 120

PARQUET = Path(__file__).parent / "ml_data" / "BTC_USD_15m_tabular.parquet"
OUT_DIR = Path(__file__).parent / "eda_charts"

FLOW_FEATURES = [
    "delta_ratio", "cvd_zscore", "large_delta",
    "bnc_delta_ratio", "okx_delta_ratio", "exchange_divergence",
]
PRICE_FEATURES = ["return_1b", "realized_vol_20b", "macd"]
FUNDING_FEATURES = ["funding_rate", "funding_deviation", "funding_zscore"]
SCORE_FEATURES = ["bull_bear_score"]

TARGET = "label_15m"


def load() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()
    return df


# ── 1. Basic info ─────────────────────────────────────────────────────────────

def section_basic(df: pd.DataFrame):
    print("=" * 60)
    print("1. BASIC INFO")
    print("=" * 60)
    print(f"Shape      : {df.shape}")
    print(f"Date range : {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"Bars/day   : {len(df) / ((df.index[-1] - df.index[0]).days + 1):.1f}")

    miss = df.isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    print(f"\nMissing values ({len(miss)} cols with >0% missing):")
    print(miss.apply(lambda x: f"{x:.1%}").to_string())


# ── 2. Label distribution ─────────────────────────────────────────────────────

def section_labels(df: pd.DataFrame, save: bool = False):
    print("\n" + "=" * 60)
    print("2. LABEL DISTRIBUTION")
    print("=" * 60)

    for col in ["label_15m", "label_1h"]:
        counts = df[col].value_counts()
        total  = counts.sum()
        print(f"\n{col}:")
        for lbl, n in counts.items():
            print(f"  {lbl:8s} {n:5d}  ({n/total:.1%})")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, ["label_15m", "label_1h"]):
        counts = df[col].value_counts()
        colors = {"up": "#26a69a", "down": "#ef5350", "neutral": "#78909c"}
        ax.bar(counts.index, counts.values,
               color=[colors.get(l, "gray") for l in counts.index])
        ax.set_title(col)
        ax.set_ylabel("count")
        for i, (lbl, n) in enumerate(counts.items()):
            ax.text(i, n + 10, f"{n/counts.sum():.1%}", ha="center", fontsize=9)
    plt.tight_layout()
    _show_or_save(fig, "label_distribution.png", save)


# ── 3. Price & return overview ────────────────────────────────────────────────

def section_price(df: pd.DataFrame, save: bool = False):
    print("\n" + "=" * 60)
    print("3. PRICE & RETURN OVERVIEW")
    print("=" * 60)

    ret = df["return_1b"].dropna()
    print(f"return_1b  mean={ret.mean():.5f}  std={ret.std():.5f}  "
          f"skew={ret.skew():.2f}  kurt={ret.kurtosis():.2f}")
    print(f"           min={ret.min():.4f}  max={ret.max():.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Price
    axes[0].plot(df.index, df["close"], lw=0.8, color="#1976d2")
    axes[0].set_title("BTC Close Price (15m)")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # Return distribution
    axes[1].hist(ret, bins=200, color="#7b1fa2", alpha=0.7, edgecolor="none")
    axes[1].axvline(0, color="red", lw=1)
    axes[1].set_title("15m Return Distribution")
    axes[1].set_xlabel("return")

    # Realized vol
    axes[2].plot(df.index, df["realized_vol_20b"], lw=0.8, color="#f57c00")
    axes[2].set_title("Realized Vol (20-bar rolling std of returns)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    _show_or_save(fig, "price_overview.png", save)


# ── 4. Flow features ──────────────────────────────────────────────────────────

def section_flow(df: pd.DataFrame, save: bool = False):
    print("\n" + "=" * 60)
    print("4. FLOW FEATURES")
    print("=" * 60)

    for col in FLOW_FEATURES:
        s = df[col].dropna()
        if len(s) == 0:
            print(f"  {col:30s} ALL NaN")
            continue
        print(f"  {col:30s} mean={s.mean():+.4f}  std={s.std():.4f}  "
              f"min={s.min():.4f}  max={s.max():.4f}")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    for ax, col in zip(axes, FLOW_FEATURES):
        s = df[col].dropna()
        if len(s) == 0:
            ax.set_title(f"{col} (all NaN)")
            continue
        ax.plot(df.index, df[col], lw=0.6, alpha=0.8)
        ax.axhline(0, color="red", lw=0.8, linestyle="--")
        ax.set_title(col)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    _show_or_save(fig, "flow_features.png", save)


# ── 5. Funding rate ───────────────────────────────────────────────────────────

def section_funding(df: pd.DataFrame, save: bool = False):
    print("\n" + "=" * 60)
    print("5. FUNDING RATE")
    print("=" * 60)

    fr = df["funding_rate"].dropna()
    print(f"  Settlements in range : {(fr != fr.shift()).sum()} changes")
    print(f"  Mean funding rate    : {fr.mean():.6f}  ({fr.mean()*3*365:.2%} annualized)")
    print(f"  Max                  : {fr.max():.6f}")
    print(f"  Min                  : {fr.min():.6f}")
    print(f"  % positive           : {(fr > 0).mean():.1%}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    axes[0].plot(df.index, df["funding_rate"] * 100, lw=0.8, color="#e91e63")
    axes[0].axhline(0, color="black", lw=0.8, linestyle="--")
    axes[0].set_title("Funding Rate (%) — forward-filled every 8h")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    axes[1].hist(fr * 100, bins=80, color="#e91e63", alpha=0.7, edgecolor="none")
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_title("Funding Rate Distribution (%)")
    axes[1].set_xlabel("funding rate (%)")

    plt.tight_layout()
    _show_or_save(fig, "funding_rate.png", save)


# ── 6. Correlation heatmap ────────────────────────────────────────────────────

def section_correlation(df: pd.DataFrame, save: bool = False):
    print("\n" + "=" * 60)
    print("6. FEATURE CORRELATION")
    print("=" * 60)

    key_cols = (FLOW_FEATURES + PRICE_FEATURES + FUNDING_FEATURES
                + SCORE_FEATURES + ["future_return_15m"])
    key_cols = [c for c in key_cols if c in df.columns]

    corr = df[key_cols].corr()

    # Top correlations with future_return_15m
    if "future_return_15m" in corr:
        top = (corr["future_return_15m"]
               .drop("future_return_15m")
               .abs()
               .sort_values(ascending=False)
               .head(10))
        print("\nTop 10 correlations with future_return_15m:")
        for feat, val in top.items():
            raw = corr["future_return_15m"][feat]
            print(f"  {feat:30s}  {raw:+.4f}")

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", center=0,
                cmap="RdBu_r", linewidths=0.3, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    _show_or_save(fig, "correlation_heatmap.png", save)


# ── 7. Mean feature value per label ──────────────────────────────────────────

def section_feature_vs_label(df: pd.DataFrame, save: bool = False):
    print("\n" + "=" * 60)
    print("7. FEATURE MEAN BY LABEL (label_15m)")
    print("=" * 60)

    key_cols = FLOW_FEATURES + ["funding_zscore", "bull_bear_score"]
    key_cols = [c for c in key_cols if c in df.columns]

    labeled = df[df[TARGET].notna()]
    stats = labeled.groupby(TARGET)[key_cols].mean()
    print(stats.round(4).to_string())

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    colors = {"up": "#26a69a", "down": "#ef5350", "neutral": "#78909c"}

    for ax, col in zip(axes, key_cols):
        vals = labeled.groupby(TARGET)[col].mean()
        ax.bar(vals.index, vals.values,
               color=[colors.get(l, "gray") for l in vals.index])
        ax.set_title(col, fontsize=9)
        ax.axhline(0, color="black", lw=0.8, linestyle="--")

    for ax in axes[len(key_cols):]:
        ax.set_visible(False)

    plt.suptitle("Mean Feature Value by Label (label_15m)", fontsize=12)
    plt.tight_layout()
    _show_or_save(fig, "feature_vs_label.png", save)


# ── 8. Bull-bear score ────────────────────────────────────────────────────────

def section_bull_bear(df: pd.DataFrame, save: bool = False):
    print("\n" + "=" * 60)
    print("8. BULL-BEAR SCORE")
    print("=" * 60)

    bs = df["bull_bear_score"].dropna()
    print(f"  mean={bs.mean():.1f}  std={bs.std():.1f}  "
          f"median={bs.median():.1f}  min={bs.min():.1f}  max={bs.max():.1f}")

    # Score accuracy: does score>50 predict up?
    labeled = df[df[TARGET].notna()].copy()
    labeled["pred"] = np.where(labeled["bull_bear_score"] > 50, "up", "down")
    labeled["correct"] = (
        ((labeled["pred"] == "up")   & (labeled[TARGET] == "up")) |
        ((labeled["pred"] == "down") & (labeled[TARGET] == "down"))
    )
    excl_neutral = labeled[labeled[TARGET] != "neutral"]
    acc = excl_neutral["correct"].mean()
    print(f"  bull_bear_score>50 directional accuracy (excl neutral): {acc:.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df.index, df["bull_bear_score"], lw=0.6, alpha=0.8, color="#ff6f00")
    axes[0].axhline(50, color="red", lw=1, linestyle="--")
    axes[0].set_title("Bull-Bear Score over Time")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    axes[1].hist(bs, bins=60, color="#ff6f00", alpha=0.7, edgecolor="none")
    axes[1].axvline(50, color="red", lw=1)
    axes[1].set_title("Bull-Bear Score Distribution")
    axes[1].set_xlabel("score (0=bearish, 100=bullish)")

    plt.tight_layout()
    _show_or_save(fig, "bull_bear_score.png", save)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _show_or_save(fig, name: str, save: bool):
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / name
        fig.savefig(path, bbox_inches="tight")
        print(f"  [saved] {path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="Save charts to eda_charts/")
    args = ap.parse_args()

    print(f"Loading {PARQUET.name} ...")
    df = load()

    section_basic(df)
    section_labels(df, args.save)
    section_price(df, args.save)
    section_flow(df, args.save)
    section_funding(df, args.save)
    section_correlation(df, args.save)
    section_feature_vs_label(df, args.save)
    section_bull_bear(df, args.save)

    print("\nEDA complete.")


if __name__ == "__main__":
    main()
