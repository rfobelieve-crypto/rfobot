"""
LAYER 3: edge_query.py

Research interface for edge discovery.

Core question: "Which conditions lead to statistically significant edge?"

Usage (Python REPL / notebook):

    from research.pipeline.edge_query import EdgeQuery
    q = EdgeQuery()

    # All BSL events
    df = q.fetch(event_side="BSL")
    print(q.summary(df))

    # BSL + rising OI + positive delta → reversal probability?
    df = q.fetch(event_side="BSL", oi_change_1bar=(0, None), delta_1bar=(0, None))
    print(q.summary(df))

    # Full breakdown
    print(q.breakdown(group_by=["event_side", "label"]))
"""
from __future__ import annotations

import sys
import os
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.db import get_db_conn

logger = logging.getLogger(__name__)


class EdgeQuery:
    """Query interface for event_features table."""

    def fetch(
        self,
        symbol:        str | None = None,
        event_side:    str | None = None,   # 'BSL' | 'SSL'
        label:         str | None = None,   # 'reversal' | 'continuation' | 'neutral'
        delta_1bar:    tuple | None = None, # (min, max) or (min, None) or (None, max)
        oi_change_1bar: tuple | None = None,
        funding_zscore: tuple | None = None,
        labeled_only:  bool = True,
    ) -> pd.DataFrame:
        """
        Load filtered events from event_features.

        Range filters: (min, max). Use None for unbounded.
        Example: delta_1bar=(0, None) → positive delta only.
        """
        conditions = []
        params     = []

        if labeled_only:
            conditions.append("label IS NOT NULL")

        if symbol:
            conditions.append("symbol = %s"); params.append(symbol)
        if event_side:
            conditions.append("event_side = %s"); params.append(event_side)
        if label:
            conditions.append("label = %s"); params.append(label)

        for col, rng in [
            ("delta_1bar",    delta_1bar),
            ("oi_change_1bar", oi_change_1bar),
            ("funding_zscore", funding_zscore),
        ]:
            if rng is not None:
                lo, hi = rng
                if lo is not None:
                    conditions.append(f"{col} >= %s"); params.append(lo)
                if hi is not None:
                    conditions.append(f"{col} <= %s"); params.append(hi)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql   = f"SELECT * FROM event_features {where} ORDER BY trigger_ts_ms"

        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])
        for col in ["delta_1bar","delta_2bar","delta_4bar",
                    "oi_change_1bar","oi_change_2bar",
                    "funding_rate","funding_zscore",
                    "pressure","delta_price_divergence","flow_acceleration",
                    "return_1bar","return_2bar","return_4bar"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # ── Statistics ────────────────────────────────────────────────────────────

    def summary(self, df: pd.DataFrame, outcome_col: str = "return_4bar") -> dict:
        """
        Compute edge statistics for a filtered DataFrame.

        Returns:
            n, win_rate, avg_return, std, sharpe, percentiles
        """
        if df.empty or outcome_col not in df.columns:
            return {"n": 0}

        clean = df[df[outcome_col].notna() & df["label"].notna()].copy()
        if clean.empty:
            return {"n": 0}

        # win_rate = P(label == 'reversal') — "winning" = expected direction
        # For both BSL and SSL, "reversal" is the profitable ICT trade
        wins     = (clean["label"] == "reversal").sum()
        win_rate = wins / len(clean)

        ret      = clean[outcome_col]
        avg      = float(ret.mean())
        std      = float(ret.std())
        sharpe   = avg / std if std > 0 else 0.0
        pcts     = ret.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).to_dict()

        return {
            "n":          len(clean),
            "win_rate":   round(win_rate, 4),
            "avg_return": round(avg * 100, 4),   # in %
            "std":        round(std * 100, 4),
            "sharpe":     round(sharpe, 4),
            "p10":        round(pcts[0.10] * 100, 4),
            "p25":        round(pcts[0.25] * 100, 4),
            "p50":        round(pcts[0.50] * 100, 4),
            "p75":        round(pcts[0.75] * 100, 4),
            "p90":        round(pcts[0.90] * 100, 4),
            "label_dist": clean["label"].value_counts().to_dict(),
        }

    def breakdown(
        self,
        group_by:    list[str] = ("event_side",),
        outcome_col: str       = "return_4bar",
    ) -> pd.DataFrame:
        """
        Group events and compute summary statistics per group.
        """
        df = self.fetch(labeled_only=False)
        if df.empty:
            return pd.DataFrame()

        records = []
        for keys, grp in df.groupby(list(group_by)):
            stat = self.summary(grp, outcome_col)
            if isinstance(keys, str):
                keys = (keys,)
            record = dict(zip(group_by, keys))
            record.update(stat)
            records.append(record)

        return pd.DataFrame(records)

    def condition_test(
        self,
        event_side:   str,
        conditions:   dict,
        outcome_col:  str = "return_4bar",
    ) -> dict:
        """
        Test a specific condition set vs. the unconditional baseline.

        conditions: dict of {col: (min, max)}
        Returns: {baseline: summary, conditional: summary, lift: float}

        Example:
            q.condition_test(
                "BSL",
                {"delta_1bar": (0.05, None), "oi_change_1bar": (0, None)},
            )
        """
        base = self.summary(self.fetch(event_side=event_side), outcome_col)
        cond_df = self.fetch(event_side=event_side, **conditions)
        cond    = self.summary(cond_df, outcome_col)

        lift = (
            round(cond["win_rate"] - base["win_rate"], 4)
            if base.get("n", 0) > 0 and cond.get("n", 0) > 0
            else None
        )

        return {"baseline": base, "conditional": cond, "win_rate_lift": lift}

    def print_report(self, event_side: str | None = None):
        """Print a quick overview report."""
        df = self.fetch(event_side=event_side, labeled_only=False)
        if df.empty:
            print("No events found.")
            return

        print(f"\n{'='*50}")
        print(f"  EVENT FEATURES REPORT  ({len(df)} events total)")
        print(f"{'='*50}")

        for side in (["BSL","SSL"] if event_side is None else [event_side]):
            sub = df[df["event_side"] == side]
            if sub.empty:
                continue
            stat = self.summary(sub)
            print(f"\n  {side}  (n={stat.get('n',0)})")
            print(f"  Win rate (reversal): {stat.get('win_rate',0)*100:.1f}%")
            print(f"  Avg 4h return:       {stat.get('avg_return',0):+.3f}%")
            print(f"  Std:                 {stat.get('std',0):.3f}%")
            print(f"  Sharpe:              {stat.get('sharpe',0):.3f}")
            print(f"  p10/p50/p90:         {stat.get('p10',0):+.2f}% / "
                  f"{stat.get('p50',0):+.2f}% / {stat.get('p90',0):+.2f}%")
            if "label_dist" in stat:
                print(f"  Labels:              {stat['label_dist']}")
        print()
