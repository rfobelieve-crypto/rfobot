# -*- coding: utf-8 -*-
"""Reverse proof for the Stage 5 concentration gate.

A gate that has never gone red is not known to measure anything
(mistake.md 2026-09-03).  Break the denominator and it must fire.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import labels as L

lb = pd.read_parquet(HERE.parent / "data" / "labels" / "BTC.parquet")
ok, _ = L.run_asserts(lb)
print("baseline BTC:", "GREEN" if not ok else f"RED {ok}")
assert not ok, "baseline must be green before the reverse proof means anything"

for factor, name in ((0.5, "atr halved"), (0.2, "atr cut to a fifth"),
                     (2.0, "atr doubled")):
    m = lb.copy()
    for tau in L.TAUS:
        m[f"r_norm_{tau}"] = m[f"r_norm_{tau}"] / factor
    f, d = L.run_asserts(m)
    share = d["r_norm_3600"]["extreme_day_share"]
    print(f"  {name:20s} extreme-day share={share*100:6.3f}%  -> "
          f"{'RED (caught)' if f else 'GREEN (BLIND)'}")
    if factor < 1 and not f:
        raise SystemExit(f"gate is blind to '{name}'")
print("Stage 5 concentration gate: reverse proof PASS")
