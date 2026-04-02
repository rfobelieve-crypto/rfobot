"""
Global configuration for the research platform.
All tunable parameters live here — nowhere else.
"""
from dataclasses import dataclass, field
from typing import Dict, List

# ── Timeframe helpers ────────────────────────────────────────────────────────

TF_SECONDS: Dict[str, int] = {
    "15m": 900,
    "1h":  3_600,
    "4h":  14_400,
    "1d":  86_400,
}

TF_MS: Dict[str, int] = {k: v * 1000 for k, v in TF_SECONDS.items()}


# ── Config dataclasses ───────────────────────────────────────────────────────

@dataclass
class ColorConfig:
    """Chart colors. Change here, nowhere else."""
    reversal:    str = "#26a69a"   # teal-green
    continuation: str = "#ef5350"  # red
    neutral:     str = "#607d8b"   # grey-blue
    signal:      str = "#ffeb3b"   # yellow signal dot
    candle_up:   str = "#26a69a"
    candle_down: str = "#ef5350"
    band_line:   str = "#546e7a"
    band_fill:   str = "rgba(84,110,122,0.15)"
    bg_color:    str = "#131722"
    grid_color:  str = "#2a2e39"
    text_color:  str = "#d1d4dc"


@dataclass
class ChartConfig:
    """Per-chart settings passed to build_chart()."""
    symbol:       str         = "BTC-USD"
    timeframe:    str         = "1h"
    score_field:  str         = "risk_adj_score"  # column to display on score bar
    lookback_days: int        = 7
    band_window:  int         = 20
    band_n_sigma: float       = 2.0
    show_oi:      bool        = False
    show_cvd:     bool        = False
    annotation:   str         = "Source: @rfo"
    colors:       ColorConfig = field(default_factory=ColorConfig)


@dataclass
class RunnerConfig:
    """Settings for the bar runner loop."""
    symbols:          List[str] = field(default_factory=lambda: ["BTC-USD"])
    timeframes:       List[str] = field(default_factory=lambda: ["1h", "4h"])
    lookback_days:    int       = 7
    interval_seconds: int       = 60
    band_window:      int       = 20
    band_n_sigma:     float     = 2.0
