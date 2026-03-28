"""
Abstract interface for all score models.
Every model (rule-based, AI, LDC, ...) must implement this contract.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoreOutput:
    """Standardised output from any scoring model."""
    reversal_score:     float
    continuation_score: float
    final_bias:         str            # "reversal" | "continuation" | "neutral"
    confidence:         float          # abs(rev − cont) / (rev + cont)
    risk_adj_score:     float          # confidence-weighted display score (0–100)
    signal:             int            # 1 = strong signal, 0 = normal
    upper_band:         Optional[float] = None
    lower_band:         Optional[float] = None


class ScoreModel(ABC):
    """
    Base class for all scoring models.

    Subclasses implement compute_score(features, config) → ScoreOutput.
    The interface is intentionally minimal so models can diverge internally
    (rule tables, ML inference, statistical tests, etc.).
    """

    @abstractmethod
    def compute_score(self, features: dict, config: dict) -> ScoreOutput:
        """
        Produce a score from assembled features.

        Args:
            features: dict from feature_assembler.assemble_features()
            config:   dict from RunnerConfig / ChartConfig
                      relevant keys: band_window, band_n_sigma

        Returns:
            ScoreOutput
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique string identifier for this model (stored in DB)."""
        ...
