"""
Model registry: maps string names → ScoreModel classes.
Callers only need to pass a string — no direct imports of model classes.
"""
from __future__ import annotations
from research.score_engine.interface import ScoreModel
from research.score_engine.rule_based import RuleBasedModel

MODEL_REGISTRY: dict[str, type[ScoreModel]] = {
    "rule_based": RuleBasedModel,
    # "ai_model":  AIModel,    # future
    # "ldc_model": LDCModel,   # future
}


def get_model(name: str) -> ScoreModel:
    """
    Return an instantiated ScoreModel by name.

    Raises:
        KeyError if name is not in registry.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown score model {name!r}. "
            f"Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]()


def list_models() -> list[str]:
    return list(MODEL_REGISTRY)
