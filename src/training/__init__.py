"""Pipeline d'entraînement du lot P2 : cleaning, découpage, pipeline scikit-learn et validation.

Le protocole est réglé dans config/training.yaml et expliqué dans docs/protocole_validation.md.
"""

from .cleaning import clean
from .config import comparison_tags, load_config, load_contract, resolve_preset
from .pipeline import PipelineResult, build_pipeline, run_pipeline
from .split import Split, temporal_split
from .validation import METRIC_KEYS, choose_decision_threshold, evaluate

__all__ = [
    "METRIC_KEYS",
    "PipelineResult",
    "Split",
    "build_pipeline",
    "choose_decision_threshold",
    "clean",
    "comparison_tags",
    "evaluate",
    "load_config",
    "load_contract",
    "resolve_preset",
    "run_pipeline",
    "temporal_split",
]
