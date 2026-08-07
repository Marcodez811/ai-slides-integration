"""Post-extraction, side-effect-free enrichment contracts."""

from .image_classifier import (
    ImageClassification,
    ImageClassifier,
    InMemoryClassificationCache,
    classification_cache_key,
)
from .table_profiler import (
    ChartCandidate,
    ChartSeries,
    ProfiledCell,
    TableProfile,
    profile_table,
    profile_tables,
)

__all__ = [
    "ChartCandidate", "ChartSeries", "ImageClassification", "ImageClassifier",
    "InMemoryClassificationCache", "ProfiledCell", "TableProfile",
    "classification_cache_key", "profile_table", "profile_tables",
]
