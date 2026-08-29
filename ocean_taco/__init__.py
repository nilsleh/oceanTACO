"""OceanTACO 0.1: reproducible ocean sampling and native-coordinate retrieval."""

__author__ = "Nils Lehmann"
__version__ = "0.1.0"

from .catalog import CORE_DATASET_REPO_ID, CORE_DATASET_REVISION, CatalogConfig
from .filter import CoverageRequirement, QueryFilter, SelectedPairs, select_queryset
from .geobox import GeoBox, PatchSize, PatchSpec, Query, TimeRange
from .manifest import PatchSet, QuerySet
from .plot import plot_ocean_sample
from .registry import MODALITY_REGISTRY, ModalitySpec
from .sampling import QueryDraw, build_queryset, draw_queryset, replay_experiment

__all__ = [
    "CORE_DATASET_REPO_ID",
    "CORE_DATASET_REVISION",
    "CatalogConfig",
    "CoverageRequirement",
    "GeoBox",
    "MODALITY_REGISTRY",
    "ModalitySpec",
    "PatchSet",
    "PatchSize",
    "PatchSpec",
    "Query",
    "QueryDraw",
    "QueryFilter",
    "QuerySet",
    "SelectedPairs",
    "TimeRange",
    "build_queryset",
    "draw_queryset",
    "plot_ocean_sample",
    "replay_experiment",
    "select_queryset",
]
