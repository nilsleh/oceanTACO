"""Pinned Core-catalog configuration and thin catalog access helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CORE_DATASET_REPO_ID = "nilsleh/OceanTACO"
CORE_DATASET_REVISION = "4a3233f8f0d0a38bb85d8122043c9ffd3b772196"


@dataclass(frozen=True, slots=True)
class CatalogConfig:
    """Configuration for Core catalog access.

    A revision is pinned by default.  Selecting ``main`` or a local checkout is
    explicit, so a training manifest can always record the exact catalog it
    was built against.
    """

    repo_id: str = CORE_DATASET_REPO_ID
    revision: str = CORE_DATASET_REVISION
    taco_path: Path | str | None = None
    cache_dir: Path | str | None = None
    timeout_seconds: float = 30.0
    retries: int = 3

    def __post_init__(self) -> None:
        if not self.repo_id:
            raise ValueError("repo_id cannot be empty.")
        if not self.revision:
            raise ValueError("revision cannot be empty.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.retries < 0:
            raise ValueError("retries must be non-negative.")
        if self.taco_path is not None:
            object.__setattr__(self, "taco_path", Path(self.taco_path))
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))

    @property
    def resolved_catalog_url(self) -> str:
        """Return the configured local catalog path or immutable dataset root."""
        if self.taco_path is not None:
            return str(self.taco_path)
        return f"https://huggingface.co/datasets/{self.repo_id}/resolve/{self.revision}/"

    def to_dict(self) -> dict[str, object]:
        """Return configuration suitable for manifest provenance."""
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "taco_path": str(self.taco_path) if self.taco_path is not None else None,
            "cache_dir": str(self.cache_dir) if self.cache_dir is not None else None,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
        }


#: Catalog metadata that makes a directory self-describing to ``tacoreader``.
#: Small (about 0.7 MB in total) and fetched alongside the first remote read,
#: so that a populated snapshot directory is itself a valid ``taco_path``.
CATALOG_METADATA_FILES: tuple[str, ...] = (
    "COLLECTION.json",
    "METADATA/level0.parquet",
    "METADATA/level1.parquet",
    "METADATA/level2.parquet",
)


def materialise_catalog_metadata(config: CatalogConfig) -> Path:
    """Download the catalog's metadata files and return their snapshot directory.

    ``hf_hub_download`` preserves the repository layout, so fetching these four
    files leaves the same ``COLLECTION.json`` / ``DATA/`` / ``METADATA/`` shape
    a full local catalog has.  Granules fetched later land in the same tree,
    which is why remote access and a local copy are one layout at different
    levels of completeness rather than two mechanisms.
    """
    from huggingface_hub import hf_hub_download

    root: Path | None = None
    for filename in CATALOG_METADATA_FILES:
        path = Path(
            hf_hub_download(
                repo_id=config.repo_id,
                filename=filename,
                revision=config.revision,
                repo_type="dataset",
                cache_dir=config.cache_dir,
            )
        )
        if filename == "COLLECTION.json":
            root = path.parent
    assert root is not None
    return root


def load_catalog(config: CatalogConfig):
    """Load the configured TACO catalog without importing optional HF tooling eagerly."""
    try:
        import tacoreader
    except ImportError as error:  # pragma: no cover - exercised in clean installs
        raise ImportError("Catalog access requires tacoreader. Install ocean_taco base dependencies.") from error

    # tacoreader 2.4 exposes a pandas backend selector, while newer 2.x
    # releases already return the pandas-compatible catalog by default.
    if hasattr(tacoreader, "use"):
        tacoreader.use("pandas")
    if config.taco_path is not None:
        return tacoreader.load(str(config.taco_path))
    # Materialise the metadata so the snapshot directory is a complete, valid
    # taco_path, but keep loading from the URL: reading the catalog from the
    # snapshot would make tacoreader emit local paths for granules that have
    # not been downloaded yet, and retrieval would fail instead of fetching.
    materialise_catalog_metadata(config)
    return tacoreader.load(config.resolved_catalog_url)
