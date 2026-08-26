"""Pinned Core-catalog configuration and thin catalog access helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CORE_DATASET_REPO_ID = "nilsleh/OceanTACO"
CORE_DATASET_REVISION = "878befc437a49cbf584353efc7346ebe705e743c"


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
    return tacoreader.load(config.resolved_catalog_url)
