"""The shipped, worker-safe access adapter for logical OceanTACO patches."""

from __future__ import annotations

import os
from typing import Any

from ..access import LocalCacheBackend
from ..catalog import CatalogConfig
from ..geobox import PatchSpec
from ..retrieve import load_hf_dataset, load_multisource_time_series_nc


class CoreSourceLoader:
    """Load released Core sources through a per-worker immutable cache.

    The adapter contains only serialisable catalog configuration at
    construction.  Catalog state and HDF5 handles are created lazily in the
    calling process and are dropped when a worker starts, so neither a catalog
    connection nor a file descriptor is shared across DataLoader workers.
    """

    def __init__(self, config: CatalogConfig) -> None:
        if config.cache_dir is None:
            raise ValueError("CoreSourceLoader requires CatalogConfig(cache_dir=...) for released Mode-1 access.")
        self.config = config
        self._catalog: Any | None = None
        self._backend: LocalCacheBackend | None = None
        self._owner_pid: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Never pickle an open catalog or HDF5 handle into a spawned worker."""
        return {"config": self.config, "_catalog": None, "_backend": None, "_owner_pid": None}

    def _reset_for_process(self) -> None:
        pid = os.getpid()
        if self._owner_pid == pid:
            return
        self._catalog = None
        self._backend = None
        self._owner_pid = pid

    def worker_init(self) -> None:
        """Forget parent state before a DataLoader worker performs any I/O."""
        self._catalog = None
        self._backend = None
        self._owner_pid = os.getpid()

    def _catalog_for_process(self):
        self._reset_for_process()
        if self._catalog is None:
            self._catalog = load_hf_dataset(self.config)
        return self._catalog

    def _backend_for_process(self) -> LocalCacheBackend:
        self._reset_for_process()
        if self._backend is None:
            self._backend = LocalCacheBackend(self.config.cache_dir, revision=self.config.revision)
        return self._backend

    def load(self, token: str, patch: PatchSpec):
        """Return a decoded native source stack or ``None`` for a logical patch."""
        return load_multisource_time_series_nc(
            self._catalog_for_process(),
            (token,),
            patch.footprint,
            patch.context,
            config=self.config,
            backend=self._backend_for_process(),
        )[token]

    def load_pair(self, components: tuple[str, str], patch: PatchSpec) -> dict[str, Any] | None:
        """Load paired variables once when they share the same Core asset."""
        first, second = components
        dataset = self.load(first, patch)
        if dataset is None:
            return None
        return {first: dataset, second: dataset}
