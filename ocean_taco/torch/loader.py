"""The shipped, worker-safe access adapter for logical OceanTACO patches."""

from __future__ import annotations

import os
import weakref
from collections.abc import Iterable
from typing import Any

from ..access import LocalCacheBackend
from ..catalog import CatalogConfig
from ..geobox import PatchSpec
from ..retrieve import (
    AssetPlan,
    load_hf_dataset,
    load_multisource_time_series_nc,
    load_planned_multisource_time_series_nc,
    plan_multisource_assets,
)


_LIVE_CORE_LOADERS: weakref.WeakSet[CoreSourceLoader] = weakref.WeakSet()
_FORK_GUARD_REGISTERED = False


def _drop_catalogs_before_fork() -> None:
    """Remove native catalog state before POSIX creates a child process."""
    for loader in tuple(_LIVE_CORE_LOADERS):
        loader._drop_for_fork()


def _register_fork_guard() -> None:
    global _FORK_GUARD_REGISTERED
    if not _FORK_GUARD_REGISTERED and hasattr(os, "register_at_fork"):
        os.register_at_fork(before=_drop_catalogs_before_fork)
        _FORK_GUARD_REGISTERED = True


class PlannedSourceLoader:
    """Worker-safe fetcher backed exclusively by parent-resolved assets."""

    def __init__(self, config: CatalogConfig, plan: AssetPlan) -> None:
        self.config = config
        self.plan = dict(plan)
        self._backend: LocalCacheBackend | None = None
        self._owner_pid: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Never pickle an HDF5 cache handle into a spawned worker."""
        return {
            "config": self.config,
            "plan": self.plan,
            "_backend": None,
            "_owner_pid": None,
        }

    def _reset_for_process(self) -> None:
        pid = os.getpid()
        if self._owner_pid == pid:
            return
        self._backend = None
        self._owner_pid = pid

    def worker_init(self) -> None:
        """Start each worker with no inherited HDF5 cache handle."""
        self._backend = None
        self._owner_pid = os.getpid()

    def _backend_for_process(self) -> LocalCacheBackend:
        self._reset_for_process()
        if self._backend is None:
            self._backend = LocalCacheBackend(
                self.config.cache_dir, revision=self.config.revision
            )
        return self._backend

    def _load_tokens(self, tokens: Iterable[str], patch: PatchSpec) -> dict[str, Any]:
        return load_planned_multisource_time_series_nc(
            self.plan,
            tokens,
            patch.footprint,
            patch.context,
            config=self.config,
            backend=self._backend_for_process(),
        )

    def load(self, token: str, patch: PatchSpec):
        """Load one source using only the serialised parent-side plan."""
        return self._load_tokens((token,), patch)[token]

    def load_pair(
        self, components: tuple[str, str], patch: PatchSpec
    ) -> dict[str, Any] | None:
        """Load paired variables together while still avoiding catalog access."""
        values = self._load_tokens(components, patch)
        first, second = components
        if values[first] is None or values[second] is None:
            return None
        return {first: values[first], second: values[second]}


class CoreSourceLoader:
    """Parent-side Core planner with a safe lazy fallback for direct use.

    :meth:`plan` resolves all catalog rows before workers exist and returns a
    :class:`PlannedSourceLoader` containing only paths/URLs.  Calling
    :meth:`load` directly retains the legacy lazy path for callers that cannot
    plan in advance, but the fork guard drops native catalog state before a
    child is created.
    """

    def __init__(self, config: CatalogConfig) -> None:
        if config.cache_dir is None:
            raise ValueError(
                "CoreSourceLoader requires CatalogConfig(cache_dir=...) for released Mode-1 access."
            )
        self.config = config
        self._catalog: Any | None = None
        self._backend: LocalCacheBackend | None = None
        self._owner_pid: int | None = None
        _LIVE_CORE_LOADERS.add(self)
        _register_fork_guard()

    def __getstate__(self) -> dict[str, Any]:
        """Never pickle an open catalog or HDF5 handle into a spawned worker."""
        return {"config": self.config, "_catalog": None, "_backend": None, "_owner_pid": None}

    def _drop_for_fork(self) -> None:
        """Discard native state in the parent before a child can inherit it."""
        self._catalog = None
        self._backend = None
        self._owner_pid = None

    def _reset_for_process(self) -> None:
        pid = os.getpid()
        if self._owner_pid == pid:
            return
        self._catalog = None
        self._backend = None
        self._owner_pid = pid

    def worker_init(self) -> None:
        """Forget parent state before an unplanned worker performs I/O."""
        self._drop_for_fork()
        self._owner_pid = os.getpid()

    def _catalog_for_process(self):
        self._reset_for_process()
        if self._catalog is None:
            self._catalog = load_hf_dataset(self.config)
        return self._catalog

    def _backend_for_process(self) -> LocalCacheBackend:
        self._reset_for_process()
        if self._backend is None:
            self._backend = LocalCacheBackend(
                self.config.cache_dir, revision=self.config.revision
            )
        return self._backend

    def plan(
        self, requests: Iterable[tuple[str, PatchSpec]]
    ) -> PlannedSourceLoader:
        """Resolve every unique ``(token, day, footprint)`` once in the parent."""
        resolved = plan_multisource_assets(
            self._catalog_for_process(),
            ((token, patch.footprint, patch.context) for token, patch in requests),
        )
        return PlannedSourceLoader(self.config, resolved)

    def load(self, token: str, patch: PatchSpec):
        """Legacy lazy retrieval for callers that deliberately skip planning."""
        return load_multisource_time_series_nc(
            self._catalog_for_process(),
            (token,),
            patch.footprint,
            patch.context,
            config=self.config,
            backend=self._backend_for_process(),
        )[token]

    def load_pair(
        self, components: tuple[str, str], patch: PatchSpec
    ) -> dict[str, Any] | None:
        """Load paired variables once when they share the same Core asset."""
        first, second = components
        dataset = self.load(first, patch)
        if dataset is None:
            return None
        return {first: dataset, second: dataset}
