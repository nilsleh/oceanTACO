"""The shipped, worker-safe access adapter for logical OceanTACO patches."""

from __future__ import annotations

import os
import weakref
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

from ..access import LocalCacheBackend
from ..catalog import CatalogConfig
from ..geobox import PatchSpec
from ..registry import get_modality
from ..retrieve import (
    AssetPlan,
    _daily_cache_key,
    _date_string,
    _days,
    _load_planned_bbox_nc,
    load_hf_dataset,
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


def _required_variables(tokens: Iterable[str]) -> dict[str, tuple[str, ...] | None]:
    """Project dense fields by shared filename; retain all ragged point fields."""
    sources = tuple(get_modality(token) for token in tokens)
    by_file: dict[str, set[str]] = {}
    for source in sources:
        by_file.setdefault(source.filename, set()).add(source.primary_variable)
    return {
        source.token: None
        if source.is_points
        else tuple(sorted(by_file[source.filename]))
        for source in sources
    }


class PlannedSourceLoader:
    """Worker-safe fetcher backed exclusively by parent-resolved assets."""

    def __init__(self, config: CatalogConfig, plan: AssetPlan) -> None:
        self.config = config
        self.plan = dict(plan)
        self._daily_cache: dict | None = None
        self._batch_variables: dict | None = None
        self._backend: LocalCacheBackend | None = None
        self._owner_pid: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Never pickle an HDF5 cache handle into a spawned worker."""
        return {
            "config": self.config,
            "plan": self.plan,
            "_daily_cache": None,
            "_batch_variables": None,
            "_backend": None,
            "_owner_pid": None,
        }

    def _reset_for_process(self) -> None:
        pid = os.getpid()
        if self._owner_pid == pid:
            return
        self._backend = None
        self._daily_cache = None
        self._batch_variables = None
        self._owner_pid = pid

    def worker_init(self) -> None:
        """Start each worker with no inherited HDF5 cache handle."""
        self.close()
        self._daily_cache = None
        self._batch_variables = None
        self._owner_pid = os.getpid()

    def _backend_for_process(self) -> LocalCacheBackend:
        """Reuse bounded worker-local handles independently of the disk cache."""
        self._reset_for_process()
        if self._backend is None:
            self._backend = LocalCacheBackend(
                self.config.cache_dir,
                revision=self.config.revision,
                max_open_files=self.config.max_open_files,
            )
        return self._backend

    def close(self) -> None:
        """Explicitly release this process's source handles."""
        if self._backend is not None:
            self._backend.close()
        self._backend = None

    def _load_tokens(self, tokens: Iterable[str], patch: PatchSpec) -> dict[str, Any]:
        tokens = tuple(tokens)
        return load_planned_multisource_time_series_nc(
            self.plan,
            tokens,
            patch.footprint,
            patch.context,
            config=self.config,
            backend=self._backend_for_process(),
            variables_by_token=self._batch_variables or _required_variables(tokens),
            daily_cache=self._daily_cache,
        )

    @contextmanager
    def batch(self, requests: Iterable[tuple[str, PatchSpec]]):
        """Reuse daily crops within a batch, grouping shared files and fields.

        Only requested crops are retained, and all batch state is discarded on
        success or failure. Context and target windows share their common days.
        """
        requests = tuple(requests)
        if self._daily_cache is not None:
            raise RuntimeError("Nested source-loader batches are not supported.")
        backend = self._backend_for_process()
        self._daily_cache = {}
        try:
            self._batch_variables = _required_variables(token for token, _ in requests)
            pending = {}
            for token, patch in requests:
                variables = self._batch_variables[token]
                for day in _days(patch.context):
                    when, box = _date_string(day), patch.footprint
                    assets = self.plan.get((token, when, box), ())
                    key = _daily_cache_key(assets, when, box, token, variables)
                    pending[key] = (assets, when, box, token, variables)
            # Group assets/variables without changing rendering or sample order.
            for key, (assets, when, box, token, variables) in sorted(
                pending.items(),
                key=lambda item: (
                    tuple(asset.location for asset in item[1][0]),
                    item[1][4] or (),
                ),
            ):
                data = _load_planned_bbox_nc(
                    assets,
                    when,
                    box,
                    token,
                    config=self.config,
                    backend=backend,
                    variables=variables,
                )
                self._daily_cache[key] = None if data is None else data.load()
            yield
        finally:
            self._daily_cache = None
            self._batch_variables = None

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
        self.config = config
        self._catalog: Any | None = None
        self._backend: LocalCacheBackend | None = None
        self._owner_pid: int | None = None
        _LIVE_CORE_LOADERS.add(self)
        _register_fork_guard()

    def __getstate__(self) -> dict[str, Any]:
        """Never pickle an open catalog or HDF5 handle into a spawned worker."""
        return {
            "config": self.config,
            "_catalog": None,
            "_backend": None,
            "_owner_pid": None,
        }

    def _drop_for_fork(self) -> None:
        """Discard native state in the parent before a child can inherit it."""
        self.close()
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
        """Reuse bounded worker-local handles independently of the disk cache."""
        self._reset_for_process()
        if self._backend is None:
            self._backend = LocalCacheBackend(
                self.config.cache_dir,
                revision=self.config.revision,
                max_open_files=self.config.max_open_files,
            )
        return self._backend

    def close(self) -> None:
        """Explicitly release this process's source handles."""
        if self._backend is not None:
            self._backend.close()
        self._backend = None

    def plan(self, requests: Iterable[tuple[str, PatchSpec]]) -> PlannedSourceLoader:
        """Resolve every unique ``(token, day, footprint)`` once in the parent."""
        resolved = plan_multisource_assets(
            self._catalog_for_process(),
            ((token, patch.footprint, patch.context) for token, patch in requests),
        )
        return PlannedSourceLoader(self.config, resolved)

    def _load_tokens(self, tokens: Iterable[str], patch: PatchSpec):
        tokens = tuple(tokens)
        plan = plan_multisource_assets(
            self._catalog_for_process(),
            ((token, patch.footprint, patch.context) for token in tokens),
        )
        return load_planned_multisource_time_series_nc(
            plan,
            tokens,
            patch.footprint,
            patch.context,
            config=self.config,
            backend=self._backend_for_process(),
            variables_by_token=_required_variables(tokens),
        )

    def load(self, token: str, patch: PatchSpec):
        """Lazy projected retrieval for callers that deliberately skip planning."""
        return self._load_tokens((token,), patch)[token]

    def load_pair(
        self, components: tuple[str, str], patch: PatchSpec
    ) -> dict[str, Any] | None:
        """Load paired variables once when they share the same Core asset."""
        values = self._load_tokens(components, patch)
        return None if any(value is None for value in values.values()) else values
