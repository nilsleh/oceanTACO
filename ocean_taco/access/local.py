"""Revision-qualified local cache used by the sole released access mode."""

from __future__ import annotations

import io
import os
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from ..catalog import CORE_DATASET_REVISION


class LocalCacheBackend:
    """Cache immutable NetCDF assets atomically with a worker-local read LRU.

    The catalog revision is part of every cache key.  A process check before
    each open discards inherited handles after a DataLoader worker is spawned,
    so workers never share an HDF5 file handle with their parent or each other.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        revision: str = CORE_DATASET_REVISION,
        max_open_files: int = 16,
    ) -> None:
        self.root = Path(root)
        self.revision = self._component(revision, name="revision")
        if max_open_files <= 0:
            raise ValueError("max_open_files must be positive.")
        self.max_open_files = max_open_files
        self._owner_pid = os.getpid()
        self._handles: OrderedDict[Path, object] = OrderedDict()

    @staticmethod
    def _component(value: str, *, name: str) -> str:
        """Validate one cache-key component rather than accepting path syntax."""
        candidate = str(value)
        if not candidate or candidate in {".", ".."} or Path(candidate).name != candidate:
            raise ValueError(f"Cache {name} must be one concrete path component.")
        return candidate

    def path_for(self, date: str, tile: str, filename: str) -> Path:
        """Resolve a cache path without permitting caller-controlled traversal."""
        return self.root / self.revision / self._component(date, name="date") / self._component(tile, name="tile") / self._component(filename, name="filename")

    def _reset_after_spawn(self) -> None:
        """Forget inherited HDF5 handles when process ownership changes."""
        current_pid = os.getpid()
        if current_pid == self._owner_pid:
            return
        # HDF5's process-global state is not fork-safe.  A child owns copies of
        # the descriptors, so closing them here cannot help the parent and may
        # touch unsafe inherited state.  Discard references and open fresh
        # read-only handles lazily in the worker instead.
        self._handles.clear()
        self._owner_pid = current_pid

    def _open_cached(self, path: Path):
        """Open ``path`` once per worker and evict least-recently-used files."""
        import xarray as xr

        self._reset_after_spawn()
        handle = self._handles.pop(path, None)
        if handle is None:
            handle = xr.open_dataset(path, engine="h5netcdf")
        self._handles[path] = handle
        while len(self._handles) > self.max_open_files:
            _, evicted = self._handles.popitem(last=False)
            evicted.close()
        return handle

    def close(self) -> None:
        """Close this process's cached read handles."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def open_or_fetch(self, date: str, tile: str, filename: str, fetch: Callable[[], bytes]):
        """Open a valid cache hit or atomically commit the fetched bytes first."""
        import xarray as xr

        path = self.path_for(date, tile, filename)
        if path.exists():
            return self._open_cached(path)
        content = fetch()
        if not content:
            raise OSError("Refusing to cache an empty asset response.")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
            # Validate before publish, so truncated/error responses cannot turn
            # into cache hits after an interrupted retrieval.
            with xr.open_dataset(io.BytesIO(content), engine="h5netcdf"):
                pass
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return self._open_cached(path)
