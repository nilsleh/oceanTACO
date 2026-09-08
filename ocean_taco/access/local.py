"""Revision-qualified local cache used by the sole released access mode."""

from __future__ import annotations

import atexit
import io
import os
import tempfile
import weakref
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from ..catalog import CORE_DATASET_REVISION

_LIVE_BACKENDS: weakref.WeakSet = weakref.WeakSet()


def _close_backends() -> None:
    for backend in tuple(_LIVE_BACKENDS):
        backend.close()


# Close in the parent before fork, where HDF5 state is still safe to touch.
# Spawn uses __getstate__; PID checks remain a defensive fallback.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_close_backends)
atexit.register(_close_backends)


class LocalCacheBackend:
    """Cache immutable NetCDF assets atomically with a worker-local read LRU.

    The catalog revision is part of every cache key.  A process check before
    each open discards inherited handles after a DataLoader worker is spawned,
    so workers never share an HDF5 file handle with their parent or each other.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        revision: str = CORE_DATASET_REVISION,
        max_open_files: int = 16,
    ) -> None:
        self.root = None if root is None else Path(root)
        self.revision = self._component(revision, name="revision")
        if isinstance(max_open_files, bool) or not isinstance(max_open_files, int) or max_open_files <= 0:
            raise ValueError("max_open_files must be positive.")
        self.max_open_files = max_open_files
        self._owner_pid = os.getpid()
        self._handles: OrderedDict[Path, object] = OrderedDict()
        self._views: dict[int, object] = {}
        self.file_opens = 0
        self.cache_hits = 0
        _LIVE_BACKENDS.add(self)

    @staticmethod
    def _component(value: str, *, name: str) -> str:
        """Validate one cache-key component rather than accepting path syntax."""
        candidate = str(value)
        if not candidate or candidate in {".", ".."} or Path(candidate).name != candidate:
            raise ValueError(f"Cache {name} must be one concrete path component.")
        return candidate

    def path_for(self, date: str, tile: str, filename: str) -> Path:
        """Resolve a cache path without permitting caller-controlled traversal."""
        if self.root is None:
            raise ValueError("Fetching assets requires a cache root; open_path does not.")
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
        self._views.clear()
        self.file_opens = self.cache_hits = 0
        self._owner_pid = current_pid

    def _open_cached(self, path: Path):
        """Open ``path`` once per worker and evict least-recently-used files."""
        import xarray as xr

        self._reset_after_spawn()
        handle = self._handles.pop(path, None)
        if handle is None:
            while len(self._handles) >= self.max_open_files:
                _, evicted = self._handles.popitem(last=False)
                self._views.pop(id(evicted), None)
                evicted.close()
            handle = xr.open_dataset(path, engine="h5netcdf")
            self.file_opens += 1
        else:
            self.cache_hits += 1
        self._handles[path] = handle
        return handle

    def close(self) -> None:
        """Close this process's cached read handles."""
        self._reset_after_spawn()
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._views.clear()

    def __getstate__(self):
        """Serialize configuration only, never live handles or derived views."""
        return {"root": self.root, "revision": self.revision, "max_open_files": self.max_open_files}

    def __setstate__(self, state):
        """Create an empty worker-owned cache after deserialization."""
        self.__init__(**state)

    def canonical_grid(self, dataset):
        """Cache a lazy coordinate-normalized view for the life of its handle."""
        from ..retrieve import _canonicalise_grid_coordinates

        self._reset_after_spawn()
        key = id(dataset)
        if key not in self._views:
            view = _canonicalise_grid_coordinates(dataset)
            if any(dataset is handle for handle in self._handles.values()):
                self._views[key] = view
            return view
        return self._views[key]

    def open_path(self, path: Path | str):
        """Open an already-materialised asset through this worker's read LRU.

        ``hf_hub_download`` performs its own atomic, revision-qualified caching,
        so a Hub asset is already immutable on disk and must not be copied into
        a second cache.  Only the fork-safe handle reuse is still wanted.
        """
        return self._open_cached(Path(path))

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
