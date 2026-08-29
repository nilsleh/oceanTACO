"""Build released OceanTACO QuerySets from a local TACO dataset.

Three stages, each independently runnable via ``--stage``:

``plan``
    Date-invariant work: global source grids, position grids for every
    (patch size, kind) pair, per-position crop plans and static denominators.
``measure``
    The only expensive stage.  Date-major: each date's regional assets are
    read once and every position of every set is measured against them.  One
    resumable shard per date; parallel over dates within a process
    (``--jobs``) and across SLURM array tasks (``--shard-index``).
``assemble``
    Reads the shards, calls :func:`ocean_taco.sampling.publish.build_queryset`
    for each set with a pure-lookup coverage callback, and publishes with
    :meth:`ocean_taco.manifest.QuerySet.write`.

Dates whose source assets are incomplete are retained with null coverage;
they are never silently dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from queryset_build import (  # noqa: E402
    COVERAGE_KEY,
    DENSE_TOKENS,
    POINT_TOKEN,
    REGIONS,
    ArgoDay,
    GlobalGrid,
    build_argo_day,
    build_global_grid,
    build_position_plan,
    canonical_lon,
    check_units,
    dense_coverage_for,
    grid_signature,
    scatter_tiles,
)

from ocean_taco.catalog import CatalogConfig, load_catalog  # noqa: E402
from ocean_taco.geobox import PatchSize, utc_isoformat  # noqa: E402
from ocean_taco.manifest import QuerySet, content_sha256  # noqa: E402
from ocean_taco.registry import get_modality, registry_sha256  # noqa: E402
from ocean_taco.retrieve import _clean_swot, _url_from_row  # noqa: E402
from ocean_taco.sampling.coverage import unavailable_dense_coverage  # noqa: E402
from ocean_taco.sampling.grids import (  # noqa: E402
    build_position_grid,
    latitude_band_counts,
)
from ocean_taco.sampling.ocean_mask import load_released_ocean_mask  # noqa: E402
from ocean_taco.sampling.publish import GRID_SPACING_RATIO, build_queryset  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_TOKENS: tuple[str, ...] = ("argo", "l3_ssh", "l3_swot")

#: Region ordering used for the ``region_mask`` bitmask on position rows.
REGION_BIT = {region: 1 << index for index, region in enumerate(REGIONS)}


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def _file_digest(path: Path, chunk: int = 1 << 20) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def source_identity(uri: str, strategy: str) -> dict[str, str]:
    """Return the recorded identity for one local source asset.

    Shards are reusable only while the exact source files they measured are
    still current. Keep this in one helper so the record written during
    measurement and the record re-checked before reuse cannot drift apart.
    """
    path = Path(uri)
    if strategy == "sha256":
        return {
            "uri": str(uri),
            "identity_kind": "sha256",
            "identity_value": _file_digest(path),
        }
    if strategy == "stat":
        stat = path.stat()
        return {
            "uri": str(uri),
            "identity_kind": "size_mtime",
            "identity_value": f"{stat.st_size}:{int(stat.st_mtime_ns)}",
        }
    raise ValueError(f"Unknown asset identity strategy {strategy!r}.")


def catalog_digest(taco_path: Path) -> tuple[str, dict[str, str]]:
    """Digest the catalog metadata that defines this local dataset revision."""
    parts: dict[str, str] = {}
    metadata = taco_path / "METADATA"
    files = sorted(metadata.glob("*.parquet")) if metadata.is_dir() else []
    collection = taco_path / "COLLECTION.json"
    if collection.is_file():
        files = [*files, collection]
    if not files:
        raise ValueError(f"No catalog metadata found under {taco_path}.")
    digest = sha256()
    for path in files:
        value = _file_digest(path)
        parts[path.name] = value
        digest.update(path.name.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest(), parts


def dataset_revision(taco_path: Path, catalog_sha: str, override: str | None) -> str:
    """Return a concrete revision identity for a *local* dataset build.

    ``CORE_DATASET_REVISION`` pins the Hugging Face repository; this build
    reads local files, so recording that constant would assert an identity
    nothing here verified.
    """
    if override:
        return override
    collection = taco_path / "COLLECTION.json"
    version = "unversioned"
    if collection.is_file():
        try:
            payload = json.loads(collection.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        for key in ("dataset_version", "version", "taco_version"):
            value = payload.get(key)
            if value:
                version = str(value)
                break
    return f"local:{version}+{catalog_sha[:16]}"


def code_commit(allow_dirty: bool) -> str:
    """Return the repository commit, refusing a dirty tracked tree by default."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if status and not allow_dirty:
        raise SystemExit(
            "Refusing to publish from a dirty working tree; commit changes or pass "
            f"--allow-dirty.\n{status}"
        )
    return f"{commit}-dirty" if status else commit


def environment_lock_hash(lock_path: Path | None) -> tuple[str, dict[str, Any]]:
    """Return a concrete environment identity and the payload behind it."""
    if lock_path is not None:
        return _file_digest(lock_path), {"lock_file": str(lock_path)}
    from importlib.metadata import PackageNotFoundError, version

    packages: dict[str, str] = {}
    for name in (
        "numpy",
        "pandas",
        "xarray",
        "pyarrow",
        "h5netcdf",
        "h5py",
        "tacoreader",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - clean-install diagnostic
            packages[name] = "absent"
    payload = {"packages": packages, "python": sys.version.split()[0]}
    return content_sha256(payload), payload


# --------------------------------------------------------------------------
# catalog access
# --------------------------------------------------------------------------


def catalog_dates(catalog) -> list[str]:
    """Return every catalog date as an ISO ``YYYY-MM-DD`` string."""
    frame = catalog.flatten() if hasattr(catalog, "flatten") else catalog
    column = None
    for candidate in ("l0:stac:time_start", "l1:istac:time_start", "l2:istac:time_start"):
        if candidate in frame.columns:
            column = candidate
            break
    if column is None:
        raise ValueError("Catalog has no recognised time-start column.")
    import pandas as pd

    values = pd.to_datetime(frame[column], utc=True, errors="coerce").dropna()
    return sorted({str(value.date()) for value in values})


def date_assets(catalog, date: str) -> dict[tuple[str, str], str]:
    """Map ``(region, token)`` to a source URI for one date.

    The catalog omits rows for assets that do not exist, so an absent key is
    exactly a missing source asset.
    """
    frame = catalog.filter_datetime(f"{date}/{date}").flatten()
    result: dict[tuple[str, str], str] = {}
    for token in BUILD_TOKENS:
        filename = get_modality(token).filename
        rows = frame[frame["l2:id"].astype(str).str.endswith(filename)]
        for _, row in rows.iterrows():
            region = str(row["l1:id"])
            if region in REGION_BIT:
                result[(region, token)] = _url_from_row(row)
    return result


# --------------------------------------------------------------------------
# stage A: plan
# --------------------------------------------------------------------------


def _open(path: str):
    import xarray as xr

    return xr.open_dataset(path, engine="h5netcdf")


def load_dense_tiles(uris: Mapping[str, str], token: str) -> dict[str, Any]:
    """Open one date's regional assets for a dense token."""
    datasets = {}
    for region, uri in sorted(uris.items()):
        dataset = _open(uri)
        datasets[region] = _clean_swot(dataset) if token == "l3_swot" else dataset
    return datasets


def dense_arrays(
    grid: GlobalGrid, tiles: Mapping[str, Any], token: str, *, want_n_obs: bool
) -> tuple[np.ndarray | None, np.ndarray | None, frozenset[str]]:
    """Scatter one date's tiles onto the global grid.

    Returns the merged values, the optional ``n_obs`` grid, and the set of
    regions that actually supplied the primary variable.

    Structural assumptions are re-validated on every date, not only on the
    reference date used to derive the axes: a silently misaligned tile would
    corrupt every coverage value measured from it.
    """
    spec = get_modality(token)
    values: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for region, dataset in tiles.items():
        if spec.primary_variable not in dataset:
            # A real dataset condition: on a few dates an asset exists but
            # carries only auxiliary fields, with the primary variable absent
            # entirely.  That is an unmeasurable closure exactly like an
            # absent file, so the region is dropped here and the caller's
            # region-completeness rule turns any footprint touching it into a
            # null coverage tuple.  Measuring the remaining regions would
            # report a partial swath as a real observation.
            continue
        if int(dataset.sizes.get("time", 1)) != 1:
            raise ValueError(
                f"{token} {region} has {dataset.sizes.get('time')} time steps; "
                "one catalog date must decode to exactly one time step."
            )
        data = dataset[spec.primary_variable]
        check_units(spec, data.attrs.get("units"))
        lat = np.asarray(dataset["lat"].values, dtype=np.float64)
        lon = canonical_lon(np.asarray(dataset["lon"].values, dtype=np.float64))
        order = np.argsort(lon)
        row, col = grid.slices[region]
        if not np.allclose(
            grid.lat[row], lat, atol=spec.regularity_tolerance, rtol=0.0
        ) or not np.allclose(
            grid.lon[col], lon[order], atol=spec.regularity_tolerance, rtol=0.0
        ):
            raise ValueError(
                f"{token} {region} axes drifted beyond the declared regularity "
                f"tolerance of {spec.regularity_tolerance}."
            )
        if "time" in data.dims:
            data = data.isel(time=0, drop=True)
        values[region] = np.asarray(data.values, dtype=np.float32)[:, order]
        if want_n_obs and "n_obs" in dataset:
            n_obs = dataset["n_obs"]
            if "time" in n_obs.dims:
                n_obs = n_obs.isel(time=0, drop=True)
            counts[region] = np.asarray(n_obs.values, dtype=np.float32)[:, order]
    if not values:
        return None, None, frozenset()
    measured = frozenset(values)
    merged = scatter_tiles(grid, values)
    if not want_n_obs:
        return merged, None, measured
    if set(counts) != set(values):
        raise ValueError(f"{token} requires product-supplied 'n_obs' on every tile.")
    return merged, scatter_tiles(grid, counts, fill=np.nan), measured


def build_reference_grids(catalog, dates: Sequence[str]) -> dict[str, GlobalGrid]:
    """Derive the global axes for each dense token from the first complete date."""
    grids: dict[str, GlobalGrid] = {}
    for token in DENSE_TOKENS:
        for date in dates:
            assets = date_assets(catalog, date)
            uris = {
                region: uri
                for (region, current), uri in assets.items()
                if current == token
            }
            if set(uris) != set(REGIONS):
                continue
            tiles = load_dense_tiles(uris, token)
            try:
                # The axes come from a date where every region carries the
                # primary variable, so a variable-less asset cannot define the
                # published grid.
                if any(
                    get_modality(token).primary_variable not in dataset
                    for dataset in tiles.values()
                ):
                    continue
                grids[token] = build_global_grid(token, tiles)
            finally:
                for dataset in tiles.values():
                    dataset.close()
            break
        else:
            raise ValueError(
                f"No date in the requested range has all {len(REGIONS)} {token} regions; "
                "cannot derive a global grid."
            )
    return grids


def region_mask_value(box) -> int:
    """Bitmask of Core regions a footprint intersects."""
    from queryset_build import footprint_regions

    return sum(REGION_BIT[region] for region in footprint_regions(box))


def build_sets(
    ocean_mask,
    grids: Mapping[str, GlobalGrid],
    patch_sizes: Sequence[int],
    kinds: Sequence[str],
) -> dict[tuple[int, str], dict[str, Any]]:
    """Build every position grid with its per-token crop plans.

    ``build_position_grid`` invokes ``static_counts(lon, lat)`` with no index,
    so the grid is built once bare to learn the centres, the plans are built
    from those centres, and a second identical call consumes them through a
    memoised lookup.  Both calls are deterministic; the row order is asserted
    to match rather than assumed.
    """
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for size in patch_sizes:
        patch_size = PatchSize(float(size), "km")
        for kind in kinds:
            spacing = patch_size.value * GRID_SPACING_RATIO[kind]
            bare = build_position_grid(
                ocean_mask,
                patch_size=patch_size,
                spacing_km=spacing,
                region_mask=region_mask_value,
                static_counts=None,
            )
            plans: dict[str, list[Any]] = {token: [] for token in DENSE_TOKENS}
            counts: dict[tuple[float, float], dict[str, int]] = {}
            for row in bare:
                lon, lat = float(row["centre_lon"]), float(row["centre_lat"])
                entry: dict[str, int] = {}
                for token in DENSE_TOKENS:
                    plan = build_position_plan(
                        grids[token],
                        patch_size=patch_size,
                        centre_lon=lon,
                        centre_lat=lat,
                        ocean_mask=ocean_mask,
                    )
                    plans[token].append(plan)
                    prefix = COVERAGE_KEY[token]
                    entry[f"{prefix}_footprint_cells"] = plan.footprint_cells
                    entry[f"{prefix}_ocean_cells"] = plan.ocean_cells
                counts[(lon, lat)] = entry
            positions = build_position_grid(
                ocean_mask,
                patch_size=patch_size,
                spacing_km=spacing,
                region_mask=region_mask_value,
                static_counts=lambda lon, lat: counts[(float(lon), float(lat))],
            )
            if len(positions) != len(bare) or any(
                left["position_id"] != right["position_id"]
                for left, right in zip(positions, bare, strict=True)
            ):
                raise ValueError(
                    "build_position_grid is not deterministic across calls; "
                    "static_counts alignment cannot be trusted."
                )
            result[(size, kind)] = {
                "patch_size": patch_size,
                "spacing_km": spacing,
                "positions": positions,
                "plans": plans,
            }
    return result


# --------------------------------------------------------------------------
# stage B: measure
# --------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _shard_path(work_dir: Path, date: str) -> Path:
    return work_dir / "shards" / f"{date}.npz"


def measure_date(
    date: str,
    sets: Mapping[tuple[int, str], dict[str, Any]],
    grids: Mapping[str, GlobalGrid],
    assets: Mapping[tuple[str, str], str],
    *,
    asset_identity: str,
) -> dict[str, Any]:
    """Measure every position of every set against one date's sources."""
    present: dict[str, frozenset[str]] = {}
    globals_: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}
    identities: dict[tuple[str, str], dict[str, str]] = {}

    for token in DENSE_TOKENS:
        uris = {
            region: uri for (region, current), uri in assets.items() if current == token
        }
        if not uris:
            present[token] = frozenset()
            globals_[token] = (None, None)
            continue
        tiles = load_dense_tiles(uris, token)
        try:
            values, n_obs, measured = dense_arrays(
                grids[token], tiles, token, want_n_obs=token == "l3_swot"
            )
        finally:
            for dataset in tiles.values():
                dataset.close()
        # A region whose asset lacks the primary variable is not "present" for
        # coverage purposes, even though its file exists and is ledgered.
        present[token] = measured
        globals_[token] = (values, n_obs)

    argo_uris = {
        region: uri
        for (region, current), uri in assets.items()
        if current == POINT_TOKEN
    }
    present[POINT_TOKEN] = frozenset(argo_uris)
    argo_day: ArgoDay | None = None
    if argo_uris:
        argo_tiles = {region: _open(uri) for region, uri in sorted(argo_uris.items())}
        try:
            argo_day = build_argo_day(argo_tiles, date)
        finally:
            for dataset in argo_tiles.values():
                dataset.close()

    for key, uri in assets.items():
        identities[key] = source_identity(uri, asset_identity)

    payload: dict[str, Any] = {}
    for key, entry in sorted(sets.items()):
        size, kind = key
        patch_size = entry["patch_size"]
        n = len(entry["positions"])
        columns = {
            name: np.full(n, -1, dtype=np.int64)
            for name in (
                "swot_valid_cells",
                "swot_valid_ocean_cells",
                "swot_n_obs_sum",
                "ssh_valid_cells",
                "ssh_valid_ocean_cells",
                "argo_profile_count",
            )
        }
        for index, position in enumerate(entry["positions"]):
            for token in DENSE_TOKENS:
                values, n_obs = globals_[token]
                plan = entry["plans"][token][index]
                coverage = dense_coverage_for(plan, present[token], values, n_obs)
                prefix = COVERAGE_KEY[token]
                columns[f"{prefix}_valid_cells"][index] = (
                    -1 if coverage.valid_cells is None else coverage.valid_cells
                )
                columns[f"{prefix}_valid_ocean_cells"][index] = (
                    -1
                    if coverage.valid_ocean_cells is None
                    else coverage.valid_ocean_cells
                )
                if prefix == "swot":
                    columns["swot_n_obs_sum"][index] = (
                        -1 if coverage.n_obs_sum is None else coverage.n_obs_sum
                    )
            if argo_day is not None and set(REGIONS) <= present[POINT_TOKEN]:
                columns["argo_profile_count"][index] = argo_day.count(
                    patch_size,
                    float(position["centre_lon"]),
                    float(position["centre_lat"]),
                )
        for name, values in columns.items():
            payload[f"{size}-{kind}/{name}"] = values
    return {"columns": payload, "identities": identities, "present": present}


def write_shard(
    work_dir: Path, date: str, plan_id: str, measured: Mapping[str, Any]
) -> Path:
    """Write one date's measurements atomically with a verifiable sidecar."""
    path = _shard_path(work_dir, date)
    path.parent.mkdir(parents=True, exist_ok=True)
    identities = {
        f"{region}|{token}": value
        for (region, token), value in sorted(measured["identities"].items())
    }
    meta = {
        "date": date,
        "plan_id": plan_id,
        "identities": identities,
        "present": {
            token: sorted(regions) for token, regions in measured["present"].items()
        },
    }
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, meta=json.dumps(meta, sort_keys=True), **measured["columns"]
        )
    os.replace(temporary, path)
    sidecar = path.with_suffix(".done.json")
    sidecar.write_text(
        json.dumps(
            {"plan_id": plan_id, "sha256": _file_digest(path)},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def shard_is_valid(
    work_dir: Path,
    date: str,
    plan_id: str,
    *,
    assets: Mapping[tuple[str, str], str] | None = None,
    asset_identity: str | None = None,
) -> bool:
    """Whether a completed shard still matches its plan and source assets.

    Callers deciding whether to *reuse* a shard pass the current date's asset
    records. This makes a corrected or re-downloaded source invalidate its old
    measurement instead of merely re-attesting stale metadata at assemble
    time. Callers which only verify an already selected shard may omit them.
    """
    if (assets is None) != (asset_identity is None):
        raise ValueError("assets and asset_identity must be supplied together.")
    path = _shard_path(work_dir, date)
    sidecar = path.with_suffix(".done.json")
    if not path.is_file() or not sidecar.is_file():
        return False
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if record.get("plan_id") != plan_id or record.get("sha256") != _file_digest(path):
        return False
    if assets is None:
        return True
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"].item()))
        expected = {
            f"{region}|{token}": source_identity(uri, asset_identity)
            for (region, token), uri in sorted(assets.items())
        }
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    return meta.get("identities") == expected


def _worker_init(config: dict[str, Any]) -> None:
    """Build the date-invariant plan once per worker process."""
    mask = load_released_ocean_mask()
    catalog = load_catalog(CatalogConfig(taco_path=Path(config["taco_path"])))
    grids = build_reference_grids(catalog, config["reference_dates"])
    _WORKER["mask"] = mask
    _WORKER["catalog"] = catalog
    _WORKER["grids"] = grids
    _WORKER["sets"] = build_sets(
        mask, grids, config["patch_sizes"], config["kinds"]
    )
    _WORKER["config"] = config


def _worker_measure(date: str) -> tuple[str, str, float]:
    config = _WORKER["config"]
    work_dir = Path(config["work_dir"])
    plan_id = config["plan_id"]
    assets = date_assets(_WORKER["catalog"], date)
    if shard_is_valid(
        work_dir,
        date,
        plan_id,
        assets=assets,
        asset_identity=config["asset_identity"],
    ):
        return date, "skipped", 0.0
    started = time.time()
    measured = measure_date(
        date,
        _WORKER["sets"],
        _WORKER["grids"],
        assets,
        asset_identity=config["asset_identity"],
    )
    write_shard(work_dir, date, plan_id, measured)
    return date, "measured", time.time() - started


# --------------------------------------------------------------------------
# stage C: assemble
# --------------------------------------------------------------------------


def load_coverage_cache(
    work_dir: Path, dates: Sequence[str], key: str, n_positions: int, plan_id: str
) -> dict[str, np.ndarray]:
    """Load one set's coverage columns for every date as flat arrays.

    Indexed ``position_index * n_dates + date_index`` so the assemble-stage
    callback is an O(1) lookup with no per-pair dictionary overhead.
    """
    names = (
        "swot_valid_cells",
        "swot_valid_ocean_cells",
        "swot_n_obs_sum",
        "ssh_valid_cells",
        "ssh_valid_ocean_cells",
        "argo_profile_count",
    )
    total = n_positions * len(dates)
    cache = {name: np.full(total, -1, dtype=np.int64) for name in names}
    for date_index, date in enumerate(dates):
        if not shard_is_valid(work_dir, date, plan_id):
            raise SystemExit(
                f"Missing or stale shard for {date}; run --stage measure first."
            )
        with np.load(_shard_path(work_dir, date), allow_pickle=False) as archive:
            for name in names:
                column = archive[f"{key}/{name}"]
                if column.size != n_positions:
                    raise ValueError(
                        f"Shard {date} has {column.size} rows for {key}/{name}, "
                        f"expected {n_positions}."
                    )
                cache[name][date_index :: len(dates)] = column
    return cache


def assets_rows(
    work_dir: Path, dates: Sequence[str], plan_id: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build the complete asset ledger: one row per date, region, and token."""
    rows: list[dict[str, Any]] = []
    tally = {"present": 0, "missing": 0}
    for date_index, date in enumerate(dates):
        with np.load(_shard_path(work_dir, date), allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"].item()))
        identities = meta["identities"]
        for region in REGIONS:
            for token in BUILD_TOKENS:
                record = identities.get(f"{region}|{token}")
                if record is None:
                    rows.append(
                        {
                            "date_index": date_index,
                            "region": region,
                            "token": token,
                            "asset_id": "",
                            "uri": "",
                            "identity_kind": "",
                            "identity_value": "",
                            "status": "missing",
                        }
                    )
                    tally["missing"] += 1
                    continue
                filename = get_modality(token).filename
                rows.append(
                    {
                        "date_index": date_index,
                        "region": region,
                        "token": token,
                        "asset_id": f"{date}/{region}/{filename}",
                        "uri": record.get("uri", ""),
                        "identity_kind": record["identity_kind"],
                        "identity_value": record["identity_value"],
                        "status": "present",
                    }
                )
                tally["present"] += 1
    return rows, tally


def _nullable(value: int) -> int | None:
    return None if value < 0 else int(value)


def assemble_set(
    key: tuple[int, str],
    entry: Mapping[str, Any],
    *,
    work_dir: Path,
    dates: Sequence[str],
    plan_id: str,
    ocean_mask,
    provenance: Mapping[str, Any],
    assets: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> QuerySet:
    """Build and publish one QuerySet."""
    size, kind = key
    name = f"{size}-{kind}"
    positions = entry["positions"]
    cache = load_coverage_cache(work_dir, dates, name, len(positions), plan_id)
    n_dates = len(dates)

    def measure_coverage(position, date_index, _date):
        from ocean_taco.sampling.coverage import DenseCoverage

        offset = int(position["position_index"]) * n_dates + date_index
        swot = DenseCoverage(
            _nullable(cache["swot_valid_cells"][offset]),
            _nullable(cache["swot_valid_ocean_cells"][offset]),
            _nullable(cache["swot_n_obs_sum"][offset]),
        )
        ssh = DenseCoverage(
            _nullable(cache["ssh_valid_cells"][offset]),
            _nullable(cache["ssh_valid_ocean_cells"][offset]),
            None,
        )
        return {
            "swot": swot,
            "ssh": ssh,
            "argo": _nullable(cache["argo_profile_count"][offset]),
        }

    static = {
        (float(row["centre_lon"]), float(row["centre_lat"])): {
            "swot_footprint_cells": int(row["swot_footprint_cells"]),
            "swot_ocean_cells": int(row["swot_ocean_cells"]),
            "ssh_footprint_cells": int(row["ssh_footprint_cells"]),
            "ssh_ocean_cells": int(row["ssh_ocean_cells"]),
        }
        for row in positions
    }
    grid_validation = {
        "latitude_band_counts": latitude_band_counts(positions),
        "position_count": len(positions),
    }
    queryset = build_queryset(
        ocean_mask=ocean_mask,
        patch_size=entry["patch_size"],
        kind=kind,
        dates=list(dates),
        tokens=list(BUILD_TOKENS),
        provenance={**provenance, "grid_validation": grid_validation},
        assets=assets,
        measure_coverage=measure_coverage,
        static_counts=lambda lon, lat: static[(float(lon), float(lat))],
        region_mask=region_mask_value,
    )
    queryset.write(output_root / name)
    return queryset


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taco-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument(
        "--stage",
        choices=("plan", "measure", "assemble", "all"),
        default="all",
    )
    parser.add_argument("--patch-size", type=int, action="append", dest="patch_sizes")
    parser.add_argument("--kind", action="append", dest="kinds")
    parser.add_argument("--date-stride", type=int, default=1)
    parser.add_argument("--max-dates", type=int, default=None)
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(32, max(1, (os.cpu_count() or 2) // 2)),
        help="worker processes over dates; memory-bound near 24-32 on a 94 GB node",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--asset-identity", choices=("sha256", "stat"), default="sha256"
    )
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--environment-lock", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.patch_sizes = sorted(args.patch_sizes or [256])
    args.kinds = list(args.kinds or ["training", "eval"])
    for kind in args.kinds:
        if kind not in GRID_SPACING_RATIO:
            parser.error(f"unknown kind {kind!r}")
    if args.work_dir is None:
        args.work_dir = args.output_root / "_work"
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must lie in [0, --shard-count).")
    return args


def compute_plan_id(
    args: argparse.Namespace,
    grids: Mapping[str, GlobalGrid],
    dates: Sequence[str],
    ocean_mask,
    sets: Mapping[tuple[int, str], Mapping[str, Any]],
    code_revision: str,
) -> str:
    """Identity of everything a shard's contents depend on.

    The position identities are part of this, not merely the inputs that
    ought to determine them.  A shard stores one row per position, so any
    change to how centres are placed -- a grid construction fix included --
    must invalidate existing shards even though every declared input above
    is unchanged.
    """
    return content_sha256(
        {
            "ocean_mask_id": ocean_mask.artifact_id,
            "patch_sizes": args.patch_sizes,
            "kinds": sorted(args.kinds),
            "tokens": list(BUILD_TOKENS),
            "dates": list(dates),
            "asset_identity": args.asset_identity,
            # A shard embodies builder logic as well as its declared inputs.
            # Without this, a resumed dirty build can mix old and new logic.
            "code_commit": code_revision,
            "grids": [grid_signature(grids[token]) for token in DENSE_TOKENS],
            "positions": {
                f"{size}-{kind}": [
                    row["position_id"] for row in sets[(size, kind)]["positions"]
                ]
                for size, kind in sorted(sets)
            },
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(CatalogConfig(taco_path=args.taco_path))
    dates = catalog_dates(catalog)[:: args.date_stride]
    if args.max_dates:
        dates = dates[: args.max_dates]
    if not dates:
        raise SystemExit("No dates selected.")
    if len(dates) > 32_767:
        raise SystemExit(
            f"{len(dates)} dates exceeds the int16 date_index column width."
        )
    print(f"[plan] {len(dates)} dates: {dates[0]} .. {dates[-1]}", flush=True)

    names = [f"{size}-{kind}" for size in args.patch_sizes for kind in args.kinds]
    if args.stage in {"assemble", "all"}:
        existing = [name for name in names if (args.output_root / name).exists()]
        if existing:
            raise SystemExit(
                "Refusing to overwrite published QuerySet directories: "
                f"{existing}. Remove them or choose another --output-root."
            )

    ocean_mask = load_released_ocean_mask()
    grids = build_reference_grids(catalog, dates)
    for token in DENSE_TOKENS:
        print(f"[plan] {token} global grid {grids[token].shape}", flush=True)

    catalog_sha, catalog_parts = catalog_digest(args.taco_path)
    revision = dataset_revision(args.taco_path, catalog_sha, args.dataset_revision)
    commit = code_commit(args.allow_dirty)
    lock_hash, lock_payload = environment_lock_hash(args.environment_lock)
    sets = build_sets(ocean_mask, grids, args.patch_sizes, args.kinds)
    plan_id = compute_plan_id(args, grids, dates, ocean_mask, sets, commit)
    for key, entry in sorted(sets.items()):
        print(
            f"[plan] {key[0]}-{key[1]}: {len(entry['positions'])} positions "
            f"(spacing {entry['spacing_km']:.2f} km)",
            flush=True,
        )
    print(f"[plan] plan_id {plan_id[:16]} in {time.time() - started:.1f}s", flush=True)

    if args.dry_run or args.stage == "plan":
        return 0

    if args.stage in {"measure", "all"}:
        assigned = [
            date
            for index, date in enumerate(dates)
            if index % args.shard_count == args.shard_index
        ]
        todo = [
            date
            for date in assigned
            if not shard_is_valid(
                args.work_dir,
                date,
                plan_id,
                assets=date_assets(catalog, date),
                asset_identity=args.asset_identity,
            )
        ]
        print(
            f"[measure] {len(todo)}/{len(assigned)} dates to measure "
            f"(shard {args.shard_index}/{args.shard_count}, jobs {args.jobs})",
            flush=True,
        )
        config = {
            "taco_path": str(args.taco_path),
            "work_dir": str(args.work_dir),
            "plan_id": plan_id,
            "patch_sizes": args.patch_sizes,
            "kinds": args.kinds,
            "asset_identity": args.asset_identity,
            # A shard embodies builder logic as well as its declared inputs.
            # Without this, a resumed dirty build can mix old and new logic.
            "code_commit": commit,
            "reference_dates": dates,
        }
        if todo and args.jobs > 1:
            with ProcessPoolExecutor(
                max_workers=min(args.jobs, len(todo)),
                initializer=_worker_init,
                initargs=(config,),
            ) as pool:
                futures = {pool.submit(_worker_measure, date): date for date in todo}
                for done, future in enumerate(as_completed(futures), start=1):
                    date, status, seconds = future.result()
                    print(
                        f"[measure] {done}/{len(todo)} {date} {status} {seconds:.1f}s",
                        flush=True,
                    )
        elif todo:
            _WORKER.update(
                {
                    "mask": ocean_mask,
                    "catalog": catalog,
                    "grids": grids,
                    "sets": sets,
                    "config": config,
                }
            )
            for done, date in enumerate(todo, start=1):
                date, status, seconds = _worker_measure(date)
                print(
                    f"[measure] {done}/{len(todo)} {date} {status} {seconds:.1f}s",
                    flush=True,
                )

    if args.stage == "measure":
        return 0

    missing = [
        date
        for date in dates
        if not shard_is_valid(
            args.work_dir,
            date,
            plan_id,
            assets=date_assets(catalog, date),
            asset_identity=args.asset_identity,
        )
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} shards missing (first: {missing[0]}); "
            "run --stage measure to completion before assembling."
        )

    assets, tally = assets_rows(args.work_dir, dates, plan_id)
    source_records = content_sha256(
        [
            {
                "date": dates[row["date_index"]],
                "region": row["region"],
                "token": row["token"],
                "status": row["status"],
                "identity_value": row["identity_value"],
            }
            for row in sorted(
                assets, key=lambda row: (row["date_index"], row["region"], row["token"])
            )
        ]
    )
    provenance = {
        "dataset_revision": revision,
        "catalog_sha256": catalog_sha,
        "registry_sha256": registry_sha256(),
        "source_records_sha256": source_records,
        "code_commit": commit,
        "environment_lock_hash": lock_hash,
    }
    for field, value in provenance.items():
        if value in (None, "", "unknown"):
            raise SystemExit(f"Provenance field {field} is not concrete.")
    print(
        f"[assemble] assets present={tally['present']} missing={tally['missing']}",
        flush=True,
    )

    published: dict[str, str] = {}
    for key, entry in sorted(sets.items()):
        name = f"{key[0]}-{key[1]}"
        queryset = assemble_set(
            key,
            entry,
            work_dir=args.work_dir,
            dates=dates,
            plan_id=plan_id,
            ocean_mask=ocean_mask,
            provenance=provenance,
            assets=assets,
            output_root=args.output_root,
        )
        published[name] = queryset.queryset_id
        print(
            f"[assemble] {name}: {len(queryset.positions)} positions x "
            f"{len(queryset.dates)} dates = {len(queryset.coverage)} coverage rows "
            f"({queryset.queryset_id[:16]})",
            flush=True,
        )

    manifest = {
        "built_at": utc_isoformat(datetime.now(timezone.utc)),
        "taco_path": str(args.taco_path),
        "dates": dates,
        "patch_sizes": args.patch_sizes,
        "kinds": args.kinds,
        "tokens": list(BUILD_TOKENS),
        "plan_id": plan_id,
        "provenance": provenance,
        "catalog_files": catalog_parts,
        "environment": lock_payload,
        "assets": tally,
        "published": published,
        "units_assumption": (
            "Source primary variables carry no 'units' attribute; the registry "
            "canonical unit is assumed. Coverage counts finite cells and is "
            "unit-independent. A present but unrecognised unit still fails."
        ),
    }
    (args.output_root / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[done] {time.time() - started:.1f}s -> {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
