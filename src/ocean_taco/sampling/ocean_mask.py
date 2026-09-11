"""Product-derived binary ocean mask used by the published position grids."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import numpy as np

MASK_ARTIFACT_VERSION = "ghrsst_0p1deg_60S_60N_v1"
RELEASED_MASK_FILENAME = "ocean_mask_0p1deg_60S_60N.npz"


def _canonical_lons(lons: np.ndarray) -> np.ndarray:
    values = np.asarray(lons, dtype=np.float64)
    return np.where(values == 180.0, -180.0, ((values + 180.0) % 360.0) - 180.0)


def _sha256_arrays(
    lats: np.ndarray, lons: np.ndarray, mask: np.ndarray, manifest: Mapping[str, Any]
) -> str:
    digest = sha256()
    for value in (lats, lons, mask.astype(np.uint8)):
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    digest.update(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class OceanMaskArtifact:
    """Frozen, binary 60°S–60°N sampling authority.

    ``mask=True`` means ocean.  It is deliberately independent of a source's
    finite-data mask: ice is ocean; cloud and swath gaps remain source support
    concerns.
    """

    lat: np.ndarray
    lon: np.ndarray
    ocean_mask: np.ndarray
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        lat = np.asarray(self.lat, dtype=np.float64)
        lon = _canonical_lons(self.lon)
        mask = np.asarray(self.ocean_mask, dtype=bool)
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError("Ocean mask coordinates must be one-dimensional.")
        if mask.shape != (lat.size, lon.size):
            raise ValueError("ocean_mask must have shape (lat, lon).")
        if lat.size == 0 or lon.size == 0:
            raise ValueError("Ocean mask cannot be empty.")
        if not np.all(np.diff(lat) > 0) or not np.all(np.diff(lon) > 0):
            raise ValueError(
                "Ocean mask latitude and canonical longitude axes must be strictly ascending."
            )
        if lat[0] < -60.0 or lat[-1] > 60.0:
            raise ValueError(
                "The released ocean-mask artifact must be limited to 60°S–60°N."
            )
        manifest = dict(self.manifest)
        manifest.setdefault("artifact_version", MASK_ARTIFACT_VERSION)
        manifest.setdefault("coordinate_convention", "[-180, 180)")
        identity_manifest = {
            key: value for key, value in manifest.items() if key != "sha256"
        }
        identifier = _sha256_arrays(lat, lon, mask, identity_manifest)
        supplied = manifest.get("sha256")
        if supplied is not None and supplied != identifier:
            raise ValueError("Ocean-mask manifest sha256 does not match mask contents.")
        manifest["sha256"] = identifier
        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "lon", lon)
        object.__setattr__(self, "ocean_mask", mask)
        object.__setattr__(self, "manifest", manifest)

    @property
    def sha256(self) -> str:
        """Content identity of the frozen artifact."""
        return self.manifest["sha256"]

    @property
    def artifact_id(self) -> str:
        """Versioned public artifact identity."""
        return f"{self.manifest['artifact_version']}:{self.sha256}"

    def nearest(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Select categorical mask values by nearest coordinate, never interpolate."""
        selected, _ = self.nearest_with_domain(lat, lon)
        return selected

    def nearest_with_domain(
        self, lat: np.ndarray, lon: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return categorical values and whether each grid cell is classified.

        ``False`` in ``ocean_mask`` means land *only* where
        ``in_mask_domain`` is true.  Keeping the latter separately prevents an
        out-of-domain cell from being silently represented as land.
        """
        lat_values = np.asarray(lat, dtype=np.float64)
        lon_values = _canonical_lons(np.asarray(lon, dtype=np.float64))
        lat_indices = np.abs(self.lat[:, None] - lat_values.reshape(1, -1)).argmin(
            axis=0
        )
        lon_indices = np.abs(self.lon[:, None] - lon_values.reshape(1, -1)).argmin(
            axis=0
        )
        selected = self.ocean_mask[np.ix_(lat_indices, lon_indices)]
        # The artifact is authoritative only over its frozen 60°S–60°N
        # support.  Clamping an out-of-domain grid cell to its edge silently
        # calls an unclassified cell ocean, so make it explicitly non-ocean.
        latitude_in_domain = (lat_values >= self.lat[0]) & (lat_values <= self.lat[-1])
        in_mask_domain = np.broadcast_to(
            latitude_in_domain.reshape(-1, 1), selected.shape
        ).copy()
        return selected & in_mask_domain, in_mask_domain

    def nearest_points(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Select categorical mask values for paired point coordinates."""
        selected, _ = self.nearest_points_with_domain(lat, lon)
        return selected

    def nearest_points_with_domain(
        self, lat: np.ndarray, lon: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return paired categorical values and their artifact-domain status."""
        lat_values, lon_values = np.broadcast_arrays(
            np.asarray(lat, dtype=np.float64),
            _canonical_lons(np.asarray(lon, dtype=np.float64)),
        )
        flat_lat, flat_lon = lat_values.reshape(-1), lon_values.reshape(-1)
        lat_indices = np.abs(self.lat[:, None] - flat_lat.reshape(1, -1)).argmin(axis=0)
        lon_indices = np.abs(self.lon[:, None] - flat_lon.reshape(1, -1)).argmin(axis=0)
        selected = self.ocean_mask[lat_indices, lon_indices]
        in_mask_domain = (flat_lat >= self.lat[0]) & (flat_lat <= self.lat[-1])
        return (selected & in_mask_domain).reshape(
            lat_values.shape
        ), in_mask_domain.reshape(lat_values.shape)

    def write(self, path: Path | str) -> Path:
        """Write a compressed immutable artifact with its manifest."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                lat=self.lat,
                lon=self.lon,
                ocean_mask=self.ocean_mask,
                manifest=json.dumps(self.manifest, sort_keys=True),
            )
        temporary.replace(path)
        return path

    @classmethod
    def read(cls, path: Path | str) -> OceanMaskArtifact:
        """Load and revalidate an artifact written by :meth:`write`."""
        with np.load(Path(path), allow_pickle=False) as archive:
            manifest = json.loads(str(archive["manifest"].item()))
            return cls(archive["lat"], archive["lon"], archive["ocean_mask"], manifest)


def load_released_ocean_mask() -> OceanMaskArtifact:
    """Load the wheel-packaged, pinned Core binary ocean-mask artifact."""
    resource = files("ocean_taco.sampling").joinpath("data", RELEASED_MASK_FILENAME)
    with as_file(resource) as path:
        return OceanMaskArtifact.read(path)


def build_ocean_mask(
    dataset,
    *,
    source_asset_ids: list[str],
    source_revision: str,
    date: str = "2024-06-01",
) -> OceanMaskArtifact:
    """Build the frozen mask from a decoded GHRSST ``l4_sst`` input dataset.

    The product declares bit semantics via ``flag_masks=[1, 2, 4, 8]`` and
    ``flag_meanings='sea land lake ice'``.  Fill values are always classified
    as non-ocean before the integer bit test.
    """
    if (
        "mask" not in dataset
        or "lat" not in dataset.coords
        or "lon" not in dataset.coords
    ):
        raise ValueError("GHRSST input must contain mask, lat, and lon.")
    source_mask = dataset["mask"]
    extra_dims = [
        dimension for dimension in source_mask.dims if dimension not in {"lat", "lon"}
    ]
    for dimension in extra_dims:
        if source_mask.sizes[dimension] != 1:
            raise ValueError(
                f"GHRSST mask has unsupported non-singleton dimension {dimension!r}."
            )
        source_mask = source_mask.isel({dimension: 0}, drop=True)
    source_mask = source_mask.transpose("lat", "lon").sortby("lat")
    values = np.asarray(source_mask.values)
    fill_value = source_mask.encoding.get(
        "_FillValue", source_mask.attrs.get("_FillValue")
    )
    valid = np.isfinite(values)
    if fill_value is not None:
        valid &= values != fill_value
    ocean = np.zeros(values.shape, dtype=bool)
    ocean[valid] = (values[valid].astype(np.int16) & 1) != 0
    lat = np.asarray(source_mask["lat"].values)
    lon = _canonical_lons(np.asarray(source_mask["lon"].values))
    order = np.argsort(lon)
    lon, ocean = lon[order], ocean[:, order]
    latitude_keep = (lat >= -60.0) & (lat <= 60.0)
    semantics = {
        "flag_masks": np.asarray(
            source_mask.attrs.get("flag_masks", []), dtype=int
        ).tolist(),
        "flag_meanings": str(source_mask.attrs.get("flag_meanings", "")),
        "fill_value": None if fill_value is None else float(fill_value),
    }
    if (
        semantics["flag_masks"] != [1, 2, 4, 8]
        or semantics["flag_meanings"] != "sea land lake ice"
    ):
        raise ValueError(
            "GHRSST mask bit semantics differ from the declared frozen-artifact contract."
        )
    manifest = {
        "artifact_version": MASK_ARTIFACT_VERSION,
        "source_revision": source_revision,
        "source_date": date,
        "source_asset_ids": sorted(source_asset_ids),
        "bit_semantics": semantics,
        "coordinate_convention": "[-180, 180)",
    }
    return OceanMaskArtifact(lat[latitude_keep], lon, ocean[latitude_keep], manifest)
