"""OceanTACO Dataset - PyTorch Dataset for ocean surface state data.

Design principles:
- Query-based: samples defined by (bbox, time_range) queries
- Trust TACO: use built-in filter_datetime and filter_bbox, then flatten
- Lazy loading: use dask for windowed reads of large files
- Fast merging: vectorized operations for multi-region results
- DataLoader compatible: no internal parallelism
"""

from __future__ import annotations

import io
import re
from datetime import timedelta
from pathlib import Path
from typing import Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tacoreader
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import Dataset

from .queries import Query

tacoreader.use("pandas")


# =============================================================================
# Constants
# =============================================================================

VAR_NAMES = {
    "glorys_ssh": "zos",
    "glorys_sst": "thetao",
    "glorys_sss": "so",
    "glorys_uo": "uo",
    "glorys_vo": "vo",
    "l4_ssh": "sla",
    "l4_sst": "analysed_sst",
    "l4_sss": "sos",
    "l4_wind": "eastward_wind",
    "l3_sst": "adjusted_sea_surface_temperature",
    "l3_sss_smos": "Sea_Surface_Salinity",
    "l3_ssh": "sla_filtered",
    "l3_swot": "ssha_filtered",
    "l3_sss_smos_asc": "Sea_Surface_Salinity",
    "l3_sss_smos_desc": "Sea_Surface_Salinity",
    "argo": "TEMP",
}

GRIDDED_SOURCES = {
    "glorys",
    "l4_ssh",
    "l4_sst",
    "l4_sss",
    "l4_wind",
    "l3_sst",
    "l3_ssh",
    "l3_swot",
    "l3_sss_smos_asc",
    "l3_sss_smos_desc",
}
POINT_SOURCES = {"argo"}

COL_VSI = "vsi_path"
TACOPAD_PREFIX = "__TACOPAD__"
_VSI_PATTERN = re.compile(r"/vsisubfile/(\d+)_(\d+),(.+)")


# =============================================================================
# Utilities
# =============================================================================


def build_file_index(
    taco_path: str, queries: list[Query], variables: list[str]
) -> list[pd.DataFrame]:
    """Pre-index files at init. Single SQL query, then split by date."""
    dataset = tacoreader.load(taco_path)

    # Find global time range across all queries
    all_dates = set()
    for q in queries:
        all_dates.add(pd.to_datetime(q.time_start).date())
        all_dates.add(pd.to_datetime(q.time_end).date())

    if not all_dates:
        dataset.close()
        return [pd.DataFrame() for _ in queries]

    time_start = min(all_dates)
    # Use exclusive upper bound so "2024-12-31T00:00:00+00:00" < "2025-01-01"
    time_end_excl = max(all_dates) + timedelta(days=1)

    # Build variable filter using tacoreader l2 column prefix
    var_conditions = []
    seen_glorys = False
    for var in variables:
        if var.startswith("glorys_"):
            if not seen_glorys:
                var_conditions.append("\"l2:data_source\" = 'glorys'")
                seen_glorys = True
        else:
            var_conditions.append(f"\"l2:data_source\" = '{var}'")
    var_filter = " OR ".join(var_conditions) if var_conditions else "1=1"

    # tacoreader flat-view SQL: tables are l0/l1/l2, columns prefixed "lN:col"
    sql = f"""
        SELECT
            "l0:stac:time_start"          AS time_start,
            "l2:internal:gdal_vsi"        AS vsi_path,
            "l2:res_deg_lat"              AS res_deg_lat,
            "l2:data_source"              AS data_source
        FROM l2
        WHERE "l2:id" NOT LIKE '{TACOPAD_PREFIX}%'
          AND "l0:stac:time_start" >= '{time_start}'
          AND "l0:stac:time_start" <  '{time_end_excl}'
          AND ({var_filter})
    """
    raw = dataset.sql(sql)
    dataset.close()

    # Normalise to pandas
    all_files: pd.DataFrame = raw.to_pandas() if hasattr(raw, "to_pandas") else raw
    all_files["time_start"] = pd.to_datetime(all_files["time_start"]).dt.date

    # Split by per-query date range (fast pandas filter)
    index = []
    for q in queries:
        q_start = pd.to_datetime(q.time_start).date()
        q_end = pd.to_datetime(q.time_end).date()
        mask = (all_files["time_start"] >= q_start) & (all_files["time_start"] <= q_end)
        index.append(all_files[mask].copy())

    return index


def parse_vsi_path(vsi_path: str) -> tuple[int, int, str] | None:
    """Extract (offset, size, filepath) from /vsisubfile/offset_size,path format."""
    if match := _VSI_PATTERN.match(vsi_path):
        return int(match[1]), int(match[2]), match[3]
    return None


def load_netcdf_var(
    vsi_path: str, var_name: str, bbox: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load single variable from NetCDF, cropped to bbox. Handles local and remote files."""
    parsed = parse_vsi_path(vsi_path)

    if parsed:
        # /vsisubfile/offset_size,path — byte-range subfile
        offset, size, filepath = parsed
        if filepath.startswith("/vsicurl/"):
            url = filepath.replace("/vsicurl/", "")
            headers = {"Range": f"bytes={offset}-{offset + size - 1}"}
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            ds = xr.open_dataset(io.BytesIO(resp.content), engine="h5netcdf")
        else:
            with open(filepath, "rb") as f:
                f.seek(offset)
                data = f.read(size)
            ds = xr.open_dataset(io.BytesIO(data), engine="h5netcdf")
    elif vsi_path.startswith("/vsicurl/"):
        # Direct /vsicurl/ URL — download entire file
        url = vsi_path.replace("/vsicurl/", "")
        resp = requests.get(url)
        resp.raise_for_status()
        ds = xr.open_dataset(io.BytesIO(resp.content), engine="h5netcdf")
    else:
        # Local file path
        ds = xr.open_dataset(vsi_path, engine="h5netcdf")

    if var_name not in ds:
        ds.close()
        return None

    lon_min, lon_max, lat_min, lat_max = bbox
    lons, lats = ds["lon"].values, ds["lat"].values

    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    lat_mask = (lats >= lat_min) & (lats <= lat_max)

    if not lon_mask.any() or not lat_mask.any():
        ds.close()
        return None

    ds = ds.isel(lon=lon_mask, lat=lat_mask)
    data = ds[var_name].values
    lats_out = ds["lat"].values
    lons_out = ds["lon"].values
    ds.close()

    return data, lats_out, lons_out


def _interpolate_to_patch(data: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Bilinear interpolation that preserves the NaN mask.

    NaN regions are excluded from contributing to interpolated values via
    a mask-weight normalization. Pixels where the interpolated mask weight
    falls below 0.01 are set back to NaN.
    """
    has_time = data.ndim == 3
    if not has_time:
        data = data[np.newaxis]  # -> (1, H, W)

    valid_mask = np.isfinite(data).astype(np.float32)
    data_filled = np.where(valid_mask.astype(bool), data, 0.0).astype(np.float32)

    h, w = target_size
    # (T, H, W) -> (T, 1, H, W) for F.interpolate
    t_data = torch.from_numpy(data_filled).unsqueeze(1)
    t_mask = torch.from_numpy(valid_mask).unsqueeze(1)

    t_data_r = F.interpolate(t_data, size=(h, w), mode="bilinear", align_corners=False)
    t_mask_r = F.interpolate(t_mask, size=(h, w), mode="bilinear", align_corners=False)

    data_r = t_data_r.numpy().squeeze(1)  # (T, H, W)
    mask_r = t_mask_r.numpy().squeeze(1)  # (T, H, W)

    # Normalize by mask weight; restore NaN where coverage is negligible
    result = np.where(mask_r > 0.01, data_r / np.maximum(mask_r, 1e-8), np.nan)

    if not has_time:
        result = result.squeeze(0)
    return result.astype(np.float32)


# =============================================================================
# Fast Grid Merging
# =============================================================================


class GridMerger:
    """Accumulates gridded data from multiple sources, computes mean."""

    __slots__ = (
        "bbox",
        "resolution",
        "target_lons",
        "target_lats",
        "shape",
        "_sum",
        "_count",
    )

    def __init__(self, bbox: tuple[float, float, float, float], resolution: float):
        self.bbox = bbox
        self.resolution = resolution

        lon_min, lon_max, lat_min, lat_max = bbox
        self.target_lons = np.arange(
            lon_min, lon_max + resolution / 2, resolution, dtype=np.float32
        )
        self.target_lats = np.arange(
            lat_min, lat_max + resolution / 2, resolution, dtype=np.float32
        )
        self.shape = (len(self.target_lats), len(self.target_lons))

        self._sum = np.zeros(self.shape, dtype=np.float64)
        self._count = np.zeros(self.shape, dtype=np.int32)

    def add(self, data: np.ndarray, src_lons: np.ndarray, src_lats: np.ndarray) -> None:
        if data.size == 0:
            return

        while data.ndim > 2:
            data = data.squeeze()

        if data.shape == self.shape:
            valid = np.isfinite(data)
            self._sum += np.where(valid, data, 0)
            self._count += valid.astype(np.int32)
            return

        # Coordinate mapping for mismatched grids
        lon_idx = np.clip(
            ((src_lons - self.bbox[0]) / self.resolution).astype(int),
            0,
            len(self.target_lons) - 1,
        )
        lat_idx = np.clip(
            ((src_lats - self.bbox[2]) / self.resolution).astype(int),
            0,
            len(self.target_lats) - 1,
        )

        # Vectorized scatter-add (replaces O(H×W) Python loop)
        li, lj = np.meshgrid(
            lat_idx[: data.shape[0]],
            lon_idx[: data.shape[1]],
            indexing="ij",
        )
        valid = np.isfinite(data)
        np.add.at(self._sum, (li[valid], lj[valid]), data[valid])
        np.add.at(self._count, (li[valid], lj[valid]), 1)

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.errstate(invalid="ignore"):
            merged = np.where(self._count > 0, self._sum / self._count, np.nan)
        return merged.astype(np.float32), self.target_lats, self.target_lons


# =============================================================================
# Main Dataset Class
# =============================================================================


class OceanTACODataset(Dataset):
    """Query-based PyTorch Dataset for OceanTACO data.

    Pre-indexes files via SQL at init, making it safe for DataLoader with num_workers > 0.
    """

    def __init__(
        self,
        taco_path: str,
        queries: list[Query],
        input_variables: list[str],
        target_variables: list[str],
        target_resolution: float | None = None,
        temporal_agg: Literal["first", "last", "mean", "stack"] = "mean",
        default_patch_size: tuple[int, int] = (128, 128),
        patch_sizes: dict[str, tuple[int, int]] | None = None,
    ):
        """Args:
        taco_path: Path to TACO dataset file
        queries: List of Query objects defining samples
        input_variables: Variables to load as inputs
        target_variables: Variables to load as targets
        target_resolution: Output grid resolution in degrees (None = use native)
        temporal_agg: How to aggregate multiple timestamps
        default_patch_size: Target (H, W) pixel size for all gridded variables. Default (128, 128).
        patch_sizes: Per-variable overrides, e.g. {"l4_ssh": (64, 64)}. Point sources are never resized.
        """
        super().__init__()

        self.taco_path = taco_path
        self.queries = queries
        self.input_variables = list(input_variables)
        self.target_variables = list(target_variables)
        self.all_variables = list(set(input_variables) | set(target_variables))
        self.target_resolution = target_resolution
        self.temporal_agg = temporal_agg
        self.default_patch_size = default_patch_size
        self.patch_sizes = patch_sizes or {}

        # Validate variables
        invalid = set(self.all_variables) - set(VAR_NAMES.keys())
        if invalid:
            raise ValueError(f"Unknown variables: {invalid}. Valid: {list(VAR_NAMES)}")

        self._file_index = build_file_index(taco_path, queries, self.all_variables)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict:
        query = self.queries[idx]
        file_df = self._file_index[idx]

        if file_df.empty:
            return self._empty_result(query)

        inputs = {
            v: self._load_variable(v, file_df, query.bbox) for v in self.input_variables
        }
        targets = {
            v: self._load_variable(v, file_df, query.bbox)
            for v in self.target_variables
        }

        return {
            "inputs": inputs,
            "targets": targets,
            "coords": self._extract_coords(inputs, targets, query.bbox),
            "metadata": {
                "bbox": query.bbox,
                "time_range": (query.time_start, query.time_end),
                "n_files": len(file_df),
            },
        }

    def _load_variable(
        self, var: str, file_df: pd.DataFrame, bbox: tuple
    ) -> dict | None:
        if var.startswith("glorys_"):
            var_df = file_df[file_df["data_source"] == "glorys"]
        else:
            var_df = file_df[file_df["data_source"] == var]

        if var_df.empty:
            return None

        nc_var = VAR_NAMES[var]
        resolution = float(var_df["res_deg_lat"].iloc[0])

        use_merger = len(var_df) > 1
        merger = GridMerger(bbox, resolution) if use_merger else None
        data_list = []
        lats_out, lons_out = None, None

        for _, row in var_df.iterrows():
            vsi_path = row[COL_VSI]
            result = load_netcdf_var(vsi_path, nc_var, bbox)
            if not result:
                continue
            data, lats, lons = result

            if data.size == 0:
                continue

            if merger:
                merger.add(data, lons, lats)
            else:
                data_list.append(data)
                if lats_out is None:
                    lats_out, lons_out = lats, lons

        if merger:
            data, lats_out, lons_out = merger.result()
        elif data_list:
            data = self._aggregate_temporal(data_list)
        else:
            return None

        if var == "l4_sst":
            data = data - 273.15

        # Resize gridded vars to target patch size
        if var not in POINT_SOURCES and self.default_patch_size is not None:
            target_size = self.patch_sizes.get(var, self.default_patch_size)
            if data.shape[-2:] != target_size:
                data = _interpolate_to_patch(data, target_size)
            # Update coords to match new pixel grid
            lon_min, lon_max, lat_min, lat_max = bbox
            h, w = target_size
            lats_out = np.linspace(lat_min, lat_max, h, dtype=np.float32)
            lons_out = np.linspace(lon_min, lon_max, w, dtype=np.float32)

        # TODO normalization

        # Handle NaN
        data = np.nan_to_num(data, nan=0.0)
        if data.ndim > 2 and data.shape[0] == 1:
            data = data.squeeze(0)

        return {
            "data": torch.from_numpy(data.astype(np.float32)),
            "lats": torch.from_numpy(lats_out.astype(np.float32))
            if lats_out is not None
            else None,
            "lons": torch.from_numpy(lons_out.astype(np.float32))
            if lons_out is not None
            else None,
        }

    def _aggregate_temporal(self, data_list: list[np.ndarray]) -> np.ndarray:
        if len(data_list) == 1:
            return data_list[0]

        shapes = [d.shape for d in data_list]
        if len(set(shapes)) > 1:
            return data_list[0]

        stacked = np.stack(data_list, axis=0)

        if self.temporal_agg == "first":
            return stacked[0]
        elif self.temporal_agg == "last":
            return stacked[-1]
        elif self.temporal_agg == "mean":
            return np.nanmean(stacked, axis=0)
        elif self.temporal_agg == "stack":
            return stacked
        return stacked[0]

    def _extract_coords(self, inputs: dict, targets: dict, bbox: tuple) -> dict:
        for var_data in list(inputs.values()) + list(targets.values()):
            if var_data and var_data.get("lats") is not None:
                return {"lat": var_data["lats"], "lon": var_data["lons"]}
        # Fallback: derive from default_patch_size
        lon_min, lon_max, lat_min, lat_max = bbox
        h, w = self.default_patch_size
        return {
            "lat": torch.linspace(lat_min, lat_max, h),
            "lon": torch.linspace(lon_min, lon_max, w),
        }

    def _empty_result(self, query: Query) -> dict:
        return {
            "inputs": {v: None for v in self.input_variables},
            "targets": {v: None for v in self.target_variables},
            "coords": self._extract_coords({}, {}, query.bbox),
            "metadata": {
                "bbox": query.bbox,
                "time_range": (query.time_start, query.time_end),
                "n_files": 0,
            },
        }

    def visualize_sample(
        self,
        sample: dict,
        figsize: tuple[int, int] | None = None,
        save_path: str | Path | None = None,
        title: str = "",
        max_cols: int = 3,
    ):
        """Visualize all variables in a sample.

        Args:
            sample: Output from __getitem__ or _execute_query
            figsize: Figure size (width, height)
            save_path: Path to save figure (None = display)
            title: Optional title prefix
            max_cols: Maximum columns in subplot grid
        """
        # Collect all variables to plot
        all_vars = {}
        for name, data in sample["inputs"].items():
            if data is not None:
                all_vars[f"[Input] {name}"] = data
        for name, data in sample["targets"].items():
            if data is not None:
                all_vars[f"[Target] {name}"] = data

        if not all_vars:
            print("No data to visualize!")
            return

        n_vars = len(all_vars)
        n_cols = min(max_cols, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols

        if figsize is None:
            figsize = (6 * n_cols, 5 * n_rows)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            subplot_kw={"projection": ccrs.PlateCarree()},
            squeeze=False,
        )
        axes = axes.flatten()

        bbox = sample["metadata"].get("bbox")
        coords = sample["coords"]

        for ax, (var_label, var_data) in zip(axes, all_vars.items()):
            self._plot_variable(ax, var_label, var_data, coords, bbox)

        # Hide unused axes
        for i in range(n_vars, len(axes)):
            axes[i].axis("off")

        # Suptitle with metadata
        metadata = sample["metadata"]
        time_range = metadata.get("time_range", ("?", "?"))
        suptitle = f"{title}\n" if title else ""
        suptitle += f"Time: {time_range[0]} to {time_range[1]}"
        if bbox:
            suptitle += (
                f" | BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"
            )
        suptitle += f" | Files: {metadata.get('n_files', '?')}"

        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        fig.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close(fig)

    def _plot_variable(
        self, ax, var_label: str, var_data: dict, coords: dict, bbox: tuple
    ):
        """Plot a single variable on an axis."""
        data = var_data["data"].detach().cpu().numpy()

        # Use coordinates from var_data if available, else from sample coords
        if var_data.get("lats") is not None:
            lats = var_data["lats"].detach().cpu().numpy()
            lons = var_data["lons"].detach().cpu().numpy()
        else:
            lats = coords["lat"].detach().cpu().numpy()
            lons = coords["lon"].detach().cpu().numpy()

        if data.size == 0:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(var_label)
            return

        # Get colormap params
        cmap_params = _get_colormap_params(var_label)

        # Handle dark background for sparse data (L3 SSH/SWOT)
        var_lower = var_label.lower()
        use_dark_bg = "ssh" in var_lower or "swot" in var_lower or "sla" in var_lower

        if use_dark_bg:
            ax.set_facecolor("black")
            land_color = "#333333"
            grid_color = "gray"
            # Expose background for sparse L3 data (masked with 0.0 in loader)
            if "l3" in var_lower:
                data = data.copy()
                data[data == 0.0] = np.nan
        else:
            land_color = "lightgray"
            grid_color = "black"

        # Handle 3D data (time, lat, lon) - take last timestep
        if data.ndim == 3:
            data = data[-1]
            var_label += " (t=-1)"

        # Dynamically set vmin/vmax from data, fallback to defaults if not set
        finite_data = data[np.isfinite(data)]
        vmin = np.nanmin(finite_data)
        vmax = np.nanmax(finite_data)

        # Plot based on data shape
        if data.ndim == 2 and lats.ndim == 1:
            # Gridded data
            mappable = ax.pcolormesh(
                lons,
                lats,
                data,
                transform=ccrs.PlateCarree(),
                cmap=cmap_params["cmap"],
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
                shading="gouraud" if "l3_swot" in var_lower else "auto",
            )
        elif data.ndim == 1:
            # Point data
            mappable = ax.scatter(
                lons,
                lats,
                c=data,
                transform=ccrs.PlateCarree(),
                cmap=cmap_params["cmap"],
                vmin=vmin,
                vmax=vmax,
                s=10,
                alpha=0.8,
            )
        else:
            ax.text(
                0.5,
                0.5,
                f"Shape: {data.shape}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(var_label)
            return

        plt.colorbar(mappable, ax=ax, label=cmap_params["label"], shrink=0.8)

        # Set extent
        if bbox:
            ax.set_extent([bbox[0], bbox[1], bbox[2], bbox[3]], crs=ccrs.PlateCarree())

        # Add map features
        ax.coastlines(linewidth=0.5, color=grid_color)
        ax.add_feature(cfeature.LAND, facecolor=land_color, edgecolor="none")
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, color=grid_color)

        ax.set_title(f"{var_label}\nshape={data.shape}", fontsize=10)


# =============================================================================
# Visualization Helpers
# =============================================================================


def _get_colormap_params(var_name: str) -> dict:
    """Get visualization parameters for a variable."""
    var_lower = var_name.lower()

    if "ssh" in var_lower or "swot" in var_lower or "sla" in var_lower:
        return {"vmin": -0.6, "vmax": 0.6, "cmap": "RdBu_r", "label": "SSH (m)"}
    elif "sst" in var_lower or "temp" in var_lower:
        return {"vmin": 0, "vmax": 40, "cmap": "RdYlBu_r", "label": "SST (°C)"}
    elif "sss" in var_lower or "sal" in var_lower:
        return {"vmin": 32, "vmax": 38, "cmap": "viridis", "label": "SSS (PSU)"}
    elif "wind" in var_lower:
        return {"vmin": -15, "vmax": 15, "cmap": "coolwarm", "label": "Wind (m/s)"}
    elif "uo" in var_lower or "vo" in var_lower:
        return {"vmin": -2, "vmax": 2, "cmap": "coolwarm", "label": "Current (m/s)"}
    else:
        return {"vmin": None, "vmax": None, "cmap": "viridis", "label": "Value"}


# =============================================================================
# Collate Function
# =============================================================================


def collate_ocean_samples(batch: list[dict]) -> dict:
    """Collate function for DataLoader.

    Handles None values and variable-size tensors by padding.
    """
    if not batch:
        return {}

    def stack_tensors(tensor_list: list[torch.Tensor | None]) -> torch.Tensor | None:
        tensors = [t for t in tensor_list if t is not None]
        if not tensors:
            return None

        # Pad to max shape
        ndim = tensors[0].ndim
        max_shape = [max(t.shape[i] for t in tensors) for i in range(ndim)]

        padded = []
        for t in tensors:
            if list(t.shape) != max_shape:
                pad = []
                for i in range(ndim - 1, -1, -1):
                    pad.extend([0, max_shape[i] - t.shape[i]])
                t = torch.nn.functional.pad(t, pad, value=0.0)
            padded.append(t)

        return torch.stack(padded, dim=0)

    # Collate inputs
    input_vars = list(batch[0]["inputs"].keys())
    inputs = {}
    for var in input_vars:
        tensors = [
            s["inputs"][var]["data"] if s["inputs"][var] else None for s in batch
        ]
        inputs[var] = stack_tensors(tensors)

    # Collate targets
    target_vars = list(batch[0]["targets"].keys())
    targets = {}
    for var in target_vars:
        tensors = [
            s["targets"][var]["data"] if s["targets"][var] else None for s in batch
        ]
        targets[var] = stack_tensors(tensors)

    # Coords from first sample
    coords = batch[0]["coords"]

    # Metadata
    metadata = {
        "bboxes": [s["metadata"]["bbox"] for s in batch],
        "time_ranges": [s["metadata"]["time_range"] for s in batch],
    }

    return {
        "inputs": inputs,
        "targets": targets,
        "coords": coords,
        "metadata": metadata,
    }
