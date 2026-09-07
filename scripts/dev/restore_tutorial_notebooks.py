#!/usr/bin/env python3
"""Regenerate the maintained tutorial notebooks from reviewed cell definitions.

This is intentionally a source generator: notebooks remain the rendered teaching
documents committed under docs/tutorials, while this file makes their narrative
and current-Core API migration reviewable as normal Python.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "tutorials"

def md(text): return nbf.v4.new_markdown_cell(dedent(text).strip())
def code(text): return nbf.v4.new_code_cell(dedent(text).strip())

SETUP = """
import os
import warnings
from io import BytesIO
from pathlib import Path

from IPython.display import Image, display

# Hub transfers report progress on stderr, which would otherwise be captured as
# notebook output and rendered as noise in the documentation.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Warnings raised by the library are part of what these tutorials demonstrate,
# but Python's default format prefixes each one with the absolute path of the
# file that raised it, which is an artifact of the machine that built the docs.
def _format_warning(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\\n"

warnings.formatwarning = _format_warning

def display_figure(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=144, bbox_inches="tight")
    display(Image(data=buffer.getvalue()))

PATCH_SIZE_KM = 256          # the published patch size
REQUESTED_ROWS = 4
SEED = 7

# Draw records are notebook output, not cached data; they land beside the notebook.
DRAW_DIR = Path("draws")
DRAW_DIR.mkdir(exist_ok=True)
"""

LOAD = """
from ocean_taco import CatalogConfig, QuerySet
from ocean_taco.retrieve import load_hf_dataset

config = CatalogConfig()
catalog = load_hf_dataset(config)
queryset = QuerySet.from_hub(PATCH_SIZE_KM, "eval")
print(f"queryset_id={queryset.queryset_id}; kind={queryset.header['kind']}; positions={len(queryset.positions)}; dates={len(queryset.dates)}")
"""

# Shared by the four rewritten tutorials only. The two Hurricane Milton
# notebooks reproduce paper figures and keep SETUP unchanged, so their
# generated source stays byte-identical to the reviewed version.
PLOTTING = """
import matplotlib.pyplot as plt
import numpy as np

# One diverging colormap across these tutorials, with an explicit range per
# source. RdBu_r centres its white on the midpoint of whatever range it is
# given, so leaving the range to autoscaling would move that centre between
# panels and make two figures of the same field look unalike.
CMAP = "RdBu_r"

# Ranges measured over drawn North Atlantic rows rather than taken from a
# product specification, so they bracket what these tutorials actually plot.
# A range far wider than one patch's own spread would flatten that patch to a
# single shade, so these are kept as tight as cross-panel comparison allows.
COLOR_RANGE = {
    "l4_sst": (20.0, 28.0),        # degrees celsius
    "l4_ssh": (-0.5, 0.5),         # metres
    "l3_swot": (-0.5, 0.5),        # metres
    "l3_ssh": (-0.5, 0.5),         # metres, nadir altimetry, same quantity as l3_swot
    "l4_sss": (32.0, 38.0),        # practical salinity units
    "velocity": (-0.8, 0.8),       # m/s, signed component, symmetric about zero
    "speed": (0.0, 0.8),           # m/s, a magnitude, so it starts at zero
    "argo_temp": (0.0, 30.0),      # degrees celsius
}

def color_limits(key):
    # Fall back to autoscaling for anything without a declared range.
    low, high = COLOR_RANGE.get(key, (None, None))
    return {"vmin": low, "vmax": high}

# Coastlines come from the Natural Earth shapefiles that ship with cartopy and
# are read from its local cache, so drawing them needs no network access while
# the notebooks execute. Only the geometries are used, not cartopy's projection
# machinery: these axes are already plain degrees, where PlateCarree is the
# identity, so the segments can go straight onto them.
import cartopy.io.shapereader as shpreader
from functools import lru_cache
from matplotlib.collections import LineCollection

@lru_cache(maxsize=None)
def _coastline_segments(resolution):
    path = shpreader.natural_earth(resolution=resolution, category="physical", name="coastline")
    segments = []
    for geometry in shpreader.Reader(path).geometries():
        parts = geometry.geoms if geometry.geom_type == "MultiLineString" else [geometry]
        segments.extend(np.asarray(part.coords) for part in parts)
    return tuple(segments)

def add_coastlines(axis, resolution=None, color="#444444", linewidth=.7, alpha=.85):
    # Draw continent contours on a plain longitude/latitude axis. The axis
    # limits must already be set, because they select both the detail level and
    # which segments are worth drawing.
    lon_min, lon_max = sorted(axis.get_xlim())
    lat_min, lat_max = sorted(axis.get_ylim())
    if resolution is None:
        # Coarser outlines suit a whole hemisphere and would look like a
        # polygon at patch scale, so the detail level follows the extent.
        span = max(lon_max - lon_min, lat_max - lat_min)
        resolution = "110m" if span > 60 else "50m" if span > 8 else "10m"
    visible = []
    for coords in _coastline_segments(resolution):
        lon, lat = coords[:, 0], coords[:, 1]
        if lat.max() < lat_min or lat.min() > lat_max:
            continue
        # Panels that run past +/-180 degrees see each landmass twice, so the
        # shifted copies are offered as well and the extent test drops them.
        for shift in (0.0, -360.0, 360.0):
            shifted = lon + shift
            if shifted.max() >= lon_min and shifted.min() <= lon_max:
                visible.append(np.column_stack([shifted, lat]))
    if visible:
        axis.add_collection(LineCollection(visible, colors=color, linewidths=linewidth,
                                           alpha=alpha, zorder=2.5))
    # A LineCollection would otherwise widen the limits to fit whole continents.
    axis.set_xlim(lon_min, lon_max)
    axis.set_ylim(lat_min, lat_max)
    return axis

def geographic_extent(record):
    # The (left, right, bottom, top) extent of a rendered record, in degrees.
    # Every gridded record carries the lon and lat vectors of the grid it was
    # rendered onto, so a panel can be drawn in degrees rather than in pixel
    # index and carry a coastline.
    lon = np.asarray(record["lon"]).ravel()
    lat = np.asarray(record["lat"]).ravel()
    return (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
"""


def write(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}})
    nbf.write(notebook, OUT / name)

write("ml_dataset.ipynb", [
    md("""# From a published QuerySet to a rendered sample

Satellite and in-situ platforms observe the ocean with different sampling
geometries. Over one 256 km patch, a gridded L4 analysis provides a complete
field at a moderate resolution, a nadir altimeter measures along a single
ground track crossing the patch, and SWOT measures a wide swath that may not
intersect the patch at all on a given date. Rendering those into one tensor
therefore requires deciding which sources must be present, what grid they
share, and how cells with no measurement behind them are marked. Those
decisions affect what a model can learn and what a metric means, so OceanTACO
requires each to be stated rather than applying a default, and records the
settings that were used.

The library splits the work into four stages, which this notebook covers in
order:

1. A **QuerySet** is a released table of patch positions and dates. It is
   published, versioned, and checksummed, so two people who name the same
   QuerySet are working from the same candidate samples.
2. A **filter** narrows that set by geography, time, or recorded coverage,
   without fetching anything. `QueryFilter(box=...)` states a rule and commits
   to no rows yet.
3. A **draw** selects specific rows and writes a record of that selection, so
   the same rows can be recovered later.
4. A **dataset** renders those rows into tensors, using one renderer per source
   to fix the grid, the masks, and the treatment of absent data.

Each stage narrows the one before it and writes a record of what it selected.
Given a rendered tensor you can therefore recover the draw record it came from,
the filter that produced its candidate rows, and the released QuerySet behind
those. The final section reproduces all of it from six stored values.

**Where to go next.** The three companion tutorials each go further on one
part:

- [QuerySet and filter deep-dive](data_retrieval_workflows.ipynb) covers
  filters, coverage evidence, and the lower-level retrieval API.
- [ML use cases and the training loader](spatio_temporal_query_generation.ipynb)
  builds the working `DataLoader`, including forecasting, super-resolution, and
  a masked training step.
- [ML renderer configuration reference](ml_configuration_cookbook.ipynb) is the
  per-renderer reference for `Resample`, `Native`, `VectorPair`, and `Points`.

**Prerequisites:** familiarity with the PyTorch `Dataset` and `DataLoader`
interfaces. No oceanography background is assumed, and the domain facts that
matter are stated where they are used."""),
    md("""## 1. Setup

The three constants below are the only inputs this notebook takes.
`CatalogConfig()` carries a pinned catalog revision, and `QuerySet.from_hub`
fetches a published QuerySet by patch size and kind. Loading the QuerySet
verifies every table against the checksums in its header, so a successful load
doubles as the integrity check."""),
    code(SETUP), code(PLOTTING), code(LOAD),
    md("""## 2. Geography comes before pixels

Fixed channel shapes make batches stack and keep architectures simple, while
resampling a sparse swath onto a coarse grid discards the fine-scale structure
that the swath was flown to measure. Since the better trade differs by source
and by application, OceanTACO requires the choice to be stated per source
rather than applying a default. The chain from a geographic patch to a pixel
array runs in four steps.

### Step 1: the patch is geographic, not pixel-shaped

A patch is specified in kilometres, and `PatchSize.to_degrees` converts that to
a longitude and latitude span at a given centre latitude. The two spans differ.
A degree of latitude is about 111 km everywhere, while a degree of longitude
shrinks by `cos(latitude)` away from the equator, so a 256 km patch spans
roughly 2.3° of longitude at the equator and about 4.6° at 60°N. The figure
below plots that relationship: one patch size in kilometres becomes a different
geographic window depending on where it sits.

### Step 2: native resolution sets the raw pixel count

Each source has a fixed native resolution baked into its files, so the same
geographic window yields a different raw array per source:

| Token | Product | Native grid step | Geometry |
|-------|---------|------------------|----------|
| `l4_sst` | L4 gridded analysis | 0.100° | dense, complete |
| `l4_ssh` | L4 DUACS | 0.125° | dense, complete |
| `l4_sss` | L4 gridded salinity | 0.125° | dense, complete |
| `glorys_*` | GLORYS reanalysis | 0.083° (1/12°) | dense, complete |
| `l3_ssh` | L3 nadir altimetry | 0.063° | sparse, along-track |
| `l3_swot` | SWOT wide swath | 0.018° | sparse, swath |
| `argo` | Argo floats | irregular | ragged points |

Those steps are the mean latitude spacing measured on the published files for
one date and region, rather than a nominal product figure. The L3 rows give the
step of the grid the tracks are stored on, which is much finer than the L4
analyses. What makes them sparse is that most of that grid is empty on any
given day.

### Step 3: the renderer fixes the output grid

`Resample((H, W), support_threshold)` puts a source on a fixed grid, while
`Native()` keeps the source's own grid and lets shapes vary between rows.
Neither is a default, because a renderer is required per source: silently
resampling would be an unrecorded scientific choice.

The second argument, `support_threshold`, sets when a resampled cell counts as
measured. When several native cells fall into one output cell, *support* is the
fraction of that output cell backed by valid source data. Cells whose support
falls below the threshold are marked invalid rather than filled with an
interpolated value, so a swath edge remains an edge instead of bleeding into
open water. It also has no default, for the same reason the renderer does
not.

`l3_swot` shows why absence needs its own representation. A swath either
crosses the patch on a given date or it does not, so on a date it misses there
is no array of the right shape with NaNs in some cells. The source is instead
structurally absent for that row, which §5 covers.

### Step 4: what this means for model design

`Native()` per source means channels no longer share a spatial shape, which
buys fidelity at the cost of per-source encoders or an explicit fusion step.
`Resample` everywhere gives uniform channels and pays interpolation. The
[renderer reference](ml_configuration_cookbook.ipynb) works through both."""),
    code("""
from ocean_taco import PatchSize
patch = PatchSize(PATCH_SIZE_KM, "km")
latitudes = [0, 30, 45, 60]
widths = [patch.to_degrees(centre_lat=lat)[0] for lat in latitudes]
plt.plot(latitudes, widths, marker="o")
plt.xlabel("centre latitude [°]")
plt.ylabel("longitude width [°]")
plt.title(f"{PATCH_SIZE_KM} km patch: latitude-aware longitude span")
display_figure(plt.gcf())
print(dict(zip(latitudes, map(lambda x: round(x, 2), widths))))
"""),
    md("""## 3. The training and eval sets

Two QuerySets are published per patch size, and they differ in how their
positions were placed. Training positions are placed stochastically, which
samples ocean conditions without imposing a lattice and lets the set grow later
without re-planning a grid. Eval positions sit on a systematic grid, which
makes coverage uniform so that a metric averaged over them is a spatial average
rather than one weighted by wherever a sampler happened to concentrate. The
`kind` field records which of the two a QuerySet is, taking the value
`"training"` or `"eval"`.

At global scale the two are hard to tell apart, since 11001 training and 6027
eval positions both cover the whole ocean. The difference lies in local
spacing, so the second row of the figure below draws a few degrees of the North
Atlantic box, with each stored position expanded to the 256 km footprint the
query actually covers. Drawing footprints rather than centres shows how much
ocean one query spans and where neighbouring queries overlap, neither of which
a dimensionless point can show.

Both placements sit on regular rows of latitude, because patch height in
degrees does not depend on longitude. Along those rows they differ: the eval
footprints repeat at a fixed longitude step, while the training footprints are
offset row by row and pack more densely, so they overlap more often. The cell
below counts the positions in the box directly.

**A QuerySet kind carries no guarantee that training and eval samples are
independent.** Both
sets span the same published dates and are drawn from the same ocean, so a
training patch and an eval patch can be neighbours in space and identical in
time. Nothing in the released artifacts prevents that, because preventing it
depends on what your model predicts and over what horizon.

In practice, choose the separation your experiment needs, apply it as a filter
to both sets, and keep the draw records. Separating in time is usually the more
defensible axis, because ocean fields are strongly correlated in space over the
scales one patch covers. The
[ML use cases notebook](spatio_temporal_query_generation.ipynb) works through
the query construction that follows from this."""),
    code("""
from ocean_taco import GeoBox, QueryFilter, select_queryset
training = QuerySet.from_hub(PATCH_SIZE_KM, "training")
evaluation = queryset
split_box = GeoBox(-80, -30, 10, 45)
train_rows = select_queryset(training, QueryFilter(box=split_box))
eval_rows = select_queryset(evaluation, QueryFilter(box=split_box))
print(f"training kind={training.header['kind']}, positions={len(training.positions)}, candidates in box={train_rows.count}")
print(f"eval     kind={evaluation.header['kind']}, positions={len(evaluation.positions)}, candidates in box={eval_rows.count}")
print(f"both sets span the same {len(training.dates)} published dates: {len(training.dates) == len(evaluation.dates)}")
for label, source in (("training", training), ("eval", evaluation)):
    lons = np.array([p["centre_lon"] for p in source.positions])
    lats = np.array([p["centre_lat"] for p in source.positions])
    inside = ((lons >= split_box.lon_min) & (lons <= split_box.lon_max)
              & (lats >= split_box.lat_min) & (lats <= split_box.lat_max))
    rows = np.unique(np.round(lats[inside], 3))
    print(f"  {label:8s} in box: {int(inside.sum()):4d} positions on {len(rows):3d} latitude rows, "
          f"{len(np.unique(np.round(lons[inside], 3))):4d} distinct longitudes")
print("Shared latitude rows, different longitude placement: the eval longitudes repeat, the training ones do not.")
print("A kind records how positions were placed. It does not separate the two sets in space or time.")
"""),
    code("""
# Top row: the whole ocean, where the two placements are hard to tell apart.
# Bottom row: query footprints in a few degrees of the North Atlantic, drawn at
# the size the patch actually covers rather than as dimensionless centres.
FOOTPRINT_BOX = GeoBox(-60, -52, 28, 34)
patch = PatchSize(PATCH_SIZE_KM, "km")
fig, axes = plt.subplots(2, 2, figsize=(11, 7.8))
sets = (("training set (stochastic)", training, "#2474a6"),
        ("eval set (systematic)", evaluation, "#238b45"))
for column, (label, source, colour) in enumerate(sets):
    lons = np.array([p["centre_lon"] for p in source.positions])
    lats = np.array([p["centre_lat"] for p in source.positions])
    # Marker size and opacity suit the global panels, which carry thousands of
    # points: one setting for both rows would smear these into a solid block.
    axes[0, column].scatter(lons, lats, s=.6, alpha=.35, color=colour, linewidths=0, rasterized=True)
    axes[0, column].add_patch(plt.Rectangle((split_box.lon_min, split_box.lat_min),
                                            split_box.lon_max - split_box.lon_min,
                                            split_box.lat_max - split_box.lat_min,
                                            fill=False, edgecolor="#d95f02", linewidth=1.6))
    axes[0, column].set(title=f"{label}: {len(lons)} positions", xlabel="longitude [°]",
                        ylabel="latitude [°]", xlim=(-180, 180), ylim=(-90, 90))
    add_coastlines(axes[0, column])

    inside = ((lons >= FOOTPRINT_BOX.lon_min) & (lons <= FOOTPRINT_BOX.lon_max)
              & (lats >= FOOTPRINT_BOX.lat_min) & (lats <= FOOTPRINT_BOX.lat_max))
    # footprint() is the same call the library uses to turn a centre into the
    # box a query covers, so these rectangles are the queries themselves.
    footprints = [patch.footprint(lon, lat) for lon, lat in zip(lons[inside], lats[inside])]
    for box in footprints:
        axes[1, column].add_patch(plt.Rectangle((box.lon_min, box.lat_min),
                                                box.lon_max - box.lon_min,
                                                box.lat_max - box.lat_min,
                                                fill=False, edgecolor=colour, linewidth=1.1, alpha=.55))
    axes[1, column].scatter(lons[inside], lats[inside], s=7, color=colour, zorder=3)
    # Limits come from the footprints rather than from the selection box, so no
    # rectangle is cut off at the edge and every overlap stays visible.
    axes[1, column].set(title=f"{len(footprints)} query footprints, {PATCH_SIZE_KM} km each",
                        xlabel="longitude [°]", ylabel="latitude [°]",
                        xlim=(min(b.lon_min for b in footprints) - .4, max(b.lon_max for b in footprints) + .4),
                        ylim=(min(b.lat_min for b in footprints) - .4, max(b.lat_max for b in footprints) + .4))
    add_coastlines(axes[1, column])
for axis in axes.ravel():
    axis.grid(alpha=.2)
fig.suptitle("Published positions, and the area the queries at those positions cover")
fig.tight_layout()
display_figure(fig)
lon_span, lat_span = patch.to_degrees(centre_lat=31.0)
print(f"At 31°N a {PATCH_SIZE_KM} km patch spans {lon_span:.2f}° of longitude and {lat_span:.2f}° of latitude.")
print("Each rectangle is a stored patch centre expanded to its footprint. No observation has been fetched yet.")
"""),
    md("""## 4. A draw, and the sample it renders

The dataset needs specific rows, so this section draws four of them from the
box above and writes the selection to a record. `replay_experiment` reads that
record back and reproduces the same rows, so a rendered batch can be recovered
later. The [deep-dive](data_retrieval_workflows.ipynb) covers
filters and draws in full, and the four rows here are what §5 renders."""),
    code("""
from ocean_taco import draw_queryset, replay_experiment
selection = QueryFilter(box=split_box)
draw = draw_queryset(queryset, requested_row_count=REQUESTED_ROWS, seed=SEED,
                     record_path=DRAW_DIR / "ml-dataset-draw.json", query_filter=selection)
replayed = replay_experiment(queryset, DRAW_DIR / "ml-dataset-draw.json")
print(f"drawn rows={len(draw.rows)}; replay reproduces them exactly: {replayed.rows == draw.rows}")
for row in draw.rows:
    print(f"  ({row['centre_lon']:7.2f}, {row['centre_lat']:6.2f}) on {row['anchor_time'][:10]}  patch={row['patch_id'][:12]}")
print(f"inclusion_probability={draw.inclusion_probability:.6g}; record={DRAW_DIR / 'ml-dataset-draw.json'}")
"""),
    md("""## 5. The sample schema, and one rendered source

`dataset[i]` returns a flat dict keyed by source token, plus a `query` entry
carrying the `PatchSpec` that produced it and an `availability` entry with one
Boolean per source. Each source's own dict holds:

| Key | Shape | Meaning |
|-----|-------|---------|
| `data` | `(T, H, W)` | decoded values, NaN where invalid |
| `source_valid` | `(T, H, W)` | the source's own validity, before rendering |
| `support_mask` | `(T, H, W)` | cells whose support met the threshold |
| `valid_mask` | `(T, H, W)` | `source_valid` and `support_mask` together |
| `support` | `(T, H, W)` | fraction of the output cell backed by source data |
| `lat`, `lon` | `(H,)`, `(W,)` | geographic coordinates of the rendered grid |
| `times` | list | the source timestamps that went into `T` |

The three masks record different causes. `source_valid` marks cells where the
instrument reported no value, `support_mask` marks cells where the renderer
could not build one from what was reported, and `valid_mask` is the conjunction
of the two, which is the mask a loss should use. Because the first two are kept
separately, a cloud-flagged pixel can still be distinguished from a swath edge
after rendering.

**Structural absence.** When a source has no asset for a position and date it
is neither dropped nor zero-filled. The renderer returns its `empty()` form:
`data` with shape `(0, H, W)`, masks zero, coordinates NaN, and
`availability[token] = False`. Because the key remains present and the batch
layout is unchanged, downstream code detects absence from the leading zero or
from `availability` without inspecting values. Zero-filling would remove that
distinction, since zero is itself a plausible SST anomaly.

The render below asks for 128×128 cells. `l4_sst` is natively about 23×23 over
a 256 km patch, so this upsamples, and the library emits a warning saying so.
The upsampling here serves a figure at display resolution. Meanwhile §2's
`native_shape` is preserved in the payload, so the raw pixel count the data
carries remains available."""),
    code("""
from ocean_taco.render import Resample
from ocean_taco.torch import OceanTACODataset
dataset = OceanTACODataset(queries=draw, sources={
    "l4_sst": Resample((128, 128), support_threshold=0.5),
    "l3_swot": Resample((128, 128), support_threshold=0.5),
}, catalog_config=config)
sample = dataset[0]
for key, value in sample.items():
    if isinstance(value, dict):
        print(key, {name: tuple(item.shape) if hasattr(item, "shape") else type(item).__name__ for name, item in value.items()})
    else:
        print(key, value)
"""),
    code("""
from ocean_taco.plot import plot_ocean_sample
# This figure stands alone rather than beside a second panel, so the colour
# range is taken from the patch itself. One open-ocean patch spans well under a
# degree, and the basin-wide range used elsewhere in these tutorials would map
# all of it onto a single shade.
values = sample["l4_sst"]["data"][sample["l4_sst"]["valid_mask"]]
low, high = float(values.min()), float(values.max())
artist = plot_ocean_sample(sample, "l4_sst", cmap=CMAP, interpolation="nearest", vmin=low, vmax=high)
artist.axes.set_title("Rendered L4 SST at 128x128 with native geographic coordinates")
add_coastlines(artist.axes)
artist.axes.figure.colorbar(artist, ax=artist.axes, label="SST [°C]")
artist.axes.figure.tight_layout()
display_figure(artist.axes.figure)
print(f"native shape before resampling: {tuple(sample['l4_sst']['native_shape'])}")
print(f"this patch spans {low:.2f} to {high:.2f} degC, a range of {high - low:.2f} degC")
"""),
    md("""## 6. Collation, native shapes, and workers

`collate_ocean_samples` is the collate function a `DataLoader` needs, and it
**collates availability separately from values**. A source absent for one batch
member changes neither the tensor layout nor the batch size, so downstream code
reads `batch["availability"]` rather than inferring presence from a shape.

**Fixed versus native shapes.** `Resample` outputs stack directly, since every
sample already shares a grid. `Native()` outputs do not, because shapes vary
row by row, and stacking them requires either padding, which invents cells, or
grouping samples that already agree. `ShapeBucketSampler` does the grouping.

**Workers.** Catalog resolution happens once in the parent process, before any
fork. `OceanTACODataset` calls `plan()` at construction to turn queries into
resolved asset locations, so worker processes never open the catalog and fetch
already-named assets instead. Since no worker touches the catalog,
`num_workers > 0` is safe here, and the same design is why construction does
visible work while `__getitem__` stays cheap. Pass
`worker_init_fn=seed_ocean_taco_worker` whenever `num_workers > 0`,
which the [training loader notebook](spatio_temporal_query_generation.ipynb)
demonstrates in a running loop.

**No implicit normalisation.** The loader returns decoded values in their
recorded units and neither centres, scales, nor fills them. §7 covers why."""),
    code("""
from torch.utils.data import DataLoader
from ocean_taco.torch import collate_ocean_samples
loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_ocean_samples)
batch = next(iter(loader))
print("batch l4_sst", tuple(batch["l4_sst"]["data"].shape))
print("availability", batch["availability"])
print("valid cells", int(batch["l4_sst"]["valid_mask"].sum()))
"""),
    md("""## 7. Normalisation, and what to persist

Normalisation happens here rather than in the loader, and it goes through
`valid_mask` so that invalid cells stay NaN. Averaging the raw array instead
would fold in whatever those cells hold and shift the statistics of every later
batch. The statistics belong to the experiment, so compute them once over the
training set and hold them fixed. Recomputing them per batch would instead make
the normalisation of a sample depend on which other samples it was batched
with."""),
    code("""
import torch
def normalise_valid(data, valid_mask, mean, std):
    # Normalise only where the mask is true, so invalid cells stay NaN rather
    # than becoming a plausible-looking zero.
    result = torch.full_like(data, float("nan"))
    scale = torch.as_tensor(std, dtype=data.dtype).clamp_min(1e-6)
    result[valid_mask] = (data[valid_mask] - mean) / scale
    return result

values, mask = sample["l4_sst"]["data"], sample["l4_sst"]["valid_mask"]
mean, std = values[mask].mean(), values[mask].std()
normalised = normalise_valid(values, mask, mean, std)
print(f"source mean={mean:.3f} std={std:.3f} over {int(mask.sum())} valid cells")
print(f"normalised mean={normalised[mask].mean():.3e} std={normalised[mask].std():.3f}")
print(f"invalid cells still NaN: {bool(torch.isnan(normalised[~mask]).all())}")
"""),
    md("""### Reproducing this sample later

Everything above is recoverable from six things, and none of them is the data
itself:

1. **The QuerySet ID**, `queryset.queryset_id`, which names the released set.
2. **The header checksums**, `table_sha256`, verified on every load, so a
   successful read is also an integrity proof.
3. **The catalog revision**, pinned in `CatalogConfig`, so the assets behind
   the rows cannot move underneath you.
4. **The filter**, the `QueryFilter` that produced the candidate rows.
5. **The draw record**, written by `draw_queryset` and read back by
   `replay_experiment`, which §4 checks.
6. **The renderer settings and normalisation statistics**, meaning the shapes,
   the `support_threshold`, and the mean and standard deviation used above.

Store those alongside model weights. Rendered tensors take more space and fix
the choices made in §2, so re-rendering at a different resolution or support
threshold requires the six values above rather than the stored arrays."""),

])

write("spatio_temporal_query_generation.ipynb", [
    md("""# ML use cases and the training loader

A machine learning setup on ocean data is mostly a statement about time and
resolution. A forecasting model requires its target strictly after its
context, a data assimilation model requires the target at the middle of the
window, and a super-resolution model requires the same patch twice at two
resolutions. Each of those is a different query, and this notebook builds four of them against the
published QuerySet, then feeds the result into a `DataLoader` that runs.

The four query shapes are:

1. **Forecasting**, where a context window is followed by a target at a lead
   time, with a check that the two windows do not overlap.
2. **Midpoint retrieval**, where the target sits at the centre of a symmetric
   window, which is the shape data assimilation and interpolation setups need.
3. **Super-resolution**, where one patch is returned at two resolutions.
4. **Multi-source sparse and dense**, where every drawn row is required to
   carry a sparse source alongside the dense fields.

The last sections then build the training pipeline: a `DataLoader` with worker
processes, batching under both fixed and native shapes, normalisation
statistics computed once, and a training step whose loss is masked.

**Coverage filters appear here but are not explained here.** The
[QuerySet and filter deep-dive](data_retrieval_workflows.ipynb) covers
`CoverageRequirement` and the null-versus-zero distinction it rests on. This
notebook uses coverage to condition a draw and links there for the reasoning.
For the concepts behind QuerySets, draws, and renderers, start with
[From a published QuerySet to a rendered sample](ml_dataset.ipynb)."""),
    code(SETUP), code(PLOTTING), code(LOAD),
    md("""## 1. Forecasting: context now, target later

A forecasting setup needs the context window and the target to be separate
objects, so that a model consuming context cannot reach the target by accident.
`QueryFilter` states the relation and the lead time, and the two datasets then
differ only in their context offsets.

The mechanism is worth stating plainly, because the same one drives every
construction in this notebook. `draw_queryset` builds every row from a single
`QueryFilter`, so all rows in a draw share its offsets. To get a second set of
rows anchored identically but pointing at a different time, copy the drawn rows
and override `context_start_offset_days` and `context_end_offset_days` on the
copies. `OceanTACODataset` accepts a plain sequence of row mappings, so those
copies are a valid dataset input.

Draw once and derive both datasets from that single draw. Drawing twice would
change which rows are eligible and silently compare different anchors.

**What round-trips and what does not.** The draw record describes the draw, so
the context rows replay exactly. The target rows are rebuilt in notebook code
from that draw, because the experiment record stores only the filter-level
offsets and therefore does not describe the overridden copies. Persist the draw
record and the override rule together, rather than assuming the whole pipeline
round-trips from the record alone.

The printed check is the one that matters: the two windows must be disjoint, or
the model can see its own target."""),
    code("""
from ocean_taco import GeoBox, QueryFilter, draw_queryset, replay_experiment, select_queryset
from ocean_taco.render import Native, Resample
from ocean_taco.torch import OceanTACODataset

BOX = GeoBox(-80, -30, 10, 45)
SOURCE = {"l4_sst": Resample((32, 32), support_threshold=0.5)}

def offset_rows(rows, start, end):
    # Copy drawn rows and point them at a different span around the same
    # anchor. OceanTACODataset accepts a plain sequence of row mappings.
    return [{**row, "context_start_offset_days": start, "context_end_offset_days": end} for row in rows]

def window_of(dataset):
    context = dataset[0]["query"].context
    return f"{context.start:%Y-%m-%d} .. {context.end:%Y-%m-%d}"

forecast_filter = QueryFilter(box=BOX, relation="forecast", target_lead_days=1,
                              context_start_offset_days=-1, context_end_offset_days=0)
forecast_draw = draw_queryset(queryset, requested_row_count=4, seed=29,
                              record_path=DRAW_DIR / "forecast-draw.json",
                              query_filter=forecast_filter)
lead = forecast_filter.target_lead_days
context_set = OceanTACODataset(queries=forecast_draw, sources=SOURCE, catalog_config=config)
target_set = OceanTACODataset(queries=offset_rows(forecast_draw.rows, lead, lead), sources=SOURCE, catalog_config=config)
context_window = context_set[0]["query"].context
target_window = target_set[0]["query"].context
print(f"anchor:  {forecast_draw.rows[0]['anchor_time'][:10]}")
print(f"context: {window_of(context_set)}")
print(f"target:  {window_of(target_set)}  (lead {lead} day)")
print(f"windows disjoint: {target_window.start > context_window.end}")
print(f"draw record replays: {replay_experiment(queryset, DRAW_DIR / 'forecast-draw.json').rows == forecast_draw.rows}")
"""),
    md("""## 2. Midpoint retrieval for assimilation and interpolation

Forecasting places the target after the context. Assimilation and
interpolation setups place it inside: the model sees a span of days on both
sides and predicts the state in the middle. That reordering is expressed
entirely through the target offsets passed to the same filter, so it needs no
library change.

`QueryFilter.relation` takes `"same_time"` or `"forecast"`, where `"forecast"`
requires a positive `target_lead_days` and `"same_time"` requires zero. The
relation alone therefore expresses only "target simultaneous with the anchor"
or "target strictly in the future". The context offsets carry the rest: they
are signed integers with only `context_end_offset_days >= context_start_offset_days`
enforced, so a symmetric window like `(-2, +2)` is legal.

That gives the construction. Use `relation="same_time"` with a symmetric
window, then override the target rows to `(0, 0)`, which is the anchor and
therefore the midpoint of the context span. The difference between this section
and the previous one is exactly which offsets the target rows carry.

Anchors near the ends of the record are rejected rather than silently
truncated, because the filter requires every date across the window to exist in
the QuerySet. A wider window therefore leaves fewer eligible rows, which the
printed counts show."""),
    code("""
HALF_WINDOW = 2
midpoint_filter = QueryFilter(box=BOX, relation="same_time",
                              context_start_offset_days=-HALF_WINDOW,
                              context_end_offset_days=HALF_WINDOW)
print(f"eligible pairs with a +/-{HALF_WINDOW} day window: {select_queryset(queryset, midpoint_filter).count}")
print(f"eligible pairs with no window at all:      {select_queryset(queryset, QueryFilter(box=BOX)).count}")

midpoint_draw = draw_queryset(queryset, requested_row_count=4, seed=11,
                              record_path=DRAW_DIR / "midpoint-draw.json",
                              query_filter=midpoint_filter)
window_set = OceanTACODataset(queries=midpoint_draw, sources=SOURCE, catalog_config=config)
centre_set = OceanTACODataset(queries=offset_rows(midpoint_draw.rows, 0, 0), sources=SOURCE, catalog_config=config)
print(f"anchor:  {midpoint_draw.rows[0]['anchor_time'][:10]}")
print(f"context: {window_of(window_set)}  -> data {tuple(window_set[0]['l4_sst']['data'].shape)}")
print(f"target:  {window_of(centre_set)}  -> data {tuple(centre_set[0]['l4_sst']['data'].shape)}")
print("The target date sits at the centre of the context span, with equal numbers of days on each side.")
"""),
    code("""
# The two query shapes drawn on a day axis, so the difference between them is
# visible as geometry rather than only as printed dates.
fig, axes = plt.subplots(2, 1, figsize=(9, 3.6), sharex=True)
layouts = (("forecast: target after the context", (-1, 0), (lead, lead)),
           (f"midpoint: target inside a +/-{HALF_WINDOW} day context", (-HALF_WINDOW, HALF_WINDOW), (0, 0)))
for axis, (title, context_span, target_span) in zip(axes, layouts):
    axis.barh(0, context_span[1] - context_span[0] + 1, left=context_span[0] - .5,
              height=.55, color="#2474a6", label="context")
    axis.barh(0, target_span[1] - target_span[0] + 1, left=target_span[0] - .5,
              height=.32, color="#d95f02", label="target")
    axis.axvline(0, color="#333333", linewidth=.9, linestyle=":")
    axis.set(title=title, yticks=[], xlim=(-3.5, 3.5), ylim=(-.6, .6))
    axis.grid(axis="x", alpha=.2)
axes[0].legend(loc="upper right", ncol=2, fontsize=9)
axes[1].set_xlabel("days relative to the anchor (dotted line)")
fig.tight_layout()
display_figure(fig)
print("Both shapes come from one draw each. Only the offsets on the target rows differ.")
"""),
    md("""## 3. Super-resolution: native grids, and a power-of-two rescaling

The sources in this catalog do not share a resolution. Over one patch, an L4
analysis arrives on a grid of a few tens of cells per side, nadir altimetry
arrives as a narrow track sampled densely along it, and the SWOT swath arrives
at a few hundred cells per side. A super-resolution setup exists because of
that spread, so the first thing to look at is the spread itself.

This section uses the 512 km QuerySet rather than the 256 km one used
elsewhere. A larger patch carries more spatial context, which super-resolution
architectures require, and it also makes the resolution differences between
products easier to see in one figure.

`Native()` renders each source on its own grid without resampling, so the shape
that comes back is the shape the product actually has over this box. The four
sources are rendered together below and their shapes printed alongside the
panels."""),
    code("""
from ocean_taco import CoverageRequirement

SUPER_PATCH_KM = 512
super_queryset = QuerySet.from_hub(SUPER_PATCH_KM, "eval")
SUPER_SOURCES = ("l4_sst", "l4_ssh", "l3_ssh", "l3_swot")

# Condition the draw so every row carries SWOT. The deep-dive notebook explains
# what this requirement reads and why null coverage is not zero coverage.
swot_present = QueryFilter(box=BOX, coverage=(CoverageRequirement("swot", "valid_fraction_ocean", 0.2),))
super_draw = draw_queryset(super_queryset, requested_row_count=8, seed=5,
                           record_path=DRAW_DIR / "superres-draw.json",
                           query_filter=swot_present)
native_set = OceanTACODataset(queries=super_draw,
                              sources={token: Native() for token in SUPER_SOURCES},
                              catalog_config=config)
native_rows = [native_set[index] for index in range(len(native_set))]
print(f"{len(native_rows)} rows drawn from the {SUPER_PATCH_KM} km QuerySet, each carrying all four sources.")
print()
print(f"{'source':10s} {'native shape':>16s}   {'cells':>8s}   degrees per cell")
for token in SUPER_SOURCES:
    record = native_rows[0][token]
    height, width = np.asarray(record["data"]).shape[-2:]
    lon, lat = np.asarray(record["lon"]).ravel(), np.asarray(record["lat"]).ravel()
    step_lon = abs(float(lon[-1] - lon[0])) / max(width - 1, 1)
    step_lat = abs(float(lat[-1] - lat[0])) / max(height - 1, 1)
    print(f"{token:10s} {f'{height} x {width}':>16s}   {height * width:>8,d}   {step_lat:.3f} lat, {step_lon:.3f} lon")
print()
print("One box, one date, four different grids. That mismatch is what a super-resolution setup addresses.")
"""),
    code("""
# The same box in all four panels, each on its own native grid. Panels share a
# colour range per quantity so the difference between them is resolution and
# coverage rather than colour scaling.
row = native_rows[0]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
for axis, token in zip(axes, SUPER_SOURCES):
    record = row[token]
    values = np.asarray(record["data"])[0]
    shown = np.where(np.asarray(record["valid_mask"])[0], values, np.nan)
    drawn = axis.imshow(shown, origin="lower", cmap=CMAP, interpolation="nearest",
                        aspect="auto", extent=geographic_extent(record), **color_limits(token))
    axis.set(title=f"{token}: {values.shape[0]} x {values.shape[1]} native",
             xlabel="longitude [°]", ylabel="latitude [°]")
    add_coastlines(axis)
    fig.colorbar(drawn, ax=axis, shrink=.85)
fig.suptitle(f"One {SUPER_PATCH_KM} km box on four native grids, at {row['query'].context.start:%Y-%m-%d}")
display_figure(fig)
print("l4_sst and l4_ssh are dense and smooth. l3_ssh is a narrow nadir track and l3_swot a wide swath,")
print("both leaving most of the box unmeasured, which is why the white areas differ between panels.")
"""),
    md("""### Configuring `Resample` for a power-of-two factor

Super-resolution architectures built on pixel shuffle or on stacked strided
convolutions need the two shapes to stand in an integer ratio, usually a power
of two, because each stage doubles one axis. `Resample` takes the output shape
directly, so the factor is whatever the caller makes it, and the calculation
worth writing down is the one that satisfies two requirements at once.

The first requirement is the power of two. The second is that neither shape
exceeds the native grid, since resampling above native invents detail the
instrument never resolved, and `Resample` warns when a target exceeds native by
more than 2x. These two are compatible, and the way to satisfy both is to take
the smallest native side across the rows being drawn, then pick the largest
power-of-two-divisible shape at or below it.

The cell below performs that calculation from the measured native shapes rather
than asserting the numbers, so the same code selects valid shapes for a
different box, patch size or source pair."""),
    code("""
FACTOR = 4          # the super-resolution factor the architecture expects
SR_PAIR = ("l3_ssh", "l3_swot")

# The fine shape must fit inside the native grid of every drawn row, so the
# binding number is the smallest side seen across all of them.
smallest_side = min(min(np.asarray(sample[token]["data"]).shape[-2:])
                    for sample in native_rows for token in SR_PAIR)
# Round down to a multiple of the factor, so dividing by it stays an integer.
fine_side = (smallest_side // FACTOR) * FACTOR
coarse_side = fine_side // FACTOR
print(f"smallest native side over {len(native_rows)} rows and {len(SR_PAIR)} sources: {smallest_side}")
print(f"fine target: {fine_side} x {fine_side}, coarse input: {coarse_side} x {coarse_side}, factor {FACTOR}x")
print(f"fine shape stays at or below native: {fine_side <= smallest_side}")
print(f"the ratio is an exact power of two: {fine_side // coarse_side == FACTOR}")

coarse_set = OceanTACODataset(queries=super_draw,
                              sources={token: Resample((coarse_side, coarse_side), support_threshold=0.5) for token in SR_PAIR},
                              catalog_config=config)
fine_set = OceanTACODataset(queries=super_draw,
                            sources={token: Resample((fine_side, fine_side), support_threshold=0.5) for token in SR_PAIR},
                            catalog_config=config)
coarse_rows = [coarse_set[index] for index in range(len(coarse_set))]
fine_rows = [fine_set[index] for index in range(len(fine_set))]
print()
print(f"coarse batch shape: {tuple(np.asarray(coarse_rows[0][SR_PAIR[0]]['data']).shape)}")
print(f"fine batch shape:   {tuple(np.asarray(fine_rows[0][SR_PAIR[0]]['data']).shape)}")
print("No upsampling warning is raised, because both shapes stay at or below native for every row.")
"""),
    code("""
# Four drawn rows rather than one, so the pair is judged across varied swath
# coverage instead of on a single favourable sample.
shown_rows = range(min(4, len(coarse_rows)))
fig, axes = plt.subplots(len(SR_PAIR) * 2, len(shown_rows),
                         figsize=(3.4 * len(shown_rows), 3.1 * len(SR_PAIR) * 2),
                         constrained_layout=True)
for column, index in enumerate(shown_rows):
    stages = [(token, f"{token} {label} {side}x{side}", rows[index][token])
              for token in SR_PAIR
              for label, side, rows in (("input", coarse_side, coarse_rows), ("target", fine_side, fine_rows))]
    for axis, (token, title, record) in zip(axes[:, column], stages):
        values = np.asarray(record["data"])[0]
        shown = np.where(np.asarray(record["valid_mask"])[0], values, np.nan)
        axis.imshow(shown, origin="lower", cmap=CMAP, interpolation="nearest",
                    aspect="auto", extent=geographic_extent(record), **color_limits(token))
        axis.set(title=title, xticks=[], yticks=[])
    axes[0, column].set_xlabel(f"row {index}")
fig.suptitle(f"{FACTOR}x super-resolution pairs across {len(shown_rows)} drawn rows")
display_figure(fig)
print(f"Each column is a different position and date. The {FACTOR}x factor holds for every one of them,")
print("while how much of the box each instrument measured changes from column to column.")
"""),
    md("""## 4. Multi-source: requiring a sparse source alongside dense fields

A model that fuses a sparse observation with dense analyses needs rows where
the sparse source is actually present. Drawing rows at random and discarding
the ones that miss would work, but it wastes fetches and makes the row count
depend on luck. Conditioning the draw on recorded coverage instead states the
requirement up front and lets the filter find rows that satisfy it, at no
download cost, because coverage reads the published fact table.

The draw in §3 already carries that requirement, so this section reuses it and
adds the dense fields. Every row carries SWOT by construction, which the
availability count confirms.

Metrics are validated per token and form a closed set. `swot` accepts
`valid_cells`, `valid_ocean_cells`, `n_obs_sum`, `valid_fraction_footprint`,
and `valid_fraction_ocean`. `ssh` accepts the same set without `n_obs_sum`, and
`argo` accepts `profile_count`. The
[deep-dive](data_retrieval_workflows.ipynb) covers what those values mean."""),
    code("""
multi_sources = {
    "l4_sst": Resample((32, 32), support_threshold=0.5),
    "l4_ssh": Resample((32, 32), support_threshold=0.5),
    "l3_swot": Resample((32, 32), support_threshold=0.5),
}
multi_set = OceanTACODataset(queries=super_draw, sources=multi_sources, catalog_config=config)
present = {token: 0 for token in multi_sources}
for index in range(len(super_draw.rows)):
    for token, available in multi_set[index]["availability"].items():
        present[token] += bool(available)
rows = len(super_draw.rows)
for token, count in present.items():
    print(f"  {token:9s} available in {count}/{rows} drawn rows")
first = multi_set[0]
print("valid fraction per source in row 0:")
for token in multi_sources:
    mask = np.asarray(first[token]["valid_mask"])
    print(f"  {token:9s} {mask.mean():.3f}" if mask.size else f"  {token:9s} structurally absent")
print("A low valid fraction on a sparse source is evidence about sampling, not about the ocean.")
"""),
    md("""## 5. Mixed output shapes in one sample

The sources in a single sample do not have to share a shape. Collation runs per
token, so a `Resample` token and a `Native()` token in the same configuration
collate into two differently-shaped stacks without interfering.

`Native()` is where shapes genuinely vary between rows, because the swath
crosses each patch differently. Stacking then needs either padding, which
invents cells, or grouping rows that already agree. `ShapeBucketSampler` does
the grouping, and §7 uses it in a loader.

Building the sampler calls `native_shapes`, which is a deliberate O(N)
rendering pass in the parent process. It is not free on a large draw, so treat
it as setup cost rather than something to call per epoch."""),
    code("""
from ocean_taco.torch import native_shapes
mixed = OceanTACODataset(queries=super_draw, sources={
    "l4_sst": Resample((32, 32), support_threshold=0.5),
    "l3_swot": Native(),
}, catalog_config=config)
for index in range(3):
    record = mixed[index]
    print(f"row {index}: l4_sst={tuple(np.asarray(record['l4_sst']['data']).shape)} "
          f"l3_swot={tuple(np.asarray(record['l3_swot']['data']).shape)}")
swot_shapes = native_shapes(mixed, "l3_swot")
print(f"native l3_swot shapes across the draw: {swot_shapes}")
print(f"distinct shapes: {len(set(swot_shapes))} of {len(swot_shapes)} rows")
"""),
    md("""## 6. Normalisation statistics, computed once

Statistics belong to the experiment rather than to the batch. Computing them
per batch would let batch composition reach the inputs, so this section
computes one mean and standard deviation over the training rows and holds them
fixed for everything downstream.

The average runs through `valid_mask`. Averaging the raw array instead would
fold in cells the instrument never measured, and since those cells are NaN the
result would be NaN. The printed comparison against a zero-filled average shows
what the mask excludes: zero-filling treats every unobserved cell as a
measured zero, so the mean moves towards zero in proportion to how much of the
patch went unobserved."""),
    code("""
import torch
training_set = QuerySet.from_hub(PATCH_SIZE_KM, "training")
stats_draw = draw_queryset(training_set, requested_row_count=8, seed=3,
                           record_path=DRAW_DIR / "stats-draw.json",
                           query_filter=QueryFilter(box=BOX))
stats_data = OceanTACODataset(queries=stats_draw, sources=SOURCE, catalog_config=config)
total, count, squares = 0.0, 0, 0.0
for index in range(len(stats_draw.rows)):
    record = stats_data[index]["l4_sst"]
    values, mask = record["data"], record["valid_mask"]
    if mask.sum():
        selected = values[mask]
        total += float(selected.sum()); squares += float((selected ** 2).sum()); count += int(mask.sum())
SST_MEAN = total / count
SST_STD = max((squares / count - SST_MEAN ** 2) ** .5, 1e-6)
print(f"training rows used: {len(stats_draw.rows)}; valid cells: {count}")
print(f"fixed statistics: mean={SST_MEAN:.3f} degC, std={SST_STD:.3f} degC")
# The same average taken two ways on a sparse source, where the mask actually
# excludes cells. l4_sst is dense and complete here, so it would show no
# difference at all.
sparse_data = OceanTACODataset(queries=super_draw, sources={"l3_swot": Resample((32, 32), support_threshold=0.5)},
                               catalog_config=config)
sparse_record = sparse_data[0]["l3_swot"]
sparse_values, sparse_mask = sparse_record["data"], sparse_record["valid_mask"]
filled = torch.nan_to_num(sparse_values, nan=0.0)
print(f"l3_swot valid cells: {int(sparse_mask.sum())} of {sparse_mask.numel()}")
print(f"  masked mean      = {float(sparse_values[sparse_mask].mean()):.4f} m")
print(f"  zero-filled mean = {float(filled.mean()):.4f} m")
print("Zero-filling counts every unsampled cell as a measured zero and pulls the average towards it.")
"""),
    md("""## 7. The training loader

Three details decide whether a loader built on this library works, and each is
easy to get wrong.

**The collate function.** `collate_ocean_samples` defaults to
`native="ragged"`, which returns `{"items": [...]}` for a `Native()` token
rather than a tensor. That default is deliberate, since padding a native grid
invents cells. For a `DataLoader` that indexes tensors, pass `native_pad_collate`,
which is the same function with `native="padded"` and which labels every padded
cell through `spatial_padding_mask`. Fixed-grid tokens stack either way.

**Worker seeding.** With `num_workers > 0`, pass
`worker_init_fn=seed_ocean_taco_worker`. It reseeds Python and NumPy from
PyTorch's per-worker seed and resets the shipped loader in each worker. Worker
safety is otherwise handled inside the library, since the dataset nulls its
backends when pickled and a fork guard drops parent catalog state, so no file
handles cross the fork.

**Bucketing shuffles by default.** `ShapeBucketSampler` shuffles within and
across buckets unless told otherwise, so pass `shuffle=False` or a fixed seed
for reproducible output.

Both regimes appear below: fixed `Resample` shapes stacking directly, and
`Native()` shapes batched through the bucket sampler."""),
    code("""
from torch.utils.data import DataLoader
from ocean_taco.torch import ShapeBucketSampler, collate_ocean_samples, native_pad_collate, seed_ocean_taco_worker

fixed_loader = DataLoader(mixed, batch_size=2, num_workers=2,
                          collate_fn=native_pad_collate,
                          worker_init_fn=seed_ocean_taco_worker)
batch = next(iter(fixed_loader))
print("fixed-shape token stacks directly:")
print(f"  l4_sst data={tuple(batch['l4_sst']['data'].shape)}")
print("native token is padded, and the padding is labelled:")
print(f"  l3_swot data={tuple(batch['l3_swot']['data'].shape)} "
      f"padding_mask={tuple(batch['l3_swot']['spatial_padding_mask'].shape)}")
print(f"  availability: {batch['availability']}")

native_only = OceanTACODataset(queries=super_draw, sources={"l3_swot": Native()}, catalog_config=config)
sampler = ShapeBucketSampler(native_shapes(native_only, "l3_swot"), batch_size=2, seed=19, shuffle=False)
bucket_loader = DataLoader(native_only, batch_sampler=sampler, collate_fn=native_pad_collate,
                           num_workers=2, worker_init_fn=seed_ocean_taco_worker)
print(f"bucketed batches: {len(sampler)}")
for index, bucketed in enumerate(bucket_loader):
    print(f"  batch {index}: {tuple(bucketed['l3_swot']['data'].shape)} "
          f"padded cells={int(bucketed['l3_swot']['spatial_padding_mask'].sum())}")
"""),
    md("""## 8. A training step with a masked loss

There is no single mask. A collated batch carries `valid_mask`, `source_valid`,
`support_mask`, `time_mask`, and `ocean_mask`, plus a top-level `availability`
dict. The relation between the first three is:

> `valid_mask = source_valid & support_mask`, further ANDed with the ocean mask
> when one is supplied.

The split exists so that a reader can tell why a cell is invalid: the source
reported nothing (`source_valid`), the renderer had too little support to build
a value (`support_mask`), or the cell is land (`ocean_mask`). Drive the loss
from `valid_mask` and use the others to explain it. For `VectorPair`,
`pair_available` is the sample-level Boolean and `valid_mask` covers cells where
both components have support.

The step below trains one small convolution to predict the midpoint field from
its context. The loss is computed only on cells valid in both the input and the
target, so unobserved cells contribute no gradient.

One batch in this small draw has no target at all, because L4 SST has no asset
for those two anchor dates and the target render is therefore structurally
absent. The loop reads `availability` and skips it. A batch of absent targets
is an ordinary event at this draw size rather than a failure, and reading
availability is how a training loop tells that case apart from a batch whose
targets are present but heavily masked."""),
    code("""
def masked_mse(prediction, target, mask):
    # Cells outside the mask contribute no gradient at all, rather than
    # contributing a difference against a filled-in value.
    if not bool(mask.any()):
        return torch.zeros((), requires_grad=True)
    return ((prediction - target)[mask] ** 2).mean()

def normalise(values, mask):
    output = torch.full_like(values, float("nan"))
    output[mask] = (values[mask] - SST_MEAN) / SST_STD
    return torch.nan_to_num(output, nan=0.0)

torch.manual_seed(0)
model = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
optimiser = torch.optim.SGD(model.parameters(), lr=0.05)

context_loader = DataLoader(
    OceanTACODataset(queries=midpoint_draw, sources=SOURCE, catalog_config=config),
    batch_size=2, num_workers=2, collate_fn=collate_ocean_samples, worker_init_fn=seed_ocean_taco_worker)
target_loader = DataLoader(
    OceanTACODataset(queries=offset_rows(midpoint_draw.rows, 0, 0), sources=SOURCE, catalog_config=config),
    batch_size=2, num_workers=2, collate_fn=collate_ocean_samples, worker_init_fn=seed_ocean_taco_worker)

for step, (context_batch, target_batch) in enumerate(zip(context_loader, target_loader)):
    context, target = context_batch["l4_sst"], target_batch["l4_sst"]
    present = target_batch["availability"]["l4_sst"]
    if not any(present):
        # Every target in this batch is structurally absent, so there is
        # nothing to compare a prediction against. Skipping is the honest
        # response, and availability is what says so.
        print(f"step {step}: no target available for any batch member ({present}), skipped")
        continue
    # Average the context days down to one channel, keeping only valid days.
    context_valid = context["valid_mask"]
    inputs = normalise(context["data"], context_valid).mean(dim=1, keepdim=True)
    targets = normalise(target["data"], target["valid_mask"])
    mask = context_valid.any(dim=1, keepdim=True) & target["valid_mask"]
    loss = masked_mse(model(inputs), targets, mask)
    optimiser.zero_grad(); loss.backward(); optimiser.step()
    print(f"step {step}: batch={tuple(inputs.shape)} targets available={present} "
          f"valid cells={int(mask.sum())} of {mask.numel()} loss={float(loss):.4f}")
print("The loss saw only cells valid in both the context and the target.")
"""),
    code("""
# Where a mask excludes cells, and why. The sparse source is the one that
# shows this: a dense L4 analysis is valid nearly everywhere, so its three
# panels would be indistinguishable.
mask_record = sparse_data[0]["l3_swot"]
data_panel = np.asarray(mask_record["data"])[0]
source_panel = np.asarray(mask_record["source_valid"])[0]
support_panel = np.asarray(mask_record["support_mask"])[0]
valid_panel = np.asarray(mask_record["valid_mask"])[0]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
panels = (("data [m]", data_panel, CMAP, color_limits("l3_swot")),
          (f"source_valid ({source_panel.mean():.2f})", source_panel, "Greys_r", {"vmin": 0, "vmax": 1}),
          (f"support_mask ({support_panel.mean():.2f})", support_panel, "Greys_r", {"vmin": 0, "vmax": 1}),
          (f"valid_mask ({valid_panel.mean():.2f}), drives the loss", valid_panel, "Greys_r", {"vmin": 0, "vmax": 1}))
mask_extent = geographic_extent(mask_record)
for axis, (title, values, colormap, limits) in zip(axes, panels):
    drawn = axis.imshow(values, origin="lower", cmap=colormap, interpolation="nearest",
                        aspect="auto", extent=mask_extent, **limits)
    axis.set(title=title, xlabel="longitude [°]", ylabel="latitude [°]")
    add_coastlines(axis)
    fig.colorbar(drawn, ax=axis, shrink=.85)
display_figure(fig)
print(f"valid_mask equals source_valid AND support_mask: "
      f"{bool((valid_panel == (source_panel & support_panel)).all())}")
"""),
    md("""### What to carry forward

The pieces this notebook assembled are the ones an experiment has to persist:
the draw record, the override rule that produced the target rows, the renderer
shapes, and the normalisation statistics computed in §6. Those five items plus
the QuerySet ID and the pinned catalog revision reproduce every tensor above,
which the [overview notebook](ml_dataset.ipynb) lists in full."""),

])

write("data_retrieval_workflows.ipynb", [
    md("""# QuerySet selection and native-coordinate retrieval

Most of the work in building an ocean dataset happens before anything is
downloaded. A published QuerySet carries a position table, a per-position and
per-date coverage table, and an asset-identity table, and a filter reads those
tables to answer questions like "how many patches in this box were actually
observed by SWOT" without fetching a single granule. This notebook covers that
selection layer first, then the retrieval API underneath it.

The two halves serve different needs. Sections 1 to 4 are the QuerySet layer:
what a filter narrows, how coverage evidence is recorded, and where a selection
lands in space and time. Sections 5 to 8 are the lower-level retrieval API,
which fetches named assets in their native coordinates and is what you reach
for when working outside the QuerySet flow.

This notebook is the reference for coverage filtering. The
[ML use cases notebook](spatio_temporal_query_generation.ipynb) uses coverage
requirements to condition a draw and links here rather than re-explaining them.
For the four-stage overview, see
[From a published QuerySet to a rendered sample](ml_dataset.ipynb)."""),
    code(SETUP), code(PLOTTING), code(LOAD),
    md("""## 1. What a filter narrows

A `QueryFilter` restricts the published set along three independent axes:
geography through `box`, time through `date_start` and `date_end`, and recorded
evidence through `coverage`. `select_queryset` applies a filter and returns the
count without fetching anything, so you can see the size of a selection before
committing to a download.

The counts below narrow in sequence. The starting number is every published
position paired with every published date, which is a large number precisely
because the QuerySet stores positions and dates as separate tables and takes
their product."""),
    code("""
from ocean_taco import CoverageRequirement, GeoBox, QueryFilter, select_queryset
box = GeoBox(-80, -30, 10, 45)
all_pairs = len(queryset.positions) * len(queryset.dates)
in_box = select_queryset(queryset, QueryFilter(box=box))
in_box_and_time = select_queryset(queryset, QueryFilter(box=box, date_start=queryset.dates[0], date_end=queryset.dates[89]))
observed = select_queryset(queryset, QueryFilter(box=box, coverage=(CoverageRequirement("ssh", "valid_cells", 1),)))
print(f"published pairs:           {all_pairs:,}  ({len(queryset.positions)} positions x {len(queryset.dates)} dates)")
print(f"inside the box:            {in_box.count:,}")
print(f"box and first 90 dates:    {in_box_and_time.count:,}")
print(f"box and observed SSH:      {observed.count:,}")
print("None of these counts fetched a granule. They read the published tables.")
"""),
    code("""
labels = ["published", "in box", "box + 90 days", "box + SSH evidence"]
counts = [all_pairs, in_box.count, in_box_and_time.count, observed.count]
fig, axis = plt.subplots(figsize=(8, 3.6))
bars = axis.bar(labels, counts, color=["#9aa5b1", "#2474a6", "#4a90c0", "#238b45"])
axis.set_yscale("log")
axis.set_ylabel("position/date pairs (log scale)")
axis.set_title("Each filter axis narrows the published set before any download")
for bar, value in zip(bars, counts):
    axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
display_figure(fig)
print("The last bar counts only pairs whose recorded SSH coverage was measured and non-zero.")
"""),
    md("""## 2. Null and zero mean different things

The coverage table records what was measured about each position and date, and
it separates two states that are easy to conflate. A **null** means the
evidence was never established for that pair, because nobody looked or the
check could not run. A **zero** means it was established and the answer was
none: SWOT crossed and contributed no valid cells.

Coercing null to zero would turn "unknown" into a confident claim of absence,
and a filter built on that would select on measurement effort rather than on
the ocean. Coverage filters therefore reject null evidence rather than treating
it as a zero that fails the threshold.

The counts below come from the published table, so they describe the release
rather than this notebook's selection."""),
    code("""
coverage = queryset.coverage
nulls = sum(row["swot_valid_cells"] is None for row in coverage)
zeros = sum(row["swot_valid_cells"] == 0 for row in coverage)
measured = sum(row["swot_valid_cells"] is not None and row["swot_valid_cells"] > 0 for row in coverage)
print(f"SWOT evidence across {len(coverage):,} published pairs:")
print(f"  null (never measured):        {nulls:,}")
print(f"  zero (measured, none found):  {zeros:,}")
print(f"  positive (measured, present): {measured:,}")
print("A coverage filter rejects null evidence rather than reading it as a zero.")
"""),
    md("""## 3. Coverage requirements

A `CoverageRequirement` names a token, a metric, and a minimum, and the filter
keeps only pairs whose recorded value meets it. Because it reads the published
fact table, stating "only positions where SWOT covered at least a fifth of the
ocean area" costs no download and no rendering.

Metrics are validated per token and form a closed set:

| Token | Metrics |
|-------|---------|
| `swot` | `valid_cells`, `valid_ocean_cells`, `n_obs_sum`, `valid_fraction_footprint`, `valid_fraction_ocean` |
| `ssh` | `valid_cells`, `valid_ocean_cells`, `valid_fraction_footprint`, `valid_fraction_ocean` |
| `argo` | `profile_count` |

When a requirement spans a context window rather than a single date, the
`aggregate` argument decides how the per-date values combine, taking `"sum"`,
`"mean"`, or `"min"`. Use `"min"` to require the threshold on every day of the
window rather than on the window as a whole."""),
    code("""
requirements = [
    ("swot", "valid_fraction_ocean", 0.2),
    ("swot", "valid_fraction_ocean", 0.5),
    ("ssh", "valid_cells", 1),
    ("argo", "profile_count", 1),
]
print(f"pairs in box with no coverage requirement: {in_box.count:,}")
for token, metric, minimum in requirements:
    selection = select_queryset(queryset, QueryFilter(box=box, coverage=(CoverageRequirement(token, metric, minimum),)))
    share = selection.count / in_box.count
    print(f"  {token:5s} {metric:22s} >= {minimum:<5} -> {selection.count:8,} pairs ({share:6.1%} of the box)")
print("Argo is the sparse extreme: most patches on most days contain no float profile at all.")
"""),
    md("""## 4. Where a selection sits

A count alone does not say whether a selection is concentrated in one corner of
the box or spread across it, and neither does it say how it is distributed
through time. The two panels below plot both for the SWOT-conditioned selection
above, which is the one the
[ML use cases notebook](spatio_temporal_query_generation.ipynb) draws from.

The left panel shows which positions survive the coverage requirement and how
often each one does. The right panel counts surviving pairs per month, which is
where a sampling artifact would show up as a gap or a spike."""),
    code("""
from collections import Counter
selected = select_queryset(queryset, QueryFilter(box=box, coverage=(CoverageRequirement("swot", "valid_fraction_ocean", 0.2),)))
position_hits, month_hits = Counter(), Counter()
for position_index, date_index in selected.iter_pairs():
    position_hits[position_index] += 1
    month_hits[queryset.dates[date_index][:7]] += 1
lons = np.array([queryset.positions[index]["centre_lon"] for index in position_hits])
lats = np.array([queryset.positions[index]["centre_lat"] for index in position_hits])
hits = np.array([position_hits[index] for index in position_hits])
print(f"{selected.count:,} selected pairs over {len(position_hits)} distinct positions and {len(month_hits)} months")
print(f"dates per position: min={hits.min()} median={int(np.median(hits))} max={hits.max()}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
drawn = axes[0].scatter(lons, lats, c=hits, s=30, cmap="viridis", edgecolor="white", linewidth=.3)
fig.colorbar(drawn, ax=axes[0], label="dates meeting the requirement")
axes[0].set(title="Selected positions, coloured by how often they qualify",
            xlabel="longitude [°]", ylabel="latitude [°]")
months = sorted(month_hits)
axes[1].bar(range(len(months)), [month_hits[month] for month in months], color="#2474a6")
axes[1].set(title="Selected pairs per month", xlabel="month", ylabel="pairs",
            xticks=range(0, len(months), 3), xticklabels=[months[index] for index in range(0, len(months), 3)])
axes[1].tick_params(axis="x", rotation=45)
for axis in axes:
    axis.grid(alpha=.2)
fig.tight_layout()
display_figure(fig)
print("Coverage varies by position and by month, which is a property of the satellite orbit rather than of the filter.")
"""),
    md("""## 5. The retrieval API: catalog rows and source tokens

The rest of this notebook works below the QuerySet layer, fetching named assets
directly. Reach for this when you need a field outside the published patch
positions, or when inspecting what an asset holds.

A source token names a variable inside an asset rather than a file. The
registry maps each token to the asset holding it and the variable to read from
that asset, which is why several tokens can share one file: the GLORYS tokens
all resolve to `glorys.nc` and differ only in which variable they select.

The registry also records the geometry, and that record governs the handling
downstream. A `dense_grid` source is merged and cropped as a field, while a
`ragged_points` source such as `argo` keeps individual float positions and is
never rasterised on retrieval."""),
    code("""
from ocean_taco.registry import MODALITY_REGISTRY
print(f"catalog URL={config.resolved_catalog_url}")
print(f"{'token':10s} {'geometry':14s} {'filename':14s} {'unit':8s} primary variable")
for token in ("l4_sst", "l4_ssh", "l4_sss", "l3_ssh", "l3_swot", "argo"):
    spec = MODALITY_REGISTRY[token]
    geometry = "ragged_points" if spec.is_points else "dense_grid"
    print(f"{token:10s} {geometry:14s} {spec.filename:14s} {spec.canonical_unit:8s} {spec.primary_variable}")
print("l4_sst and l4_ssh name different files, while the GLORYS tokens share one and differ only in variable.")
"""),
    md("""## 6. One named tile

The smallest retrieval unit is one asset for one date in one of the eight named
Core regions. Regions are resolved by name rather than by bounding-box query,
because the region set is fixed and immutable while `tacoreader`'s bbox
argument convention has changed across releases. Asking for a named tile is
therefore the most direct call available, and it returns the file's own
variables and dimensions with nothing merged or cropped."""),
    code("""
from ocean_taco.retrieve import load_tile_nc
date = queryset.dates[0][:10]
tile = load_tile_nc(catalog, date, "NORTH_ATLANTIC", "l4_sst", config=config)
print("date", date, "tile sizes", None if tile is None else dict(tile.sizes))
if tile is not None:
    print("variables", list(tile.data_vars))
"""),
    md("""## 7. Box retrieval and merge

A geographic box usually spans more than one region tile, so `load_bbox_nc`
resolves every intersecting tile, fetches each once, merges them on their
shared coordinates, and crops the result to the box. The returned field keeps
its **native coordinates**, since this call retrieves rather than renders, so
there is no target grid and no interpolation.

Three return values mean three different things. `None` means no asset matched
the request. An empty field means the asset existed and held nothing inside the
box. Invalid coordinates or dates raise `ValueError` rather than returning
something falsy."""),
    code("""
from ocean_taco import TimeRange
from ocean_taco.retrieve import load_bbox_nc, load_multisource_time_series_nc
sst = load_bbox_nc(catalog, date, box, "l4_sst", config=config)
print("box", box.to_dict())
print("sizes", None if sst is None else dict(sst.sizes))
if sst is not None:
    print("coordinates", {key: (round(float(sst[key].min()), 2), round(float(sst[key].max()), 2)) for key in ("lat", "lon")})
"""),
    code("""
if sst is None:
    print("No matching SST asset: no field is plotted.")
else:
    variable = next(name for name in sst.data_vars if sst[name].ndim >= 2)
    field = np.asarray(sst[variable]).squeeze()[::8, ::8]
    fig, axis = plt.subplots(figsize=(8, 4))
    image = axis.imshow(field, origin="lower", aspect="auto", cmap=CMAP, interpolation="nearest",
                        extent=(float(sst["lon"].min()), float(sst["lon"].max()),
                                float(sst["lat"].min()), float(sst["lat"].max())),
                        **color_limits("l4_sst"))
    axis.set(xlabel="longitude [°]", ylabel="latitude [°]", title=f"Retrieved box: {variable}")
    add_coastlines(axis)
    fig.colorbar(image, ax=axis, label=variable)
    fig.tight_layout()
    display_figure(fig)
    print("The image is a decimated display of the returned native-coordinate field.")
"""),
    md("""## 8. A multi-source closed time range

Requesting several sources over one time range returns a dict keyed by token,
each entry carrying that source's own time axis. Those axes differ, because the
products have different native temporal sampling and retrieval does not
reconcile them onto a common cadence. Reconciliation is a modelling decision,
so it belongs to the renderer and the context window rather than to the fetch.

The requested interval is **closed**, meaning both endpoints are included."""),
    code("""
window = TimeRange(queryset.dates[0], queryset.dates[1])
stack = load_multisource_time_series_nc(catalog, ("l4_sst", "l4_ssh", "l3_swot"), box, window, config=config)
for token, value in stack.items():
    steps = 0 if value is None else int(value.sizes.get("time", 1))
    print(f"{token:9s} time steps={steps:2d}  " + ("no asset matched" if value is None else f"variables={list(value.data_vars)[:4]}"))
print("Different step counts are expected. These products sample time differently.")
"""),
    md("""## 9. Points and antimeridian boxes

Two behaviours are worth stating explicitly, because each is a place where a
plausible-looking wrong answer is easy to produce.

**Points keep their own coordinates.** Argo returns float positions and their
measurements. Rasterising on retrieval would invent structure between floats
that nothing observed, so an empty point set is returned as a valid result
meaning the floats were not there.

**A box crossing the antimeridian becomes two segments.** A single interval
from 170 to −170 would be either empty or global depending on which comparison
ran first, so the geometry is represented explicitly as two rectangles and
every downstream test stays a simple interval comparison."""),
    code("""
argo = load_bbox_nc(catalog, date, box, "argo", config=config)
wrapped = GeoBox(170, -170, 10, 30, wraps_antimeridian=True)
print("Argo records in the box:", 0 if argo is None else next(iter(argo.sizes.values())))
print(f"wrapped request has {len(wrapped.segments())} explicit segments:")
for segment in wrapped.segments():
    print("   ", segment.to_dict())
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
if argo is not None and all(key in argo for key in ("lon", "lat")):
    axes[0].scatter(np.asarray(argo["lon"]), np.asarray(argo["lat"]), s=26, alpha=.75,
                    color="#dd8a45", edgecolor="white", linewidth=.4)
else:
    axes[0].text(.5, .5, "No Argo points in this selection", ha="center", va="center", transform=axes[0].transAxes)
axes[0].set(title="Native Argo point locations", xlabel="longitude [°]", ylabel="latitude [°]")
for segment in wrapped.segments():
    axes[1].fill_between([segment.lon_min, segment.lon_max], segment.lat_min, segment.lat_max,
                         alpha=.5, color="#dd8a45")
axes[1].axvline(180, color="#333333", linestyle=":", linewidth=1)
axes[1].axvline(-180, color="#333333", linestyle=":", linewidth=1)
axes[1].set(xlim=(-190, 190), ylim=(0, 35), xlabel="longitude [°]", ylabel="latitude [°]",
            title="A box crossing 180° is stored as two rectangles")
for axis in axes:
    add_coastlines(axis)
    axis.grid(alpha=.2)
fig.tight_layout()
display_figure(fig)
print("The two rectangles are what every downstream interval test sees.")
"""),
    code("""
for value in [tile, sst, argo, *stack.values()]:
    if value is not None and callable(close := getattr(value, "close", None)):
        close()
print("Closed opened datasets.")
"""),

])

write("ml_configuration_cookbook.ipynb", [
    md("""# ML renderer configuration reference

A renderer turns one source over one patch into an array, and the choice of
renderer sets the tensor structure everything downstream has to handle: whether
samples stack into a batch without padding, what appears when a source is
absent for a position and date, and how much of the native resolution
survives. Each configuration below states the structure it produces and what it
does with missing data.

The four renderers are `Resample`, which fixes the output grid, `Native`, which
keeps the source's own grid, `VectorPair`, which renders two components as one
field, and `Points`, which returns ragged records.

Every configuration runs against the same pinned catalog revision, the same
published eval QuerySet, and the same drawn row, so the differences between
them come from the rendering configuration alone.

This is a reference rather than a pipeline guide. For the end-to-end training
loader see the
[ML use cases notebook](spatio_temporal_query_generation.ipynb), and for
filters and coverage see the
[QuerySet and filter deep-dive](data_retrieval_workflows.ipynb)."""),
    code(SETUP), code(PLOTTING), code(LOAD),
    md("""## The row every configuration uses

Every configuration below renders the same drawn row. Holding the row fixed
means any difference in the printed output comes from the renderer rather than
from the data. The helpers below print the tensor structure a configuration
produces and draw a rendered source: shapes, masks, and whether the source was
available at all.

The draw is conditioned on recorded SWOT and SSH coverage, so the shared row
carries the sparse source alongside the dense analyses and every panel below
has data to show. The [deep-dive](data_retrieval_workflows.ipynb) explains what
those requirements read."""),
    code("""
from ocean_taco import CoverageRequirement, GeoBox, PatchSize, QueryFilter, draw_queryset, select_queryset
from ocean_taco.render import Native, Points, Resample, VectorPair
from ocean_taco.torch import OceanTACODataset

shared_draw = draw_queryset(queryset, requested_row_count=1, seed=29,
                            record_path=DRAW_DIR / "cookbook-draw.json",
                            query_filter=QueryFilter(box=GeoBox(-80, -30, 10, 45),
                                                     coverage=(CoverageRequirement("swot", "valid_fraction_ocean", 0.2),
                                                               CoverageRequirement("ssh", "valid_cells", 1))))
row = shared_draw.rows[0]
print(f"row: ({row['centre_lon']:.2f}, {row['centre_lat']:.2f}) on {row['anchor_time'][:10]}")

def render(sources):
    # Render the shared row under one configuration and report what came back.
    sample = OceanTACODataset(queries=shared_draw, sources=sources, catalog_config=config)[0]
    for token in sources:
        record, available = sample[token], sample["availability"][token]
        shape = tuple(np.asarray(record["data"]).shape)
        valid = int(np.asarray(record["valid_mask"]).sum())
        print(f"  {token:10s} data={str(shape):18s} available={str(available):5s} valid_cells={valid}")
    return sample

def show_grid(axis, record, title, component=None, key=None):
    # Draw a rendered source, or state its absence rather than inventing pixels.
    data = np.asarray(record["data"])
    if data.shape[0] == 0:
        axis.text(.5, .5, "structurally absent for\\nthis position and date", ha="center", va="center", transform=axis.transAxes)
        axis.set(title=f"{title}: shape {data.shape}", xticks=[], yticks=[])
        return
    image = data[0] if component is None else data[0][component]
    # The record carries the grid it was rendered onto, so the panel is drawn in
    # degrees and can take a coastline instead of counting pixels.
    extent = geographic_extent(record)
    drawn = axis.imshow(image, origin="lower", cmap=CMAP, interpolation="nearest",
                        aspect="auto", extent=extent, **color_limits(key))
    axis.set(title=f"{title}: shape {data.shape}", xlabel="longitude [°]", ylabel="latitude [°]")
    add_coastlines(axis)
    axis.figure.colorbar(drawn, ax=axis, shrink=.8)
"""),
    md("""## `Resample`: fixed grids and multimodal fusion

`Resample((H, W), support_threshold)` puts every source on one grid, so each
becomes `(T, H, W)` with the same `H` and `W`, channels concatenate, and
batches stack without padding. The cost is interpolation, paid by every source
whose native grid differs from the target.

Fusion is the same configuration with another source added. Adding `l3_swot`
to two complete L4 fields exercises the sparse case: a swath either crosses
this patch on this date or it does not, and the output reports which.

`support_threshold` decides what happens when several native cells fall into
one output cell. Support is the fraction of the output cell backed by valid
source data, and cells below the threshold are marked invalid rather than
filled with an interpolated guess.

The 64x64 target used throughout this notebook sits above the native grid of
the dense L4 sources, which are about 23x31 and 19x24 cells over a 256 km
patch, so `Resample` warns once per token that it is upsampling. The warnings
below are expected and are kept visible rather than suppressed. A uniform
comparison grid keeps these configurations readable side by side, while
`native_shape` in each payload records the resolution the data actually
carries."""),
    code("""
fixed = {"l4_sst": Resample((64, 64), .5), "l4_ssh": Resample((64, 64), .5)}
print("fixed grids:")
fixed_sample = render(fixed)
fusion = {**fixed, "l3_swot": Resample((64, 64), .5)}
print("with SWOT fused in:")
fusion_sample = render(fusion)
"""),
    code("""
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
for axis, token in zip(axes, ("l4_sst", "l4_ssh", "l3_swot")):
    show_grid(axis, fusion_sample[token], token, key=token)
display_figure(fig)
print("All three share one model-facing grid, and their masks stay separate.")
"""),
    md("""## `VectorPair`: two components as one field

Rendered as independent sources, an eastward and a northward velocity
component can disagree about where they are valid: a cell ends up with a valid
`u` and an invalid `v`, which yields a direction no measurement supports.

`VectorPair` renders both components as a unit. The result is `(T, 2, H, W)`
rather than two `(T, H, W)` entries, `valid_mask` covers cells where **both**
components have support, and `pair_available` is the sample-level Boolean. The
two components come from the same underlying asset, so they are also fetched
once rather than twice."""),
    code("""
vectors = {"velocity": VectorPair(Resample((64, 64), .5))}
print("components:", vectors["velocity"].components)
velocity_sample = render(vectors)
velocity = velocity_sample["velocity"]
data = np.asarray(velocity["data"])
if data.shape[0]:
    speed = np.hypot(data[0][0], data[0][1])
    print(f"u range=[{np.nanmin(data[:, 0]):.3f}, {np.nanmax(data[:, 0]):.3f}] m/s")
    print(f"v range=[{np.nanmin(data[:, 1]):.3f}, {np.nanmax(data[:, 1]):.3f}] m/s")
    print(f"speed max={np.nanmax(speed):.3f} m/s, pair_available={bool(velocity['pair_available'])}")
"""),
    code("""
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
# The components are signed, so they get a range symmetric about zero, which
# puts the colormap's white at no flow. Speed is a magnitude and starts at zero.
show_grid(axes[0], velocity, "eastward (u)", component=0, key="velocity")
show_grid(axes[1], velocity, "northward (v)", component=1, key="velocity")
data = np.asarray(velocity["data"])
if data.shape[0]:
    speed = np.hypot(data[0][0], data[0][1])
    extent = geographic_extent(velocity)
    drawn = axes[2].imshow(speed, origin="lower", cmap=CMAP, interpolation="nearest",
                           aspect="auto", extent=extent, **color_limits("speed"))
    # The arrows are placed on the same degree axes as the image, so the quiver
    # grid is subsampled from the record's own longitude and latitude vectors.
    step = max(1, speed.shape[0] // 16)
    grid_x, grid_y = np.meshgrid(np.asarray(velocity["lon"]).ravel()[::step],
                                 np.asarray(velocity["lat"]).ravel()[::step])
    axes[2].quiver(grid_x, grid_y, data[0][0][::step, ::step], data[0][1][::step, ::step], color="#222222", scale=6)
    axes[2].set(title="speed with direction", xlabel="longitude [°]", ylabel="latitude [°]")
    add_coastlines(axes[2])
    fig.colorbar(drawn, ax=axes[2], shrink=.8, label="m s⁻¹")
else:
    axes[2].text(.5, .5, "pair unavailable", ha="center", va="center", transform=axes[2].transAxes)
display_figure(fig)
print("One shared mask governs both components, so every drawn arrow has support in each.")
"""),
    md("""## Sparse and dense sources in one configuration

**A missing SWOT cell records that the satellite did not sample there**, which
is the distinction that mixing a complete L4 analysis with a sparse L3 swath
has to preserve. A value cannot carry it, since zero is a perfectly plausible
sea-level anomaly. The masks carry it instead, so a model consuming both
sources should read `valid_mask` rather than testing values against a
sentinel, and a loss should be masked by it."""),
    code("""
sparse_dense = {"l4_sst": Resample((64, 64), .5), "l3_swot": Resample((64, 64), .5)}
sparse_sample = render(sparse_dense)
for token in sparse_dense:
    mask = np.asarray(sparse_sample[token]["valid_mask"])
    if mask.size:
        print(f"{token:10s} valid fraction={mask.mean():.3f}")
print("A low valid fraction is evidence about sampling, not about the ocean.")
"""),
    md("""## `Points`: ragged Argo records

Argo measurements are float profiles at their own positions rather than a
field, so `Points` returns ragged records carrying coordinates and pressures
instead of a grid, and the count varies from row to row. Zero points is a valid
result and a valid batch member, because floats are sparse and most patches on
most days contain none.

`variable` selects which measurement to expose, and `pres_range` limits the
depth band in decibars. Omitting `pres_range` takes the shallowest usable level
per profile, while the configuration below asks for the top 200 dbar so a full
profile is visible.

The row rendered here comes from a draw conditioned on `profile_count`, which
is how to obtain a patch containing floats without naming coordinates by hand.
The survey cell after it measures how uncommon that is across an unconditioned
draw.

The left panel below carries a single marker because a float profiles where it
drifts, so all of its records share one position and differ in depth, which the
right panel resolves as a temperature profile. Across the unconditioned draw
surveyed above, one float in a 256 km patch is the ordinary count."""),
    code("""
points = {"argo": Points(variable="TEMP", pres_range=(0, 200))}
argo_draw = draw_queryset(queryset, requested_row_count=1, seed=29,
                          record_path=DRAW_DIR / "argo-draw.json",
                          query_filter=QueryFilter(box=GeoBox(-80, -30, 10, 45),
                                                   coverage=(CoverageRequirement("argo", "profile_count", 1),)))
argo_row = argo_draw.rows[0]
print(f"row: ({argo_row['centre_lon']:.2f}, {argo_row['centre_lat']:.2f}) on {argo_row['anchor_time'][:10]}")
argo_sample = OceanTACODataset(queries=argo_draw, sources=points, catalog_config=config)[0]
record = argo_sample["argo"]
print("available:", argo_sample["availability"]["argo"])
for key in ("data", "lat", "lon", "pres"):
    if key in record:
        print(f"  {key:6s} shape={tuple(np.asarray(record[key]).shape)}")
print(f"{int(np.asarray(record['data']).size)} point records. The count is per patch and varies.")
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
lon, lat = np.asarray(record.get("lon", [])), np.asarray(record.get("lat", []))
values = np.asarray(record["data"]).reshape(-1)
if lon.size:
    # Draw the patch footprint too. Every record here belongs to one float
    # profiling at a single location, so without the outline the panel would
    # read as a plotting failure rather than as genuine sparsity.
    lon_span, lat_span = PatchSize(PATCH_SIZE_KM, "km").to_degrees(centre_lat=argo_row["centre_lat"])
    axes[0].add_patch(plt.Rectangle((argo_row["centre_lon"] - lon_span / 2, argo_row["centre_lat"] - lat_span / 2),
                                    lon_span, lat_span, fill=False, edgecolor="#2474a6",
                                    linewidth=1.4, linestyle="--", label="patch footprint"))
    drawn = axes[0].scatter(lon, lat, c=values[:lon.size], s=90, cmap=CMAP, zorder=3,
                            edgecolor="black", linewidth=.5, label=f"{lon.size} records",
                            **color_limits("argo_temp"))
    fig.colorbar(drawn, ax=axes[0], label="TEMP [°C]")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set(xlim=(argo_row["centre_lon"] - lon_span * .65, argo_row["centre_lon"] + lon_span * .65),
                ylim=(argo_row["centre_lat"] - lat_span * .65, argo_row["centre_lat"] + lat_span * .65))
    add_coastlines(axes[0])
    pres = np.asarray(record["pres"]).reshape(-1)
    axes[1].scatter(values[:pres.size], pres, s=26, color="#dd8a45")
    axes[1].invert_yaxis()
    axes[1].set(title="The same records as a profile", xlabel="TEMP [°C]", ylabel="pressure [dbar]")
else:
    for axis in axes:
        axis.text(.5, .5, "no Argo profiles in this patch", ha="center", va="center", transform=axis.transAxes)
axes[0].set_title("Argo records inside the 256 km patch")
axes[0].set(xlabel="longitude [°]", ylabel="latitude [°]")
display_figure(fig)
print("Points keep their own coordinates and pressures, and nothing is rasterised onto a grid.")
"""),
    code("""
# How often does an unconditioned draw contain floats at all?
survey = draw_queryset(queryset, requested_row_count=20, seed=29,
                       record_path=DRAW_DIR / "argo-survey-draw.json")
survey_dataset = OceanTACODataset(queries=survey, sources=points, catalog_config=config)
counts = [int(np.asarray(survey_dataset[index]["argo"]["data"]).size) for index in range(len(survey.rows))]
print(f"rows with at least one profile: {sum(count > 0 for count in counts)} of {len(counts)}")
print("Zero is the ordinary case at this patch size, which is why the configuration above conditions its draw.")
"""),
    md("""## `Native`: the source grid, and bucketing

Where `Resample` pays interpolation for a uniform grid, `Native()` keeps the
source's own grid, so no cell is interpolated or invented. Shapes then vary
from row to row, and varying shapes cannot be stacked. `ShapeBucketSampler` handles that by grouping rows whose native
shapes already agree, so each batch is internally uniform without padding.

Pass `shuffle=False` for a reproducible bucket ordering, since it shuffles
within and across buckets by default and a printed bucket list would otherwise
depend on the seed. The
[ML use cases notebook](spatio_temporal_query_generation.ipynb) runs this
sampler against a real `DataLoader`."""),
    code("""
native_sample = render({"l3_swot": Native()})
resampled = np.asarray(fusion_sample["l3_swot"]["data"])
native = np.asarray(native_sample["l3_swot"]["data"])
print(f"native shape={native.shape}, resampled shape={resampled.shape}")
print("Native preserves the source grid exactly. Resample fixes the shape and pays interpolation for it.")
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
show_grid(axes[0], native_sample["l3_swot"], "Native", key="l3_swot")
show_grid(axes[1], fusion_sample["l3_swot"], "Resample (64, 64)", key="l3_swot")
display_figure(fig)
print("Same row and same colour range, so the difference between the panels is resolution alone.")
"""),
    code("""
from ocean_taco.torch import ShapeBucketSampler
shapes = [(32, 48), (32, 48), (40, 48), (40, 48), (32, 48)]
sampler = ShapeBucketSampler(shapes, batch_size=2, seed=19, shuffle=False)
batches = list(sampler)
print("native shapes:", shapes)
print("batches:", batches)
for batch in batches:
    print("  ", [shapes[index] for index in batch], "-> uniform:", len({shapes[index] for index in batch}) == 1)
"""),
    md("""## Regional and antimeridian boxes

A regional box resolves to a single segment. A box crossing the antimeridian
resolves to two, because a single interval from 170 to −170 would be either
empty or global depending on which comparison ran first. Splitting it keeps
every downstream test a simple interval comparison, and the
[deep-dive](data_retrieval_workflows.ipynb) plots the geometry."""),
    code("""
regional = GeoBox(-80, -30, 10, 45)
wrapped = GeoBox(170, -170, 10, 30, wraps_antimeridian=True)
print("regional segments:", len(regional.segments()))
for segment in wrapped.segments():
    print("  wrapped segment:", segment.to_dict())
print(f"regional selection: {select_queryset(queryset, QueryFilter(box=regional)).count:,} pairs")
"""),
    md("""## Normalisation

Normalisation belongs to the experiment, not to the loader, and has to be
recorded alongside it. The loader therefore returns decoded values in their
recorded units and normalises nothing. Apply it through `valid_mask` so that
invalid cells stay NaN rather than becoming a plausible-looking zero, and
compute the statistics once over the training rows rather than per batch.

`l3_swot` is the informative source here, since its swath covers only part of
the patch, so the mask genuinely excludes cells and the NaN behaviour is
visible. The comparison at the end comes out differently under zero-filling,
which treats every unobserved cell as a measured zero and so biases the mean
towards zero."""),
    code("""
import torch
def normalise_valid(data, mask, mean, std):
    output = torch.full_like(data, float("nan"))
    output[mask] = (data[mask] - mean) / torch.as_tensor(std, dtype=data.dtype).clamp_min(1e-6)
    return output

values, mask = fusion_sample["l3_swot"]["data"], fusion_sample["l3_swot"]["valid_mask"]
mean, std = values[mask].mean(), values[mask].std()
normalised = normalise_valid(values, mask, mean, std)
print(f"source mean={mean:.4f} std={std:.4f} over {int(mask.sum())} of {mask.numel()} cells")
print(f"normalised mean={normalised[mask].mean():.3e} std={normalised[mask].std():.3f}")
print(f"{int((~mask).sum())} invalid cells, all still NaN: {bool(torch.isnan(normalised[~mask]).all())}")
filled = torch.nan_to_num(values, nan=0.0)
print(f"masked mean={mean:.4f} against zero-filled mean={filled.mean():.4f}")
"""),
])

write("plot_hurricane_milton.ipynb", [
    md("""# Hurricane Milton: wind and SSH across products

The maintained helper produces the four-date, three-column projected figure:
L4 wind with vectors, L3 along-track SSH, and L3 SWOT. The dense SWOT product
is always requested; a missing source is shown as missing rather than hidden."""),
    code(SETUP),
    code("""
from ocean_taco import CatalogConfig
from ocean_taco.retrieve import load_hf_dataset
from ocean_taco.viz.paper.plot_hurricane_milton import DEFAULT_DATES, close_data, load_date, make_figure
config = CatalogConfig()
catalog = load_hf_dataset(config)
print(f"catalog={config.resolved_catalog_url}; revision={config.revision}; dates={DEFAULT_DATES}")
"""),
    md("## Retrieve every product and measure the local execution path"),
    code("""
from time import perf_counter
started = perf_counter()
rows = {date: load_date(catalog, date, config=config) for date in DEFAULT_DATES}
for date, products in rows.items(): print(date, {token: dict(data.sizes) for token, data in products.items()})
print(f"elapsed={perf_counter() - started:.1f}s; cache={config.cache_dir}")
if any("l3_swot" not in products for products in rows.values()): raise RuntimeError("L3 SWOT is required for this tutorial figure.")
"""),
    code("""
from IPython.display import display
figure = make_figure(rows, DEFAULT_DATES)
display_figure(figure)
close_data(rows)
import matplotlib.pyplot as plt
plt.close(figure)
print("Closed datasets and figure.")
"""),
])

write("plot_hurricane_milton_cross_product.ipynb", [
    md("""# Hurricane Milton: SSH cross-product comparison

This workflow overlays L4 DUACS, L3 along-track, and L3 SWOT in projected
geography, then compares each L3 product with L4 after interpolation to the
observation coordinates. It reports deterministic-subsample visualisation,
correlation, RMSE, and a 1:1 reference line."""),
    code(SETUP),
    code("""
from ocean_taco import CatalogConfig
from ocean_taco.retrieve import load_hf_dataset
from ocean_taco.viz.paper.plot_hurricane_milton_cross_product import close_products, load_products, make_figure
DATE = "2024-10-09"
config = CatalogConfig()
catalog = load_hf_dataset(config)
data = load_products(catalog, DATE, config=config)
print(f"date={DATE}; revision={config.revision}; products={list(data)}")
if set(data) != {"l4_ssh", "l3_ssh", "l3_swot"}: raise RuntimeError("This tutorial requires all three SSH products, including L3 SWOT.")
"""),
    code("""
from IPython.display import display
figure = make_figure(data, DATE)
display_figure(figure)
close_products(data)
import matplotlib.pyplot as plt
plt.close(figure)
print("Closed datasets and figure. The scatter uses no nearest-grid shortcut and never fills missing L4 values with zero.")
"""),
])

