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


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


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
    import matplotlib.pyplot as plt
    plt.close(fig)

PATCH_SIZE_KM = 256          # the published patch size
REQUESTED_ROWS = 4
SEED = 7
QUERYSET_REVISION = "d4d6eede189819347a760a87c4d047c8e5bafd52"
TRAINING_POSITION_COUNTS = {128: 35_000, 256: 8_800, 512: 2_170}
TRAINING_MAXIMUM_PAIR_IOU = 0.20

# Draw records are notebook output, not cached data; they land beside the notebook.
DRAW_DIR = Path("draws")
DRAW_DIR.mkdir(exist_ok=True)
"""

LOAD = """
from ocean_taco import CatalogConfig, QuerySet
from ocean_taco.retrieve import load_hf_dataset

config = CatalogConfig(revision=QUERYSET_REVISION)
catalog = load_hf_dataset(config)
queryset = QuerySet.from_hub(PATCH_SIZE_KM, "eval", revision=QUERYSET_REVISION)
print(f"queryset_id={queryset.queryset_id}; kind={queryset.header['kind']}; positions={len(queryset.positions)}; dates={len(queryset.dates)}")
"""

# Shared by the four rewritten tutorials only. The two Hurricane Milton
# notebooks reproduce paper figures and keep SETUP unchanged, so their
# generated source stays byte-identical to the reviewed version.
PLOTTING = """
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
from math import cos, radians

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
    # Measured over drawn rows across the whole tutorial box, which reaches 45N
    # and therefore spans winter mid-latitudes as well as the tropics. A range
    # of 20-28 was calibrated on subtropical rows alone and clipped half of
    # every draw to a single flat shade.
    "l4_sst": (2.0, 32.0),         # degrees celsius
    "l4_ssh": (-0.5, 0.5),         # metres
    "l3_swot": (-0.5, 0.5),        # metres
    "l3_ssh": (-0.5, 0.5),         # metres, nadir altimetry, same quantity as l3_swot
    "l4_sss": (32.0, 38.0),        # practical salinity units
    "velocity": (-0.8, 0.8),       # m/s, signed component, symmetric about zero
    "speed": (0.0, 0.8),           # m/s, a magnitude, so it starts at zero
    "argo_temp": (0.0, 30.0),      # degrees celsius
}

# The physical quantity each source carries, so a panel can name what it shows
# and a colourbar can be labelled rather than left as bare numbers.
UNITS = {
    "l4_sst": "degC",
    "l4_ssh": "m",
    "l3_swot": "m",
    "l3_ssh": "m",
    "l4_sss": "PSU",
    "velocity": "m/s",
    "speed": "m/s",
    "argo_temp": "degC",
}

QUANTITY = {
    "l4_sst": "sea surface temperature",
    "l4_ssh": "sea surface height anomaly",
    "l3_swot": "sea surface height anomaly",
    "l3_ssh": "sea surface height anomaly",
    "l4_sss": "sea surface salinity",
    "velocity": "current velocity component",
    "speed": "current speed",
    "argo_temp": "temperature",
}

def color_limits(key):
    # Fall back to autoscaling for anything without a declared range.
    low, high = COLOR_RANGE.get(key, (None, None))
    return {"vmin": low, "vmax": high}

def source_label(token, shape=None):
    # "l4_sst 45x51 [degC]" -- the variable, what it measures, and its units,
    # so no panel depends on the surrounding prose to say what it holds.
    unit = UNITS.get(token)
    parts = [token]
    if shape is not None:
        parts.append(f"{shape[0]}x{shape[1]}")
    return " ".join(parts) + (f" [{unit}]" if unit else "")

def colorbar_label(token):
    quantity, unit = QUANTITY.get(token), UNITS.get(token)
    if quantity and unit:
        return f"{quantity} [{unit}]"
    return QUANTITY.get(token) or (f"[{unit}]" if unit else "")

def add_colorbar(fig, drawn, axes, token, shrink=.7):
    # Every figure states its range rather than leaving the reader to infer it
    # from the colours, so two figures of the same field are comparable.
    bar = fig.colorbar(drawn, ax=axes, shrink=shrink, label=colorbar_label(token))
    low, high = COLOR_RANGE.get(token, (None, None))
    if low is not None:
        bar.set_ticks(np.linspace(low, high, 5))
    return bar

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

def map_aspect(extent):
    # Degrees of longitude are shorter than degrees of latitude away from the
    # equator, so a patch is not square in degrees. Passing this to imshow
    # keeps the panel and its coastlines in true proportion; aspect="auto"
    # would stretch both to fill whatever box the subplot happens to have.
    lon_min, lon_max, lat_min, lat_max = extent
    mid = radians((lat_min + lat_max) / 2)
    return 1.0 / max(cos(mid), 1e-6)

def batch_member(batch, token, index):
    # One sample's worth of a collated batch, in the shape the single-record
    # helpers above already accept. Collation stacks the grid vectors as
    # lat (B, H) and lon (B, W), so slicing the leading axis recovers the
    # record that geographic_extent and the imshow calls expect.
    stacked = batch[token]
    record = {key: stacked[key][index] for key in ("data", "valid_mask", "lat", "lon")
              if key in stacked}
    for key in ("source_valid", "support_mask", "spatial_padding_mask"):
        if key in stacked:
            record[key] = stacked[key][index]
    return record

def batch_panel_grid(batch, token, columns=4, day=0, title=None, panel_title=None,
                     figure_width=3.3, panel_height=3.0, show_coastlines=True):
    # A whole batch drawn as a grid of maps, one panel per batch member. Batch
    # size, not the caller, sets the number of panels, so a loader that returns
    # fewer rows than asked for still draws every row it did return.
    values = np.asarray(batch[token]["data"])
    count = values.shape[0]
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(figure_width * columns, panel_height * rows),
                             constrained_layout=True)
    flat = axes.ravel()
    limits = color_limits(token)
    available = batch.get("availability", {}).get(token, [True] * count)
    drawn = None
    for index in range(count):
        axis, record = flat[index], batch_member(batch, token, index)
        axis.set(xticks=[], yticks=[])
        axis.set_title(panel_title(batch, index) if panel_title else f"row {index}", fontsize=9)
        extent = geographic_extent(record)
        if not available[index] or not np.isfinite(extent).all():
            # A row whose source has no asset for its date is rendered with no
            # grid at all, so there is nothing to place on a map. Saying so in
            # the panel keeps the batch's own geometry, where panel N is row N.
            axis.text(.5, .5, "no data\\nfor this row", ha="center", va="center",
                      fontsize=9, color="#777777", transform=axis.transAxes)
            continue
        field = np.asarray(record["data"])[day]
        # Cells the renderer could not fill are left blank rather than drawn as
        # a value, so a sparse source reads as sparse.
        shown = np.where(np.asarray(record["valid_mask"])[day], field, np.nan)
        drawn = axis.imshow(shown, origin="lower", cmap=CMAP, interpolation="nearest",
                            aspect=map_aspect(extent), extent=extent, **limits)
        if show_coastlines:
            add_coastlines(axis)
    for axis in flat[count:]:
        axis.axis("off")
    if drawn is not None:
        add_colorbar(fig, drawn, axes.ravel().tolist(), token, shrink=.6)
    heading = f"{title} -- {source_label(token)}" if title else source_label(token)
    fig.suptitle(heading, fontsize=11)
    return fig

def batch_dates(batch):
    # The context start date of every member of a collated batch. Collation
    # keeps the per-sample query objects in a list, so the dates survive
    # batching and a panel can say which day it is showing.
    return [f"{spec.context.start:%Y-%m-%d}" for spec in batch["query"]]
"""


def write(name, cells):
    if name == "ml_dataset.ipynb":
        for cell in cells:
            if cell.cell_type == "code":
                cell.source = "\n".join(
                    line for line in cell.source.splitlines()
                    if not line.lstrip().startswith("#")
                )
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
    )
    nbf.write(notebook, OUT / name)


write(
    "ml_dataset.ipynb",
    [
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

The four constants below select the patch size, draw size, seed, and immutable
dataset revision. `CatalogConfig()` and `QuerySet.from_hub` use that same revision,
so anyone can fetch the eval and training QuerySets without a local checkout.
Loading verifies every table against the checksums in its header, so a successful
load is also an integrity check."""),
        code(SETUP),
        code(PLOTTING),
        code(LOAD),
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
positions were placed. Training positions are a seeded, stratified best-candidate
random sample of eligible ocean-mask cells. An 18 × 36 equal-area global
stratification assigns each ocean region a share of the training density; candidate
centres use cosine-latitude area weights. Every stored centre is ocean and its full
patch remains within the mask domain, while coastal context remains eligible. For
each selection the sampler considers a pool of 16 candidates and keeps the best
acceptable one; no retained footprint overlaps a selected footprint by more than
0.20 IoU. The target counts are 35,000 positions at 128 km, 8,800 at 256 km, and
2,170 at 512 km.
Eval positions sit on a systematic grid, which makes coverage uniform so that a
metric averaged over them is a spatial average rather than one weighted by
wherever a sampler happened to concentrate. The
`kind` field records which of the two a QuerySet is, taking the value
`"training"` or `"eval"`.

At global scale the two are hard to tell apart, since 8800 training and 6027
eval positions both cover the whole ocean. The difference lies in local
spacing, so the second row of the figure below draws a few degrees of the North
Atlantic box, with each stored position expanded to the 256 km footprint the
query actually covers. Drawing footprints rather than centres shows how much
ocean one query spans and where neighbouring queries overlap, neither of which
a dimensionless point can show.

The training centres do not follow latitude rows or a longitude lattice. Their
seed, candidate-pool size, and hard IoU ceiling are stored in the QuerySet header,
so the random sample is reproducible. Eval centres retain their regular physical
spacing.
The cell below reports the local counts without claiming a shared row
structure.

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
training = QuerySet.from_hub(PATCH_SIZE_KM, "training", revision=QUERYSET_REVISION)
training_sampling = training.header["position_sampling"]
assert len(training.positions) == TRAINING_POSITION_COUNTS[PATCH_SIZE_KM]
assert training_sampling["method"] == "stratified_best_candidate_ocean/v1"
assert training_sampling["maximum_pair_iou"] == TRAINING_MAXIMUM_PAIR_IOU
evaluation = queryset
split_box = GeoBox(-80, -30, 10, 45)
train_rows = select_queryset(training, QueryFilter(box=split_box))
eval_rows = select_queryset(evaluation, QueryFilter(box=split_box))
print(f"training kind={training.header['kind']}, positions={len(training.positions)}, candidates in box={train_rows.count}, sampler={training_sampling['method']}, max IoU={training_sampling['maximum_pair_iou']:.2f}, id={training.queryset_id[:12]}")
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
print("Training centres use seeded stratified best-candidate sampling with a 0.20 IoU cap; eval centres follow a systematic grid.")
print("A kind records how positions were placed. It does not separate the two sets in space or time.")
"""),
        code("""
# Top row: the whole ocean, where the two placements are hard to tell apart.
# Bottom row: query footprints in a few degrees of the North Atlantic, drawn at
# the size the patch actually covers rather than as dimensionless centres.
FOOTPRINT_BOX = GeoBox(-60, -52, 28, 34)
patch = PatchSize(PATCH_SIZE_KM, "km")
fig, axes = plt.subplots(2, 2, figsize=(11, 7.8))
sets = (("training set (best-candidate, IoU ≤ 0.20)", training, "#2474a6"),
        ("eval set (systematic)", evaluation, "#238b45"))
for column, (label, source, colour) in enumerate(sets):
    lons = np.array([p["centre_lon"] for p in source.positions])
    lats = np.array([p["centre_lat"] for p in source.positions])
    # Marker size and opacity suit the global panels, which carry thousands of
    # points: one setting for both rows would smear these into a solid block.
    axes[0, column].scatter(lons, lats, s=.6, alpha=.35, color=colour, linewidths=0, rasterized=True)
    axes[0, column].add_patch(plt.Rectangle((FOOTPRINT_BOX.lon_min, FOOTPRINT_BOX.lat_min),
                                            FOOTPRINT_BOX.lon_max - FOOTPRINT_BOX.lon_min,
                                            FOOTPRINT_BOX.lat_max - FOOTPRINT_BOX.lat_min,
                                            fill=False, edgecolor="#d95f02", linewidth=1.6))
    axes[0, column].set(title=f"{label}: {len(lons)} positions", xlabel="longitude [°]",
                        ylabel="latitude [°]", xlim=(-180, 180), ylim=(-90, 90))
    add_coastlines(axes[0, column])

    inside = ((lons >= FOOTPRINT_BOX.lon_min) & (lons <= FOOTPRINT_BOX.lon_max)
              & (lats >= FOOTPRINT_BOX.lat_min) & (lats <= FOOTPRINT_BOX.lat_max))
    # footprint() is the same call the library uses to turn a centre into the
    # box a query covers, so these rectangles are the queries themselves.
    footprints = [patch.footprint(lon, lat) for lon, lat in zip(lons[inside], lats[inside])]
    latitude_levels = len(np.unique(np.round(lats[inside], 3)))
    for box in footprints:
        axes[1, column].add_patch(plt.Rectangle((box.lon_min, box.lat_min),
                                                box.lon_max - box.lon_min,
                                                box.lat_max - box.lat_min,
                                                fill=False, edgecolor=colour, linewidth=1.1, alpha=.55))
    axes[1, column].scatter(lons[inside], lats[inside], s=7, color=colour, zorder=3)
    # Limits come from the footprints rather than from the selection box, so no
    # rectangle is cut off at the edge and every overlap stays visible.
    axes[1, column].set(title=f"{len(footprints)} footprints; {latitude_levels} latitude levels",
                        xlabel="longitude [°]", ylabel="latitude [°]",
                        xlim=(min(b.lon_min for b in footprints) - .4, max(b.lon_max for b in footprints) + .4),
                        ylim=(min(b.lat_min for b in footprints) - .4, max(b.lat_max for b in footprints) + .4))
    add_coastlines(axes[1, column])
for axis in axes.ravel():
    axis.grid(alpha=.2)
fig.suptitle("Published positions, and the area the queries at those positions cover")
fig.tight_layout()
display_figure(fig)
print(f"global outline and footprint zoom: lon {FOOTPRINT_BOX.lon_min} to {FOOTPRINT_BOX.lon_max}, lat {FOOTPRINT_BOX.lat_min} to {FOOTPRINT_BOX.lat_max}")
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

`dataset[i]` returns a flat dictionary keyed by source token, a `query` entry
containing the source `PatchSpec`, and one Boolean per source in
`availability`. Each source dictionary contains:

| Key | Shape | Meaning |
|-----|-------|---------|
| `data` | `(T, H, W)` | decoded values, NaN where invalid |
| `source_valid` | `(T, H, W)` | positive source support on the output grid |
| `support_mask` | `(T, H, W)` | cells whose support met the threshold |
| `valid_mask` | `(T, H, W)` | source support, threshold, and ocean mask combined |
| `support` | `(T, H, W)` | fraction of the output cell backed by source data |
| `lat`, `lon` | `(H,)`, `(W,)` | geographic coordinates of the rendered grid |
| `times` | list | the source timestamps that went into `T` |

For resampled scalars and vectors, `source_valid` is `support > 0` on the
output grid. `support_mask` is true where support meets the configured
threshold. `valid_mask` combines both with the ocean mask when present; use
it for inputs and losses. With `Native()`, `source_valid` stays on the native
grid. These masks describe support, not individual instrument quality flags.

**Structural absence.** If no asset exists for a position and date, the renderer
returns `empty()`: `data` has shape `(0, H, W)`, masks are zero, coordinates
are NaN, and `availability[token] = False`. The key and batch layout remain
present. Detect absence through the leading zero or `availability`, rather
than through the values; zero can be a valid SST anomaly.

The render below requests 128×128 cells. `l4_sst` is about 23×23 natively over
a 256 km patch, so this example upsamples for display. The library emits a
warning. `native_shape` remains in the payload and records the raw shape."""),
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

`collate_ocean_samples` is the `DataLoader` collate function. It collates
availability separately from values. An absent source leaves tensor layout and
batch size unchanged; downstream code reads `batch["availability"]`.

**Fixed and native shapes.** `Resample` outputs stack directly because every
sample shares a grid. `Native()` outputs can differ by row. Stack them by
padding or by grouping samples with matching shapes; `ShapeBucketSampler`
performs the grouping.

**Workers and source access.** Keep using PyTorch `DataLoader`.
`CoreSourceLoader` resolves catalog assets in the parent during dataset
construction; `PlannedSourceLoader` reads those assets in workers. PyTorch
handles batch scheduling, shuffling, and prefetching. Pass
`worker_init_fn=seed_ocean_taco_worker` when `num_workers > 0`.

**Automatic batch reuse.** PyTorch calls the dataset's `__getitems__` method
for a batch. The built-in planned loader groups shared assets and variables
and reuses daily crops across overlapping context/target windows, while
returning samples in the requested order. No manual batching call is needed.
Crop caches are cleared after each batch; custom source loaders keep their
ordered per-item calls. ML reads project requested dense variables, while Argo
retains its point fields.

**File handles.** `CatalogConfig(max_open_files=16)` sets the default limit
per source-loader cache in each process. Handles are reused even without
`cache_dir`; local assets are opened in place. Fork/spawn workers start with
fresh handles. `dataset.source_loader.close()` releases handles in the calling
process; worker handles belong to the workers.

For repeated training epochs, construct one DataLoader and reuse it:

```python
from torch.utils.data import DataLoader
from ocean_taco.torch import collate_ocean_samples, seed_ocean_taco_worker

training_loader = DataLoader(
    dataset, batch_size=2, num_workers=2,
    collate_fn=collate_ocean_samples,
    worker_init_fn=seed_ocean_taco_worker,
    persistent_workers=True, prefetch_factor=2,
)
```

Iterate `training_loader` for each epoch to retain worker file caches. When
using `num_workers=0`, omit `prefetch_factor` and leave `persistent_workers`
false. The single-batch example below uses zero workers for easy inspection.
See the [training notebook](spatio_temporal_query_generation.ipynb) for
workload examples; worker and prefetch settings should be measured on your data.

**No implicit normalisation.** The loader returns decoded values in their
recorded units without centring, scaling, or filling them. Section 7 covers
normalisation."""),
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
    ],
)

write(
    "spatio_temporal_query_generation.ipynb",
    [
        md("""# ML use cases and the training loader

A machine learning setup on ocean data is mostly a statement about time and
resolution. A forecasting model needs its target strictly after its context, an
assimilation model needs the target in the middle of the window, and a
super-resolution model needs the same patch from instruments of different
resolution. Each is a different query, and this notebook builds four of them
against the published QuerySet, then feeds each into a `DataLoader` that runs.

The four query shapes are:

1. **Forecasting**, where a context window is followed by a target at a lead
   time, with a check that the two windows do not overlap.
2. **Midpoint retrieval**, where the target sits at the centre of a symmetric
   window, which is the shape assimilation and interpolation setups need.
3. **Super-resolution**, where a coarse observation and two dense analyses
   supply the inputs and a finer instrument over the same patch supplies the
   target.
4. **Multi-source**, where every drawn row is required to carry a sparse
   observation alongside the dense analyses.

The last two sections build the training pipeline: batching under fixed and
native shapes, normalisation statistics computed once, and a training step
whose loss is masked.

**Every section draws rows that actually carry the data it renders.** Coverage
requirements do that work at selection time, so no figure below has a hole in
it. The [QuerySet and filter deep-dive](data_retrieval_workflows.ipynb) covers
`CoverageRequirement` and the null-versus-zero distinction it rests on. For the
concepts behind QuerySets, draws, and renderers, start with
[From a published QuerySet to a rendered sample](ml_dataset.ipynb)."""),
        code(SETUP),
        code(PLOTTING),
        code(LOAD),
        md("""## Why this notebook works at 512 km

Patch size decides how much ocean a sample contains, and it interacts with the
render shape. A 256 km patch at 31°N spans about 2.7° of longitude, which is
roughly 27 cells of an 0.1° L4 analysis. Asking `Resample` for a 32x32 output
from that grid is asking for more cells than the instrument measured, and
`Resample` switches to bilinear interpolation when a target side exceeds
native. The result looks smooth because it *is* smooth: the extra detail is
interpolation, not ocean.

A 512 km patch spans about 5.4°, and the same sources arrive at 45x51 or larger.
That is enough to render at or below native everywhere, so every panel in this
notebook shows measured cells rather than interpolated ones. It is also enough
area to show a front or an eddy rather than a single gradient.

The published sizes are 128, 256 and 512 km, so 512 is the largest patch
available and the one used throughout below."""),
        code("""
SUPER_PATCH_KM = 512
queryset = QuerySet.from_hub(SUPER_PATCH_KM, "eval")
print(f"queryset_id={queryset.queryset_id[:16]}...; kind={queryset.header['kind']}; "
      f"positions={len(queryset.positions)}; dates={len(queryset.dates)}")
"""),
        md("""## 1. Forecasting: context now, target later

A forecasting setup needs the context window and the target to be separate
objects, so that a model consuming context cannot reach the target by accident.
`QueryFilter` states the relation and the lead time, and the two datasets then
differ only in their context offsets.

`draw_queryset` builds every row from a single `QueryFilter`, so all rows in a
draw share its offsets. To get the target rows, copy the drawn rows and point
them at the target day. `OceanTACODataset` accepts a plain sequence of row
mappings, so those copies are a valid dataset input.

Two details make the copies honest. The lead comes from the filter rather than
being retyped at the call site, so the two cannot drift apart. And the copies
clear `relation` and `target_lead_days`, because a target row is not itself a
forecast anchor — it is the day being predicted. The library renders the
context window of whatever row it is given; constructing the target is the
caller's job.

Draw once and derive both datasets from that single draw. Drawing twice would
change which rows are eligible and silently compare different anchors."""),
        code("""
from ocean_taco import (CoverageRequirement, GeoBox, QueryFilter, draw_queryset,
                        replay_experiment, select_queryset)
from ocean_taco.render import Native, Resample
from ocean_taco.torch import OceanTACODataset

BOX = GeoBox(-80, -30, 10, 45)
BATCH_SIZE = 4

# At 512 km the L4 analyses arrive at 45x51 or larger, so a 40x40 output stays
# at or below native for every drawn row and no cell is interpolated.
SST = {"l4_sst": Resample((40, 40), support_threshold=0.5)}

def target_rows(draw, query_filter):
    # The target day, derived from the filter's own lead so the two cannot
    # disagree. The copies are plain same-time rows: a target is the day being
    # predicted, not another forecast anchor.
    lead = query_filter.target_lead_days
    return [{**row, "context_start_offset_days": lead, "context_end_offset_days": lead,
             "relation": "same_time", "target_lead_days": 0}
            for row in draw.rows]

def window_of(dataset):
    context = dataset[0]["query"].context
    return f"{context.start:%Y-%m-%d} .. {context.end:%Y-%m-%d}"

forecast_filter = QueryFilter(box=BOX, relation="forecast", target_lead_days=1,
                              context_start_offset_days=-1, context_end_offset_days=0)
forecast_draw = draw_queryset(queryset, requested_row_count=BATCH_SIZE, seed=29,
                              record_path=DRAW_DIR / "forecast-draw.json",
                              query_filter=forecast_filter)
lead = forecast_filter.target_lead_days
context_set = OceanTACODataset(queries=forecast_draw, sources=SST, catalog_config=config)
target_set = OceanTACODataset(queries=target_rows(forecast_draw, forecast_filter),
                              sources=SST, catalog_config=config)
print(f"anchor:  {forecast_draw.rows[0]['anchor_time'][:10]}")
print(f"context: {window_of(context_set)}")
print(f"target:  {window_of(target_set)}  (lead {lead} day)")
print(f"windows disjoint: {target_set[0]['query'].context.start > context_set[0]['query'].context.end}")
print(f"draw record replays: {replay_experiment(queryset, DRAW_DIR / 'forecast-draw.json').rows == forecast_draw.rows}")
"""),
        md("""### The batch this query shape produces

The offsets above retrieve data from OceanTACO into tensors with dates and shapes once a `DataLoader`
renders them, so for illustrative purposes this section and every folloing one is building towards creating a single
batch.

For batching, we need to account for the various different resolutions that yield different pixel sizes when retrieving a geographical extent. `collate_ocean_samples` is
passed as `collate_fn` because the default PyTorch collation cannot stack these
samples, and `seed_ocean_taco_worker` is passed as `worker_init_fn` because
worker processes otherwise inherit one seed. §4 returns to both, along with
batching sources whose shape varies between rows.

In the figure below, the first two columns are the context day and the target
day, and the third is the difference between them.

These calls already use the optimized path: PyTorch automatically requests
batches through `OceanTACODataset.__getitems__`, and the planned source loader
shares asset reads and overlapping context/target daily crops within a batch.
The file cache defaults to `CatalogConfig(max_open_files=16)` per loader and
process, independently of `cache_dir`.

`one_batch` below creates a loader for a single illustration. For repeated
training, construct one loader outside the epoch loop and reuse it with
`persistent_workers=True` and, for example, `prefetch_factor=2` when workers
are enabled. With zero workers, omit prefetch and keep persistence false.
See the [overview's worker example](ml_dataset.ipynb). Batch crop caches are
always cleared after the batch; persistent workers retain only their bounded
file caches.
"""),
        code("""
import torch
from torch.utils.data import DataLoader
from ocean_taco.torch import (ShapeBucketSampler, collate_ocean_samples,
                              native_pad_collate, seed_ocean_taco_worker)

def one_batch(dataset, batch_size=BATCH_SIZE, collate=collate_ocean_samples):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2,
                        collate_fn=collate, worker_init_fn=seed_ocean_taco_worker)
    return next(iter(loader))

def require_complete(batch, tokens, label):
    # A hole in a use-case figure means the draw was wrong, not that the figure
    # should apologise for it. Fail here rather than publish an empty panel.
    for token in tokens:
        available = batch["availability"][token]
        if not all(available):
            raise AssertionError(f"{label}: {token} missing in {available.count(False)} rows")
    print(f"{label}: all {len(batch['availability'][tokens[0]])} rows carry {', '.join(tokens)}")

context_batch = one_batch(context_set)
target_batch = one_batch(target_set)
require_complete(context_batch, ["l4_sst"], "context")
require_complete(target_batch, ["l4_sst"], "target")
print(f"context batch: {tuple(context_batch['l4_sst']['data'].shape)}  (batch, days, height, width)")
print(f"target batch:  {tuple(target_batch['l4_sst']['data'].shape)}")
"""),
        code("""
count = context_batch["l4_sst"]["data"].shape[0]
fig, axes = plt.subplots(count, 3, figsize=(11.5, 3.1 * count), constrained_layout=True)
DIFF_LIMIT = 1.5
for row in range(count):
    context_record = batch_member(context_batch, "l4_sst", row)
    target_record = batch_member(target_batch, "l4_sst", row)
    extent = geographic_extent(context_record)
    context_field = np.where(np.asarray(context_record["valid_mask"])[-1],
                             np.asarray(context_record["data"])[-1], np.nan)
    target_field = np.where(np.asarray(target_record["valid_mask"])[0],
                            np.asarray(target_record["data"])[0], np.nan)
    # Each row scales to its own patch. These panels compare two days of one
    # position, and a range wide enough for the whole box -- 45N in winter to
    # the tropics in summer -- renders any single patch as one flat shade.
    pair = np.concatenate([context_field[np.isfinite(context_field)].ravel(),
                           target_field[np.isfinite(target_field)].ravel()])
    row_limits = {"vmin": float(pair.min()), "vmax": float(pair.max())}
    for column, (field, limits, cmap) in enumerate((
            (context_field, row_limits, CMAP),
            (target_field, row_limits, CMAP),
            (target_field - context_field, {"vmin": -DIFF_LIMIT, "vmax": DIFF_LIMIT}, "PuOr_r"))):
        axis = axes[row, column]
        drawn = axis.imshow(field, origin="lower", cmap=cmap, interpolation="nearest",
                            aspect=map_aspect(extent), extent=extent, **limits)
        add_coastlines(axis)
        axis.set(xticks=[], yticks=[])
        if row == 0:
            axis.set_title(("context (last day)", f"target (+{lead} day)",
                            "target - context")[column], fontsize=10)
        if column == 1:
            fig.colorbar(drawn, ax=axis, shrink=.85, label=colorbar_label("l4_sst"))
        elif column == 2:
            fig.colorbar(drawn, ax=axis, shrink=.85, label=f"change [{UNITS['l4_sst']}]")
        else:
            axis.set_ylabel(f"{batch_dates(context_batch)[row]}", fontsize=9)
fig.suptitle(f"Forecasting at {SUPER_PATCH_KM} km: {source_label('l4_sst', (40, 40))}, "
             f"{lead} day lead", fontsize=12)
display_figure(fig)
print("Each row carries its own scale, so a 27 degC tropical patch and a 3-19 degC winter front")
print("are both legible. The third column is the change a forecast model has to predict.")
"""),
        md("""## 2. Midpoint retrieval for assimilation and interpolation

Forecasting places the target after the context. Assimilation and
interpolation setups place it inside: the model sees a span of days on both
sides and predicts the state in the middle. That reordering is expressed
entirely through the offsets passed to the same filter, so it needs no library
change.

`QueryFilter.relation` takes `"same_time"` or `"forecast"`, where `"forecast"`
requires a positive `target_lead_days` and `"same_time"` requires zero. The
relation alone therefore expresses only "target simultaneous with the anchor"
or "target strictly in the future". The context offsets carry the rest: they
are signed integers with only `context_end_offset_days >= context_start_offset_days`
enforced, so a symmetric window like `(-2, +2)` is legal.

Use `relation="same_time"` with a symmetric window, then point the target rows
at `(0, 0)`, the anchor and therefore the midpoint of the span. The difference
between this section and the previous one is exactly which offsets the target
rows carry.

Anchors near the ends of the record are rejected rather than silently
truncated, because the filter requires every date across the window to exist in
the QuerySet. A wider window therefore leaves fewer eligible rows."""),
        code("""
HALF_WINDOW = 2
midpoint_filter = QueryFilter(box=BOX, relation="same_time",
                              context_start_offset_days=-HALF_WINDOW,
                              context_end_offset_days=HALF_WINDOW)
print(f"eligible pairs with a +/-{HALF_WINDOW} day window: {select_queryset(queryset, midpoint_filter).count:,d}")
print(f"eligible pairs with no window at all:      {select_queryset(queryset, QueryFilter(box=BOX)).count:,d}")

midpoint_draw = draw_queryset(queryset, requested_row_count=BATCH_SIZE, seed=11,
                              record_path=DRAW_DIR / "midpoint-draw.json",
                              query_filter=midpoint_filter)
window_set = OceanTACODataset(queries=midpoint_draw, sources=SST, catalog_config=config)
centre_set = OceanTACODataset(queries=target_rows(midpoint_draw, midpoint_filter),
                              sources=SST, catalog_config=config)
print(f"anchor:  {midpoint_draw.rows[0]['anchor_time'][:10]}")
print(f"context: {window_of(window_set)}  -> data {tuple(window_set[0]['l4_sst']['data'].shape)}")
print(f"target:  {window_of(centre_set)}  -> data {tuple(centre_set[0]['l4_sst']['data'].shape)}")
print("The target sits at the centre of the context span, with equal numbers of days on each side.")
"""),
        code("""
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
        md("""### The batch this query shape produces

The schematic above is the intent; the batch is the result. It differs from
§1's in one visible way: the context tensor carries five days per row instead
of two, because the window spans two days either side of the anchor.

The figure holds one batch member fixed and steps across its five context days,
which is the axis this query shape adds. It scales to that patch's own range
rather than the shared one, because it compares days within a single patch and
the shared range would render five near-identical blocks. The member shown is
the one with the most spatial structure, since a patch of uniform open ocean
would show five identical panels whatever happened between them."""),
        code("""
window_batch = one_batch(window_set)
centre_batch = one_batch(centre_set)
require_complete(window_batch, ["l4_sst"], "context window")
require_complete(centre_batch, ["l4_sst"], "midpoint target")
days = window_batch["l4_sst"]["data"].shape[1]
print(f"context batch: {tuple(window_batch['l4_sst']['data'].shape)}  ({days} days per row)")
print(f"target batch:  {tuple(centre_batch['l4_sst']['data'].shape)}  (the middle day alone)")

spread = [float(np.nanstd(np.asarray(window_batch["l4_sst"]["data"])[index]))
          for index in range(window_batch["l4_sst"]["data"].shape[0])]
member = int(np.nanargmax(spread))
record = batch_member(window_batch, "l4_sst", member)
values = np.asarray(record["data"])
extent = geographic_extent(record)
low, high = float(np.nanmin(values)), float(np.nanmax(values))

fig, axes = plt.subplots(1, days, figsize=(3.3 * days, 3.4), constrained_layout=True)
for axis, day, offset in zip(axes, range(days), range(-HALF_WINDOW, HALF_WINDOW + 1)):
    axis.set(title=f"anchor {offset:+d} d" if offset else "anchor (target day)", xticks=[], yticks=[])
    valid = np.asarray(record["valid_mask"])[day]
    drawn = axis.imshow(np.where(valid, values[day], np.nan), origin="lower", cmap=CMAP,
                        interpolation="nearest", aspect=map_aspect(extent), extent=extent,
                        vmin=low, vmax=high)
    add_coastlines(axis)
fig.colorbar(drawn, ax=axes.tolist(), shrink=.7, label=colorbar_label("l4_sst"))
fig.suptitle(f"Row {member} across its +/-{HALF_WINDOW} day context window: "
             f"{source_label('l4_sst', (40, 40))}, scaled to this patch ({low:.1f} to {high:.1f})")
display_figure(fig)
print("The window supplies days on both sides of the target, so a model here interpolates in time")
print("rather than extrapolating forward as it does in section 1.")
"""),
        md("""## 3. Super-resolution: coarse inputs and a finer target

Super-resolution needs two views of the same patch at two resolutions. The
useful pair is not one array resampled twice — that manufactures the coarse
version by throwing away detail, and a model trained on it learns to undo a
known downsampling rather than to recover real structure.

This catalog carries a genuine pair. `l3_ssh` is nadir altimetry, a narrow
track sampled densely along its length, arriving at roughly 73x60 over a 512 km
box. `l3_swot` is the SWOT swath, arriving at roughly 256x210 over the same
box. That is a real difference in instrument resolution, and it is the one a
super-resolution model exists to bridge.

The nadir track alone is a thin input. Operational super-resolution for sea
surface height conditions on more: `l4_sst` and `l4_ssh` are gridded analyses
that cover the whole patch every day, and both carry information about the
structure the swath resolves. Sea surface temperature fronts sit where dynamic
height gradients sit, and the L4 height analysis is the coarse field the swath
refines. So the input side of this section is three sources — one sparse
observation and two dense analyses — and the target is the single fine swath.

Each input keeps its own grid rather than sharing one. Forcing them onto a
common shape would mean resampling every input down to the coarsest of them,
discarding nadir resolution to match the L4 height grid. Encoders that take
several inputs at several scales are the normal architecture here, so the
query shape should preserve what each instrument actually resolved."""),
        code("""
# Both instruments present on every drawn row. aggregate="min" applies the
# requirement to each context day rather than to their sum, so a row cannot
# qualify on one good day. The nadir threshold is low because a track covers a
# line: requiring 0.1 of the box from it selects nothing at all.
sr_filter = QueryFilter(box=BOX, coverage=(
    CoverageRequirement("swot", "valid_fraction_ocean", 0.2, aggregate="min"),
    CoverageRequirement("ssh", "valid_fraction_ocean", 0.02, aggregate="min")))
print(f"rows carrying both instruments every day: {select_queryset(queryset, sr_filter).count:,d}")

sr_draw = draw_queryset(queryset, requested_row_count=BATCH_SIZE, seed=5,
                        record_path=DRAW_DIR / "superres-draw.json", query_filter=sr_filter)
SR_INPUTS = ("l3_ssh", "l4_sst", "l4_ssh")
SR_TARGET = "l3_swot"
SR_TOKENS = SR_INPUTS + (SR_TARGET,)
native_set = OceanTACODataset(queries=sr_draw, sources={token: Native() for token in SR_TOKENS},
                              catalog_config=config)
native_rows = [native_set[index] for index in range(len(native_set))]
print()
print(f"{'source':10s} {'role':8s} {'native shape':>16s}   {'cells':>8s}   degrees per cell")
for token in SR_TOKENS:
    record = native_rows[0][token]
    height, width = np.asarray(record["data"]).shape[-2:]
    lon, lat = np.asarray(record["lon"]).ravel(), np.asarray(record["lat"]).ravel()
    step_lon = abs(float(lon[-1] - lon[0])) / max(width - 1, 1)
    step_lat = abs(float(lat[-1] - lat[0])) / max(height - 1, 1)
    role = "target" if token == SR_TARGET else "input"
    print(f"{token:10s} {role:8s} {f'{height} x {width}':>16s}   {height * width:>8,d}   "
          f"{step_lat:.3f} lat, {step_lon:.3f} lon")
"""),
        md("""### Choosing the shapes

Super-resolution architectures built on pixel shuffle or stacked strided
convolutions need the input and target shapes in an integer ratio, usually a
power of two, because each stage doubles one axis. `Resample` takes the output
shape directly, so the factor is whatever the caller makes it, and the
calculation worth writing down satisfies two requirements at once.

The first is the power of two, taken here against the nadir input, which is the
observation the target refines. The second is that **every** shape stays at or
below its own source's native grid, since resampling above native invents
detail the instrument never resolved. That is why each shape binds against its
own source rather than all of them against the smallest: binding everything
against the coarsest input would discard exactly the resolution advantage the
nadir track and the swath have.

Support thresholds differ per source because the sampling geometry does. A
threshold of 0.5 asks that half an output cell be covered before it is filled,
which suits a wide swath and the gridded analyses. A nadir track covers a line
rather than an area, so no cell of any useful size reaches half, and the same
threshold would empty the source entirely."""),
        code("""
FACTOR = 4

def largest_multiple_at_or_below(rows, token, factor):
    smallest = min(min(np.asarray(row[token]["data"]).shape[-2:]) for row in rows)
    return (smallest // factor) * factor, smallest

coarse_shapes, coarse_side = {}, None
for token in SR_INPUTS:
    side, native = largest_multiple_at_or_below(native_rows, token, FACTOR)
    coarse_shapes[token] = (side, side)
    if token == "l3_ssh":
        coarse_side = side
    print(f"{token:9s} smallest native side {native:>3d}  -> input {side} x {side}")

fine_side = coarse_side * FACTOR
swath_native = min(min(np.asarray(row[SR_TARGET]["data"]).shape[-2:]) for row in native_rows)
print(f"{SR_TARGET:9s} smallest native side {swath_native:>3d}  -> target {fine_side} x {fine_side}")
print(f"factor against the nadir input: {fine_side // coarse_side}x")
print(f"every shape at or below its own native grid: "
      f"{all(side <= min(min(np.asarray(row[token]['data']).shape[-2:]) for row in native_rows) for token, (side, _) in coarse_shapes.items()) and fine_side <= swath_native}")

SR_SUPPORT = {"l3_ssh": 0.0, "l4_sst": 0.5, "l4_ssh": 0.5, "l3_swot": 0.5}
sr_sources = {token: Resample(shape, support_threshold=SR_SUPPORT[token])
              for token, shape in coarse_shapes.items()}
sr_sources[SR_TARGET] = Resample((fine_side, fine_side), support_threshold=SR_SUPPORT[SR_TARGET])
sr_set = OceanTACODataset(queries=sr_draw, sources=sr_sources, catalog_config=config)
sr_batch = one_batch(sr_set)
require_complete(sr_batch, list(SR_TOKENS), "super-resolution")
print()
for token in SR_TOKENS:
    fractions = sr_batch[token]["valid_mask"].flatten(1).float().mean(dim=1)
    print(f"  {token:9s} {tuple(sr_batch[token]['data'].shape)!s:20s} "
          f"valid fraction per row {[round(float(value), 2) for value in fractions]}")
print()
print("No upsampling warning is raised, because each shape stays at or below its own native grid.")
"""),
        md("""### The batch this query shape produces

The first three rows are the inputs and the bottom row is the target, one
column per batch member. Two things are visible at once.

The resolution gap is the first: the nadir panels step in visible blocks and
the swath panels resolve structure inside a track of the same width. That
contrast is the section's subject, and it is what forcing every source onto one
output grid would erase.

The second is what the dense inputs add. The two L4 analyses fill their patch
where both observations leave most of it empty, and the temperature field
carries fronts in the same places the swath shows height structure. A model
given only the nadir track would have to invent the rest of the patch; given
the analyses as well, it has a coarse field everywhere and a sharp observation
along one track.

The white areas on the observation rows are honest. Neither instrument covers a
whole patch on a given day, and where the two tracks do not coincide the
observation input and the target describe different parts of the box. A model
on this pair has to handle that, which is why the geometry is worth looking at
before choosing a loss."""),
        code("""
count = sr_batch[SR_TARGET]["data"].shape[0]
fig, axes = plt.subplots(len(SR_TOKENS), count, squeeze=False,
                         figsize=(3.4 * count, 3.3 * len(SR_TOKENS)), constrained_layout=True)
for row, token in enumerate(SR_TOKENS):
    for column in range(count):
        axis = axes[row, column]
        record = batch_member(sr_batch, token, column)
        extent = geographic_extent(record)
        field = np.where(np.asarray(record["valid_mask"])[0], np.asarray(record["data"])[0], np.nan)
        drawn = axis.imshow(field, origin="lower", cmap=CMAP, interpolation="nearest",
                            aspect=map_aspect(extent), extent=extent, **color_limits(token))
        add_coastlines(axis)
        axis.set(xticks=[], yticks=[])
        if row == 0:
            axis.set_title(batch_dates(sr_batch)[column], fontsize=9)
    shape = (fine_side, fine_side) if token == SR_TARGET else coarse_shapes[token]
    role = "target: " if token == SR_TARGET else "input: "
    axes[row, 0].set_ylabel(role + source_label(token, shape), fontsize=9)
    fig.colorbar(drawn, ax=axes[row, :].tolist(), shrink=.85, label=colorbar_label(token))
fig.suptitle(f"Super-resolution at {SUPER_PATCH_KM} km: nadir track and two L4 analyses "
             f"-> SWOT swath, {fine_side // coarse_side}x on the nadir grid", fontsize=12)
display_figure(fig)
print("The nadir input steps in visible blocks; the target resolves structure inside the same track.")
print("The two L4 rows are the dense context that makes the rest of the patch predictable.")
"""),
        md("""## 4. Multi-source: a sparse observation beside dense analyses

A model that fuses a sparse observation with dense analyses needs rows where
the sparse source is actually present. Drawing at random and discarding the
misses would work, but it wastes fetches and makes the row count depend on
luck. Conditioning the draw on recorded coverage states the requirement up
front and costs no downloads, because coverage reads the published fact table
rather than the granules.

Section 3 rendered the same three tokens, but its subject was resolution: what
each instrument resolves, and how the shapes are chosen. The subject here is
**selection** — how a draw is conditioned so that every row carries the sparse
source at all, and what the resulting geometry costs a fusion model. So this
section draws its own rows against a stricter swath requirement rather than
reusing §3's, and renders every source on one shared grid, because a fusion
model that concatenates its inputs channel-wise needs them aligned.

The dense sources carry no coverage evidence of their own — the fact table
records `swot`, `ssh` and `argo` only — so their completeness is checked after
the draw rather than required before it."""),
        code("""
multi_filter = QueryFilter(box=BOX, coverage=(
    CoverageRequirement("swot", "valid_fraction_ocean", 0.25, aggregate="min"),))
multi_draw = draw_queryset(queryset, requested_row_count=BATCH_SIZE, seed=17,
                           record_path=DRAW_DIR / "multisource-draw.json",
                           query_filter=multi_filter)
DENSE_SHAPE = (40, 40)
multi_sources = {
    "l4_sst": Resample(DENSE_SHAPE, support_threshold=0.5),
    "l4_ssh": Resample(DENSE_SHAPE, support_threshold=0.5),
    "l3_swot": Resample((128, 128), support_threshold=0.5),
}
multi_set = OceanTACODataset(queries=multi_draw, sources=multi_sources, catalog_config=config)
multi_batch = one_batch(multi_set)
require_complete(multi_batch, list(multi_sources), "multi-source")
rows = multi_batch["l4_sst"]["data"].shape[0]
for token in multi_sources:
    fractions = multi_batch[token]["valid_mask"].flatten(1).float().mean(dim=1)
    print(f"  {token:9s} valid fraction per row {[round(float(value), 2) for value in fractions]}")
print()
print("A low valid fraction on the swath is evidence about sampling, not about the ocean.")
"""),
        md("""### The batch this query shape produces

Every row carries all three sources by construction. The figure shows what
carrying them looks like, and the contrast between the rows is the point: the
analyses fill their patch, and the swath crosses it at a different angle and a
different width in every column.

A model fusing these three has to handle that variation per sample rather than
assume a fixed observation geometry."""),
        code("""
tokens = list(multi_sources)
fig, axes = plt.subplots(len(tokens), rows, squeeze=False,
                         figsize=(3.2 * rows, 3.3 * len(tokens)), constrained_layout=True)
for row, token in enumerate(tokens):
    for column in range(rows):
        axis = axes[row, column]
        record = batch_member(multi_batch, token, column)
        extent = geographic_extent(record)
        field = np.where(np.asarray(record["valid_mask"])[0], np.asarray(record["data"])[0], np.nan)
        drawn = axis.imshow(field, origin="lower", cmap=CMAP, interpolation="nearest",
                            aspect=map_aspect(extent), extent=extent, **color_limits(token))
        add_coastlines(axis)
        axis.set(xticks=[], yticks=[])
        if row == 0:
            axis.set_title(batch_dates(multi_batch)[column], fontsize=9)
    shape = DENSE_SHAPE if token.startswith("l4_") else (128, 128)
    axes[row, 0].set_ylabel(source_label(token, shape), fontsize=9)
    fig.colorbar(drawn, ax=axes[row, :].tolist(), shrink=.85, label=colorbar_label(token))
fig.suptitle(f"One batch of {rows} rows: two dense analyses and the SWOT swath", fontsize=12)
display_figure(fig)
print("The white area in the l3_swot row is the part of the patch the satellite did not overfly.")
print("Conditioning the draw on coverage is what makes every one of these rows usable.")
"""),
        md("""## 5. Batching native shapes: padding against bucketing

The sources in one sample need not share a shape. Collation runs per token, so
a `Resample` token and a `Native()` token collate into two differently-shaped
stacks without interfering.

`Native()` is where shapes genuinely vary between rows, because the swath
crosses each patch differently. Stacking then needs either padding, which
invents cells, or grouping rows that already agree. Both appear below, on the
same draw, so the trade-off is visible as numbers rather than asserted.

Building the sampler calls `native_shapes`, a deliberate O(N) rendering pass in
the parent process. It is not free on a large draw, so treat it as setup cost
rather than something to call per epoch."""),
        code("""
from ocean_taco.torch import native_shapes

# A larger draw than the sections above, because bucketing only has something
# to show when several rows share a shape. Eight rows with eight distinct
# shapes would make every bucket a singleton and every comparison trivial.
bucket_draw = draw_queryset(queryset, requested_row_count=16, seed=23,
                            record_path=DRAW_DIR / "bucket-draw.json",
                            query_filter=multi_filter)
mixed = OceanTACODataset(queries=bucket_draw, catalog_config=config, sources={
    "l4_sst": Resample(DENSE_SHAPE, support_threshold=0.5),
    "l3_swot": Native(),
})
shapes = native_shapes(mixed, "l3_swot")
distinct = {}
for shape in shapes:
    distinct[tuple(shape)] = distinct.get(tuple(shape), 0) + 1
print(f"native l3_swot shapes across {len(shapes)} rows: {len(distinct)} distinct")
for shape, occurrences in sorted(distinct.items(), key=lambda item: -item[1])[:5]:
    print(f"  {shape[0]:>3d} x {shape[1]:<4d} in {occurrences} row(s)")
"""),
        code("""
BUCKET_BATCH = 4
padded_batch = one_batch(mixed, batch_size=BUCKET_BATCH, collate=native_pad_collate)
fixed, native = padded_batch["l4_sst"], padded_batch["l3_swot"]
print(f"l4_sst  stacks directly: {tuple(fixed['data'].shape)}")
print(f"l3_swot padded up to:    {tuple(native['data'].shape)}")
draw_order_padding = int(native["spatial_padding_mask"].sum())

sampler = ShapeBucketSampler(shapes, batch_size=BUCKET_BATCH, seed=19, shuffle=False)
bucket_loader = DataLoader(mixed, batch_sampler=sampler, collate_fn=native_pad_collate,
                           num_workers=2, worker_init_fn=seed_ocean_taco_worker)
bucketed_padding, sizes = 0, []
for bucketed in bucket_loader:
    bucketed_padding += int(bucketed["l3_swot"]["spatial_padding_mask"].sum())
    sizes.append(bucketed["l3_swot"]["data"].shape[0])
print()
print(f"batches of {BUCKET_BATCH} in draw order: 1 shown, {draw_order_padding:,d} padded cells")
print(f"bucketed into {len(sizes)} batches of sizes {sizes}: {bucketed_padding:,d} padded cells")
print("Grouping by shape removes the padding. The cost is that a batch no longer holds")
print("an independent sample of the draw, which matters if batch composition reaches the loss.")
"""),
        md("""### What padding looks like

`native_pad_collate` pads each row up to the largest in its batch and records
where it did so in `spatial_padding_mask`. Padded cells are not measurements,
and not missing measurements either: they are an artifact of putting rows of
different sizes in one tensor, which is why they get their own mask rather than
being folded into `valid_mask`."""),
        code("""
count = native["data"].shape[0]
fig, axes = plt.subplots(2, count, squeeze=False, figsize=(3.0 * count, 6.4),
                         constrained_layout=True)
for column in range(count):
    swot = np.where(np.asarray(native["valid_mask"])[column][0],
                    np.asarray(native["data"])[column][0], np.nan)
    drawn = axes[0, column].imshow(swot, origin="lower", cmap=CMAP, interpolation="nearest",
                                   aspect="auto", **color_limits("l3_swot"))
    axes[1, column].imshow(np.asarray(native["spatial_padding_mask"])[column], origin="lower",
                           cmap="Greys", interpolation="nearest", aspect="auto", vmin=0, vmax=1)
    true_shape = native["true_shapes"][column]
    axes[0, column].set_title(f"{true_shape[1]}x{true_shape[2]} native", fontsize=9)
    for row in range(2):
        axes[row, column].set(xticks=[], yticks=[])
axes[0, 0].set_ylabel(source_label("l3_swot") + ", padded", fontsize=9)
axes[1, 0].set_ylabel("spatial_padding_mask", fontsize=9)
add_colorbar(fig, drawn, axes[0, :].tolist(), "l3_swot", shrink=.85)
fig.suptitle(f"One padded batch of {count} native rows, with the padding it required", fontsize=12)
display_figure(fig)
print("The dark region in the lower row is padding, and it differs per row because the native")
print("shapes do. These panels are drawn in array space, not on a map, so they are not to scale.")
"""),
        md("""## 6. Normalisation statistics, then a training step

Statistics belong to the experiment rather than to the batch. Computing them
per batch would let batch composition reach the inputs, so this section
computes one mean and standard deviation over the training rows and holds them
fixed for everything downstream.

They are accumulated by **streaming the whole training draw** rather than
reading one batch. A mean over a single batch of four patches is not a
statistic; it is a sample of whatever four positions the draw happened to
return. Summing values, squares and counts across the loader gives the same
answer a full pass would, at one batch of memory.

The average runs through `valid_mask`. Averaging the raw array instead would
fold in cells the instrument never measured, and since those are NaN the result
would be NaN. The comparison against a zero-filled average shows what the mask
excludes: zero-filling treats every unobserved cell as a measured zero, so the
mean moves towards zero in proportion to how much of the patch went
unobserved."""),
        code("""
training_set = QuerySet.from_hub(SUPER_PATCH_KM, "training")
stats_draw = draw_queryset(training_set, requested_row_count=32, seed=3,
                           record_path=DRAW_DIR / "stats-draw.json",
                           query_filter=QueryFilter(box=BOX))
stats_loader = DataLoader(OceanTACODataset(queries=stats_draw, sources=SST, catalog_config=config),
                          batch_size=8, num_workers=2, collate_fn=collate_ocean_samples,
                          worker_init_fn=seed_ocean_taco_worker)

total, total_square, cells, rows_seen = 0.0, 0.0, 0, 0
for batch in stats_loader:
    values, mask = batch["l4_sst"]["data"], batch["l4_sst"]["valid_mask"]
    selected = values[mask]
    total += float(selected.sum())
    total_square += float((selected ** 2).sum())
    cells += int(mask.sum())
    rows_seen += values.shape[0]
SST_MEAN = total / cells
SST_STD = (total_square / cells - SST_MEAN ** 2) ** 0.5
print(f"streamed {rows_seen} training rows in batches of 8; {cells:,d} valid cells")
print(f"fixed statistics: mean={SST_MEAN:.3f} {UNITS['l4_sst']}, std={SST_STD:.3f} {UNITS['l4_sst']}")
"""),
        code("""
sparse_batch = one_batch(OceanTACODataset(
    queries=multi_draw, sources={"l3_swot": Resample((128, 128), support_threshold=0.5)},
    catalog_config=config))
sparse_values, sparse_mask = sparse_batch["l3_swot"]["data"], sparse_batch["l3_swot"]["valid_mask"]
filled = torch.nan_to_num(sparse_values, nan=0.0)
masked_mean = float(sparse_values[sparse_mask].mean())
print(f"l3_swot valid cells: {int(sparse_mask.sum()):,d} of {sparse_mask.numel():,d}")
print(f"  masked mean      = {masked_mean:.4f} {UNITS['l3_swot']}")
print(f"  zero-filled mean = {float(filled.mean()):.4f} {UNITS['l3_swot']}")
print("Zero-filling counts every unsampled cell as a measured zero and pulls the average towards it.")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), constrained_layout=True)
raw = sparse_values[sparse_mask].flatten().numpy()
axes[0].hist(raw, bins=60, color="#2474a6")
axes[0].axvline(masked_mean, color="#333333", linestyle="--", linewidth=1)
axes[0].set(title=f"{source_label('l3_swot')}, measured cells only",
            xlabel=colorbar_label("l3_swot"), ylabel="cells")
axes[1].hist(filled.flatten().numpy(), bins=60, color="#d95f02")
axes[1].axvline(float(filled.mean()), color="#333333", linestyle="--", linewidth=1)
axes[1].axvline(masked_mean, color="#2474a6", linestyle=":", linewidth=1.4)
axes[1].set(title="the same source zero-filled, every unsampled cell counted",
            xlabel=colorbar_label("l3_swot"), ylabel="cells")
fig.suptitle("What the mask excludes: the spike at zero is cells the swath never visited", fontsize=12)
display_figure(fig)
print("The dotted line marks the honest average; the gap to the dashed one is the error")
print("that zero-filling introduces.")
"""),
        md("""### A training step with a masked loss

There is no single mask. A collated batch carries `valid_mask`, `source_valid`,
`support_mask`, `time_mask`, and `ocean_mask`, plus a top-level `availability`
dict. The relation between the first three is:

> `valid_mask = source_valid & support_mask`, further ANDed with the ocean mask
> when one is supplied.

The split exists so a reader can tell *why* a cell is invalid: the source
reported nothing (`source_valid`), the renderer had too little support to build
a value (`support_mask`), or the cell is land (`ocean_mask`). Drive the loss
from `valid_mask` and read the others when a row looks wrong.

The step below uses the midpoint draw from §2: context on both sides,
target in the middle, loss over cells valid in both."""),
        code("""
def masked_mse(prediction, target, mask):
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
loader_args = dict(batch_size=2, num_workers=2, collate_fn=collate_ocean_samples,
                   worker_init_fn=seed_ocean_taco_worker)
context_loader = DataLoader(OceanTACODataset(queries=midpoint_draw, sources=SST,
                                             catalog_config=config), **loader_args)
target_loader = DataLoader(OceanTACODataset(queries=target_rows(midpoint_draw, midpoint_filter),
                                            sources=SST, catalog_config=config), **loader_args)

for step, (context_batch_step, target_batch_step) in enumerate(zip(context_loader, target_loader)):
    context, target = context_batch_step["l4_sst"], target_batch_step["l4_sst"]
    context_valid = context["valid_mask"]
    inputs = normalise(context["data"], context_valid).mean(dim=1, keepdim=True)
    targets = normalise(target["data"], target["valid_mask"])
    mask = context_valid.any(dim=1, keepdim=True) & target["valid_mask"]
    loss = masked_mse(model(inputs), targets, mask)
    optimiser.zero_grad(); loss.backward(); optimiser.step()
    print(f"step {step}: batch={tuple(inputs.shape)} valid cells={int(mask.sum()):,d} "
          f"of {mask.numel():,d} loss={float(loss.detach()):.4f}")
print("The loss saw only cells valid in both the context and the target.")
"""),
        code("""
mask_record = OceanTACODataset(queries=multi_draw, catalog_config=config,
                               sources={"l3_swot": Resample((128, 128), support_threshold=0.5)})[0]["l3_swot"]
data_panel = np.asarray(mask_record["data"])[0]
source_panel = np.asarray(mask_record["source_valid"])[0]
support_panel = np.asarray(mask_record["support_mask"])[0]
valid_panel = np.asarray(mask_record["valid_mask"])[0]
extent = geographic_extent(mask_record)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.0), constrained_layout=True)
panels = ((source_label("l3_swot"), np.where(valid_panel, data_panel, np.nan), CMAP, color_limits("l3_swot")),
          (f"source_valid ({source_panel.mean():.2f})", source_panel, "Greys_r", {"vmin": 0, "vmax": 1}),
          (f"support_mask ({support_panel.mean():.2f})", support_panel, "Greys_r", {"vmin": 0, "vmax": 1}),
          (f"valid_mask ({valid_panel.mean():.2f})", valid_panel, "Greys_r", {"vmin": 0, "vmax": 1}))
for axis, (title, panel, cmap, limits) in zip(axes, panels):
    drawn = axis.imshow(panel, origin="lower", cmap=cmap, interpolation="nearest",
                        aspect=map_aspect(extent), extent=extent, **limits)
    add_coastlines(axis)
    axis.set(title=title, xticks=[], yticks=[])
    fig.colorbar(drawn, ax=axis, shrink=.8)
fig.suptitle("Where a mask excludes cells, and why. The sparse source is the one that shows it.",
             fontsize=12)
display_figure(fig)
print(f"valid_mask equals source_valid AND support_mask: "
      f"{bool((valid_panel == (source_panel & support_panel)).all())}")
"""),
        md("""### What to carry forward

The pieces this notebook assembled are the ones an experiment has to persist:
the draw record, the target-row rule that derives the target offsets from the
filter, the renderer shapes and support thresholds, and the normalisation
statistics from §6. Those plus the QuerySet ID and the pinned catalog revision
reproduce every tensor above, which the [overview
notebook](ml_dataset.ipynb) lists in full."""),
    ],
)

write(
    "data_retrieval_workflows.ipynb",
    [
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
        code(SETUP),
        code(PLOTTING),
        code(LOAD),
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

For a box spanning region tiles, `load_bbox_nc` resolves the intersecting
assets and aligns their lightweight coordinate axes before selecting the box.
It reads the requested spatial slices before reindexing and merging values,
preserving coordinate clustering, inclusive boundaries, and wrapped longitude
order. Overlap conflicts are checked inside the requested footprint. The returned field keeps
its **native coordinates**, since this call retrieves rather than renders, so
there is no target grid and no interpolation.

Three return values mean three different things. `None` means no asset matched
the request. An empty field means the asset existed and held nothing inside the
box. Invalid coordinates or dates raise `ValueError` rather than returning
something falsy.

Scientific retrieval keeps all source variables by default; only the ML source
adapter projects the dense variables requested by its renderers. Shared GLORYS
tokens therefore still expose the full file through this scientific API.
"""),
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
    sst_extent = (float(sst["lon"].min()), float(sst["lon"].max()),
                  float(sst["lat"].min()), float(sst["lat"].max()))
    image = axis.imshow(field, origin="lower", aspect=map_aspect(sst_extent), cmap=CMAP,
                        interpolation="nearest", extent=sst_extent, **color_limits("l4_sst"))
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
    ],
)

write(
    "ml_configuration_cookbook.ipynb",
    [
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
        code(SETUP),
        code(PLOTTING),
        code(LOAD),
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
                        aspect=map_aspect(extent), extent=extent, **color_limits(key))
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

When configured independently, eastward and northward velocity components can
have different valid regions. A valid `u` with an invalid `v` does not define
a measured vector.

`VectorPair` produces both components together as `(T, 2, H, W)`. Its
`valid_mask` requires support for both components, and `pair_available` gives
sample-level availability. The shared asset is fetched once.

For `VectorPair(Resample(...))`, `source_valid`, `support`, `support_mask`, and
`valid_mask` all have shape `(T, H, W)` on the output grid; the component axis
appears only in `data`. `source_valid` equals `support > 0`, with support
computed jointly from both components. This corrects the earlier native-grid
`source_valid` shape and allows batches with differing native shapes or missing
records to collate. `VectorPair(Native())` keeps its native masks.
"""),
        code("""vectors = {"velocity": VectorPair(Resample((64, 64), .5))}
print("components:", vectors["velocity"].components)
velocity_sample = render(vectors)
velocity = velocity_sample["velocity"]
data = np.asarray(velocity["data"])
mask_shape = (data.shape[0], *data.shape[-2:])
for key in ("source_valid", "support", "support_mask", "valid_mask"):
    assert tuple(velocity[key].shape) == mask_shape
np.testing.assert_array_equal(velocity["source_valid"], np.asarray(velocity["support"]) > 0)
if data.shape[0]:
    speed = np.hypot(data[0][0], data[0][1])
    print(f"u range=[{np.nanmin(data[:, 0]):.3f}, {np.nanmax(data[:, 0]):.3f}] m/s")
    print(f"v range=[{np.nanmin(data[:, 1]):.3f}, {np.nanmax(data[:, 1]):.3f}] m/s")
    print(f"speed max={np.nanmax(speed):.3f} m/s, pair_available={bool(velocity['pair_available'])}")"""),
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
                           aspect=map_aspect(extent), extent=extent, **color_limits("speed"))
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
    ],
)

write(
    "plot_hurricane_milton.ipynb",
    [
        md("""# Hurricane Milton: wind and SSH across products

The maintained helper produces the four-date, three-column projected figure:
L4 wind with vectors, L3 along-track SSH, and L3 SWOT. The dense SWOT product
is always requested; a missing source is shown as missing rather than hidden."""),
        code(SETUP),
        code("""
from ocean_taco import CatalogConfig
from ocean_taco.retrieve import load_hf_dataset
from ocean_taco.figures.hurricane_milton import DEFAULT_DATES, close_data, load_date, make_figure
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
    ],
)

write(
    "plot_hurricane_milton_cross_product.ipynb",
    [
        md("""# Hurricane Milton: SSH cross-product comparison

This workflow overlays L4 DUACS, L3 along-track, and L3 SWOT in projected
geography, then compares each L3 product with L4 after interpolation to the
observation coordinates. It reports deterministic-subsample visualisation,
correlation, RMSE, and a 1:1 reference line."""),
        code(SETUP),
        code("""
from ocean_taco import CatalogConfig
from ocean_taco.retrieve import load_hf_dataset
from ocean_taco.figures.hurricane_milton_cross_product import close_products, load_products, make_figure
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
    ],
)
