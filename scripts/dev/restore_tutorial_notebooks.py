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

def write(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}})
    nbf.write(notebook, OUT / name)

write("ml_dataset.ipynb", [
    md("""# From a published QuerySet to a PyTorch batch

A QuerySet is a released *population* of positions and dates. A filter selects
from that population; a recorded draw selects rows reproducibly; an
`OceanTACODataset` renders those rows into model-facing samples. Each stage
narrows the one before it, and each records what it selected, so a batch can be
traced back to the released population it came from."""),
    md("""## 1. Setup

Nothing here is configured. `CatalogConfig()` carries a pinned catalog
revision, and `QuerySet.from_hub` fetches a published QuerySet by patch size
and kind, so the only inputs are the three constants below. Loading the
QuerySet verifies every table against the checksums in its header, which means
a successful load is itself the integrity check."""),
    code(SETUP), code(LOAD),
    md("""## 2. Geography comes before pixels

A patch is specified in kilometres. Its longitude span grows with latitude
because a degree of longitude becomes shorter away from the equator. Renderers
then choose either a fixed model grid or the source's native grid; neither
choice is implicit normalisation."""),
    code("""
from ocean_taco import PatchSize
import matplotlib.pyplot as plt
patch = PatchSize(PATCH_SIZE_KM, "km")
latitudes = [0, 30, 45, 60]
widths = [patch.to_degrees(centre_lat=lat)[0] for lat in latitudes]
plt.plot(latitudes, widths, marker="o")
plt.xlabel("centre latitude [°]"); plt.ylabel("longitude width [°]")
plt.title(f"{PATCH_SIZE_KM} km patch: latitude-aware longitude span")
display_figure(plt.gcf())
print(dict(zip(latitudes, map(lambda x: round(x, 2), widths))))
"""),
    md("""## 3. Train and evaluation populations

Kinds communicate intended use, but they do not alone prevent leakage. Create
disjoint temporal and spatial policies for the experiment, retain their draw
records, and inspect selected row IDs before training."""),
    code("""
from ocean_taco import GeoBox, QueryFilter, select_queryset
from ocean_taco.manifest import QuerySet
training = QuerySet.from_hub(PATCH_SIZE_KM, "training")
evaluation = queryset
split_box = GeoBox(-80, -30, 10, 45)
train_population = select_queryset(training, QueryFilter(box=split_box))
eval_population = select_queryset(evaluation, QueryFilter(box=split_box))
print(f"training kind={training.header['kind']}, candidates={train_population.count}")
print(f"evaluation kind={evaluation.header['kind']}, candidates={eval_population.count}")
print("Use an explicit time guard band as well; a QuerySet kind is not a leakage proof.")
"""),
    code("""
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
for axis, label, source, colour in zip(
    axes, ("training population", "evaluation population"),
    (training, evaluation), ("#2474a6", "#238b45"),
):
    axis.scatter([p["centre_lon"] for p in source.positions], [p["centre_lat"] for p in source.positions],
                 s=1.5, alpha=.18, color=colour, rasterized=True)
    axis.set(title=label, xlabel="longitude [°]", ylabel="latitude [°]")
    axis.grid(alpha=.2)
fig.suptitle("Published positions: stochastic training and systematic evaluation")
fig.tight_layout()
from IPython.display import display
display_figure(fig)
print("Each point is a stored patch centre, not a rendered observation.")
"""),
    md("## 4. Filter, draw, and replay at most eight rows"),
    code("""
from ocean_taco import QueryFilter, draw_queryset, replay_experiment
selection = QueryFilter(box=GeoBox(-80, -30, 10, 45))
draw = draw_queryset(queryset, requested_row_count=min(REQUESTED_ROWS, 8), seed=SEED,
                     record_path=DRAW_DIR / "ml-dataset-draw.json", query_filter=selection)
replayed = replay_experiment(queryset, DRAW_DIR / "ml-dataset-draw.json")
assert replayed.rows == draw.rows
for row in draw.rows:
    print(row["position_id"][:12], row["anchor_time"], row["centre_lon"], row["centre_lat"], row["patch_id"][:12])
print(f"inclusion_probability={draw.inclusion_probability:.6g}; record={DRAW_DIR / 'ml-dataset-draw.json'}")
"""),
    code("""
import matplotlib.pyplot as plt
fig, axis = plt.subplots(figsize=(7, 4))
axis.scatter([p["centre_lon"] for p in queryset.positions], [p["centre_lat"] for p in queryset.positions],
             s=1.2, alpha=.10, color="#9aa5b1", label="published evaluation population")
axis.scatter([row["centre_lon"] for row in draw.rows], [row["centre_lat"] for row in draw.rows],
             s=65, marker="*", color="#d95f02", label="recorded draw")
axis.set(xlabel="longitude [°]", ylabel="latitude [°]", title="A replayable draw stays attached to its source population")
axis.legend(loc="best"); axis.grid(alpha=.2)
fig.tight_layout()
from IPython.display import display
display_figure(fig)
print("The orange stars are the rows used below; their IDs are stored in the draw record.")
"""),
    md("""## 5. Dataset schema and one real render

Every source has data, coordinates, validity/support masks, and availability.
An unavailable source is represented structurally, rather than being silently
dropped. Here fixed resampling makes a conventional dense model input."""),
    code("""
from ocean_taco.render import Resample
from ocean_taco.torch import OceanTACODataset
dataset = OceanTACODataset(queries=draw, sources={
    "l4_sst": Resample((48, 48), support_threshold=0.5),
    "l3_swot": Resample((48, 48), support_threshold=0.5),
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
artist = plot_ocean_sample(sample, "l4_sst")
artist.axes.set_title("Rendered L4 SST with native geographic coordinates")
artist.axes.figure.colorbar(artist, ax=artist.axes, label="SST")
artist.axes.figure.tight_layout()
display_figure(artist.axes.figure)
"""),
    md("""## 6. Collation, native shapes, and workers

`collate_ocean_samples` preserves availability masks. Fixed `Resample`
outputs are easy to batch; `Native()` keeps exact shapes and should be paired
with `ShapeBucketSampler`. Planning occurs before worker processes; no
normalisation is performed by the loader."""),
    code("""
from torch.utils.data import DataLoader
from ocean_taco.torch import collate_ocean_samples
loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_ocean_samples)
batch = next(iter(loader))
print("batch l4_sst", tuple(batch["l4_sst"]["data"].shape))
print("availability", batch["availability"])
print("valid cells", int(batch["l4_sst"]["valid_mask"].sum()))
"""),
    code("""
import matplotlib.pyplot as plt
tokens = list(batch["availability"])
available = [sum(batch["availability"][token]) for token in tokens]
fig, axis = plt.subplots(figsize=(6, 3.5))
axis.bar(tokens, available, color=["#2474a6", "#8c510a"])
size = len(batch["availability"][tokens[0]])
axis.set(ylim=(0, size + .5), ylabel="available records in batch", title="Availability is collated separately from values")
for index, value in enumerate(available):
    axis.text(index, value + .05, f"{value}/{size}", ha="center")
fig.tight_layout()
from IPython.display import display
display_figure(fig)
print("A source can be unavailable for one batch member without changing the batch tensor layout.")
"""),
    code("""
import torch
def normalise_valid(data, valid_mask, mean, std):
    result = torch.full_like(data, float("nan"))
    result[valid_mask] = (data[valid_mask] - mean) / torch.as_tensor(std, dtype=data.dtype).clamp_min(1e-6)
    return result
example = torch.tensor([1.0, float("nan"), 3.0])
print(normalise_valid(example, torch.isfinite(example), 2.0, 1.0).tolist())
print("Persist the QuerySet ID, header checksums, filter, draw record, renderer settings, and normalisation statistics.")
"""),

])

write("spatio_temporal_query_generation.ipynb", [
    md("""# Spatio-temporal QuerySet intuition

QuerySets factor a stable geographic position table from date-dependent
coverage evidence. They are not generated ad hoc at training time. This
notebook uses the real local evaluation and training populations to show how
selection, coverage, cadence, and replay fit together."""),
    code(SETUP), code(LOAD),
    md("## 1. Population geography: random training versus systematic evaluation"),
    code("""
import matplotlib.pyplot as plt
from ocean_taco.manifest import QuerySet
training = QuerySet.from_hub(PATCH_SIZE_KM, "training")
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
for axis, label, source, colour in zip(axes, ("training", "evaluation"), (training, queryset), ("#3182bd", "#2f855a")):
    axis.scatter([p["centre_lon"] for p in source.positions], [p["centre_lat"] for p in source.positions], s=1.2, alpha=.18, color=colour)
    axis.set(title=f"{label}: stored positions", xlabel="longitude", ylabel="latitude")
display_figure(plt.gcf())
print("The plot displays stored positions, not rendered data or an inferred coverage surface.")
"""),
    md("## 2. Null is not zero"),
    code("""
coverage = queryset.coverage
nulls = sum(row["swot_valid_cells"] is None for row in coverage)
zeros = sum(row["swot_valid_cells"] == 0 for row in coverage)
print(f"SWOT evidence: null={nulls} (not measured); zero={zeros} (measured absent)")
print("A coverage filter rejects null evidence; it must not coerce null to zero.")
"""),
    md("## 3. Box, date, and coverage filters"),
    code("""
from ocean_taco import CoverageRequirement, GeoBox, QueryFilter, select_queryset
box = GeoBox(-80, -30, 10, 45)
base = select_queryset(queryset, QueryFilter(box=box))
covered = select_queryset(queryset, QueryFilter(box=box, coverage=(CoverageRequirement("ssh", "valid_cells", 1),)))
print(f"box-only pairs={base.count}; with observed SSH coverage={covered.count}")
print("Coverage acts on the published fact table; it triggers no source download.")
"""),
    code("""
import matplotlib.pyplot as plt
all_pairs = len(queryset.positions) * len(queryset.dates)
labels = ["published pairs", "in box", "with SSH evidence"]
counts = [all_pairs, base.count, covered.count]
fig, axis = plt.subplots(figsize=(8, 3.5))
bars = axis.bar(labels, counts, color=["#9aa5b1", "#3182bd", "#2f855a"])
axis.set_yscale("log"); axis.set_ylabel("position/date pairs (log scale)")
axis.set_title("Filtering narrows published evidence before any rendering")
for bar, value in zip(bars, counts):
    axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
from IPython.display import display
display_figure(fig)
print("The final bar counts only rows with observed SSH coverage; null evidence was excluded explicitly.")
"""),
    md("## 4. Recorded draws and temporal guard bands"),
    code("""
from ocean_taco import draw_queryset, replay_experiment
guarded = QueryFilter(box=box, date_start=queryset.dates[2], date_end=queryset.dates[-3], context_start_offset_days=0, context_end_offset_days=0)
draw = draw_queryset(queryset, requested_row_count=min(4, select_queryset(queryset, guarded).count), seed=42,
                     record_path=DRAW_DIR / "query-generation-draw.json", query_filter=guarded)
assert replay_experiment(queryset, DRAW_DIR / "query-generation-draw.json").rows == draw.rows
print(f"drawn={len(draw.rows)}; inclusion_probability={draw.inclusion_probability:.6g}")
print("Guard bands must be part of the recorded split policy, not an after-the-fact convention.")
"""),
    md("## 5. Cadence and overlap are modelling choices"),
    code("""
dates = [value[:10] for value in queryset.dates[:8]]
print("first stored dates:", dates)
print("Training draws are stochastic; evaluation coverage is systematic. Choose cadence and spatial buffers for the scientific independence your metric claims.")
"""),
    code("""
import matplotlib.pyplot as plt
import numpy as np
dates = [value[:10] for value in queryset.dates[:8]]
index = np.arange(len(dates))
fig, axis = plt.subplots(figsize=(9, 2.7))
axis.scatter(index, np.zeros_like(index), s=70, color="#2f855a", label="stored evaluation dates")
axis.axvspan(.8, 2.2, color="#f6ad55", alpha=.35, label="example temporal guard band")
axis.set(xticks=index, xticklabels=dates, ylim=(-.8, .8), yticks=[], title="Cadence and guard bands belong to the split policy")
axis.tick_params(axis="x", rotation=30); axis.legend(loc="upper right")
fig.tight_layout()
from IPython.display import display
display_figure(fig)
print("The shaded interval is a policy illustration; it does not alter the published dates.")
"""),

])

write("data_retrieval_workflows.ipynb", [
    md("""# Native-coordinate retrieval workflows

This guide starts with catalog identity, then retrieves one tile, a geographic
box, a multi-source time range, and native Argo points. Retrieval returns
native coordinates and may return `None` for no matching asset; it never
fabricates a dense field."""),
    code(SETUP), code(LOAD),
    md("## 1. Catalog rows and source tokens"),
    code("""
from ocean_taco.registry import MODALITY_REGISTRY
print(f"catalog URL={config.resolved_catalog_url}")
for token in ("l4_sst", "l4_ssh", "l3_swot", "argo"):
    spec = MODALITY_REGISTRY[token]
    print(token, "filename=", spec.filename, "points=", spec.is_points, "variables=", spec.available_variables[:3])
"""),
    code("""
import matplotlib.pyplot as plt
from ocean_taco.registry import MODALITY_REGISTRY
tokens = ("l4_sst", "l4_ssh", "l3_swot", "argo")
fig, axis = plt.subplots(figsize=(7, 3))
axis.barh(tokens, [not MODALITY_REGISTRY[token].is_points for token in tokens], color=["#3182bd", "#3182bd", "#756bb1", "#dd8a45"])
axis.set(xlim=(0, 1.15), xticks=(0, 1), xticklabels=("point records", "gridded field"), title="Catalog modality families")
fig.tight_layout()
from IPython.display import display
display_figure(fig)
print("This summary comes from the registry used by the retrieval calls above.")
"""),
    md("## 2. One named tile"),
    code("""
from ocean_taco.retrieve import load_tile_nc
date = queryset.dates[0][:10]
tile = load_tile_nc(catalog, date, "NORTH_ATLANTIC", "l4_sst", config=config)
print("date", date, "tile sizes", None if tile is None else dict(tile.sizes))
if tile is not None: print("variables", list(tile.data_vars))
"""),
    md("## 3. Box retrieval and merge"),
    code("""
from ocean_taco import GeoBox, TimeRange
from ocean_taco.retrieve import load_bbox_nc, load_multisource_time_series_nc
box = GeoBox(-80, -30, 25, 45)
sst = load_bbox_nc(catalog, date, box, "l4_sst", config=config)
print("box", box.to_dict())
print("sizes", None if sst is None else dict(sst.sizes), "coordinates", None if sst is None else {key: (float(sst[key].min()), float(sst[key].max())) for key in ("lat", "lon")})
"""),
    code("""
import matplotlib.pyplot as plt
import numpy as np
if sst is None:
    print("No matching SST asset: no field is plotted.")
else:
    variable = next(name for name in sst.data_vars if sst[name].ndim >= 2)
    field = np.asarray(sst[variable]).squeeze()[::8, ::8]
    fig, axis = plt.subplots(figsize=(8, 4))
    im = axis.imshow(field, origin="lower", aspect="auto", cmap="turbo",
                     extent=(float(sst["lon"].min()), float(sst["lon"].max()), float(sst["lat"].min()), float(sst["lat"].max())))
    axis.set(xlabel="longitude [°]", ylabel="latitude [°]", title=f"Retrieved box: {variable}")
    fig.colorbar(im, ax=axis, label=variable)
    fig.tight_layout()
    from IPython.display import display
    display_figure(fig)
    print("The image is a decimated display of the returned native-coordinate field.")
"""),
    md("## 4. A multi-source closed time range"),
    code("""
window = TimeRange(queryset.dates[0], queryset.dates[1])
stack = load_multisource_time_series_nc(catalog, ("l4_sst", "l4_ssh", "l3_swot"), box, window, config=config)
for token, value in stack.items():
    print(token, None if value is None else {"sizes": dict(value.sizes), "variables": list(value.data_vars)[:4]})
"""),
    code("""
import matplotlib.pyplot as plt
tokens = list(stack)
steps = [0 if value is None else int(value.sizes.get("time", 1)) for value in stack.values()]
fig, axis = plt.subplots(figsize=(7, 3.5))
axis.bar(tokens, steps, color=["#3182bd", "#6baed6", "#756bb1"])
axis.set(ylabel="retrieved time steps", title="The multi-source request preserves each source's cadence")
for index, value in enumerate(steps): axis.text(index, value + .03, str(value), ha="center")
fig.tight_layout()
from IPython.display import display
display_figure(fig)
print("Different bar heights are expected: these products have different native temporal sampling.")
"""),
    md("## 5. Native point data, antimeridians, and validation and edge cases"),
    code("""
argo = load_bbox_nc(catalog, date, box, "argo", config=config)
wrapped = GeoBox(170, -170, 10, 30, wraps_antimeridian=True)
print("Argo records", 0 if argo is None else next(iter(argo.sizes.values())))
print("wrapped request has", len(wrapped.segments()), "explicit segments:", [segment.to_dict() for segment in wrapped.segments()])
print("Retrieval behavior: cache hits reuse immutable files; None means no catalog match; empty points are valid; invalid coordinates/dates raise ValueError.")
"""),
    code("""
import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
if argo is not None and all(key in argo for key in ("lon", "lat")):
    axes[0].scatter(np.asarray(argo["lon"]), np.asarray(argo["lat"]), s=10, alpha=.6, color="#dd8a45")
else:
    axes[0].text(.5, .5, "No Argo points in this selection", ha="center", va="center", transform=axes[0].transAxes)
axes[0].set(title="Native Argo point locations", xlabel="longitude [°]", ylabel="latitude [°]")
for segment in wrapped.segments():
    axes[1].fill_between([segment.lon_min, segment.lon_max], segment.lat_min, segment.lat_max, alpha=.45)
axes[1].set(xlim=(-190, 190), ylim=(0, 35), xlabel="longitude [°]", ylabel="latitude [°]", title="Antimeridian request splits into two boxes")
fig.tight_layout()
from IPython.display import display
display_figure(fig)
print("An empty point set is still a valid retrieval result; antimeridian geometry is represented explicitly.")
"""),
    code("""
for value in [tile, sst, argo, *stack.values()]:
    if value is not None and callable(close := getattr(value, "close", None)):
        close()
print("Closed opened datasets.")
"""),

])

write("ml_configuration_cookbook.ipynb", [
    md("""# ML configuration cookbook

A renderer decides what shape a source becomes, and that choice is not free: it
determines whether samples can be stacked into a batch, what happens when a
source is absent for a position and date, and how much of the native resolution
survives. Each recipe below states the tensor structure it produces, what it
does with empty data, and why its renderer and collator fit that structure.
Every recipe runs against the same pinned catalog revision and the same
published evaluation QuerySet loaded in the setup cells, so the recipes differ
only in their rendering configuration."""),
    code(SETUP), code(LOAD),
    md("## Fixed grids and multimodal fusion"),
    code("""
from ocean_taco.render import Native, Points, Resample, VectorPair
fixed = {"l4_sst": Resample((64, 64), .5), "l4_ssh": Resample((64, 64), .5)}
fusion = {**fixed, "l3_swot": Resample((64, 64), .5)}
print("layout: dense [T,H,W] per source; unavailable source has an empty structured payload")
print("fixed", fixed, "fusion", fusion)
"""),
    md("## Sparse/dense sources and vector pairs"),
    code("""
sparse_dense = {"l4_sst": Resample((64, 64), .5), "l3_swot": Resample((64, 64), .5)}
vectors = {"velocity": VectorPair(Resample((64, 64), .5))}
print("Sparse SWOT keeps its own valid/support mask; do not treat missing cells as observations.")
print("VectorPair shares availability across eastward/northward components:", vectors["velocity"].components)
"""),
    md("## Argo points"),
    code("""
points = {"argo": Points(variable="TEMP", pres_range=(0, 10))}
print("layout: ragged records with lat/lon/pressure/profile IDs; zero points is a valid batch member.")
print(points)
"""),
    md("## Forecasting with two datasets"),
    code("""
from ocean_taco import QueryFilter, draw_queryset
forecast = QueryFilter(relation="forecast", target_lead_days=1, context_start_offset_days=-1, context_end_offset_days=0)
draw = draw_queryset(queryset, requested_row_count=4, seed=19, record_path=DRAW_DIR / "forecast-draw.json", query_filter=forecast)
print(f"context/target draw rows={len(draw.rows)}; relation={forecast.relation}; lead={forecast.target_lead_days}d")
print("Use separate context and target datasets so a context model cannot accidentally consume the target.")
"""),
    md("## Native shapes and bucketing"),
    code("""
from ocean_taco.torch import ShapeBucketSampler
sampler = ShapeBucketSampler([(32, 48), (32, 48), (40, 48)], batch_size=2, seed=19)
print("Native keeps source shape; ShapeBucketSampler batches equal shapes:", list(sampler))
print("Use fixed Resample for ordinary dense fusion; use Native only when the model handles ragged resolution.")
"""),
    md("## Regional, antimeridian, leakage, and normalisation"),
    code("""
from ocean_taco import GeoBox
import torch
regional = GeoBox(-80, -30, 10, 45)
wrapped = GeoBox(170, -170, 10, 30, wraps_antimeridian=True)
def normalise_valid(data, mask, mean, std):
    output = torch.full_like(data, float("nan")); output[mask] = (data[mask] - mean) / max(std, 1e-6); return output
print("regional", regional.to_dict(), "wrapped segments", [segment.to_dict() for segment in wrapped.segments()])
print("Leakage control: disjoint QuerySet filters, temporal guard bands, and persisted draw records.")
print("normalisation", normalise_valid(torch.tensor([1., float("nan"), 3.]), torch.tensor([True, False, True]), 2., 1.).tolist())
"""),

    md("""## Retrieved sample gallery\n\nThe following figures use the same recorded local draw as the recipes above. They show actual rendered values and their validity evidence, so the configuration choices can be inspected rather than inferred from object representations."""),
    code("""
import matplotlib.pyplot as plt
import numpy as np
from ocean_taco import draw_queryset
from ocean_taco.render import Points, Resample
from ocean_taco.torch import OceanTACODataset
recipe_draw = draw_queryset(queryset, requested_row_count=1, seed=19, record_path=DRAW_DIR / "cookbook-draw.json")
recipe_dataset = OceanTACODataset(queries=recipe_draw, sources={"l4_sst": Resample((64, 64), .5), "l3_swot": Resample((64, 64), .5)}, catalog_config=config)
recipe_sample = recipe_dataset[0]

def show_grid(axis, record, title):
    # Draw a rendered source, or state its absence rather than inventing pixels.
    data = np.asarray(record["data"])
    if data.shape[0] == 0:
        axis.text(.5, .5, "structurally absent for\\nthis position and date", ha="center", va="center", transform=axis.transAxes)
        axis.set(title=f"{title}: shape {data.shape}", xticks=[], yticks=[])
        return
    image = axis.imshow(data[0], origin="lower", cmap="turbo", aspect="auto")
    axis.set(title=f"{title}: shape {data.shape}", xlabel="x pixel", ylabel="y pixel")
    axis.figure.colorbar(image, ax=axis, shrink=.8)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for axis, token in zip(axes, ("l4_sst", "l3_swot")):
    show_grid(axis, recipe_sample[token], token)
from IPython.display import display
display_figure(fig)
for token in ("l4_sst", "l3_swot"):
    print(token, "data", np.asarray(recipe_sample[token]["data"]).shape, "valid_mask", np.asarray(recipe_sample[token]["valid_mask"]).shape)
print("A source with no asset for this position and date renders with a leading zero dimension; the batch layout is unchanged.")
"""),
    code("""
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
axes[0].imshow(np.asarray(recipe_sample["l3_swot"]["data"])[0], origin="lower", cmap="coolwarm", aspect="auto")
axes[0].set(title="Sparse SWOT values", xlabel="x pixel", ylabel="y pixel")
axes[1].imshow(np.asarray(recipe_sample["l3_swot"]["valid_mask"])[0], origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
axes[1].set(title="Separate validity mask", xlabel="x pixel", ylabel="y pixel")
from IPython.display import display
display_figure(fig)
print("Sparse values and their validity evidence must travel together into the model.")
"""),
    code("""
import matplotlib.pyplot as plt
argo_dataset = OceanTACODataset(queries=recipe_draw, sources={"argo": Points(variable="TEMP", pres_range=(0, 10))}, catalog_config=config)
argo_record = argo_dataset[0]["argo"]
data = np.asarray(argo_record["data"])
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
if data.size:
    axes[0].scatter(np.asarray(argo_record["lon"]), np.asarray(argo_record["lat"]), c=data, s=18, cmap="turbo")
    axes[1].scatter(data, np.asarray(argo_record["pres"]), s=18, color="#dd8a45")
    axes[1].invert_yaxis()
else:
    for axis in axes: axis.text(.5, .5, "No points in this draw", ha="center", va="center", transform=axis.transAxes)
axes[0].set(title="Argo records in geographic space", xlabel="longitude [°]", ylabel="latitude [°]")
axes[1].set(title="Argo temperature profile", xlabel="temperature", ylabel="pressure")
from IPython.display import display
display_figure(fig)
print("Ragged point data are visualised as records, not fabricated into a dense grid.")
"""),
    code("""
import matplotlib.pyplot as plt
fig, axis = plt.subplots(figsize=(8, 2.8))
axis.broken_barh([(-1, 1), (1, 1)], (4, 5), facecolors=["#3182bd", "#dd8a45"])
axis.axvline(0, color="black", lw=1)
axis.set(xlim=(-1.5, 2.5), ylim=(0, 14), yticks=[], xlabel="days relative to anchor", title="Forecast recipe: context and target are distinct windows")
axis.text(-.5, 10, "context", ha="center"); axis.text(1.5, 10, "target", ha="center")
fig.tight_layout()
from IPython.display import display
display_figure(fig)
print("The timeline represents the QueryFilter relation used above, not an inferred observation schedule.")
"""),
    code("""
import matplotlib.pyplot as plt
from ocean_taco.render import Native
native_dataset = OceanTACODataset(queries=recipe_draw, sources={"l3_swot": Native()}, catalog_config=config)
native_record = native_dataset[0]["l3_swot"]
native_data = np.asarray(native_record["data"])
fixed_data = np.asarray(recipe_sample["l3_swot"]["data"])
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
show_grid(axes[0], native_record, "Native")
show_grid(axes[1], recipe_sample["l3_swot"], "Resample (64, 64)")
axes[2].imshow(np.asarray(native_record["valid_mask"])[0], origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
axes[2].set(title="Native-grid valid-data mask", xlabel="x pixel", ylabel="y pixel")
display_figure(fig)
print(f"native shape={native_data.shape}; resampled shape={fixed_data.shape}")
print("Native keeps whatever the source stored, so shapes vary between rows; Resample fixes them at the cost of interpolation.")
"""),
    code("""
import matplotlib.pyplot as plt
fig, axis = plt.subplots(figsize=(8, 3.4))
axis.fill_between([-80, -30], 10, 45, alpha=.35, color="#3182bd", label="regional box")
for segment in wrapped.segments():
    axis.fill_between([segment.lon_min, segment.lon_max], segment.lat_min, segment.lat_max, alpha=.5, color="#dd8a45", label="antimeridian segment")
axis.set(xlim=(-190, 190), ylim=(0, 50), xlabel="longitude [°]", ylabel="latitude [°]", title="Regional and antimeridian selections are explicit geometry")
handles, labels = axis.get_legend_handles_labels()
axis.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
fig.tight_layout()
from IPython.display import display
display_figure(fig)
print("Regional bounds, wrapped bounds, and validity-aware normalisation are all explicit configuration choices.")
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

