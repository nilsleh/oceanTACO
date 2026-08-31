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

Ocean data does not arrive as an image dataset. A single patch of ocean is
observed by instruments with different native resolutions, different coverage,
and different failure modes: a gridded L4 analysis is complete everywhere, a
nadir altimeter samples a line, and SWOT sees a swath that may miss the patch
entirely on the day you asked for. Deciding what a *sample* means under those
conditions is most of the work, and it is a scientific decision rather than a
loader detail.

OceanTACO separates that decision into four stages, and this notebook walks
through all four:

1. A **QuerySet** is a released *population* of patch positions and dates. It
   is published, versioned, and checksummed, so two people who name the same
   QuerySet are talking about the same set of candidate samples.
2. A **filter** narrows that population by geography or time, without drawing
   anything. `QueryFilter(box=...)` states a policy; it does not yet commit to
   rows.
3. A **draw** selects specific rows from the filtered population and writes a
   record of what it selected, so the same rows can be recovered later.
4. A **dataset** renders those rows into tensors, applying a renderer per
   source that decides the grid, the masks, and what absence looks like.

Each stage narrows the one before it, and each records what it selected. That
is what makes a batch traceable: given a tensor, you can name the draw record
it came from, the filter that produced its population, and the released
QuerySet that population was drawn from.

**Prerequisites:** familiarity with PyTorch `Dataset` and `DataLoader`. No
oceanography background is assumed; the domain facts that matter are stated
where they are used."""),
    md("""## 1. Setup

Nothing here is configured. `CatalogConfig()` carries a pinned catalog
revision, and `QuerySet.from_hub` fetches a published QuerySet by patch size
and kind, so the only inputs are the three constants below. Loading the
QuerySet verifies every table against the checksums in its header, which means
a successful load is itself the integrity check."""),
    code(SETUP), code(LOAD),
    md("""## 2. Geography comes before pixels

Two goals compete whenever ocean data is turned into tensors. A model wants a
stable interface: fixed channel shapes, so batches stack and architectures stay
simple. The data wants to keep its native resolution, because resampling a
sparse swath onto a coarse grid destroys exactly the structure the swath was
flown to measure. OceanTACO resolves this by making the choice explicit per
source rather than picking a default, and the chain from geography to pixels
runs in four steps.

### Step 1: the patch is geographic, not pixel-shaped

A patch is specified in kilometres, and `PatchSize.to_degrees` converts that to
a longitude and latitude span at a given centre latitude. The two are not
equal. A degree of latitude is about 111 km everywhere, but a degree of
longitude shrinks by `cos(latitude)` as you move away from the equator, so a
256 km patch spans roughly 2.3° of longitude at the equator and about 4.6° at
60°N. The figure below is that relationship: the same patch in kilometres is a
different geographic window depending on where it sits.

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
one date and region, not a nominal product figure. The L3 rows are the step of
the grid the tracks are stored on, which is much finer than the L4 analyses;
what makes them sparse is that most of that grid is empty on any given day.

`l3_swot` is the case that breaks naive handling. It is not a grid with gaps;
it is a swath that either crosses the patch on that date or does not. When it
does not, there is no array to return with the right shape and some NaNs — the
correct answer is that the source is structurally absent, which §5 covers.

### Step 3: the renderer decides the output grid

`Resample((H, W), support_threshold)` puts a source on a fixed grid.
`Native()` keeps the source's own grid, so shapes vary between rows. Neither is
a default: a renderer is required per source, because silently resampling would
be an unrecorded scientific choice.

`support_threshold` is the parameter that makes resampling honest. When
several native cells fall into one output cell, *support* is the fraction of
that output cell backed by valid source data. Cells below the threshold are
marked invalid rather than filled with an interpolated guess, so a swath edge
stays an edge instead of bleeding into open water. It has no default for the
same reason the renderer has none.

### Step 4: what this means for model design

`Native()` per source means channels no longer share a spatial shape, which
buys fidelity at the cost of needing per-source encoders or an explicit fusion
step. `Resample` everywhere gives uniform channels and pays interpolation. The
cookbook works through both."""),
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

Two QuerySets are published per patch size, and they differ in how their
positions were placed. The training population is placed stochastically, which
gives an unbiased sample of ocean conditions and lets the population grow
without re-planning a grid. The evaluation population is placed systematically
on a fixed grid, which makes coverage uniform and makes a metric computed over
it interpretable as a spatial average rather than a weighted one. The figure
below shows the difference directly: irregular density on the left, a regular
lattice on the right.

The `kind` field records that intent, and it is worth being precise about what
it does not do. **A QuerySet kind is not a leakage proof.** Both published sets
span the same 858 dates, and their positions are drawn from the same ocean, so
a training patch and an evaluation patch can be neighbours in space and
identical in time. Nothing in the released artifacts prevents that, because
preventing it is an experiment-design decision that depends on what your model
is predicting and over what horizon.

In practice: choose an explicit split policy, apply it as a filter to both
populations, and keep the draw records. A temporal guard band is usually the
more defensible axis, because ocean fields are strongly autocorrelated in space
over the scales one patch covers. The `spatio_temporal_query_generation`
notebook works through it."""),
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

The three masks are separate on purpose. `source_valid` says the instrument
did not report a value there; `support_mask` says the renderer could not build
one from what was reported; `valid_mask` is the conjunction and is what a loss
should be masked by. Keeping them apart is what lets you tell a cloud-flagged
pixel from a swath edge after the fact.

**Structural absence.** When a source has no asset for a position and date it
is neither dropped nor zero-filled. The renderer returns its `empty()` form:
`data` with shape `(0, H, W)`, masks zero, coordinates NaN, and
`availability[token] = False`. The leading zero is the signal — the key is
still there and the batch layout is unchanged, so absence is representable
rather than inferred from what is missing. Zero-filling would be worse than
either, since zero is a plausible SST anomaly."""),
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

`collate_ocean_samples` is the collate function `DataLoader` needs, and its
central property is that **availability is collated separately from values**: a
source absent for one batch member neither changes the tensor layout nor
shrinks the batch. Downstream code reads `batch["availability"]` rather than
inferring presence from a shape.

**Fixed versus native shapes.** `Resample` outputs stack directly, since every
sample already shares a grid. `Native()` outputs do not: shapes vary row by
row, and stacking them requires either padding, which invents cells, or
grouping samples that already agree. `ShapeBucketSampler` does the grouping,
batching together rows whose native shapes match. Pass `shuffle=False` when you
need a reproducible bucket ordering; it shuffles within buckets by default.

**Workers.** Catalog resolution happens once, in the parent process, before any
fork. `OceanTACODataset` calls `plan()` at construction to turn queries into
resolved asset locations, so worker processes never open the catalog — they
fetch already-named assets. That is what makes `num_workers > 0` safe here, and
it is also why construction does visible work while `__getitem__` stays cheap.

**No implicit normalisation.** The loader returns decoded values in their
recorded units and neither centres, scales, nor fills them; §7 covers why that
is deliberate."""),
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
    md("""## 7. Normalisation, and what to persist

Normalisation happens here rather than in the loader, and it goes through
`valid_mask` so invalid cells stay NaN. Averaging the raw array instead would
fold in whatever those cells hold and shift the statistics of every subsequent
batch. The statistics belong to the experiment: compute them once over the
training population and hold them fixed, since recomputing per batch leaks
batch composition into the inputs."""),
    code("""
import torch
def normalise_valid(data, valid_mask, mean, std):
    # Normalise only where the mask is true; invalid cells stay NaN rather than
    # becoming a plausible-looking zero.
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
    md("""### Reproducing this batch later

Everything above is recoverable from six things, and none of them is the data
itself:

1. **The QuerySet ID** — `queryset.queryset_id`, which names the released
   population.
2. **The header checksums** — `table_sha256`, which is verified on every load,
   so a successful read is also an integrity proof.
3. **The catalog revision** — pinned in `CatalogConfig`, so the assets behind
   the rows cannot move underneath you.
4. **The filter** — the `QueryFilter` that produced the population.
5. **The draw record** — written to disk by `draw_queryset`, replayable with
   `replay_experiment`, which is what the assertion in §4 checks.
6. **The renderer settings and normalisation statistics** — the shapes,
   `support_threshold`, and the mean and standard deviation used above.

Store those alongside model weights. Storing the rendered tensors instead is
both larger and less useful, because it cannot be re-rendered at a different
resolution or with a different support threshold."""),

])

write("spatio_temporal_query_generation.ipynb", [
    md("""# Spatio-temporal QuerySet intuition

QuerySets factor a stable geographic position table from date-dependent
coverage evidence. They are not generated ad hoc at training time. This
notebook uses the real local evaluation and training populations to show how
selection, coverage, cadence, and replay fit together."""),
    code(SETUP), code(LOAD),
    md("""## 1. Population geography: random training versus systematic evaluation

The two published QuerySets place their positions by different rules, and the
difference is visible before any data is fetched. Training positions are drawn
stochastically, which samples ocean conditions without imposing a lattice and
lets the population be extended later without re-planning. Evaluation positions
sit on a systematic grid, which makes coverage uniform so that a metric
averaged over them is a spatial average rather than one weighted by wherever
the sampler happened to concentrate. Both are plotted below as stored
positions: each point is a patch centre recorded in the release, not an
observation and not a coverage surface."""),
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
    md("""## 2. Null is not zero

The coverage table records what was measured about each position and date, and
it distinguishes two things that are easy to conflate. A **null** means the
evidence was never established for that pair — nobody looked, or the check
could not run. A **zero** means it was established and the answer was none:
SWOT crossed and contributed no valid cells. Coercing null to zero would turn
"unknown" into a confident claim of absence, and any filter built on it would
silently select on measurement effort rather than on the ocean."""),
    code("""
coverage = queryset.coverage
nulls = sum(row["swot_valid_cells"] is None for row in coverage)
zeros = sum(row["swot_valid_cells"] == 0 for row in coverage)
print(f"SWOT evidence: null={nulls} (not measured); zero={zeros} (measured absent)")
print("A coverage filter rejects null evidence; it must not coerce null to zero.")
"""),
    md("""## 3. Box, date, and coverage filters

A `QueryFilter` narrows the population along three independent axes: geography
via `box`, time via `date_start` and `date_end`, and observed evidence via
`coverage`. The coverage requirement is the one worth dwelling on, because it
reads the published fact table rather than the data — asking for at least one
valid SSH cell costs no download and no rendering. That is what makes it usable
as a population-level policy: you can state "only positions where SSH was
actually observed" and see the count before committing to fetch anything."""),
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
    md("""## 4. Recorded draws and temporal guard bands

A draw commits to specific rows and writes a record of that commitment, which
`replay_experiment` can reproduce exactly — the assertion below is the check.
The filter applied here also drops the first two and last two stored dates,
which is a **temporal guard band**: a margin at the split boundary that keeps
a training patch and an evaluation patch from sitting adjacent in time. Ocean
fields are autocorrelated over days, so adjacency at the boundary leaks. The
guard band belongs in the recorded filter rather than in a convention applied
afterwards, because only the recorded version is reproducible."""),
    code("""
from ocean_taco import draw_queryset, replay_experiment
guarded = QueryFilter(box=box, date_start=queryset.dates[2], date_end=queryset.dates[-3], context_start_offset_days=0, context_end_offset_days=0)
draw = draw_queryset(queryset, requested_row_count=min(4, select_queryset(queryset, guarded).count), seed=42,
                     record_path=DRAW_DIR / "query-generation-draw.json", query_filter=guarded)
assert replay_experiment(queryset, DRAW_DIR / "query-generation-draw.json").rows == draw.rows
print(f"drawn={len(draw.rows)}; inclusion_probability={draw.inclusion_probability:.6g}")
print("Guard bands must be part of the recorded split policy, not an after-the-fact convention.")
"""),
    md("""## 5. Cadence and overlap are modelling choices

Neither the published cadence nor the spacing between positions is a claim
about statistical independence. The release stores dates at a fixed interval
and positions on a known grid; whether two of them are independent enough for
your metric depends on what the model predicts and over what horizon. A
forecast at seven days needs a wider guard band than a same-day reconstruction,
and a mesoscale target needs more spatial separation than a basin-scale one.
The shaded band below illustrates a policy over the stored dates; it changes
nothing in the release."""),
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
    md("""## 1. Catalog rows and source tokens

A source token is not a filename. The registry maps each token to the asset
that holds it and the variable to read from that asset, which is why several
tokens can share one file: the GLORYS tokens all resolve to `glorys.nc` and
differ only in which variable they select. The registry also records the
geometry, which decides everything downstream — a `dense_grid` source is
merged and cropped as a field, while a `ragged_points` source like `argo`
keeps individual float positions and is never rasterised on retrieval."""),
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
    md("""## 2. One named tile

The smallest retrieval unit is one asset for one date in one of the eight named
Core regions. Regions are resolved by name rather than by bounding-box query,
because the region set is fixed and immutable while `tacoreader`'s bbox
argument convention has changed across releases. Asking for a named tile is
therefore the most direct thing you can do, and it returns the file's own
variables and dimensions with nothing merged or cropped."""),
    code("""
from ocean_taco.retrieve import load_tile_nc
date = queryset.dates[0][:10]
tile = load_tile_nc(catalog, date, "NORTH_ATLANTIC", "l4_sst", config=config)
print("date", date, "tile sizes", None if tile is None else dict(tile.sizes))
if tile is not None: print("variables", list(tile.data_vars))
"""),
    md("""## 3. Box retrieval and merge

A geographic box usually spans more than one region tile, so `load_bbox_nc`
resolves every intersecting tile, fetches each once, merges them on their
shared coordinates, and crops the result to the box. The returned field keeps
its **native coordinates** — this is retrieval, not rendering, so there is no
target grid and no interpolation. A `None` return means no asset matched, which
is a different statement from an empty field: the latter says the asset existed
and had nothing in the box."""),
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
    md("""## 4. A multi-source closed time range

Requesting several sources over one time range returns a dict keyed by token,
each entry carrying that source's own time axis. The axes differ, and the bar
chart below shows it: the products have different native temporal sampling, and
retrieval does not reconcile them onto a common cadence. Reconciliation is a
modelling decision, so it belongs to the renderer and the context window rather
than to the fetch. Note also that the requested interval is **closed** — both
endpoints are included."""),
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
    md("""## 5. Point data, antimeridians, and edge cases

Three behaviours worth stating explicitly, because each is a place where a
plausible-looking wrong answer is easy to produce.

**Points stay points.** Argo returns float positions and their measurements,
not a gridded field. Rasterising on retrieval would invent structure between
floats that nothing observed.

**The antimeridian is split, not wrapped.** A box crossing 180° is represented
as two explicit segments, so every downstream comparison stays a simple
interval test. A single interval from 170 to −170 would either be empty or
cover the whole globe, depending on which comparison ran first.

**Absence and error are distinguished.** `None` means no catalog match; an
empty point set is a valid result meaning the floats were not there; invalid
coordinates or dates raise `ValueError` rather than returning something
falsy."""),
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
    md("""## The row every recipe uses

Each recipe below renders the same drawn row, so the differences between them
come from the rendering configuration and nothing else. The helper prints the
tensor structure a configuration produces, which is the thing worth comparing:
shapes, masks, and whether the source was available at all."""),
    code("""
import matplotlib.pyplot as plt
import numpy as np
from ocean_taco import GeoBox, QueryFilter, draw_queryset, select_queryset
from ocean_taco.render import Native, Points, Resample, VectorPair
from ocean_taco.torch import OceanTACODataset

recipe_draw = draw_queryset(queryset, requested_row_count=1, seed=29,
                            record_path=DRAW_DIR / "cookbook-draw.json",
                            query_filter=QueryFilter(box=GeoBox(-80, -30, 10, 45)))
row = recipe_draw.rows[0]
print(f"row: ({row['centre_lon']:.2f}, {row['centre_lat']:.2f}) on {row['anchor_time'][:10]}")

def render(sources):
    # Render the shared row under one configuration and report what came back.
    sample = OceanTACODataset(queries=recipe_draw, sources=sources, catalog_config=config)[0]
    for token in sources:
        record, available = sample[token], sample["availability"][token]
        shape = tuple(np.asarray(record["data"]).shape)
        valid = int(np.asarray(record["valid_mask"]).sum())
        print(f"  {token:10s} data={str(shape):18s} available={str(available):5s} valid_cells={valid}")
    return sample

def show_grid(axis, record, title, component=None):
    # Draw a rendered source, or state its absence rather than inventing pixels.
    data = np.asarray(record["data"])
    if data.shape[0] == 0:
        axis.text(.5, .5, "structurally absent for\\nthis position and date", ha="center", va="center", transform=axis.transAxes)
        axis.set(title=f"{title}: shape {data.shape}", xticks=[], yticks=[])
        return
    image = data[0] if component is None else data[0][component]
    drawn = axis.imshow(image, origin="lower", cmap="turbo", aspect="auto")
    axis.set(title=f"{title}: shape {data.shape}", xlabel="x pixel", ylabel="y pixel")
    axis.figure.colorbar(drawn, ax=axis, shrink=.8)
"""),
    md("""## Fixed grids and multimodal fusion

Putting every source on one `Resample` grid is the conventional dense setup:
each source becomes `(T, H, W)` with the same `H` and `W`, so channels
concatenate and batches stack without padding. The cost is interpolation, paid
by every source whose native grid differs from the target.

Fusion is the same recipe with another source added. Adding `l3_swot` to two
complete L4 fields is the interesting case, because SWOT is sparse: it either
crosses this patch on this date or it does not, and the output below reports
which."""),
    code("""
fixed = {"l4_sst": Resample((64, 64), .5), "l4_ssh": Resample((64, 64), .5)}
print("fixed grids:"); fixed_sample = render(fixed)
fusion = {**fixed, "l3_swot": Resample((64, 64), .5)}
print("with SWOT fused in:"); fusion_sample = render(fusion)
"""),
    code("""
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
for axis, token in zip(axes, ("l4_sst", "l4_ssh", "l3_swot")):
    show_grid(axis, fusion_sample[token], token)
display_figure(fig)
print("All three share one model-facing grid; their masks stay separate.")
"""),
    md("""## Vector pairs

An eastward and a northward velocity component are not two independent
sources. They are one vector field, and rendering them separately would let
them disagree: a cell could end up with a valid `u` and an invalid `v`,
producing a direction that no measurement supports.

`VectorPair` renders both components as a unit. The result is `(T, 2, H, W)`
rather than two `(T, H, W)` entries, `valid_mask` describes cells where **both**
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
    print(f"speed max={np.nanmax(speed):.3f} m/s; pair_available={bool(velocity['pair_available'])}")
"""),
    code("""
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
show_grid(axes[0], velocity, "eastward (u)", component=0)
show_grid(axes[1], velocity, "northward (v)", component=1)
data = np.asarray(velocity["data"])
if data.shape[0]:
    speed = np.hypot(data[0][0], data[0][1])
    drawn = axes[2].imshow(speed, origin="lower", cmap="magma", aspect="auto")
    step = max(1, speed.shape[0] // 16)
    grid_y, grid_x = np.mgrid[0:speed.shape[0]:step, 0:speed.shape[1]:step]
    axes[2].quiver(grid_x, grid_y, data[0][0][::step, ::step], data[0][1][::step, ::step], color="white", scale=6)
    axes[2].set(title="speed with direction", xlabel="x pixel", ylabel="y pixel")
    fig.colorbar(drawn, ax=axes[2], shrink=.8, label="m s⁻¹")
else:
    axes[2].text(.5, .5, "pair unavailable", ha="center", va="center", transform=axes[2].transAxes)
display_figure(fig)
print("One shared mask governs both components, so every drawn arrow is supported by both.")
"""),
    md("""## Sparse and dense sources together

Mixing a complete L4 analysis with a sparse L3 swath in one configuration is
the common multimodal case, and the thing to get right is that **a missing
SWOT cell is not an observation of zero**. It is a cell the satellite did not
sample. The masks carry that distinction; the values alone cannot, because zero
is a perfectly plausible sea-level anomaly.

A model that consumes both should therefore read `valid_mask` rather than
testing values against a sentinel, and a loss should be masked by it."""),
    code("""
sparse_dense = {"l4_sst": Resample((64, 64), .5), "l3_swot": Resample((64, 64), .5)}
sparse_sample = render(sparse_dense)
for token in sparse_dense:
    mask = np.asarray(sparse_sample[token]["valid_mask"])
    if mask.size:
        print(f"{token:10s} valid fraction={mask.mean():.3f}")
print("A low valid fraction is evidence about sampling, not about the ocean.")
"""),
    md("""## Argo points

Argo is not a field. Each record is a float profile at its own position, so
`Points` returns ragged records with coordinates and pressures rather than a
grid, and the count varies from row to row. Zero points is a valid result and a
valid batch member: floats are sparse, and most patches on most days contain
none.

`variable` selects which measurement to expose, and `pres_range` limits the
depth band in decibars. Omitting it takes the shallowest usable level per
profile; the recipe below asks for the top 200 dbar so a full profile is
visible."""),
    code("""
from ocean_taco.geobox import PatchSpec, PatchSize
# Argo is sparse against a 256 km patch: most drawn rows contain no floats at
# all, so this recipe names a patch that does, and then measures how common
# that is across the shared draw.
float_patch = PatchSpec(centre_lon=-47.1, centre_lat=21.9, patch_size=PatchSize(PATCH_SIZE_KM, "km"),
                        anchor_time="2025-04-23T00:00:00Z",
                        context_start_offset_days=0, context_end_offset_days=0)
points = {"argo": Points(variable="TEMP", pres_range=(0, 200))}
argo_sample = OceanTACODataset(queries=[float_patch], sources=points, catalog_config=config)[0]
record = argo_sample["argo"]
print("available:", argo_sample["availability"]["argo"])
for key in ("data", "lat", "lon", "pres"):
    if key in record:
        print(f"  {key:6s} shape={tuple(np.asarray(record[key]).shape)}")
print(f"{int(np.asarray(record['data']).size)} point records; the count is per patch and varies.")
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
lon, lat = np.asarray(record.get("lon", [])), np.asarray(record.get("lat", []))
values = np.asarray(record["data"]).reshape(-1)
if lon.size:
    drawn = axes[0].scatter(lon, lat, c=values[:lon.size], s=60, cmap="turbo", edgecolor="black", linewidth=.4)
    fig.colorbar(drawn, ax=axes[0], label="TEMP [°C]")
    pres = np.asarray(record["pres"]).reshape(-1)
    axes[1].scatter(values[:pres.size], pres, s=26, color="#dd8a45")
    axes[1].invert_yaxis()
    axes[1].set(title="The same records as a profile", xlabel="TEMP [°C]", ylabel="pressure [dbar]")
else:
    for axis in axes:
        axis.text(.5, .5, "no Argo profiles in this patch", ha="center", va="center", transform=axis.transAxes)
axes[0].set(title="Argo records at native positions", xlabel="longitude [°]", ylabel="latitude [°]")
display_figure(fig)
print("Points keep their own coordinates and pressures; nothing is rasterised onto a grid.")
"""),
    code("""
# How often does a drawn patch contain floats at all?
survey = draw_queryset(queryset, requested_row_count=20, seed=29,
                       record_path=DRAW_DIR / "argo-survey-draw.json")
survey_dataset = OceanTACODataset(queries=survey, sources=points, catalog_config=config)
counts = [int(np.asarray(survey_dataset[index]["argo"]["data"]).size) for index in range(len(survey.rows))]
print(f"rows with at least one profile: {sum(count > 0 for count in counts)} of {len(counts)}")
print("Zero is the ordinary case at this patch size, and it is a valid batch member.")
"""),
    md("""## Forecasting with two datasets

A forecasting setup needs the context window and the target to be separate
objects, so that a model consuming context cannot reach the target by
accident. The `QueryFilter` states the relation and the lead time; the two
datasets then differ only in their context offsets.

The output below is the check that matters: the two windows must not overlap.
If they do, the model can see its own target."""),
    code("""
# One draw, so both datasets describe the same rows; only the offsets differ.
# Drawing twice would change which rows are eligible and silently compare
# different anchors.
forecast = QueryFilter(relation="forecast", target_lead_days=1,
                       context_start_offset_days=-1, context_end_offset_days=0)
forecast_draw = draw_queryset(queryset, requested_row_count=1, seed=29,
                              record_path=DRAW_DIR / "forecast-draw.json", query_filter=forecast)
source = {"l4_sst": Resample((64, 64), .5)}
lead = forecast.target_lead_days
target_row_maps = [
    {**row, "context_start_offset_days": lead, "context_end_offset_days": lead}
    for row in forecast_draw.rows
]
context_rows = OceanTACODataset(queries=forecast_draw, sources=source, catalog_config=config)
target_rows = OceanTACODataset(queries=target_row_maps, sources=source, catalog_config=config)
context_window = context_rows[0]["query"].context
target_window = target_rows[0]["query"].context
print(f"anchor: {forecast_draw.rows[0]['anchor_time'][:10]}")
print(f"context window: {context_window.start:%Y-%m-%d} .. {context_window.end:%Y-%m-%d}")
print(f"target  window: {target_window.start:%Y-%m-%d} .. {target_window.end:%Y-%m-%d}")
print(f"windows disjoint: {target_window.start > context_window.end}")
"""),
    md("""## Native shapes and bucketing

`Native()` returns the source's own grid, so nothing is interpolated and
nothing is invented — but shapes then vary from row to row, and varying shapes
cannot be stacked. `ShapeBucketSampler` solves that by grouping rows whose
native shapes already agree, so each batch is internally uniform without
padding.

Pass `shuffle=False` for a reproducible bucket ordering; it shuffles within
buckets by default, which makes a printed bucket list seed-dependent."""),
    code("""
native_sample = render({"l3_swot": Native()})
resampled = np.asarray(fusion_sample["l3_swot"]["data"])
native = np.asarray(native_sample["l3_swot"]["data"])
print(f"native shape={native.shape}; resampled shape={resampled.shape}")
print("Native preserves the source grid exactly; Resample fixes the shape at the cost of interpolation.")
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
show_grid(axes[0], native_sample["l3_swot"], "Native")
show_grid(axes[1], fusion_sample["l3_swot"], "Resample (64, 64)")
display_figure(fig)
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

A regional box is an ordinary filter. A box crossing the antimeridian is not:
it is represented as two explicit segments, because a single interval from 170
to −170 is either empty or global depending on which comparison runs first.
Splitting it keeps every downstream test a simple interval comparison."""),
    code("""
regional = GeoBox(-80, -30, 10, 45)
wrapped = GeoBox(170, -170, 10, 30, wraps_antimeridian=True)
print("regional segments:", len(regional.segments()))
for segment in wrapped.segments():
    print("  wrapped segment:", segment.to_dict())
print(f"regional population: {select_queryset(queryset, QueryFilter(box=regional)).count} pairs")
"""),
    md("""## Normalisation

The loader returns decoded values in their recorded units and normalises
nothing, because normalisation is an experiment-level choice that has to be
recorded with the experiment. Apply it through `valid_mask`, so invalid cells
stay NaN rather than becoming a plausible-looking zero, and compute the
statistics once over the training population rather than per batch."""),
    code("""
import torch
def normalise_valid(data, mask, mean, std):
    output = torch.full_like(data, float("nan"))
    output[mask] = (data[mask] - mean) / torch.as_tensor(std, dtype=data.dtype).clamp_min(1e-6)
    return output

# l3_swot is the informative case: its swath covers only part of the patch, so
# the mask actually excludes cells and the NaN behaviour is visible.
values, mask = fusion_sample["l3_swot"]["data"], fusion_sample["l3_swot"]["valid_mask"]
mean, std = values[mask].mean(), values[mask].std()
normalised = normalise_valid(values, mask, mean, std)
print(f"source mean={mean:.4f} std={std:.4f} over {int(mask.sum())} of {mask.numel()} cells")
print(f"normalised mean={normalised[mask].mean():.3e} std={normalised[mask].std():.3f}")
print(f"{int((~mask).sum())} invalid cells, all still NaN: {bool(torch.isnan(normalised[~mask]).all())}")
# Zero-filling before averaging is the failure this guards against: it treats
# every unobserved cell as a measured zero.
filled = torch.nan_to_num(values, nan=0.0)
print(f"masked mean={mean:.4f} vs zero-filled mean={filled.mean():.4f}")
print("Leakage control is separate: disjoint filters, temporal guard bands, and persisted draw records.")
"""),

    md("""## Geometry of the selections used above

The regional box and the antimeridian box drawn together, so the two-segment
representation is visible as geometry rather than only as a printed dict."""),
    code("""
fig, axis = plt.subplots(figsize=(8, 3.4))
axis.fill_between([-80, -30], 10, 45, alpha=.35, color="#3182bd", label="regional box")
for segment in wrapped.segments():
    axis.fill_between([segment.lon_min, segment.lon_max], segment.lat_min, segment.lat_max, alpha=.5, color="#dd8a45", label="antimeridian segment")
axis.set(xlim=(-190, 190), ylim=(0, 50), xlabel="longitude [°]", ylabel="latitude [°]",
         title="Regional and antimeridian selections are explicit geometry")
handles, labels = axis.get_legend_handles_labels()
axis.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
fig.tight_layout()
display_figure(fig)
print("The wrapped box is two rectangles, which is what every downstream interval test sees.")
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

