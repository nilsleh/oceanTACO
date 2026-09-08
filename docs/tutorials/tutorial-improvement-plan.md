# OceanTACO Tutorial Improvements

## Context

The six tutorial notebooks in `docs/tutorials/` need a pass to make them
release-ready. The user is reviewing them one by one and dictating the changes
required for each. This document accumulates those requests into a single
hand-off plan.

**Key constraint:** the `.ipynb` files are generated artifacts. All narrative
and code changes must be made in `scripts/dev/restore_tutorial_notebooks.py`
(1095 lines), then regenerated and re-executed with
`scripts/dev/execute_tutorial_notebooks.py`. Editing a notebook directly is
reverted on the next regeneration.

## Before doing anything: environment and two hazards

These three items are blockers. Skipping them either corrupts the published
docs or fills a quota-limited volume.

### Environment

`HF_HOME` is currently unset and `~/.cache/huggingface` holds no OceanTACO data
at all, so every execution run downloads granules onto `/p/home`, which has a
tight quota. The venv activation script also prepends its own `site-packages`
and shadows the worktree copy of `ocean_taco`, which fails in confusing ways
during the module rename below.

```sh
export HF_HOME=/p/project1/hai_uqmethodbox/nils/hf-home
source <venv>/activate.sh && unset PYTHONPATH
cd /p/project1/hai_uqmethodbox/nils/oceanTACO-pr1
```

The notebooks keep using `QuerySet.from_hub`, so the published docs retain the
honest zero-config install story. A full local mirror exists at
`/p/project1/hai_uqmethodbox/data/new_ssh_dataset_taco_folder/OceanTACO`
(858 date directories) but is deliberately **not** wired into the notebooks.

### Regeneration and execution are one operation

`docs/conf.py` sets `nb_execution_mode = "off"`, so **stored cell outputs are
the published artifact**. The generator writes notebooks with no outputs.
Running the generator and committing before `execute_tutorial_notebooks.py`
finishes therefore publishes blank code cells to the docs site. Never commit
between the two commands.

### Keep the two prior plan documents

`notebook-remediation-plan.md` is the only record of the two hazards above, of
the marker-size diagnosis in §1c, of the cartopy offline-data risk, and of
several known library bugs. `notebook-restoration-plan.md` holds the visual
review acceptance gate. Both are already excluded from the Sphinx build by the
`tutorials/*-plan.md` glob in `docs/conf.py`, so they cost nothing to keep.
Salvage their content before considering deletion.

### Generator layout

| Notebook | `restore_tutorial_notebooks.py` lines |
|---|---|
| `ml_dataset.ipynb` | 70–413 |
| `spatio_temporal_query_generation.ipynb` | 414–544 |
| `data_retrieval_workflows.ipynb` | 545–709 |
| `ml_configuration_cookbook.ipynb` | 710–1030 |
| `plot_hurricane_milton.ipynb` | 1031–1065 |
| `plot_hurricane_milton_cross_product.ipynb` | 1066–end |

Shared preamble: `SETUP` (line 22) and `LOAD` (line 56) string constants are
injected into notebooks; changes there affect several tutorials at once.

**Paper-notebook isolation:** `SETUP` currently provides `display`, but it does
not provide `matplotlib.pyplot`. Keep `SETUP` and `LOAD` unchanged so the two
Milton notebooks retain identical generated source. If the four rewritten
notebooks need a common plotting import, introduce a separate non-paper setup
fragment and use it only there.

Note: the generator line numbers throughout this document refer to the file as
it stands today. They will drift as soon as editing begins, so treat them as
locators for the first pass, not as durable addresses.

The ranges above were verified correct, with one caveat: each stated end line is
the blank separator *after* the closing `])` of that notebook's cell list, so an
insertion at the end line lands outside the list. Insert one line earlier.

Cell format: `md("...")` and `code("...")` (generator lines 19–20) both apply
`textwrap.dedent` then `.strip()`, so triple-quoted bodies must be uniformly
indented or dedent silently no-ops. Because the bodies are Python string
literals, any literal backslash must be doubled.

## Requested changes

### 1. `ml_dataset.ipynb` (generator lines 70–412)

#### 1a. Prose rewrite (applies to the whole notebook)

The writing reads as machine-generated and needs a full polish pass. Ban the
following patterns outright and rewrite every instance:

- **"It is not X, it is Y"** antithesis. Instances include: "Ocean data does not
  arrive as an image dataset"; "it is a scientific decision rather than a loader
  detail"; "It is not a grid with gaps; it is a swath that..."; "absence is
  representable rather than inferred"; "Remote access and a local catalog are
  not two mechanisms; they are the same layout"; "Each point is a stored patch
  centre, not a rendered observation."
- **"what X looks like"** — e.g. "what absence looks like" (§ intro, stage 4).
- **"X decides the Y"** — e.g. "Step 3: the renderer decides the output grid";
  "a renderer per source that decides the grid".
- **Semicolons and em dashes.** Remove all of them; split into sentences.
- The §1 opener **"Nothing here is configured."** Replace the whole paragraph.
- **"one real render"** in the §5 heading. Nonsensical as written (real as
  opposed to what?). Rewrite the heading and section intro.

#### 1b. Terminology

Invented ML-sounding vocabulary that no ML practitioner uses. Verified against
the package: **"leakage", "split policy", "guard band", and "recipe" have zero
hits in `ocean_taco/`** and must be replaced with plain language.

**"Population" goes too**, everywhere in tutorial prose. Replace with plain
wording: the published training and eval sets, or simply "the training set" /
"the eval set". The `kind` field takes values `"training"` and `"eval"`, and
that is what the prose should say.

**Correction to an earlier claim in this plan:** "population" does *not* appear
nowhere in `ocean_taco/`. It appears **14 times**, and several are
reader-visible through `docs/api/*.md` autodoc:

| File | Lines |
|---|---|
| `filter.py` | 56, 287, 304 |
| `manifest.py` | 5, 358 |
| `torch/dataset.py` | 102 |
| `viz/queryset_maps.py` | 5, 84 |
| `sampling/draw.py` | 19, 21, 25, 95 |
| `sampling/publish.py` | 52 |

(`generate_dataset/new_format_ssh_data.py:942` is an unrelated statistics use,
"population std", and must be left alone.)

The other four terms were verified genuinely absent: **"leakage", "split
policy", "guard band", and "recipe" have zero hits in `ocean_taco/`.**

Decide the scope for "population" in package docstrings. The minimum is the
error at `dataset.py:102` below. Doing only that leaves the tutorials saying
"the training set" while the API docs rendered beside them still say
"population", so sweeping the docstrings listed above is the consistent choice.

**Also reword the user-facing error** at `ocean_taco/torch/dataset.py:102`, which
a reader can trigger by passing a QuerySet where a draw is required. It
currently reads:

> "A published QuerySet is a population, not a sample list; pass a QueryDraw or
> experiment_record."

It contains "population", the banned "not X, it is Y" antithesis, and a
semicolon all at once. Reword it to plainly say a QuerySet is the full published
set and that a `QueryDraw` or `experiment_record` is required, so the error
agrees with the tutorials rather than contradicting them.

"Contract" appears only in internal docstrings (`geobox.py`, `temporal.py`),
never in tutorial prose, so nothing to do there.

#### 1c. Figures

- **§3 published-positions scatter** (lines 224–240): the two panels look nearly
  identical at global scale, so the figure fails to make its point. Add a
  zoomed-in North Atlantic view with bounding boxes drawn, so the stochastic vs
  systematic placement difference is actually visible.

  **The zoom is only half the fix.** The root cause is recorded in
  `notebook-remediation-plan.md`: the marker settings (`s=2..3, alpha=.15..4`)
  were tuned against 1488 points and now render 6027 eval / 11001 training
  positions as a solid smear. Retune marker size and alpha alongside the zoom,
  or the panels stay illegible at any scale.

  Also note `CARTOPY_USER_DIR` was dropped in an earlier rewrite, so
  `coastlines()` / `cfeature.LAND` may attempt a network fetch during execution.
  Adding map figures increases the exposure.

- **§5 `plot_ocean_sample` L4 SST render** (lines 310–317): output is blurry.

  **Diagnosed: this is not a plotting bug.** The sample is rendered at
  `Resample((48, 48))` (generator line 300) and then upscaled into a 144-dpi
  figure by `display_figure`. It is blurry because it is genuinely 48x48.

  Two changes are needed, and the second puts package code in scope:

  1. Raise the demo render to `Resample((128, 128), support_threshold=0.5)`.
  2. **Add `cmap` and `interpolation` keyword parameters to
     `plot_ocean_sample`** (`ocean_taco/plot.py:18`, the `imshow` at line 54).
     The helper currently accepts neither, so it always uses matplotlib's
     default `viridis`. Without this the unified colormap decision below cannot
     reach this figure at all. Keep both parameters optional and defaulted to
     current behaviour so existing callers are unaffected.

  While in `plot.py`, note that `time_index` is already bounds-checked before
  indexing (lines 45–46), so the `IndexError` issue recorded in the remediation
  plan appears fixed. Confirm before acting on it.
- **§6 "available records in batch" bar chart** (lines 351–365): remove. It
  conveys nothing.

#### 1d. Structure

- **Notebook section 4, "Filter, draw, and replay at most eight rows"**
  (lines 241–266): remove. Not useful in this notebook. Two dependencies must be
  handled: the notebook's own section 5 builds its dataset from `draw`, and its
  section 7 reproducibility list references "the assertion in §4". The removal
  must keep a draw available for section 5 and fix the dangling cross-reference.
- **Scope gap:** the notebook is titled "to a PyTorch batch" but never shows how
  to build a working DataLoader. It only surfaces difficulties and nuances. Add
  a prominent link to the **repurposed `spatio_temporal_query_generation`
  notebook (§3)**, which is where that material will live. Note: not the
  cookbook, which stays a renderer reference (§2) and does not cover batching.

**Title decision:** retitle this notebook **"From a published QuerySet to a
rendered sample"**. The DataLoader material moves to §3, so retaining a title
that promises a batch would be misleading.

#### 1e. Code hygiene (noticed while reading)

- `import matplotlib.pyplot as plt` is repeated in four cells (lines 178, 225,
  254, 352) and `from IPython.display import display` three times (237, 263,
  362). The counts are right but the justification is only half right:
  **`SETUP` provides `display` but never imports matplotlib.** So the three
  `display` imports are straightforwardly redundant and can go, while removing
  the four `plt` imports breaks the notebook unless `SETUP` gains
  `import matplotlib.pyplot as plt` first. Add it in the non-paper setup
  fragment, then deduplicate only the four rewritten notebooks.

  The same redundant `display` import also appears at generator lines 489, 538,
  577, 627, 655, 697 (and 1056, 1086 in the out-of-scope Milton sections). Line
  627 is an indented, function-scope import.
- §3 prose hardcodes "858 dates" as a literal. Derive it or drop it.

### 2. `ml_configuration_cookbook.ipynb` (generator lines 710–1029)

**Decided scope: this stays an API reference tour**, organized by renderer
(`Resample` / `Native` / `VectorPair` / `Points`), cleaned up rather than
restructured. It does *not* become the use-case or pipeline guide. The working
DataLoader material goes to §3 instead, so the cookbook is not stretched to
carry it. Its forecasting section (lines 904–935) also moves to §3, where it
joins the other spatio-temporal use cases.

Because the organizing principle is unchanged, the changes below are a cleanup
pass, not a rewrite. A short design pass on prose and figures is still worth
doing before editing, but the structure is settled.

#### 2a. Prose

Same ban-list as §1a applies in full. Confirmed instances in this notebook:

- Opening line 713, **"A renderer decides what shape a source becomes"** —
  singled out by the user as especially bad. Rewrite the whole intro paragraph.
- "It is not X, it is Y": "Argo is not a field" (851); "a missing SWOT cell is
  not an observation of zero. It is a cell the satellite did not sample"
  (832–835); "They are one vector field" (791); "A regional box is an ordinary
  filter. A box crossing the antimeridian is not" (971).
- Em dash at line 939; semicolons at 726, 858, 893, 951.
- Line 1009: **"Leakage control"** and **"temporal guard bands"** — invented
  terms, remove (see §1b).
- Line 989: "the training population" — say "the training set" (see §1b).

#### 2b. Terminology: drop "recipe"

No one uses this word for this. It is load-bearing throughout: the title
concept, "The row every recipe uses" (722), "Each recipe below..." (716, 724),
"the recipe below asks for" (858), "this recipe names a patch" (863). Choose a
plain replacement (e.g. configuration / example / setup) and apply it
consistently, including any retitling of the notebook itself.

**Decision:** use **"configuration"** throughout and retitle the notebook
**"ML renderer configuration reference"**. Replace the hard-coded Argo
`PatchSpec` with a deterministic one-row draw using
`CoverageRequirement("argo", "profile_count", 1)`. The cookbook may link to
§4 for coverage semantics, but must not reintroduce the deep-dive explanation.

#### 2c. Figures

- **Colormaps are inconsistent and wrong.** `show_grid` uses `turbo` (759),
  velocity speed uses `magma` (818), Argo scatter uses `turbo` (882). Unify on
  `RdBu_r` across every plot in the notebook. See the colormap decision in
  Cross-cutting work for the `vmin`/`vmax` requirement that comes with it.
- **"Geometry of the selections used above"** (1012–1028): the final section and
  its rectangle plot do not communicate anything useful. Remove or replace.

#### 2d. Settled implementation notes

- §"Argo points" uses the coverage-qualified draw defined in §2b rather than a
  magic patch.
- There is no end-to-end `DataLoader` example here. That is now by design: it
  belongs to §3, not to this notebook.

### 3. `spatio_temporal_query_generation.ipynb` → repurposed (lines 414–543)

**The notebook as it stands is not useful and should be replaced, not patched.**
It duplicates `ml_dataset` §3 (same `from_hub(..., "training")` call, same
two-panel scatter, same narrative, same global-scale illegibility), and its
distinctive material is built on the invented "temporal guard band" / "split
policy" vocabulary that §1b removes.

**This notebook becomes the end-to-end ML notebook.** No separate new notebook
is created. It covers involved query construction *and* the working training
loader in one place. Retitle accordingly (the current "intuition" title goes).

#### 3a. Use cases to drive the rework

1. **Forecasting** — context window plus target at a lead time, with the
   disjoint-window check. Move/absorb the cookbook's thin forecasting section
   (generator lines 904–935).
2. **Spatio-temporal queries that are not forecasting** — retrieve a time
   horizon whose target is the **midpoint** of the window, covering data
   assimilation and interpolation-style setups.
3. **Super-resolution** — sources returned at different resolutions, i.e. a
   coarse input and a fine target for the same patch.
4. **Multi-source sparse + dense** — queries that must satisfy several sources
   at once, e.g. requiring SWOT coverage alongside dense L4 fields.
5. **Differently-resized returned sources** — mixed output shapes across sources
   in one sample.

#### How to build the midpoint / data-assimilation workflow

This must be supported, and it **is** supported today with no library change.
Verified against the code:

- `QueryFilter.relation` is a closed `Literal["same_time", "forecast"]`
  (`ocean_taco/filter.py:70`), where `forecast` requires `target_lead_days > 0`
  (line 86) and `same_time` requires `0` (line 84). So `relation` alone only
  expresses "target simultaneous" or "target strictly in the future".
- The context offsets are **signed ints with no positivity constraint**. Only
  `context_end_offset_days >= context_start_offset_days` is enforced
  (`filter.py:82`, `geobox.py:224`). A symmetric window `(-N, +N)` is therefore
  legal, and `_anchor_domain_valid` (`filter.py:211-215`) correctly requires
  every date across that span to exist in the QuerySet, so anchors near the
  record boundaries are rejected rather than silently truncated.
- `OceanTACODataset` builds each `PatchSpec` by reading
  `context_start_offset_days` / `context_end_offset_days` off the row dict
  (`ocean_taco/torch/dataset.py:30-36`), so per-row overrides work.

**The construction:** use `relation="same_time"` with a symmetric context
window, e.g. `QueryFilter(context_start_offset_days=-N,
context_end_offset_days=+N)`. Draw once. Build the input dataset from that draw
directly, then build the target dataset from row copies with the offsets
overridden to `(0, 0)`, which is the anchor and therefore the midpoint of the
context window. This is the same override pattern the cookbook already uses for
forecasting (generator lines 923–928), so reuse it rather than inventing a new
mechanism.

Draw once and derive both datasets from that single draw. Drawing twice changes
which rows are eligible and silently compares different anchors.

Teach alongside it that `relation="forecast"` covers the strictly-future case,
and that the difference between the two is which offsets the target rows carry.

**One caveat to state plainly in the notebook.** `draw_queryset` and
`replay_experiment` build every row from a single `QueryFilter`
(`sampling/draw.py:124-133`), so all rows in a draw share its offsets. The
override therefore produces a hand-built `Sequence[Mapping]`, which
`OceanTACODataset` accepts (third `__init__` branch, `torch/dataset.py:113-119`)
but which the experiment record no longer describes: `_record` stores only the
filter-level offsets. The *draw* stays replayable, and the derived target rows
are reconstructed in notebook code from that draw. Say so rather than implying
the whole pipeline round-trips.

#### How to build the remaining three use cases

**3. Super-resolution.** Input and target share the anchor and differ only in
renderer shape. Build both datasets from the same draw:

```python
coarse = {"l4_sst": Resample((32, 32), support_threshold=0.5)}
fine   = {"l4_sst": Resample((128, 128), support_threshold=0.5)}
```

Verified as supported: `sources` is a plain `Mapping[str, Renderer]` and each
token gets its own renderer instance (`torch/dataset.py:120-122`), with nothing
cross-checking shapes between tokens.

Watch the upsampling warning at `render/resample.py:104-111`, which fires once
per token when the target exceeds native resolution by more than 2x. It fires
only when **up**sampling, so a coarse-input / fine-target pair that keeps both
shapes at or below native never trips it. Print `native_shape` (preserved in the
payload) and choose shapes against it. If a demo must upsample, say in the prose
that the warning is expected rather than letting it appear unexplained in
committed output.

**4. Multi-source sparse + dense.** Condition the draw so every row is
guaranteed to carry the sparse source alongside the dense fields:

```python
QueryFilter(coverage=(CoverageRequirement(token="swot",
                                          metric="valid_fraction_ocean",
                                          minimum=0.2), ))
```

Metrics are validated per token (`filter.py:25-47`) and are a closed set:
`swot` accepts `valid_cells`, `valid_ocean_cells`, `n_obs_sum`,
`valid_fraction_footprint`, `valid_fraction_ocean`; `ssh` the same minus
`n_obs_sum`; `argo` only `profile_count`. `aggregate` is one of
`"sum" | "mean" | "min"`.

This is the honest replacement for the cookbook's hardcoded Argo patch (§2d):
instead of naming coordinates that happen to contain floats, state the coverage
requirement and let the filter find rows. Link to §4 for the full explanation
rather than re-teaching coverage here.

**5. Differently-resized returned sources.** Mixed output shapes in one sample,
e.g. `{"l4_sst": Resample((64, 64), 0.5), "l3_swot": Native()}`. Collation is
per-token (`collate_ocean_samples` loops tokens independently), so this collates
cleanly into differently-shaped stacks. This is where `ShapeBucketSampler`
earns its place, since `Native()` output shapes vary per row.

Note `ShapeBucketSampler` **shuffles by default** (`sampler.py:55`), so pass
`shuffle=False` or a fixed seed for reproducible committed output. Building it
calls `native_shapes`, documented as a deliberate O(N) rendering pass in the
parent process, so it is not free on a large draw.

#### 3b. Material to carry over

Both the null-vs-zero material (lines 444–459) and coverage-conditioned
selection (`CoverageRequirement`, lines 460–491) **move to §4**, which is now the
QuerySet/filter deep-dive. This notebook uses coverage filters in its
multi-source use case but links to §4 for the explanation instead of teaching
them again.

#### 3c. Material to drop

- **§1** entirely (lines 422–443): duplicates `ml_dataset` §3.
- **§4/§5 guard-band and "split policy" framing** (lines 493–541), including the
  plot legend "example temporal guard band" and title "Cadence and guard bands
  belong to the split policy". If temporal separation between splits is still
  worth covering, state it in plain language.
- Em dashes at 465, 496; semicolons at 457, 458, 474, 484, 509, 520, 540.

#### 3d. Must also include: the working training pipeline

- A real `DataLoader` that runs, with `num_workers > 0`.
- Batching under both regimes: stacked `Resample` outputs, and `Native()`
  outputs via `ShapeBucketSampler`.
- Normalisation statistics computed once over the training split, held fixed.
- A minimal model step and a masked loss, so mask semantics are demonstrated
  rather than described.

**Three API details that will otherwise be got wrong.** All verified:

1. **`collate_ocean_samples` defaults to `native="ragged"`**, which returns an
   *uncollated* `{"items": [...]}` list rather than tensors
   (`torch/dataset.py:604-637`). **`native_pad_collate` is the DataLoader-ready
   wrapper.** A notebook that passes the default as `collate_fn` and then
   indexes tensors will not work.
2. **`num_workers > 0` needs `worker_init_fn=seed_ocean_taco_worker`.** It is
   exported from `ocean_taco/torch/` for exactly this and reseeds Python/NumPy
   plus resets the shipped loader per worker. Worker safety is otherwise
   deliberately engineered (`__getstate__` nulls backends, a fork guard drops
   parent catalog state), so no file handles cross the fork.
3. **There is no single mask.** A collated batch carries `valid_mask`,
   `source_valid`, `support_mask`, `time_mask`, `ocean_mask`, plus a top-level
   `availability` dict. The semantics are clean and worth stating once:

   > `valid_mask = source_valid & support_mask`, further ANDed with the ocean
   > mask when one is supplied.

   The split exists so a reader can tell *why* a cell is invalid: no source data
   (`source_valid`), insufficient interpolation support (`support_mask`), or
   land (`ocean_mask`). Drive the loss from `valid_mask`; use the others to
   explain it. For `VectorPair`, `pair_available` is the sample-level Boolean
   and `valid_mask` covers cells where both components have support.

Reuse: `OceanTACODataset`, `native_pad_collate`, `ShapeBucketSampler`,
`seed_ocean_taco_worker` (all `ocean_taco/torch/`), `draw_queryset` /
`replay_experiment`, and the `SETUP` / `LOAD` generator constants.
`ml_dataset` §1d links here.

### 4. `data_retrieval_workflows.ipynb` → QuerySet & filter deep-dive (lines 545–708)

**New role:** the in-depth notebook for interacting with the QuerySet — filters,
selection, and visualizing what a selection actually contains. The existing
retrieval material is **kept and reframed**, not dropped: filters and selection
become the body of the notebook, and the direct retrieval API is presented after
it as the lower-level path for working outside the QuerySet flow. Retitle to
match the new emphasis.

#### 4a. Add: QuerySet and filter material (the new body)

- `QueryFilter` across its axes: `box`, `date_start` / `date_end`, `coverage`.
- `select_queryset` and reading counts before committing to any fetch.
- **Coverage filtering lives here** (moved from §3): `CoverageRequirement`, and
  the null-vs-zero distinction salvaged from
  `spatio_temporal_query_generation` §2–§3 (generator lines 444–491). Retitle
  "Null is not zero" (banned antithesis pattern). §3 links here for the
  explanation rather than teaching it again.
- Visualization of selections: where the filtered rows sit geographically and
  temporally.

#### 4b. Keep, reframed: direct retrieval

`load_tile_nc` (§2), `load_bbox_nc` (§3), `load_multisource_time_series_nc`
(§4), and the registry/token material (§1) all stay, positioned as the
lower-level API. The closed-interval note and the `None` vs empty vs
`ValueError` distinctions are worth keeping.

#### 4c. Prose

Ban-list from §1a. Confirmed instances:
- "A source token is not a filename" (555); "this is retrieval, not rendering"
  (601); "it never fabricates a dense field" (551); "which is a different
  statement from an empty field" (603); "Points stay points... not a gridded
  field" (664); "The antimeridian is split, not wrapped" (668).
- "X decides the Y": "The registry also records the geometry, which decides
  everything downstream" (558).
- Em dashes at 559, 601, 638; semicolons at 653, 682, 699.

#### 4d. Figures

- **`cmap="turbo"` at line 622** — apply the same unified `RdBu_r` decision from
  §2c, including its shared `vmin`/`vmax` requirement.
- **§1 modality barh chart** (569–580): plots a boolean as bar length, giving
  four bars of length 0 or 1. One bit per token. Replace with a table or drop.
- **§5 antimeridian panel** (693–695): duplicates the cookbook plot already
  slated for removal in §2c. Keep at most one instance across the whole tutorial
  set, and this notebook is the better home for it.

### 5 & 6. The two Hurricane Milton notebooks — **out of scope**

`plot_hurricane_milton.ipynb` (lines 1031–1064) and
`plot_hurricane_milton_cross_product.ipynb` (lines 1066–1094) reproduce paper
figures and are considered fine as they are. Their thin narrative is appropriate
to that purpose, and their figures must **not** be restyled.

Consequences for the rest of the plan:

- **The unified colormap decision (§2c) does not apply to these two.** Their
  plotting lives in `ocean_taco/viz/paper/plot_hurricane_milton.py` and
  `plot_hurricane_milton_cross_product.py`, which must stay untouched so the
  figures keep matching the paper. Confine colormap changes to the tutorial
  generator and any non-paper `viz` helpers.
- Two minor prose instances of the banned patterns exist here ("a missing source
  is shown as missing rather than hidden", line 1036; "no nearest-grid shortcut
  and never fills missing L4 values with zero", line 1092). Leave them alone
  unless the user says otherwise, since these notebooks are not being reworked.

## Resulting tutorial architecture

The rework reassigns roles across the four non-paper notebooks. Filenames stay
the same (so `execute_tutorial_notebooks.py`'s hardcoded list needs no change),
but titles and content change substantially.

| Notebook | New role |
|---|---|
| `ml_dataset` | General overview and entry point. Concepts and the four stages, linking out to the others. |
| `data_retrieval_workflows` | QuerySet and filter deep-dive: filters, selection, coverage, visualization; direct retrieval API as the lower-level path. |
| `spatio_temporal_query_generation` | ML use cases (forecasting, midpoint/assimilation, super-resolution, multi-source) plus the working training loader. |
| `ml_configuration_cookbook` | Renderer API reference, organized by renderer. |
| both `plot_hurricane_milton*` | Unchanged paper-figure reproductions. |

Cross-links to establish: `ml_dataset` → the other three; `spatio_temporal_query_generation` → `data_retrieval_workflows` for coverage-filter explanation.

## Module rename: `manifest.py` → `queryset.py`

"Manifest" is out-of-touch jargon for a module whose job is to define
`QuerySet` (line 357) and `PatchSet` (line 811). Rename it to `queryset.py`,
after the primary class and the concept the library is built around.

**Also fix the tutorials to stop reaching into it.** `ocean_taco/__init__.py:9`
already re-exports both classes, so `from ocean_taco import QuerySet` is the
correct public import. The tutorial generator currently does
`from ocean_taco.manifest import QuerySet` at lines **214** and **435**, using a
private path for an already-public class. That is a bug independent of the
rename, and fixing it removes the word from reader-visible code.

Scope: rename the module, update all import sites, fix the two tutorial imports
to use the public re-export. **Verified count: 20 import statements across 16
`.py` files**, not 24 (the higher figure counts two notebooks and two markdown
plan documents alongside real code). Breakdown: 7 package-internal
(`__init__.py`, `filter.py`, `torch/dataset.py`, `viz/queryset_maps.py`,
`sampling/{grids,publish,draw}.py`), 7 in tests across 4 files, 6 in scripts
across 5 files.

Both tutorial imports are also **redundant regardless of the rename**: `LOAD`
already does `from ocean_taco import CatalogConfig, QuerySet`, so `QuerySet` is
in the kernel namespace before either cell runs.

This lands as **one combined change** together with the notebook rewrites, not
as a separate commit. Note the working tree currently carries an unrelated SWOT
phase-figures thread (`README.md`, `docs/dataset_description.md`,
`ocean_taco/registry.py`, and four PNGs under `docs/images/`). Stage
selectively, or that work gets swept into the same commit.

**Two traps for the executing agent:**

1. **`scripts/release/verify_wheel.py:18`** lists `"ocean_taco/manifest.py"` as a
   hardcoded string in its `_REQUIRED` set. It is not an import, so an
   import-based search will miss it, and a stale entry fails the release check.
2. **`tests/test_package_contract.py` uses a `manifest={}` keyword argument**
   (lines 84, 93, 305, 347, 478, 640, 668) that is unrelated to the module name.
   A blind find-and-replace over the word will break these tests.

Import sites to update span `ocean_taco/` (`__init__.py`, `catalog.py`,
`filter.py`, `geobox.py`, `sampling/*`, `torch/dataset.py`, `viz/queryset_maps.py`,
`benchmarks/*`), `scripts/release/*`, and `tests/*`.

**Not affected:** `docs/api/*.md` reference only public paths
(`.. autoclass:: ocean_taco.QuerySet`), confirming the module was always
internal. The two historical plan docs in `docs/tutorials/` mention the old path
but are being ignored per the user's instruction.

Verify with `python -c "import ocean_taco; ocean_taco.QuerySet"`, the full test
suite, and `python scripts/release/verify_wheel.py` if a wheel is available.

## Cross-cutting work

1. **Prose ban-list** (§1a) applies to notebooks 1–4 uniformly: no "not X but Y"
   antithesis, no "what X looks like", no "X decides the Y", no semicolons, no
   em dashes. Consider invoking the `prose-writing-style` skill during execution.
2. **Terminology** (§1b): remove "leakage", "split policy", "guard band",
   "recipe", and "population" from tutorial prose. Say "the training set" /
   "the eval set", matching the `"training"` and `"eval"` values the `kind`
   field actually takes. The first four have zero hits in `ocean_taco/`;
   "population" has 14, so §1b also asks for a decision on the package
   docstrings that render into `docs/api/`.
3. **Colormaps**: **`RdBu_r`** across notebooks 1–4. Replaces the current mix of
   `turbo` (`show_grid`, Argo scatter, box retrieval) and `magma` (velocity
   speed). These four sites (generator lines 622, 759, 818, 882) are the only
   explicit cmaps in the generator. Does **not** touch `ocean_taco/viz/paper/`.

   **Set an explicit shared `vmin`/`vmax` per source.** `RdBu_r` is a diverging
   map, so with autoscaling it centres white at each panel's own data midpoint.
   On sequential fields such as SST, speed and Argo temperature that makes every
   patch read as half red / half blue with the split moving between figures.
   Fixing `vmin`/`vmax` per source keeps the colour mapping stable and
   comparable across panels and notebooks.

   Reaching the §5 L4 SST figure additionally requires the `cmap` parameter
   added to `plot_ocean_sample` in §1c.
4. **Import hygiene**: `IPython.display.display` is re-imported in many cells
   despite being available from `SETUP`. `matplotlib.pyplot` is re-imported too,
   but is **not** in `SETUP`, so add it there before deduplicating. See §1e.

## Verification

All work happens in `scripts/dev/restore_tutorial_notebooks.py`. Never edit the
`.ipynb` files directly.

```sh
export HF_HOME=/p/project1/hai_uqmethodbox/nils/hf-home
source <venv>/activate.sh && unset PYTHONPATH
cd /p/project1/hai_uqmethodbox/nils/oceanTACO-pr1

# The rename must be green first: the generator imports from the package.
python -c "import ocean_taco; ocean_taco.QuerySet; ocean_taco.PatchSet"
python -m pytest tests/

rm -f docs/tutorials/draws/*.json                     # clear stale draw records
python scripts/dev/restore_tutorial_notebooks.py && \
python scripts/dev/execute_tutorial_notebooks.py      # ONE operation, no commit between
```

The `&&` is not cosmetic. The generator writes notebooks with no outputs and
`nb_execution_mode = "off"` means stored outputs are what the docs publish, so a
commit landing between the two steps ships blank code cells.

Execution is network-bound against the pinned Hub revision, with a 1800 s
per-cell timeout across six notebooks run serially. Budget accordingly, and
expect the first run after setting `HF_HOME` to download granules afresh.

**About `docs/tutorials/draws/`:** five JSON draw records are written at
execution time into that directory (`DRAW_DIR`, resolved relative to the
notebook's own directory). They are untracked, so this is runtime hygiene rather
than a commit concern, but §1d removes the section that writes
`ml-dataset-draw.json` and §3 rewrites the owner of `query-generation-draw.json`.
Clear the directory before a verification run so stale records cannot mask a
missing draw.

`execute_tutorial_notebooks.py` runs with `allow_errors=True`, writes partial
output and a traceback into notebook metadata on failure, prints a
`DONE {...}` JSON line per notebook and a `SUMMARY n/6` line, and exits non-zero
if any cell errored. Its `NOTEBOOKS` list (lines 21–28) needs no change, since
filenames are unchanged.

Checks beyond a clean exit:

1. **Figures render and are legible.** Specifically confirm the reworked §3
   scatter (zoomed North Atlantic with boxes) actually shows the
   stochastic-vs-systematic difference, and that the L4 SST render is no longer
   blurry.
2. **The training loop in `spatio_temporal_query_generation` actually runs**,
   including `num_workers > 0` and the masked loss, rather than only being
   described.
3. **Milton notebooks are byte-comparable in figure content** to before the
   change, confirming the paper figures were not restyled.
4. **Grep the generator for regressions** after the prose pass:
   ```sh
   grep -n "—\|;\|recipe\|guard band\|leakage\|split policy\|population\|manifest\|turbo\|magma" \
     scripts/dev/restore_tutorial_notebooks.py
   ```
   Remaining hits should fall only in the Milton sections, which are out of
   scope. Note the semicolon match will also hit legitimate Python, so read the
   results rather than counting them.
5. **Every code cell has a non-`None`, monotonically increasing
   `execution_count`.** This is the check that caught stale-output drift last
   time, and `allow_errors=True` means a clean exit alone does not prove it.
6. **Docs build**: `sphinx-build -W` (warnings as errors), which will catch
   cross-references broken by section renumbering. `docs/conf.py` wires the
   notebooks through `myst_nb`, and all three `*-plan.md` files are already
   excluded from the build.
7. `docs/tutorials/index.md` needs three specific fixes, verified:
   - It **violates this plan's own ban-list**: "Remote access and a local
     catalog are not two mechanisms; they are the same layout" (quoted as a §1a
     example without noticing it lives here, not in the generator), plus
     "decides per catalog row".
   - Its **toctree order contradicts the new architecture**, currently reading
     `ml_dataset, ml_configuration_cookbook, spatio_temporal_query_generation,
     data_retrieval_workflows`, which places the renderer reference second.
     Reorder to match the role table above.
   - It claims every notebook fetches `QuerySet.from_hub(256, "eval")`, but the
     generator also calls `from_hub(PATCH_SIZE_KM, "training")`. Already wrong
     today, independent of this rework.

   The "Each notebook builds its own `CatalogConfig()`" sentence is accurate and
   only needs revisiting if setup changes. Note `index.md` does *not* describe
   the notebooks one by one, so that part of the update is smaller than it
   looks.

## Open items for the executing agent

- The cookbook (§2) warrants a short design pass on prose and figures before
  editing, though its structure is settled as a renderer reference.
- §1's removal of the draw/replay section must keep a `QueryDraw` available for
  the dataset cell, because `OceanTACODataset` rejects a raw QuerySet
  (`ocean_taco/torch/dataset.py:102`).
- §3 is the largest single piece of work: it is a rewrite, not an edit, and it
  carries five use cases plus a working training loop. Consider landing it in
  its own pass even though the change as a whole is one commit.
- **Resolved: keep both prior plan documents.** See the blockers section at the
  top. They hold operational context recorded nowhere else and are already
  excluded from the Sphinx build. Their stale `ocean_taco/manifest.py`
  references are a two-line fix, not grounds for deletion. One conflict worth
  recording: `notebook-restoration-plan.md` §6 demands reworking the Hurricane
  notebooks, which `notebook-remediation-plan.md` investigated and dismissed as
  already satisfied. This plan follows the latter.

- **Prose acceptance criteria.** The ban-list says what *not* to write and gives
  no positive target. Restore the standard from `notebook-restoration-plan.md`:
  *visual review is an explicit acceptance gate; successful Python execution is
  not sufficient.* Exported originals for comparison live at
  `/p/project1/hai_uqmethodbox/nils/oceanTACO/docs/tutorials` (the sibling main
  checkout, since this directory is a linked worktree). Invoking the
  `prose-writing-style` skill during execution is worth doing.

- **Three items in this plan are already done** and should not consume time:
  `docs/tutorials/execution-report.local.json` no longer exists, both
  `scripts/dev/` scripts are now tracked, and the cookbook generator/disk drift
  is reconciled at 29 cells.

## Suggested order of work

1. Environment setup and the blockers at the top of this document.
2. Module rename, with `pytest tests/` green before anything else.
3. Notebooks 1, 2 and 4: prose, terminology, figures, colormaps.
4. Notebook 3 last, as its own pass, now that its five use cases are specified.
5. `index.md`, then one atomic regenerate-and-execute run.

## Decisions confirmed after review

This section resolves the conditional wording above. It takes precedence where
an earlier section says “choose”, “confirm”, or “decide”.

### Titles, navigation, and paper scope

- Retitle the four non-paper notebooks: **From a published QuerySet to a
  rendered sample**, **QuerySet filters and native retrieval**,
  **End-to-end spatio-temporal ML workflows**, and **ML renderer configuration
  reference**.
- In `index.md`, order the tutorials: overview, QuerySet/filter deep-dive, ML
  workflows, renderer reference, then the two paper reproductions.
- Keep the two older plan documents as unlinked historical records, with a
  short superseded notice if they are edited for stale imports.
- Do not alter the Milton notebook definitions, their paper plotting helpers,
  or their styling. Do not add imports to shared `SETUP` or `LOAD`; use a
  non-paper setup fragment for `matplotlib.pyplot` if needed. Verification
  checks unchanged source, not a new pixel-hash requirement.

### Concrete end-to-end ML examples

Every one of the five §3 examples must draw once, derive all paired datasets
from that draw, create a `DataLoader(num_workers=2)`, execute a model forward
pass, and calculate a mask-aware loss.

1. Forecasting uses L4 SST context offsets `(-1, 0)` and an L4 SST target at
   `(1, 1)`, with a disjoint-window assertion.
2. Midpoint assimilation uses sparse SWOT context offsets `(-1, 1)` and a dense
   L4 SSH target at `(0, 0)`. Include the sparse observation mask as an input
   channel.
3. Super-resolution uses L4 SSH at `32×32` as input and coverage-qualified
   `l3_swot` SSH anomaly at `96×96` as target. Explain that the products are
   related but distinct, use an upsampling model, and restrict loss to the SWOT
   target `valid_mask`.
4. Multi-source sparse+dense uses L4 SST, L4 SSH, and coverage-qualified SWOT
   at a common fixed shape. Replace NaNs only for model input and concatenate
   the corresponding validity masks so missing observations remain explicit.
5. Mixed fixed output sizes use L4 SST at `48×48`, L4 SSH at `64×64`, and SWOT
   at `96×96`. Encode each source independently, adaptively pool its features,
   then fuse features rather than concatenating spatial tensors.

The separate native-grid example uses `native_shapes`,
`ShapeBucketSampler(batch_size=2, shuffle=False)`, and `native_pad_collate`.
Choose and record a small candidate-draw seed that yields a bucket with at
least two rows. Assert equal native shapes and an all-false
`spatial_padding_mask` in every emitted batch. This is the deep-learning-safe
way to stack native grids without inventing padded pixels.

For normalisation, include reusable streaming masked-statistics code and run it
once over a deterministic 16-row draw from the published training set. Store
per-token mean and standard deviation and reuse them for all tutorial training
steps. State that production replaces this small demonstration draw with the
complete training selection.

### Renderer and terminology decisions

- Replace “recipe” with **“configuration”** everywhere and use the renderer
  reference title above.
- Replace the cookbook’s magic Argo `PatchSpec` with a one-row deterministic
  draw using `CoverageRequirement("argo", "profile_count", 1)`.
- Use `RdBu_r` for every scalar field plot in notebooks 1–4. Use explicit
  masked-data limits: symmetric bounds for signed SSH and velocity components,
  documented quantile limits for SST, Argo temperature, and speed, and units on
  every colorbar. Set the L4 SST artist to `RdBu_r` with nearest interpolation.
- Make the train/eval placement comparison a global-plus-North-Atlantic
  four-panel figure. Draw `GeoBox(-80, -30, 10, 45)` on the global panels and
  use the same bounds for both zoom panels.
- Remove the availability bar chart and cookbook geometry section. Replace the
  retrieval modality bar chart with a table. Keep the antimeridian figure only
  in the retrieval notebook.
- Apply the prose and terminology pass to `index.md` and reader-facing package
  docstrings, including the raw-QuerySet error. Leave unrelated statistical
  uses of “population” unchanged.

### Breaking module rename and acceptance criteria

- Rename `manifest.py` to `queryset.py` as a breaking change. Do not leave a
  compatibility shim. Update actual imports, generated tutorial imports, and
  the wheel inventory, while preserving unrelated `manifest={}` fields and
  benchmark-manifest terminology.
- Before regeneration, pass package imports and the full test suite. Then run
  regeneration and notebook execution as one operation, build docs with
  warnings as errors, and review the stored figures and prose manually.
- Treat the final implementation as complete only when every code cell executes
  cleanly, the multi-worker and native-bucket assertions run, all five ML paths
  complete their forward/loss step, and the non-paper tutorials meet the visual
  review gate.
