# OceanTACO Tutorial Improvements, round 2

## Purpose

`tutorial-improvement-plan.md` (round 1) has been implemented in full and
verified. This document collects the **second** review pass: the user is
inspecting the six notebooks one by one and dictating final fixes, which are
recorded here as they arrive and ticked off as they land.

Round 1 is history. Do not re-litigate it here. Record only what the current
review asks for, plus anything the round-1 verification left open.

The filename must **end in** `-plan.md` for `docs/conf.py:34`
(`tutorials/*-plan.md`) to exclude it from the Sphinx build. That is why this
is `tutorial-improvement-v2-plan.md` and not `...-plan-v2.md`, which the glob
does **not** match and which would publish this working document as a page.

## Status board

One row per notebook. `Reviewed` flips when the user has inspected that
notebook and dictated their findings; `Fixes` counts the open items in the
per-notebook sections below.

| # | Notebook | Title | Reviewed | Open fixes |
|---|---|---|---|---|
| 1 | `ml_dataset` | From a published QuerySet to a rendered sample | 2026-09-02 | — (X1, M1, M2 done) |
| 2 | `data_retrieval_workflows` | QuerySet selection and native-coordinate retrieval | 2026-09-02 | D1 (X1 folded in) |
| 3 | `spatio_temporal_query_generation` | ML use cases and the training loader | 2026-09-02 | — (Q0, Q1, Q2, X1 done) |
| 4 | `ml_configuration_cookbook` | ML renderer configuration reference | 2026-09-02 | — (C1 done) |
| 5 | `plot_hurricane_milton` | Hurricane Milton: wind and SSH across products | out of scope | — |
| 6 | `plot_hurricane_milton_cross_product` | Hurricane Milton: SSH cross-product comparison | out of scope | — |

Notebooks 5 and 6 reproduce paper figures. They are out of scope unless the
user says otherwise; see "Constraints that carry over".

## Baseline as of 2026-09-02

Measured from the working tree, not assumed. Re-measure before trusting these
after any regeneration.

| Notebook | cells | code | md words | figures | exec counts clean |
|---|---|---|---|---|---|
| `ml_dataset` | 20 | 11 | 2181 | 3 | yes |
| `data_retrieval_workflows` | 26 | 16 | 1140 | 4 | yes |
| `spatio_temporal_query_generation` | 24 | 14 | 1694 | 3 | yes |
| `ml_configuration_cookbook` | 26 | 17 | 1109 | 4 | yes |
| `plot_hurricane_milton` | 6 | 4 | 54 | 1 | yes |
| `plot_hurricane_milton_cross_product` | 4 | 3 | 43 | 1 | yes |

Verified green at this baseline:

- 90 tests passed, 7 skipped, 1 deselected
- 6/6 notebooks execute cleanly; every code cell has a non-`None`,
  monotonically increasing `execution_count`
- banned-term grep returns 0 across generator, notebooks and `index.md`
- both Milton notebooks' **cell source is byte-identical to `HEAD`**
- `ocean_taco/viz/paper/` shows 0 changed files
- `import ocean_taco.manifest` raises `ModuleNotFoundError`
- docs build succeeds with 1 warning, which belongs to the unrelated SWOT
  thread (`docs/dataset_description.md:13`, missing `swot-mission-phases`
  xref target), not to this work

The change is **staged but not committed** (26 files). The SWOT phase-figures
thread is deliberately left unstaged.

## Environment

The scratchpad is wiped between sessions. Recreate `env.sh`:

```sh
export HF_HOME=/p/project1/hai_uqmethodbox/nils/hf-home
source /p/project1/hai_uqmethodbox/nils/sc_venv_template/activate.sh
cd /p/project1/hai_uqmethodbox/nils/oceanTACO-pr1
```

Two corrections to round 1's stated environment, both learned the hard way:

- **Do not `unset PYTHONPATH`.** Round 1's plan said to. It breaks the
  environment: the venv `site-packages` is on `PYTHONPATH` and carries
  `nbformat`/`nbclient`.
- **Source `activate.sh`; never call the venv `python` directly.** The
  activation script sources `modules.sh`, without which every interpreter call
  dies with `libpython3.12.so.1.0: cannot open shared object file`.

`HF_HOME` must point at project space because `/p/home` has a tight quota.

Packages added to the venv to make execution and the docs build work, none of
which were present originally: `nbformat`, `nbclient`, `cartopy`, `h5netcdf`,
the Sphinx extensions (`sphinx_copybutton`, `sphinx_autodoc_typehints`), plus
`pip install -e . --no-deps --no-build-isolation` so the Jupyter kernel can
import `ocean_taco` at all.

## Constraints that carry over

These held through round 1 and still bind. Breaking any of them corrupts the
published docs or the paper figures.

1. **Never edit `.ipynb` files directly.** They are generated.
   `scripts/dev/restore_tutorial_notebooks.py` (1648 lines) is the source of
   truth; direct edits are reverted on the next generator run.
2. **Regeneration and execution are one operation.** `docs/conf.py` sets
   `nb_execution_mode = "off"`, so stored outputs *are* the published artifact,
   and the generator writes notebooks with no outputs. Committing between the
   generator and `execute_tutorial_notebooks.py` publishes blank code cells.
3. **`SETUP` and `LOAD` stay byte-identical.** That is what keeps the two
   Milton notebooks' generated source identical to the reviewed version.
   Round-1 plotting helpers went into a separate `PLOTTING` fragment used only
   by the four rewritten notebooks, precisely so this invariant holds. Put any
   new shared helper in `PLOTTING`, never in `SETUP`.
4. **`ocean_taco/viz/paper/` stays untouched**, and Milton figures are not
   restyled, so they keep matching the paper.
5. **Stage selectively.** The working tree carries an unrelated SWOT
   phase-figures thread: `README.md`, `docs/dataset_description.md`,
   `ocean_taco/registry.py`, four `docs/images/swot_phase_*.png`, and
   `scripts/dev/swot_phase_figures.py`. Do not sweep it into this commit.
6. **Prose ban-list**, notebooks 1-4: no "It is not X, it is Y" antithesis, no
   "what X looks like", no "X decides the Y", no semicolons, no em dashes.
7. **Terminology**: "leakage", "split policy", "guard band", "recipe" and
   "population" stay out of tutorial prose and out of docstrings that render
   into `docs/api/`. Say "the training set" / "the eval set", matching the
   `kind` field values `"training"` / `"eval"`.
   `generate_dataset/new_format_ssh_data.py:942` ("population std") is an
   unrelated statistics use and is left alone.
8. **Keep the prior plan documents** (`notebook-remediation-plan.md`,
   `notebook-restoration-plan.md`, `tutorial-improvement-plan.md`). They are
   the only record of several hazards and known library bugs.

## Generator layout, current

Line numbers moved substantially in round 1. These are current.

| Fragment / notebook | `restore_tutorial_notebooks.py` |
|---|---|
| `SETUP` | 22 |
| `LOAD` | 56 |
| `PLOTTING` (four rewritten notebooks only) | 69 |
| `def write(name, cells)` | 100 |
| `ml_dataset.ipynb` | 104 |
| `spatio_temporal_query_generation.ipynb` | 486 |
| `data_retrieval_workflows.ipynb` | 942 |
| `ml_configuration_cookbook.ipynb` | 1250 |
| `plot_hurricane_milton.ipynb` | 1584 |
| `plot_hurricane_milton_cross_product.ipynb` | 1619 |

Shared colour handling lives in `PLOTTING` as `CMAP = "RdBu_r"`, a
`COLOR_RANGE` dict of per-source `(vmin, vmax)` pairs measured over drawn North
Atlantic rows, and a `color_limits(key)` helper that falls back to autoscaling
for unknown keys. Two entries exist for velocity on purpose: `"velocity"` is
signed and symmetric about zero for u/v components, `"speed"` starts at zero
because it is a magnitude.

## Requested fixes

One section per notebook, filled in as the user dictates. Each item gets an
ID, the verbatim request, the diagnosis, the generator location to change, and
a status.

Item statuses: `open` / `in progress` / `done` / `wontfix (reason)`.

### 1. `ml_dataset` — From a published QuerySet to a rendered sample

Reviewed 2026-09-02, against the freshly regenerated and executed notebook
(6/6 clean, Milton source byte-identical to `HEAD`, cell source identical to
what round 1 staged).

The prose finding restates X1, which already holds the diagnosis and the
skeleton. It is not duplicated here; see "Cross-cutting" below. The two
figure findings are new and are recorded as M1 and M2.

#### M1. The zoom row shows points, and should show query bounding boxes — `done`

**Request (2026-09-02):** "the published positions thing is useless zoom, the
zoom should rather show bounding boxes of actual queries".

**Location.** Generator lines 296-321, the two-by-two "Published positions"
figure, plus the prose that sets it up at lines 244-255.

**Diagnosis.** The bottom row plots the same thing as the top row, only over a
smaller extent: `axes[1, column].scatter(lons[inside], lats[inside], s=26)`
draws patch *centres* as dots. A centre is dimensionless, so the zoom adds no
information the top row lacks. It cannot show what a reader zooms in to see,
namely how much ocean one query actually covers, whether neighbouring queries
overlap, and how far training and eval footprints tile differently. The
notebook has already told the reader (§2) that a 256 km patch is
latitude-dependent in longitude span, and then draws nothing that has a span.

The prose currently compensates for the figure in words. Line 253-255 concedes
the figure is the weaker evidence: "the count is the more reliable read of the
two, because at this zoom the eye picks up the shared row structure more
readily than the longitude offsets". A figure that needs its own caption to
apologise for it is the thing to replace, not to caption better. Round 1's
verification note ("Look at every figure") flagged exactly this class.

**Proposal, not yet approved.** Replace the bottom-row scatter with drawn
footprints:

- `PatchSize(PATCH_SIZE_KM, "km").footprint(centre_lon, centre_lat)` already
  returns the exact `GeoBox` for a query (`ocean_taco/geobox.py:180`). No new
  geometry is needed, and the figure then uses the same call the library uses.
- Draw each footprint as an unfilled `Rectangle` with a low-alpha edge, so
  overlap reads as darker edges where boxes stack.
- The box holds 254 eval and 362 training positions, which is too many
  rectangles to read. Narrow the bottom row to a sub-box of a few degrees, or
  cap it at the first N rows of latitude, so individual footprints stay
  distinguishable. **Open question for the user:** which of the two, and what
  extent.
- Keep the printed counts cell as-is. It is measured and correct.
- Prose at 244-255 needs rewriting to match, since it currently describes a
  point scatter and argues the figure is unreliable. This overlaps X1 and
  should be done in the same pass.

**Interaction.** This changes code cells, so the figure and its output change.
It does not touch `SETUP`/`LOAD`; a shared footprint helper, if one is wanted
across notebooks, goes in `PLOTTING` per constraint 3.

#### M2. Maps have no continent contours — `done`

**Request (2026-09-02):** "the maps should all have continent contours".

**Diagnosis.** No map in the four rewritten notebooks draws a coastline. Every
geographic panel is a bare `scatter` or `imshow` on plain longitude/latitude
axes, so a reader cannot place a patch against any landmark, and an ocean
dataset renders as an unlabelled rectangle. `grep` for `cartopy|coastline|ccrs`
over the generator returns nothing.

**Inventory of affected panels**, by generator line. "Global" panels need
contours most; patch-scale panels over open ocean may show no coast at all,
which is itself informative but should be decided per panel.

| Line | Notebook | Panel | Extent |
|---|---|---|---|
| 303 | `ml_dataset` | published positions, global | global |
| 312 | `ml_dataset` | published positions, zoom (M1) | N Atlantic box |
| 674 | `ml_dataset` | rendered patch | patch |
| 925 | `spatio_temporal_query_generation` | rendered field | patch |
| 1087 | `data_retrieval_workflows` | coverage hits scatter | global |
| 1171 | `data_retrieval_workflows` | retrieved box | box |
| 1223 | `data_retrieval_workflows` | native Argo locations | global |
| 1233 | `data_retrieval_workflows` | Argo, second panel | -190..190 |
| 1317 | `ml_configuration_cookbook` | rendered image | patch |
| 1394 | `ml_configuration_cookbook` | speed | patch |
| 1474 | `ml_configuration_cookbook` | Argo scatter | patch |

**Feasibility, checked.** `cartopy` 0.25.0 is installed and its Natural Earth
shapefiles are already cached under
`~/.local/share/cartopy/shapefiles/natural_earth/physical/` at 110m, 50m and
10m for both `coastline` and `land`. So contours need **no network access at
execution time**, which matters because the executor runs offline on a compute
node. Round 1 added `cartopy` to the venv but never used it.

**Proposal, not yet approved.** Two routes, and the user should pick:

- **(a) Full cartopy axes.** `subplot_kw={"projection": ccrs.PlateCarree()}`
  plus `ax.coastlines(resolution=...)`. Correct, and gives real projection
  handling, but every affected panel changes construction, `imshow` gains a
  `transform=`, and the `imshow` panels currently use `aspect="auto"`, which a
  GeoAxes will fight. Largest blast radius.
- **(b) A `PLOTTING` helper that draws contours on ordinary axes.** Read the
  Natural Earth `land`/`coastline` geometries once and plot them as line
  collections onto the existing plain lon/lat axes, clipped to each panel's
  limits. PlateCarree on plain degree axes is the identity, so the contours
  land correctly with no projection machinery, no `transform=` on any existing
  call, and no change to `aspect`. Smaller diff, one shared helper, satisfies
  "all maps" uniformly. **Recommended.**

Either way the helper belongs in `PLOTTING` (constraint 3: never `SETUP`),
which is exactly what `PLOTTING` was created for in round 1.

**Open questions for the user.**

1. Route (a) or (b).
2. Whether patch-scale panels (674, 925, 1317, 1394, 1474) get contours too.
   They sit over open ocean, so the coastline will often be off-panel and add
   nothing visible. Applying it everywhere is uniform and matches "all";
   restricting to global and box panels avoids drawing nothing.
3. Resolution per scale: 110m is right for global panels, 50m for box panels;
   10m only if a patch-scale panel actually touches a coast.

**Interaction.** Code-cell changes in all four rewritten notebooks, so all
their figures change and must be re-inspected by eye. The Milton notebooks are
out of scope (constraints 3 and 4), so "all maps" here means the four
rewritten notebooks. **The user should confirm** that reading, since the
Milton figures are also maps without contours, and changing them needs an
explicit decision recorded in section 5-6.

### 2. `data_retrieval_workflows` — QuerySet selection and native-coordinate retrieval

Reviewed 2026-09-02.

#### D1. Rethink the notebook: filters exist to build QuerySets for ML — `open`

**Request (2026-09-02):** "I really don't get the point of
data_retrieveal_workflows it is more about filtering a query set, but all the
figures are unintuitive and meaningless, it should rather focus on how to use
the filters to construct query sets that are then used in ML workflows.
rethink this tutoiral".

**Location.** Generator lines 942-1250, 26 cells, 16 code, 1140 words.

**Diagnosis. The notebook has no single subject, and says so itself.** Its own
intro concedes the split: "The two halves serve different needs. Sections 1 to
4 are the QuerySet layer [...] Sections 5 to 8 are the lower-level retrieval
API". A tutorial that opens by explaining that it is two documents is two
documents. The title carries the same seam: "QuerySet selection **and**
native-coordinate retrieval".

**The second half duplicates the API reference.** Sections 5-9 demonstrate
`load_tile_nc`, `load_bbox_nc` and `load_multisource_time_series_nc`. All three
are already autodoc'd in `docs/api/remote.md:10-14`, along with
`load_hf_dataset` and `load_bbox_swot_nc`. So roughly half the notebook is a
prose restatement of generated reference material, which is why it reads as
having no point: the ML reader never needs it, and the reader who does need it
is better served by the API page.

**The filtering half never reaches ML.** Sections 1-4 build selections and then
stop at a count. `select_queryset` is called repeatedly, its `.count` printed,
and no selection is ever handed to a draw, a dataset or a loader. The one thing
that would motivate the whole notebook, a filter producing a QuerySet that
trains something, is absent. Meanwhile the *other* notebook
(`spatio_temporal_query_generation`) uses `CoverageRequirement` and explicitly
links here for the explanation, so the dependency runs the wrong way: the
motivating use lives elsewhere and this notebook holds only the mechanism.

**The figures are unintuitive, per-figure.** Four figures, and none of them
answers a question a reader arrives with.

| Line | Figure | Why it does not land |
|---|---|---|
| 1087 | positions coloured by how often they qualify, plus pairs-per-month bar | Shows a property of the **satellite orbit**, not of the filter. Its own closing line admits this: "Coverage varies by position and by month, which is a property of the satellite orbit rather than of the filter." A figure in a filtering tutorial whose caption says it is not about filtering. |
| 1171 | one retrieved SST box, decimated `[::8, ::8]` | A single unremarkable field. Decimation is display-only and unexplained, and nothing is compared against anything. |
| 1223 | native Argo point locations | A scatter of dots with no context, no coastline, no relation to any query. |
| 1233 | antimeridian box as two rectangles | The one genuinely useful figure. It shows a real mechanism a reader would otherwise get wrong. Keep it. |

**Proposed rethink, not yet approved.** Recast the notebook so its subject is
*constructing a QuerySet for an ML workflow with filters*, and cut what does
not serve that.

- **New spine:** a filter is how you turn the published set into the specific
  training set your experiment needs, and every filter choice is visible in the
  resulting set, so the notebook should end holding a dataset it built.
- **Keep and re-aim sections 1-4.** Each filter axis (`box`, dates, `coverage`)
  stays, but each is shown by **what it changes about the resulting set**, not
  by a count. Retain the null-versus-zero distinction (§2), which is real,
  non-obvious, and the kind of thing only a tutorial teaches.
- **End in an ML artifact.** The final section should take the filtered
  QuerySet into `draw_queryset` and a `DataLoader`, matching Q2's
  batch-of-8-and-visualise pattern, so the notebook demonstrates the thing it
  is about. This also inverts the current cross-link direction.
- **Cut sections 5-8**, or reduce them to a short pointer to
  `docs/api/remote.md`. They are reference material, already generated, and
  they are what makes the notebook feel pointless.
- **Keep §9's antimeridian figure**, which is a genuine mechanism, and fold it
  into the filtering half where a box is being constructed.
- **Replace the three weak figures** with ones that show a filter's effect:
  the same box before and after a coverage requirement; the surviving positions
  against the discarded ones; the drawn rows that a filtered set yields.

**Open questions for the user.**

1. **How far to cut §§5-9.** Delete outright, or keep a single short section
   showing one native retrieval for the reader working outside the QuerySet
   flow? Deleting is cleaner and the API page covers it; keeping one worked
   example costs little. **Recommendation: keep one, cut the rest.**
2. **Does the notebook keep its name and slot**, or does it become something
   like "Building a training set with filters"? The title change follows from
   the rethink, and `docs/tutorials/index.md` plus the cross-links in the other
   three notebooks would need updating with it.
3. **Overlap with `spatio_temporal_query_generation`.** If this notebook ends
   in a `DataLoader` and that one builds loaders per use case, the boundary
   needs stating: this one about **which rows** (selection), that one about
   **what shape** (query construction). Confirm that split.

**Interaction.** This is the largest of the round-2 items and it changes cell
count, fetches and cross-links. It should be sequenced **after** Q0 (revision
repin) and **after** M2 (contours), since the replacement figures are maps.
Whether its prose rewrite is folded in here rather than run as a separate X1
pass depends on how much survives the restructure: rewriting prose that is
about to be deleted is wasted, so **X1 for this notebook should wait for D1**.

### 3. `spatio_temporal_query_generation` — ML use cases and the training loader

Reviewed 2026-09-02.

#### Q0. Repin the catalog revision to latest `main` — `done`

**Request (2026-09-02):** "we have the other query sets also published on HF so
no problem at all, don't act like this is a huge deal" / "you can repin to
latest main HF version it has everything".

**Measured.** The pinned revision
`95a7cfca2723f5f3b3d55592520651ce1c4a55c4` publishes only `256-eval` and
`256-training`. Latest `main`, sha `4a3233f8f0d0a38bb85d8122043c9ffd3b772196`,
publishes **128, 256 and 512, each in eval and training**. So larger patches
are already available and Q1 is unblocked.

**Location.** One line: `ocean_taco/catalog.py:9`,
`CORE_DATASET_REVISION`. A repo-wide grep for the old sha returns that file
only, so nothing else needs editing.

**Consequence.** Every notebook re-fetches against the new revision, so all
six re-execute with new data. The two Milton notebooks execute against this
revision too, and although constraint 3 protects their *source*, their
**outputs** will change. Their figures reproduce paper figures, so they must be
inspected after the repin to confirm they still match. This is the one place
where Q0 touches the paper-figure constraint.

**Status of `ocean_taco/catalog.py`.** It is already staged as part of round 1,
so this edit joins the existing staged change rather than opening a new file.

#### Q1. The super-resolution section is wrong and must be rebuilt — `done`

**Request (2026-09-02):** "supre-resolution section completely wrong, why is
there coarse input that is also somehow masked? why you making that
assumptions. The patches should likely be much larger to have more context so
cover a larger area, and it should be full l4sst, l4_ssh and then both
altimetry and swot shown as well for that bounding box. first as raw inputs
such that we can show the pixel size mismatches that might occur and then also
explain how to use the resample functionality, because most super res models
need like a power 2 super resolution factor, so we want to show how to
congigure that correctly, and then the visualizations always need to show
diverse sapmles, it cannot just be a single sample", then clarifying: "l3_ssh
and l3_swot should be shown in super resolution, and the text is clearly
wrong, please follow the feedback I gave".

**Location.** Generator lines 638-680 (§3): one `md` cell and two `code` cells.

**Diagnosis.** Five defects, each confirmed against the generator.

1. **The coarse input is a masked sparse swath, which is not a
   super-resolution input.** Both tokens are `l3_swot` (lines 665-666), a
   sparse along-track swath. Rendered at 32x32 it is not a low-resolution
   version of the target, it is the same sparse data on a coarser grid, with
   holes and a `support_mask`. The section teaches the wrong setup.
2. **The premise is an unstated assumption.** "Super-resolution needs the same
   anchor rendered twice, differing only in output shape" is one narrow
   construction presented as the definition.
3. **The patch is too small for context.** 256 km, against
   `BOX = GeoBox(-80, -30, 10, 45)` (line 549).
4. **Only one source is shown**, where the reader needs the multi-product
   resolution contrast.
5. **One sample.** `coarse, fine = coarse_set[0], fine_set[0]` (line 667)
   plots row 0 of a 6-row draw.

**Resolved by the user, no longer open.**

- **Patch size: use the 512 km QuerySet** for this section, available under Q0.
  Larger patch, more context, as requested.
- **Sources: `l4_sst`, `l4_ssh`, `l3_ssh` and `l3_swot`** over the same
  bounding box. `l3_ssh` is nadir altimetry (`registry.py:107`,
  variable `sla_filtered`) and `l3_swot` is the SWOT swath
  (`registry.py:108`). Both confirmed present in the registry.
- **Super-resolution pair: `l3_ssh` and `l3_swot`.** These are the two shown at
  two resolutions. `l4_sst` and `l4_ssh` appear as the dense context fields in
  the raw-input stage.

**What the section becomes.** Two stages, in this order.

- **Stage 1, raw native inputs.** `l4_sst`, `l4_ssh`, `l3_ssh` and `l3_swot`
  over one 512 km box at their **native** grids, via `Native()`, so the
  pixel-size mismatch between products is directly visible. This motivating
  observation is currently absent. `native_shape` is already in every payload,
  so the mismatch can be both drawn and printed.
- **Stage 2, `Resample` at a power-of-two factor.** Show how to configure
  `Resample` so low-res and high-res shapes stand in a power-of-two ratio,
  which is what super-resolution architectures require. **The current text
  argues the opposite** and is simply wrong: line 644 says "Choose the shapes
  against the native resolution rather than against a convenient power of
  two". The two constraints are not opposed. The section should derive a shape
  pair that is both a power-of-two factor apart and at or below native, showing
  the calculation from `native_shape` rather than asserting constants.
- **Diverse samples.** Multiple drawn rows in every panel set, never `[0]`.

**Remaining open question.** Which power-of-two factor to demonstrate (2x or
4x), which follows from the 512 km native shapes and should be measured once
Q0 lands rather than guessed.

#### Q2. Text-only examples need visualisations — `done` (2026-09-07)

**Request (2026-09-02):** "all example cases need clear visualizations, only
text output is unacceptable and doesn't help to illustrate the conceps".

**Diagnosis, measured.** 14 code cells, and **only 3 produce a figure**. The
per-cell inventory from the executed notebook:

| Cell | Section | Figures | Text outputs |
|---|---|---|---|
| 4 | 1. Forecasting | 0 | 3 |
| 5 | 1. Forecasting | **1** | 2 |
| 6 | 3. Super-resolution | 0 | 1 |
| 7 | 3. Super-resolution | **1** | 2 |
| 8 | 4. Multi-source | 0 | 2 |
| 9 | 5. Mixed output shapes | 0 | 4 |
| 10 | 6. Normalisation | 0 | 2 |
| 11 | 7. Training loader | 0 | 5 |
| 12 | 8. Masked loss | 0 | 1 |
| 13 | 8. Masked loss | **1** | 2 |

So **sections 2, 4, 5, 6 and 7 are entirely text-only**: midpoint retrieval,
multi-source sparse-plus-dense, mixed output shapes, normalisation statistics,
and the training loader. Each teaches a spatial or structural concept and each
currently asks the reader to reconstruct it from printed shapes.

**Candidate visualisations**, one per uncovered section, as a proposal:

- **§2 midpoint retrieval:** a timeline like §1's, showing the target centred
  in a symmetric window. §1 already has exactly this figure, so the asymmetry
  between the two sections is unmotivated.
- **§4 multi-source:** panels of the dense fields with the sparse SWOT swath
  overlaid, showing what "requiring a sparse source" actually selects.
- **§5 mixed output shapes:** the differing native shapes drawn at their true
  relative sizes, which is the whole point of the section and is currently only
  a printed tuple.
- **§6 normalisation:** value histograms before and after normalisation,
  computed through the mask.
- **§7 training loader:** a rendered batch as a grid of panels, showing what a
  `DataLoader` batch actually contains.

**Resolved by the user: every case goes through a `DataLoader`.** "since this
is about query generation that should also turn into dataloders which means the
tutorial should create dataloaders, from which we then pull one batch, mayb
size 8 and then visualize the batch for all cases".

This supersedes the per-section candidate list above and sets one uniform
pattern for the whole notebook: **each use case builds its `DataLoader`, pulls
a single batch of 8, and visualises that batch.** It answers the §6/§7 question
too, since a batch is visualisable in every case, and it satisfies Q1's
"diverse samples" requirement by construction rather than as a separate rule.

It also fixes a structural oddity: §7 is currently titled "The training loader"
and is the only place a `DataLoader` appears, which makes the loader look like
a final step rather than the thing every query shape feeds. Building one per
case makes the notebook match its own title.

**Consequences to work through when implementing.**

- Batch size 8 against the current `requested_row_count` values (§3 draws 6)
  means the draws must request at least 8 rows.
- `ShapeBucketSampler` already exists for the native-shape case (§5), where a
  batch of 8 cannot stack unless shapes agree. §5's loader needs it; the others
  do not.
- Batch visualisation is a repeated operation across five sections, so the
  panel-grid helper belongs in `PLOTTING` (constraint 3), not repeated inline.
- Fetch cost rises: five loaders times 8 rows, against today's handful of
  single samples. Execution time should be measured after the first section is
  converted, before the rest follow.

**Interaction.** Every new panel is a map, so **M2 (contours) should be settled
before these figures are drawn**, otherwise they get drawn twice.

**Outcome (2026-09-07).** Implemented and verified. The notebook went from 16
code cells / 4 figures to **21 / 11**, with every section now building a
`DataLoader`, pulling one batch of `BATCH_SIZE = 8` and visualising it.

- Sections 1, 2, 4, 5 and 6 gained batch figures; §3's final figure was
  converted from four hand-picked rows to the whole batch.
- Two helpers landed in `PLOTTING`: `batch_panel_grid` (with `batch_member` and
  `batch_dates`), which draws a collated batch as a grid of maps.
- **§7 did not dissolve, it narrowed.** Its collate and worker-seeding material
  moved to §1 where the first loader is built, leaving §7 to the one case the
  uniform pattern does not cover: `ShapeBucketSampler`. It now closes with a
  measured comparison against §5's padded batch, 0 padded cells bucketed
  against 35,796 in draw order.
- **Fetch cost was not the risk the plan expected.** The whole notebook
  executes in **~120 s**, because the sections reuse three draws rather than
  making five independent ones.

Three defects surfaced only once every drawn row was plotted, which is the
argument for the batch pattern in miniature:

1. The `l3_ssh` support threshold, recorded below.
2. §2's window figure rendered five identical maroon blocks, because the shared
   `COLOR_RANGE` is built for cross-panel comparison and this figure compares
   days within one patch. It now picks the highest-variance batch member and
   scales to that patch's own range.
3. §6's before/after histogram overlaid two identical shapes on a twin axis and
   showed nothing. It is now four panels in two columns.

Prose consequences, both from output that no longer matched the text: §8's
claim that a whole batch is skipped is false at 8 rows (three batches carry a
partly absent target, none is fully absent), and a `float(loss)` on a
graph-attached scalar raised a new `UserWarning`, fixed with `.detach()`.

**Found while implementing Q2 (2026-09-07): `l3_ssh` must not carry a support
threshold.** Q1's rebuilt §3 resampled both super-resolution sources with
`support_threshold=0.5`. `l3_ssh` is nadir altimetry, which covers a line
rather than an area, so no output cell of a useful size ever reaches half
coverage. Measured over the 8-row 512 km draw: at 13x13 the source is
structurally absent in **8 of 8 rows** at threshold 0.5, against 7 of 8
available at 0.0. The coarse half of the super-resolution pair was therefore
empty in every column, which only became visible once the batch figure drew all
eight rows rather than a printed shape. Fixed by setting the threshold per
source (`l3_ssh` 0.0, `l3_swot` 0.5) and saying why in the prose. The user
confirmed: "for l3_ssh there shouldn't be such a threshold".

### 4. `ml_configuration_cookbook` — ML renderer configuration reference

Reviewed 2026-09-02.

#### C1. Rewrite the prose — `done`

**Request (2026-09-02):** "again writing needs to be improved significatly".

**Location.** Generator lines 1250-1584. 9 markdown cells, 1109 words, 23
paragraphs of 8 words or more.

**Diagnosis: the signature profile here differs from `ml_dataset`,** measured
rather than assumed, so X1's `ml_dataset` recipe does not transfer unchanged.

| Signature | `ml_dataset` | `ml_configuration_cookbook` |
|---|---|---|
| S1 conceits / personification | several | **0** |
| S2 shaped short closers | many | **3 of 23** |
| S3 clefts "X is what Y" | 7 total across notebooks | **1** ("A uniform comparison grid is what makes these...") |
| S4 self-grading | 6 total across notebooks | **1** ("This notebook is the per-renderer reference.") |

The four X1 signatures are largely **absent**. Sweeping for personification
verbs (`wants`, `pulls`, `punishes`, `disagree`, `honest`) returns nothing. So
whatever makes this notebook read as machine-written is a **fifth signature**
that X1 does not name, and the rewrite needs it identified before drafting.

**What the measurement actually shows.** Two things stand out.

- **S5. Template-uniform section structure.** Every one of the 9 sections opens
  the same way: a one-sentence definition of the renderer, immediately followed
  by a `so`/`because` clause giving its consequence. Extracted openers:
  "`Resample((H, W), support_threshold)` puts every source on one grid, so
  each...", "`Native()` returns the source's own grid, so nothing is
  interpolated and...", "Each configuration renders the same drawn row, so the
  comparison isolates the...". Nine sections built to one template is the
  cadence a reader registers as generated, even though each individual sentence
  is defensible. `SKILL.md`'s point applies directly: the prose sits in the
  middle of the safe region and reads as machine-written while passing every
  rule.
- **S6. Uniform paragraph length.** 23 paragraphs, and the closer length
  distribution is flat: 15 of 23 close on a 20-32 word sentence. There is
  almost no short sentence anywhere, which is the mirror image of `ml_dataset`
  (where the problem was short shaped closers). Unvarying rhythm at either
  extreme reads as generated.

**Consequence for the fix.** `ml_dataset`'s rewrite removes conceits, clefts
and shaped closers. This notebook has almost none of those. Its rewrite is
about **breaking the template**: varying how a section opens (some with the
signature, some with the problem it solves, some with the failure it prevents),
and varying sentence length so the rhythm stops being uniform. Applying X1's
`ml_dataset` edits here would change little.

**A reference notebook may want some uniformity.** This is a reference, and
parallel structure across parallel renderers is a legitimate choice, not
automatically a defect. **Open question for the user:** how far to break the
template. Options: (a) vary every section opener; (b) keep the parallel
definition-first structure, which suits a reference, and vary only rhythm and
paragraph length; (c) restructure so the four renderers are genuinely
contrasted rather than listed in parallel.

**Method.** As X1: skeleton first, then draft, then `references/checklist.md`
scoped to what changed. The skeleton for this notebook is **not yet built** and
should be, before any drafting, since the diagnosis above says the section
structure itself is in question.

**Interaction.** Prose-only, so no figure or output changes, but still requires
the full regenerate-and-execute cycle (constraint 2). Independent of M1/M2.

**Relation to X1.** X1 currently says the other three notebooks follow
`ml_dataset` "since the same four signatures run through all of them". This
measurement **contradicts that** for `ml_configuration_cookbook`. X1's scope
note should be corrected once the user confirms the reading, and the remaining
two notebooks measured individually rather than assumed to match `ml_dataset`.

### 5-6. Hurricane Milton notebooks

Out of scope. Any change here breaks constraint 3 or 4, so it needs an
explicit decision from the user recorded in this section first.

### Cross-cutting

#### X1. Rewrite the prose under `prose-writing-style`, starting with `ml_dataset` — `in progress` (`ml_dataset`, `spatio_temporal_query_generation` done; `data_retrieval_workflows` blocked on D1)

**Request (2026-09-02):** "general advice /prose-writing-style should be used
to rewrite all the text, because it is still horrendous AI patterns", then,
scoping the first pass: "the prose writing skill has not the patterns list
included and should be used to improve the writing in ml_dataset.ipynb because
it is horrendous".

**The pattern list is out.** The user first mentioned the Patina list
(`https://github.com/devswha/patina/blob/main/docs/PATTERNS-EN.md`) as a
possible aid, then established it is not part of the skill. Confirmed: a grep
across `~/.claude/skills/prose-writing-style/` returns no mention of it, and
the skill ships only `SKILL.md`, `references/corrections.md`,
`references/moves.md` and `references/checklist.md`. **The skill alone governs
this rewrite.** The Patina sweep in `scripts/dev/prose_audit.py` stays as a
cheap regression net for round 1's ban-list, not as a source of instructions.

**Scope.** `ml_dataset` first, 2181 words, the largest and worst of the four.

**Correction (2026-09-02): the "same four signatures run through all of them"
claim is withdrawn.** It was asserted, not measured. Per-notebook measurement
(see "Decisions taken") shows `ml_dataset` carries 4 personifications, 10 short
closers, 5 clefts and 7 self-gradings, while `ml_configuration_cookbook`
carries 1/3/1/2 and needs a different fix entirely (C1). Each notebook is
measured before its rewrite is drafted, and `data_retrieval_workflows` is
rewritten only after D1 restructures it. All prose lives in
`scripts/dev/restore_tutorial_notebooks.py`, `ml_dataset` at lines 104-485.

**Why a mechanical pass will not do it.** Sweeping the 6124 words against the
Patina list returns almost nothing: 4 colon-reveals, 4 "from X to Y" ranges,
and zero for AI vocabulary, copula avoidance, negative parallelism, participle
chains, filler, hedging, conclusion signals, throat-clearing, em dashes, emoji
and rhetorical questions. Round 1's ban-list already removed the surface tells,
which is exactly the failure `SKILL.md` opens on: rules mark the boundary of
acceptable prose and cannot locate its centre, so prose aimed at the middle of
the safe region reads as machine-written while passing every check.

**Diagnosis: four signatures**, each counted rather than asserted. Counts are
across all four notebooks; the `ml_dataset` examples are quoted verbatim.

- **S1. Organizing conceits and framing devices** (corrections entry 1). The
  prose repeatedly builds a frame and pays it off. §2 opens "Two goals compete
  whenever ocean data becomes tensors" and personifies both sides ("A model
  wants...", "The data pulls the other way"). The title opens on "instruments
  that disagree about almost everything". §3's "what the two placements share
  and where they part" is the same move at sentence scale, and §2's "`l3_swot`
  is the case that punishes naive handling" personifies the data again. Entry
  1 says these get **removed entirely** and re-stated as ordinary exposition,
  not reworded.
- **S2. Shaped closers, 19 paragraphs** (entry 7). Each ends on a short
  unquantified verdict: "It has no default for the same reason the renderer
  has none", "Nothing cross-checks shapes between tokens", "Recomputing per
  batch would let batch composition reach the inputs". Entry 7 replaces the
  quotable closing construction with substance, and §4.1 warns that the number
  is what gets evicted when a sentence aims at a beat.
- **S3. Cleft emphasis carrying the claim, 7 instances** of "X is what Y":
  "`support_threshold` is what keeps resampling honest", "The leading zero is
  the signal", "That is what makes `num_workers > 0` safe here". A cleft
  exists to place stress, so it is shape-driven by construction (§4.1).
  "Honest" is separately an evaluative verdict standing where the mechanism
  belongs (entry 4).
- **S4. Self-grading the exposition, 6 instances** (entry 4). "this notebook
  walks through all four", "That is the honest trade", "The three masks stay
  separate on purpose", "Keeping them apart is what lets you...". Entry 4
  requires claims about one's own exposition to be hedged or cut while the
  technical claims stay flat and quantified. The current prose inverts that.

**Skeleton for `ml_dataset`**, built before drafting as `SKILL.md` requires.
Spine: *rendering ocean observations into tensors requires choices the data
cannot make for you, so OceanTACO requires each one explicitly and records it,
which is what makes a rendered sample reproducible from artifacts far smaller
than the data.*

| § | Claim | Evidence | Dependency on previous |
|---|---|---|---|
| Intro | Four stages, each recording its selection | the four named APIs | — |
| 2 | Geography, native resolution and renderer choice together fix the array | `to_degrees` figure, measured grid steps | supplies the vocabulary §§4-5 use |
| 3 | `kind` records placement, not independence | measured row and longitude counts | narrows §2: which rows may be drawn |
| 4 | A draw names rows and replays exactly | `replayed.rows == draw.rows` | supplies the rows §5 renders |
| 5 | Sample schema, three masks, structural absence | printed shapes | renders §4's rows with §2's settings |
| 6 | Collation keeps availability separate from values | batch shapes | generalizes §5 to a batch |
| 7 | Normalisation is the caller's, computed through the mask | printed statistics | depends on §5's mask and §6's batching |
| Close | Six artifacts reproduce everything | each named above | collects what §§1-7 produced |

Every dependency is a real relation rather than "comes after", so the section
order holds and no section needs cutting. The rewrite is therefore at the
paragraph and sentence level, not structural.

**Worked pairs.** Drafted against the skill to calibrate scale, then reverted
pending approval. These are the intended target, not applied text.

*S1, the §2 opening conceit.* Before: "Two goals compete whenever ocean data
becomes tensors. A model wants a stable interface, meaning fixed channel shapes
so that batches stack and architectures stay simple. The data pulls the other
way, because resampling a sparse swath onto a coarse grid destroys the
structure the swath was flown to measure." After: "Fixed channel shapes make
batches stack and keep architectures simple, while resampling a sparse swath
onto a coarse grid discards the fine-scale structure that the swath was flown
to measure. Since the better trade differs by source and by application,
OceanTACO requires the choice to be stated per source rather than applying a
default." The competing-goals frame and both personifications go, the same two
technical facts remain, and the relation between them is stated with "while"
and "Since".

*S3 and S2 together, the `support_threshold` paragraph.* Before: "`support_
threshold` is what keeps resampling honest. [...] It has no default for the
same reason the renderer has none." After: "The second argument,
`support_threshold`, sets when a resampled cell counts as measured. [...] It
also has no default, for the same reason the renderer does not." The cleft
becomes a plain definition, "honest" becomes the operational criterion, and the
closer states the fact without the cadence.

*S4, §5's structural-absence paragraph.* Before: "The leading zero is the
signal. The key is still present and the batch layout is unchanged, so a reader
can distinguish an absent source from a present one." After: "Because the key
remains present and the batch layout is unchanged, downstream code detects
absence from the leading zero or from `availability` without inspecting
values." The planted headline sentence merges into the sentence that explains
it, and the unmarked run gains its connective.

**Method.**

1. The skeleton above comes first, and is done.
2. Rewrite `ml_dataset` in the generator, section by section, then stop for
   review before touching the other three.
3. Orient before sharpening (entry 2) and define units at first use (entry 3),
   both of which suit tutorial register.
4. Keep every number, every table, and the long multi-clause explanatory
   sentences. Corrections' "What he left alone" is explicit that density is
   not the problem: conceits, self-grading and shaped closers are.
5. Run `references/checklist.md` last, scoped to what was rewritten, plus the
   Patina sweep as a regression net.

**Expected size.** The reverted draft covering roughly half of `ml_dataset`
touched 124 lines against 115, so the rewrite is close to length-neutral and
does not shorten the tutorial.

**Constraint interaction.** Code cells are untouched by this item, so the
figures and outputs do not change. Prose-only edits to the generator still
require the full regenerate-and-execute cycle (constraint 2), because the
generator rewrites every `.ipynb` from scratch. The Milton notebooks contain 97
words of prose between them and are out of scope (constraints 3 and 4).

**Verification.** The existing banned-term and ban-list greps, plus the Patina
sweep, plus `references/checklist.md` scoped to what was rewritten.

## Decisions taken 2026-09-02

Every open question in the item sections below has been answered by the user.
Where an item's prose still poses a question, this table is what governs.

| # | Question | Decision |
|---|---|---|
| M2 | contour route and scope | **`PLOTTING` helper on plain axes, all 11 map panels.** Not cartopy GeoAxes. Patch-scale panels included, even where the coast falls off-panel. |
| M1 | how to narrow the footprint zoom | **A smaller sub-box, a few degrees**, so individual 256 km footprints and their overlap are legible. |
| D1 | how much retrieval API survives | **Keep one worked native-retrieval example, cut the rest**, pointing to `docs/api/remote.md`. |
| D1 | title and slot | **Rename to match the new subject**; keep the existing filename. `index.md` and cross-links in the other three notebooks follow. |
| C1 | how far to break the template | **Vary openers and rhythm, keep the parallel definition-first structure**, which suits a reference. |
| Q2 | do §6 and §7 get figures | **Figures everywhere, fitted to the topic.** §6 gets before/after normalisation histograms through the mask, not a map grid. §7 may dissolve, since every section now builds a loader. |
| Q0 | Milton figure risk | **Repin and proceed; check the Milton figures at final verification**, not as a gate. |
| X1 | shared or per-notebook diagnosis | **Measure each notebook before drafting.** X1's one-shared-diagnosis claim is withdrawn. |

### Per-notebook prose measurement

Run after the X1 decision above, so no notebook's rewrite is drafted against
another's diagnosis. Counts are over markdown cells of the executed notebooks.

| Notebook | words | paras | S1 personification | S2 short closers | S3 clefts | S4 self-grading |
|---|---|---|---|---|---|---|
| `ml_dataset` | 2181 | 34 | 4 | 10 | 5 | 7 |
| `spatio_temporal_query_generation` | 1694 | 34 | 2 | 9 | 1 | 6 |
| `data_retrieval_workflows` | 1140 | 24 | 0 | 5 | 1 | 4 |
| `ml_configuration_cookbook` | 1109 | 23 | 1 | 3 | 1 | 2 |

`ml_dataset` carries the S1-S4 load, which is why X1 starts there.
`spatio_temporal_query_generation` is second on S2 and S4 but has almost no
clefts. The two right-hand notebooks are low on every X1 signature, and the
cookbook's problem is S5/S6 (template uniformity and flat rhythm) instead. So
the rewrite differs per notebook, and X1's "same four signatures" scope note is
**withdrawn**.

## Build order

Items have hard dependencies. This order respects them; the rationale for each
edge is in the item sections.

| Step | Item | Why here |
|---|---|---|
| 1 | **Q0** repin `catalog.py:9` | Everything else executes against the new revision. Doing it later would invalidate every figure built before it. |
| 2 | **M2** `PLOTTING` coastline helper | Every figure added by M1, Q1, Q2 and D1 is a map. Building it first means each new figure gets contours once rather than twice. |
| 3 | **M1** footprint zoom in `ml_dataset` | Small, self-contained, and exercises the M2 helper on a real panel before the larger items depend on it. |
| 4 | **Q1** rebuild super-resolution | Needs Q0 (512 km QuerySet) and M2. Introduces the multi-source native-grid figures that Q2's batch pattern then generalises. |
| 5 | **Q2** loaders and batch-of-8 across `spatio_temporal_query_generation` | Needs Q1's section to exist in final form. Adds the `PLOTTING` batch-grid helper that D1 reuses. |
| 6 | **D1** rethink `data_retrieval_workflows` | Largest item. Reuses Q2's batch-grid helper and its loader pattern for the closing section. Renaming touches `index.md` and cross-links, so it lands after the notebooks it links to are stable. |
| 7 | **X1** prose, per-notebook | Deliberately last for the three restructured notebooks: prose rewritten before D1 or Q1 restructures a section is wasted. `ml_dataset` prose could start earlier, but M1 changes §3's text anyway. |
| 8 | **C1** cookbook prose | Independent of every code item, since the cookbook has no figure work. Can run in parallel with any step, or last. |

**Regeneration cadence.** Steps 1-6 each change code cells, so each ends with
the full regenerate-and-execute cycle (constraint 2) and a look at every
changed figure. Steps 7-8 are prose-only but still need the cycle, because the
generator rewrites every `.ipynb` from scratch.

**Two helpers land in `PLOTTING`** (constraint 3, never `SETUP`): the coastline
helper at step 2 and the batch-grid panel helper at step 5. Both are used by
more than one notebook, which is what `PLOTTING` exists for.

## Open risks

Recorded rather than resolved, because each is only decidable once the work
starts.

- **Fetch cost and execution time.** Q2 builds five loaders of 8 rows where
  today there are a handful of single samples, and Q1 adds four sources over a
  512 km box. Execution time should be measured after step 4 and again after
  the first converted section in step 5, before the rest follow. Today's full
  cycle is a few minutes; this could change materially.
- **The Milton figures under Q0.** Their source stays byte-identical
  (constraint 3 is about source, not outputs), but they re-execute against the
  new revision. The user has decided to check them at final verification rather
  than gate on it. If they shift, that is a decision point, not a bug to fix
  silently, since they reproduce paper figures.
- **512 km patches at high latitude.** `PatchSize.footprint` raises if a
  footprint crosses a pole, and a 512 km patch spans twice the longitude of a
  256 km one at the same latitude. The North Atlantic box is far from the
  poles, so this should not bite, but Q1's draws should be checked rather than
  assumed.
- **Power-of-two factor.** Q1 must satisfy both a power-of-two ratio and
  fine-shape-at-or-below-native. Whether 2x or 4x is achievable follows from
  the 512 km native shapes, which are measurable only after step 1.
- **`ShapeBucketSampler` and batch-of-8.** §5's native-shape section cannot
  stack a batch of 8 unless shapes agree. The sampler exists for this, but
  whether 8 rows of agreeing shape are available in the drawn set is unverified.

## Carried over from round 1

Open points that round 1 surfaced but did not close.

- **The commit.** Round 1 says the work "lands as one combined change together
  with the notebook rewrites, not as a separate commit". The 26 files are
  staged and the user has not asked for the commit, so it has not been made.
  Round-2 fixes should join the same commit.
- **Two round-1 claims were corrected against measurement**, and the reasoning
  should survive any re-edit of those passages:
  - The training/eval placement difference is **not** visible as
    lattice-versus-scatter when zoomed. Both sets sit on regular latitude rows
    (23 training, 17 eval, spacing std ~0.05). They differ in *longitude*
    placement (362 vs 254 distinct longitudes). The prose says this and a code
    cell prints the counts.
  - The single-panel L4 SST render uses a **per-patch** `vmin`/`vmax`, not the
    shared range. That patch spans ~0.9 degrees celsius against a basin-wide
    range, so the shared limits flattened it to one shade. A comment records
    why. Shared ranges are kept wherever panels are actually compared
    side by side.
- **`plot_ocean_sample` bounds-checking**: round 1's open question about an
  `IndexError` on `time_index` is resolved. It is already bounds-checked at
  `ocean_taco/plot.py:60` and raises a clear `ValueError`.

## Verification, run after every batch of fixes

```sh
source env.sh

# 1. tests
pytest -q

# 2. regenerate AND execute as one operation, never commit in between
python scripts/dev/restore_tutorial_notebooks.py
python scripts/dev/execute_tutorial_notebooks.py

# 3. banned terms, expect 0
grep -rniE 'leakage|split policy|guard band|\brecipe\b|\bpopulation\b' \
  scripts/dev/restore_tutorial_notebooks.py docs/tutorials/*.ipynb \
  docs/tutorials/index.md | grep -v -- '-plan.md'

# 4. prose sweep: Patina regression net plus the S1-S4 signatures (item X1).
#    Baseline before the rewrite: 7 S3 clefts, 6 S4 self-grading, 19 S2
#    shaped closers, 4 colon reveals, 4 false ranges. Expect these to fall,
#    and expect the Patina rows to stay at 0.
python scripts/dev/prose_audit.py

# 5. Milton source identical to HEAD, and paper viz untouched
git status --short ocean_taco/viz/paper/     # expect empty

# 6. docs build, expect only the known SWOT xref warning
sphinx-build -b html docs docs/_build/html
```

Plus two checks that no exit code will give you:

- **Execution counts.** Every code cell needs a non-`None`, monotonically
  increasing `execution_count`, or the published page shows unrun cells.
- **Look at every figure.** This is the check that matters most. Round 1's
  four worst defects (a prose claim contradicted by its own figure, a
  washed-out SST render, clamped velocity components, an uninformative
  single-dot Argo panel) all produced clean exit codes and correct-looking
  code. Only viewing the rendered PNGs caught them.


## Progress log

### 2026-09-07 — resumed after an interrupted cycle

The 2026-09-02 session ended mid-cycle: the generator ran at 16:32 and
execution only reached `spatio_temporal_query_generation` at 16:35, leaving the
other five notebooks with **no stored outputs**. Because `nb_execution_mode` is
`"off"`, committing in that state would have published blank code cells. No
execution failure was recorded; the run simply stopped.

Verified before re-executing: all six notebooks' cell sources were byte-identical
to a fresh generator run, so execution proceeded without regenerating and without
destroying the one good notebook's outputs.

**Landed on 2026-09-02 but never marked**, confirmed by reading the diff:

- **M2** — `_coastline_segments` / `add_coastlines` in `PLOTTING`, reading the
  cached Natural Earth shapefiles, wired into the map panels. Route (b) as
  decided.
- **M1** — the "Published positions" zoom row now draws footprint
  `Rectangle`s over a sub-box instead of a point scatter.
- **Q1** — §3 rebuilt as "native grids, and a power-of-two rescaling", with
  `FACTOR = 4` and a multi-row pair figure.
- **Q0** — `CORE_DATASET_REVISION` repinned to `4a3233f8`.
- Unplanned but kept: `strip_environment_noise` in
  `execute_tutorial_notebooks.py`, filtering TqdmWarning / IProgress /
  unauthenticated-Hub stderr out of stored outputs while leaving `ocean_taco`'s
  own warnings, which the tutorials demonstrate, in place.

**X1, `ml_dataset` — done 2026-09-07.** Rewritten in the generator against
`prose-writing-style`, using the skeleton and worked pairs already in this
document. Measured on the regenerated notebook: self-grading 7 → 1, clefts
5 → 0, shaped closers 10 → 4, Patina rows unchanged at 0. Word count 2181 →
2221, so length-neutral as predicted. The §2 competing-goals conceit and both
personifications are gone, the `support_threshold` cleft is a plain definition,
and §3's prose now describes the footprint figure M1 introduced rather than the
point scatter it replaced, including dropping the sentence that conceded the
figure was the weaker evidence.

Two semicolons introduced during the rewrite tripped the round-1 ban-list and
were replaced with sentence-initial connectives, which entry 5 prefers anyway.

**Still open:** X1 for the remaining three notebooks (`data_retrieval_workflows`
only after D1 restructures it), D1, C1, Q2.

### 2026-09-07 (continued) — X1 for `spatio_temporal_query_generation`, and C1

**Scope for this session, set by the user:** prose items only (X1 and C1). D1
and Q2 are left open for the user to direct, since D1 restructures a notebook
and Q2 adds figures.

**X1, `spatio_temporal_query_generation` — done.** Measured before drafting
rather than assumed to match `ml_dataset`, as this document requires. The
profile sat between the other two notebooks: only 2 personifications and 2
clefts, but 13 short closers across 44 paragraphs. Fixed the two
personifications (a model that "wants" its target; zero-filling that "pulls"
the mean), both clefts, and one genuine shaped closer ("That is a different
query, and it needs no library change", which now states that the reordering
lives in the target offsets). Clefts and self-grading both measure 0 after.

**C1 — done.** Its skeleton was built first, as this document required, and is
in the session scratchpad. The diagnosis held up: the four X1 signatures were
largely absent and the real problems were S5 and S6.

- **S5, template-uniform openers: 9 of 9 → 3 of 9.** `VectorPair` now opens on
  the failure it prevents, `Native` on the cost `Resample` pays, the sparse
  section on what a missing cell records, and normalisation on whose choice it
  is. Three definition-first openers were kept deliberately, per the user's
  decision (option b) that parallel structure suits a reference.
- **S6, uniform rhythm: closers in the flat 20-32 word band 15 of 23 → 11 of
  23**, closer-length stdev now 9.6.
- The one cleft ("A uniform comparison grid is what makes these...") and the
  one self-grading sentence ("This notebook is the per-renderer reference.")
  are both gone.

**Verification at this point.** `tests/` 90 passed / 7 skipped / 1 deselected,
matching the baseline. Banned-term sweep returns 0. Milton notebooks' cell
source byte-identical to `HEAD`, and `ocean_taco/viz/paper/` untouched, so
constraints 3 and 4 hold. Regeneration and execution were run as one operation.

Note: `pytest -q` from the repo root fails collection on six
`ocean_taco/test_*.py` files because the execution venv has no `aiohttp`. That
is the pre-existing import breakage the exit plan already records, unrelated to
the tutorials; `pytest tests/` is the suite the baseline refers to.


## Appendix: the C1 skeleton, built 2026-09-07

Preserved here because the session scratchpad it was written in is wiped
between sessions.

Spine: the renderer chosen per source fixes the tensor structure downstream code
must handle — how samples stack, what absence looks like, and how much native
resolution survives — so the four renderers are picked by the structure an
experiment needs, not by preference.

| § | Claim | Evidence | Dependency on previous |
|---|---|---|---|
| Intro | Renderer choice fixes tensor structure; four renderers, one shared row | the four named APIs | — |
| Row | One drawn row, coverage-conditioned, so differences are renderer-only | printed shapes/masks | supplies the control every later section varies against |
| Resample | Fixed grid: channels concatenate, batches stack, cost is interpolation | 64x64 vs native 23x31, 19x24; upsample warnings | first configuration on the shared row |
| VectorPair | Two components as one unit prevents mixed-validity directions | (T,2,H,W), pair_available | narrows Resample: some sources must not be rendered independently |
| Sparse+dense | Absence must live in masks, not values, since 0 is a valid anomaly | SWOT valid_mask vs L4 | generalizes Resample to a sparse source |
| Points | Ragged records, count varies, zero is valid | profile_count survey | contradicts the grid assumption of every prior section |
| Native | Source grid preserved; varying shapes need bucketing | ShapeBucketSampler bucket list | supplies the alternative to Resample's interpolation |
| Boxes | Antimeridian box resolves to two segments | segment count | narrows how any of the above resolve geographically |
| Normalisation | Statistics are experiment-level, applied through valid_mask | zero-fill vs masked mean | depends on the masks established by Resample/sparse sections |

Every dependency is a real relation, so the section order holds. Per the user's
decision (option b), the parallel definition-first structure stays; the fix is
S5/S6 — vary how sections open and vary rhythm, since 9 of 9 currently open
"definition + so/because consequence" and 15 of 23 paragraphs close on a 20-32
word sentence.

Openers to vary (keep ~3 definition-first, since it is a reference):
- Resample: keep definition-first (it is the baseline every later section varies from)
- VectorPair: open on the failure it prevents (mixed-validity direction)
- Sparse+dense: open on the problem (what a missing cell means)
- Points: open on the data's shape (profiles, not a field) — already close
- Native: open on the cost Resample pays, i.e. dependency-stating transition
- Normalisation: open on the choice being the experiment's, not the loader's
