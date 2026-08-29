# Notebook restoration plan

## Purpose

Restore the tutorials as rich, visually useful teaching documents while migrating
any obsolete API calls to the current pinned-Core API.  This is a restoration and
modernisation effort, not a reduction-to-smoke-tests effort.

The editorial and visual baseline is the existing tutorial set in
`/p/project1/hai_uqmethodbox/nils/oceanTACO/docs/tutorials`.  The working copy in
this branch must preserve its explanatory depth and figure intent while using the
current APIs and an immutable Hugging Face revision.

## Audit findings (2026-08-29)

The current PR notebooks execute, but execution success hid a serious quality
regression.  The original material was replaced instead of migrated.

| Tutorial | Original | Current PR | Regression to correct |
| --- | ---: | ---: | --- |
| `ml_dataset.ipynb` | 29 cells; 15 prose cells; 439 code lines; 3 figures | 7 cells; 1 prose cell; 53 code lines; 1 figure | Lost the conceptual walkthrough, resolution reasoning, training/evaluation comparison, schema walkthrough, batching explanation, and persistence guidance. |
| `spatio_temporal_query_generation.ipynb` | 17 cells; 6 prose cells; 280 code lines; 3 figures | 7 cells; 1 prose cell; 52 code lines; no figures | Lost parameter intuition, map-based comparisons, overlap/cadence experiments, and practical guidance. |
| `data_retrieval_workflows.ipynb` | 25 cells; 12 prose cells; 147 code lines | 7 cells; 1 prose cell; 30 code lines | Lost the retrieval API map, step-by-step catalog-to-asset explanation, contracts, and edge-case guidance. |
| `plot_hurricane_milton.ipynb` | 4-date, 3-column Cartopy figure with wind vectors, coastlines, track/eye markers, and colourbars | One generic two-product `pcolormesh` image | Lost the paper-quality figure and silently made SWOT opt-in. |
| `plot_hurricane_milton_cross_product.ipynb` | Projected product overlay, storm eye, legend, and L4/L3 correlation + RMSE plot | Separate generic panels and nearest-grid scatter | Lost the actual cross-product comparison semantics and its quantitative summary. |
| `ml_configuration_cookbook.ipynb` | New tutorial required by this branch | One prose cell and terse recipes | Needs to become a real recipe book, not an API inventory. |

The current PR also uses an in-memory QuerySet fallback because the claimed
released QuerySet path is absent at the pinned revision.  A fallback is acceptable
for a temporary diagnosis, but must not be presented as a released-artifact
tutorial or used for final committed output.

## Non-negotiable restoration principles

1. Preserve the original learning journey.  Markdown explains *why*, then a
   small executable cell demonstrates *how*, then a meaningful output or figure
   confirms the result.
2. Preserve visual intent.  Maps must retain geographic context, labels,
   colour/scaling choices, legends, and annotations that make them interpretable.
3. Modernise interfaces, not pedagogical scope.  Replace retired
   `ocean_taco.dataset.*` calls with `CatalogConfig`, `QuerySet`, `GeoBox`,
   `TimeRange`, retrieval, render, and torch APIs without deleting concepts.
4. Use one pinned `CatalogConfig` and one shared cache root in every notebook.
   Configuration constants appear in the first code cell and are printed in the
   output.  No notebook may install from `main` or silently select an unpinned
   catalog.
5. Final tutorial outputs use a real, published QuerySet at the pinned revision.
   The notebook must fail clearly when that artifact is unavailable; it must not
   manufacture a population while claiming to load a release.
6. Keep demonstrations small (at most eight requested QuerySet rows), but do not
   omit a modality or visualization merely to make execution appear cheap.
   Expensive Hurricane/SWOT rendering must have an explicitly measured remote
   execution path and documented resource/cache behaviour.

## Restoration work

### 1. Establish a trustworthy data and visual baseline

- Obtain or publish the actual `v1` QuerySet artifacts first, with a single
  canonical path and header/manifest identity.  Decide whether the public path
  is `release/querysets/v1/<patch-size>-<kind>/`; remove the ambiguous `pilot10`
  wording from tutorials once this is settled.
- Record the pinned revision, artifact path, header checksum, source date range,
  and small selected row IDs in committed notebook output.
- Export the original notebook figures to a review directory outside Git and
  compare them side by side with regenerated PR figures.  Visual review is an
  explicit acceptance gate; successful Python execution is not sufficient.
- Profile the Hurricane source assets individually (download size, peak RSS,
  elapsed time, cache reuse).  Use that evidence to choose the supported execution
  environment rather than defaulting away from L3 SWOT.

### 2. Restore `ml_dataset.ipynb` as the flagship tutorial

Rebuild the original narrative with current interfaces and real release rows:

1. Explain the sample/manifest mental model and the distinction between a
   released QuerySet population, a filter, a recorded draw, and a torch sample.
2. Introduce patch geography and latitude-aware shape implications with a small
   map or schematic that is generated in the notebook.
3. Show train and evaluation QuerySets separately, including a visible
   spatial/temporal split and a warning that a QuerySet kind alone does not
   guarantee leakage freedom.
4. Filter and draw at most eight rows; show the selected positions, dates,
   inclusion probability, record path, and replay check.
5. Build `OceanTACODataset` and show the complete sample schema, including
   empty/missing records, native coordinates, validity masks, and metadata.
6. Visualise a real rendered source with `plot_ocean_sample`; retain a useful
   geographic display rather than a bare tensor dump.
7. Explain `DataLoader`, collation, availability masks, fixed resampling versus
   native shapes, `ShapeBucketSampler`, workers, planning, and the lack of
   implicit normalization.
8. Close with safe NaN-aware normalization code and reproducibility/persistence
   guidance.

Each conceptual section needs explanatory prose, a small runnable example, and
an inspectable output.  The final notebook should be comparable in depth to the
original 29-cell tutorial, not necessarily line-for-line identical.

### 3. Restore query-generation intuition

`spatio_temporal_query_generation.ipynb` keeps the new required topics
(filtering, coverage null-versus-zero, draw/replay, train/eval QuerySets), but
reintroduces visual intuition:

- map several filtered/drawn position sets with clear legends and shared bounds;
- contrast random/drawn training selection with systematic evaluation coverage;
- show how box, date, and coverage filters alter selected populations;
- show temporal cadence/guard-band effects in a small timeline or table;
- explain null coverage as “not measured” and zero as “measured absent” before
  filtering on either;
- retain concrete practical rules and leakage cautions.

### 4. Rebuild the retrieval tutorial as a guided workflow

Keep the original retrieval chapter structure, but map each chapter to public
current APIs only:

- pinned catalog configuration and catalog identity;
- catalog row selection and source-token/registry semantics;
- one tile retrieval, box retrieval/merge, and multisource `TimeRange` retrieval;
- L3 SWOT’s source-specific geometry/shape behaviour;
- antimeridian boxes as two explicit segments;
- native ragged Argo point retrieval;
- cache, `None`, empty result, coordinate, and date-range error contracts.

Use small real requests and display dimensions, variables, coordinate ranges, and
selected catalog evidence.  Do not turn private helper functions into public
teaching APIs merely because the original notebook exposed them.

### 5. Expand the configuration cookbook into annotated recipes

Give every required recipe its own prose heading, minimal complete code block,
and observed result or schema:

- fixed grids;
- multimodal fusion;
- sparse/dense data;
- `VectorPair`;
- Argo points;
- forecasting with two datasets;
- `Native`, `native_shapes`, and `ShapeBucketSampler`;
- regional and antimeridian selection;
- train/eval leakage controls;
- NaN-safe normalization.

Every recipe must state the intended model contract, important failure/empty-data
behaviour, and why the selected renderer/collator is appropriate.

### 6. Restore Hurricane visual workflows before re-executing them

Do not continue using the PR’s generic inline plotting replacement.  Port the
existing paper figure implementations to the current retrieval API, preserving:

- `CatalogConfig` at the pinned revision and shared cache;
- Gulf-of-Mexico projected maps, coastlines, land layer, readable grid labels;
- all three products: L4 wind/L4 SSH, L3 along-track SSH, and L3 SWOT;
- Hurricane Milton track and eye markers;
- original multi-date wind/SSH layout and separate wind/SSH colourbars;
- cross-product overlay, product legend, deterministic subsampling, correlation,
  RMSE, 1:1 line, and labelled axes.

The port belongs in the maintained visualization helpers where it can be tested;
the notebooks should configure, invoke, explain, and display those helpers.
Close datasets and figures after use.  No `INCLUDE_DENSE_SWOT = False` default is
acceptable in the final Hurricane tutorials.

### 7. Execution and review gates

After content restoration:

1. Execute every tutorial against the pinned catalog and published QuerySets.
2. Validate every code cell has a non-error output, but do not add artificial
   `print()` output where a useful displayed result belongs.
3. Inspect every embedded PNG and compare the Hurricane and flagship ML figures
   against the original visual baseline.
4. Confirm no cache, downloaded HF asset, standalone generated PNG, checkpoint,
   or docs build output is staged.
5. Run `pytest tests -q`, `sphinx-build -W -b html docs docs/_build/html`, the
   notebook executor, and the stale-pilot path grep from the original task.
6. Commit content restoration separately from helper/API ports and separately
   from the final executed notebook-output refresh.

## Definition of done

The completed tutorial set reads like an expert-authored learning sequence,
not a set of passing integration probes.  A reader can understand the data model,
choose a configuration, interpret the output figures, and reproduce a small real
experiment at the pinned revision.  The Hurricane notebooks reproduce the
scientific comparisons that their titles promise, including L3 SWOT, and all
committed outputs demonstrate that result.
