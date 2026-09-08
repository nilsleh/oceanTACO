# Notebook remediation plan

Follow-up to `notebook-restoration-plan.md`. That document set the quality bar
and audited the regression; this one sequences the remaining work against the
**current working tree** (2026-08-31) and records the decisions taken.

**Phase 1 was redesigned on 2026-08-31.** The earlier draft made the four
`OCEANTACO_*` environment variables optional and added a publish script; the
current design deletes them outright and moves the work into the library, so
that a reader needs only `pip install` plus a preamble that names no paths.
See "Decisions taken" and Phase 1.

Worktree: `/p/project1/hai_uqmethodbox/nils/oceanTACO-pr1`
Originals for reference: `/p/project1/hai_uqmethodbox/nils/oceanTACO/docs/tutorials`

## Where things actually stand

The working tree is a **partial** restoration. Content is much improved over
`HEAD` (which still holds the 7-cell stubs the audit flagged), but it falls
short on several of the restoration plan's own non-negotiables.

| Notebook | original | working tree | committed outputs |
| --- | --- | --- | --- |
| `ml_dataset` | 29 cells / 1812 md-words / 3 figs | 19 / 243 / — | **none** |
| `spatio_temporal_query_generation` | 17 / 275 / 3 | 15 / 82 / — | **none** |
| `data_retrieval_workflows` | 25 / 405 / 0 | 18 / 79 / — | **none** |
| `ml_configuration_cookbook` | (new) | 21 / 72 / 6 figs | 14 outputs |
| `plot_hurricane_milton` | 6 / 41 / 2 | 6 / 54 / — | **none** |
| `..._cross_product` | 5 / 42 / 1 | 4 / 43 / — | **none** |

Five of six notebooks have `execution_count: None` throughout.
`docs/conf.py` sets `nb_execution_mode = "off"`, so Sphinx renders **stored**
outputs — the published docs would show empty code cells.
`docs/tutorials/execution-report.local.json` claims all six executed cleanly,
but it predates the current edits (report 08-29, notebooks 08-31) and
disagrees with the files (it claims 9 code cells for `ml_dataset`; there are 12).

### Two findings that shape everything below

**1. The notebooks are generated.** `scripts/dev/restore_tutorial_notebooks.py`
is the source of truth; it reproduces five of six `.ipynb` files exactly.
**All content fixes go in the generator, never in the `.ipynb` files** —
direct edits are silently reverted on the next run.

The cookbook has already drifted: the generator emits 22 cells, disk has 21
(generator adds a `## Retrieved sample gallery` heading and a
Native-vs-`Resample` figure; disk has an older `ShapeBucketSampler` bar chart).
The cookbook is also the *only* notebook with committed outputs, so
**running the generator today destroys those 6 figures.** Regeneration and
execution must happen as one operation, never committed in between.

**2. The preamble is one constant, not six copies.** The ~20-line block
duplicated across six notebooks is a single `SETUP` string at
`restore_tutorial_notebooks.py:23`, rendered six times. There is no source
duplication to remove — the one copy is simply wrong. No helper module is
needed; adding one would push docs scaffolding into the shipped public API.

### Decisions taken

- **One revision pin covers both catalog and QuerySets.** Re-pin
  `CORE_DATASET_REVISION` from `878befc4…` to
  `95a7cfca2723f5f3b3d55592520651ce1c4a55c4` (head of `main`). See
  "Revision" below for the evidence and why this is output-neutral.
- **Remote assets are fetched with `hf_hub_download`,** not the hand-rolled
  `requests.get` retry loop. This is what removes the cache directory from the
  reader's vocabulary entirely.
- **`QuerySet.from_hub()` replaces environment-variable path assembly.**
- **The 256 km sets are what the notebooks use.** The notebooks currently
  hardcode `PATCH_SIZE_KM = 512`; **switch to 256**. The other four sets are not
  uploaded, so nothing in the tutorials may reference 128 or 512.
- **Prose** target ~1200–1500 words for `ml_dataset`.
- **Out of scope, by decision:** `validate_release_evidence.py`, the
  `dataset_revision` sentinel reconciliation, and any queryset rebuild.

---

## Environment

Maintainer work (regenerating and executing notebooks) requires the project
venv. It **must be sourced**, and `PYTHONPATH` cleared afterwards — the
activation script prepends its own `site-packages`, which shadows the worktree
copy of `ocean_taco`:

```sh
source /p/project1/hai_uqmethodbox/nils/oceanTACO/sc-venv-template-uv/activate.sh
unset PYTHONPATH
cd /p/project1/hai_uqmethodbox/nils/oceanTACO-pr1
```

Provides Python 3.12 with `nbformat`, `nbclient`, `xarray`, `torch`,
`matplotlib`, `cartopy`, `pyarrow`, and `huggingface_hub` 1.29.0
(with the `hf` CLI; `huggingface-cli` is deprecated in 1.x).

**The four `OCEANTACO_*` variables are deleted, not made optional.** After
Phase 1 the notebooks read no project-specific environment at all. The only
variable that remains relevant is the standard `HF_HOME`, and it is guidance
rather than a requirement:

```sh
export HF_HOME=/path/with/space/hf-home    # optional but recommended
```

The catalog is **291 GB** (measured on the cluster copy). Tutorial draws touch
only the granules their eight rows need, but HF's default
`~/.cache/huggingface` is frequently on a small or quota-limited volume, so
readers should point `HF_HOME` somewhere with room. On this cluster keep it in
project space — not `/p/home` (tight quota), not `/tmp` (node-local).

### One access mode, two levels of completeness

`hf_hub_download` preserves the repository layout. A fetched granule lands at:

```
$HF_HOME/hub/datasets--nilsleh--OceanTACO/snapshots/<revision>/DATA/<date>/<region>/l3_ssh.nc
```

`snapshots/<revision>/` is a symlink tree mirroring the repo exactly (content
is deduplicated in `blobs/`). Verified 2026-08-31 by fetching one granule and
opening it with `h5netcdf`.

The consequence is worth teaching in the tutorials: **a populated snapshot
directory is a valid `taco_path`.** It has the same
`COLLECTION.json` / `DATA/` / `METADATA/` shape as the 291 GB cluster catalog,
just with fewer granules present. Remote access and a full local copy are not
two mechanisms — they are the same layout at different levels of completeness,
which is also why `_is_local_location` (`ocean_taco/retrieve.py:135`) decides
per catalog row rather than from configuration.

### Revision

`list_repo_files` at the previously pinned `878befc4…` returns **zero**
`querysets/` files: the 256 km sets were uploaded to `main` afterwards. The
old plan's assumption that one pin covered both artifacts was wrong.

Re-pinning to `95a7cfca2723f5f3b3d55592520651ce1c4a55c4` is verified
output-neutral at blob level (2026-08-31):

| check | result |
| --- | --- |
| `DATA/` paths identical | yes |
| `DATA/` blobs differing | **0** |
| `COLLECTION.json` identical | yes |
| files added | the 8 queryset files, and nothing else |

The new revision is a strict superset, so no notebook's scientific output can
change. The only cost is one re-download of already-cached granules, because
the revision is part of every cache key.

### The published QuerySets (256 km)

| | `256-eval` | `256-training` |
| --- | ---: | ---: |
| positions | 6027 | 11001 |
| dates | 858 (2023-03-29 → 2025-08-02) | 858 (same range) |
| grid spacing | 230.4 km | 170.67 km |
| `kind` | `eval` | `training` |
| size | 22 MB | 40 MB |

Both carry `tokens = ["argo", "l3_ssh", "l3_swot"]` and
`patch_size = {value: 256.0, unit: "km"}`.

Consequences for the notebooks:

- Set `PATCH_SIZE_KM = 256`, one named constant driving both
  `from_hub(PATCH_SIZE_KM, "eval")` and `from_hub(PATCH_SIZE_KM, "training")`.
- **The populations are ~4× larger than 512** (6027 vs 1488 eval positions).
  Draws stay capped at eight rows, so cost is unchanged, but any figure that
  scatters the *whole* population needs its marker size and alpha retuned — the
  current `s=2..3, alpha=.15..4` settings were chosen against 1488 points and
  will read as a solid smear at 6027/11001. Applies to `ml_dataset` cells 16-17
  and `spatio_temporal` cell 4.
- The filter-funnel figure's "published pairs" bar becomes
  `6027 × 858 ≈ 5.2 M` pairs. The log scale already handles it; the printed
  counts simply change.
- `GeoBox(-80, -30, 10, 45)`, used as the split box throughout, is unchanged in
  meaning and will now select proportionally more candidates.

Because the source dates are identical between the two sets, the temporal guard
band in `spatio_temporal` §4 remains the meaningful split axis, and the
"a QuerySet kind is not a leakage proof" warning stays exactly as relevant.

---

## Phase 0 — Safety net (before touching the generator)

The cookbook's 6 PNGs exist only in the working tree and are about to be
overwritten.

1. Copy all six `.ipynb` to a scratch directory outside the repo.
2. Export the originals' figures from `../oceanTACO/docs/tutorials/` to the same
   place. Restoration plan §1 makes side-by-side visual comparison an explicit
   acceptance gate, and it is impossible after regeneration.
3. Reconcile the cookbook drift deliberately: **keep both** figures. The
   Native-vs-`Resample` comparison is a real recipe result; the bucket chart is
   the only visual for `ShapeBucketSampler`. Pass `shuffle=False` there —
   `ShapeBucketSampler.__iter__` shuffles by default
   (`ocean_taco/torch/sampler.py:55`), so the printed bucket list is otherwise
   seed-dependent and unexplainable.

## Phase 1 — Zero-config data access

The goal is a preamble that names no paths, no cache, and no environment
variables. Reaching it requires small library changes; they are the point, not
a detour, because every workaround the notebooks perform today exists to route
around a gap in the shipped package.

### 1.1 `ocean_taco/catalog.py`

- `CORE_DATASET_REVISION = "95a7cfca2723f5f3b3d55592520651ce1c4a55c4"`.
- `cache_dir` keeps its `None` default and its meaning as an **optional
  override**; `hf_hub_download` handles caching when it is unset.

### 1.2 `ocean_taco/retrieve.py` — the one hot-path change

Replace the `fetch()` closure in `_download_dataset` (~line 157) and
`_download_location` (~line 192). Both currently hand-roll a `requests.get`
retry loop against the Hub.

- Derive the repo-relative path by stripping the
  `https://huggingface.co/datasets/{repo_id}/resolve/{revision}/` prefix that
  `resolved_catalog_url` (`catalog.py:47`) built. The suffix is exactly
  `hf_hub_download`'s `filename`, so no path mapping is required — these are
  URLs we construct ourselves.
- Call `hf_hub_download(repo_id=…, filename=<suffix>, revision=…,
  repo_type="dataset")`, then `xr.open_dataset(path, engine="h5netcdf")`.
- Keep the existing `requests.get` path as a fallback for any non-Hub HTTP URL,
  so a third-party mirror still works.
- Leave `_is_local_location` untouched — local catalogs already open assets in
  place and must keep doing so.

Keep `LocalCacheBackend` (`ocean_taco/access/local.py`). Its fork-safe HDF5
handle LRU (`_reset_after_spawn`, `max_open_files`) is `DataLoader`-worker
safety, orthogonal to caching, and not something `hf_hub_download` provides.

### 1.3 `ocean_taco/torch/loader.py`

Delete the `cache_dir is None` raise at `loader.py:113`. It is a policy
assertion, not a technical requirement — the retrieval layer below handles
`None` fine, and with `hf_hub_download` there is always a cache. Update the
companion message at `dataset.py:138`.

### 1.4 `ocean_taco/manifest.py` — `QuerySet.from_hub()`

```python
@classmethod
def from_hub(cls, patch_km: int = 256, kind: str = "eval", *,
             repo_id: str = CORE_DATASET_REPO_ID,
             revision: str = CORE_DATASET_REVISION) -> QuerySet:
```

`snapshot_download(allow_patterns=f"querysets/{patch_km}-{kind}/*")`, then
delegate to the existing `read()`. `read()` keeps its local-path-only contract
and its sha256 verification unchanged — `from_hub` is a thin locator on top.
Import `huggingface_hub` lazily inside the method, matching how `load_catalog`
defers its `tacoreader` import.

### 1.5 `pyproject.toml`

- Move `huggingface_hub` from the `[hf]` extra into core `dependencies`. It is
  now on the default retrieval path, so leaving it optional would make the base
  install broken by default. (It was previously declared but imported nowhere
  in `ocean_taco/` — the extra had no consumer.)
- Add `nbformat` to `[tutorials]` and make that extra self-sufficient
  (`tutorials = […, "ocean_taco[viz]"]`). `docs/tutorials/index.md` tells
  readers to install it, but it currently pulls neither `matplotlib` nor
  `cartopy`.
- **Distribution name.** `pip install oceantaco` does **not** resolve: PEP 503
  normalizes `_`, `-`, and `.` alike, so `ocean_taco` and `ocean-taco` are the
  same name but `oceantaco` is a different one. Either document
  `pip install ocean-taco`, or rename the distribution to `oceantaco` now —
  cheap, since nothing has been pushed to PyPI yet. Decide before publishing;
  the docs must match whichever is chosen.

### 1.6 The preamble

`SETUP` (`scripts/dev/restore_tutorial_notebooks.py:22-49`) and `LOAD`
(lines 51-63) collapse to roughly:

```python
from pathlib import Path
from io import BytesIO
from IPython.display import Image, display

from ocean_taco import CatalogConfig, QuerySet
from ocean_taco.retrieve import load_hf_dataset

def display_figure(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=144, bbox_inches="tight")
    display(Image(data=buffer.getvalue()))

PATCH_SIZE_KM = 256          # the published patch size
REQUESTED_ROWS = 4
SEED = 7

config = CatalogConfig()
catalog = load_hf_dataset(config)
queryset = QuerySet.from_hub(PATCH_SIZE_KM, "eval")
print(f"queryset_id={queryset.queryset_id}; kind={queryset.header['kind']}; "
      f"positions={len(queryset.positions)}; dates={len(queryset.dates)}")
```

Deleted outright: `PACKAGE_ROOT` and `sys.path.insert` (unnecessary for an
installed package, and actively harmful in Colab where the absolute path does
not exist); `CACHE_DIR`; `QUERYSET_ROOT`; `CATALOG_PATH`; `QUERYSET_DIRECTORY`;
the pre-`read` `header.json` load; `revision=header["dataset_revision"]`; and
the `artifact_revision`/`header_sha256` print.

That print is the publication-verification ceremony, and it earns nothing:
`QuerySet.read` already recomputes and verifies every table's sha256 and raises
on mismatch (`ocean_taco/manifest.py:601-604`), so a successful load is itself
the integrity proof. Printing twelve hex characters demonstrates nothing
further. Passing `revision=header["dataset_revision"]` was also actively wrong —
it overrode the pinned catalog revision with the artifact's
`local:0.1.0+…` build string, which is not a git SHA and poisons cache keys.

Same edit in the two Milton notebooks (generator lines ~595, ~632), which build
their own `CatalogConfig` and currently inherit `QUERYSET_ROOT`, `SEED`,
`PATCH_SIZE_KM` and `REQUESTED_ROWS` that they never use.

`ml_dataset` §3 and `spatio_temporal` §1 use
`QuerySet.from_hub(PATCH_SIZE_KM, "training")`.

**All content edits go in the generator, never in the `.ipynb` files** —
direct edits are silently reverted on the next run.

### 1.7 Docs, same commit

- `docs/tutorials/index.md` — stop presenting `scripts/dev/execute_notebooks.sh`
  as the reader entry point (it sources a hardcoded JSC venv at line 6; it is a
  maintainer tool). Document `HF_HOME` and the snapshot-is-a-catalog point from
  "One access mode" above. Note that the notebooks are generated.
- Replace every `pilot10` path: `getting_started.md:48`,
  `dataset-ml-loader.md:8,10,25`, `train-eval-splits.md:9,21,22`. Note
  `getting_started.md:48` points at `release/querysets/pilot10/512-eval`, a
  path that does not exist in this repo — replace with
  `QuerySet.from_hub(256, "eval")`.
- `getting_started.md:25` no longer needs to tell users to invent a `cache_dir`.

## Phase 2 — Remaining QuerySet uploads

**The 256 km sets are published and are all the tutorials need.** Phase 1
consumes them through `QuerySet.from_hub()`; no publish script is on the
critical path, and the `QuerySet.read`-accepts-a-repo-id idea from the earlier
draft is superseded by `from_hub()`.

Remaining and optional: `{128,512}-{eval,training}`, at
`../oceanTACO/release/querysets/v1/` (264 MB across six sets). Upload them only
if the tutorials are later meant to demonstrate multiple patch sizes; until
then nothing in the docs may reference 128 or 512. The manual procedure is in
the appendix. **Exclude `_work/`** — a 334 MB `shards` directory sits inside
the same tree and must not be published.

**Known inconsistency, deliberately not fixed:** published headers keep
`dataset_revision = "local:0.1.0+71c238b6e2fd19bd"` while the notebooks use the
pinned catalog default. With the preamble no longer reading or printing that
field, the mismatch is invisible to readers; reconciling it would require a
queryset rebuild (a 2 h / 48 CPU / 90 GB Slurm job), which is out of scope.

## Phase 3 — Weave the appended figures into the narrative

Figures are currently bolted on after the narrative ends. This is reordering
elements in the generator's `write(...)` lists.

**`ml_dataset`** — population scatter → §3 after `select_queryset`; draw map →
§4 after `draw_queryset`/`replay_experiment`; availability bar chart → §6 after
the `DataLoader`/collate cell. The last is also a correctness fix: it reads
`batch["availability"]` three cells after that variable is introduced.

**`data_retrieval_workflows`** — registry barh → §1; decimated SST image → §3
after `load_bbox_nc`; time-step bar → §4 after
`load_multisource_time_series_nc`; Argo/antimeridian → §5. Then **move the
`.close()` cell to last**. Cells 14–17 currently use `sst`, `stack`, `argo`
*after* close. Verified: this xarray version lazily reopens rather than raising,
so it is not a crash — but it teaches "close, then keep using", the opposite of
the intended lesson.

**`spatio_temporal_query_generation`** — filter-funnel chart → §3 after the
box/coverage cell (it consumes `base` and `covered` defined there); date
timeline → §5, which is explicitly about cadence and currently has no figure.

## Phase 4 — Prose expansion

Target ~1200–1500 md-words for `ml_dataset` (from 243). Not the original's 1812:
that includes install/imports boilerplate modern packaging makes obsolete, and
restoration plan §2 asks for "comparable depth", not parity. Source from
`../oceanTACO/docs/tutorials/ml_dataset.ipynb`:

- **§2 "Geography comes before pixels" (47w → ~350w).** Port original cell 16,
  the 424-word resolution chain — the single most valuable prose in the old set.
  Its five steps survive; only API names change: bbox → `PatchSize.to_degrees`;
  native-resolution table verbatim (still factually correct, including the SWOT
  sparse-swath paragraph); `default_patch_size` → `Resample` vs `Native`;
  `_interpolate_to_patch` → `support_threshold` (the current mechanism for the
  same idea, and otherwise undocumented anywhere in the tutorials); the
  model-design implications essentially unchanged.
- **§0 intro (66w → ~180w)** — original cell 0, plus the
  population/filter/draw/sample four-stage distinction the current text only
  names.
- **§3 train/eval (37w → ~150w)** — originals 9 and 12 for random-vs-systematic;
  keep the leakage warning.
- **§5 schema (37w → ~180w)** — original cell 20. Enumerate `data`,
  `source_valid`, `support_mask`, `valid_mask`, `support`, `lat`/`lon`, `times`,
  and the structural-absence contract (`Resample.empty()` → `(0,H,W)`,
  availability `False`) rather than a dropped key.
- **§6 collation (40w → ~200w)** — original cell 23, extended with
  `ShapeBucketSampler`, worker/planning semantics (`plan()` resolves catalog rows
  before fork, `ocean_taco/torch/dataset.py:143`), and the
  no-implicit-normalisation contract.
- **New closing (~120w)** — original cell 25's persistence guidance: QuerySet ID,
  header `table_sha256`, filter, draw record, renderer settings, normalisation
  statistics.

`spatio_temporal` (82w) and `data_retrieval` (79w) need proportionally less.
Every `## N.` heading in those two is currently a bare title — give each a
40–80 word paragraph explaining *why* before the code.

## Phase 5 — Cookbook recipes show results, not claims

Generator cells 4, 6, 8, 12, 14 build a config dict and `print()` a prose claim.
Replace each with an observed result, reusing the `recipe_dataset` /
`recipe_draw` machinery already present in cells 15–20.

- **Fixed grids / fusion** — show real `tuple(sample[token]["data"].shape)`; one
  tensor shape retires `print("layout: dense [T,H,W]…")`.
- **`VectorPair`** — the largest gap: it never produces a sample anywhere in the
  docs. Build `{"velocity": VectorPair(Resample((64, 64), 0.5))}`, show the
  `(T,2,H,W)` shape and shared availability
  (`ocean_taco/torch/dataset.py:227-231`), and plot both components via
  `plot_ocean_sample(..., component=0/1)`.
- **Argo** — move the existing gallery figure adjacent to its recipe.
- **Forecasting** — currently draws rows but never renders; build the two
  datasets the prose recommends and show that their context windows differ.
- **`Native` / `ShapeBucketSampler`** — pair with the Native-vs-`Resample`
  figure plus the bucket chart (`shuffle=False`).
- **Normalisation** — apply `normalise_valid` to a real `sample[token]["data"]`
  with its `valid_mask`, not a hand-built 3-element tensor.

Each recipe becomes: heading → prose stating the model contract and empty-data
behaviour → minimal code → observed output. This dissolves the "gallery"
section, which is the same antipattern as Phase 3.

## Phase 6 — Library bug fixes (separate commit)

- **`ocean_taco/plot.py:46-49`** indexes before it bounds-checks:
  `image = data[time_index]` runs before the `time_index >= data.shape[0]`
  guard, leaving the guard unreachable on the `ndim == 3` path. With
  `Resample.empty()` returning `(0,H,W)`, this raises `IndexError` instead of
  the intended `ValueError`.
- **`ocean_taco/viz/paper/plot_hurricane_milton.py`** —
  `axes[row, 0].set_ylabel(date)` does not render on a cartopy `GeoAxes`, so the
  date row labels are invisible; use `axis.text(..., transform=axis.transAxes)`.
  And wind speed is computed from `eastward_wind_max`/`northward_wind_max` while
  the quiver uses `eastward_wind`/`northward_wind`, so the colour and the arrows
  depict different quantities. Both are only visible once executed.

Note: contrary to restoration plan §6, the hurricane helpers are in good shape —
Mercator projection, `cfeature.LAND`, coastlines, gridline labels, the full
Milton track, per-date eye markers, two separate colourbars, all three products
requested unconditionally, and no `INCLUDE_DENSE_SWOT` flag. §6 is close to
already satisfied.

## Phase 7 — Execute and verify

Order matters, because of the Phase 0 hazard.

1. `python scripts/dev/restore_tutorial_notebooks.py` — this zeroes the
   cookbook's outputs. **Do not commit here.**
2. `python scripts/dev/execute_tutorial_notebooks.py` immediately. Its
   `REQUIRED_ENV` check (`execute_tutorial_notebooks.py:31`) must be updated —
   it still demands `OCEANTACO_QUERYSET_ROOT` and `OCEANTACO_CATALOG_PATH`,
   which Phase 1 deletes. Preferred over `execute_notebooks.sh`: it uses `allow_errors=True` and
   persists partial output plus a traceback into notebook metadata, which is far
   better for diagnosis. It writes `ocean_taco_execution_failure` into notebook
   metadata on failure — that must not reach a commit.
3. Validate: every code cell has output, none is `output_type == "error"`, and —
   **a new check the current validator lacks** — `execution_count` is non-`None`
   and monotonically increasing. That is what catches the stale-output /
   fresh-source drift Phase 0 found.
4. Visual review against the Phase 0 baseline (restoration plan §1, §7.3).
5. **Clean-install contract — run this first, it is the point of Phase 1.** In
   a fresh venv, `pip install -e ".[tutorials]"`, point `HF_HOME` at an empty
   directory, and run the preamble with **no** `OCEANTACO_*` variables set. It
   must work. Then assert the fetched granule resolves under
   `snapshots/<rev>/DATA/<date>/<region>/` and that `tacoreader` loads that
   snapshot directory as a `taco_path`.
6. **Local path unregressed.** With `taco_path` set to the cluster catalog,
   confirm assets still open in place and nothing is written to any cache.
7. `pytest tests -q`, plus the `remote`-marked tests that
   `pyproject.toml:88` deselects by default — they are exactly the ones
   covering the `hf_hub_download` change. Then
   `sphinx-build -W -b html docs docs/_build/html`; `grep -r pilot10` returns
   nothing outside `_build/`; `grep -r OCEANTACO_QUERYSET_ROOT docs/ scripts/`
   returns nothing.
8. Confirm nothing unwanted is staged: no cache directory, no downloaded HF
   assets, no `docs/_build/`, no standalone PNGs.

**`l4_sst` verification.** The 256-eval header lists
`tokens = ["argo", "l3_ssh", "l3_swot"]` with no `l4_sst` (identical to 512 in
this respect), yet `ml_dataset`'s
flagship figure renders `l4_sst`. Traced: the header token list gates only the
coverage/assets *fact tables* (`ocean_taco/manifest.py:400,428`); rendering
resolves through the modality registry and catalog
(`ocean_taco/torch/loader.py:161-180`), and `l4_sst` is registered
(`ocean_taco/registry.py:86`) with assets on disk. **Expected outcome: it
renders.** Confirm during execution by asserting `availability["l4_sst"] is
True`. If false, swap the flagship figure to `l3_swot`, and the cookbook's
`l4_sst`/`l4_ssh` to `l3_ssh`/`l3_swot`.

**Housekeeping.** `execution-report.local.json` is untracked, stale, disagrees
with the files, embeds two absolute cluster paths, and — verified via
`git check-ignore` — is **not** gitignored despite the `.local.` infix. Delete
it and add `docs/tutorials/*.local.*` to `.gitignore`; provenance already lives
in the notebooks' own printed output.
`scripts/dev/restore_tutorial_notebooks.py` and
`scripts/dev/execute_tutorial_notebooks.py` are likewise untracked and
unignored — **commit both**. The generator is now the source of truth for six
documents; leaving it untracked guarantees the next person edits the `.ipynb`
files and loses the work.

**Commit split** (restoration plan §7.6): (1) library zero-config access —
revision re-pin, `hf_hub_download`, `from_hub()`, `pyproject` deps;
(2) collapsed preamble + doc path fixes; (3) library bug fixes; (4) generator
content restoration — prose, figures, recipes; (5) executed output refresh.

---

## Open risks

- **`hf_hub_download` on the retrieval hot path.** This is the one behavioural
  change reaching non-tutorial users. Mitigated by the non-Hub HTTP fallback
  and by the local-path check in Phase 7.
- **Mirror vs. HF byte-identity still unverified.** The *revision* question is
  settled (0 differing `DATA/` blobs between the old and new pins), but the
  local cluster mirror at
  `/p/project1/hai_uqmethodbox/data/new_ssh_dataset_taco_folder/OceanTACO` was
  never checksummed against the Hub. Dropping `taco_path` will change committed
  outputs if the two diverge — worth discovering, but it will look like a
  regression.
- **Hurricane cost unprofiled.** Four dates with L3 SWOT unconditionally on.
  Restoration plan §1 asks for this profile and nobody has produced it; it
  determines whether these two notebooks can ever live in CI.
- **Cartopy offline data.** The old helper set `CARTOPY_USER_DIR`; the rewrite
  dropped it, so `coastlines()` / `cfeature.LAND` may attempt a network fetch at
  execution time. The old tree cached shapefiles in `docs/tutorials/.cartopy/`.
- All runtime claims here come from source reading. Five of six notebooks have
  never been executed in their current form, so unknown failures are likely on
  the first real run.

---

## Appendix — Uploading the QuerySets by hand

**The 256 km sets are already uploaded**; this procedure is retained for the
remaining `{128,512}-{eval,training}` sets, should they be needed. Everything
below is run by the maintainer.

**Any upload moves `main`.** Because `CORE_DATASET_REVISION` is now pinned to a
specific commit, a new upload is invisible to the library until the pin is
advanced. That is the desired behaviour — reproducibility by default — but it
means publishing 128/512 is a two-step operation: upload, then re-pin and
re-run Phase 7.

### 1. Environment and authentication

```sh
source /p/project1/hai_uqmethodbox/nils/oceanTACO/sc-venv-template-uv/activate.sh
unset PYTHONPATH
hf auth login          # paste a token with WRITE scope on nilsleh/OceanTACO
hf auth whoami         # confirm
```

No token is currently configured on this account (`~/.cache/huggingface/token`
is absent and `HF_TOKEN` is unset). Create one at
<https://huggingface.co/settings/tokens>. Prefer a short-lived fine-grained
token scoped to the single dataset repo. Do not commit it or paste it into a
notebook.

### 2. Check what will be uploaded

```sh
export QS=/p/project1/hai_uqmethodbox/nils/oceanTACO/release/querysets/v1
du -sh "$QS"/*/            # six sets, 264 MB total
du -sh "$QS"/_work         # 334 MB — must NOT be uploaded
ls "$QS"/256-eval          # header.json + 3 parquet files
```

### 3. Upload, one set at a time

Each set goes to `querysets/<patch>-<kind>/` at the repo root — no `v1` or
`pilot10` parent.

```sh
for set in 128-eval 128-training 512-eval 512-training; do   # 256-* already published
  hf upload nilsleh/OceanTACO "$QS/$set" "querysets/$set" \
    --repo-type dataset \
    --include "*.json" --include "*.parquet" \
    --commit-message "Add $set QuerySet artifacts" \
    --create-pr
done
```

Notes:

- `--repo-type dataset` is required; the CLI defaults to `model`.
- The explicit `--include` filters are a second guard against `_work/`; because
  each upload targets one set directory, `_work/` is already outside the tree
  being pushed. Verify with `du` beforehand regardless.
- `--create-pr` opens a PR per set instead of committing straight to `main`, so
  the upload is reviewable and revertible. Drop it only once you are satisfied
  with a dry run.
- 264 MB over six commits; expect several minutes. `hf upload` is resumable, so
  a failed set can simply be re-run.

### 4. Verify from a clean cache

```sh
python - <<'PY'
from ocean_taco.manifest import QuerySet
qs = QuerySet.from_hub(256, "eval")
print(qs.queryset_id, qs.header["kind"], len(qs.positions), len(qs.dates))
PY
```

`QuerySet.read` recomputes every table's sha256 against `header.json`, so a
successful read is also an end-to-end integrity check of the upload. Expect `queryset_id` starting `2b55f024`, kind `eval`, 6027 positions,
858 dates. The check is not instant — the coverage
table is the bulk of the 22 MB and is hashed in full.

### 5. After an upload lands

- Advance `CORE_DATASET_REVISION` to the new head of `main`, and verify the
  `DATA/` blobs are unchanged before doing so (see "Revision" above for the
  comparison used for the 256 upload).
- Re-run Phase 7 so the committed outputs reflect the published artifacts.
- Update `docs/getting_started.md`, `docs/dataset-ml-loader.md`, and
  `docs/train-eval-splits.md` to the published `querysets/<patch>-<kind>` path,
  using **256** in every example until the other sizes are published.
