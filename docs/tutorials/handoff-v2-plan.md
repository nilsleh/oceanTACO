# Handoff: OceanTACO tutorial improvements, round 2

Rewritten 2026-09-07, replacing the 2026-09-02 planning handoff. Round 2 is now
**almost fully implemented and committed**. One item remains, D1.

Read this file, then the relevant item sections of
`docs/tutorials/tutorial-improvement-v2-plan.md`, then wait for the user.

## Where things stand

Everything through 2026-09-07 is committed. `14c9937` "Redesign the tutorial
notebooks and rewrite their prose" sits on round 1's `da4ba37`, and `80d29c0`
"Build a DataLoader in every ML use case and visualise its batch" implements Q2
on top of it. The branch is 29 commits ahead.

| Item | Notebook | State |
| --- | --- | --- |
| M1 footprint zoom | `ml_dataset` | done |
| M2 continent contours | all four | done |
| Q0 catalog repin | all | done |
| Q1 super-resolution rebuild | `spatio_temporal_query_generation` | done |
| C1 prose | `ml_configuration_cookbook` | done |
| X1 prose | `ml_dataset`, `spatio_temporal_query_generation` | done |
| X1 prose | `data_retrieval_workflows` | **blocked on D1** |
| Q2 batch visualisations | `spatio_temporal_query_generation` | done (`80d29c0`) |
| **D1 rethink** | `data_retrieval_workflows` | **open** |

The two Hurricane Milton notebooks are out of scope and their cell source is
byte-identical to the reviewed version.

**Verified at `80d29c0`:** `pytest tests/` gives 90 passed, 7 skipped, 1
deselected. All six notebooks execute with no errors, monotonic non-`None`
execution counts and **24 figures**, up from 17 because Q2 added 7 to the ML
use cases notebook. Both Milton notebooks' cell source is byte-identical to
`HEAD`, `ocean_taco/viz/paper/` is clean, the banned-term grep returns 0, and
`sphinx-build` succeeds with one warning, the known unrelated SWOT xref at
`docs/dataset_description.md:13`.

## The one remaining item

D1 is fully specified in the plan document, and **every open question in it has
already been answered by the user** - see the "Decisions taken 2026-09-02"
table, which governs where the item's own prose still poses a question. Do not
re-ask them.

### D1 — rethink `data_retrieval_workflows` (plan line 290)

The largest remaining item. The notebook has no single subject: its own intro
concedes it is two documents, sections 5-8 restate what `docs/api/remote.md`
already autodocs, and the filtering half never reaches an ML artifact.

Decided: **keep one worked native-retrieval example and cut the rest**, pointing
at `docs/api/remote.md`; **rename to match the new subject** while keeping the
filename, updating `index.md` and the cross-links in the other three notebooks.
Keep §9's antimeridian figure, which is the one good one. End the notebook in a
`DataLoader`, which inverts the current cross-link direction.

**X1 for this notebook waits for D1** — rewriting prose about to be deleted is
wasted. Fold the prose pass into the restructure.

### Q2 — done, and what it leaves for D1

Committed as `80d29c0`. Every use case now builds a `DataLoader`, pulls one
batch of `BATCH_SIZE = 8` and visualises it. 16 code cells and 4 figures became
21 and 11. Three things carry into D1:

- **`batch_panel_grid` is in `PLOTTING`** alongside `batch_member` and
  `batch_dates`, and D1's closing section should reuse it rather than writing a
  fourth panel loop. It takes a collated batch and a token, and a
  `panel_title=lambda batch, index: ...` for per-panel labels.
- **Absent rows are routine and must be guarded.** A row whose source has no
  asset for its date collates with NaN `lat`/`lon`, and `imshow` with an
  `extent=` then raises `ValueError: Axis limits cannot be NaN or Inf`. The
  helper checks `availability` and `np.isfinite` and labels the panel instead.
  Any new figure that indexes a batch directly needs the same guard.
- **Fetch cost is not the constraint the plan feared.** The whole notebook
  executes in ~120 s, because sections reuse three draws rather than making one
  each. D1 can afford a real loader in its closing section.

One correction worth carrying: `support_threshold` is per source, not per
configuration. `l3_ssh` is a nadir track and needs 0.0, since it never covers
half an output cell and silently reports itself unavailable at 0.5.

## Environment, needed before any command

```sh
source /p/project1/hai_uqmethodbox/nils/.oceantaco-nbrun/env.sh
cd /p/project1/hai_uqmethodbox/nils/oceanTACO-pr1
```

That `env.sh` is the whole environment and is what the executor uses. Traps,
all already paid for:

- **Do not add `module load` to it.** That re-injects the cluster's MPI h5py,
  whose `H5Pset_dxpl_mpio` symbol does not resolve against this venv, and every
  granule open then fails with a misleading `No module named 'h5py'`.
- **A bare `cd` in a compound Bash command resets the shell** and drops the
  environment. Re-source it when that happens.
- The repo also carries `env.sh` at its root, which is a *different*,
  older environment. The `.oceantaco-nbrun` one is the one that works.

## Non-negotiable constraints

Full list in the plan under "Constraints that carry over". The four that break
things silently:

1. **Never edit `.ipynb` files.** They are generated from
   `scripts/dev/restore_tutorial_notebooks.py`, now 1841 lines. Direct edits
   are reverted on the next generator run.
2. **Regeneration and execution are one operation.** `docs/conf.py` sets
   `nb_execution_mode = "off"`, so stored outputs *are* the published artifact.
   Committing between `restore_tutorial_notebooks.py` and
   `execute_tutorial_notebooks.py` publishes blank code cells to the docs site.
   This is exactly what went wrong on 2026-09-02; see below.
3. **`SETUP` and `LOAD` stay byte-identical**, which is what keeps the two
   Milton notebooks' generated source identical to the reviewed version. New
   shared helpers go in `PLOTTING`.
4. **Stage selectively.** The tree carries an unrelated SWOT phase-figures
   thread that must not join a tutorial commit: `README.md`,
   `docs/dataset_description.md`, `ocean_taco/registry.py`, four
   `docs/images/swot_phase_*.png`, and `scripts/dev/swot_phase_figures.py`.
   The one Sphinx warning belongs to that thread, not to this work.

## What went wrong on 2026-09-02, so you avoid it

The session ended mid-cycle. The generator ran at 16:32 and execution reached
only one of six notebooks at 16:35, leaving five with **no stored outputs**.
Because of constraint 2 a commit at that point would have published blank code
cells for five tutorials. Nothing had failed; the run simply stopped, and the
plan document still said every item was `open`, so the completed work was
invisible until the diff was read.

Two habits prevent a repeat:

- **Run regeneration and execution as one tracked background task**, then check
  that six `DONE` lines came back with `"errors": []`. Backgrounding through a
  shell wrapper once spawned four concurrent runners writing the same files.
- **Update the plan's status board in the same session as the work**, not at
  the end.

Before executing, you can safely skip regeneration if the notebooks already
match the generator: regenerate into a scratch copy of the repo layout and
compare cell sources. This avoids destroying stored outputs unnecessarily.

## The verification sequence

```sh
# 1. tests. NOTE: use tests/, not a bare pytest -q from the root.
pytest tests/ -q            # expect 90 passed, 7 skipped, 1 deselected

# 2. syntax-gate every code cell (AST-parse) after regenerating, before executing

# 3. regenerate AND execute as one operation, never commit in between
python scripts/dev/restore_tutorial_notebooks.py
python scripts/dev/execute_tutorial_notebooks.py

# 4. banned terms, expect 0
grep -rniE 'leakage|split policy|guard band|\brecipe\b|\bpopulation\b' \
  scripts/dev/restore_tutorial_notebooks.py docs/tutorials/*.ipynb \
  docs/tutorials/index.md | grep -v -- '-plan.md'

# 5. prose sweep
python scripts/dev/prose_audit.py

# 6. Milton source identical to HEAD, and paper viz untouched
git status --short ocean_taco/viz/paper/     # expect empty

# 7. docs build, expect only the known SWOT xref warning
sphinx-build -b html docs docs/_build/html
```

Plus the check no exit code gives you: **every code cell needs a non-`None`,
monotonically increasing execution count**, and every figure should be looked
at.

**`pytest -q` from the repo root does not work.** Six `ocean_taco/test_*.py`
files fail collection because the execution venv has no `aiohttp`. That is
pre-existing breakage unrelated to the tutorials, but it means the root-level
invocation in older plan text is wrong; `pytest tests/` is the suite the
baseline refers to.

## Prose work, if you touch it

`/prose-writing-style` governs, and only it. The Patina list is **not** part of
the skill; `scripts/dev/prose_audit.py` sweeps it as a cheap regression net for
round 1's ban-list, not as a source of instructions.

Read `references/corrections.md` in full before drafting — it is the only
positive model of the target voice. Build the skeleton first.

**Measure each notebook before drafting rather than assuming it matches
`ml_dataset`.** This mattered in practice: `ml_dataset` carried organising
conceits, clefts and short shaped closers, while `ml_configuration_cookbook`
had almost none of those and instead opened all nine sections on one
definition-plus-consequence template. The same recipe would have done nothing
for the cookbook. The C1 skeleton is preserved in the plan's appendix.

Watch for two things that bit this work: the round-1 ban-list includes
**semicolons**, and the audit's "S2 closer" rows include false positives such as
table lead-ins and cross-reference sentences — read them before acting.

## Useful facts

- A working document must be named `*-plan.md` to stay out of the Sphinx build
  (`docs/conf.py:34`). `...-plan-v2.md` does **not** match that glob and would
  publish as a docs page. This file is `handoff-v2-plan.md` for that reason.
- Untracked and deliberately so: this file, the three other plan documents,
  `scripts/dev/prose_audit.py`, and the root `env.sh`. The user has not asked
  for them to be committed.
- Baseline after `14c9937`: `ml_dataset` 11 code cells / 3 figures;
  `data_retrieval_workflows` 16 / 4; `spatio_temporal_query_generation` 16 / 4;
  `ml_configuration_cookbook` 17 / 4; the two Milton notebooks 4 / 1 and 3 / 1.
- Packages added to the venv so execution and the docs build work: `nbformat`,
  `nbclient`, `cartopy`, `h5netcdf`, `sphinx_copybutton`,
  `sphinx_autodoc_typehints`, plus `pip install -e . --no-deps
  --no-build-isolation` so the Jupyter kernel can import `ocean_taco`.
- `cartopy`'s Natural Earth shapefiles are already cached, so contours need no
  network access at execution time.
