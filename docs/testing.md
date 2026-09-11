# Testing and verification

OceanTACO keeps quick deterministic tests separate from tests that need a
local port, Hugging Face, slow I/O, or PyTorch worker processes. Start from the
repository root after activating the project environment and clearing an
inherited `PYTHONPATH`.

```sh
source /path/to/venv/bin/activate
unset PYTHONPATH
pytest tests -q
```

## Pytest tiers

| Tier | Command | Purpose |
| --- | --- | --- |
| Default/offline | `pytest tests -q` | Fast, hermetic contract and fixture tests; excludes `slow` and `remote`. |
| Local | `pytest tests -q -m local` | Tests the guarded local OceanTACO port when it is available. |
| Remote | `pytest tests -q -m remote` | Exercises the pinned Hugging Face catalog path. |
| Slow | `pytest tests -q -m slow` | Longer integration checks. |
| Workers | `pytest tests -q -m workers` | High-value multiprocessing/worker-boundary regressions. |

Marker expressions are composable, for example
`pytest tests -q -m 'remote or workers'`. The default selection deliberately
does not download data.

## Fixture data

The portable test package includes synthetic NetCDF/QuerySet fixtures. They
exercise canonical coordinates, unavailable records, ragged points, collation,
and worker planning without depending on a machine-specific data checkout.
Do not replace them with downloaded Hugging Face assets or commit cache data.

## Local throughput measurement

Loader throughput depends on patch size, output grid, source mix, and worker
count, so it is measured rather than quoted. `scripts/dev/benchmark_local_draw.py`
runs a draw end to end against a local catalog. It is repository tooling and is
not shipped in the wheel, and `--queryset` takes a path to a QuerySet directory
in the checkout rather than a published set name.

```bash
python scripts/dev/benchmark_local_draw.py --taco-path /path/to/OceanTACO \
    --queryset release/querysets/v2/256-training --grid-size 64 \
    --workers 2 --prefetch-factor 1 --epochs 5 --persistent-workers \
    --json-output throughput.json
```

Its defaults are 128 rows, 128 km patches, a 128 × 128 grid, and four workers.
Useful sweeps are the published 128/256/512 km sets, grid sizes 32 through 256,
worker counts 0/2/4/8, and prefetch factors 1 and 2. `--date-end`,
`--context-start`, `--context-end`, and `--shuffle` vary dates and context
windows, while `--sources` also accepts `argo` and `glorys_currents`.
`--diagnostic-patch -90 -56 64` requests one explicit 64 km seam location and
bypasses the coverage-backed draw; repeat the flag for further locations.

The report separates QuerySet reading, drawing, planning and loader
construction, first-batch latency, post-first-batch throughput, delivery and
worker service latency percentiles, file opens, cache hits, and process peak
RSS. RSS is a per-process high-water mark and includes inherited parent memory
under fork. A short prefetched run overstates post-first-batch throughput, so
compare complete epoch rates and repeat each measurement with identical
settings.

## Documentation and notebooks

Treat the documentation build as a warning-free check:

```sh
sphinx-build -W -b html docs docs/_build/html
```

The tutorial notebooks are edited directly and executed against the pinned
Hugging Face revision. Executing rewrites stored outputs, so the result is
reviewed before it is committed:

```sh
python scripts/dev/execute_tutorial_notebooks.py
```

The executor runs with `allow_errors=True`, so a failing cell leaves its
partial output and a traceback in the notebook's metadata for inspection
instead of aborting the run; it returns non-zero if any cell errored. Set
`HF_HOME` to a volume with room first. The script does not add `docs/_build`,
notebook checkpoints, or downloaded assets to the worktree.
