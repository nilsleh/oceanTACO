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

## Documentation and notebooks

Treat the documentation build as a warning-free check:

```sh
sphinx-build -W -b html docs docs/_build/html
```

The tutorial notebooks are generated from
`scripts/dev/restore_tutorial_notebooks.py` and executed against the pinned
Hugging Face revision. Regenerating clears stored outputs, so the two steps run
together and the result is reviewed before it is committed:

```sh
python scripts/dev/restore_tutorial_notebooks.py
python scripts/dev/execute_tutorial_notebooks.py
```

The executor runs with `allow_errors=True`, so a failing cell leaves its
partial output and a traceback in the notebook's metadata for inspection
instead of aborting the run; it returns non-zero if any cell errored. Set
`HF_HOME` to a volume with room first. Neither script adds `docs/_build`,
notebook checkpoints, or downloaded assets to the worktree.
