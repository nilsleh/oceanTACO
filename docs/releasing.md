# Releasing

OceanTACO is published to PyPI as `oceantaco`. Publishing is automated: the
{file}`.github/workflows/publish-pypi.yml` workflow runs when a GitHub Release is
published, and uploads through PyPI trusted publishing from the `pypi` environment.
No API token is stored in the repository, and nothing is uploaded from a developer
machine.

The steps below take a reviewed `main` to a published release.

## 1. Verify the release candidate

Run the offline gates from a clean checkout of `main`. These are the same checks CI
runs on every pull request, and all four must pass before tagging.

```sh
ruff check --no-fix ocean_taco tests
pytest tests -q
sphinx-build -W -b html docs docs/_build/html
python -m build && twine check dist/*
```

For a first release of a new major or minor version, also execute the six notebooks in
{file}`docs/tutorials` against the live Hub and run the README and
{doc}`getting_started` snippets verbatim. The cheap checks above all pass on prose
defects — a wrong unit or a stale generated page only shows up when the documented
commands are actually run. Budget roughly an hour and 1.4 GB of downloads, and point
`HF_HOME` at a scratch volume first. An `HF_TOKEN` avoids the unauthenticated Hub rate
limit and makes this substantially faster.

## 2. Update the version

The version is declared in three independent places, and they must agree:

| File | Field |
| --- | --- |
| {file}`pyproject.toml` | `project.version` |
| {file}`ocean_taco/__init__.py` | `__version__` |
| {file}`docs/conf.py` | `release` |

Change all three and open a pull request for it. Nothing checks that they match, so a
missed one ships a wheel whose `ocean_taco.__version__` disagrees with the version pip
resolved.

## 3. Dry run on TestPyPI

Once the version pull request is merged, publish a dry run from the Actions tab:
run the **Publish to TestPyPI** workflow (`workflow_dispatch`) against `main`. It
builds, checks metadata, smoke-installs the wheel into a clean virtual environment and
uploads to TestPyPI from the `testpypi` environment.

Confirm the upload installs and imports on its own:

```sh
python -m venv /tmp/oceantaco-testpypi
/tmp/oceantaco-testpypi/bin/pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  oceantaco
/tmp/oceantaco-testpypi/bin/python -c "import ocean_taco; print(ocean_taco.__version__)"
```

The `--extra-index-url` is required: TestPyPI does not mirror the runtime
dependencies, so without it the install fails resolving `torch` and `xarray`.

A version number can be uploaded to TestPyPI only once. If the dry run reveals a
defect, fix it and dry-run the next version rather than re-uploading the same one.

## 4. Tag and publish

Tag the merge commit on `main` and publish a GitHub Release for it.

```sh
git checkout main && git pull
git tag -a v0.1.0 -m "OceanTACO 0.1.0"
git push origin v0.1.0
```

Then create the release from that tag and write the release notes in its body. The
notes are the only per-version record OceanTACO keeps, so say what changed since the
previous release and call out anything that breaks an existing workflow. Publishing the
release — not saving it as a draft — triggers the PyPI workflow.

```sh
gh release create v0.1.0 --title "v0.1.0" --notes "$(cat <<'NOTES'
First public release. See https://oceantaco.readthedocs.io/en/latest/ for the
tutorials and API reference.
NOTES
)"
```

`--generate-notes` fills the body from the merged pull requests instead, which is a
useful starting point to edit down.

## 5. Confirm the published release

```sh
python -m venv /tmp/oceantaco-release
/tmp/oceantaco-release/bin/pip install oceantaco
/tmp/oceantaco-release/bin/python -c "import ocean_taco; print(ocean_taco.__version__)"
```

Check that the [PyPI project page](https://pypi.org/project/oceantaco/) renders the
README and that Read the Docs has built the new tag.

## Notes

- **PyPI and TestPyPI are configured separately.** They are independent registries with
  independent accounts, and a trusted publisher registered on one grants nothing on the
  other. Each needs its own entry, pointing at its own workflow:

  | Registry | Workflow | Environment |
  | --- | --- | --- |
  | pypi.org | `publish-pypi.yml` | `pypi` |
  | test.pypi.org | `publish-testpypi.yml` | `testpypi` |

  Both are registered for `oceantaco` under repository `nilsleh/oceanTACO`. A mismatch
  in any of those three fields fails at the upload step, after the build and smoke
  install have already passed.
- **A first upload needs a *pending* publisher.** Trusted publishing normally authorises
  a workflow against an existing project, so before a name's first release it has to be
  registered as a pending publisher instead. It converts to an ordinary trusted
  publisher once the first upload lands, and needs no further attention.
- **Releases are immutable.** A version cannot be re-uploaded to PyPI once published,
  and yanking hides a release without freeing the version. Mistakes are corrected by
  releasing a new patch version.
- **The wheel size is a guard.** Both publish workflows fail if the built wheel reaches
  400 KB. The shipped package is around 150 KB, so crossing that bound means
  repository-only code from {file}`tools/` or {file}`scripts/` has leaked into the
  distribution.
