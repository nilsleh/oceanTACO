"""Assert the 0.1 wheel contains only the documented public package surface.

Run this after ``python -m build``.  It intentionally inspects the wheel rather
than relying on setuptools' package-discovery configuration: a top-level
``test_*.py`` module can otherwise slip into a distribution even when its
subpackage is excluded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

_REQUIRED = {
    "ocean_taco/__init__.py",
    "ocean_taco/catalog.py",
    "ocean_taco/geobox.py",
    "ocean_taco/manifest.py",
    "ocean_taco/filter.py",
    "ocean_taco/registry.py",
    "ocean_taco/retrieve.py",
    "ocean_taco/temporal.py",
    "ocean_taco/access/__init__.py",
    "ocean_taco/render/__init__.py",
    "ocean_taco/sampling/__init__.py",
    "ocean_taco/sampling/coverage.py",
    "ocean_taco/sampling/data/ocean_mask_0p1deg_60S_60N.npz",
    "ocean_taco/sampling/draw.py",
    "ocean_taco/sampling/grids.py",
    "ocean_taco/sampling/publish.py",
    "ocean_taco/torch/__init__.py",
}
_EXCLUDED_PREFIXES = (
    "ocean_taco/benchmarks/",
    "ocean_taco/dataset/",
    "ocean_taco/generate_dataset/",
    "ocean_taco/viz/",
    "tests/",
)
_EXCLUDED_FILES = {
    "ocean_taco/preflight.py",
    "ocean_taco/splits.py",
    "ocean_taco/sampling/generate.py",
}


def inspect_wheel(path: Path) -> list[str]:
    """Return all release-boundary violations in ``path``."""
    with ZipFile(path) as archive:
        names = set(archive.namelist())
    errors = [
        f"wheel is missing required file: {name}" for name in sorted(_REQUIRED - names)
    ]
    for name in sorted(names):
        if name.startswith(_EXCLUDED_PREFIXES):
            errors.append(f"wheel contains excluded repository-only content: {name}")
        if name in _EXCLUDED_FILES:
            errors.append(f"wheel contains removed superseded API: {name}")
        if name.startswith("ocean_taco/") and Path(name).name.startswith("test_"):
            errors.append(f"wheel contains pytest module: {name}")
    return errors


def main() -> None:
    """Validate one built wheel and exit non-zero on a boundary violation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    arguments = parser.parse_args()
    errors = inspect_wheel(arguments.wheel)
    if errors:
        parser.error("\n".join(errors))
    print(f"wheel boundary verified: {arguments.wheel}")


if __name__ == "__main__":
    main()
