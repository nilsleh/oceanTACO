"""Offline tests for the release-boundary and evidence-gate commands."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from scripts.release.validate_release_evidence import validate
from scripts.release.verify_wheel import _REQUIRED, inspect_wheel


def _write_wheel(path: Path, extra_names: tuple[str, ...] = ()) -> None:
    """Create the smallest wheel-shaped archive needed by the boundary test."""
    with ZipFile(path, "w") as archive:
        for name in _REQUIRED:
            archive.writestr(name, b"")
        for name in extra_names:
            archive.writestr(name, b"")


def test_wheel_inspection_accepts_the_documented_surface(tmp_path):
    wheel = tmp_path / "ocean_taco-0.1.0-py3-none-any.whl"
    _write_wheel(wheel)

    assert inspect_wheel(wheel) == []


def test_wheel_inspection_rejects_legacy_and_pytest_content(tmp_path):
    wheel = tmp_path / "ocean_taco-0.1.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        (
            "ocean_taco/dataset/dataset.py",
            "ocean_taco/preflight.py",
            "ocean_taco/test_accidental.py",
        ),
    )

    errors = inspect_wheel(wheel)
    assert any("repository-only" in error for error in errors)
    assert any("removed superseded API" in error for error in errors)
    assert any("pytest module" in error for error in errors)


def test_release_evidence_template_is_an_intentional_release_stop():
    root = Path(__file__).resolve().parents[1]
    template = json.loads(
        (root / "release/evidence/release-evidence.template.json").read_text(
            encoding="utf-8"
        )
    )

    errors = validate(template, root=root)

    assert any("reference_recipe is not complete" in error for error in errors)
    assert any("ocean_taco.dataset still exists" in error for error in errors)
