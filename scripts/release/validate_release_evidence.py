"""Validate the evidence bundle required by design revision 9, section 11.

The validator is deliberately a release stop, not a generator.  It cannot turn
fixture-only results, a placeholder manifest, or a claimed benchmark into
evidence.  A release owner records paths to committed reports and generated
artifacts; this command verifies that the record is complete and that the
six QuerySets are structurally suitable for the 0.1 release.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ocean_taco.catalog import CORE_DATASET_REVISION
from ocean_taco.manifest import QuerySet

REQUIRED_STEPS = (
    "reference_recipe",
    "core_integration",
    "reference_querysets",
    "runtime_benchmark",
    "paper_reproduction",
    "legacy_removal",
    "packaging",
    "core_pilot",
)
REFERENCE_SETS = {
    "128-training": (128, "training", 44_227),
    "128-eval": (128, "eval", 24_289),
    "256-training": (256, "training", 11_010),
    "256-eval": (256, "eval", 6_043),
    "512-training": (512, "training", 2_722),
    "512-eval": (512, "eval", 1_482),
}
REFERENCE_DATE_COUNT = 858


def _project_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as stream:
        return str(tomllib.load(stream)["project"]["version"])


def _evidence_paths(
    step: Mapping[str, Any], root: Path, name: str, errors: list[str]
) -> None:
    paths = step.get("evidence")
    if not isinstance(paths, list) or not paths:
        errors.append(f"{name} must name at least one committed evidence path.")
        return
    for value in paths:
        if not isinstance(value, str) or not value:
            errors.append(f"{name} has a non-string evidence path.")
            continue
        path = root / value
        if not path.is_file():
            errors.append(f"{name} evidence path does not exist: {value}")


def _step(
    document: Mapping[str, Any], name: str, root: Path, errors: list[str]
) -> Mapping[str, Any] | None:
    steps = document.get("steps")
    if not isinstance(steps, Mapping):
        errors.append("release evidence must contain an object named 'steps'.")
        return None
    value = steps.get(name)
    if not isinstance(value, Mapping):
        errors.append(f"release evidence is missing the {name!r} step.")
        return None
    if value.get("status") != "complete":
        errors.append(f"{name} is not complete.")
    _evidence_paths(value, root, name, errors)
    return value


def _require_fields(
    value: Mapping[str, Any], fields: Sequence[str], name: str, errors: list[str]
) -> None:
    for field in fields:
        if field not in value or value[field] in (None, "", [], {}):
            errors.append(f"{name} is missing required field {field!r}.")


def _validate_querysets(
    value: Mapping[str, Any], root: Path, mask_id: str, errors: list[str]
) -> None:
    entries = value.get("manifests")
    if not isinstance(entries, Mapping):
        errors.append(
            "reference_querysets must map each reference set to its manifest directory."
        )
        return
    if set(REFERENCE_SETS) != set(entries):
        errors.append(
            "reference_querysets.manifests must contain exactly the six released training/eval sets."
        )
        return
    for name, (size, kind, expected_positions) in REFERENCE_SETS.items():
        location = entries[name]
        if not isinstance(location, str) or not location:
            errors.append(f"reference queryset {name} has no manifest path.")
            continue
        directory = root / location
        try:
            queryset = QuerySet.read(directory)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"reference queryset {name} cannot be read: {exc}")
            continue
        if not queryset.positions:
            errors.append(f"reference queryset {name} is empty.")
        if queryset.header.get("dataset_revision") != CORE_DATASET_REVISION:
            errors.append(
                f"reference queryset {name} does not use the pinned Core revision."
            )
        if queryset.header.get("ocean_mask_id") != mask_id:
            errors.append(
                f"reference queryset {name} does not use the recorded ocean-mask identity."
            )
        patch = queryset.header.get("patch_size")
        if (
            not isinstance(patch, Mapping)
            or patch.get("unit") != "km"
            or patch.get("value") != size
        ):
            errors.append(f"reference queryset {name} has the wrong PatchSize.")
        if queryset.header.get("kind") != kind:
            errors.append(f"reference queryset {name} has the wrong grid kind.")
        if len(queryset.positions) != expected_positions:
            errors.append(
                f"reference queryset {name} has {len(queryset.positions)} positions, expected {expected_positions}."
            )
        if len(queryset.dates) != REFERENCE_DATE_COUNT:
            errors.append(
                f"reference queryset {name} has {len(queryset.dates)} dates, expected {REFERENCE_DATE_COUNT}."
            )
        if len(queryset.coverage) != len(queryset.positions) * len(queryset.dates):
            errors.append(
                f"reference queryset {name} does not contain the complete position/date coverage product."
            )
        if not queryset.assets:
            errors.append(f"reference queryset {name} has no staged asset identities.")


def validate(document: Mapping[str, Any], *, root: Path) -> list[str]:
    """Return every unmet section-11 release criterion in ``document``."""
    errors: list[str] = []
    if document.get("schema_version") != 1:
        errors.append("release evidence schema_version must be 1.")
    if document.get("release_version") != _project_version(root):
        errors.append("release evidence release_version must match pyproject.toml.")
    if document.get("dataset_revision") != CORE_DATASET_REVISION:
        errors.append("release evidence must record the pinned CORE_DATASET_REVISION.")

    steps = {name: _step(document, name, root, errors) for name in REQUIRED_STEPS}
    recipe = steps["reference_recipe"]
    if recipe is not None:
        _require_fields(
            recipe,
            (
                "recipe_id",
                "patch_sizes_km",
                "grid_spacings",
                "canonical_dates_sha256",
                "registered_tokens",
                "ocean_mask_id",
            ),
            "reference_recipe",
            errors,
        )
        if recipe.get("patch_sizes_km") != [128, 256, 512]:
            errors.append("reference_recipe.patch_sizes_km must be [128, 256, 512].")
        if recipe.get("grid_spacings") != {"training": "2L/3", "eval": "0.9L"}:
            errors.append(
                "reference_recipe must declare the fixed training/eval grid spacings."
            )
        mask_id = recipe.get("ocean_mask_id")
    else:
        mask_id = ""

    integration = steps["core_integration"]
    if integration is not None:
        _require_fields(
            integration,
            ("command", "environment", "asset_checksums"),
            "core_integration",
            errors,
        )
        if integration.get("dataset_revision") != CORE_DATASET_REVISION:
            errors.append("core_integration must record the pinned Core revision.")

    querysets = steps["reference_querysets"]
    if querysets is not None:
        _require_fields(
            querysets,
            ("regeneration_command", "regeneration_hashes"),
            "reference_querysets",
            errors,
        )
        _validate_querysets(querysets, root, str(mask_id), errors)

    runtime = steps["runtime_benchmark"]
    if runtime is not None:
        _require_fields(
            runtime,
            (
                "command",
                "hardware",
                "cache_states",
                "worker_count",
                "duration_seconds",
                "p50_seconds",
                "p95_seconds",
                "samples_per_second",
                "golden_patchset",
            ),
            "runtime_benchmark",
            errors,
        )
        if runtime.get("access_mode") != "local_immutable_cache":
            errors.append(
                "runtime_benchmark must exercise the shipped local immutable-cache access mode."
            )
        if runtime.get("worker_count") != 4:
            errors.append("runtime_benchmark must use exactly four workers.")

    paper = steps["paper_reproduction"]
    if paper is not None:
        _require_fields(
            paper,
            ("baseline_fixtures", "reproduction_command", "tutorial_command"),
            "paper_reproduction",
            errors,
        )

    legacy = steps["legacy_removal"]
    if legacy is not None:
        if legacy.get("legacy_namespace_absent") is not True:
            errors.append(
                "legacy_removal must explicitly confirm legacy_namespace_absent=true."
            )
        if (root / "ocean_taco" / "dataset").exists():
            errors.append(
                "ocean_taco.dataset still exists; it must be removed after paper reproduction passes."
            )

    packaging = steps["packaging"]
    if packaging is not None:
        _require_fields(
            packaging,
            (
                "base_smoke",
                "hf_smoke",
                "viz_smoke",
                "tutorials_smoke",
                "docs_build",
                "wheel",
                "testpypi_smoke",
            ),
            "packaging",
            errors,
        )

    pilot = steps["core_pilot"]
    if pilot is not None:
        _require_fields(
            pilot,
            (
                "patchset",
                "coverage_distribution",
                "valid_fraction_distribution",
                "point_count_distribution",
                "science_review",
            ),
            "core_pilot",
            errors,
        )
    return errors


def main() -> None:
    """Read a release-evidence JSON file and report all blockers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evidence", type=Path, help="Path to the committed release-evidence JSON file."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve evidence paths.",
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    try:
        document = json.loads(arguments.evidence.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read release evidence: {exc}")
    if not isinstance(document, Mapping):
        parser.error("release evidence must be a JSON object.")
    errors = validate(document, root=root)
    if errors:
        print("Release evidence is incomplete:", file=sys.stderr)
        print(*(f"- {error}" for error in errors), sep="\n", file=sys.stderr)
        raise SystemExit(1)
    print(f"release evidence verified: {arguments.evidence}")


if __name__ == "__main__":
    main()
