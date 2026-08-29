"""Tests for the manifest-local QuerySet world-map diagnostic."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from ocean_taco.filter import QueryFilter, select_queryset
from ocean_taco.geobox import GeoBox, PatchSize, utc_isoformat
from ocean_taco.manifest import QuerySet, content_sha256, position_id
from ocean_taco.viz import queryset_maps


def _queryset(kind: str) -> QuerySet:
    dates = tuple(
        utc_isoformat(f"2024-01-0{day}T00:00:00Z") for day in range(1, 6)
    )
    grid_id = f"fixture-{kind}"
    locations = ((179.7, 0.0), (10.0, 40.0), (-175.0, -10.0))
    positions = tuple(
        {
            "position_index": index,
            "position_id": position_id(
                grid_id=grid_id, centre_lon=longitude, centre_lat=latitude
            ),
            "centre_lon": longitude,
            "centre_lat": latitude,
            "region_mask": 1,
            "swot_footprint_cells": 4,
            "swot_ocean_cells": 3,
            "ssh_footprint_cells": 4,
            "ssh_ocean_cells": 3,
        }
        for index, (longitude, latitude) in enumerate(locations)
    )
    coverage_values = (
        ((1, None, 1, 1, 1), (1, 1, 1, 1, 1)),
        ((1, 0, 1, 1, 1), (1, 1, 1, 1, 0)),
        ((2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
    )
    coverage = tuple(
        {
            "position_index": position_index,
            "date_index": date_index,
            "swot_valid_cells": swot_value,
            "swot_valid_ocean_cells": swot_value,
            "swot_n_obs_sum": swot_value,
            "ssh_valid_cells": ssh_value,
            "ssh_valid_ocean_cells": ssh_value,
            "argo_profile_count": None,
        }
        for position_index, (swot_values, ssh_values) in enumerate(coverage_values)
        for date_index, (swot_value, ssh_value) in enumerate(
            zip(swot_values, ssh_values, strict=True)
        )
    )
    header = {
        "patch_size": {"value": 100.0, "unit": "km"},
        "kind": kind,
        "grid_spacing_km": 66.666666666667 if kind == "training" else 90.0,
        "grid_id": grid_id,
        "dataset_revision": "fixture-revision",
        "catalog_sha256": "catalog",
        "registry_sha256": "registry",
        "source_records_sha256": "records",
        "ocean_mask_id": "mask",
        "ocean_mask_sha256": "mask-hash",
        "dates": list(dates),
        "date_sha256": content_sha256(list(dates)),
        "tokens": ["argo", "l3_ssh", "l3_swot"],
        "parquet_profile": {"writer": "pyarrow"},
        "code_commit": "commit",
        "environment_lock_hash": "environment",
    }
    assets = tuple(
        {
            "date_index": date_index,
            "region": "FIXTURE",
            "token": token,
            "asset_id": f"{token}-{date_index}",
            "uri": f"fixture://{token}/{date_index}",
            "identity_kind": "sha256",
            "identity_value": f"identity-{token}-{date_index}",
            "status": "present",
        }
        for date_index in range(len(dates))
        for token in header["tokens"]
    )
    return QuerySet(
        header=header, positions=positions, coverage=coverage, assets=assets
    )


def _position_indices(positions) -> tuple[int, ...]:
    return tuple(int(position["position_index"]) for position in positions)


def test_default_dates_and_report_rows_use_three_shared_representatives():
    training = _queryset("training")
    evaluation = _queryset("eval")

    dates = queryset_maps.select_common_dates(training, evaluation)
    assert dates == (training.dates[0], training.dates[2], training.dates[-1])

    rows = queryset_maps.build_report_rows(
        training, evaluation, dates=dates
    )
    assert [(row.set_label, row.coverage_label) for row in rows] == [
        ("Training", "All positions"),
        ("Training", "SWOT + SSH covered"),
        ("Evaluation", "All positions"),
        ("Evaluation", "SWOT + SSH covered"),
    ]
    assert all(len(row.panels) == 3 for row in rows)
    assert [panel.count for panel in rows[1].panels] == [3, 3, 2]


def test_region_and_bbox_scenarios_match_queryfilter_selection():
    training = _queryset("training")
    date = training.dates[0]
    ordinary = queryset_maps.SpatialScenario(
        "North Atlantic subset", GeoBox(0.0, 20.0, 30.0, 50.0)
    )
    wrapped = queryset_maps.SpatialScenario(
        "Dateline subset", GeoBox(170.0, -170.0, -20.0, 20.0, True)
    )

    scenarios = queryset_maps.build_spatial_scenarios(
        regions=["NORTH_ATLANTIC"], boxes=[ordinary, wrapped]
    )
    assert scenarios[1].box == queryset_maps.CORE_REGION_BOUNDS["NORTH_ATLANTIC"]
    for scenario, expected_indices in ((ordinary, (1,)), (wrapped, (0, 2))):
        expected = tuple(
            position_index
            for position_index, _ in select_queryset(
                training,
                QueryFilter(date_start=date, date_end=date, box=scenario.box),
            ).iter_pairs()
        )
        assert expected == expected_indices
        actual = queryset_maps.panel_positions(
            training,
            date=date,
            scenario=scenario,
            require_coverage=False,
        )
        assert _position_indices(actual) == expected


def test_covered_panels_require_positive_non_null_swot_and_ssh_evidence():
    training = _queryset("training")
    scenario = queryset_maps.SpatialScenario("Global")

    # On index 1, position 0 is null, position 1 is zero, and only position 2
    # has positive evidence from both grids.
    positions = queryset_maps.panel_positions(
        training,
        date=training.dates[1],
        scenario=scenario,
        require_coverage=True,
    )
    assert _position_indices(positions) == (2,)

    positive = queryset_maps.panel_positions(
        training,
        date=training.dates[0],
        scenario=scenario,
        require_coverage=True,
    )
    assert _position_indices(positive) == (0, 1, 2)


def test_footprints_are_latitude_aware_and_split_at_antimeridian():
    size = PatchSize(100.0, "km")
    equatorial = queryset_maps.footprint_rectangles(
        size, {"centre_lon": 0.0, "centre_lat": 0.0}
    )
    high_latitude = queryset_maps.footprint_rectangles(
        size, {"centre_lon": 0.0, "centre_lat": 60.0}
    )
    assert equatorial[0].height == pytest.approx(100.0 / 111.32)
    assert high_latitude[0].width == pytest.approx(2 * equatorial[0].width)

    wrapped = queryset_maps.footprint_rectangles(
        size, {"centre_lon": 179.8, "centre_lat": 0.0}
    )
    assert len(wrapped) == 2
    assert wrapped[0].lon_min > 179.0
    assert wrapped[0].lon_min + wrapped[0].width == pytest.approx(180.0)
    assert wrapped[1].lon_min == -180.0
    assert sum(rectangle.width for rectangle in wrapped) == pytest.approx(
        equatorial[0].width
    )


def test_cli_parses_repeatable_wrapped_bbox_scenarios_independently():
    arguments = queryset_maps._argument_parser().parse_args(
        [
            "--train",
            "train",
            "--eval",
            "eval",
            "--output",
            "report.pdf",
            "--bbox",
            "ordinary",
            "0",
            "20",
            "30",
            "50",
            "--bbox",
            "wrapped",
            "170",
            "-170",
            "-20",
            "20",
            "--wraps-antimeridian",
        ]
    )
    scenarios = tuple(
        queryset_maps._parse_bbox(values, wraps_antimeridian=wraps)
        for values, wraps in arguments.bbox
    )
    assert not scenarios[0].box.wraps_antimeridian
    assert scenarios[1].box.wraps_antimeridian


@pytest.mark.parametrize("suffix", (".pdf", ".png"))
def test_render_report_writes_static_output_without_cartopy_downloads(
    monkeypatch, tmp_path: Path, suffix: str
):
    pytest.importorskip("cartopy")
    pytest.importorskip("matplotlib")
    training = _queryset("training")
    evaluation = _queryset("eval")
    rows = queryset_maps.build_report_rows(
        training,
        evaluation,
        dates=queryset_maps.select_common_dates(training, evaluation),
    )
    monkeypatch.setattr(queryset_maps, "_add_coastlines", lambda *args: None)

    output = queryset_maps.render_report(rows, tmp_path / f"report{suffix}")
    assert output == tmp_path / f"report{suffix}"
    assert output.is_file()
    assert output.stat().st_size > 0
