"""The single source of truth for Core modality semantics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Literal

GeometryClass = Literal["dense_grid", "ragged_points"]
RenderClass = Literal["dense_scalar", "vector_pair", "categorical", "ragged_points"]
TimeKind = Literal["daily_label", "instant", "interval", "point_time"]


@dataclass(frozen=True, slots=True)
class ModalitySpec:
    """Validated source record used by retrieval, render, and provenance."""

    token: str
    data_source: str
    filename: str
    primary_variable: str
    geometry: GeometryClass
    render_class: RenderClass
    source_time_kind: TimeKind
    selection: str
    aggregation: str
    canonical_unit: str
    accepted_units: tuple[str, ...]
    nominal_interval: str
    tolerance: str
    coordinate_dtypes: tuple[str, ...]
    regularity_tolerance: float
    qc_provenance: str
    analysis_window_note: str = "unknown"
    product_version: str = "Core pinned revision"
    valid_date_range: tuple[str, str] = ("2023-03-29", "2025-08-02")
    known_discontinuities: tuple[str, ...] = ()
    citation: str = "See dataset documentation."
    vector_pair_tokens: tuple[str, str] | tuple[()] = ()
    available_variables: tuple[str, ...] = ()

    @property
    def is_points(self) -> bool:
        """Whether the source has a ragged point geometry."""
        return self.geometry == "ragged_points"


def _grid(
    token: str,
    data_source: str,
    filename: str,
    variable: str,
    unit: str,
    units: tuple[str, ...],
    *,
    source_time_kind: TimeKind = "daily_label",
    known_discontinuities: tuple[str, ...] = (),
    render_class: RenderClass = "dense_scalar",
    vector_pair_tokens: tuple[str, str] | tuple[()] = (),
) -> ModalitySpec:
    return ModalitySpec(
        token=token,
        data_source=data_source,
        filename=filename,
        primary_variable=variable,
        geometry="dense_grid",
        render_class=render_class,
        source_time_kind=source_time_kind,
        selection="exact_day",
        aggregation="stack",
        canonical_unit=unit,
        accepted_units=units,
        nominal_interval="P1D",
        tolerance="P0D",
        coordinate_dtypes=("float32", "float64"),
        regularity_tolerance=1e-4,
        qc_provenance="Decoded source validity; no loader-side QC coercion.",
        known_discontinuities=known_discontinuities,
        vector_pair_tokens=vector_pair_tokens,
    )


MODALITY_REGISTRY: dict[str, ModalitySpec] = {
    "l4_ssh": _grid("l4_ssh", "l4_ssh", "l4_ssh.nc", "sla", "m", ("m", "meter", "metre")),
    "l4_sst": _grid(
        "l4_sst",
        "l4_sst",
        "l4_sst.nc",
        "analysed_sst",
        "degC",
        ("degC", "degree_Celsius", "degrees_Celsius", "celsius", "C"),
        source_time_kind="instant",
        known_discontinuities=("Raw grid/time layout changes across the Core product transition.",),
    ),
    "l4_sss": _grid("l4_sss", "l4_sss", "l4_sss.nc", "sos", "PSS-78", ("PSS-78", "1", "psu")),
    "l4_wind": _grid("l4_wind", "l4_wind", "l4_wind.nc", "eastward_wind", "m s-1", ("m s-1", "m/s", "m s^-1")),
    "l3_sst": _grid(
        "l3_sst",
        "l3_sst",
        "l3_sst.nc",
        "adjusted_sea_surface_temperature",
        "degC",
        ("degC", "degree_Celsius", "degrees_Celsius", "celsius", "C"),
        source_time_kind="instant",
    ),
    "l3_ssh": _grid("l3_ssh", "l3_ssh", "l3_ssh.nc", "sla_filtered", "m", ("m", "meter", "metre"), source_time_kind="instant"),
    "l3_swot": _grid("l3_swot", "l3_swot", "l3_swot.nc", "ssha_filtered", "m", ("m", "meter", "metre"), source_time_kind="instant"),
    "l3_sss_smos_asc": _grid("l3_sss_smos_asc", "l3_sss_smos_asc", "l3_sss_asc.nc", "Sea_Surface_Salinity", "PSS-78", ("PSS-78", "1", "psu"), source_time_kind="instant"),
    "l3_sss_smos_desc": _grid("l3_sss_smos_desc", "l3_sss_smos_desc", "l3_sss_desc.nc", "Sea_Surface_Salinity", "PSS-78", ("PSS-78", "1", "psu"), source_time_kind="instant"),
    "glorys_ssh": _grid("glorys_ssh", "glorys", "glorys.nc", "zos", "m", ("m", "meter", "metre")),
    "glorys_sst": _grid("glorys_sst", "glorys", "glorys.nc", "thetao", "degC", ("degC", "degree_Celsius", "degrees_Celsius", "celsius", "C")),
    "glorys_sss": _grid("glorys_sss", "glorys", "glorys.nc", "so", "PSS-78", ("PSS-78", "1", "psu")),
    "glorys_uo": _grid(
        "glorys_uo",
        "glorys",
        "glorys.nc",
        "uo",
        "m s-1",
        ("m s-1", "m/s", "m s^-1"),
        render_class="vector_pair",
        vector_pair_tokens=("glorys_uo", "glorys_vo"),
    ),
    "glorys_vo": _grid(
        "glorys_vo",
        "glorys",
        "glorys.nc",
        "vo",
        "m s-1",
        ("m s-1", "m/s", "m s^-1"),
        render_class="vector_pair",
        vector_pair_tokens=("glorys_uo", "glorys_vo"),
    ),
    "argo": ModalitySpec(
        token="argo",
        data_source="argo",
        filename="argo.nc",
        primary_variable="TEMP",
        geometry="ragged_points",
        render_class="ragged_points",
        source_time_kind="point_time",
        selection="all_in_window",
        aggregation="none",
        canonical_unit="degC",
        accepted_units=("degC", "degree_Celsius", "degrees_Celsius", "celsius", "C"),
        nominal_interval="PT0S",
        tolerance="PT0S",
        coordinate_dtypes=("float32", "float64"),
        regularity_tolerance=0.0,
        qc_provenance="argopy research mode: delayed-mode, QC flag 1, adjusted values upstream.",
        available_variables=("TEMP", "PSAL", "TEMP_ERROR", "PSAL_ERROR", "PRES", "PRES_ERROR", "PLATFORM_NUMBER", "CYCLE_NUMBER", "DIRECTION"),
    ),
}


def get_modality(token: str) -> ModalitySpec:
    """Return a source record or fail rather than guessing a filename."""
    try:
        return MODALITY_REGISTRY[token]
    except KeyError as error:
        raise ValueError(f"Unknown OceanTACO source token {token!r}.") from error


def registry_payload() -> dict[str, dict]:
    """Return a stable, JSON-compatible registry representation."""
    return {token: asdict(spec) for token, spec in sorted(MODALITY_REGISTRY.items())}


def registry_sha256() -> str:
    """Content identity for registry provenance."""
    encoded = json.dumps(registry_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()
