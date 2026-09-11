#!/usr/bin/env python3
"""generate_taco_tables.py
-----------------------
Reads OceanTACO NetCDF files for one date/region and produces two LaTeX tables:
  - table_short.tex  : compact overview table (one row per source, key vars)
  - table_long.tex   : full longtable with every variable + description

Usage:
    python generate_taco_tables.py \
        [--date 2023_03_29] \
        [--region NORTH_ATLANTIC] \
        [--data-root /path/to/OceanTACO/DATA] \
        [--output-dir .]
"""

import argparse
from pathlib import Path
<<<<<<< HEAD
from typing import List, Tuple
=======
>>>>>>> 69f25b242d65fa4373bdc83b62f7564863583780

try:
    import netCDF4 as nc4
    _BACKEND = "netCDF4"
except ImportError:
    _BACKEND = "xarray"


# ---------------------------------------------------------------------------
# Hard-coded display metadata per source
# Keys: file(s) used, display name, level, resolution LaTeX string,
#       availability, citation key
# ---------------------------------------------------------------------------
SOURCE_META = [
    {
        "files": ["glorys.nc"],
        "search_globs": ["glorys/glorys_*.nc"],
        "name": "GLORYS-12",
        "level": "Reanalysis",
        "resolution": r"$1/12^\circ$ ($\approx 9.3$ km)",
        "availability": "Core + Extended",
        "cite": r"\citep{jean2021copernicus}",
    },
    {
        "files": ["l4_ssh.nc"],
        "search_globs": ["l4_ssh/l4_ssh_*.nc"],
        "name": "DUACS SSH Product",
        "level": "L4",
        "resolution": r"$0.125^\circ$ ($\approx 13.9$ km)",
        "availability": "Core + Extended",
        "cite": r"\citep{taburet2019duacs}",
    },
    {
        "files": ["l4_sst.nc"],
        "search_globs": ["l4_sst/l4_sst_*.nc"],
        "name": "OSTIA SST (Met Office)",
        "level": "L4",
        "resolution": r"$0.05^\circ$ ($\approx 5.6$ km)",
        "availability": "Core + Extended",
        "cite": r"\citep{copernicus_sst_l4}",
    },
    {
        "files": ["l4_sss.nc"],
        "search_globs": ["l4_sss/l4_sss_*.nc"],
        "name": "Multi Observation SSS",
        "level": "L4",
        "resolution": r"$0.125^\circ$ ($\approx 13.9$ km)",
        "availability": "Core + Extended",
        "cite": r"\citep{copernicus_sss_l4}",
    },
    {
        "files": ["l4_wind.nc"],
        "search_globs": ["l4_wind/l4_wind_*.nc"],
        "name": "Global Ocean Daily Wind",
        "level": "L4",
        "resolution": r"$0.25^\circ$ ($\approx 27.8$ km)",
        "availability": "Core + Extended",
        "cite": r"\citep{copernicus_wind_l4}",
    },
    {
        "files": ["l3_ssh.nc"],
        "search_globs": ["l3_ssh/l3_ssh_*.nc"],
        "name": "Altimetry Along-Track",
        "level": "L3",
        "resolution": r"$\sim0.06^\circ \times 0.09^\circ$",
        "availability": "Core + Extended",
        "cite": r"\citep{copernicus_sealevel_l3_my}",
    },
    {
        "files": ["l3_swot.nc"],
        "search_globs": ["l3_swot/l3_swot_*.nc"],
        "name": "SWOT",
        "level": "L3",
        "resolution": r"$\sim0.02^\circ \times 0.03^\circ$",
        "availability": "Core only",
        "cite": r"\citep{aviso_swot_l3_lr_ssh}",
    },
    {
        "files": ["l3_sst.nc"],
        "search_globs": ["l3_sst/l3_sst_*.nc"],
        "name": "L3 SST",
        "level": "L3",
        "resolution": r"$0.10^\circ$ ($\approx 11.1$ km)",
        "availability": "Core + Extended",
        "cite": r"\citep{copernicus_sst_l3}",
    },
    {
        "files": ["l3_sss_asc.nc", "l3_sss_desc.nc"],  # merged as one row
        "search_globs": ["l3_sss_smos_asc/l3_sss_smos_asc_*.nc", "l3_sss_smos_desc/l3_sss_smos_desc_*.nc"],
        "name": "L3 Salinity (SMOS)",
        "level": "L3",
        "resolution": r"$\sim0.23^\circ \times 0.26^\circ$",
        "availability": "Core + Extended",
        "cite": r"\citep{copernicus_smos_l3}",
    },
    {
        "files": ["argo.nc"],
        "search_globs": ["argo/argo_*.nc"],
        "name": "Argo",
        "level": "In-situ",
        "resolution": "--",
        "availability": "Core + Extended",
        "cite": r"\citep{wong_argo_2020}",
        # Prioritize ocean-state variables in the compact table.
        "short_key_vars": [
            "TEMP",
            "TEMP_ADJUSTED",
            "PSAL",
            "PSAL_ADJUSTED",
            "PRES",
        ],
    },
]

# Variables that are coordinates/dimensions and should be excluded from output
EXCLUDE_VARS = {
    "lat", "lon", "time", "depth", "nv",
    "N_POINTS", "track", "lat_bnds", "lon_bnds", "crs",
}

# ---------------------------------------------------------------------------
# Fallback metadata for variables whose NetCDF attributes are empty
# (l3_ssh.nc and l3_swot.nc store no long_name/units in their files)
# ---------------------------------------------------------------------------
HARDCODED_VAR_META = {
    # shared by both l3_ssh and l3_swot
    "mdt":                 ("Mean Dynamic Topography",                      "m"),
    "mdt_sem":             ("Standard Error of Mean of MDT",                "m"),
    "obs_mean_lon":        ("Mean longitude of observations per grid cell", "degrees_east"),
    "obs_mean_lat":        ("Mean latitude of observations per grid cell",  "degrees_north"),
    "n_obs":               ("Number of observations per grid cell",         "1"),
    # l3_ssh specific
    "sla_filtered":        ("Sea Level Anomaly (filtered)",                 "m"),
    "sla_filtered_sem":    ("Standard Error of Mean of SLA (filtered)",     "m"),
    "adt":                 ("Absolute Dynamic Topography",                  "m"),
    "adt_sem":             ("Standard Error of Mean of ADT",                "m"),
    # l3_swot specific
    "ssha_filtered":       ("Sea Surface Height Anomaly (filtered)",        "m"),
    "ssha_filtered_sem":   ("Standard Error of Mean of SSHA (filtered)",    "m"),
    "ssha_unfiltered":     ("Sea Surface Height Anomaly (unfiltered)",      "m"),
    "ssha_unfiltered_sem": ("Standard Error of Mean of SSHA (unfiltered)",  "m"),
    "adt_filtered":        ("Absolute Dynamic Topography (filtered)",       "m"),
    "adt_filtered_sem":    ("Standard Error of Mean of ADT (filtered)",     "m"),
    "adt_unfiltered":      ("Absolute Dynamic Topography (unfiltered)",     "m"),
    "adt_unfiltered_sem":  ("Standard Error of Mean of ADT (unfiltered)",   "m"),
    # l4_sst (OSTIA)
    "analysed_sst":        ("Daily analysed sea surface temperature", "degree_Celsius"),
    "analysis_error":      ("Estimated standard deviation of analysed sea surface temperature error", "K"),
    "sea_ice_fraction":    ("Sea-ice area fraction", "%"),
    "mask":                ("Land-sea-ice-lake classification mask", "1"),
    # l3_sst
    "sst_dtime":                    ("Time difference from reference time", "seconds"),
    "sea_surface_temperature":      ("Sea surface temperature", "degree_Celsius"),
    "adjusted_sea_surface_temperature": ("Bias-adjusted sea surface temperature", "degree_Celsius"),
    "sses_bias":                    ("Sensor-specific error statistic bias estimate", "K"),
    "sses_standard_deviation":      ("Sensor-specific error statistic standard deviation", "K"),
    "quality_level":                ("Per-pixel quality level", "1"),
    "sources_of_sst":               ("Source sensor identifier for sea surface temperature", "1"),
    "bias_to_reference_sst":        ("Bias to the reference sea surface temperature used for cross-calibration", "K"),
    # l4_wind daily product
    "eastward_wind":       ("Daily mean of the stress-equivalent eastward wind component at 10 m", "m s-1"),
    "northward_wind":      ("Daily mean of the stress-equivalent northward wind component at 10 m", "m s-1"),
    "eastward_wind_std":   ("Daily standard deviation of the stress-equivalent eastward wind component at 10 m", "m s-1"),
    "northward_wind_std":  ("Daily standard deviation of the stress-equivalent northward wind component at 10 m", "m s-1"),
    "eastward_wind_min":   ("Daily minimum of the stress-equivalent eastward wind component at 10 m", "m s-1"),
    "northward_wind_min":  ("Daily minimum of the stress-equivalent northward wind component at 10 m", "m s-1"),
    "eastward_wind_max":   ("Daily maximum of the stress-equivalent eastward wind component at 10 m", "m s-1"),
    "northward_wind_max":  ("Daily maximum of the stress-equivalent northward wind component at 10 m", "m s-1"),
    # Track and masking metadata
    "primary_track":       ("Index of the first contributing track per grid cell", "1"),
    "is_overlap":          ("Flag indicating that multiple tracks contributed to the grid cell", "1"),
    "track_ids":           ("Per-track source file identifier", "1"),
    "track_times":         ("Per-track representative acquisition timestamp", "1"),
    "track_platforms":     ("Per-track satellite platform name", "1"),
}


# ---------------------------------------------------------------------------
# NetCDF reading
# ---------------------------------------------------------------------------

def _get_var_attrs_netcdf4(path: Path) -> list[tuple[str, str, str]]:
    """Return list of (varname, long_name, units) for all data variables."""
    results = []
    with nc4.Dataset(str(path)) as ds:
        for name, var in ds.variables.items():
            if name in EXCLUDE_VARS:
                continue
            long_name = getattr(var, "long_name", "").strip()
            units = getattr(var, "units", "").strip()
            results.append((name, long_name, units))
    return results


def _get_var_attrs_xarray(path: Path) -> list[tuple[str, str, str]]:
    import xarray as xr
    results = []
    with xr.open_dataset(str(path), mask_and_scale=False) as ds:
        for name in list(ds.coords) + list(ds.data_vars):
            if name in EXCLUDE_VARS:
                continue
            if name in ds.coords and name not in ds.data_vars:
                continue
            attrs = ds[name].attrs
            long_name = attrs.get("long_name", "").strip()
            units = attrs.get("units", "").strip()
            results.append((name, long_name, units))
    return results


def collect_variables(path: Path) -> list[tuple[str, str, str]]:
    """Return (varname, long_name, units) for every data variable in a .nc file."""
    if not path.exists():
        print(f"  WARNING: file not found: {path}")
        return []
    if _BACKEND == "netCDF4":
        raw = _get_var_attrs_netcdf4(path)
    else:
        raw = _get_var_attrs_xarray(path)
    result = []
    for varname, long_name, units in raw:
        if varname in HARDCODED_VAR_META:
            fallback_long, fallback_units = HARDCODED_VAR_META[varname]
            # Prefer curated wording for consistency across products.
            long_name = fallback_long
            units = fallback_units or units
        result.append((varname, long_name, units))
    return result


def resolve_source_files(source: dict, data_root: Path, region_dir: Path) -> List[Path]:
    """Resolve representative files for a source from known dataset layouts."""
    found: List[Path] = []

    # Layout 1: OceanTACO style: DATA/<date>/<region>/<name>.nc
    for fname in source["files"]:
        p = region_dir / fname
        if p.exists():
            found.append(p)

    # Layout 2: formatted region-final style: <data_root>/<modality>/<prefix>_*.nc
    for rel_glob in source.get("search_globs", []):
        matches = sorted((data_root).glob(rel_glob))
        if matches:
            found.append(matches[0])

    # Deduplicate while preserving order
    unique: List[Path] = []
    seen = set()
    for p in found:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def collect_source_variables(
    source: dict, data_root: Path, region_dir: Path
) -> List[Tuple[str, str, str]]:
    """Collect and deduplicate variables across representative files for a source."""
    seen = set()
    all_vars: List[Tuple[str, str, str]] = []
    for path in resolve_source_files(source, data_root, region_dir):
        for entry in collect_variables(path):
            if entry[0] not in seen:
                seen.add(entry[0])
                all_vars.append(entry)
    return all_vars


# ---------------------------------------------------------------------------
# Description builder
# ---------------------------------------------------------------------------

_TRIVIAL_UNITS = {"", "1", "dimensionless"}


def build_description(varname: str, long_name: str, units: str) -> str:
    """Return LaTeX description: \\textbf{var} -- long_name (units)."""
    desc = long_name if long_name else varname
    desc = format_definition_sentence(desc)
    desc = escape_underscore(desc)
    units_latex = latex_units(units) if units else ""
    if units_latex and units_latex not in _TRIVIAL_UNITS:
        desc = f"{desc} ({units_latex})"
    return rf"\textbf{{{varname}}} -- {desc}"


def format_definition_sentence(text: str) -> str:
    """Ensure definitions start with a capital letter and end with punctuation."""
    out = (text or "").strip()
    if not out:
        return out
    out = out[0].upper() + out[1:]
    if out[-1] not in ".!?":
        out += "."
    return out


def escape_underscore(s: str) -> str:
    """Escape underscores for LaTeX (outside math mode)."""
    import re

    # Escape only underscores that are not already escaped.
    return re.sub(r"(?<!\\)_", r"\\_", s)


def latex_units(units: str) -> str:
    """Convert a raw CF units string to a LaTeX-safe representation."""
    import re

    if not units:
        return units

    # Already contains LaTeX — pass through unchanged
    if "$" in units or "\\" in units:
        return units

    _EXACT = {
        "degrees_C":       r"$^\circ$C",
        "degree_C":        r"$^\circ$C",
        "degrees_Celsius": r"$^\circ$C",
        "degree_Celsius":  r"$^\circ$C",
        "degrees_north":   r"$^\circ$N",
        "degrees_south":   r"$^\circ$S",
        "degrees_east":    r"$^\circ$E",
        "degrees_west":    r"$^\circ$W",
        "degree_north":    r"$^\circ$N",
        "degree_east":     r"$^\circ$E",
        "kelvin":          r"K",
        "Kelvin":          r"K",
        "1e-3":            r"$10^{-3}$",
        "1e-6":            r"$10^{-6}$",
        "0.001":           r"$10^{-3}$",
        ".001":            r"$10^{-3}$",
        "psu":             r"psu",
        "PSU":             r"psu",
        "decibar":         r"dbar",
        "percent":         r"\%",
        "%":               r"\%",
        "km":              r"km",
        "m":               r"m",
        "1":               r"1",
    }
    if units in _EXACT:
        return _EXACT[units]

    # "seconds since ..." / "nanoseconds since ..." — drop the epoch
    since_match = re.match(r'^([\w]+)\s+since\s+', units)
    if since_match:
        return since_match.group(1)

    # Slash-fraction: "kg/m3", "m/s", "W/m2" → "kg\,m$^{-3}$"
    slash_match = re.match(r'^([A-Za-z]+)\s*/\s*([A-Za-z]+)(\d*)$', units)
    if slash_match:
        num, den, exp = slash_match.group(1), slash_match.group(2), slash_match.group(3)
        power = int(exp) if exp else 1
        return rf"{num}\,{den}$^{{-{power}}}$"

    # Space-separated compound units with exponents: "m s-1", "m2 s-2"
    token_pattern = re.compile(r'^([A-Za-z]+)([+-]?\d+)?$')
    tokens = units.split()
    if tokens and all(token_pattern.match(t) for t in tokens):
        parts = []
        for t in tokens:
            m = token_pattern.match(t)
            sym, exp = m.group(1), m.group(2)
            if exp is None or exp == "1":
                parts.append(sym)
            else:
                parts.append(rf"{sym}$^{{{exp}}}$")
        return r"\,".join(parts)

    # Fallback: escape underscores
    return escape_underscore(units)


# ---------------------------------------------------------------------------
# Short table
# ---------------------------------------------------------------------------

SHORT_TABLE_HEADER = r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.1}
\resizebox{\textwidth}{!}{
\begin{tabular}{|p{3cm}|c|p{3.3cm}|p{5.5cm}|c|}
\hline
\textbf{Data Source} & \textbf{Level} & \textbf{Resolution (deg/km)$^{\dagger}$} & \textbf{Key Variables} & \textbf{Availability} \\
\hline
"""

SHORT_TABLE_FOOTER = r"""\end{tabular}
}
\vspace{0.5em}
\caption{Overview of the primary data sources used in OceanTACO. The core dataset spans
29 March 2023 to 1 August 2025, while the \textbf{extended dataset} covers
1 January 2015 to 1 April 2023 preceding the SWOT mission.
A full description of all variables can be found in Table~\ref{tab:ocean_taco_variable_dictionary}.
Key variables here are shown with simplified aliases for readability.
\footnotesize $^{\dagger}$Horizontal kilometer resolutions are approximate values estimated at the equator.}
\label{tab:data_sources_core}
\end{table}
"""

MAX_KEY_VARS = 5

# Human-readable aliases used only in the short overview table.
SHORT_VAR_ALIASES = {
    # GLORYS
    "zos": "SSH",
    "thetao": "Temperature",
    "so": "Salinity",
    "uo": "U current",
    "vo": "V current",
    # L4 DUACS
    "sla": "SLA",
    "err_sla": "SLA error",
    "ugosa": "U geostrophic anomaly",
    "err_ugosa": "U geostrophic error",
    "vgosa": "V geostrophic anomaly",
    "err_vgosa": "V geostrophic error",
    # L4 SSS
    "sos": "SSS",
    "dos": "Surface density",
    "sos_error": "SSS error",
    "dos_error": "Surface density error",
    # L3/L4 SSH and SWOT
    "mdt": "MDT",
    "mdt_sem": "MDT uncertainty",
    "n_obs": "Obs. count",
    "obs_mean_lon": "Obs. mean lon",
    "obs_mean_lat": "Obs. mean lat",
    "sla_filtered": "SLA (filtered)",
    "sla_filtered_sem": "SLA uncertainty",
    "ssha_filtered": "SSHA (filtered)",
    "ssha_filtered_sem": "SSHA uncertainty",
    "ssha_unfiltered": "SSHA (unfiltered)",
    "ssha_unfiltered_sem": "SSHA uncertainty (unfiltered)",
    "adt": "ADT",
    "adt_sem": "ADT uncertainty",
    "adt_filtered": "ADT (filtered)",
    "adt_filtered_sem": "ADT uncertainty (filtered)",
    "adt_unfiltered": "ADT (unfiltered)",
    "adt_unfiltered_sem": "ADT uncertainty (unfiltered)",
    "primary_track": "Primary track id",
    "is_overlap": "Track overlap flag",
    "track_ids": "Track ids",
    "track_times": "Track timestamps",
    "track_platforms": "Track platforms",
    # SST
    "analysed_sst": "SST",
    "analysis_error": "SST analysis error",
    "sea_surface_temperature": "SST",
    "adjusted_sea_surface_temperature": "SST (bias corrected)",
    "sses_bias": "SST uncertainty bias",
    "sses_standard_deviation": "SST uncertainty SD",
    "quality_level": "SST quality flag",
    "sources_of_sst": "SST source id",
    "bias_to_reference_sst": "SST vs reference bias",
    "sst_dtime": "SST time offset",
    # SSS
    "sea_surface_salinity": "SSS",
    "sss": "SSS",
    "sss_rain_corrected": "SSS (rain corrected)",
    "Sea_Surface_Salinity": "SSS",
    "Sea_Surface_Salinity_Rain_Corrected": "SSS (rain corrected)",
    "Sea_Surface_Salinity_Error": "SSS uncertainty",
    "Sea_Surface_Salinity_Rain_Corrected_Error": "SSS uncertainty (rain corrected)",
    "X_Swath": "Swath cross-track distance",
    "PSAL": "SSS",
    "PSAL_ADJUSTED": "SSS (adjusted)",
    "PSAL_QC": "SSS QC",
    # Argo hydrography
    "TEMP": "Temperature",
    "TEMP_ADJUSTED": "Temperature (adjusted)",
    "TEMP_QC": "Temperature QC",
    "PRES": "Pressure",
    "PRES_ADJUSTED": "Pressure (adjusted)",
    "PRES_QC": "Pressure QC",
    "CYCLE_NUMBER": "Float cycle number",
    "DIRECTION": "Profile direction",
    # Wind
    "eastward_wind": "U10 wind",
    "northward_wind": "V10 wind",
    "eastward_wind_std": "U10 wind SD",
    "northward_wind_std": "V10 wind SD",
    "eastward_wind_min": "U10 wind min",
    "northward_wind_min": "V10 wind min",
    "eastward_wind_max": "U10 wind max",
    "northward_wind_max": "V10 wind max",
    # Misc
    "sea_ice_fraction": "Sea-ice fraction",
    "mask": "Surface mask",
}


def short_var_alias(varname: str, long_name: str) -> str:
    """Return a compact display alias for the short table."""
    if varname in SHORT_VAR_ALIASES:
        return SHORT_VAR_ALIASES[varname]
    if long_name:
        return format_definition_sentence(long_name).rstrip(".")
    return varname


def generate_short_table(sources_data: list, output_path: Path) -> None:
    lines = [SHORT_TABLE_HEADER]
    for src, variables in sources_data:
        name = escape_underscore(src["name"])
        level = src["level"]
        res = src["resolution"]
        avail = src["availability"]

        var_names = [v[0] for v in variables]
        var_meta = {name: (long_name, units) for name, long_name, units in variables}
        preferred = src.get("short_key_vars", [])
        ordered_names = [v for v in preferred if v in var_names]
        ordered_names += [v for v in var_names if v not in ordered_names]
        key_vars = ", ".join(
            escape_underscore(short_var_alias(v, var_meta.get(v, ("", ""))[0]))
            for v in ordered_names[:MAX_KEY_VARS]
        )

        lines.append(
            f"{name} & {level} & {res} & {key_vars} & {avail} \\\\\n\\hline\n"
        )
    lines.append(SHORT_TABLE_FOOTER)
    output_path.write_text("".join(lines))
    print(f"  Written: {output_path}")


# ---------------------------------------------------------------------------
# Long table
# ---------------------------------------------------------------------------

LONG_TABLE_HEADER = r"""\begin{scriptsize}
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.05}

\begin{longtable}{|p{2.2cm}|c|p{2.2cm}|p{8.3cm}|p{2cm}|}
\caption{OceanTACO full variable dictionary across Core and Extended releases.
Time coverage is 2015-01-01 to 2025-08-01 (Extended: 2015-01-01 to 2023-04-01; Core: 2023-03-29 to 2025-08-01).
All datasets are time-stamped; time is stored as coordinate metadata and is not listed as a data variable.
Units follow NetCDF/CF metadata as stored in the files.
Missing values are represented as NaN for floating-point fields and by integer fill values for index/count fields (typically $-1$).} \\
\label{tab:ocean_taco_variable_dictionary} \\
\hline
\textbf{Data Source} & \textbf{Level} & \textbf{Resolution} & \textbf{Variables (with definition)} & \textbf{Reference} \\
\hline
\endfirsthead

\hline
\textbf{Data Source} & \textbf{Level} & \textbf{Resolution} & \textbf{Variables (with definition)} & \textbf{Reference} \\
\hline
\endhead

"""

LONG_TABLE_FOOTER = r"""\end{longtable}
\end{scriptsize}
"""


def generate_long_table(sources_data: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [LONG_TABLE_HEADER]
    for src, variables in sources_data:
        name = escape_underscore(src["name"])
        level = src["level"]
        res = src["resolution"]
        cite = src["cite"]

        # Build variable descriptions block
        if variables:
            var_lines = []
            for i, (varname, long_name, units) in enumerate(variables):
                desc = build_description(
                    escape_underscore(varname), long_name, units
                )
                if i < len(variables) - 1:
                    var_lines.append(desc + r" \newline")
                else:
                    var_lines.append(desc)
            var_block = "\n".join(var_lines)
        else:
            var_block = r"\textit{(no data variables found)}"

        lines.append(
            f"{name} & {level} & {res} &\n{var_block}\n&\n{cite} \\\\\n\\hline\n\n"
        )
    lines.append(LONG_TABLE_FOOTER)
    output_path.write_text("".join(lines))
    print(f"  Written: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    default_data_root = (
        "/p/project1/hai_uqmethodbox/data/new_ssh_dataset_formatted_region_final"
    )
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for OceanTACO dataset."
    )
    parser.add_argument("--date", default="2023_03_29",
                        help="Date folder name, e.g. 2023_03_29")
    parser.add_argument("--region", default="NORTH_ATLANTIC",
                        help="Region folder name, e.g. NORTH_ATLANTIC")
    parser.add_argument("--data-root", default=default_data_root,
                        help="Path to OceanTACO DATA directory")
    parser.add_argument("--output-dir", default=str(script_dir),
                        help="Directory for output .tex files")
    parser.add_argument("--short-name", default="table_short.tex",
                        help="Output filename for short source overview table")
    parser.add_argument("--long-name", default="tables/full_dataset_variables.tex",
                        help="Output filename for long variable dictionary table")
    args = parser.parse_args()

    region_dir = Path(args.data_root) / args.date / args.region
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {region_dir}")
    print(f"Backend: {_BACKEND}")

    # Collect variables for each source
    sources_data = []
    for src in SOURCE_META:
        print(f"  {src['name']} ({', '.join(src['files'])})")
        variables = collect_source_variables(src, Path(args.data_root), region_dir)
        print(f"    -> {len(variables)} data variables")
        sources_data.append((src, variables))

    print("\nGenerating tables...")
    generate_short_table(sources_data, output_dir / args.short_name)
    primary_long_path = output_dir / args.long_name
    generate_long_table(sources_data, primary_long_path)

    # Keep legacy output name in sync for downstream docs that still include it.
    legacy_long_path = output_dir / "table_long.tex"
    if legacy_long_path != primary_long_path:
        generate_long_table(sources_data, legacy_long_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
