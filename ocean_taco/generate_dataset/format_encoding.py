"""Encoding helpers for dataset formatting pipeline."""

import logging

import numpy as np


def get_variable_encoding(var_name):
    """Get NetCDF compression and packing encoding for a variable."""
    # Base encoding with high compression
    base = {"zlib": True, "complevel": 5, "_FillValue": -32767}

    # Packing: int16 = 2 bytes vs float32 = 4 bytes (50% size reduction)
    v = var_name.lower()

    # EXCLUSIONS: Do NOT pack these variables (keep as float32 or int per netcdf defaults)
    # Time deltas, angles, masks, counts that don't fit
    if any(
        x in v
        for x in [
            "time",
            "date",
            "mask",
            "quality",
            "flag",
            "status",
            "source",
            "angle",
        ]
    ):
        # If it looks like a flag/status, maybe int16 is fine, but safe default is float32 for safety
        if any(x in v for x in ["flag", "status", "source", "count", "number"]):
            return {**base, "dtype": "int16"}
        return {**base, "dtype": "float32"}

    # SSH / SLA / MDT / ZOS / ADT (Meters)
    # Range typically -2m to 2m, but MDT can be -4m in some places (geoid-related).
    # Previous scale 0.0001 -> range +/- 3.2m (Too small for MDT=-3.99)
    # New scale 0.0005 -> range +/- 16.38m. Precision 0.5 mm (needed for SSHA ~10m).
    if any(x in v for x in ["ssh", "sla", "mdt", "zos", "adt"]):
        return {**base, "dtype": "int16", "scale_factor": 0.0005, "add_offset": 0.0}

    # SST / Thetao (Degrees Celsius)
    # Range typically -2 to 35 C.
    # Precision: 0.001 C
    if any(x in v for x in ["sst", "thetao", "temperature"]):
        return {**base, "dtype": "int16", "scale_factor": 0.001, "add_offset": 20.0}

    # SSS / Salinity (PSU)
    # Some products can exceed 45 PSU (observed up to ~62.7 in SMOS fields).
    # Precision: 0.001 PSU
    # Use exact match or start/end to avoid catching 'so' within 'source' or 'solar'.
    # 'sos' = L4 SSS Copernicus variable; 'salinity' in v catches 'sea_surface_salinity' (L3 SMOS).
    if v in ["sss", "so", "sos", "salinity"] or "salinity" in v or v.startswith("sss_") or v.endswith("_sss"):
        return {**base, "dtype": "int16", "scale_factor": 0.002, "add_offset": 30.0}

    # Ocean Velocities (m/s)
    # Range -5 to 5 m/s is sufficient for ocean currents (uo, vo)
    # Precision: 0.001 m/s
    if any(x in v for x in ["uo", "vo", "current"]):
        return {**base, "dtype": "int16", "scale_factor": 0.001, "add_offset": 0.0}

    # Wind Speed (m/s)
    # Hurricanes can exceed 32 m/s (Cat 1) up to ~80 m/s.
    # Previous scale 0.001 clipped at 32.7 m/s.
    # New scale 0.002 = 65 m/s (risky).
    # New scale 0.005 = 163 m/s (safe). Precision 0.5 cm/s.
    if any(x in v for x in ["wind", "speed"]):
        return {**base, "dtype": "int16", "scale_factor": 0.01, "add_offset": 0.0}

    # Metadata / counts (keep as is or specific integer types)
    if "n_obs" in v or "count" in v:
        # NO scale_factor/add_offset for actual integers to prevents float promotion/casting errors
        return {**base, "dtype": "int16", "_FillValue": -1}
    if "track" in v or "overlap" in v or "mask" in v or "num" in v:
        # NO scale_factor/add_offset for actual integers
        return {**base, "dtype": "int8", "_FillValue": -1}

    # Default for unknown floats
    return {**base, "dtype": "float32"}  # Fallback to float32 with compression


def check_encoding_safety(ds, encoding_dict):
    """Check if data fits within the encoding range to avoid silent clipping."""
    for var_name, enc in encoding_dict.items():
        if var_name not in ds:
            continue

        # Only check integer packed variables
        if enc.get("dtype") != "int16":
            continue

        data = ds[var_name].values
        if np.isnan(data).all():
            continue

        scale = enc.get("scale_factor", 1.0)
        offset = enc.get("add_offset", 0.0)

        # Calculate representable range
        # int16 range is -32768 to 32767.
        # _FillValue is usually -32767 or similar, so usable range is approx +/- 32000 steps

        # Max pos val: 32767 * scale + offset
        # Min neg val: -32767 * scale + offset (-32768 often reserved)

        max_rep = 32760 * scale + offset
        min_rep = -32760 * scale + offset

        act_max = np.nanmax(data)
        act_min = np.nanmin(data)

        if act_max > max_rep or act_min < min_rep:
            logging.warning(
                f"⚠️  Variable '{var_name}' range [{act_min:.2f}, {act_max:.2f}] exceeds encoding limits [{min_rep:.2f}, {max_rep:.2f}]. Data will be clipped!"
            )


def clear_encoding(ds):
    """Clear all encoding from dataset."""
    ds.encoding.clear()
    for var in list(ds.data_vars) + list(ds.coords):
        if hasattr(ds[var], "encoding"):
            ds[var].encoding.clear()
    return ds
