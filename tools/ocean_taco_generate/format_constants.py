"""Constants for dataset formatting pipeline."""

SPATIAL_REGIONS = {
    "SOUTH_PACIFIC_WEST": {"lat": (-90, 0), "lon": (-180, -90)},
    "SOUTH_ATLANTIC": {"lat": (-90, 0), "lon": (-90, 0)},
    "SOUTH_INDIAN": {"lat": (-90, 0), "lon": (0, 90)},
    "SOUTH_PACIFIC_EAST": {"lat": (-90, 0), "lon": (90, 180)},
    "NORTH_PACIFIC_WEST": {"lat": (0, 90), "lon": (-180, -90)},
    "NORTH_ATLANTIC": {"lat": (0, 90), "lon": (-90, 0)},
    "NORTH_INDIAN": {"lat": (0, 90), "lon": (0, 90)},
    "NORTH_PACIFIC_EAST": {"lat": (0, 90), "lon": (90, 180)},
}

NETCDF_ENCODINGS = {
    "glorys": {"zlib": True, "complevel": 4},
    "l4_ssh": {"zlib": True, "complevel": 4},
    "l4_sst": {"zlib": True, "complevel": 4},
    "l4_sss": {"zlib": True, "complevel": 4},
    "l4_wind": {"zlib": True, "complevel": 4},
    "l3_ssh": {"zlib": True, "complevel": 4},
    "l3_sst": {"zlib": True, "complevel": 4},
    "l3_swot": {"zlib": True, "complevel": 4},
    "l3_sss_smos_asc": {"zlib": True, "complevel": 4},
    "l3_sss_smos_desc": {"zlib": True, "complevel": 4},
    "argo": {"zlib": True, "complevel": 4},
}

L3_SWOT_VARS = ("ssha_filtered", "ssha_unfiltered", "mdt")
