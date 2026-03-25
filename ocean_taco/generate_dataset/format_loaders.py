"""Data loading helpers for formatting pipeline."""

import glob
import os
from datetime import datetime

import xarray as xr


def load_glorys_data(data_dir, date_str):
    """Load GLORYS data."""
    year, month = date_str[:4], date_str[4:6]
    # data/new_ssh_dataset/glorys/GLOBAL_MULTIYEAR_PHY_001_030/
    glorys_dir = os.path.join(
        data_dir,
        "glorys",
        "GLOBAL_MULTIYEAR_PHY_001_030",
        "cmems_mod_glo_phy_my_0.083deg_P1D-m_202311",
        year,
        month,
    )
    pattern = os.path.join(glorys_dir, f"mercatorglorys12v1_gl12_mean_{date_str}_R*.nc")
    files = glob.glob(pattern)

    return xr.open_dataset(files[0]) if files else None


def load_l4_ssh_data(data_dir, date_str):
    """Load L4 SSH data."""
    year, month = date_str[:4], date_str[4:6]
    # if date_obj < datetime(2025, 5, 1):
    #     subdir = "SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057/c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_202411"
    #     pattern = f"dt_global_twosat_phy_l4_{date_str}_vDT*.nc"
    # else:
    #     subdir = "SEALEVEL_GLO_PHY_L4_NRT_008_046/cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D_202506"
    #     pattern = f"nrt_global_allsat_phy_l4_{date_str}_*.nc"
    # data/new_ssh_dataset/l4_ssh/SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057/c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_202411
    # data/new_ssh_dataset/l4_ssh/SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057/c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_202411/2023/06/dt_global_twosat_phy_l4_20230602_vDT2024.nc
    subdir = "SEALEVEL_GLO_PHY_L4_MY_008_047/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_202411"
    pattern = f"dt_global_allsat_phy_l4_{date_str}_*.nc"
    l4_dir = os.path.join(data_dir, "l4_ssh", subdir, year, month)
    files = glob.glob(os.path.join(l4_dir, pattern))
    return xr.open_dataset(files[0]) if files else None


def load_l4_sst_data(data_dir, date_str):
    """Load L4 SST data from REP or NRT product depending on date.

    Download splits at 2024-01-16:
      REP: METOFFICE-GLO-SST-L4-REP-OBS-SST  (product SST_GLO_SST_L4_REP_OBSERVATIONS_010_011)
      NRT: METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2 (product SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001)
    Copernicusmarine creates:
      l4_sst/<PRODUCT>/<DATASET_ID>_<ver>/YYYY/MM/<file>.nc
    """
    year, month = date_str[:4], date_str[4:6]
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    nrt_start = datetime(2024, 1, 16)

    if date_obj < nrt_start:
        # REP product
        product = "SST_GLO_SST_L4_REP_OBSERVATIONS_010_011"
        dataset_glob = "METOFFICE-GLO-SST-L4-REP-OBS-SST_*"
        fname_pattern = f"{date_str}*-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB*REP*.nc"
    else:
        # NRT product
        product = "SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001"
        dataset_glob = "METOFFICE-GLO-SST-L4-NRT-OBS-SST*_*"
        fname_pattern = f"{date_str}*-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB*.nc"

    dataset_dirs = glob.glob(os.path.join(data_dir, "l4_sst", product, dataset_glob))
    if not dataset_dirs:
        return None
    dataset_dir = sorted(dataset_dirs)[-1]
    l4_dir = os.path.join(dataset_dir, year, month)
    files = glob.glob(os.path.join(l4_dir, fname_pattern))
    return xr.open_dataset(files[0]) if files else None


def load_l4_sss_data(data_dir, date_str):
    """Load L4 SSS data from MY/NRT products with robust fallback.

    In practice, MY covers historical years (including 2021-2022 in this
    dataset), while NRT may start later. To avoid silent coverage gaps, this
    loader checks both products and returns the first match for the requested
    date.
    """
    year, month = date_str[:4], date_str[4:6]
    date_obj = datetime.strptime(date_str, "%Y%m%d")

    product = "MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013"
    fname_pattern = f"*daily_{date_str}T*.nc"

    # Prefer MY for historical range, NRT for latest dates, but always fallback.
    if date_obj <= datetime(2022, 12, 31):
        dataset_globs = [
            "cmems_obs-mob_glo_phy-sss_my_multi_P1D_*",
            "cmems_obs-mob_glo_phy-sss_nrt_multi_P1D_*",
        ]
    else:
        dataset_globs = [
            "cmems_obs-mob_glo_phy-sss_nrt_multi_P1D_*",
            "cmems_obs-mob_glo_phy-sss_my_multi_P1D_*",
        ]

    for dataset_glob in dataset_globs:
        dataset_dirs = glob.glob(os.path.join(data_dir, "l4_sss", product, dataset_glob))
        if not dataset_dirs:
            continue
        for dataset_dir in sorted(dataset_dirs, reverse=True):
            l4_dir = os.path.join(dataset_dir, year, month)
            files = sorted(glob.glob(os.path.join(l4_dir, fname_pattern)))
            if files:
                return xr.open_dataset(files[0])

    return None


def load_l4_wind_data(data_dir, date_str):
    """Load L4 wind data."""
    fpath = os.path.join(data_dir, "l4_wind", f"l4_wind_daily_{date_str}.nc")
    return xr.open_dataset(fpath) if os.path.exists(fpath) else None


def load_l3_sst_data(data_dir, date_str):
    """Load L3 SST data from the MY ODYSSEA product.

    Download uses dataset_id='cmems_obs-sst_glo_phy_my_l3s_P1D-m' which
    belongs to product SST_GLO_PHY_L3S_MY_010_039.  Copernicusmarine creates:
      l3_sst/SST_GLO_PHY_L3S_MY_010_039/cmems_obs-sst_glo_phy_my_l3s_P1D-m_<ver>/YYYY/MM/<file>.nc
    The version suffix (e.g. _202311) is resolved via glob so it doesn't
    need to be hardcoded.
    """
    year, month = date_str[:4], date_str[4:6]
    # Glob for the version-suffixed dataset directory
    dataset_dirs = glob.glob(
        os.path.join(
            data_dir,
            "l3_sst",
            "SST_GLO_PHY_L3S_MY_010_039",
            "cmems_obs-sst_glo_phy_my_l3s_P1D-m_*",
        )
    )
    if not dataset_dirs:
        return None
    # Expect exactly one match; take the latest alphabetically if multiple
    dataset_dir = sorted(dataset_dirs)[-1]
    l3_dir = os.path.join(dataset_dir, year, month)
    pattern = os.path.join(l3_dir, f"{date_str}*-IFR-L3S_GHRSST-*.nc")
    files = glob.glob(pattern)
    return xr.open_dataset(files[0]) if files else None


def load_l3_swot_data(data_dir, date_str, return_paths_only=False):
    """Load L3 SWOT data files."""
    pattern = os.path.join(
        data_dir, "l3_swot", f"SWOT_L3_LR_SSH_Basic_*_{date_str}T*_v*.nc"
    )
    files = glob.glob(pattern)
    if return_paths_only:
        return files if files else None
    return [xr.open_dataset(f) for f in files] if files else None


def load_l3_ssh_data(data_dir, date_str, return_paths_only=False):
    """Load L3 SSH along-track data files."""
    pattern = os.path.join(data_dir, "l3_ssh", "**", f"*{date_str}*.nc")
    candidates = glob.glob(pattern, recursive=True)

    files = []
    if candidates:
        import re

        # Strict matching: date_str must be the measurement date
        # (first of the two 8-digit sequences at the end)
        # Filename format: dt_global_{sat}_phy_l3_1hz_{MEASUREMENT_DATE}_{PRODUCTION_DATE}.nc
        # We ensure date_str is followed by _\d{8}.nc
        date_pattern = re.compile(rf".*_{date_str}_\d{{8}}\.nc$")
        files = [f for f in candidates if date_pattern.match(f)]

    if return_paths_only:
        return files if files else None
    return [xr.open_dataset(f) for f in files] if files else None


def load_l3_sss_smos_data(data_dir, date_str):
    """Load L3 SMOS SSS data, separated by ascending and descending passes.

    Returns:
        Dictionary with 'asc' and 'desc' keys, each containing list of file paths
    """
    year = date_str[:4]
    asc_dir = os.path.join(
        data_dir,
        "l3_sss_smos",
        "MULTIOBS_GLO_PHY_SSS_L3_MYNRT_015_014",
        "cmems_obs-mob_glo_phy-sss_mynrt_smos-asc_P1D_202411",
        year,
    )
    des_dir = os.path.join(
        data_dir,
        "l3_sss_smos",
        "MULTIOBS_GLO_PHY_SSS_L3_MYNRT_015_014",
        "cmems_obs-mob_glo_phy-sss_mynrt_smos-des_P1D_202411",
        year,
    )

    asc_files = glob.glob(os.path.join(asc_dir, f"*{date_str}*.nc"))
    desc_files = glob.glob(os.path.join(des_dir, f"*{date_str}*.nc"))

    return {
        "asc": asc_files if asc_files else None,
        "desc": desc_files if desc_files else None,
    }


def load_argo_data(data_dir, date_str):
    """Load Argo data."""
    fpath = os.path.join(data_dir, "argo", f"argo_{date_str}.nc")
    return xr.open_dataset(fpath) if os.path.exists(fpath) else None
