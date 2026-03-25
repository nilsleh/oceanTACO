#!/usr/bin/env python3
"""Dataset-specific download functions used by the OceanTACO generator."""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import argopy
import copernicusmarine
import pandas as pd
import xarray as xr
from argopy import DataFetcher as ArgoDataFetcher

from ocean_taco.generate_dataset.download_date_filters import (
    create_copernicus_glorys_date_filter,
    create_l3_sst_date_filter,
    create_l4_sst_date_filter,
    create_ssh_date_filter,
    create_sss_date_filter,
    create_sss_smos_date_filter,
    create_wind_date_filter,
)
from ocean_taco.generate_dataset.download_tracker import DownloadTracker


def download_glorys_data(
    date_min, date_max, root_dir, tracker: DownloadTracker, dry_run=True
):
    """Download GLORYS SSH data with error tracking."""
    dataset_name = "glorys"
    tracker.logger.info(f"{'[DRY RUN] ' if dry_run else ''}Downloading GLORYS data...")

    glorys_dir = os.path.join(root_dir, "glorys")

    try:
        request_data = copernicusmarine.get(
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            output_directory=glorys_dir,
            regex=create_copernicus_glorys_date_filter(date_min, date_max),
            skip_existing=True,
            dry_run=dry_run,
        )

        if dry_run:
            files_count = (
                len(request_data.files) if hasattr(request_data, "files") else 0
            )
            tracker.log_download_attempt(
                dataset_name,
                (date_min, date_max),
                "success",
                {"files_found": files_count},
            )
        else:
            downloaded_files = list(Path(glorys_dir).glob("*.nc"))
            tracker.log_download_attempt(
                dataset_name,
                (date_min, date_max),
                "success",
                {"files_downloaded": len(downloaded_files)},
            )

        return request_data

    except Exception as e:
        tracker.log_error(dataset_name, e, {"date_min": date_min, "date_max": date_max})
        tracker.log_download_attempt(
            dataset_name, (date_min, date_max), "failed", {"error": str(e)}
        )
        raise


def download_l4_ssh_data(
    date_min, date_max, root_dir, tracker: DownloadTracker, dry_run=True
):
    """Download L4 SSH data from the MY all-sat collection."""
    dataset_name = "l4_ssh"
    tracker.logger.info(f"{'[DRY RUN] ' if dry_run else ''}Downloading L4 SSH data...")

    out_dir = os.path.join(root_dir, "l4_ssh")

    try:
        req = copernicusmarine.get(
            dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
            output_directory=out_dir,
            regex=create_ssh_date_filter(date_min, date_max),
            skip_existing=True,
            dry_run=dry_run,
        )

        files_count = len(req.files) if hasattr(req, "files") else 0
        tracker.logger.info(f"  Found {files_count} files")

        tracker.log_download_attempt(
            dataset_name, (date_min, date_max), "success", {"files_found": files_count}
        )
        return {"all_sat": req}

    except Exception as e:
        tracker.log_error(dataset_name, e, {"date_range": (date_min, date_max)})
        tracker.log_download_attempt(
            dataset_name, (date_min, date_max), "failed", {"error": str(e)}
        )
        raise


def download_l3_ssh_data(
    date_min,
    date_max,
    root_dir,
    tracker: DownloadTracker,
    dry_run=True,
    satellites=None,
):
    """Download L3 SSH data with per-satellite error tracking (MY/NRT split)."""
    dataset_name = "l3_ssh"
    tracker.logger.info(
        f"{'[DRY RUN] ' if dry_run else ''}Downloading L3 SSH data (MY and NRT products)..."
    )

    satellites = satellites or ["s3a", "s3b", "c2n", "alg", "j3n", "s6a", "h2b"]

    l3_dir = os.path.join(root_dir, "l3_ssh")
    results = {}
    success = []
    failed = []

    cutoff_date = datetime(2025, 8, 2)

    for sat in satellites:
        tracker.logger.info(f"  Processing satellite: {sat}")

        req_configs = []
        dt_min = datetime.strptime(date_min, "%Y-%m-%d")
        dt_max = datetime.strptime(date_max, "%Y-%m-%d")

        if dt_min <= cutoff_date:
            end_my_dt = min(dt_max, cutoff_date)
            end_my_str = end_my_dt.strftime("%Y-%m-%d")
            did = (
                f"cmems_obs-sl_glo_phy-ssh_my_{sat}-lr-l3-duacs_PT1S"
                if sat == "s6a"
                else f"cmems_obs-sl_glo_phy-ssh_my_{sat}-l3-duacs_PT1S"
            )
            req_configs.append(("MY", date_min, end_my_str, did))

        if dt_max > cutoff_date:
            start_nrt_dt = max(dt_min, cutoff_date + timedelta(days=1))
            if start_nrt_dt <= dt_max:
                start_nrt_str = start_nrt_dt.strftime("%Y-%m-%d")
                nrt_sat = "al" if sat == "alg" else sat
                did = (
                    f"cmems_obs-sl_glo_phy-ssh_nrt_{nrt_sat}-hr-l3-duacs_PT1S"
                    if sat == "s6a"
                    else f"cmems_obs-sl_glo_phy-ssh_nrt_{nrt_sat}-l3-duacs_PT1S"
                )
                req_configs.append(("NRT", start_nrt_str, date_max, did))

        sat_success = False
        for r_type, start, end, dataset_id in req_configs:
            tracker.logger.info(f"    [{r_type}] ID: {dataset_id}")
            try:
                req = copernicusmarine.get(
                    dataset_id=dataset_id,
                    output_directory=l3_dir,
                    regex=create_ssh_date_filter(start, end),
                    dry_run=dry_run,
                    skip_existing=True,
                )

                if dry_run and hasattr(req, "files") and len(req.files) > 0:
                    sat_success = True
                    tracker.logger.info(f"    {len(req.files)} files found ({r_type})")
                    for f in req.files[:3]:
                        tracker.logger.info(f"      {f.file_path}")
                    if len(req.files) > 3:
                        tracker.logger.info(f"      ... and {len(req.files) - 3} more")
                elif not dry_run:
                    sat_success = True

                results[f"{sat}_{r_type}"] = req

            except Exception as e:
                tracker.log_error(
                    f"{dataset_name}_{sat}_{r_type}",
                    e,
                    {"dataset_id": dataset_id, "date_range": (start, end)},
                )
                tracker.logger.warning(f"    Failed ({r_type}): {e}")

        if sat_success:
            success.append(sat)

    if not success and satellites:
        failed = satellites

    if not success:
        status = "failed"
    elif failed:
        status = "partial"
    else:
        status = "success"

    tracker.log_download_attempt(
        dataset_name,
        (date_min, date_max),
        status,
        {"satellites_success": success, "satellites_failed": failed},
    )

    tracker.logger.info(f"  L3 SSH: {len(success)} successful, {len(failed)} failed")

    if success:
        tracker.logger.info(f"  Successfully downloaded: {', '.join(success)}")
    if failed:
        tracker.logger.info(f"  Failed satellites: {', '.join(failed)}")

    if not success:
        raise RuntimeError(f"All L3 SSH satellites failed: {failed}")

    return results


def download_l3_sst_data(
    date_min, date_max, root_dir, tracker: DownloadTracker, dry_run=True
):
    """Download L3 SST data from the single MY ODYSSEA product."""
    dataset_name = "l3_sst"
    tracker.logger.info(
        f"{'[DRY RUN] ' if dry_run else ''}Downloading SST L3 infrared data..."
    )

    out_dir = os.path.join(root_dir, "l3_sst")
    try:
        req = copernicusmarine.get(
            dataset_id="cmems_obs-sst_glo_phy_my_l3s_P1D-m",
            output_directory=out_dir,
            regex=create_l3_sst_date_filter(date_min, date_max),
            dry_run=dry_run,
            skip_existing=True,
        )
        files_count = len(req.files) if hasattr(req, "files") else 0
        tracker.log_download_attempt(
            dataset_name, (date_min, date_max), "success", {"files": files_count}
        )
        if dry_run and hasattr(req, "files"):
            tracker.logger.info(f"    MY: {files_count} files")
            for f in req.files[:5]:
                tracker.logger.info(f"      {f.file_path}")
            if files_count > 5:
                tracker.logger.info(f"      ... and {files_count - 5} more")
        return {"MY": req}
    except Exception as e:
        tracker.log_error(dataset_name, e, {"date_range": (date_min, date_max)})
        tracker.log_download_attempt(
            dataset_name, (date_min, date_max), "failed", {"error": str(e)}
        )
        raise


def download_l3_sss_smos_data(
    date_min, date_max, root_dir, tracker: DownloadTracker, dry_run=True
):
    """Download SMOS SSS L3 data (ascending and descending orbits)."""
    dataset_name = "l3_sss_smos"
    tracker.logger.info(
        f"{'[DRY RUN] ' if dry_run else ''}Downloading SMOS SSS L3 data..."
    )

    out_dir = os.path.join(root_dir, "l3_sss_smos")
    results = {}
    errors = []

    try:
        tracker.logger.info("  Downloading SMOS ascending orbit data...")
        req_asc = copernicusmarine.get(
            dataset_id="cmems_obs-mob_glo_phy-sss_mynrt_smos-asc_P1D",
            output_directory=out_dir,
            regex=create_sss_smos_date_filter(date_min, date_max),
            dry_run=dry_run,
            skip_existing=True,
        )
        results["ascending"] = req_asc

        if dry_run and hasattr(req_asc, "files"):
            tracker.logger.info(f"    Ascending: {len(req_asc.files)} files")
            for f in req_asc.files[:3]:
                tracker.logger.info(f"      {f.file_path}")
            if len(req_asc.files) > 3:
                tracker.logger.info(f"      ... and {len(req_asc.files) - 3} more")

    except Exception as e:
        errors.append(("ascending", e))
        tracker.log_error(
            f"{dataset_name}_ascending", e, {"date_range": (date_min, date_max)}
        )

    try:
        tracker.logger.info("  Downloading SMOS descending orbit data...")
        req_des = copernicusmarine.get(
            dataset_id="cmems_obs-mob_glo_phy-sss_mynrt_smos-des_P1D",
            output_directory=out_dir,
            regex=create_sss_smos_date_filter(date_min, date_max),
            dry_run=dry_run,
            skip_existing=True,
        )
        results["descending"] = req_des

        if dry_run and hasattr(req_des, "files"):
            tracker.logger.info(f"    Descending: {len(req_des.files)} files")
            for f in req_des.files[:3]:
                tracker.logger.info(f"      {f.file_path}")
            if len(req_des.files) > 3:
                tracker.logger.info(f"      ... and {len(req_des.files) - 3} more")

    except Exception as e:
        errors.append(("descending", e))
        tracker.log_error(
            f"{dataset_name}_descending", e, {"date_range": (date_min, date_max)}
        )

    if errors and not results:
        status = "failed"
    elif errors:
        status = "partial"
    else:
        status = "success"

    total_files = sum(
        len(req.files) if hasattr(req, "files") else 0 for req in results.values()
    )

    tracker.log_download_attempt(
        dataset_name,
        (date_min, date_max),
        status,
        {
            "orbits": list(results.keys()),
            "total_files": total_files,
            "errors": len(errors),
        },
    )

    if errors and not results:
        raise RuntimeError(f"All SMOS orbits failed: {errors}")

    return results


def download_l4_sst_data(
    date_min, date_max, root_dir, tracker: DownloadTracker, dry_run=True
):
    """Download L4 SST data with REP/NRT split as needed."""
    dataset_name = "l4_sst"
    tracker.logger.info(f"{'[DRY RUN] ' if dry_run else ''}Downloading SST L4 data...")

    out_dir = os.path.join(root_dir, "l4_sst")

    nrt_start = datetime(2024, 1, 16)
    dt_min = datetime.strptime(date_min, "%Y-%m-%d")
    dt_max = datetime.strptime(date_max, "%Y-%m-%d")

    if dt_max < nrt_start:
        dataset_id = "METOFFICE-GLO-SST-L4-REP-OBS-SST"
    elif dt_min >= nrt_start:
        dataset_id = "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"
    else:
        results = {}
        rep_end = (nrt_start - timedelta(days=1)).strftime("%Y-%m-%d")
        req_rep = copernicusmarine.get(
            dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
            output_directory=out_dir,
            regex=create_l4_sst_date_filter(date_min, rep_end),
            skip_existing=True,
            dry_run=dry_run,
        )
        results["REP"] = req_rep

        req_nrt = copernicusmarine.get(
            dataset_id="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
            output_directory=out_dir,
            regex=create_l4_sst_date_filter(nrt_start.strftime("%Y-%m-%d"), date_max),
            skip_existing=True,
            dry_run=dry_run,
        )
        results["NRT"] = req_nrt

        files_count = sum(
            len(req.files) if hasattr(req, "files") else 0 for req in results.values()
        )
        tracker.log_download_attempt(
            dataset_name, (date_min, date_max), "success", {"files": files_count}
        )
        if dry_run:
            for key, req in results.items():
                if hasattr(req, "files"):
                    tracker.logger.info(f"    {key}: {len(req.files)} files")
        return results

    req = copernicusmarine.get(
        dataset_id=dataset_id,
        output_directory=out_dir,
        regex=create_l4_sst_date_filter(date_min, date_max),
        skip_existing=True,
        dry_run=dry_run,
    )
    files_count = len(req.files) if hasattr(req, "files") else 0
    tracker.log_download_attempt(
        dataset_name, (date_min, date_max), "success", {"files": files_count}
    )
    if dry_run and hasattr(req, "files"):
        for f in req.files[:5]:
            tracker.logger.info(f"    {f.file_path}")
        if len(req.files) > 5:
            tracker.logger.info(f"    ... and {len(req.files) - 5} more")
    return req


def download_l4_sss_data(
    date_min, date_max, root_dir, tracker: DownloadTracker, dry_run=True
):
    """Download L4 SSS data with MY/NRT split by cutoff date."""
    dataset_name = "l4_sss"
    tracker.logger.info(f"{'[DRY RUN] ' if dry_run else ''}Downloading SSS L4 data...")

    out_dir = os.path.join(root_dir, "l4_sss")

    try:
        cutoff_date = datetime(2022, 12, 31)
        dt_min = datetime.strptime(date_min, "%Y-%m-%d")
        dt_max = datetime.strptime(date_max, "%Y-%m-%d")
        results = {}
        files_count = 0

        if dt_max <= cutoff_date:
            req_my = copernicusmarine.get(
                dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1D",
                output_directory=out_dir,
                regex=create_sss_date_filter(date_min, date_max),
                skip_existing=True,
                dry_run=dry_run,
            )
            results["MY"] = req_my
            files_count += len(req_my.files) if hasattr(req_my, "files") else 0
        elif dt_min > cutoff_date:
            req_nrt = copernicusmarine.get(
                dataset_id="cmems_obs-mob_glo_phy-sss_nrt_multi_P1D",
                output_directory=out_dir,
                regex=create_sss_date_filter(date_min, date_max),
                skip_existing=True,
                dry_run=dry_run,
            )
            results["NRT"] = req_nrt
            files_count += len(req_nrt.files) if hasattr(req_nrt, "files") else 0
        else:
            my_end = cutoff_date.strftime("%Y-%m-%d")
            req_my = copernicusmarine.get(
                dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1D",
                output_directory=out_dir,
                regex=create_sss_date_filter(date_min, my_end),
                skip_existing=True,
                dry_run=dry_run,
            )
            results["MY"] = req_my
            files_count += len(req_my.files) if hasattr(req_my, "files") else 0

            nrt_start = (cutoff_date + timedelta(days=1)).strftime("%Y-%m-%d")
            req_nrt = copernicusmarine.get(
                dataset_id="cmems_obs-mob_glo_phy-sss_nrt_multi_P1D",
                output_directory=out_dir,
                regex=create_sss_date_filter(nrt_start, date_max),
                skip_existing=True,
                dry_run=dry_run,
            )
            results["NRT"] = req_nrt
            files_count += len(req_nrt.files) if hasattr(req_nrt, "files") else 0

        tracker.log_download_attempt(
            dataset_name, (date_min, date_max), "success", {"files": files_count}
        )
        return results
    except Exception as e:
        tracker.log_error(dataset_name, e, {"date_range": (date_min, date_max)})
        tracker.log_download_attempt(
            dataset_name, (date_min, date_max), "failed", {"error": str(e)}
        )
        raise


def download_l4_wind_data(
    date_min, date_max, root_dir, tracker: DownloadTracker, dry_run=True
):
    """Download L4 wind data across legacy and current products."""
    dataset_name = "l4_wind"
    tracker.logger.info(f"{'[DRY RUN] ' if dry_run else ''}Downloading Wind L4 data...")

    out_dir = os.path.join(root_dir, "l4_wind")
    start_date = datetime.strptime(date_min, "%Y-%m-%d")
    end_date = datetime.strptime(date_max, "%Y-%m-%d")

    products = {
        "legacy": {
            "dataset_id": "cmems_obs-wind_glo_phy_my_l4_0.25deg_PT1H",
            "start": datetime(1994, 5, 31),
            "end": datetime(2009, 10, 31),
        },
        "current": {
            "dataset_id": "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
            "start": datetime(2007, 10, 10),
            "end": datetime(2025, 12, 31),
        },
    }

    results = {}
    errors = []

    for product_name, product_info in products.items():
        overlap_start = max(start_date, product_info["start"])
        overlap_end = min(end_date, product_info["end"])

        if overlap_start <= overlap_end:
            tracker.logger.info(f"  Processing {product_name} product")

            try:
                req = copernicusmarine.get(
                    dataset_id=product_info["dataset_id"],
                    output_directory=out_dir,
                    regex=create_wind_date_filter(
                        overlap_start.strftime("%Y-%m-%d"),
                        overlap_end.strftime("%Y-%m-%d"),
                    ),
                    skip_existing=True,
                    dry_run=dry_run,
                )
                results[product_name] = req

                if dry_run and hasattr(req, "files"):
                    tracker.logger.info(f"    {len(req.files)} files")

            except Exception as e:
                errors.append((product_name, e))
                tracker.log_error(
                    f"{dataset_name}_{product_name}",
                    e,
                    {
                        "date_range": (
                            overlap_start.strftime("%Y-%m-%d"),
                            overlap_end.strftime("%Y-%m-%d"),
                        )
                    },
                )

    if errors and not results:
        status = "failed"
    elif errors:
        status = "partial"
    else:
        status = "success"

    tracker.log_download_attempt(
        dataset_name,
        (date_min, date_max),
        status,
        {"products": list(results.keys()), "errors": len(errors)},
    )

    if errors and not results:
        raise RuntimeError(f"All wind products failed: {errors}")

    return results


def download_argo_data(
    date_min, date_max, root_dir, tracker: DownloadTracker, dry_run=True, region=None
):
    """Download Argo data with error tracking and regional chunking."""
    dataset_name = "argo"
    tracker.logger.info(f"{'[DRY RUN] ' if dry_run else ''}Downloading Argo data...")

    argo_dir = os.path.join(root_dir, "argo")
    Path(argo_dir).mkdir(parents=True, exist_ok=True)

    argopy.set_options(src="erddap", mode="research")

    try:
        if dry_run:
            from argopy import IndexFetcher

            tracker.logger.info("  Querying Argo index for date range...")
            idx_fetcher = IndexFetcher(src="erddap").region(
                [-180, 180, -70, 70, 0, 100, date_min, date_max]
            )

            index_df = idx_fetcher.to_dataframe()

            tracker.log_download_attempt(
                dataset_name,
                (date_min, date_max),
                "success",
                {
                    "profiles": len(index_df),
                    "unique_floats": index_df["file"].nunique()
                    if len(index_df) > 0
                    else 0,
                },
            )

            tracker.logger.info(f"  Found {len(index_df)} Argo profiles")
            if len(index_df) > 0:
                tracker.logger.info(
                    f"  Date range: {index_df['date'].min()} to {index_df['date'].max()}"
                )
                tracker.logger.info(f"  Unique floats: {index_df['file'].nunique()}")

            return {"profiles_count": len(index_df), "output_dir": argo_dir}

        tracker.logger.info("  Downloading Argo data in regional chunks...")

        if region is None:
            lon_chunks = [
                (-180, -120),
                (-120, -60),
                (-60, 0),
                (0, 60),
                (60, 120),
                (120, 180),
            ]
            lat_chunks = [(-70, -30), (-30, 10), (10, 50), (50, 70)]
            regions = [
                (lon[0], lon[1], lat[0], lat[1]) for lon in lon_chunks for lat in lat_chunks
            ]
        else:
            regions = [region]

        all_data = []
        successful_chunks = 0
        failed_chunks = []

        for i, (lon_min, lon_max, lat_min, lat_max) in enumerate(regions, 1):
            tracker.logger.info(
                f"  [{i}/{len(regions)}] Region: lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}]"
            )
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    fetcher = ArgoDataFetcher(src="erddap").region(
                        [
                            lon_min,
                            lon_max,
                            lat_min,
                            lat_max,
                            0,
                            100,
                            date_min,
                            date_max,
                        ]
                    )
                    ds = fetcher.to_xarray()
                    if len(ds.N_POINTS) > 0:
                        all_data.append(ds)
                        tracker.logger.info(f"    {len(ds.N_POINTS)} profiles")
                    else:
                        tracker.logger.info("    No data in this region")
                    successful_chunks += 1
                    break
                except FileNotFoundError:
                    tracker.logger.info(
                        "    No data found for this region (FileNotFoundError)"
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        tracker.logger.warning(
                            f"    Retry {attempt + 1}/{max_retries} after {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        failed_chunks.append((lon_min, lon_max, lat_min, lat_max))
                        tracker.log_error(
                            f"{dataset_name}_region_{i}",
                            e,
                            {
                                "region": (lon_min, lon_max, lat_min, lat_max),
                                "date_range": (date_min, date_max),
                            },
                        )
                        tracker.logger.warning(
                            f"    Failed after {max_retries} attempts"
                        )

        if not all_data:
            tracker.log_download_attempt(
                dataset_name,
                (date_min, date_max),
                "failed",
                {
                    "failed_chunks": len(failed_chunks),
                    "successful_chunks": successful_chunks,
                },
            )
            tracker.logger.error("  No Argo data downloaded from any region")
            raise RuntimeError("All Argo regions failed")

        tracker.logger.info(f"  Merging {len(all_data)} regional datasets...")
        combined_ds = xr.concat(all_data, dim="N_POINTS")

        time_dt = pd.to_datetime(combined_ds["TIME"].values)
        combined_ds = combined_ds.assign_coords(time=("N_POINTS", time_dt))

        for attr in ["Fetched_by", "Fetched_constraints"]:
            if attr in combined_ds.attrs:
                del combined_ds.attrs[attr]

        tracker.logger.info("  Splitting data by date...")
        dates = time_dt.date
        unique_dates = sorted(set(dates))

        tracker.logger.info(f"  Found {len(unique_dates)} unique dates")

        files_created = []
        total_points = 0

        for date in unique_dates:
            date_str = date.strftime("%Y%m%d")
            date_mask = dates == date
            ds_day = combined_ds.isel(N_POINTS=date_mask)

            if len(ds_day.N_POINTS) == 0:
                continue

            output_file = os.path.join(argo_dir, f"argo_{date_str}.nc")
            ds_day.to_netcdf(output_file)
            files_created.append(output_file)
            total_points += len(ds_day.N_POINTS)

            tracker.logger.info(
                f"    {date}: {len(ds_day.N_POINTS)} points -> {os.path.basename(output_file)}"
            )

        if failed_chunks and successful_chunks:
            status = "partial"
        elif failed_chunks:
            status = "failed"
        else:
            status = "success"

        tracker.log_download_attempt(
            dataset_name,
            (date_min, date_max),
            status,
            {
                "files_created": len(files_created),
                "total_points": total_points,
                "chunks_successful": successful_chunks,
                "chunks_failed": len(failed_chunks),
            },
        )

        tracker.logger.info(
            f"  Downloaded Argo data: {total_points} total profiles across {len(files_created)} files"
        )

        return {
            "output_dir": argo_dir,
            "files_created": len(files_created),
            "total_points": total_points,
            "regions_processed": successful_chunks,
        }

    except Exception as e:
        tracker.log_error(dataset_name, e, {"date_range": (date_min, date_max)})
        tracker.log_download_attempt(
            dataset_name, (date_min, date_max), "failed", {"error": str(e)}
        )
        raise
