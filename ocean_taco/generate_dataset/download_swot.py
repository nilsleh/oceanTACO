#!/usr/bin/env python3
"""SWOT FTP cataloging and download helpers."""

import concurrent.futures
import ftplib
import json
import os
import re
import time
from datetime import datetime
from ftplib import FTP

import pandas as pd

from ocean_taco.generate_dataset.download_tracker import DownloadTracker

SWOT_FTP_ROOTS = {
    "l2": "/swot_products/l2_karin/l2_lr_ssh/PGC0/Basic/",
    "l3": "/swot_products/l3_karin_nadir/l3_lr_ssh/v2_0_1/Basic/",
}


def build_swot_file_catalog(
    root_dir, ftp_user, ftp_pass, swot_level="l2", force_rebuild=False
):
    """Build or load a cached SWOT file catalog from AVISO FTP."""
    catalog_file = os.path.join(root_dir, f"swot_{swot_level}_file_catalog.json")
    if (not force_rebuild) and os.path.exists(catalog_file):
        try:
            stat = os.stat(catalog_file)
            age_days = (datetime.now().timestamp() - stat.st_mtime) / 86400
            if age_days < 7:
                print(f"  Using cached catalog (age: {age_days:.1f} days)")
                with open(catalog_file) as f:
                    return json.load(f)
        except Exception as e:
            print(f"  Error reading cached catalog: {e}")

    print(f"  Building SWOT {swot_level.upper()} file catalog...")
    ftp = FTP("ftp-access.aviso.altimetry.fr")
    ftp.login(user=ftp_user, passwd=ftp_pass)
    print("  Connected to FTP server")

    catalog = {}
    ftp.cwd(SWOT_FTP_ROOTS[swot_level])
    cycle_dirs = []
    ftp.retrlines(
        "LIST",
        lambda x: cycle_dirs.append(x.split()[-1])
        if x.startswith("d") and "cycle_" in x
        else None,
    )
    cycle_dirs = sorted(
        [d for d in cycle_dirs if d.startswith("cycle_")],
        key=lambda x: int(x.replace("cycle_", "")),
    )
    print(f"  Found {len(cycle_dirs)} cycle directories")
    total_files = 0
    for i, cycle_dir in enumerate(cycle_dirs, 1):
        print(f"  [{i}/{len(cycle_dirs)}] Cataloging {cycle_dir}...")
        try:
            ftp.cwd(cycle_dir)
            file_entries = []
            ftp.retrlines("LIST", file_entries.append)
            cycle_files = []
            for entry in file_entries:
                if not entry.endswith(".nc"):
                    continue
                parts = entry.split()
                filename = parts[-1]
                size = int(parts[4]) if len(parts) > 4 else 0
                if swot_level == "l2":
                    date_match = re.search(r"_(\d{8})T(\d{6})_", filename)
                    if date_match:
                        f_date = date_match.group(1)
                        f_time = date_match.group(2)
                    else:
                        f_date = None
                        f_time = None
                    ver_match = re.search(r"_PGC0_(\d{2})\.nc$", filename)
                    version = int(ver_match.group(1)) if ver_match else 1
                else:
                    date_match = re.search(r"_(\d{8})T(\d{6})_", filename)
                    if date_match:
                        f_date = date_match.group(1)
                        f_time = date_match.group(2)
                    else:
                        f_date = None
                        f_time = None
                    ver_match = re.search(r"v(\d+\.\d+\.\d+)\.nc$", filename)
                    version = ver_match.group(1) if ver_match else "unknown"
                cycle_files.append(
                    {
                        "filename": filename,
                        "size": size,
                        "date": f_date,
                        "time": f_time,
                        "version": version,
                    }
                )
            catalog[cycle_dir] = cycle_files
            total_files += len(cycle_files)
            ftp.cwd("..")
        except Exception as e:
            print(f"    Error cataloging {cycle_dir}: {e}")
            ftp.cwd(SWOT_FTP_ROOTS[swot_level])
            continue
    ftp.quit()
    meta = {
        "created": datetime.now().isoformat(),
        "total_cycles": len(cycle_dirs),
        "total_files": total_files,
        "cycles": catalog,
    }
    os.makedirs(root_dir, exist_ok=True)
    with open(catalog_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Catalog complete: {total_files} files across {len(cycle_dirs)} cycles")
    return meta


def catalog_to_dataframe(catalog):
    """Convert nested SWOT catalog to a flat DataFrame for date queries."""
    records = []
    for cycle, files in catalog.get("cycles", {}).items():
        for f in files:
            if f.get("date"):
                records.append(
                    {
                        "cycle": cycle,
                        "filename": f.get("filename"),
                        "date": f.get("date"),
                        "time": f.get("time"),
                        "version": f.get("version"),
                        "size": f.get("size"),
                    }
                )
    df = pd.DataFrame(records)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _download_swot_file(task):
    """Download one SWOT file using FTP, with retries and temporary part file."""
    fi, ftp_user, ftp_pass, base_dir, swot_level = task
    fname = fi["filename"]
    cycle = fi["cycle"]
    size = fi.get("size", 0)
    local_path = os.path.join(base_dir, fname)
    ftp_root = SWOT_FTP_ROOTS[swot_level]

    if os.path.exists(local_path) and (
        size == 0 or os.path.getsize(local_path) == size
    ):
        return fname, "skip"
    for attempt in range(1, 5):
        try:
            ftp = ftplib.FTP("ftp-access.aviso.altimetry.fr", timeout=90)
            ftp.login(ftp_user, ftp_pass)
            ftp.cwd(ftp_root)
            ftp.cwd(cycle)
            with open(local_path + ".part", "wb") as out:
                ftp.retrbinary(f"RETR {fname}", out.write)
            ftp.quit()
            os.replace(local_path + ".part", local_path)
            if size and os.path.getsize(local_path) != size:
                raise OSError("size mismatch")
            return fname, "ok"
        except Exception as e:
            try:
                ftp.close()
            except Exception:
                pass
            if attempt == 4:
                return fname, f"fail:{e}"
            time.sleep(2 * attempt)
    return fname, "fail"


def parallel_swot_download(
    files_to_download, ftp_user, ftp_pass, swot_dir, swot_level, max_workers=4
):
    """Download SWOT files in parallel threads."""
    os.makedirs(swot_dir, exist_ok=True)
    tasks = [(fi, ftp_user, ftp_pass, swot_dir, swot_level) for fi in files_to_download]
    results = {"ok": 0, "skip": 0, "fail": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for _, status in ex.map(_download_swot_file, tasks):
            if status in results:
                results[status] += 1
            else:
                results["fail"] += 1
            if (results["ok"] + results["skip"] + results["fail"]) % 10 == 0:
                print(f"Progress: {results}")
    print("Final:", results)
    return results


def download_swot_data(
    date_min,
    date_max,
    root_dir,
    ftp_user,
    ftp_pass,
    tracker: DownloadTracker,
    swot_level="l2",
    dry_run=True,
    force_rebuild_catalog=False,
):
    """Download SWOT data via AVISO FTP with catalog-based date filtering."""
    dataset_name = f"{swot_level}_swot"
    tracker.logger.info(
        f"{'[DRY RUN] ' if dry_run else ''}Downloading SWOT {swot_level.upper()} data via FTP..."
    )
    swot_dir = os.path.join(root_dir, f"{swot_level}_swot")
    os.makedirs(swot_dir, exist_ok=True)
    start_date = datetime.strptime(date_min, "%Y-%m-%d")
    end_date = datetime.strptime(date_max, "%Y-%m-%d")

    def in_range(dstr):
        if not dstr:
            return False
        d = datetime.strptime(dstr, "%Y%m%d")
        return start_date <= d <= end_date

    try:
        tracker.logger.info("  Loading SWOT file catalog...")
        catalog = build_swot_file_catalog(
            root_dir, ftp_user, ftp_pass, swot_level, force_rebuild_catalog
        )

        if not catalog or "cycles" not in catalog:
            tracker.log_download_attempt(
                dataset_name,
                (date_min, date_max),
                "failed",
                {"error": "Catalog unavailable"},
            )
            tracker.logger.error("  Catalog unavailable; aborting SWOT download.")
            return None

        tracker.logger.info(
            f"  Catalog loaded: {catalog['total_files']} files across {catalog['total_cycles']} cycles"
        )

        matching_files = []
        cycles_with_data = []
        for cycle, files in catalog["cycles"].items():
            sel = [dict(f, cycle=cycle) for f in files if in_range(f.get("date"))]
            if sel:
                matching_files.extend(sel)
                cycles_with_data.append(cycle)
                tracker.logger.info(f"    {cycle}: {len(sel)} file(s) in range")

        tracker.logger.info(
            f"  Date-filtered files: {len(matching_files)} across {len(cycles_with_data)} cycles"
        )

        if not matching_files:
            tracker.log_download_attempt(
                dataset_name,
                (date_min, date_max),
                "skipped",
                {"files_found": 0, "cycles_with_data": 0},
            )
            tracker.logger.info("  No matching SWOT files.")
            return {
                "files_found": 0,
                "files_to_download": 0,
                "output_dir": swot_dir,
                "cycles_with_data": 0,
            }

        if swot_level == "l2":

            def get_highest_version(files):
                grouped = {}
                for fi in files:
                    m = re.search(r"(.+_PGC0_)(\d{2})\.nc$", fi["filename"])
                    if not m:
                        continue
                    base = m.group(1)
                    ver = fi.get("version", 0)
                    cur = grouped.get(base)
                    if (cur is None) or (ver > cur["version"]):
                        grouped[base] = {"version": ver, "file": fi}
                return [v["file"] for v in grouped.values()]

            highest_version_files = get_highest_version(matching_files)
            if not highest_version_files:
                highest_version_files = matching_files
            files_to_download = highest_version_files
        else:
            files_to_download = matching_files

        tracker.logger.info(f"  Files to download: {len(files_to_download)}")

        existing = 0
        files_to_download_final = []
        for fi in files_to_download:
            fname = fi["filename"]
            local_path = os.path.join(swot_dir, fname)
            exp_size = fi.get("size", 0)
            if os.path.exists(local_path) and (
                exp_size == 0 or os.path.getsize(local_path) == exp_size
            ):
                existing += 1
                continue
            files_to_download_final.append(fi)

        tracker.logger.info(f"  Existing (skip): {existing}")
        tracker.logger.info(f"  Need download: {len(files_to_download_final)}")

        if dry_run:
            for f in files_to_download_final[:10]:
                tracker.logger.info(
                    f"    {f['filename']} {f['size'] / 1e6:.1f}MB date={f.get('date')}"
                )
            if len(files_to_download_final) > 10:
                tracker.logger.info(f"    ... {len(files_to_download_final) - 10} more")
            total_size_mb = sum(f.get("size", 0) for f in files_to_download_final) / 1e6
            tracker.logger.info(f"  Size estimate: {total_size_mb:.1f} MB")

            tracker.log_download_attempt(
                dataset_name,
                (date_min, date_max),
                "success",
                {
                    "files_found": len(matching_files),
                    "files_to_download": len(files_to_download_final),
                    "total_size_mb": total_size_mb,
                    "cycles_with_data": len(cycles_with_data),
                },
            )

            return {
                "files_found": len(matching_files),
                "files_to_download": len(files_to_download_final),
                "total_size_mb": total_size_mb,
                "output_dir": swot_dir,
                "cycles_with_data": len(cycles_with_data),
                "file_list": [f["filename"] for f in files_to_download_final],
            }

        if not files_to_download_final:
            tracker.log_download_attempt(
                dataset_name,
                (date_min, date_max),
                "success",
                {
                    "files_found": len(matching_files),
                    "files_downloaded": 0,
                    "cycles_with_data": len(cycles_with_data),
                },
            )
            tracker.logger.info("  No new SWOT files to download.")
            return {
                "files_found": len(matching_files),
                "files_downloaded": 0,
                "output_dir": swot_dir,
                "cycles_with_data": len(cycles_with_data),
            }

        results = parallel_swot_download(
            files_to_download_final,
            ftp_user,
            ftp_pass,
            swot_dir,
            swot_level,
            max_workers=4,
        )

        if results["fail"] > 0 and results["ok"] == 0:
            status = "failed"
        elif results["fail"] > 0:
            status = "partial"
        else:
            status = "success"

        tracker.log_download_attempt(
            dataset_name,
            (date_min, date_max),
            status,
            {
                "files_found": len(matching_files),
                "downloaded": results["ok"],
                "skipped": results["skip"],
                "failed": results["fail"],
                "cycles_with_data": len(cycles_with_data),
            },
        )

        return {
            "files_found": len(matching_files),
            "output_dir": swot_dir,
            "cycles_with_data": len(cycles_with_data),
            "download_results": results,
        }

    except Exception as e:
        tracker.log_error(dataset_name, e, {"date_range": (date_min, date_max)})
        tracker.log_download_attempt(
            dataset_name, (date_min, date_max), "failed", {"error": str(e)}
        )
        raise
