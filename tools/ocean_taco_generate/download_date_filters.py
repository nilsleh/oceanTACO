#!/usr/bin/env python3
"""Date range and regex helpers for OceanTACO source downloads."""

from datetime import datetime, timedelta


def _week_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split an inclusive date range into weekly spans."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    spans = []
    cur = start
    while cur <= end:
        w_end = cur + timedelta(days=6)
        if w_end > end:
            w_end = end
        spans.append((cur.isoformat(), w_end.isoformat()))
        cur = w_end + timedelta(days=1)
    return spans


def regex_date_filter(start_date, end_date):
    """Return YYYYMMDD strings for each day in the inclusive date range."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates if dates else []


def create_l3_sst_date_filter(start_date, end_date):
    """Create regex filter for L3 SST ODYSSEA files."""
    dates = regex_date_filter(start_date, end_date)
    if not dates:
        return "^$"
    if len(dates) == 1:
        return rf".*/{dates[0]}\d{{0,6}}-IFR-L3S_GHRSST-SSTfnd-ODYSSEA-GLO.*\.nc$"
    dates_pattern = "|".join(dates)
    return rf".*/({dates_pattern})\d{{0,6}}-IFR-L3S_GHRSST-SSTfnd-ODYSSEA-GLO.*\.nc$"


def create_l4_sst_date_filter(start_date, end_date):
    """Create regex filter for L4 SST METOFFICE files (REP and NRT products)."""
    dates = regex_date_filter(start_date, end_date)
    if not dates:
        return "^$"
    if len(dates) == 1:
        return rf".*/{dates[0]}\d{{0,6}}.*\.nc$"
    dates_pattern = "|".join(dates)
    return rf".*/({dates_pattern})\d{{0,6}}.*\.nc$"


def create_sss_date_filter(start_date, end_date):
    """Create regex filter for L4 SSS daily files."""
    dates = regex_date_filter(start_date, end_date)
    if not dates:
        return "^$"
    if len(dates) == 1:
        return f".*daily_{dates[0]}T.*\\.nc"
    return f".*daily_({'|'.join(dates)})T.*\\.nc"


def create_copernicus_glorys_date_filter(start_date, end_date):
    """Create regex filter for GLORYS daily mean files."""
    dates = regex_date_filter(start_date, end_date)
    if not dates:
        return "^$"
    if len(dates) == 1:
        return f".*_mean_{dates[0]}_R.*\\.nc"
    dates_pattern = "|".join(dates)
    return f".*_mean_({dates_pattern})_R.*\\.nc"


def create_ssh_date_filter(start_date, end_date):
    """Create regex filter for SSH product files."""
    dates = regex_date_filter(start_date, end_date)
    if not dates:
        return "^$"
    if len(dates) == 1:
        return f".*_{dates[0]}_\\d{{8}}\\.nc"
    return f".*_({'|'.join(dates)})_\\d{{8}}\\.nc"


def create_wind_date_filter(start_date, end_date):
    """Create regex filter for hourly wind product files."""
    dates = regex_date_filter(start_date, end_date)
    if not dates:
        return "^$"
    if len(dates) == 1:
        return f".*_{dates[0]}\\d{{2}}_.*\\.nc"
    dates_pattern = "|".join(dates)
    return f".*_({dates_pattern})\\d{{2}}_.*\\.nc"


def create_sss_smos_date_filter(start_date, end_date):
    """Create regex filter for SMOS SSS files across asc/des naming variants."""
    dates = regex_date_filter(start_date, end_date)
    if not dates:
        return "^$"

    if len(dates) == 1:
        return f".*CSF2Q[AD]_{dates[0]}[T_].*\\.nc|.*CSF2Q[AD]_{dates[0]}\\.nc"

    dates_pattern = "|".join(dates)
    return (
        f".*CSF2Q[AD]_({dates_pattern})[T_].*\\.nc|.*CSF2Q[AD]_({dates_pattern})\\.nc"
    )
