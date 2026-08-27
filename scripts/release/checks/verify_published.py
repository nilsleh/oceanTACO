import argparse
import collections
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ocean_taco.filter import CoverageRequirement, QueryFilter, select_queryset
from ocean_taco.manifest import QuerySet
from ocean_taco.sampling.grids import build_position_grid
from ocean_taco.sampling.ocean_mask import load_released_ocean_mask
from ocean_taco.sampling.publish import GRID_SPACING_RATIO
from scripts.release.validate_release_evidence import REFERENCE_SETS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(os.environ.get("ROOT", "release/querysets/v1")))
parser.add_argument("--positions-only", action="store_true")
args = parser.parse_args()
root = args.root
mask = load_released_ocean_mask()
ids = {}
headers = []
for name, (size, kind, reference_count) in REFERENCE_SETS.items():
    directory = root / name
    header = __import__("json").loads((directory / "header.json").read_text(encoding="utf-8"))
    patch_size = __import__("ocean_taco.geobox", fromlist=["PatchSize"]).PatchSize(
        float(header["patch_size"]["value"]), str(header["patch_size"]["unit"])
    )
    expected_positions = build_position_grid(
        mask, patch_size=patch_size, spacing_km=patch_size.value * GRID_SPACING_RATIO[kind]
    )
    actual_identity = [
        (row["position_id"], row["centre_lon"], row["centre_lat"])
        for row in pq.read_table(directory / "positions.parquet", columns=["position_id", "centre_lon", "centre_lat"]).to_pylist()
    ]
    expected_identity = [
        (row["position_id"], row["centre_lon"], row["centre_lat"])
        for row in expected_positions
    ]
    assert abs(len(expected_positions) - reference_count) / reference_count <= 0.01, (
        name, len(expected_positions), reference_count
    )
    assert actual_identity == expected_identity, f"{name}: published positions differ from independently rebuilt grid"
    if args.positions_only:
        continue
    qs = QuerySet.read(directory)
    D=len(qs.dates)
    assert len(qs.positions) == len(expected_positions), (
        name, len(qs.positions), len(expected_positions)
    )
    assert len(qs.coverage)==len(qs.positions)*D, "cartesian product broken"
    assert len(qs.assets)==8*3*D, "asset ledger incomplete"
    ids[name]=qs.queryset_id
    nulls=collections.Counter(); bad=collections.Counter()
    for r in qs.coverage:
        p=qs.positions[r["position_index"]]
        for tok in ("swot","ssh"):
            vc,voc=r[f"{tok}_valid_cells"],r[f"{tok}_valid_ocean_cells"]
            if vc is None:
                nulls[tok]+=1
                if voc is not None: bad["half_null"]+=1
                continue
            if voc>vc: bad["ocean>valid"]+=1
            if vc>p[f"{tok}_footprint_cells"]: bad["valid>footprint"]+=1
        if r["argo_profile_count"] is None: nulls["argo"]+=1
        if r["swot_valid_cells"]==0 and r["swot_n_obs_sum"] not in (0,None): bad["nobs_no_cells"]+=1
    n=len(qs.coverage)
    st=collections.Counter(a["status"] for a in qs.assets)
    sel=select_queryset(qs,QueryFilter(coverage=(CoverageRequirement("swot","valid_ocean_cells",1.0),)))
    print(f"{name:14s} pos={len(expected_positions):6d} dates={D} cov={n:>10,} assets={len(qs.assets):,} "
          f"null%: swot={100*nulls['swot']/n:5.2f} ssh={100*nulls['ssh']/n:5.2f} argo={100*nulls['argo']/n:5.2f} "
          f"violations={sum(bad.values())} {dict(st)} swot>=1:{sel.count:,}", flush=True)
    if bad: print("   !!! VIOLATIONS", dict(bad), flush=True)
    headers.append(qs.header)
if args.positions_only:
    raise SystemExit(0)
print("\ndistinct queryset ids:", len(set(ids.values()))==6)
for index, hdr in enumerate(headers):
    for k in ("dataset_revision", "catalog_sha256", "registry_sha256", "source_records_sha256", "code_commit", "environment_lock_hash"):
        v = str(hdr[k])
        assert v and v != "unknown", (index, k)
    print(f"   provenance[{index}] {str(hdr["code_commit"])[:64]}")
print("tokens:", headers[0]["tokens"])
