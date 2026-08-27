"""Independent spot-check of the published v1 sets.

Uses only RELEASED measurement functions on directly-opened source files;
no builder code participates in computing the expected values.
"""
import os, random, numpy as np, xarray as xr
from ocean_taco.manifest import QuerySet
from ocean_taco.registry import get_modality
from ocean_taco.sampling.ocean_mask import load_released_ocean_mask
from ocean_taco.sampling.coverage import measure_dense_coverage, measure_argo_profile_count
from ocean_taco.retrieve import _REGION_BOUNDS, _intersects, _merge_points, _clean_swot

ROOT="/p/project1/hai_uqmethodbox/data/new_ssh_dataset_taco_folder/OceanTACO/DATA"
mask=load_released_ocean_mask()
random.seed(20260826)
ok=fail=0; details=[]
for setname in ("512-training","256-eval","128-training"):
    qs=QuerySet.read(f"release/querysets/v1/{setname}")
    ps=qs.patch_size
    for _ in range(10):
        pi=random.randrange(len(qs.positions)); di=random.randrange(len(qs.dates))
        pos=qs.positions[pi]; lon0,lat0=float(pos["centre_lon"]),float(pos["centre_lat"])
        date=qs.dates[di][:10]; dd=date.replace("-","_")
        box=ps.footprint(lon0,lat0)
        regs=[r for r,b in _REGION_BOUNDS.items() if _intersects(box,b)]
        row=qs.coverage_row(pi,di)
        for token,key in (("l3_ssh","ssh"),("l3_swot","swot")):
            spec=get_modality(token)
            paths=[f"{ROOT}/{dd}/{r}/{spec.filename}" for r in regs]
            usable=[]
            for p in paths:
                if not os.path.exists(p): usable=None; break
                d=xr.open_dataset(p,engine="h5netcdf")
                if spec.primary_variable not in d: d.close(); usable=None; break
                usable.append((p,d))
            if usable is None:
                for _,d in (usable or []): d.close()
                if row[f"{key}_valid_cells"] is not None:
                    print(f"EXPECTED NULL {setname} {date} {token} {lon0:.2f},{lat0:.2f} got {row[f'{key}_valid_cells']}"); fail+=1
                else: ok+=1
                continue
            parts=[]
            for p,ds in usable:
                if token=="l3_swot": ds=_clean_swot(ds)
                lon=np.asarray(ds["lon"].values,float)
                ds=ds.assign_coords(lon=np.where(lon==180.0,-180.0,((lon+180.0)%360.0)-180.0)).sortby("lon").sortby("lat")
                keep=[spec.primary_variable]+(["n_obs"] if token=="l3_swot" else [])
                parts.append(ds[keep].load()); ds.close()
            merged=xr.combine_by_coords(parts,combine_attrs="override") if len(parts)>1 else parts[0]
            merged[spec.primary_variable].attrs["units"]=spec.canonical_unit
            ref=measure_dense_coverage(merged, token=token, patch_size=ps, centre_lon=lon0, centre_lat=lat0,
                                       ocean_mask=mask, n_obs_variable="n_obs" if token=="l3_swot" else None)
            got=(row[f"{key}_valid_cells"],row[f"{key}_valid_ocean_cells"])
            exp=(ref.valid_cells,ref.valid_ocean_cells)
            if got!=exp: print(f"MISMATCH {setname} {date} {token} {lon0:.2f},{lat0:.2f} got={got} exp={exp}"); fail+=1
            else: ok+=1
            if token=="l3_swot":
                if row["swot_n_obs_sum"]!=ref.n_obs_sum:
                    print(f"NOBS MISMATCH {setname} {date} {lon0:.2f},{lat0:.2f} got={row['swot_n_obs_sum']} exp={ref.n_obs_sum}"); fail+=1
                else: ok+=1
        ap=[f"{ROOT}/{dd}/{r}/argo.nc" for r in _REGION_BOUNDS]
        if all(os.path.exists(p) for p in ap):
            tiles=[xr.open_dataset(p,engine="h5netcdf") for p in ap]
            ref=measure_argo_profile_count(_merge_points(tiles),patch_size=ps,centre_lon=lon0,centre_lat=lat0,date=date)
            if row["argo_profile_count"]!=ref: print(f"ARGO MISMATCH {setname} {date} {lon0:.2f},{lat0:.2f} got={row['argo_profile_count']} exp={ref}"); fail+=1
            else: ok+=1
            for t in tiles: t.close()
        else:
            if row["argo_profile_count"] is not None: print(f"EXPECTED ARGO NULL {setname} {date}"); fail+=1
            else: ok+=1
    print(f"...{setname} done", flush=True)
print(f"\nSPOT-CHECK: {ok} agreements, {fail} mismatches")
