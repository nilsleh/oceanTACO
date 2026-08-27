import collections
from ocean_taco.manifest import QuerySet
from ocean_taco.filter import QueryFilter, CoverageRequirement, select_queryset
EXPECT={"512-eval":576,"512-training":1139,"256-eval":2751,"256-training":4766,"128-eval":10206,"128-training":20015}
ids={}; hdr=None
for name,exp in EXPECT.items():
    qs=QuerySet.read(f"release/querysets/v1/{name}")
    D=len(qs.dates)
    assert len(qs.positions)==exp,(name,len(qs.positions))
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
    print(f"{name:14s} pos={exp:6d} dates={D} cov={n:>10,} assets={len(qs.assets):,} "
          f"null%: swot={100*nulls['swot']/n:5.2f} ssh={100*nulls['ssh']/n:5.2f} argo={100*nulls['argo']/n:5.2f} "
          f"violations={sum(bad.values())} {dict(st)} swot>=1:{sel.count:,}", flush=True)
    if bad: print("   !!! VIOLATIONS", dict(bad), flush=True)
    hdr=qs.header
print("\ndistinct queryset ids:", len(set(ids.values()))==6)
for k in ("dataset_revision","catalog_sha256","registry_sha256","source_records_sha256","code_commit","environment_lock_hash"):
    v=str(hdr[k]); assert v and v!="unknown", k
    print(f"   {k:24s} {v[:64]}")
print("tokens:", hdr["tokens"])
