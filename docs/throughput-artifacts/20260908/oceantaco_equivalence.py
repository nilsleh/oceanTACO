import sys

sys.path.insert(0, sys.argv[1])
import resource
import time

import torch

from ocean_taco import CatalogConfig, PatchSize, PatchSpec
from ocean_taco.render import Native, Points, Resample, VectorPair
from ocean_taco.torch import OceanTACODataset

port = "/p/project1/hai_uqmethodbox/nils/oceanTACO/results/generation_audit_20260828/port_20230329_verified/taco/OceanTACO"
# Exercise interior, a regional seam, the antimeridian, equator and both hemispheres.
positions = [(-55, 25), (-90, -56), (179.8, -30), (45, 0), (55, 56)]
specs = [
    PatchSpec(lon, lat, PatchSize(128, "km"), "2023-03-29", 0, 0)
    for lon, lat in positions
]
# The local port has one day: this also tests missing-day padding/availability.
specs.append(
    PatchSpec(
        -55,
        25,
        PatchSize(128, "km"),
        "2023-03-29",
        0,
        1,
        target_start_offset_days=1,
        target_end_offset_days=1,
    )
)
sources = (
    "l3_ssh",
    "l3_swot",
    "l4_ssh",
    "l4_sst",
    "l4_sss",
    "l4_wind",
    "glorys_ssh",
    "glorys_sst",
    "glorys_sss",
)
outputs = []
start = time.perf_counter()
for renderer in (Native(), Resample((32, 32), 0.5), Resample((128, 128), 0.5)):
    dataset = OceanTACODataset(
        queries=specs,
        sources={
            **{t: renderer for t in sources},
            "currents": VectorPair(renderer),
            "argo": Points(),
        },
        catalog_config=CatalogConfig(taco_path=port),
    )
    for i in range(len(dataset)):
        outputs.append(dataset[i])
        print(
            type(renderer).__name__,
            i,
            round(time.perf_counter() - start, 2),
            flush=True,
        )
    if hasattr(dataset.source_loader, "close"):
        dataset.source_loader.close()
torch.save(outputs, sys.argv[2])
print(
    "METRICS",
    time.perf_counter() - start,
    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    flush=True,
)
