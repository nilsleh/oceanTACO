# Dataset Description

OceanTACO is a multi-source oceanographic dataset covering five sea surface variables, organized as regional NetCDF tiles and hosted on HuggingFace.

**HuggingFace dataset:** [nilsleh/OceanTACO](https://huggingface.co/datasets/nilsleh/OceanTACO)

---

## Dataset Versions and Coverage

OceanTACO is released in two temporal versions to support both SWOT-era and longer pre-SWOT analyses.

- **Core version:** `2023-03-29` to `2025-08-01` (spans both the SWOT calibration and science phases — see [SWOT mission phases](#swot-mission-phases) below, which changes how `l3_swot` behaves across dates).
- **Extended version:** `2015-01-01` to `2023-03-29` (all modalities except SWOT).
- **Shared boundary date:** both versions meet at `2023-03-29`, which allows seamless concatenation.

Temporal indexing is daily. The core period contains 856 daily indices, and the extended period contains 3009 daily indices.

---

(swot-mission-phases)=
## SWOT Mission Phases

The Core period spans two different SWOT orbits. `l3_swot` behaves very
differently in each, and both regimes are routinely mistaken for broken or
duplicated data.

| Phase | Dates | Repeat cycle | What a fixed bounding box looks like |
|---|---|---|---|
| Calibration ("fast-sampling") | `2023-03-29` – `2023-07-10` | 1 day | The swath sits in the **same place every day**; a daily mosaic looks *identical* |
| Orbit change | `2023-07-11` – `2023-07-25` | — | **No `l3_swot.nc` is published** on these dates |
| Science | `2023-07-26` – `2025-08-02` | 21 days | The swath **moves every day**; a small box is **empty (all-NaN) on most days** |

### During calibration, an unchanging picture is not unchanging data

In the 1-day repeat orbit the swath *footprint* repeats (day-to-day overlap of
the observed pixels is ~99%), so a mosaic over a fixed box looks static. The
values inside the swath still evolve:

![SWOT calibration phase mosaic](images/swot_phase_calibration.png)

Differencing consecutive days shows coherent, evolving mesoscale structure, with
mean absolute differences of roughly 0.02–0.05 m. No two days are pixel-identical:

![SWOT calibration phase day-to-day differences](images/swot_phase_differences.png)

### During science, most days over a small box are legitimately empty

In the 21-day repeat orbit the swath moves daily. Over a fixed box, revisits land
at lags of 0, 11 and 21/22 days — the 21-day cycle plus its ascending / descending
sub-cycle — so most days carry no observation at all. All-NaN days are
pixel-identical to one another, which is a second, unrelated way a mosaic can
appear "the same":

![SWOT science phase mosaic](images/swot_phase_science.png)

The two regimes are clearest side by side over a full year of coverage: a flat
plateau while the orbit repeats daily, a hard gap at the orbit change, then a
regular 21-day oscillation.

![SWOT coverage over a fixed box across both mission phases](images/swot_revisit_coverage.png)

### Gaps and no-data semantics

SWOT is gridded at ~2 km with `processing = bin_mean_no_smoothing`. There is **no
gap-filling**, so `NaN` always means "not observed here on this day" and never
zero. Averaging or training over SWOT should mask on finite values rather than
assuming dense coverage.

Beyond the orbit-change window, 33 dates publish no `l3_swot.nc` in any region
(SWOT-wide outages, not per-region formatting misses): `2023-05-20/21`,
`2023-07-11`–`2023-07-25`, `2023-09-22`–`2023-09-26`, `2023-12-23`–`2023-12-27`,
`2024-05-11/12`, `2024-10-28`, `2025-01-11/12`, `2025-04-26/27`. Other modalities
are unaffected on those dates.

The figures above are regenerated with:

```sh
python scripts/dev/swot_phase_figures.py --out docs/images
```

---

## Processing Levels and Sensor Semantics

OceanTACO combines products with different observational and modeling characteristics:

- **L3 observations:** preserve native or near-native sampling geometry and sparse coverage.
- **L4 products:** gap-filled mapped fields optimized for spatial completeness.
- **Reanalysis (GLORYS):** physically consistent model-assimilated fields.
- **In situ (Argo):** independent profile observations for validation and cross-checking.

These levels are complementary but not interchangeable. Differences in sampling, mapping, and assimilation should be considered when comparing products.

---

## Data Sources

OceanTACO aggregates products from five observational categories:

| Category | Sources | Variables |
|---|---|---|
| **L4 gridded** (fused/interpolated) | DUACS, CMEMS | SSH (SLA), SST, SSS, Wind |
| **L3 along-track** (swath) | DUACS, SMOS, SWOT | SSH, SSS ascending/descending |
| **GLORYS reanalysis** | CMEMS GLORYS12 | SSH, SST, SSS, currents (u/v) |
| **Argo floats** | Argo GDAC | Temperature profiles (point source) |

---

## Variables

The following variables are available in OceanTACO. Use the **token** string when constructing `OceanTACODataset`.

| Token | NetCDF variable | Description | Units |
|---|---|---|---|
| `l4_ssh` | `sla` | L4 Sea Level Anomaly | m |
| `l4_sst` | `analysed_sst` | L4 Sea Surface Temperature (auto-converted to °C on load) | °C |
| `l4_sss` | `sos` | L4 Sea Surface Salinity | PSU |
| `l4_wind` | `eastward_wind` | L4 Eastward Wind | m/s |
| `l3_sst` | `adjusted_sea_surface_temperature` | L3 SST | K |
| `l3_sss_smos_asc` | `Sea_Surface_Salinity` | L3 SMOS SSS (ascending pass) | PSU |
| `l3_sss_smos_desc` | `Sea_Surface_Salinity` | L3 SMOS SSS (descending pass) | PSU |
| `l3_ssh` | `sla_filtered` | L3 along-track SSH | m |
| `l3_swot` | `ssha_filtered` | SWOT SSH anomaly | m |
| `argo` | `TEMP` | Argo float temperature profiles (point source) | °C |
| `glorys_ssh` | `zos` | GLORYS reanalysis SSH | m |
| `glorys_sst` | `thetao` | GLORYS reanalysis SST | °C |
| `glorys_sss` | `so` | GLORYS reanalysis Salinity | PSU |
| `glorys_uo` | `uo` | GLORYS reanalysis eastward current | m/s |
| `glorys_vo` | `vo` | GLORYS reanalysis northward current | m/s |

---

## Ocean Regions

The global ocean is divided into 8 equal 90°×90° tiles. Each region corresponds to one directory in the dataset.

| Region | Longitude | Latitude |
|---|---|---|
| `SOUTH_PACIFIC_WEST` | −180° to −90° | −90° to 0° |
| `SOUTH_ATLANTIC` | −90° to 0° | −90° to 0° |
| `SOUTH_INDIAN` | 0° to 90° | −90° to 0° |
| `SOUTH_PACIFIC_EAST` | 90° to 180° | −90° to 0° |
| `NORTH_PACIFIC_WEST` | −180° to −90° | 0° to 90° |
| `NORTH_ATLANTIC` | −90° to 0° | 0° to 90° |
| `NORTH_INDIAN` | 0° to 90° | 0° to 90° |
| `NORTH_PACIFIC_EAST` | 90° to 180° | 0° to 90° |

![Globalregion overview figure](images/fig03.png)

---

## Spatial and Temporal Indexing

OceanTACO uses a fixed global indexing model designed for reproducible cross-source querying:

- The ocean is partitioned into 8 fixed regional tiles.
- Data are indexed daily.
- Each sample is queryable by time window, region, data source, and variable token.

Because the internal sample layout is consistent across products and processing levels, the same data-access workflow can be reused across sensors and studies.

---

## Data Format

### Local directory structure

When downloaded locally, OceanTACO follows this layout:

```
DATA/
└── <YYYY_MM_DD>/
    └── <REGION_NAME>/
        ├── l4_ssh.nc
        ├── l4_sst.nc
        ├── l4_sss.nc
        ├── l3_ssh.nc
        ├── l3_swot.nc
        ├── glorys.nc
        └── ...
```

### NetCDF encoding

Files use HDF5/NetCDF4 with scaled `int16` encoding and lossless `zlib` compression. The tile format is compatible with `xarray` and `h5netcdf`. Spatial coordinates follow a regular lat/lon grid; Argo profiles use an unstructured point dimension.

---

## Processing Workflow and Known Limitations

OceanTACO generation follows three high-level steps:

1. Regional tiling of daily global products.
2. Conservative binning/regridding of sparse L3 observations.
3. Storage encoding with scaled `int16` and lossless compression.

Important interpretation caveats:

- **Projection:** data are stored in WGS84 (`EPSG:4326`), which is not area-preserving and introduces stronger distortion toward high latitudes.
- **L3 gridding behavior:** binning preserves observed sampling patterns; additional gap-filling is not introduced at this stage.
- **Uncertainty interpretation:** aggregated per-cell uncertainty primarily reflects within-track variability and may not fully capture between-track sampling differences.
- **Cross-level comparison:** L4 and reanalysis fields include mapping/assimilation effects and should be interpreted accordingly when compared to L3 or in situ observations.

---

## HuggingFace Access

OceanTACO is hosted on HuggingFace and can be accessed without downloading the full dataset.

### Catalog retrieval

```python
from ocean_taco import CatalogConfig, GeoBox
from ocean_taco.retrieve import load_bbox_nc, load_hf_dataset, load_tile_nc

config = CatalogConfig()
catalog = load_hf_dataset(config)

# Load one named-region tile.
tile = load_tile_nc(catalog, "2024-06-01", "NORTH_ATLANTIC", "l4_sst", config=config)

# Or merge every tile intersecting a named geographic box.
ds = load_bbox_nc(
    catalog,
    "2024-06-01",
    GeoBox(-80.0, -30.0, 25.0, 50.0),
    "l4_sst",
    config=config,
)
```

### Download full snapshot (huggingface_hub)

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(repo_id="nilsleh/OceanTACO", repo_type="dataset")
```
---

## Licenses

- **Code**: Apache 2.0
- **Dataset**: Creative Commons Attribution 4.0 International (CC BY 4.0)

See the [OceanTACO Dataset Card](https://huggingface.co/datasets/nilsleh/OceanTACO) for full license information, required attribution, acknowledgements, and citations.
