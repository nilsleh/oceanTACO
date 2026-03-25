"""Description of file."""

import os
from pathlib import Path

import cartopy
import pandas as pd

from ocean_taco.dataset.dataset import OceanTACODataset


def _configure_cartopy_dir(path: str):
    """Configure cartopy data directory."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CARTOPY_USER_DIR"] = str(p)
    cartopy.config["data_dir"] = str(p)


def test_geo_dataset(
    dataset: OceanTACODataset,
    bbox: tuple[float, float, float, float] = (120, 160, 20, 60),
    time_slice: tuple[pd.Timestamp, pd.Timestamp] = None,
    out_path: str = "test_geo_dataset.png",
):
    """Test OceanTACODataset (geospatial query).

    Args:
        dataset: OceanTACODataset instance
        bbox: Spatial bounds
        time_slice: Temporal bounds (start, end)
        out_path: Output file path
    """
    print(f"\n{'=' * 60}")
    print("Testing OceanTACODataset (Geospatial Query)")
    print(f"{'=' * 60}")
    print(f"BBox: {bbox}")
    print(f"Time slice: {time_slice}")

    # Retrieve data using geoslice query
    time_start = time_slice[0] if time_slice else pd.Timestamp("2024-01-01")
    time_end = time_slice[1] if time_slice else time_start
    sample = dataset[
        (slice(bbox[0], bbox[1]), slice(bbox[2], bbox[3]), slice(time_start, time_end))
    ]
    dataset.visualize_sample(sample, save_path=out_path)


def test_geo_dataset_direct_call(
    dataset: OceanTACODataset,
    bbox: tuple[float, float, float, float] = (120, 160, 20, 60),
    time_slice: tuple[pd.Timestamp, pd.Timestamp] = None,
    out_path: str = "test_geo_dataset_direct.png",
):
    """Test OceanTACODataset using direct get_region() call.

    Args:
        dataset: OceanTACODataset instance
        bbox: Spatial bounds
        time_slice: Temporal bounds (start, end)
        out_path: Output file path
    """
    print(f"\n{'=' * 60}")
    print("Testing OceanTACODataset (Direct get_region())")
    print(f"{'=' * 60}")
    print(f"BBox: {bbox}")
    print(f"Time slice: {time_slice}")

    time_start = time_slice[0] if time_slice else pd.Timestamp("2024-01-01")
    time_end = time_slice[1] if time_slice else time_start
    sample = dataset[
        (slice(bbox[0], bbox[1]), slice(bbox[2], bbox[3]), slice(time_start, time_end))
    ]

    dataset.visualize_sample(sample, save_path=out_path)


def main():
    """Run tests for both dataset types."""
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from ocean_taco.dataset.dataset import (
        Query,
        collate_ocean_samples,
        generate_queries,
    )

    # Configure cartopy
    _configure_cartopy_dir("./.cartopy")

    queries = [
        Query(bbox=(120, 170, 20, 55), time_start="2023-04-01"),
        Query(bbox=(120, 170, 20, 55), time_start="2023-04-02"),
    ]

    # Or generate them
    queries = generate_queries(
        bbox=(120, 170, 20, 55),
        date_start="2024-08-01",
        date_end="2024-08-05",
        temporal_window=1,  # daily
    )

    import h5py

    # Extract the embedded file first, or if you have a standalone copy:
    with h5py.File(
        "data/new_ssh_dataset_formatted_region/l3_swot/l3_swot_NORTH_ATLANTIC_20230329.nc",
        "r",
    ) as f:
        for var in f.keys():
            ds = f[var]
            if hasattr(ds, "chunks"):
                print(f"{var}: shape={ds.shape}, chunks={ds.chunks}, dtype={ds.dtype}")

    taco_paths = [
        "data/new_ssh_dataset_taco/OceanTACO_part0001.tacozip",
        "data/new_ssh_dataset_taco/OceanTACO_part0002.tacozip",
        "data/new_ssh_dataset_taco/OceanTACO_part0003.tacozip",
        "data/new_ssh_dataset_taco/OceanTACO_part0004.tacozip",
        "data/new_ssh_dataset_taco/OceanTACO_part0005.tacozip",
        "data/new_ssh_dataset_taco/OceanTACO_part0006.tacozip",
    ]
    # import tacoreader
    # df = tacoreader.load(taco_paths)

    # 2. Create dataset
    from ocean_taco.dataset.dataset import OceanTACODatasetV2

    dataset = OceanTACODatasetV2(
        # taco_path="data/new_ssh_dataset_taco/OceanTACO.tacozip",
        taco_path=taco_paths,
        queries=queries,
        input_variables=["l3_ssh", "l4_ssh", "glorys_ssh", "glorys_sss", "l4_wind"],
        target_variables=["l3_swot", "argo", "l4_wind", "l3_sss_smos_asc"],
        # lazy_threshold_mb=10.0,  # Use dask for files >10MB
        l3_ssh_satellites=["Sentinel-3A", "Jason-3"],
    )

    out = dataset[0]

    dataset.visualize_sample(out, save_path="test_index_dataset.png")

    # 3. Use with DataLoader
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_ocean_samples,
    )

    # benchmark dataloader speed
    import time

    start_time = time.time()
    for batch in tqdm(loader):
        continue
    end_time = time.time()
    print(f"DataLoader iteration time: {end_time - start_time:.2f} seconds")

    # ocean_mask = "nils/oceanTACO/ocean_mask_cache/ocean_mask_0.100deg.npz"
    # from torch.utils.data import DataLoader
    # from tqdm import tqdm

    # cfg = OmegaConf.load("ocean_taco/torchgeo_config.yaml")
    # inter_ds = instantiate(cfg.dataset)

    # # toi argument to sampler needs to be pd.Interval
    # toi = pd.Interval(pd.Timestamp("2023-04-01"), pd.Timestamp("2023-04-04"))

    # # make polygon roi betwenn -70 and 70 lat and -180 and 180 lon
    # roi = shapely_box(-180, -70, 180, 70)

    # sampler = PrecomputedOceanSampler(
    #     dataset=inter_ds,
    #     size=128,
    #     length=1000,
    #     toi=toi,
    #     roi=roi,
    #     ocean_mask_path=ocean_mask,
    #     ocean_threshold=0.9,
    # )
    # need a triple tuple of x, y , t as a slice slice(i, i), slice(i, i), slice(MINT, MAXT)
    # with the coordinates of lon lat and time over japan
    # query = (
    #     slice(120, 170),
    #     slice(20, 55),
    #     slice(pd.Timestamp("2023-04-02"), pd.Timestamp("2023-04-03")),
    # )

    # query = (slice(-51.25334807653633, -19.253348076536327, None), slice(32.572989623862455, 64.57298962386245, None), slice(pd.Timestamp('2023-04-01 00:00:00'), pd.Timestamp('2023-04-02 00:00:00'), None))

    # sample = inter_ds[query]

    # sample = inter_ds[query]

    # geo_slice = next(iter(sampler))

    # sample = inter_ds[geo_slice]

    # inter_ds.plot_sample(sample, save_path="test_index_dataset.png")

    # for query in tqdm(sampler):
    #     sample = inter_ds[query]

    # collate_fn = instantiate(cfg.collate_fn)

    # dl = DataLoader(
    #     inter_ds, sampler=sampler, batch_size=8, num_workers=4, collate_fn=collate_fn
    # )

    # for batch in tqdm(dl):
    #     continue

    # Common parameters
    taco_path = (
        # "/p/scratch/hai_uqmethodbox/data/ssh_dataset_taco/SeaSurfaceState.tacozip",
        # "/p/scratch/hai_uqmethodbox/data/new_ssh_dataset_taco/SeaSurfaceState_part0001.tacozip",
        "data/new_ssh_dataset_taco/OceanTACO.tacozip"
    )
    # input_vars = ["glorys_ssh", "l3_ssh", "l4_ssh", "l4_wind", "argo"]
    input_vars = ["l3_ssh", "l4_ssh", "glorys_ssh", "glorys_sss", "l4_wind"]
    target_vars = ["l3_swot", "argo", "l4_wind", "l3_sss"]
    bbox = (120, 170, 20, 55)
    time_slice = (pd.Timestamp("2023-04-01"), pd.Timestamp("2023-04-02"))
    geo_query = {"bbox": bbox, "time_slice": time_slice}

    # Test 2: Geospatial dataset with dict query (simulating QuerySampler)
    print("\n" + "=" * 60)
    print("TEST 2: Geospatial Dataset (Dict Query Interface)")
    print("=" * 60)

    geo_dataset = OceanTACODataset(
        taco_path="data/new_ssh_dataset_taco/OceanTACO.tacozip",
        input_variables=input_vars,
        target_variables=target_vars,
    )

    # Query a 2-day window
    test_geo_dataset(
        dataset=geo_dataset,
        bbox=bbox,
        time_slice=time_slice,
        out_path="test_geo_dataset_dict_query.png",
    )

    # Test 3: Geospatial dataset with direct get_region() call
    print("\n" + "=" * 60)
    print("TEST 3: Geospatial Dataset (Direct get_region())")
    print("=" * 60)

    test_geo_dataset_direct_call(
        dataset=geo_dataset,
        bbox=bbox,
        time_slice=time_slice,
        out_path="test_geo_dataset_direct.png",
    )

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - test_index_dataset.png (Index-based access)")
    print("  - test_geo_dataset_dict_query.png (Dict query interface)")
    print("  - test_geo_dataset_direct.png (Direct get_region() call)")


if __name__ == "__main__":
    main()
