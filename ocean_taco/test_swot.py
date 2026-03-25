"""Description of file."""

import json
import pathlib
import time

import fsspec
import fsspec.implementations.http as http_impl
import geopandas as gpd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import xarray as xr
from pyarrow import fs


def convert_swot_to_geoparquet(
    swot_file: str,
    output_file: str = "swot_scaled.parquet",
    compression: str = "zstd",
    compression_level: int = 20,
    row_group_size: int = 10_000,
) -> pathlib.Path:
    swot_path = pathlib.Path(swot_file)
    out_path = pathlib.Path(output_file)

    # Decode CF properly (includes scale/offset, units, time handling)
    ds = xr.open_dataset(swot_path, decode_cf=True)

    arrays = []
    names = []
    var_metadata = {}
    global_1d = {}

    for var_name, var in ds.variables.items():
        raw = var.values
        dims = var.dims
        attrs = var.attrs

        # Only keep 2D data
        if dims == ("num_lines", "num_pixels"):
            arrays.append(pa.array(raw.astype("float32").ravel(), type=pa.float32()))
            names.append(var_name)
            var_metadata[var_name] = {k: str(v) for k, v in attrs.items()}

        elif len(dims) == 1:
            try:
                global_1d[var_name] = {
                    "values": [float(x) for x in raw],
                    "attrs": {k: str(v) for k, v in attrs.items()},
                }
            except Exception:
                continue

    # Write table with metadata
    table = pa.Table.from_arrays(arrays, names=names)
    metadata = {
        b"swot_global_attrs": json.dumps(
            {k: str(v) for k, v in ds.attrs.items()}
        ).encode(),
        b"swot_variable_attrs": json.dumps(var_metadata).encode(),
        b"swot_1d_variables": json.dumps(global_1d).encode(),
    }

    pq.write_table(
        table.replace_schema_metadata(metadata),
        out_path,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        use_dictionary=True,
        use_byte_stream_split=True,
        write_statistics=True,
    )

    return out_path.resolve()


def read_bbox_geoparquet(path, bbox, columns=None):
    """Filter a GeoParquet (local or HTTP) by lat/lon bounding box.

    Args:
        path (str): Local path or HTTP(s) URL to .parquet file or directory
        bbox (tuple): (minx, miny, maxx, maxy)
        columns (list[str], optional): Columns to select

    Returns:
        GeoDataFrame with geometry and requested columns.
    """
    minx, miny, maxx, maxy = bbox

    # Determine filesystem & path
    if path.startswith(("http://", "https://")):
        fs_impl = http_impl.HTTPFileSystem()
        arrow_fs = fs.PyFileSystem(fs.FSSpecHandler(fs_impl))
        dataset_path = path
    else:
        arrow_fs = fs.LocalFileSystem()
        dataset_path = path

    # Always ensure lon/lat for geometry filtering and creation
    cols = set(columns or [])
    cols.update(["longitude", "latitude"])

    dataset = ds.dataset(dataset_path, filesystem=arrow_fs, format="parquet")

    table = dataset.to_table(
        columns=list(cols) if columns else None,
        filter=(
            (ds.field("longitude") >= minx)
            & (ds.field("longitude") <= maxx)
            & (ds.field("latitude") >= miny)
            & (ds.field("latitude") <= maxy)
        ),
    )

    df = table.to_pandas()
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    gdf = gdf.drop(columns=["longitude", "latitude"])
    return gdf


def read_bbox_from_netcdf(path, var_name, bbox, dropna=True):
    minx, miny, maxx, maxy = bbox
    file_obj = fsspec.open(path, mode="rb").open()
    ds = xr.open_dataset(file_obj, engine="h5netcdf", decode_cf=True, chunks="auto")

    lon = ds["longitude"]
    lat = ds["latitude"]

    # Precompute mask into memory
    mask = ((lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)).compute()

    da = ds[var_name]

    # Apply mask using numpy array
    da_bbox = da.where(mask, drop=dropna)

    return da_bbox


# From NETCDF to Parquet
output = convert_swot_to_geoparquet(
    "data/SWOT_L2_LR_SSH_Basic_008_496_20231231T231538_20240101T000707_PGC0_01.nc",
    output_file="swot_clean.parquet",
)

# Read a parquet file with a bounding box
pq_file = (
    "https://huggingface.co/datasets/csaybar/playground/resolve/main/swot_clean.parquet"
)
bbox = (0, -10, 360, 10)  # minlon, minlat, maxlon, maxlat
start_time = time.time()
gdf1 = read_bbox_geoparquet(pq_file, bbox, columns=["ssh_karin"])
end_time = time.time()
print(f"Time taken to read GeoParquet: {end_time - start_time:.2f} seconds")


# Read a NetCDF file with a bounding box
netcdf_url = "https://huggingface.co/datasets/csaybar/playground/resolve/main/SWOT_L2_LR_SSH_Basic_008_496_20231231T231538_20240101T000707_PGC0_01.nc"
start_time = time.time()
gdf2 = read_bbox_from_netcdf(netcdf_url, "ssh_karin", (0, -10, 360, 10))
end_time = time.time()
print(f"Time taken to read NetCDF: {end_time - start_time:.2f} seconds")
