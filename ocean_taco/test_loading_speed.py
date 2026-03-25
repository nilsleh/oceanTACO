"""Description of file."""

import time

from torch.utils.data import DataLoader

from ocean_taco.dataset.dataset import OceanTACODataset
from ocean_taco.dataset.queries import PatchSize, QueryGenerator, QuerySampler

# --- CONFIGURATION ---
taco_path = "data/new_ssh_dataset_taco/SeaSurfaceState_Regionalized.tacozip"
input_vars = ["l4_ssh", "glorys_ssh", "l4_sst", "l4_sss", "l3_ssh"]
target_vars = ["glorys_sst"]
num_queries = 20
num_workers = 0
min_date = "2023-03-29"
max_date = "2023-04-03"

# --- GENERATE QUERIES ---
generator = QueryGenerator(land_mask_path=".ocean_mask_cache/land_mask.npy")
queries = generator.generate_training_queries(
    n_queries=num_queries,
    patch_size=PatchSize(16.0, "deg"),
    date_range=(min_date, max_date),
    bbox_constraint=(-75.0, -40.0, 20.0, 50.0),
    max_land_fraction=0.8,
    seed=42,
)


# --- DATASET & LOADER ---
dataset = OceanTACODataset(
    taco_path=taco_path, input_variables=input_vars, target_variables=target_vars
)
sampler = QuerySampler(queries, shuffle=False)


class QueryDatasetAdapter:
    def __init__(self, ds, query_list):
        self.ds = ds
        self.query_list = query_list

    def __len__(self):
        return len(self.query_list)

    def __getitem__(self, idx):
        return self.ds[self.query_list[idx].to_geoslice()]


loader = DataLoader(
    QueryDatasetAdapter(dataset, queries),
    batch_size=10,
    sampler=sampler,
    num_workers=num_workers,
)

# --- MEASURE LOADING SPEED ---
start = time.time()
for i, batch in enumerate(loader):
    print(f"Loaded batch {i + 1} with keys: {list(batch.keys())}")
