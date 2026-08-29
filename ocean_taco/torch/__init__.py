"""PyTorch map-style surface for rendered OceanTACO samples."""

import random

import numpy as np
from torch.utils.data import get_worker_info

from .dataset import OceanTACODataset, collate_ocean_samples, native_pad_collate
from .loader import CoreSourceLoader
from .sampler import ShapeBucketSampler


def seed_ocean_taco_worker(worker_id: int) -> None:
    """Seed Python/NumPy and reset the shipped loader from PyTorch's worker seed."""
    info = get_worker_info()
    if info is None:
        return
    seed = info.seed
    random.seed(seed)
    np.random.seed(seed % (2**32))
    loader = getattr(info.dataset, "source_loader", None)
    initialise = getattr(loader, "worker_init", None)
    if callable(initialise):
        initialise()


__all__ = [
    "CoreSourceLoader",
    "OceanTACODataset",
    "ShapeBucketSampler",
    "collate_ocean_samples",
    "native_pad_collate",
    "seed_ocean_taco_worker",
]
