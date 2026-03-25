"""Compute dataset normalization statistics for variables.."""

import json
import os
from pathlib import Path

import cartopy
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Register custom resolver if not already registered (to avoid errors if run multiple times)
if not OmegaConf.has_resolver("len"):
    OmegaConf.register_new_resolver("len", lambda x: len(x))


class WelfordStats:
    """Compute running statistics using Welford's online algorithm (memory efficient)."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from mean
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def update(self, data: np.ndarray):
        """Update statistics with new data batch."""
        # Flatten and remove invalid values (NaN, inf)
        values = data.ravel()
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return

        # Welford's algorithm for numerical stability
        for x in values:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2
            self.min_val = min(self.min_val, float(x))
            self.max_val = max(self.max_val, float(x))

    def finalize(self) -> dict:
        """Compute final statistics."""
        if self.n < 2:
            return {
                "mean": 0.0,
                "std": 1.0,
                "variance": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }

        variance = self.M2 / self.n
        std = np.sqrt(variance)

        return {
            "mean": float(self.mean),
            "std": float(std),
            "variance": float(variance),
            "min": float(self.min_val),
            "max": float(self.max_val),
            "count": int(self.n),
        }


class ReservoirSampler:
    """Collect samples using reservoir sampling for percentile computation.
    Memory-efficient approach that maintains a fixed-size random sample.
    """

    def __init__(self, reservoir_size: int = 10_000_000):
        """Initialize reservoir sampler.

        Args:
            reservoir_size: Maximum number of samples to keep in memory
        """
        self.reservoir_size = reservoir_size
        self.reservoir = []
        self.n_seen = 0

    def update(self, data: np.ndarray):
        """Update reservoir with new data using reservoir sampling algorithm."""
        values = data.ravel()
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return

        for x in values:
            self.n_seen += 1

            if len(self.reservoir) < self.reservoir_size:
                # Fill reservoir until it reaches capacity
                self.reservoir.append(float(x))
            else:
                # Replace with probability reservoir_size / n_seen
                j = np.random.randint(0, self.n_seen)
                if j < self.reservoir_size:
                    self.reservoir[j] = float(x)

    def compute_percentiles(self, percentiles: list[float] = [2, 98]) -> dict:
        """Compute percentiles from collected samples.

        Args:
            percentiles: List of percentile values to compute (e.g., [2, 98])

        Returns:
            Dictionary with percentile values
        """
        if len(self.reservoir) < 10:
            return None

        reservoir_array = np.array(self.reservoir)
        percentile_values = np.percentile(reservoir_array, percentiles)

        return {
            "percentiles": {
                f"p{int(p)}": float(v) for p, v in zip(percentiles, percentile_values)
            },
            "samples_collected": len(self.reservoir),
            "total_samples_seen": self.n_seen,
        }


def _configure_cartopy_dir(path: str):
    """Configure cartopy data directory."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CARTOPY_USER_DIR"] = str(p)
    cartopy.config["data_dir"] = str(p)


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("COMPUTE STATISTICS (Hydra)")
    print("=" * 60)

    _configure_cartopy_dir("./.cartopy")

    # 1. Disable loading existing statistics
    # This ensures we get raw values (though still nan_to_num=0.0 in dataset,
    # but we will use masks to filter)
    print("Disabling existing stats loading to compute from raw data...")
    if "paths" in cfg:
        # We need to preserve the path to save later, but unset it for instantiation
        save_path = cfg.paths.get("stats_path", "dataset_statistics.json")
        cfg.paths.stats_path = None
    else:
        save_path = "dataset_statistics.json"

    # 2. Instantiate DataModule from config
    # We explicitly remove stats_path from dataset configs if they have it set
    print("Instantiating DataModule...")
    try:
        # Iterate over datasets to clear stats_path
        if "datasets" in cfg:
            for i in range(len(cfg.datasets)):
                if "stats_path" in cfg.datasets[i]:
                    cfg.datasets[i].stats_path = None

        dm = instantiate(cfg.datamodule)

        # Double check: ensure datasets created in setup won't have stats
        # The datamodule uses self.stats_path for creating union dataset
        dm.stats_path = None

    except Exception as e:
        raise RuntimeError("Failed to instantiate datamodule from Hydra config") from e

    # 3. Setup and get DataLoader
    print("Setting up DataModule (this may build indices)...")
    dm.setup("fit")
    train_loader = dm.train_dataloader()

    # 4. Initialize accumulators
    accumulators = {}
    samplers = {}

    print(f"\nProcessing {len(train_loader)} batches...")

    ignored_keys = {
        "target",
        "condition",
        "obs",
        "bounds",
        "batch_idx",
        "coords",
        "query",
        "query_idx",
    }

    # 5. Iterate and Accumulate
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        # Iterate over keys in the batch
        for key, value in batch.items():
            # Skip non-tensor data and metadata
            if not isinstance(value, torch.Tensor):
                continue
            if key in ignored_keys:
                continue
            if key.endswith("_mask") or key in ["lats", "lons", "times", "mask"]:
                continue

            # Identify data variable
            # value shape: [B, C, H, W] or [B, H, W] or [B, C]

            # Initialize if new
            if key not in accumulators:
                accumulators[key] = WelfordStats()
                samplers[key] = ReservoirSampler()

            # Convert to numpy
            data_np = value.detach().cpu().numpy()

            # Handle Masking
            # In XarrayDataset, NaNs are replaced by 0.0, and a mask is provided.
            # We MUST use the mask to filter out invalid 0.0s.

            mask_key = f"{key}_mask"
            valid_data = data_np.ravel()  # Default flat

            if mask_key in batch:
                mask = batch[mask_key].detach().cpu().numpy().astype(bool)

                # Check shapes (mask might be [B, 1, H, W] while data is [B, C, H, W] or mixed)
                if mask.shape == data_np.shape:
                    valid_data = data_np[mask]
                elif mask.ndim == data_np.ndim:
                    # Attempt strict masking if dims match
                    try:
                        valid_data = data_np[mask]
                    except Exception as e:
                        raise ValueError(f"Failed to apply mask for key '{key}'") from e
                else:
                    # If mask is missing channel dim (common if mask is spatial only)
                    # Try broadcasting
                    pass

                # If selection resulted in array, use it. If flat 0.0s exist, filter?
                # Actually, simplify: Flatten both
                flat_data = data_np.ravel()
                flat_mask = mask.ravel()

                if len(flat_data) == len(flat_mask):
                    valid_data = flat_data[flat_mask]
                elif len(flat_data) % len(flat_mask) == 0:
                    # Broadcast mask (e.g. 1 mask for C channels)
                    ratio = len(flat_data) // len(flat_mask)
                    tiled_mask = np.tile(flat_mask, ratio)
                    if len(tiled_mask) == len(flat_data):
                        valid_data = flat_data[tiled_mask]

            # Filter Inf/NaN (just in case)
            valid_data = valid_data[np.isfinite(valid_data)]

            accumulators[key].update(valid_data)
            samplers[key].update(valid_data)

    # 6. Finalize and Save
    print("\nFinalizing statistics...")
    final_stats = {}

    for key in accumulators:
        stats = accumulators[key].finalize()
        perc = samplers[key].compute_percentiles()

        if stats["count"] > 0:
            if perc:
                stats.update(perc)
            final_stats[key] = stats

            print(f"  {key}:")
            print(f"    Mean: {stats['mean']:.4f}")
            print(f"    Std:  {stats['std']:.4f}")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        else:
            print(f"  {key}: No valid data found!")

    # Save
    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(final_stats, f, indent=2)
        print(f"\nSaved statistics to {out_path}")
    else:
        print(json.dumps(final_stats, indent=2))


if __name__ == "__main__":
    main()
