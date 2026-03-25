"""Description of file."""

import torch

from ocean_taco.dataset.dataset import SeaSurfaceStateDataset

# Import WelfordStats from compute_statistics
from ocean_taco.generate_dataset.statistics import WelfordStats


def compute_dataset_stats(dataset, max_samples=10):
    """Compute statistics for all variables in dataset."""
    # Initialize stats collectors for each variable
    input_stats = {var: WelfordStats() for var in dataset.input_variables}
    target_stats = {var: WelfordStats() for var in dataset.target_variables}

    print(f"Computing statistics from {min(max_samples, len(dataset))} samples...")

    for idx in range(min(max_samples, len(dataset))):
        print(
            f"  Processing sample {idx + 1}/{min(max_samples, len(dataset))}", end="\r"
        )

        sample = dataset[idx]

        # Update input variable stats
        for var in dataset.input_variables:
            data = sample["inputs"][var]["data"]
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            input_stats[var].update(data)

        # Update target variable stats
        for var in dataset.target_variables:
            data = sample["targets"][var]["data"]
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            target_stats[var].update(data)

    print()  # New line after progress

    # Finalize and print results
    print("\n" + "=" * 60)
    print("INPUT VARIABLES (should be ~N(0,1) if normalized)")
    print("=" * 60)
    for var, stats in input_stats.items():
        result = stats.finalize()
        if result["count"] > 0:
            print(f"\n{var}:")
            print(f"  Mean: {result['mean']:.6f}  (expect ~0)")
            print(f"  Std:  {result['std']:.6f}  (expect ~1)")
            print(f"  Range: [{result['min']:.6f}, {result['max']:.6f}]")
            print(f"  Count: {result['count']:,}")

    print("\n" + "=" * 60)
    print("TARGET VARIABLES (should be ~N(0,1) if normalized)")
    print("=" * 60)
    for var, stats in target_stats.items():
        result = stats.finalize()
        if result["count"] > 0:
            print(f"\n{var}:")
            print(f"  Mean: {result['mean']:.6f}  (expect ~0)")
            print(f"  Std:  {result['std']:.6f}  (expect ~1)")
            print(f"  Range: [{result['min']:.6f}, {result['max']:.6f}]")
            print(f"  Count: {result['count']:,}")


if __name__ == "__main__":
    # Quick test with first 10 samples
    dataset = SeaSurfaceStateDataset(
        taco_zip_path="/p/scratch/hai_uqmethodbox/data/ssh_dataset_taco/SeaSurfaceState.tacozip",
        input_variables=["l2_swot", "l4_ssh"],
        target_variables=["glorys_ssh"],
        min_date="2024-01-01",
        max_date="2024-01-10",
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Statistics loaded: {hasattr(dataset, 'dataset_statistics')}")

    compute_dataset_stats(dataset, max_samples=10)
