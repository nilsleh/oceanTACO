"""Description of file."""

import glob
import os
import time

import matplotlib.pyplot as plt
import tacoreader


def measure_load_times(pattern="SeaSurfaceState_part*.tacozip"):
    """Measure how long tacoreader.load() takes as the number of input files increases."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    load_times = []
    num_files_list = list(range(1, len(paths) + 1))

    print(f"Found {len(paths)} files. Measuring load time incrementally...\n")

    df = tacoreader.load(paths)

    for n in num_files_list:
        subset = paths[:n]
        start = time.perf_counter()
        df = tacoreader.load(subset if n > 1 else subset[0])
        end = time.perf_counter()
        elapsed = end - start
        del df
        load_times.append(elapsed)

        print(f"Loaded {n:3d} file(s) in {elapsed:.3f} seconds")

    return num_files_list, load_times


def plot_results(num_files_list, load_times):
    plt.figure(figsize=(8, 5))
    plt.plot(num_files_list, load_times, marker="o")
    plt.xlabel("Number of input files")
    plt.ylabel("Load time (seconds)")
    plt.title("tacoreader.load() Performance vs. Number of Files")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tacoreader_load_performance.png")


if __name__ == "__main__":
    num_files_list, load_times = measure_load_times(
        os.path.join("/mnt/SSD2/nils/datasets/SeaTACO", "SeaSurfaceState_part*.tacozip")
    )
    plot_results(num_files_list, load_times)
