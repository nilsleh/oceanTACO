#!/usr/bin/env python3
"""Benchmark OceanTACO dataset performance."""

import time
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from torch.utils.data import DataLoader

from ocean_taco.dataset.dataset import (
    OceanTACODatasetV2,
    Query,
    collate_ocean_samples,
    generate_queries,
)

app = typer.Typer(help="OceanTACO Dataset Benchmark CLI")
console = Console()

DEFAULT_PATH = "OceanTACO.tacozip"


def parse_bbox(bbox_str: str) -> tuple[float, float, float, float]:
    """Parse 'lon_min,lon_max,lat_min,lat_max' string."""
    try:
        parts = tuple(float(x) for x in bbox_str.split(","))
        if len(parts) != 4:
            raise ValueError
        return parts
    except ValueError:
        console.print(
            "[red]Invalid bbox. Expected: lon_min,lon_max,lat_min,lat_max[/red]"
        )
        raise typer.Exit(1)


@app.command()
def benchmark(
    taco_path: str = typer.Option(
        DEFAULT_PATH, "--path", "-p", help="Path or URL to TACO dataset"
    ),
    start_date: datetime = typer.Option(
        datetime(2024, 1, 1), "--start", "-s", formats=["%Y-%m-%d"], help="Start date"
    ),
    end_date: datetime = typer.Option(
        datetime(2024, 2, 1), "--end", "-e", formats=["%Y-%m-%d"], help="End date"
    ),
    batch_size: int = typer.Option(8, "--batch-size", "-b"),
    num_workers: int = typer.Option(2, "--workers", "-w"),
    bbox: str = typer.Option(
        "120,150,20,55", "--bbox", help="lon_min,lon_max,lat_min,lat_max"
    ),
) -> None:
    """Run benchmark on OceanTACO dataset."""
    bbox_tuple = parse_bbox(bbox)

    config_table = Table(show_header=False, box=None)
    config_table.add_column("Key", style="cyan")
    config_table.add_column("Value")
    config_table.add_row("Path", taco_path)
    config_table.add_row("Date Range", f"{start_date.date()} → {end_date.date()}")
    config_table.add_row("BBox", bbox)
    config_table.add_row("Batch Size", str(batch_size))
    config_table.add_row("Workers", str(num_workers))
    console.print(Panel(config_table, title="[bold]Config[/bold]"))

    with console.status("[green]Generating queries..."):
        queries = generate_queries(
            bbox=bbox_tuple, date_start=start_date.date(), date_end=end_date.date()
        )
    console.print(f"Queries: [cyan]{len(queries)}[/cyan]")

    with console.status("[green]Indexing dataset (SQL)..."):
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

        dataset = OceanTACODatasetV2(
            taco_path=taco_paths,
            queries=queries,
            input_variables=["l3_ssh", "l4_ssh", "l4_sst"],
            target_variables=["l3_swot", "l3_sss_smos_asc"],
        )
    console.print(f"Samples: [cyan]{len(dataset)}[/cyan]")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_ocean_samples,
    )

    total_samples = 0
    total_batches = 0
    start_time = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing", total=len(loader))

        for batch in loader:
            bs = batch_size
            for var_data in batch["inputs"].values():
                if var_data is not None:
                    bs = var_data.shape[0]
                    break

            total_samples += bs
            total_batches += 1
            progress.advance(task)

    duration = time.perf_counter() - start_time

    results = Table(title="[bold]Results[/bold]", show_header=False)
    results.add_column("Metric", style="cyan")
    results.add_column("Value", style="green", justify="right")
    results.add_row("Time", f"{duration:.2f}s")
    results.add_row("Batches", str(total_batches))
    results.add_row("Throughput", f"{total_batches / duration:.2f} batch/s")
    results.add_row("Throughput", f"{total_samples / duration:.2f} sample/s")
    if total_batches > 0:
        results.add_row("Latency", f"{duration / total_batches * 1000:.1f} ms/batch")
    else:
        results.add_row("Latency", "N/A")
    console.print(results)


@app.command()
def info() -> None:
    """Show configuration and environment variables."""
    table = Table(title="[bold]Config[/bold]")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    table.add_column("Source", style="dim")

    table.add_row("Default Path", DEFAULT_PATH, "default")
    console.print(table)

    console.print(
        "\n[dim]Env vars: OCEANTACO_HF_TOKEN, OCEANTACO_CACHE_DIR, OCEANTACO_DEFAULT_PATH[/dim]"
    )


@app.command()
def test(
    taco_path: str = typer.Option(DEFAULT_PATH, "--path", "-p"),
    query_date: datetime = typer.Option(
        datetime(2024, 4, 1), "--date", "-d", formats=["%Y-%m-%d"]
    ),
    bbox: str = typer.Option("160,170,30,45", "--bbox"),
) -> None:
    """Test single query and inspect output."""
    bbox_tuple = parse_bbox(bbox)

    query = Query(bbox=bbox_tuple, time_start=query_date.date())
    console.print(f"[bold]Query:[/bold] {query}")

    taco_paths = [
        "data/new_ssh_dataset_taco/OceanTACO_part0001.tacozip",
        "data/new_ssh_dataset_taco/OceanTACO_part0002.tacozip",
        "data/new_ssh_dataset_taco/OceanTACO_part0003.tacozip",
        "data/new_ssh_dataset_taco/OceanTACO_part0004.tacozip",
        "data/new_ssh_dataset_taco/OceanTACO_part0005.tacozip",
        "data/new_ssh_dataset_taco/OceanTACO_part0006.tacozip",
    ]

    with console.status("[green]Loading..."):
        dataset = OceanTACODatasetV2(
            taco_path=taco_paths,
            queries=[query],
            input_variables=["l4_ssh", "l4_sst"],
            target_variables=["l3_swot"],
        )
        sample = dataset[0]

    table = Table(title="[bold]Sample[/bold]")
    table.add_column("Variable", style="cyan")
    table.add_column("Role", style="yellow")
    table.add_column("Shape", style="green")

    for name, data in sample["inputs"].items():
        shape = str(tuple(data["data"].shape)) if data else "None"
        table.add_row(name, "input", shape)

    for name, data in sample["targets"].items():
        shape = str(tuple(data["data"].shape)) if data else "None"
        table.add_row(name, "target", shape)

    console.print(table)
    console.print(
        f"\n[dim]bbox={sample['metadata']['bbox']}, time={sample['metadata']['time_range']}[/dim]"
    )


if __name__ == "__main__":
    app()
