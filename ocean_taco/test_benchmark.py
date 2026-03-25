#!/usr/bin/env python3
"""Description of file."""

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
from test_dataset import (
    CONFIG,
    OceanTACODataset,
    Query,
    collate_ocean_samples,
    generate_queries,
)
from torch.utils.data import DataLoader

app = typer.Typer(help="OceanTACO Dataset Benchmark CLI")
console = Console()


def parse_bbox(bbox_str: str) -> tuple[float, float, float, float]:
    """Parse a comma-separated bbox string.

    Args:
        bbox_str: Format 'lon_min,lon_max,lat_min,lat_max'.

    Returns:
        Tuple of four floats.
    """
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


def _print_config(
    taco_path: str,
    start_date: datetime,
    end_date: datetime,
    bbox: str,
    batch_size: int,
    num_workers: int,
) -> None:
    """Display benchmark configuration as a Rich panel."""
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("Path", taco_path)
    table.add_row("Date Range", f"{start_date.date()} → {end_date.date()}")
    table.add_row("BBox", bbox)
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Workers", str(num_workers))
    console.print(Panel(table, title="[bold]Config[/bold]"))


def _print_results(
    title: str, duration: float, total_batches: int, total_samples: int
) -> None:
    """Display benchmark results as a Rich table."""
    table = Table(title=f"[bold]{title}[/bold]", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Time", f"{duration:.2f}s")
    table.add_row("Batches", str(total_batches))
    table.add_row("Throughput", f"{total_batches / duration:.2f} batch/s")
    table.add_row("Throughput", f"{total_samples / duration:.2f} sample/s")
    table.add_row("Latency", f"{duration / total_batches * 1000:.1f} ms/batch")
    console.print(table)


@app.command()
def benchmark(
    taco_path: str = typer.Option(
        CONFIG.default_path, "--path", "-p", help="Path or URL to TACO dataset"
    ),
    start_date: datetime = typer.Option(
        datetime(2024, 1, 1), "--start", "-s", formats=["%Y-%m-%d"], help="Start date"
    ),
    end_date: datetime = typer.Option(
        datetime(2024, 1, 15), "--end", "-e", formats=["%Y-%m-%d"], help="End date"
    ),
    batch_size: int = typer.Option(8, "--batch-size", "-b"),
    num_workers: int = typer.Option(0, "--workers", "-w"),
    bbox: str = typer.Option(
        "120,140,20,40", "--bbox", help="lon_min,lon_max,lat_min,lat_max"
    ),
    compare: bool = typer.Option(
        True,
        "--compare/--no-compare",
        help="Compare DataLoader batching vs direct __getitem__ loop",
    ),
) -> None:
    """Run throughput benchmark on OceanTACO dataset."""
    bbox_tuple = parse_bbox(bbox)
    _print_config(taco_path, start_date, end_date, bbox, batch_size, num_workers)

    with console.status("[green]Generating queries..."):
        queries = generate_queries(
            bbox=bbox_tuple, date_start=start_date.date(), date_end=end_date.date()
        )
    console.print(f"Queries: [cyan]{len(queries)}[/cyan]")

    with console.status("[green]Indexing dataset (SQL)..."):
        dataset = OceanTACODataset(
            taco_path=taco_path,
            queries=queries,
            input_variables=["l3_ssh", "l4_ssh", "glorys_ssh", "glorys_sss", "l4_wind"],
            target_variables=["l3_swot", "l4_wind", "l3_sss_smos_asc"],
        )
    console.print(f"Samples: [cyan]{len(dataset)}[/cyan]")

    if compare:
        # DataLoader benchmark
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_ocean_samples,
        )

        # Direct __getitem__ benchmark (no DataLoader)
        total_samples, total_batches = 0, 0
        start_time = time.perf_counter()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Direct __getitem__", total=len(dataset))

            for i in range(len(dataset)):
                _ = dataset[i]
                total_samples += 1
                total_batches += 1
                progress.advance(task)

        duration = time.perf_counter() - start_time
        _print_results("Direct __getitem__", duration, total_batches, total_samples)

        total_samples, total_batches = 0, 0
        start_time = time.perf_counter()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("DataLoader", total=len(loader))

            for batch in loader:
                # Infer actual batch size from first non-None input tensor
                bs = batch_size
                for var_data in batch["inputs"].values():
                    if var_data is not None:
                        bs = var_data.shape[0]
                        break

                total_samples += bs
                total_batches += 1
                progress.advance(task)

        duration = time.perf_counter() - start_time
        _print_results("DataLoader", duration, total_batches, total_samples)

    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_ocean_samples,
        )

        total_samples, total_batches = 0, 0
        start_time = time.perf_counter()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("DataLoader", total=len(loader))

            for batch in loader:
                # Infer actual batch size from first non-None input tensor
                bs = batch_size
                for var_data in batch["inputs"].values():
                    if var_data is not None:
                        bs = var_data.shape[0]
                        break

                total_samples += bs
                total_batches += 1
                progress.advance(task)

        duration = time.perf_counter() - start_time
        _print_results("DataLoader", duration, total_batches, total_samples)


@app.command()
def info() -> None:
    """Show configuration and environment variables."""
    table = Table(title="[bold]Config[/bold]")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    table.add_column("Source", style="dim")

    table.add_row("Default Path", CONFIG.default_path, "env/default")
    table.add_row("Cache Dir", str(CONFIG.cache_dir), "env/default")
    table.add_row("HF Token", "***" if CONFIG.hf_token else "(not set)", "env")
    console.print(table)
    console.print(
        "\n[dim]Env vars: OCEANTACO_HF_TOKEN, OCEANTACO_CACHE_DIR, OCEANTACO_DEFAULT_PATH[/dim]"
    )


@app.command()
def test(
    taco_path: str = typer.Option(CONFIG.default_path, "--path", "-p"),
    query_date: datetime = typer.Option(
        datetime(2023, 4, 1), "--date", "-d", formats=["%Y-%m-%d"]
    ),
    bbox: str = typer.Option("160,170,30,45", "--bbox"),
) -> None:
    """Test single query and inspect output shapes."""
    bbox_tuple = parse_bbox(bbox)
    query = Query(bbox=bbox_tuple, time_start=query_date.date())
    console.print(f"[bold]Query:[/bold] {query}")

    with console.status("[green]Loading..."):
        dataset = OceanTACODataset(
            taco_path=taco_path,
            queries=[query],
            input_variables=["l4_ssh", "l4_sst", "glorys_ssh", "glorys_sss", "l4_wind"],
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
