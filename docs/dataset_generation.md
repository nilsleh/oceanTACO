# Dataset Generation Pipeline

This page explains how to generate OceanTACO from raw source data in three steps:

1. Download raw source data (`download.py`)
2. Format source data into regional NetCDF tiles + inventory (`format.py`)
3. Build a TACO dataset from the formatted inventory (`build_taco.py`)

The goal is to give you a self-contained workflow you can run and adapt.

## Prerequisites

Install generation dependencies from the repository root:

```sh
pip install -e ".[generate]"
```

If you use conda in this repo:

```sh
conda activate testpy311
pip install -e ".[generate]"
```

## Directory Flow

The three steps use this flow:

- Step 1 output (`--output-dir` in `download.py`): raw source files (for example `./ssh_state_data`)
- Step 2 input (`--data-dir` in `format.py`): same raw source directory
- Step 2 output (`--output-dir` in `format.py`): formatted regional files + inventory parquet (for example `./formatted_ssh_data`)
- Step 3 input: `--data-dir` points to formatted directory, `--inventory-path` points to inventory parquet
- Step 3 output (`--output-dir` in `build_taco.py`): final OceanTACO folder/parts

## Quick End-to-End Commands

```sh
# 1) Download raw data (dry run by default)
python ocean_taco/generate_dataset/download.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-04 \
  --output-dir ./ssh_state_data \
  --download

# 2) Format into regional NetCDF + inventory parquet
python ocean_taco/generate_dataset/format.py \
  --date-min 2024-01-01 \
  --date-max 2024-01-04 \
  --data-dir ./ssh_state_data \
  --output-dir ./formatted_ssh_data \
  --inventory-path file_inventory.parquet \
  --processes 4

# 3) Build TACO from formatted files
python ocean_taco/generate_dataset/build_taco.py \
  --data-dir ./formatted_ssh_data \
  --output-dir ./tortilla \
  --inventory-path ./formatted_ssh_data/file_inventory.parquet
```

## Step 1: Download Raw Data (`download.py`)

Script:

- `ocean_taco/generate_dataset/download.py`

Example:

```sh
python ocean_taco/generate_dataset/download.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --output-dir ./ssh_state_data \
  --download
```

### CLI Options

| Argument | Type | Default | Purpose |
|---|---|---|---|
| `--start-date` | str (`YYYY-MM-DD`) | `2024-01-01` | Start date (inclusive). |
| `--end-date` | str (`YYYY-MM-DD`) | `2024-01-04` | End date (inclusive). |
| `--output-dir` | path | `./ssh_state_data` | Root directory for downloaded raw files. |
| `--log-dir` | path | `<output-dir>/logs` | Optional custom log/report directory. |
| `--dry-run` | flag | `False` | Preview actions only. If neither mode flag is set, script still behaves as dry run. |
| `--download` | flag | `False` | Perform real downloads. |
| `--weekly-batches` | flag | `False` | Split date range into weekly segments. |
| `--aviso-ftp-user` | str | `""` | AVISO FTP username (for SWOT FTP flows). |
| `--aviso-ftp-pass` | str | `""` | AVISO FTP password (for SWOT FTP flows). |
| `--swot-level` | choice `l2|l3` | `l2` | SWOT product level for SWOT download paths. |
| `--continue-on-error` | flag | `False` | Continue with remaining datasets if one fails. |

### Important Current Behavior

- In the current `main()` implementation, only `L3 SSH` is actively enabled in the `download_functions` list.
- Other dataset download calls are present but currently commented out in `download.py`.
- The script writes structured logs and a JSON report in the log directory.

## Step 2: Format and Regionalize Data (`format.py`)

Script:

- `ocean_taco/generate_dataset/format.py`

This step:

- reads raw source files
- splits/aggregates into 8 spatial regions
- writes formatted regional NetCDF files
- writes/updates `file_inventory.parquet`

Example:

```sh
python ocean_taco/generate_dataset/format.py \
  --date-min 2024-01-01 \
  --date-max 2024-01-31 \
  --data-dir ./ssh_state_data \
  --output-dir ./formatted_ssh_data \
  --inventory-path file_inventory.parquet \
  --processes 4
```

### CLI Options

| Argument | Type | Default | Purpose |
|---|---|---|---|
| `--date-min` | str (`YYYY-MM-DD`) | `2024-01-01` | Start date (inclusive). |
| `--date-max` | str (`YYYY-MM-DD`) | `2024-01-04` | End date (inclusive). |
| `--data-dir` | path | `./ssh_state_data` | Root directory containing raw downloaded data. |
| `--output-dir` | path | `./formatted_ssh_data` | Root directory for regional formatted outputs. |
| `--inventory-path` | filename/path | `file_inventory.parquet` | Inventory filename (written under output dir). |
| `--processes`, `-p` | int | `2` | Number of worker processes. |
| `--include-l3-swot`, `--no-include-l3-swot` | bool toggle | `True` | Include or skip L3 SWOT processing. |
| `--include-l3-ssh`, `--no-include-l3-ssh` | bool toggle | `True` | Include or skip L3 SSH processing. |
| `--include-argo`, `--no-include-argo` | bool toggle | `True` | Include or skip Argo processing. |
| `--only-vars` | list of strings | `None` | Restrict processing to specific sources (for example `l4_ssh`). |
| `--update-existing-inventory` | flag | `False` | Merge into existing inventory instead of replacing it. |

### Output You Should Expect

- Regional files under modality-specific directories in `--output-dir`
- Inventory parquet at:

```text
<output-dir>/<inventory-path>
```

By default that is:

```text
./formatted_ssh_data/file_inventory.parquet
```

## Step 3: Build TACO (`build_taco.py`)

Script:

- `ocean_taco/generate_dataset/build_taco.py`

This step:

- loads formatted inventory parquet
- groups files into Date -> Region -> Files hierarchy
- writes final TACO output
- optionally verifies readability

Example:

```sh
python ocean_taco/generate_dataset/build_taco.py \
  --data-dir ./formatted_ssh_data \
  --output-dir ./tortilla \
  --inventory-path ./formatted_ssh_data/file_inventory.parquet
```

### CLI Options

| Argument | Type | Default | Purpose |
|---|---|---|---|
| `--data-dir` | path | `./formatted_ssh_data` | Root directory with formatted regional files. |
| `--output-dir` | path | `./tortilla` | Output directory for built OceanTACO artifacts. |
| `--inventory-path` | path | required | Input inventory parquet path. |
| `--include-l3-swot`, `--no-include-l3-swot` | bool toggle | `True` | Include/skip L3 SWOT during build. |
| `--include-argo`, `--no-include-argo` | bool toggle | `True` | Include/skip Argo during build. |
| `--verify` | flag | `True` behavior | Verify built TACO by loading it with `tacoreader`. |
| `--start-date` | str (`YYYY-MM-DD`) | `None` | Optional inventory filter lower bound. |
| `--end-date` | str (`YYYY-MM-DD`) | `None` | Optional inventory filter upper bound. |
| `--analyze-duplicates-only` | flag | `False` | Run duplicate analysis and exit without build. |
| `--duplicate-report-path` | path | `None` | Optional parquet output for duplicate-analysis details. |

## Recommended Workflow Patterns

### 1. Safe first pass (dry run + small date range)

```sh
python ocean_taco/generate_dataset/download.py --start-date 2024-01-01 --end-date 2024-01-03 --dry-run
python ocean_taco/generate_dataset/format.py --date-min 2024-01-01 --date-max 2024-01-03 --processes 2
python ocean_taco/generate_dataset/build_taco.py --inventory-path ./formatted_ssh_data/file_inventory.parquet
```

### 2. Incremental updates

If you rerun formatting for a new date window and want to keep old inventory entries:

```sh
python ocean_taco/generate_dataset/format.py \
  --date-min 2024-02-01 \
  --date-max 2024-02-07 \
  --update-existing-inventory
```

### 3. Duplicate investigation before build

```sh
python ocean_taco/generate_dataset/build_taco.py \
  --inventory-path ./formatted_ssh_data/file_inventory.parquet \
  --analyze-duplicates-only \
  --duplicate-report-path ./reports/duplicates.parquet
```

## Troubleshooting

- If build fails, first verify `--inventory-path` exists and points to the formatted step output.
- If output looks incomplete, confirm which data sources are currently enabled in `download.py` `download_functions`.
- For long date ranges, start with `--weekly-batches` in download and moderate `--processes` in format.
- Use `--continue-on-error` in download only when partial results are acceptable.
