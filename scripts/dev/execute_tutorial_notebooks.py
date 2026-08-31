#!/usr/bin/env python3
"""Execute tutorial notebooks in place and leave their outputs for review.

The notebooks configure themselves through the pinned catalog and
``QuerySet.from_hub``, so this runner needs no project-specific environment.
It executes with ``allow_errors=True`` and persists partial output plus a
traceback into notebook metadata, then returns non-zero if any cell errored.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = [
    "docs/tutorials/ml_dataset.ipynb",
    "docs/tutorials/spatio_temporal_query_generation.ipynb",
    "docs/tutorials/data_retrieval_workflows.ipynb",
    "docs/tutorials/ml_configuration_cookbook.ipynb",
    "docs/tutorials/plot_hurricane_milton.ipynb",
    "docs/tutorials/plot_hurricane_milton_cross_product.ipynb",
]


def error_messages(notebook: nbformat.NotebookNode) -> list[str]:
    return [
        f"{output.ename}: {output.evalue}"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type == "error"
    ]


def execute(path: Path) -> dict[str, object]:
    notebook = nbformat.read(path, as_version=4)
    print(f"START {path}", flush=True)
    try:
        NotebookClient(
            notebook,
            timeout=1800,
            kernel_name="python3",
            allow_errors=True,
        ).execute(cwd=str(path.parent))
    except Exception:
        # Persist the partial output as well: it is essential for inspection.
        failure = traceback.format_exc()
        notebook.metadata["ocean_taco_execution_failure"] = failure
        print(f"EXECUTOR FAILURE {path}\n{failure}", flush=True)
    finally:
        nbformat.write(notebook, path)

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = error_messages(notebook)
    result = {
        "notebook": str(path.relative_to(ROOT)),
        "code_cells": len(code_cells),
        "executed_cells": sum(cell.execution_count is not None for cell in code_cells),
        "cells_with_output": sum(bool(cell.outputs) for cell in code_cells),
        "errors": errors,
    }
    print("DONE " + json.dumps(result), flush=True)
    return result


def main() -> int:
    # The notebooks configure themselves; only the standard HF_HOME is relevant,
    # and huggingface_hub supplies its own default when it is unset.
    results = [execute(ROOT / name) for name in NOTEBOOKS]
    failed = [result for result in results if result["errors"]]
    print(
        f"SUMMARY {len(results) - len(failed)}/{len(results)} notebooks executed cleanly",
        flush=True,
    )
    for result in failed:
        print(f"FAILED {result['notebook']}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
